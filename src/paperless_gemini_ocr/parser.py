"""The Paperless-ngx parser: Tesseract's parser with Gemini reading the scans.

It extends the built-in ``RasterisedDocumentParser`` so that archive PDF,
thumbnail, page count and metadata work exactly as without the plugin. Only
the text of scanned pages comes from the model, requested for all pages at
once while the local part of the import runs.

Rules it follows:

* A model failure never fails an import. The affected pages get local OCR
  text and the document is tagged, so nothing is silently worse.
* "Reprocess" on an existing document raises instead, so good text is never
  replaced by fallback text. Paperless logs the error and leaves the
  document as it was.
* Documents without scanned pages are handed to the built-in parser untouched.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from documents.parsers import ParseError
from documents.signals import document_consumption_finished
from documents.signals import document_consumption_started
from paperless.parsers.tesseract import RasterisedDocumentParser

from . import __version__
from . import state
from .config import Config
from .pages import _inherited
from .pages import analyse_pdf
from .pages import normalise_text
from .pages import pdf_page_texts
from .pages import prepare_image
from .pipeline import DocumentResult
from .pipeline import assemble
from .pipeline import read_pages

if TYPE_CHECKING:
    from .pages import Page

logger = logging.getLogger("paperless.parsing.gemini_ocr")


class GeminiOcrParser(RasterisedDocumentParser):
    name: str = "Gemini OCR"
    version: str = __version__
    author: str = "pipamann"
    url: str = "https://github.com/pipamann/paperless-gemini-ocr"

    # Honours PAPERLESS_REMOTE_OCR_MODE and the workflow action for remote OCR.
    uses_remote_service: bool = True

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        if mime_type not in cls.supported_mime_types():
            return None
        if not Config.from_env().api_key:
            # Not configured: stay invisible, the built-in parser takes over.
            return None
        return 30

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        document_path = Path(document_path)
        in_consume = state.take_consume_flag(document_path)
        cfg = Config.from_env()
        started = time.monotonic()
        workdir = Path(self.tempdir) / "gemini-ocr"
        workdir.mkdir(exist_ok=True)

        try:
            if mime_type == "application/pdf":
                pages = analyse_pdf(document_path, cfg, workdir)
            else:
                pages = prepare_image(document_path, mime_type, workdir)
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            if not in_consume:
                raise ParseError(
                    f"Gemini OCR could not process the document ({reason}); "
                    "content left unchanged",
                ) from e
            self.log.warning("Gemini OCR: %s; using local OCR only", reason)
            super().parse(document_path, mime_type, produce_archive=produce_archive)
            state.set_report(state.Report(document_error=reason))
            return

        scan_pages = [p for p in pages if p.kind == "gemini"]
        if not scan_pages:
            self.log.debug(
                "Gemini OCR: no scanned pages in %d page(s), built-in parser only",
                len(pages),
            )
            super().parse(document_path, mime_type, produce_archive=produce_archive)
            return

        # The model reads the pages while the local part of the import runs.
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gemini-ocr-doc",
        ) as pool:
            future = pool.submit(read_pages, pages, cfg, workdir)
            try:
                if cfg.local_ocr == "always":
                    super().parse(
                        document_path,
                        mime_type,
                        produce_archive=produce_archive,
                    )
                    archive_from_ocr = True
                else:
                    archive_from_ocr = self._parse_local(
                        document_path,
                        mime_type,
                        produce_archive=produce_archive,
                    )
            finally:
                outcome = self._collect(future, scan_pages)

        local_texts: dict[int, str] = {}
        if outcome.failures:
            if not in_consume:
                raise ParseError(
                    "Gemini OCR failed for page(s) "
                    f"{', '.join(str(n) for n in sorted(outcome.failures))}; "
                    "content left unchanged",
                )
            local_texts = self._local_texts(document_path, mime_type, pages, outcome)
            state.set_report(
                state.Report(total_pages=len(pages), failures=dict(outcome.failures)),
            )

        self.text = assemble(pages, outcome, local_texts)

        # OCRmyPDF turns pages itself; turning them again would undo that.
        if cfg.rotate != "off" and not archive_from_ocr:
            self._apply_rotation(
                pages,
                outcome,
                (180,) if cfg.rotate == "upside-down" else (90, 180, 270),
            )

        self.log.info(
            "Gemini OCR: %d page(s), %d read by %s in %.1f s (%d request(s), "
            "%d input / %d output tokens), %d with embedded text, %d failed; "
            "import parsing took %.1f s",
            len(pages),
            len(outcome.results),
            cfg.model,
            outcome.seconds,
            outcome.attempts,
            outcome.input_tokens,
            outcome.output_tokens,
            sum(1 for p in pages if p.kind == "embedded"),
            len(outcome.failures),
            time.monotonic() - started,
        )

    def _collect(self, future, scan_pages: list[Page]) -> DocumentResult:
        """Result of the model run; an unexpected error counts as all pages failed."""
        try:
            return future.result()
        except Exception as e:
            self.log.warning("Gemini OCR: reading failed (%s)", type(e).__name__)
            return DocumentResult(
                failures={p.number: f"{type(e).__name__}" for p in scan_pages},
            )

    # ------------------------------------------------------------------
    # Local part
    # ------------------------------------------------------------------

    def _parse_local(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool,
    ) -> bool:
        """Archive PDF and embedded text without running Tesseract.

        Uses the built-in parser's own OCR_MODE=off path. Should that path
        fail on an odd file, the normal path with OCR gets a try before the
        import is given up, as it would without the plugin. Returns True when
        that happened, i.e. when OCRmyPDF produced the archive.
        """
        from paperless.models import ModeChoices

        configured_mode = self.settings.mode
        self.settings.mode = ModeChoices.OFF
        try:
            super().parse(document_path, mime_type, produce_archive=produce_archive)
            return False
        except Exception as e:
            self.log.warning(
                "Gemini OCR: import without local OCR failed (%s), retrying with it",
                type(e).__name__,
            )
        finally:
            self.settings.mode = configured_mode
        self.text = None
        self.archive_path = None
        super().parse(document_path, mime_type, produce_archive=produce_archive)
        return True

    def _local_texts(
        self,
        document_path: Path,
        mime_type: str,
        pages: list[Page],
        outcome: DocumentResult,
    ) -> dict[int, str]:
        """Best local text for exactly the pages the model failed on.

        Local OCR where it yields something, otherwise whatever text layer the
        page brought along (a scanner's own, for instance).
        """
        failed = sorted(outcome.failures)
        by_number = {p.number: p for p in pages}
        texts = {n: by_number[n].embedded_text for n in failed}
        for number, text in self._ocr_failed_pages(
            document_path,
            mime_type,
            failed,
            len(pages),
        ).items():
            if text.strip():
                texts[number] = text
        return texts

    def _ocr_failed_pages(
        self,
        document_path: Path,
        mime_type: str,
        failed: list[int],
        total_pages: int,
    ) -> dict[int, str]:
        """Run Tesseract over the failed pages only and return their text.

        A separate pass with forced OCR: the regular modes may skip OCR (a
        PDF that already has text) or produce no usable output file (no
        archive wanted). Text, archive and settings of the import itself are
        put back afterwards, whatever happens here.
        """
        import pikepdf
        from paperless.models import ModeChoices

        configured_mode = self.settings.mode
        if configured_mode == ModeChoices.OFF:
            # OCR is switched off for this installation; respect that.
            return {}

        tempdir = Path(self.tempdir)
        workdir = tempdir / "gemini-ocr"
        kept_text = self.text
        kept_archive = Path(self.archive_path) if self.archive_path else None
        aside: Path | None = None
        outputs = [tempdir / "archive.pdf", tempdir / "archive-fallback.pdf"]
        try:
            if mime_type == "application/pdf":
                subset = workdir / "failed-pages.pdf"
                with pikepdf.open(document_path) as source:
                    only_failed = pikepdf.Pdf.new()
                    for number in failed:
                        only_failed.pages.append(source.pages[number - 1])
                    only_failed.save(subset)
                    only_failed.close()
                numbers = failed
            else:
                subset = document_path
                numbers = list(range(1, total_pages + 1))

            if kept_archive is not None and kept_archive.is_file():
                aside = tempdir / "gemini-ocr-first-pass-archive.pdf"
                kept_archive.replace(aside)
            for output in outputs:
                output.unlink(missing_ok=True)

            self.settings.mode = ModeChoices.FORCE
            self.text = None
            self.archive_path = None
            super().parse(subset, mime_type, produce_archive=False)
            whole = self.text or ""

            per_page: list[str] | None = None
            for output in outputs:
                if output.is_file():
                    per_page = pdf_page_texts(output)
                    break
            if len(numbers) == 1:
                return {numbers[0]: whole or normalise_text((per_page or [""])[0])}
            if per_page is not None and len(per_page) == len(numbers):
                found = {
                    n: normalise_text(t)
                    for n, t in zip(numbers, per_page, strict=True)
                    if n in failed
                }
                if any(found.values()) or not whole:
                    return found
            # No page boundaries to go by: keep what OCR found, once.
            return {failed[0]: whole}
        except Exception as e:
            self.log.warning(
                "Gemini OCR: local OCR for the failed pages failed too (%s)",
                type(e).__name__,
            )
            return {}
        finally:
            self.settings.mode = configured_mode
            for output in outputs:
                output.unlink(missing_ok=True)
            if aside is not None and kept_archive is not None:
                aside.replace(kept_archive)
            self.text = kept_text
            self.archive_path = kept_archive

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _apply_rotation(
        self,
        pages: list[Page],
        outcome: DocumentResult,
        accepted: tuple[int, ...],
    ) -> None:
        """Turn pages upright in the archive PDF. The original is never touched."""
        turns = {
            number: result.rotation
            for number, result in outcome.results.items()
            if result.rotation in accepted
        }
        if not turns:
            return
        if not (self.archive_path and Path(self.archive_path).is_file()):
            self.log.info(
                "Gemini OCR: page(s) %s look rotated; no archive file is produced "
                "for this document, so nothing was turned",
                ", ".join(str(n) for n in sorted(turns)),
            )
            return
        try:
            import pikepdf

            with pikepdf.open(self.archive_path, allow_overwriting_input=True) as pdf:
                if len(pdf.pages) != len(pages):
                    self.log.warning(
                        "Gemini OCR: archive has %d page(s), document %d; not rotating",
                        len(pdf.pages),
                        len(pages),
                    )
                    return
                for number, degrees in turns.items():
                    page = pdf.pages[number - 1]
                    current = int(_inherited(page.obj, "/Rotate") or 0)
                    page.obj["/Rotate"] = (current + degrees) % 360
                pdf.save(self.archive_path)
            self.log.info(
                "Gemini OCR: turned page(s) %s upright in the archive file",
                ", ".join(str(n) for n in sorted(turns)),
            )
        except Exception as e:
            self.log.warning("Gemini OCR: rotating the archive failed (%s)", type(e).__name__)


# ----------------------------------------------------------------------
# Tag and note for documents with failed pages
# ----------------------------------------------------------------------


def _review_tag(name: str):
    """The tag for flagged documents; created on first use, safe against a race."""
    from django.db import IntegrityError
    from django.db import transaction
    from documents.models import Tag

    for _ in range(3):
        tag = Tag.objects.filter(name__iexact=name).first()
        if tag is not None:
            return tag
        try:
            with transaction.atomic():
                return Tag.objects.create(name=name, matching_algorithm=Tag.MATCH_NONE)
        except IntegrityError:
            # Another worker created it at the same moment.
            continue
    raise RuntimeError(f"tag {name!r} can neither be found nor created")


def _flag_document(document, report: state.Report) -> None:
    """Tag and note, each in its own savepoint.

    Neither may undo the import this runs inside of, and a problem with one
    must not take the other with it.
    """
    from django.db import transaction
    from documents.models import Note

    cfg = Config.from_env()
    target = getattr(document, "root_document", None) or document
    if cfg.failure_tag:
        try:
            tag = _review_tag(cfg.failure_tag)
            with transaction.atomic():
                target.tags.add(tag)
        except Exception:
            logger.exception("Gemini OCR: could not tag document %s", target.pk)
    if cfg.failure_note:
        try:
            with transaction.atomic():
                Note.objects.create(document=target, note=report.note())
        except Exception:
            logger.exception("Gemini OCR: could not add a note to document %s", target.pk)


def _on_consumption_started(sender, filename=None, **kwargs) -> None:
    state.consume_started(filename)


def _on_consumption_finished(sender, document=None, **kwargs) -> None:
    report = state.pop_report()
    if report is None or document is None:
        return
    try:
        logger.warning(
            "Gemini OCR: document %s needs review: %s",
            document.pk,
            report.note(),
        )
        _flag_document(document, report)
    except Exception:
        logger.exception("Gemini OCR: could not flag document %s", document.pk)


document_consumption_started.connect(
    _on_consumption_started,
    dispatch_uid="paperless_gemini_ocr.started",
)
document_consumption_finished.connect(
    _on_consumption_finished,
    dispatch_uid="paperless_gemini_ocr.finished",
)
