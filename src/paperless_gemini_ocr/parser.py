"""The Paperless-ngx parser: Tesseract's parser with Gemini reading the scans.

It extends the built-in ``RasterisedDocumentParser`` so that archive PDF,
thumbnail, page count and metadata work exactly as without the plugin. Only
the text of scanned pages comes from the model, requested for all pages at
once while the local part of the import runs.

Rules it follows:

* A model failure never fails an import. The affected pages get local OCR
  text and the document is tagged, so nothing is silently worse.
* "Reprocess" on an existing document fails loudly instead, so good text is
  never replaced by fallback text.
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
from .pipeline import assemble
from .pipeline import read_pages

if TYPE_CHECKING:
    from .pages import Page
    from .pipeline import DocumentResult

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
        ocr_ran = cfg.local_ocr == "always"
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gemini-ocr-doc",
        ) as pool:
            future = pool.submit(read_pages, pages, cfg, workdir)
            try:
                if ocr_ran:
                    super().parse(
                        document_path,
                        mime_type,
                        produce_archive=produce_archive,
                    )
                else:
                    self._parse_local(
                        document_path,
                        mime_type,
                        produce_archive=produce_archive,
                    )
            finally:
                outcome = future.result()

        local_texts: dict[int, str] = {}
        if outcome.failures:
            if not in_consume:
                raise ParseError(
                    "Gemini OCR failed for page(s) "
                    f"{', '.join(str(n) for n in sorted(outcome.failures))}; "
                    "content left unchanged",
                )
            if not ocr_ran:
                ocr_ran = self._run_fallback_ocr(
                    document_path,
                    mime_type,
                    produce_archive=produce_archive,
                )
            local_texts = self._local_page_texts(
                document_path,
                mime_type,
                pages,
                outcome,
            )
            state.set_report(
                state.Report(total_pages=len(pages), failures=dict(outcome.failures)),
            )

        self.text = assemble(pages, outcome, local_texts)

        if cfg.rotate != "off" and not ocr_ran:
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

    # ------------------------------------------------------------------
    # Local part
    # ------------------------------------------------------------------

    def _parse_local(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool,
    ) -> None:
        """Archive PDF and embedded text without running Tesseract.

        Uses the built-in parser's own OCR_MODE=off path. Should that path
        fail on an odd file, the normal path with OCR gets a try before the
        import is given up, as it would without the plugin.
        """
        from paperless.models import ModeChoices

        configured_mode = self.settings.mode
        self.settings.mode = ModeChoices.OFF
        try:
            super().parse(document_path, mime_type, produce_archive=produce_archive)
            return
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

    def _run_fallback_ocr(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool,
    ) -> bool:
        """Run the built-in parser with OCR for pages the model failed on.

        Returns True when it replaced the result of the OCR-free run. On an
        error that result is restored: the document is imported either way.
        """
        kept_text = self.text
        kept_archive: Path | None = None
        if self.archive_path and Path(self.archive_path).is_file():
            kept_archive = Path(self.tempdir) / "archive-without-ocr.pdf"
            Path(self.archive_path).replace(kept_archive)
        self.text = None
        self.archive_path = None
        try:
            super().parse(document_path, mime_type, produce_archive=produce_archive)
            return True
        except Exception as e:
            self.log.warning(
                "Gemini OCR: local OCR for the failed pages failed too (%s)",
                type(e).__name__,
            )
            self.text = kept_text
            self.archive_path = kept_archive
            return False

    def _local_page_texts(
        self,
        document_path: Path,
        mime_type: str,
        pages: list[Page],
        outcome: DocumentResult,
    ) -> dict[int, str]:
        """Local text for exactly the pages the model failed on."""
        failed = sorted(outcome.failures)
        if len(pages) == 1:
            # The built-in parser's own text, as an import without the plugin has it.
            return {1: self.text or ""}
        source: Path | None = None
        if self.archive_path and Path(self.archive_path).is_file():
            source = Path(self.archive_path)
        elif mime_type == "application/pdf":
            source = document_path
        if source is not None:
            texts = pdf_page_texts(source)
            if texts is not None and len(texts) == len(pages):
                return {n: normalise_text(texts[n - 1]) for n in failed}
        self.log.warning(
            "Gemini OCR: no per-page local text available, page(s) %s stay empty",
            ", ".join(str(n) for n in failed),
        )
        return {}

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


def _flag_document(document, report: state.Report) -> None:
    from django.db import transaction
    from documents.models import Note
    from documents.models import Tag

    cfg = Config.from_env()
    target = getattr(document, "root_document", None) or document
    # Own savepoint: an error here must not undo the import it runs inside of.
    with transaction.atomic():
        if cfg.failure_tag:
            tag = Tag.objects.filter(name__iexact=cfg.failure_tag).first()
            if tag is None:
                tag = Tag.objects.create(
                    name=cfg.failure_tag,
                    matching_algorithm=Tag.MATCH_NONE,
                )
            target.tags.add(tag)
        if cfg.failure_note:
            Note.objects.create(document=target, note=report.note())


def _on_consumption_started(sender, filename=None, **kwargs) -> None:
    state.consume_started(filename)


def _on_consumption_finished(sender, document=None, **kwargs) -> None:
    report = state.pop_report()
    if report is None or document is None:
        return
    try:
        _flag_document(document, report)
        logger.warning(
            "Gemini OCR: document %s flagged for review: %s",
            document.pk,
            report.note(),
        )
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
