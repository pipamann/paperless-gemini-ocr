"""Read all scan pages of one document at the same time and put the text together."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from .gemini import GeminiClient
from .gemini import GeminiError
from .gemini import PageResult
from .pages import AnalysisError
from .pages import load_payload
from .pages import turn_payload

if TYPE_CHECKING:
    from .config import Config
    from .pages import Page

logger = logging.getLogger("paperless.parsing.gemini_ocr")

# A page with at least this much text of its own (a scanner's text layer) is
# not believed to be empty just because the model says so.
_TEXT_LAYER_CHARS = 50


@dataclass
class DocumentResult:
    # Keyed by 1-based page number; only pages of kind "gemini" appear.
    results: dict[int, PageResult] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def attempts(self) -> int:
        return sum(r.attempts for r in self.results.values())

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results.values())

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens + r.thinking_tokens for r in self.results.values())


def load_prompt(cfg: Config) -> str:
    if cfg.prompt_file:
        try:
            text = Path(cfg.prompt_file).read_text(encoding="utf-8").strip()
            if text:
                return text
            logger.warning("Prompt file %s is empty, using the built-in prompt", cfg.prompt_file)
        except (OSError, UnicodeError) as e:
            logger.warning(
                "Prompt file %s not readable (%s), using the built-in prompt",
                cfg.prompt_file,
                type(e).__name__,
            )
    return resources.files(__package__).joinpath("prompt.txt").read_text(encoding="utf-8").strip()


def _read_page(
    client: GeminiClient,
    data: bytes,
    mime_type: str,
    prompt: str,
    deadline: float,
    *,
    search_when_refused: bool = True,
    known_refusal: GeminiError | None = None,
) -> PageResult:
    """Read a page, and read it again the right way up when it is not.

    A page the model reports as rotated is turned by that much and read
    again. If the new reading reports the page as upright, its text is taken
    and the rotation counts as confirmed. If it reports another rotation (the
    model mixes up 90 and 270), the page is turned on; no way round is read
    twice, so a page costs four requests at most.

    The API now and then refuses one particular view of a page outright
    (``blockReason``) and accepts the same page turned. A refused page is
    therefore looked at the other ways round as well, and counts as read only
    if one of them is confirmed as upright. ``search_when_refused=False``
    leaves that out: the caller has another view of the page to try first.
    ``known_refusal`` is the refusal the caller already got for the page as
    it is; that request is not sent again.

    Apart from that the first reading stays whatever goes wrong on the way:
    reading again can only make a page better.
    """
    readings: dict[int, PageResult] = {}
    tried: set[int] = set()
    refused: GeminiError | None = None
    failed_attempts = 0

    def look(turn: int) -> PageResult | None:
        """Reading of the page turned clockwise by ``turn``; None if that view failed."""
        nonlocal refused, failed_attempts
        tried.add(turn)
        view, view_mime = data, mime_type
        if turn:
            try:
                view, view_mime = turn_payload(data, mime_type, turn)
            except Exception as e:
                # Nothing was sent, so nothing counts as an attempt.
                logger.debug("Turning a page by %d failed: %s", turn, type(e).__name__)
                return None
        try:
            readings[turn] = client.transcribe(view, view_mime, prompt, deadline)
        except GeminiError as e:
            if turn == 0:
                if not e.blocked:
                    raise
                refused = e
            failed_attempts += e.attempts
            return None
        except Exception as e:
            if turn == 0:
                raise
            failed_attempts += 1
            logger.debug("Reading a page turned by %d failed: %s", turn, type(e).__name__)
            return None
        return readings[turn]

    at, current = 0, None
    if known_refusal is not None:
        tried.add(0)
        refused = known_refusal
        failed_attempts += known_refusal.attempts
    else:
        current = look(0)
    if current is None and not search_when_refused:
        raise refused
    # The turn at which the page was read as upright, once that has happened.
    upright: int | None = None
    while True:
        if current is not None and current.rotation == 0:
            upright = at
            break
        reported = current is not None and current.rotation in (90, 180, 270)
        turn = (at + current.rotation) % 360 if reported else None
        if turn is None or turn in tried:
            # No way round to go by. The others are worth a look when the
            # model keeps reporting a rotation, or when there is no reading
            # to keep because the page as it is was refused.
            if not reported and refused is None:
                break
            left = [other for other in (180, 90, 270) if other not in tried]
            if not left:
                break
            turn = left[0]
        if time.monotonic() >= deadline - 1:
            break
        reading = look(turn)
        if reading is not None:
            at, current = turn, reading

    spent = list(readings.values())
    attempts = sum(r.attempts for r in spent) + failed_attempts
    first = readings.get(0)
    if upright and (readings[upright].text or not (first and first.text)):
        result = PageResult(text=readings[upright].text, rotation=upright, rotation_confirmed=True)
    elif first is not None:
        result = first
    else:
        # The page as it is was refused, and no other view was read as
        # upright. What a turned view reads is not reliable enough to pass
        # as the page's text.
        refused.attempts = attempts
        raise refused
    result.attempts = attempts
    result.seconds = sum(r.seconds for r in spent)
    result.input_tokens = sum(r.input_tokens for r in spent)
    result.output_tokens = sum(r.output_tokens for r in spent)
    result.thinking_tokens = sum(r.thinking_tokens for r in spent)
    return result


def read_pages(
    pages: list[Page],
    cfg: Config,
    workdir: Path,
    client: GeminiClient | None = None,
) -> DocumentResult:
    """Send every page of kind "gemini" to the model.

    Never raises: whatever goes wrong, for one page or for all of them, ends
    up in ``failures``.
    """
    todo = [p for p in pages if p.kind == "gemini"]
    outcome = DocumentResult()
    if not todo:
        return outcome

    started = time.monotonic()
    deadline = started + cfg.document_timeout
    own_client = client is None
    try:
        prompt = load_prompt(cfg)
        client = client or GeminiClient(cfg)
    except Exception as e:
        for page in todo:
            outcome.failures[page.number] = f"setup failed ({type(e).__name__})"
        return outcome

    def read(page: Page) -> PageResult:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise GeminiError("document time limit reached", retryable=False, attempts=0)
        data, mime_type = load_payload(page, cfg, workdir, deadline)
        if not cfg.reread_rotated:
            result = client.transcribe(data, mime_type, prompt, deadline)
        elif page.source_pdf is not None and mime_type == "application/pdf":
            try:
                result = _read_page(
                    client,
                    data,
                    mime_type,
                    prompt,
                    deadline,
                    search_when_refused=False,
                )
            except GeminiError as refused:
                if not refused.blocked:
                    raise
                # What the API refuses is one particular rendering of the
                # page. Sent as a picture, the same page often gets through
                # (10 of 18 refused pages did).
                try:
                    picture, picture_mime = load_payload(
                        page,
                        replace(cfg, page_input="image"),
                        workdir,
                        deadline,
                    )
                    result = _read_page(client, picture, picture_mime, prompt, deadline)
                    result.attempts += refused.attempts
                except GeminiError as again:
                    again.attempts += refused.attempts
                    raise
                except Exception as e:
                    # No picture to be had (too large, rendering failed):
                    # the PDF page turned the other ways round is still worth a look.
                    logger.debug("Refused page could not be sent as a picture: %s", type(e).__name__)
                    result = _read_page(
                        client,
                        data,
                        mime_type,
                        prompt,
                        deadline,
                        known_refusal=refused,
                    )
        else:
            result = _read_page(client, data, mime_type, prompt, deadline)
        if not result.text and len(page.embedded_text) >= _TEXT_LAYER_CHARS:
            raise GeminiError(
                "the model found no text on a page whose own text layer has some",
                retryable=False,
                attempts=result.attempts,
            )
        return result

    try:
        with ThreadPoolExecutor(
            max_workers=min(cfg.concurrency, len(todo)),
            thread_name_prefix="gemini-ocr",
        ) as pool:
            futures = {page.number: pool.submit(read, page) for page in todo}
            for number, future in futures.items():
                try:
                    outcome.results[number] = future.result()
                except GeminiError as e:
                    outcome.failures[number] = f"{e} after {e.attempts} attempt(s)"
                except AnalysisError as e:
                    outcome.failures[number] = str(e)
                except Exception as e:
                    # Rendering, file access, anything unforeseen: the page
                    # counts as failed, the document still gets imported.
                    # Only the kind of error is kept; the text of a foreign
                    # exception goes into the document note otherwise.
                    outcome.failures[number] = type(e).__name__
    except Exception as e:
        for page in todo:
            if page.number not in outcome.results:
                outcome.failures.setdefault(page.number, type(e).__name__)
    finally:
        if own_client:
            try:
                client.close()
            except Exception:
                pass
    outcome.seconds = time.monotonic() - started
    return outcome


def assemble(
    pages: list[Page],
    outcome: DocumentResult,
    local_texts: dict[int, str] | None = None,
) -> str:
    """Document text in page order.

    Pages the model read use its text, pages with embedded text keep it, and
    pages the model failed on use ``local_texts`` (local OCR) where available.
    """
    local_texts = local_texts or {}
    parts: list[str] = []
    for page in pages:
        if page.kind == "embedded":
            text = page.embedded_text
        elif page.kind == "gemini":
            result = outcome.results.get(page.number)
            text = result.text if result is not None else local_texts.get(page.number, "")
        else:
            text = ""
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
