"""Read all scan pages of one document at the same time and put the text together."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from .gemini import GeminiClient
from .gemini import GeminiError
from .gemini import PageResult
from .pages import load_payload

if TYPE_CHECKING:
    from .config import Config
    from .pages import Page

logger = logging.getLogger("paperless.parsing.gemini_ocr")


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
        except OSError as e:
            logger.warning(
                "Prompt file %s not readable (%s), using the built-in prompt",
                cfg.prompt_file,
                type(e).__name__,
            )
    return resources.files(__package__).joinpath("prompt.txt").read_text(encoding="utf-8").strip()


def read_pages(
    pages: list[Page],
    cfg: Config,
    workdir: Path,
    client: GeminiClient | None = None,
) -> DocumentResult:
    """Send every page of kind "gemini" to the model. Never raises for a page."""
    todo = [p for p in pages if p.kind == "gemini"]
    outcome = DocumentResult()
    if not todo:
        return outcome

    prompt = load_prompt(cfg)
    started = time.monotonic()
    deadline = started + cfg.document_timeout
    own_client = client is None
    client = client or GeminiClient(cfg)

    def read(page: Page) -> PageResult:
        data, mime_type = load_payload(page, cfg, workdir)
        return client.transcribe(data, mime_type, prompt, deadline)

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
                except Exception as e:
                    # Rendering, file access, anything unforeseen: the page
                    # counts as failed, the document still gets imported.
                    outcome.failures[number] = f"{type(e).__name__}: {e}"
    finally:
        if own_client:
            client.close()
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
