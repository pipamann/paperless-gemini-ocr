"""Configuration, read from environment variables on every document.

All variables share the prefix ``PAPERLESS_GEMINI_OCR_``. Reading them per
document (instead of once at import) keeps the plugin testable and lets a
container restart pick up changes without any caching surprises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PREFIX = "PAPERLESS_GEMINI_OCR_"


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(PREFIX + name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_bool(name: str, default: bool) -> bool:
    value = _get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    value = _get(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _get_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = _get(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        return default


def _get_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = (_get(name) or default).lower()
    return value if value in choices else default


def _read_api_key() -> str | None:
    key = _get("API_KEY")
    if key:
        return key
    key_file = _get("API_KEY_FILE")
    if key_file:
        try:
            return Path(key_file).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


@dataclass(frozen=True)
class Config:
    api_key: str | None = None
    model: str = "gemini-3.8-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # "low" / "medium" / "high" are sent as thinkingLevel, a number as
    # thinkingBudget, "default" sends nothing and leaves it to the model.
    thinking: str = "low"
    # low / medium / high / default
    media_resolution: str = "high"
    max_output_tokens: int = 16384

    # How a PDF page reaches the model: as a single-page PDF ("pdf") or as a
    # rendered JPEG ("image").
    page_input: str = "pdf"
    render_dpi: int = 200

    # Pages of one document that are read at the same time.
    concurrency: int = 8
    # Seconds for one request, attempts per page, seconds for a whole document.
    request_timeout: float = 90.0
    max_attempts: int = 3
    document_timeout: float = 600.0

    # "embedded": pages with real embedded text keep that text and never leave
    # the server. "gemini": every page is read by the model.
    digital_pages: str = "embedded"
    # Share of the page covered by images from which a page counts as a scan,
    # whatever text layer the scanner added.
    scan_image_coverage: float = 0.7
    # Next to an image that takes a fifth of the page or more, embedded text
    # shorter than this is not trusted to be the page's whole content.
    min_text_chars: int = 200

    # "fallback": Tesseract only runs when the model failed for a page.
    # "always": Tesseract runs for every document, as without the plugin, and
    # the archive PDF keeps its text layer.
    local_ocr: str = "fallback"
    # Turn pages upright in the archive PDF when the model reports a rotation.
    # "upside-down": only pages reported as rotated by 180 degrees. The model
    # is reliable there, while it sometimes confuses 90 and 270.
    # "all": every reported rotation. "off": never.
    rotate: str = "upside-down"

    # Tag (and note) for documents where the model failed and local OCR text
    # was used instead. Empty string disables the tag.
    failure_tag: str = "ocr-review"
    failure_note: bool = True

    prompt_file: str | None = None

    @classmethod
    def from_env(cls) -> Config:
        defaults = cls()
        return cls(
            api_key=_read_api_key(),
            model=_get("MODEL", defaults.model),
            base_url=(_get("BASE_URL", defaults.base_url) or "").rstrip("/"),
            thinking=(_get("THINKING", defaults.thinking) or "").lower(),
            media_resolution=_get_choice(
                "MEDIA_RESOLUTION",
                defaults.media_resolution,
                ("low", "medium", "high", "default"),
            ),
            max_output_tokens=_get_int(
                "MAX_OUTPUT_TOKENS",
                defaults.max_output_tokens,
                256,
            ),
            page_input=_get_choice("PAGE_INPUT", defaults.page_input, ("pdf", "image")),
            render_dpi=_get_int("RENDER_DPI", defaults.render_dpi, 72),
            concurrency=_get_int("CONCURRENCY", defaults.concurrency, 1),
            request_timeout=_get_float(
                "REQUEST_TIMEOUT",
                defaults.request_timeout,
                5.0,
            ),
            max_attempts=_get_int("MAX_ATTEMPTS", defaults.max_attempts, 1),
            document_timeout=_get_float(
                "DOCUMENT_TIMEOUT",
                defaults.document_timeout,
                10.0,
            ),
            digital_pages=_get_choice(
                "DIGITAL_PAGES",
                defaults.digital_pages,
                ("embedded", "gemini"),
            ),
            scan_image_coverage=min(
                1.0,
                _get_float("SCAN_IMAGE_COVERAGE", defaults.scan_image_coverage, 0.05),
            ),
            min_text_chars=_get_int("MIN_TEXT_CHARS", defaults.min_text_chars, 1),
            local_ocr=_get_choice(
                "LOCAL_OCR",
                defaults.local_ocr,
                ("fallback", "always"),
            ),
            rotate=_get_choice(
                "ROTATE",
                defaults.rotate,
                ("upside-down", "all", "off"),
            ),
            failure_tag=os.environ.get(PREFIX + "FAILURE_TAG", defaults.failure_tag).strip(),
            failure_note=_get_bool("FAILURE_NOTE", defaults.failure_note),
            prompt_file=_get("PROMPT_FILE"),
        )
