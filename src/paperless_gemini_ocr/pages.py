"""Decide per page whether the model has to read it, and prepare the upload.

A page goes to the model when it is a scan: an image fills most of the page
(whatever text layer the scanner added), or the page has no usable text layer
at all. A page with real embedded text keeps that text, which is exact, and
never leaves the server.

Nothing in here imports Django or Paperless, so it can be tested on its own.
It needs ``pdftotext`` and ``pdftoppm`` (poppler), which Paperless-ngx ships.
"""

from __future__ import annotations

import base64
import logging
import re
import subprocess
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config

logger = logging.getLogger("paperless.parsing.gemini_ocr")

# Image types the model accepts as they are.
GEMINI_IMAGE_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"},
)
# Inline requests are limited to 20 MB; base64 adds a third.
MAX_INLINE_BYTES = 14 * 1024 * 1024
_MAX_FORM_DEPTH = 6
# Upper bound on the drawing instructions looked at per page, nested forms
# included. A crafted PDF can make forms call each other with a huge fan-out;
# past the budget the page simply counts as a scan.
_INSTRUCTION_BUDGET = 200_000
# Same idea for the decoded size of the content streams, checked before they
# are parsed, so a tiny stream that inflates to gigabytes is never unpacked.
_DECODED_BYTE_BUDGET = 32 * 1024 * 1024
# Seconds the page analysis of one document may take; later pages then count
# as scans without being looked at.
_ANALYSIS_SECONDS = 120.0
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


class AnalysisError(Exception):
    """The document could not be taken apart; the caller falls back to local OCR."""


class _BudgetExceeded(Exception):
    """A page needs more analysis work than a real document plausibly does."""


def _run_length_decode(data: bytes, limit: int) -> bytes | None:
    out = bytearray()
    index = 0
    while index < len(data):
        length = data[index]
        index += 1
        if length == 128:
            break
        if length < 128:
            out += data[index : index + length + 1]
            index += length + 1
        else:
            out += data[index : index + 1] * (257 - length)
            index += 1
        if len(out) > limit:
            return None
    return bytes(out)


def _decode_stage(name: str, data: bytes, limit: int) -> bytes | None:
    """Undo one stream filter with at most ``limit`` bytes of output, else None."""
    try:
        if name in ("/FlateDecode", "/Fl"):
            inflater = zlib.decompressobj()
            out = inflater.decompress(data, limit + 1)
            if len(out) > limit or inflater.unconsumed_tail:
                return None
            return out
        if name in ("/ASCII85Decode", "/A85"):
            body = b"".join(data.split())
            if not body.startswith(b"<~"):
                body = b"<~" + body
            if not body.endswith(b"~>"):
                body += b"~>"
            return base64.a85decode(body, adobe=True)
        if name in ("/ASCIIHexDecode", "/AHx"):
            body = b"".join(data.split()).rstrip(b">")
            if len(body) % 2:
                body += b"0"
            return bytes.fromhex(body.decode("ascii"))
        if name in ("/RunLengthDecode", "/RL"):
            return _run_length_decode(data, limit)
    except (ValueError, zlib.error):
        return None
    # LZW and anything else: no cheap way to bound the output.
    return None


def _decoded_size(stream, limit: int) -> int:
    """Decoded length of a stream, or ``limit + 1`` once it may be larger.

    Every filter of the chain is undone with a cap on its output, so nothing
    is ever unpacked beyond ``limit``. A chain with a filter that cannot be
    bounded counts as too large.
    """
    import pikepdf

    data = stream.read_raw_bytes()
    filters = stream.get("/Filter")
    if filters is None:
        names: list[str] = []
    elif isinstance(filters, pikepdf.Name):
        names = [str(filters)]
    else:
        names = [str(f) for f in filters]
    for name in names:
        if len(data) > limit:
            return limit + 1
        decoded = _decode_stage(name, data, limit)
        if decoded is None:
            return limit + 1
        data = decoded
    return len(data) if len(data) <= limit else limit + 1


class _Budget:
    """Work one page may cause, shared by the page and all forms it paints."""

    def __init__(self) -> None:
        self.instructions = _INSTRUCTION_BUDGET
        self.decoded_bytes = _DECODED_BYTE_BUDGET

    def charge_stream(self, stream) -> None:
        """Call before a stream is parsed."""
        self.decoded_bytes -= _decoded_size(stream, self.decoded_bytes)
        if self.decoded_bytes < 0:
            raise _BudgetExceeded

    def charge_instructions(self, count: int) -> None:
        # At least one unit, so empty forms cannot be called for free.
        self.instructions -= max(1, count)
        if self.instructions < 0:
            raise _BudgetExceeded


@dataclass
class Page:
    number: int  # 1-based
    kind: str  # "gemini" | "embedded" | "blank"
    reason: str = ""
    embedded_text: str = ""
    payload: Path | None = None
    mime_type: str | None = None
    # Set for PDF pages, so the page can be rendered when the single-page PDF
    # is too large or images are configured.
    source_pdf: Path | None = None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalise_text(text: str | None) -> str:
    """Same clean-up Paperless applies to extracted text (``post_process_text``)."""
    if not text:
        return ""
    collapsed = re.sub(r"([^\S\r\n]+)", " ", text)
    no_leading = re.sub(r"([\n\r]+)([^\S\n\r]+)", "\\1", collapsed)
    no_trailing = re.sub(r"([^\S\n\r]+)$", "", no_leading)
    return no_trailing.strip().replace("\0", " ")


def looks_broken(text: str) -> bool:
    """Text layer whose font mapping is lost: replacement or private-use characters."""
    if not text:
        return False
    bad = text.count("�") + 5 * text.count("(cid:")
    bad += sum(1 for ch in text if "" <= ch <= "")
    return bad / len(text) > 0.1


def pdf_page_texts(path: Path) -> list[str] | None:
    """Embedded text of every page via one pdftotext run, or None on failure."""
    try:
        proc = subprocess.run(
            ["pdftotext", "-q", "-layout", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("pdftotext failed: %s", type(e).__name__)
        return None
    pieces = proc.stdout.decode("utf-8", errors="replace").split("\f")
    # pdftotext ends every page with a form feed, so the last piece is empty.
    if pieces and not pieces[-1].strip():
        pieces.pop()
    return pieces


# ---------------------------------------------------------------------------
# Image coverage
# ---------------------------------------------------------------------------


def _concat(m: tuple, ctm: tuple) -> tuple:
    """PDF ``cm``: the new matrix ``m`` is applied before the current one."""
    return (
        m[0] * ctm[0] + m[1] * ctm[2],
        m[0] * ctm[1] + m[1] * ctm[3],
        m[2] * ctm[0] + m[3] * ctm[2],
        m[2] * ctm[1] + m[3] * ctm[3],
        m[4] * ctm[0] + m[5] * ctm[2] + ctm[4],
        m[4] * ctm[1] + m[5] * ctm[3] + ctm[5],
    )


def _inherited(obj, key: str):
    """Page attributes such as /Resources may sit on a parent /Pages node."""
    for _ in range(50):
        if obj is None:
            return None
        value = obj.get(key)
        if value is not None:
            return value
        obj = obj.get("/Parent")
    return None


def _page_area(page) -> float:
    box = _inherited(page.obj, "/CropBox") or _inherited(page.obj, "/MediaBox")
    x0, y0, x1, y1 = (float(v) for v in box)
    return abs((x1 - x0) * (y1 - y0))


def _image_area(
    stream,
    resources,
    ctm: tuple,
    limit: float,
    depth: int,
    budget: _Budget,
) -> tuple[float, int]:
    """Summed area of all images a content stream paints, and its instruction count.

    The caller has charged ``budget`` for the stream's bytes already.
    """
    import pikepdf

    xobjects = resources.get("/XObject") if resources is not None else None
    area = 0.0
    stack: list[tuple] = []
    instructions = pikepdf.parse_content_stream(stream)
    budget.charge_instructions(len(instructions))
    for instruction in instructions:
        operator = str(instruction.operator)
        if operator == "q":
            stack.append(ctm)
        elif operator == "Q":
            if stack:
                ctm = stack.pop()
        elif operator == "cm":
            ctm = _concat(tuple(float(v) for v in instruction.operands), ctm)
        elif operator == "INLINE IMAGE":
            area += min(limit, abs(ctm[0] * ctm[3] - ctm[1] * ctm[2]))
        elif operator == "Do" and xobjects is not None:
            xobject = xobjects.get(str(instruction.operands[0]))
            if xobject is None:
                continue
            subtype = str(xobject.get("/Subtype"))
            if subtype == "/Image":
                area += min(limit, abs(ctm[0] * ctm[3] - ctm[1] * ctm[2]))
            elif subtype == "/Form" and depth < _MAX_FORM_DEPTH:
                budget.charge_stream(xobject)
                matrix = xobject.get("/Matrix")
                inner = (
                    _concat(tuple(float(v) for v in matrix), ctm)
                    if matrix is not None
                    else ctm
                )
                inner_resources = xobject.get("/Resources")
                sub_area, _ = _image_area(
                    xobject,
                    inner_resources if inner_resources is not None else resources,
                    inner,
                    limit,
                    depth + 1,
                    budget,
                )
                area += sub_area
    return area, len(instructions)


def image_coverage(page) -> tuple[float, bool]:
    """Share of the page covered by images (0..1), and whether it paints anything.

    Overlapping images are counted twice and the sum is capped at 1. That only
    errs towards "scan", which costs a model call but never loses text.
    """
    import pikepdf

    page_area = _page_area(page)
    if page_area <= 0:
        return 0.0, True
    budget = _Budget()
    contents = page.obj.get("/Contents")
    if contents is not None:
        streams = [contents] if isinstance(contents, pikepdf.Stream) else list(contents)
        for stream in streams:
            budget.charge_stream(stream)
    area, count = _image_area(
        page,
        _inherited(page.obj, "/Resources"),
        _IDENTITY,
        page_area,
        0,
        budget,
    )
    return min(1.0, area / page_area), count > 0


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


def classify(text: str, coverage: float, paints: bool, cfg: Config) -> tuple[str, str]:
    """Return (kind, reason) for one PDF page."""
    has_text = len(text) >= cfg.min_text_chars
    if not paints and not has_text:
        return "blank", "empty page"
    if cfg.digital_pages == "gemini":
        return "gemini", "all pages configured for the model"
    if coverage >= cfg.scan_image_coverage:
        return "gemini", f"scan (images cover {coverage:.0%})"
    if not has_text:
        return "gemini", "no text layer"
    if looks_broken(text):
        return "gemini", "text layer unreadable"
    return "embedded", "embedded text"


def analyse_pdf(path: Path, cfg: Config, workdir: Path) -> list[Page]:
    """Classify every page and write a single-page PDF for each one the model reads."""
    import pikepdf

    texts = pdf_page_texts(path)
    try:
        pdf = pikepdf.open(path)
    except Exception as e:
        raise AnalysisError(f"cannot open PDF ({type(e).__name__})") from e

    pages: list[Page] = []
    with pdf:
        total = len(pdf.pages)
        if texts is None or len(texts) != total:
            # Without a reliable per-page text layer every page is read by the model.
            texts = [""] * total
        give_up_at = time.monotonic() + _ANALYSIS_SECONDS
        for index, pdf_page in enumerate(pdf.pages):
            number = index + 1
            text = normalise_text(texts[index])
            try:
                if time.monotonic() > give_up_at:
                    raise _BudgetExceeded
                coverage, paints = image_coverage(pdf_page)
                kind, reason = classify(text, coverage, paints, cfg)
            except Exception as e:
                kind, reason = "gemini", f"page not analysable ({type(e).__name__})"
            page = Page(number=number, kind=kind, reason=reason, embedded_text=text)
            if kind == "gemini":
                page.source_pdf = path
                target = workdir / f"page-{number:05d}.pdf"
                try:
                    single = pikepdf.Pdf.new()
                    single.pages.append(pdf_page)
                    single.save(target)
                    single.close()
                    page.payload = target
                    page.mime_type = "application/pdf"
                except Exception as e:
                    # load_payload renders the page instead.
                    logger.debug("Page %d not splittable: %s", number, type(e).__name__)
            pages.append(page)
    return pages


def render_page(pdf_path: Path, number: int, dpi: int, workdir: Path) -> Path:
    prefix = workdir / f"render-{number:05d}"
    subprocess.run(
        [
            "pdftoppm",
            "-q",
            "-r",
            str(dpi),
            "-jpeg",
            "-f",
            str(number),
            "-l",
            str(number),
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return prefix.with_suffix(".jpg")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _save_jpeg(frame, target: Path) -> None:
    """Write an RGB JPEG that fits into one request, shrinking if it must."""
    image = frame.convert("RGB")
    quality = 90
    for _ in range(6):
        image.save(target, "JPEG", quality=quality)
        if target.stat().st_size <= MAX_INLINE_BYTES:
            return
        if quality > 75:
            quality -= 10
        else:
            image = image.resize((image.width * 3 // 4, image.height * 3 // 4))
    raise AnalysisError("image cannot be reduced to an uploadable size")


def prepare_image(path: Path, mime_type: str, workdir: Path) -> list[Page]:
    """One page per image; multi-page TIFFs give one page per frame."""
    if mime_type in GEMINI_IMAGE_TYPES and path.stat().st_size <= MAX_INLINE_BYTES:
        return [
            Page(number=1, kind="gemini", reason="image", payload=path, mime_type=mime_type),
        ]

    from PIL import Image

    pages: list[Page] = []
    try:
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1) if image.format == "TIFF" else 1
            for index in range(frames):
                image.seek(index)
                target = workdir / f"frame-{index + 1:05d}.jpg"
                _save_jpeg(image, target)
                pages.append(
                    Page(
                        number=index + 1,
                        kind="gemini",
                        reason="image",
                        payload=target,
                        mime_type="image/jpeg",
                    ),
                )
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(f"cannot read image ({type(e).__name__})") from e
    return pages


# ---------------------------------------------------------------------------
# Upload payload
# ---------------------------------------------------------------------------


def load_payload(page: Page, cfg: Config, workdir: Path) -> tuple[bytes, str]:
    """Bytes and MIME type to send for a page. Safe to call from worker threads."""
    is_pdf_page = page.source_pdf is not None
    use_file = (
        page.payload is not None
        and page.payload.stat().st_size <= MAX_INLINE_BYTES
        and not (is_pdf_page and cfg.page_input == "image")
    )
    if use_file:
        return page.payload.read_bytes(), page.mime_type or "application/octet-stream"
    if not is_pdf_page:
        raise AnalysisError("page too large to upload")
    dpi = cfg.render_dpi
    for _ in range(3):
        rendered = render_page(page.source_pdf, page.number, dpi, workdir)
        if rendered.stat().st_size <= MAX_INLINE_BYTES:
            return rendered.read_bytes(), "image/jpeg"
        dpi = max(72, dpi * 2 // 3)
    raise AnalysisError("page too large to upload")
