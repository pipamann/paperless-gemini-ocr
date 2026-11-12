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
import os
import re
import subprocess
import tempfile
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
_DECODED_BYTE_BUDGET = 8 * 1024 * 1024
# Upper bound for the tokens in those streams, counted on the raw bytes before
# parsing: the parser builds one object per instruction.
_TOKEN_BUDGET = 1_500_000
_TOKEN_BREAKS = b" \n\r\t\f\0()<>[]{}/%"
# pdftotext output beyond this size is not read (a real 1000-page document
# stays far below).
_TEXT_OUTPUT_LIMIT = 64 * 1024 * 1024
# An image from this share of the page on is taken to carry content of its
# own; next to it, a short text is not accepted as the page's text.
_SIGNIFICANT_IMAGE = 0.2
_MAX_ANNOTATIONS = 500
# Seconds the page analysis of one document may take; later pages then count
# as scans without being looked at.
_ANALYSIS_SECONDS = 120.0
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
# ASCII85 data beyond this size is not decoded: Python's decoder keeps a list
# entry per four bytes of output, about fifty times the size of the data.
_A85_INPUT_LIMIT = 256 * 1024
# Longer edge of a rendered page in pixels. A3 at 200 dpi has 3300; a page
# box of any size, real plotter sheet or crafted, never makes a larger bitmap.
_MAX_RENDER_EDGE = 5000
_MIN_RENDER_EDGE = 256
# Images: frames of one file, pixels of one frame (larger ones are refused),
# and the size from which a frame is halved before it is converted.
_MAX_FRAMES = 2000
_MAX_FRAME_PIXELS = 120_000_000
_WORK_PIXELS = 40_000_000


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
            # "z" stands for four zero bytes.
            if len(body) > _A85_INPUT_LIMIT or 4 * body.count(b"z") + len(body) > limit:
                return None
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


def _decoded(stream, limit: int) -> bytes | None:
    """Decoded stream data, or None once it may be larger than ``limit``.

    Every filter of the chain is undone with a cap on its output, so nothing
    is ever unpacked beyond ``limit``. A chain with a filter that cannot be
    bounded counts as too large.
    """
    import pikepdf

    if stream.get("/DecodeParms") is not None or stream.get("/DP") is not None:
        # Predictors and the like change the bytes after decoding; a content
        # stream has no use for them, so such a stream is not looked into.
        return None
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
            return None
        decoded = _decode_stage(name, data, limit)
        if decoded is None:
            return None
        data = decoded
    return data if len(data) <= limit else None


class _Budget:
    """Work one page may cause, shared by the page and all forms it paints."""

    def __init__(self) -> None:
        self.instructions = _INSTRUCTION_BUDGET
        self.decoded_bytes = _DECODED_BYTE_BUDGET
        self.tokens = _TOKEN_BUDGET

    def charge_stream(self, stream) -> None:
        """Call before a stream is parsed."""
        data = _decoded(stream, self.decoded_bytes)
        if data is None:
            raise _BudgetExceeded
        self.decoded_bytes -= len(data)
        # Every instruction is followed by white space or a delimiter, so the
        # number of those bytes is an upper bound for the instructions.
        self.tokens -= len(data) - len(data.translate(None, _TOKEN_BREAKS)) + 1
        if self.tokens < 0:
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
    # Longer edge of a PDF page in points (1/72 inch), 0 when unknown.
    long_edge: float = 0.0


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
    bad = text.count("\ufffd") + 5 * text.count("(cid:")
    bad += sum(1 for ch in text if "\ue000" <= ch <= "\uf8ff")
    return bad / len(text) > 0.1


# Words of three letters or more from which local OCR output counts as text.
# Specks, punch holes and scanner streaks give a few stray characters at most.
_TEXT_WORDS = 3
_WORD = re.compile(r"[^\W\d_]{3,}")


def reads_like_text(text: str | None) -> bool:
    """Whether OCR output is more than the noise an empty page produces."""
    if not text:
        return False
    return sum(1 for _ in _WORD.finditer(text)) >= _TEXT_WORDS


def pdf_page_texts(path: Path) -> list[str] | None:
    """Embedded text of every page via one pdftotext run, or None on failure.

    Output size and run time are capped, so a crafted file cannot fill the
    memory through this call.
    """
    try:
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                ["pdftotext", "-q", "-layout", "-enc", "UTF-8", str(path), "-"],
                stdout=out,
                stderr=subprocess.DEVNULL,
            )
            give_up_at = time.monotonic() + 120
            while True:
                try:
                    code = proc.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    too_big = os.fstat(out.fileno()).st_size > _TEXT_OUTPUT_LIMIT
                    if too_big or time.monotonic() > give_up_at:
                        proc.kill()
                        proc.wait()
                        logger.warning("pdftotext stopped: output too large or too slow")
                        return None
            if code != 0 or os.fstat(out.fileno()).st_size > _TEXT_OUTPUT_LIMIT:
                logger.warning("pdftotext failed (exit %s)", code)
                return None
            out.seek(0)
            data = out.read()
    except OSError as e:
        logger.warning("pdftotext failed: %s", type(e).__name__)
        return None
    pieces = data.decode("utf-8", errors="replace").split("\f")
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
            elif subtype == "/Form":
                if depth >= _MAX_FORM_DEPTH:
                    # Not looked at is not the same as empty.
                    raise _BudgetExceeded
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
    annotations = _annotation_area(page, page_area)
    return min(1.0, (area + annotations) / page_area), count > 0 or annotations > 0


def _annotation_area(page, page_area: float) -> float:
    """Area of annotations that may carry page content, e.g. a scan placed as a stamp.

    What an appearance draws is not looked into. Form fields and links count
    by size alone: ordinary ones are small, while a field or link over a
    fifth of the page or more may well hold an image.
    """
    annots = page.obj.get("/Annots")
    if annots is None:
        return 0.0
    area = 0.0
    for index, annot in enumerate(annots):
        if index >= _MAX_ANNOTATIONS:
            raise _BudgetExceeded
        if annot.get("/AP") is None or annot.get("/Rect") is None:
            continue
        subtype = str(annot.get("/Subtype"))
        if subtype == "/Popup":
            continue
        x0, y0, x1, y1 = (float(v) for v in annot.get("/Rect"))
        own = min(page_area, abs((x1 - x0) * (y1 - y0)))
        if subtype in ("/Link", "/Widget") and own < _SIGNIFICANT_IMAGE * page_area:
            continue
        area += own
    return area


def _charge_resource_forms(resources, budget: _Budget, seen: set, depth: int = 0) -> None:
    """Charge every form a resource dictionary lists, painted or not.

    The clean-up of a page copy (``remove_unreferenced_resources``) parses
    all of them, also those the page never paints, which ``image_coverage``
    therefore never saw. It does so once per name, so a form listed under a
    thousand names is charged a thousand times; only the way down into a
    form's own resources is taken once.
    """
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is None:
        return
    for name in xobjects.keys():
        budget.charge_instructions(1)
        xobject = xobjects.get(name)
        if xobject is None or str(xobject.get("/Subtype")) != "/Form":
            continue
        budget.charge_stream(xobject)
        key = xobject.objgen
        if key in seen:
            continue
        seen.add(key)
        if depth >= _MAX_FORM_DEPTH:
            raise _BudgetExceeded
        _charge_resource_forms(xobject.get("/Resources"), budget, seen, depth + 1)


def _long_edge(page) -> float:
    try:
        box = _inherited(page.obj, "/MediaBox") or _inherited(page.obj, "/CropBox")
        x0, y0, x1, y1 = (float(v) for v in box)
        return max(abs(x1 - x0), abs(y1 - y0))
    except Exception:
        return 0.0


def _has_appearances(page) -> bool:
    annots = page.obj.get("/Annots")
    if annots is None:
        return False
    return any(annot.get("/AP") is not None for annot in annots)


def classify(text: str, coverage: float, paints: bool, cfg: Config) -> tuple[str, str]:
    """Return (kind, reason) for one PDF page.

    The costly mistake is to call a scan "embedded": its content would be
    lost. So everything doubtful goes to the model.
    """
    if not paints and not text:
        return "blank", "empty page"
    if cfg.digital_pages == "gemini":
        return "gemini", "all pages configured for the model"
    if coverage >= cfg.scan_image_coverage:
        return "gemini", f"scan (images cover {coverage:.0%})"
    if not text:
        return "gemini", "no text layer"
    if looks_broken(text):
        return "gemini", "text layer unreadable"
    if coverage >= _SIGNIFICANT_IMAGE and len(text) < cfg.min_text_chars:
        return "gemini", f"image ({coverage:.0%} of the page) with little text"
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
            analysed = True
            try:
                if time.monotonic() > give_up_at:
                    raise _BudgetExceeded
                coverage, paints = image_coverage(pdf_page)
                kind, reason = classify(text, coverage, paints, cfg)
            except Exception as e:
                analysed = False
                kind, reason = "gemini", f"page not analysable ({type(e).__name__})"
            page = Page(number=number, kind=kind, reason=reason, embedded_text=text)
            if kind == "gemini":
                page.source_pdf = path
                page.long_edge = _long_edge(pdf_page)
                target = workdir / f"page-{number:05d}.pdf"
                try:
                    if not analysed:
                        # The clean-up below reads the page content without
                        # the limits of the analysis. A page that went past
                        # them is rendered by pdftoppm instead (load_payload).
                        raise AnalysisError("page was not analysable")
                    if _has_appearances(pdf_page):
                        # Annotation appearances may draw with the page's
                        # resources, which the clean-up below cannot see.
                        # Such a page is rendered instead (load_payload).
                        raise AnalysisError("page has annotation appearances")
                    # Raises when the forms need more work than the analysis
                    # allows; the page is rendered then.
                    _charge_resource_forms(
                        _inherited(pdf_page.obj, "/Resources"),
                        _Budget(),
                        set(),
                    )
                    single = pikepdf.Pdf.new()
                    single.pages.append(pdf_page)
                    # Pages often share one resource dictionary. Without this
                    # the upload would carry the images and text of other
                    # pages along, unused but readable.
                    single.pages[0].remove_unreferenced_resources()
                    # Streams are copied as they are, nothing is unpacked.
                    single.save(
                        target,
                        compress_streams=False,
                        stream_decode_level=pikepdf.StreamDecodeLevel.none,
                    )
                    single.close()
                    if target.stat().st_size > MAX_INLINE_BYTES:
                        # Too large to send; no use keeping the copy around.
                        _discard(target)
                        raise AnalysisError("page too large to upload as PDF")
                    page.payload = target
                    page.mime_type = "application/pdf"
                except Exception as e:
                    # load_payload renders the page instead.
                    logger.debug("Page %d not splittable: %s", number, type(e).__name__)
            pages.append(page)
    return pages


def render_edge(long_edge: float, dpi: int) -> int:
    """Pixels for the longer edge of a rendered page: ``dpi``, within fixed bounds."""
    # Unknown size: assume A4.
    wanted = round((long_edge if long_edge > 0 else 842.0) / 72 * dpi)
    return max(_MIN_RENDER_EDGE, min(_MAX_RENDER_EDGE, wanted))


def render_page(
    pdf_path: Path,
    number: int,
    edge: int,
    workdir: Path,
    timeout: float = 180.0,
) -> Path:
    """Render one page as JPEG whose longer edge has ``edge`` pixels.

    The size is given in pixels and not as a resolution, so the bitmap
    pdftoppm allocates does not depend on what the page claims as its size.
    """
    prefix = workdir / f"render-{number:05d}"
    subprocess.run(
        [
            "pdftoppm",
            "-q",
            "-scale-to",
            str(edge),
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
        timeout=max(1.0, timeout),
    )
    return prefix.with_suffix(".jpg")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


# Orientation tag (EXIF) -> PIL.Image.Transpose value that turns the pixels
# upright. Only the turned ones: img2pdf cannot express the mirrored ones in a
# PDF and ignores them, and the model has to see what the archive shows.
_ORIENTATION = {3: 3, 6: 4, 8: 2}


def _upright_rgb(frame):
    """RGB copy of bounded size, as a viewer shows the image.

    The archive PDF is made by img2pdf, which honours the orientation tag,
    and the model has to see the page the same way round. Transparent areas
    become white instead of whatever colour hides under them.
    """
    from PIL import Image

    pixels = frame.width * frame.height
    if pixels > _MAX_FRAME_PIXELS:
        raise AnalysisError("image has too many pixels")
    # Pillow turns a TIFF by its tag while loading and drops the tag; for
    # JPEG and PNG it leaves both alone. Asked after loading, the tag says
    # what is still to be done.
    frame.load()
    try:
        orientation = frame.getexif().get(0x0112)
    except Exception:
        orientation = None
    transparent = frame.mode in ("RGBA", "LA", "PA") or (
        frame.mode == "P" and "transparency" in frame.info
    )
    image = frame
    if pixels > _WORK_PIXELS:
        if image.mode == "1":
            image = image.convert("L")
        factor = 2 if pixels <= 4 * _WORK_PIXELS else 3
        try:
            image = image.reduce(factor)
        except ValueError:
            image = image.resize(
                (max(1, image.width // factor), max(1, image.height // factor)),
                Image.Resampling.NEAREST,
            )
    if transparent:
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")
    method = _ORIENTATION.get(orientation)
    if method is not None:
        image = image.transpose(Image.Transpose(method))
    return image


def _save_jpeg(frame, target: Path) -> None:
    """Write an RGB JPEG that fits into one request, shrinking if it must."""
    image = _upright_rgb(frame)
    # Local OCR, should it be needed for this frame, wants a resolution.
    dpi = frame.info.get("dpi")
    extra = {}
    try:
        # The longer edge stays the longer edge when the image is turned.
        shrunk = max(image.size) / max(frame.size)
        if dpi and float(dpi[0]) > 1 and float(dpi[1]) > 1:
            extra["dpi"] = (
                max(1, round(float(dpi[0]) * shrunk)),
                max(1, round(float(dpi[1]) * shrunk)),
            )
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        extra = {}
    quality = 90
    for _ in range(6):
        image.save(target, "JPEG", quality=quality, **extra)
        if target.stat().st_size <= MAX_INLINE_BYTES:
            return
        if quality > 75:
            quality -= 10
        else:
            image = image.resize(
                (max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)),
            )
            if "dpi" in extra:
                extra["dpi"] = (
                    max(1, extra["dpi"][0] * 3 // 4),
                    max(1, extra["dpi"][1] * 3 // 4),
                )
    raise AnalysisError("image cannot be reduced to an uploadable size")


def prepare_image(path: Path, mime_type: str, workdir: Path) -> list[Page]:
    """One page per image; multi-page TIFFs give one page per frame."""
    if mime_type in GEMINI_IMAGE_TYPES and path.stat().st_size <= MAX_INLINE_BYTES:
        # Sent as it is. The model honours the orientation tag like img2pdf
        # does (checked with gemini-3.8-flash), so both see the same page.
        return [
            Page(number=1, kind="gemini", reason="image", payload=path, mime_type=mime_type),
        ]

    from PIL import Image

    pages: list[Page] = []
    give_up_at = time.monotonic() + _ANALYSIS_SECONDS
    try:
        with Image.open(path) as image:
            multi = image.format == "TIFF"
            index = 0
            while True:
                if time.monotonic() > give_up_at:
                    raise AnalysisError("image takes too long to prepare")
                if index:
                    # Frames are taken one by one. Asking for their number
                    # first would walk the whole file before any limit.
                    if not multi:
                        break
                    try:
                        image.seek(index)
                    except EOFError:
                        break
                    if index >= _MAX_FRAMES:
                        raise AnalysisError(f"image file has more than {_MAX_FRAMES} frames")
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
                index += 1
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(f"cannot read image ({type(e).__name__})") from e
    return pages


# ---------------------------------------------------------------------------
# Upload payload
# ---------------------------------------------------------------------------


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def load_payload(
    page: Page,
    cfg: Config,
    workdir: Path,
    deadline: float | None = None,
) -> tuple[bytes, str]:
    """Bytes and MIME type to send for a page. Safe to call from worker threads.

    ``deadline`` is a ``time.monotonic()`` value; rendering never runs past it.
    The copy or rendering of a PDF page is deleted once it is read, so a long
    document does not sit in the temporary folder a second time while the
    model works. Image files stay: local OCR may need them for a failed page.
    """
    is_pdf_page = page.source_pdf is not None
    use_file = (
        page.payload is not None
        and page.payload.stat().st_size <= MAX_INLINE_BYTES
        and not (is_pdf_page and cfg.page_input == "image")
    )
    if use_file:
        data = page.payload.read_bytes()
        if is_pdf_page:
            _discard(page.payload)
        return data, page.mime_type or "application/octet-stream"
    if not is_pdf_page:
        raise AnalysisError("page too large to upload")
    dpi = cfg.render_dpi
    for _ in range(3):
        timeout = 180.0
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 1:
                raise AnalysisError("document time limit reached while rendering")
        rendered = render_page(
            page.source_pdf,
            page.number,
            render_edge(page.long_edge, dpi),
            workdir,
            timeout,
        )
        fits = rendered.stat().st_size <= MAX_INLINE_BYTES
        data = rendered.read_bytes() if fits else b""
        _discard(rendered)
        if fits:
            return data, "image/jpeg"
        dpi = max(72, dpi * 2 // 3)
    raise AnalysisError("page too large to upload")
