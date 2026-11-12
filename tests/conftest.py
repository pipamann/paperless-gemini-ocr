"""Test helpers: synthetic PDFs and stand-ins for the Paperless modules.

The stand-ins let the parser's decisions (fallback, reprocess, rotation,
flagging) be tested without a Paperless installation. The real integration
is covered by the tests in ``integration/``.
"""

from __future__ import annotations

import io
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pikepdf
import pytest
from pikepdf import Dictionary
from pikepdf import Name
from PIL import Image

A4 = (595, 842)


# ---------------------------------------------------------------------------
# Synthetic PDFs
# ---------------------------------------------------------------------------


def _image_xobject(pdf: pikepdf.Pdf) -> pikepdf.Stream:
    buffer = io.BytesIO()
    Image.new("L", (100, 140), 200).save(buffer, "JPEG")
    return pikepdf.Stream(
        pdf,
        buffer.getvalue(),
        Type=Name.XObject,
        Subtype=Name.Image,
        Width=100,
        Height=140,
        ColorSpace=Name.DeviceGray,
        BitsPerComponent=8,
        Filter=Name.DCTDecode,
    )


def _font(pdf: pikepdf.Pdf):
    return pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica),
    )


def add_page(pdf: pikepdf.Pdf, kind: str, text: str = "") -> None:
    """Append a page: scan, scan_text, text, text_logo, blank, form_scan."""
    pdf.add_blank_page(page_size=A4)
    page = pdf.pages[-1]
    full = b"q 595 0 0 842 0 0 cm /Im0 Do Q\n"
    visible = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET\n".encode()
    hidden = f"BT 3 Tr /F1 12 Tf 50 700 Td ({text}) Tj ET\n".encode()
    resources = Dictionary(
        XObject=Dictionary(Im0=_image_xobject(pdf)),
        Font=Dictionary(F1=_font(pdf)),
    )
    if kind == "blank":
        return
    if kind == "scan":
        content = full
    elif kind == "scan_text":
        content = full + hidden
    elif kind == "text":
        content = visible
    elif kind == "text_logo":
        content = b"q 100 0 0 50 40 760 cm /Im0 Do Q\n" + visible
    elif kind == "form_scan":
        form = pikepdf.Stream(
            pdf,
            b"q 1 0 0 1 0 0 cm /Im0 Do Q",
            Type=Name.XObject,
            Subtype=Name.Form,
            BBox=[0, 0, 1, 1],
            Resources=Dictionary(XObject=Dictionary(Im0=_image_xobject(pdf))),
        )
        resources = Dictionary(XObject=Dictionary(Fm0=form))
        content = b"q 595 0 0 842 0 0 cm /Fm0 Do Q\n"
    else:
        raise ValueError(kind)
    page.obj["/Resources"] = resources
    page.obj["/Contents"] = pikepdf.Stream(pdf, content)


def make_pdf(path: Path, kinds: list[str], text: str = "Invoice 12345 total 99.50 EUR net") -> Path:
    pdf = pikepdf.Pdf.new()
    for kind in kinds:
        add_page(pdf, kind, text)
    pdf.save(path)
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def factory(kinds: list[str], name: str = "doc.pdf", **kwargs) -> Path:
        return make_pdf(tmp_path / name, kinds, **kwargs)

    return factory


@pytest.fixture
def workdir(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# Stand-ins for Paperless
# ---------------------------------------------------------------------------


class ParseError(Exception):
    pass


class _Signal:
    def __init__(self) -> None:
        self.receivers: list = []

    def connect(self, receiver, dispatch_uid=None) -> None:
        self.receivers.append(receiver)

    def send(self, sender=None, **kwargs) -> None:
        for receiver in self.receivers:
            receiver(sender, **kwargs)


class _ModeChoices:
    OFF = "off"
    AUTO = "auto"


class StubTesseractParser:
    """Behaves like the built-in parser as far as the plugin can tell."""

    calls: list[str] = []
    fail_modes: set[str] = set()

    def __init__(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp(prefix="stub-parser-"))
        self.settings = SimpleNamespace(mode=_ModeChoices.AUTO)
        self.text: str | None = None
        self.archive_path: Path | None = None
        self.log = logging.getLogger("stub")

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    @classmethod
    def supported_mime_types(cls) -> dict[str, str]:
        return {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/tiff": ".tif"}

    def get_text(self) -> str:
        return self.text or ""

    def parse(self, document_path, mime_type, *, produce_archive=True) -> None:
        mode = self.settings.mode
        type(self).calls.append(mode)
        if mode in type(self).fail_modes:
            raise ParseError(f"stub failure in mode {mode}")
        self.text = f"local text ({mode})"
        if produce_archive:
            archive = self.tempdir / "archive.pdf"
            if mime_type == "application/pdf":
                shutil.copy(document_path, archive)
            else:
                make_pdf(archive, ["scan"])
            self.archive_path = archive


def _install_stubs() -> None:
    documents = types.ModuleType("documents")
    parsers = types.ModuleType("documents.parsers")
    parsers.ParseError = ParseError
    signals = types.ModuleType("documents.signals")
    signals.document_consumption_started = _Signal()
    signals.document_consumption_finished = _Signal()
    paperless = types.ModuleType("paperless")
    paperless_parsers = types.ModuleType("paperless.parsers")
    tesseract = types.ModuleType("paperless.parsers.tesseract")
    tesseract.RasterisedDocumentParser = StubTesseractParser
    models = types.ModuleType("paperless.models")
    models.ModeChoices = _ModeChoices
    for name, module in {
        "documents": documents,
        "documents.parsers": parsers,
        "documents.signals": signals,
        "paperless": paperless,
        "paperless.parsers": paperless_parsers,
        "paperless.parsers.tesseract": tesseract,
        "paperless.models": models,
    }.items():
        sys.modules.setdefault(name, module)


_install_stubs()


@pytest.fixture
def stub_parser():
    StubTesseractParser.calls = []
    StubTesseractParser.fail_modes = set()
    return StubTesseractParser


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("PAPERLESS_GEMINI_OCR_"):
            monkeypatch.delenv(key)
