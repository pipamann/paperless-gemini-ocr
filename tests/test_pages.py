from __future__ import annotations

import pikepdf
import pytest
from PIL import Image

from paperless_gemini_ocr.config import Config
from paperless_gemini_ocr.pages import _concat
from paperless_gemini_ocr.pages import analyse_pdf
from paperless_gemini_ocr.pages import load_payload
from paperless_gemini_ocr.pages import looks_broken
from paperless_gemini_ocr.pages import normalise_text
from paperless_gemini_ocr.pages import prepare_image

CFG = Config(api_key="k")


def kinds(pages):
    return [p.kind for p in pages]


def test_scan_page_goes_to_model(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].mime_type == "application/pdf"
    with pikepdf.open(pages[0].payload) as single:
        assert len(single.pages) == 1


def test_scanner_text_layer_does_not_hide_a_scan(pdf_factory, workdir):
    """A ScanSnap-style PDF has plenty of text, but it is still a scan."""
    pages = analyse_pdf(pdf_factory(["scan_text"]), CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert "12345" in pages[0].embedded_text


def test_born_digital_page_keeps_embedded_text(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["text"]), CFG, workdir)
    assert kinds(pages) == ["embedded"]
    assert pages[0].embedded_text == "Invoice 12345 total 99.50 EUR net"
    assert pages[0].payload is None


def test_logo_does_not_make_a_page_a_scan(pdf_factory, workdir):
    assert kinds(analyse_pdf(pdf_factory(["text_logo"]), CFG, workdir)) == ["embedded"]


def test_blank_page(pdf_factory, workdir):
    assert kinds(analyse_pdf(pdf_factory(["blank"]), CFG, workdir)) == ["blank"]


def test_mixed_document_keeps_page_order(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["text", "scan", "blank", "scan_text"]), CFG, workdir)
    assert kinds(pages) == ["embedded", "gemini", "blank", "gemini"]
    assert [p.number for p in pages] == [1, 2, 3, 4]
    assert pages[1].payload != pages[3].payload


def test_image_inside_form_xobject_counts(pdf_factory, workdir):
    assert kinds(analyse_pdf(pdf_factory(["form_scan"]), CFG, workdir)) == ["gemini"]


def test_resources_inherited_from_pages_node(pdf_factory, workdir, tmp_path):
    source = pdf_factory(["scan"])
    target = tmp_path / "inherited.pdf"
    with pikepdf.open(source) as pdf:
        page = pdf.pages[0]
        pdf.Root.Pages["/Resources"] = page.obj["/Resources"]
        del page.obj["/Resources"]
        pdf.save(target)
    assert kinds(analyse_pdf(target, CFG, workdir)) == ["gemini"]


def test_text_page_without_text_layer_goes_to_model(pdf_factory, workdir):
    """Too little text to count as a layer, but the page is not empty."""
    pages = analyse_pdf(pdf_factory(["text"], text="Hi"), CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].reason == "no text layer"


def test_all_pages_to_model_when_configured(pdf_factory, workdir):
    cfg = Config(api_key="k", digital_pages="gemini")
    assert kinds(analyse_pdf(pdf_factory(["text", "blank"]), cfg, workdir)) == ["gemini", "blank"]


def test_broken_text_layer_detected():
    assert looks_broken("�" * 30 + "abc")
    assert looks_broken("(cid:12)(cid:7)(cid:99)")
    assert not looks_broken("Rechnung Nr. 4711 vom 01.02.2026")
    assert not looks_broken("")


def test_normalise_text_matches_paperless():
    # Trailing blanks inside the text stay, exactly as in post_process_text.
    assert normalise_text("  a   b \n   c\0d  ") == "a b \nc d"
    assert normalise_text(None) == ""


def test_matrix_concat_scales_then_moves():
    scale = (2.0, 0.0, 0.0, 3.0, 0.0, 0.0)
    move = (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)
    assert _concat(scale, move) == (2.0, 0.0, 0.0, 3.0, 10.0, 20.0)


def test_jpeg_is_sent_as_it_is(tmp_path, workdir):
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    pages = prepare_image(path, "image/jpeg", workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].payload == path
    data, mime = load_payload(pages[0], CFG, workdir)
    assert mime == "image/jpeg"
    assert data == path.read_bytes()


def test_multi_page_tiff_gives_one_page_per_frame(tmp_path, workdir):
    path = tmp_path / "fax.tif"
    frames = [Image.new("L", (60, 80), shade) for shade in (255, 128, 0)]
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:])
    pages = prepare_image(path, "image/tiff", workdir)
    assert [p.number for p in pages] == [1, 2, 3]
    assert all(p.mime_type == "image/jpeg" for p in pages)
    assert len({p.payload for p in pages}) == 3


def test_page_rendered_as_image_when_configured(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan", "scan"]), CFG, workdir)
    cfg = Config(api_key="k", page_input="image", render_dpi=72)
    data, mime = load_payload(pages[1], cfg, workdir)
    assert mime == "image/jpeg"
    assert data[:2] == b"\xff\xd8"


def test_unreadable_pdf_raises(tmp_path, workdir):
    from paperless_gemini_ocr.pages import AnalysisError

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(AnalysisError):
        analyse_pdf(path, CFG, workdir)


def test_nested_forms_with_huge_fan_out_stop_at_the_budget():
    """Six levels of forms, each calling the next 60 times: 60^6 calls if unbounded."""
    import time

    from pikepdf import Dictionary
    from pikepdf import Name

    from paperless_gemini_ocr.pages import _BudgetExceeded
    from paperless_gemini_ocr.pages import image_coverage

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    calls = b"/Fm0 Do " * 60
    inner = pikepdf.Stream(pdf, b"", Type=Name.XObject, Subtype=Name.Form, BBox=[0, 0, 1, 1])
    for _ in range(6):
        inner = pikepdf.Stream(
            pdf,
            calls,
            Type=Name.XObject,
            Subtype=Name.Form,
            BBox=[0, 0, 1, 1],
            Resources=Dictionary(XObject=Dictionary(Fm0=inner)),
        )
    page = pdf.pages[0]
    page.obj["/Resources"] = Dictionary(XObject=Dictionary(Fm0=inner))
    page.obj["/Contents"] = pikepdf.Stream(pdf, calls)

    started = time.monotonic()
    with pytest.raises(_BudgetExceeded):
        image_coverage(page)
    assert time.monotonic() - started < 20


def test_stream_that_inflates_hugely_is_never_unpacked():
    """100 kB that inflate to 100 MB of operators: refused before parsing."""
    import time
    import zlib

    from pikepdf import Name

    from paperless_gemini_ocr.pages import _BudgetExceeded
    from paperless_gemini_ocr.pages import image_coverage

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    bomb = zlib.compress(b"q Q " * 25_000_000, 9)
    assert len(bomb) < 200_000
    pdf.pages[0].obj["/Contents"] = pikepdf.Stream(pdf, bomb, Filter=Name.FlateDecode)

    started = time.monotonic()
    with pytest.raises(_BudgetExceeded):
        image_coverage(pdf.pages[0])
    assert time.monotonic() - started < 5


def test_contents_array_and_compressed_streams_still_work(pdf_factory, workdir, tmp_path):
    """Real PDFs split and compress their content streams."""
    source = pdf_factory(["scan"])
    target = tmp_path / "split.pdf"
    with pikepdf.open(source) as pdf:
        page = pdf.pages[0]
        page.obj["/Contents"] = pikepdf.Array(
            [pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm"), pikepdf.Stream(pdf, b"/Im0 Do Q")],
        )
        pdf.save(target, compress_streams=True)
    pages = analyse_pdf(target, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].reason.startswith("scan")


def _coverage_of(content: bytes, **stream_keys):
    from paperless_gemini_ocr.pages import image_coverage

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    pdf.pages[0].obj["/Contents"] = pikepdf.Stream(pdf, content, **stream_keys)
    return image_coverage(pdf.pages[0])


def test_chained_filters_cannot_hide_a_bomb():
    import zlib

    from pikepdf import Name

    from paperless_gemini_ocr.pages import _BudgetExceeded

    twice = zlib.compress(zlib.compress(b"q Q " * 25_000_000, 9), 9)
    assert len(twice) < 20_000
    with pytest.raises(_BudgetExceeded):
        _coverage_of(twice, Filter=pikepdf.Array([Name.FlateDecode, Name.FlateDecode]))


def test_filter_without_size_bound_is_refused():
    from pikepdf import Name

    from paperless_gemini_ocr.pages import _BudgetExceeded

    with pytest.raises(_BudgetExceeded):
        _coverage_of(b"\x80\x0b\x60\x50\x22\x0c\x0c\x85\x01", Filter=Name.LZWDecode)


def test_ascii85_then_flate_content_is_analysed():
    """An old but legitimate way to store page content."""
    import base64
    import zlib

    from pikepdf import Name

    encoded = base64.a85encode(zlib.compress(b"q 1 0 0 1 0 0 cm Q"), adobe=True)[2:]
    coverage, paints = _coverage_of(
        encoded,
        Filter=pikepdf.Array([Name.ASCII85Decode, Name.FlateDecode]),
    )
    assert (coverage, paints) == (0.0, True)


def test_run_length_decoding_is_bounded():
    from paperless_gemini_ocr.pages import _run_length_decode

    assert _run_length_decode(bytes([2, 65, 66, 67, 254, 90, 128]), 100) == b"ABCZZZ"
    assert _run_length_decode(bytes([129, 0]) * 1000, 1000) is None
