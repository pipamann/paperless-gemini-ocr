from __future__ import annotations

from pathlib import Path

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
from paperless_gemini_ocr.pages import probe_text
from paperless_gemini_ocr.pages import reads_like_text
from paperless_gemini_ocr.pages import turn_payload

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


def test_short_born_digital_page_keeps_its_text(pdf_factory, workdir):
    """A page that only says "Approved" is still exact text, not a scan."""
    pages = analyse_pdf(pdf_factory(["text"], text="Approved"), CFG, workdir)
    assert kinds(pages) == ["embedded"]
    assert pages[0].embedded_text == "Approved"


def test_drawing_without_text_goes_to_model(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["vector"]), CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].reason == "no text layer"


def test_large_image_with_a_caption_goes_to_model(pdf_factory, workdir):
    """A photographed receipt with a page number below it: the image is the content."""
    pages = analyse_pdf(pdf_factory(["text_photo"], text="Seite 3"), CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert "little text" in pages[0].reason


def test_large_figure_in_a_text_page_stays_embedded(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["text_photo"], text="Bericht " * 40), CFG, workdir)
    assert kinds(pages) == ["embedded"]


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


def test_exported_page_does_not_carry_other_pages_along(tmp_path, workdir):
    """Pages sharing one resource dictionary: only the page's own objects are uploaded."""
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import add_page

    pdf = pikepdf.Pdf.new()
    add_page(pdf, "scan")
    add_page(pdf, "text", "Other page with a secret IBAN DE00 1111 2222")
    secret = pikepdf.Stream(
        pdf,
        b"BT /F1 12 Tf 50 700 Td (CONFIDENTIAL-OTHER-PAGE) Tj ET",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=[0, 0, 595, 842],
    )
    shared = pdf.pages[0].obj["/Resources"]
    shared["/XObject"]["/FmOther"] = secret
    shared["/Font"] = pdf.pages[1].obj["/Resources"]["/Font"]
    pdf.pages[1].obj["/Resources"] = shared
    path = tmp_path / "shared.pdf"
    pdf.save(path)

    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini", "embedded"]
    with pikepdf.open(pages[0].payload) as single:
        resources = single.pages[0].obj["/Resources"]
        assert "/FmOther" not in resources.get("/XObject", {})
    assert b"CONFIDENTIAL-OTHER-PAGE" not in pages[0].payload.read_bytes()


def test_scan_hidden_in_deeply_nested_forms_is_not_called_digital(tmp_path, workdir):
    """Seven levels of forms around the page image: unseen is not the same as empty."""
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    inner = pikepdf.Stream(
        pdf,
        b"/Im0 Do",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=[0, 0, 1, 1],
        Resources=Dictionary(XObject=Dictionary(Im0=_image_xobject(pdf))),
    )
    for _ in range(7):
        inner = pikepdf.Stream(
            pdf,
            b"/Fm0 Do",
            Type=Name.XObject,
            Subtype=Name.Form,
            BBox=[0, 0, 1, 1],
            Resources=Dictionary(XObject=Dictionary(Fm0=inner)),
        )
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
    page = pdf.pages[0]
    page.obj["/Resources"] = Dictionary(XObject=Dictionary(Fm0=inner), Font=Dictionary(F1=font))
    page.obj["/Contents"] = pikepdf.Stream(
        pdf,
        b"q 595 0 0 842 0 0 cm /Fm0 Do Q BT 3 Tr /F1 12 Tf 50 700 Td (scanner text layer of this page) Tj ET",
    )
    path = tmp_path / "nested.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]


def test_scan_placed_as_a_stamp_annotation_goes_to_model(tmp_path, workdir):
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    appearance = pikepdf.Stream(
        pdf,
        b"q 595 0 0 842 0 0 cm /Im0 Do Q",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=[0, 0, 595, 842],
        Resources=Dictionary(XObject=Dictionary(Im0=_image_xobject(pdf))),
    )
    pdf.pages[0].obj["/Annots"] = pikepdf.Array(
        [
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Stamp,
                    Rect=[0, 0, 595, 842],
                    AP=Dictionary(N=appearance),
                ),
            ),
        ],
    )
    path = tmp_path / "stamp.pdf"
    pdf.save(path)
    assert kinds(analyse_pdf(path, CFG, workdir)) == ["gemini"]


def test_many_tiny_instructions_are_refused_before_parsing():
    """Under the byte limit, but millions of instructions: one object each if parsed."""
    from paperless_gemini_ocr.pages import _BudgetExceeded

    with pytest.raises(_BudgetExceeded):
        _coverage_of(b"q Q " * 1_000_000)


def _annotated_page(tmp_path, subtype, rect, inline_image=False, text="caption below the picture"):
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject
    from .conftest import add_page

    pdf = pikepdf.Pdf.new()
    add_page(pdf, "text", text)
    if inline_image:
        resources = Dictionary()
        content = b"q 595 0 0 842 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI Q"
    else:
        resources = Dictionary(XObject=Dictionary(Im0=_image_xobject(pdf)))
        content = b"q 595 0 0 842 0 0 cm /Im0 Do Q"
    appearance = pikepdf.Stream(
        pdf,
        content,
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=[0, 0, 595, 842],
        Resources=resources,
    )
    pdf.pages[0].obj["/Annots"] = pikepdf.Array(
        [
            pdf.make_indirect(
                Dictionary(
                    Type=Name.Annot,
                    Subtype=Name(subtype),
                    Rect=rect,
                    AP=Dictionary(N=appearance),
                ),
            ),
        ],
    )
    path = tmp_path / "annotated.pdf"
    pdf.save(path)
    return path


@pytest.mark.parametrize("inline_image", [False, True])
def test_scan_inside_a_page_sized_form_field_goes_to_model(tmp_path, workdir, inline_image):
    path = _annotated_page(tmp_path, "/Widget", [0, 0, 595, 842], inline_image=inline_image)
    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    # Not exported as a PDF page: it is rendered, annotation included.
    assert pages[0].payload is None
    data, mime = load_payload(pages[0], Config(api_key="k", render_dpi=72), workdir)
    assert mime == "image/jpeg"


def test_ordinary_form_fields_do_not_make_a_page_a_scan(tmp_path, workdir):
    pages = analyse_pdf(_annotated_page(tmp_path, "/Widget", [50, 600, 300, 620]), CFG, workdir)
    assert kinds(pages) == ["embedded"]


def test_page_past_the_analysis_limits_is_rendered_not_exported(tmp_path, workdir):
    """Exporting would read the page content again, without the limits."""
    import zlib

    from pikepdf import Name

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    bomb = zlib.compress(b"q Q " * 25_000_000, 9)
    pdf.pages[0].obj["/Contents"] = pikepdf.Stream(pdf, bomb, Filter=Name.FlateDecode)
    path = tmp_path / "bomb.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert "not analysable" in pages[0].reason
    assert pages[0].payload is None


def test_stream_with_predictor_is_not_looked_into():
    from pikepdf import Dictionary
    from pikepdf import Name

    from paperless_gemini_ocr.pages import _BudgetExceeded

    with pytest.raises(_BudgetExceeded):
        _coverage_of(
            b"x\x9c+T\x08\x04\x00\x01\xe5\x00\xf3",
            Filter=Name.FlateDecode,
            DecodeParms=Dictionary(Predictor=2),
        )


def test_rendering_respects_the_document_time_limit(pdf_factory, workdir):
    import time

    from paperless_gemini_ocr.pages import AnalysisError

    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    cfg = Config(api_key="k", page_input="image")
    with pytest.raises(AnalysisError, match="time limit"):
        load_payload(pages[0], cfg, workdir, deadline=time.monotonic() + 0.5)


# ---------------------------------------------------------------------------
# Limits added after the multi-model review
# ---------------------------------------------------------------------------


def test_ascii85_cannot_blow_up_while_decoding():
    from paperless_gemini_ocr.pages import _decode_stage

    limit = 8 * 1024 * 1024
    # "z" is four zero bytes: small input, large output and a larger work list.
    assert _decode_stage("/ASCII85Decode", b"z" * (3 * 1024 * 1024), limit) is None
    assert _decode_stage("/A85", b"z" * (300 * 1024), limit) is None
    assert _decode_stage("/A85", b"87cURD]i,\"Ebo80~>", limit) == b"Hello World!"


def test_unused_form_in_the_resources_keeps_the_page_from_being_exported(tmp_path, workdir):
    """The clean-up of a page copy would unpack a form the page never paints."""
    import zlib

    from pikepdf import Array
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    page = pdf.pages[0]
    form = pikepdf.Stream(pdf, b"")
    form.write(zlib.compress(b"q Q " * (4 * 1024 * 1024), 9), filter=Name.FlateDecode)
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 1, 1])
    page.obj["/Resources"] = Dictionary(
        XObject=Dictionary(Im0=_image_xobject(pdf), Unused=form),
    )
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm /Im0 Do Q\n")
    path = tmp_path / "unused-form.pdf"
    pdf.save(path)

    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].reason.startswith("scan")
    # Rendered by pdftoppm instead of exported with pikepdf.
    assert pages[0].payload is None
    data, mime = load_payload(pages[0], CFG, workdir)
    assert mime == "image/jpeg"


def test_small_unused_form_does_not_prevent_the_export(tmp_path, workdir):
    from pikepdf import Array
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    page = pdf.pages[0]
    form = pikepdf.Stream(pdf, b"q Q", Type=Name.XObject, Subtype=Name.Form, BBox=Array([0, 0, 1, 1]))
    page.obj["/Resources"] = Dictionary(
        XObject=Dictionary(Im0=_image_xobject(pdf), Unused=form),
    )
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm /Im0 Do Q\n")
    path = tmp_path / "small-form.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert pages[0].payload is not None


def test_rendered_size_does_not_follow_a_huge_page_box(tmp_path, workdir):
    """A page that claims 1400 inches is rendered at a fixed pixel size."""
    from paperless_gemini_ocr.pages import render_edge

    assert render_edge(842, 200) == 2339  # A4 at 200 dpi
    assert render_edge(100_000, 200) == 5000
    assert render_edge(0, 200) == 2339
    assert render_edge(20, 200) == 256

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    pdf.pages[0].obj["/MediaBox"] = pikepdf.Array([0, 0, 100_000, 70_000])
    pdf.pages[0].obj["/Contents"] = pikepdf.Stream(pdf, b"0 0 m 90000 60000 l S\n")
    path = tmp_path / "huge-box.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, Config(api_key="k", page_input="image"), workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].long_edge == 100_000
    data, mime = load_payload(pages[0], Config(api_key="k", page_input="image"), workdir)
    assert mime == "image/jpeg"
    import io

    with Image.open(io.BytesIO(data)) as rendered:
        assert max(rendered.size) == 5000


def test_tiff_with_too_many_frames_is_refused(tmp_path, workdir, monkeypatch):
    from paperless_gemini_ocr import pages as pages_module
    from paperless_gemini_ocr.pages import AnalysisError

    monkeypatch.setattr(pages_module, "_MAX_FRAMES", 3)
    frames = [Image.new("1", (20, 20), 1) for _ in range(4)]
    path = tmp_path / "many.tif"
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:])
    with pytest.raises(AnalysisError, match="more than 3 frames"):
        prepare_image(path, "image/tiff", workdir)
    # Exactly at the limit is fine.
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:3])
    assert len(prepare_image(path, "image/tiff", workdir)) == 3


def test_image_with_too_many_pixels_is_refused(tmp_path, workdir, monkeypatch):
    from paperless_gemini_ocr import pages as pages_module
    from paperless_gemini_ocr.pages import AnalysisError

    monkeypatch.setattr(pages_module, "_MAX_FRAME_PIXELS", 10_000)
    path = tmp_path / "big.tif"
    Image.new("1", (200, 200), 1).save(path, "TIFF")
    with pytest.raises(AnalysisError, match="too many pixels"):
        prepare_image(path, "image/tiff", workdir)


def test_large_frame_is_halved_before_conversion(tmp_path, workdir, monkeypatch):
    from paperless_gemini_ocr import pages as pages_module

    monkeypatch.setattr(pages_module, "_WORK_PIXELS", 10_000)
    path = tmp_path / "fax.tif"
    Image.new("1", (200, 150), 1).save(path, "TIFF", dpi=(300, 300))
    pages = prepare_image(path, "image/tiff", workdir)
    with Image.open(pages[0].payload) as frame:
        assert frame.size == (100, 75)
        assert frame.info["dpi"] == (150, 150)


def test_transparent_areas_become_white(tmp_path, workdir):
    """Black text on a transparent black background must not turn black on black."""
    image = Image.new("RGBA", (60, 80), (0, 0, 0, 0))
    for x in range(20, 40):
        image.putpixel((x, 40), (0, 0, 0, 255))
    path = tmp_path / "transparent.tif"
    image.save(path, "TIFF")
    pages = prepare_image(path, "image/tiff", workdir)
    with Image.open(pages[0].payload) as frame:
        assert frame.getpixel((5, 5))[0] > 240
        assert frame.getpixel((30, 40))[0] < 60


def test_converted_image_is_shown_as_its_orientation_tag_says(tmp_path, workdir):
    """img2pdf turns the archive page by the tag; the model must see the same."""
    image = Image.new("RGB", (60, 80), "white")
    for x in range(30):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 0))  # mark the top left corner
    path = tmp_path / "tagged.tif"
    image.save(path, "TIFF", tiffinfo={274: 3})  # shown turned by 180 degrees
    pages = prepare_image(path, "image/tiff", workdir)
    with Image.open(pages[0].payload) as frame:
        assert frame.size == (60, 80)
        assert frame.getpixel((2, 2))[0] > 200
        assert frame.getpixel((57, 77))[0] < 60


def test_jpeg_orientation_tag_is_applied_when_the_image_is_converted():
    """Pillow leaves JPEG pixels as stored, so the plugin turns them itself."""
    import io

    from paperless_gemini_ocr.pages import _upright_rgb

    image = Image.new("RGB", (60, 80), "white")
    for x in range(30):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # shown turned a quarter clockwise
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95, exif=exif)
    buffer.seek(0)
    with Image.open(buffer) as stored:
        shown = _upright_rgb(stored)
    assert shown.size == (80, 60)
    # The mark from the top left corner is now at the top right.
    assert shown.getpixel((77, 2))[0] < 60
    assert shown.getpixel((2, 2))[0] > 200


def test_frame_resolution_is_kept_for_local_ocr(tmp_path, workdir):
    path = tmp_path / "scan.tif"
    Image.new("L", (60, 80), 255).save(path, "TIFF", dpi=(300, 300))
    pages = prepare_image(path, "image/tiff", workdir)
    with Image.open(pages[0].payload) as frame:
        assert frame.info["dpi"] == (300, 300)


def test_mirrored_orientation_tags_are_left_alone():
    """img2pdf ignores them for the archive, so the model sees the stored pixels too."""
    import io

    from paperless_gemini_ocr.pages import _upright_rgb

    image = Image.new("RGB", (60, 80), "white")
    for x in range(30):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 4  # mirrored top to bottom
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95, exif=exif)
    buffer.seek(0)
    with Image.open(buffer) as stored:
        shown = _upright_rgb(stored)
    assert shown.getpixel((2, 2))[0] < 60


def test_form_listed_under_many_names_is_charged_every_time(tmp_path, workdir):
    """The clean-up parses a form once per name, so the budget counts it that often."""
    import zlib

    from pikepdf import Array
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    page = pdf.pages[0]
    form = pikepdf.Stream(pdf, b"")
    form.write(zlib.compress(b"q Q " * 25_000, 9), filter=Name.FlateDecode)  # 100 kB
    form["/Type"] = Name.XObject
    form["/Subtype"] = Name.Form
    form["/BBox"] = Array([0, 0, 1, 1])
    xobjects = Dictionary(Im0=_image_xobject(pdf))
    for index in range(200):  # 200 x 100 kB is past the 8 MiB budget
        xobjects[f"/Alias{index}"] = form
    page.obj["/Resources"] = Dictionary(XObject=xobjects)
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm /Im0 Do Q\n")
    path = tmp_path / "aliases.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].payload is None


def test_forms_sharing_the_page_resources_can_still_be_exported(tmp_path, workdir):
    """A form whose resources are the page's own dictionary is no endless loop."""
    from pikepdf import Array
    from pikepdf import Dictionary
    from pikepdf import Name

    from .conftest import _image_xobject

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    page = pdf.pages[0]
    resources = pdf.make_indirect(Dictionary(XObject=Dictionary(Im0=_image_xobject(pdf))))
    form = pikepdf.Stream(
        pdf,
        b"q 1 0 0 1 0 0 cm /Im0 Do Q",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=Array([0, 0, 1, 1]),
        Resources=resources,
    )
    resources["/XObject"]["/Fm0"] = form
    page.obj["/Resources"] = resources
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm /Im0 Do Q\n")
    path = tmp_path / "shared-resources.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert kinds(pages) == ["gemini"]
    assert pages[0].payload is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        (None, False),
        (" . | — ,, 1 7 ii", False),
        ("Seite 3 von 5", False),
        ("Invoice 12345 total 99.50 EUR net", False),
        ("Invoice 12345 of the Example Company, total 99.50 EUR net", True),
        # What a full OCR run made of two blank, speckled scan pages.
        ((Path(__file__).parent / "noise_sample.txt").read_text(encoding="utf-8"), False),
        ("Sehr geehrte Damen und Herren, anbei die Rechnung", True),
        ("חשבונית מספר שלוש מאות עשרים וחמש", True),
    ],
)
def test_reads_like_text(text, expected):
    assert reads_like_text(text) is expected


def test_copy_of_a_pdf_page_is_deleted_once_it_is_read(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan", "scan"]), CFG, workdir)
    copy = pages[0].payload
    assert copy.is_file()
    data, mime = load_payload(pages[0], CFG, workdir)
    assert mime == "application/pdf"
    assert data.startswith(b"%PDF")
    assert not copy.exists()
    assert pages[1].payload.is_file()


def test_rendered_page_is_deleted_once_it_is_read(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    data, mime = load_payload(pages[0], Config(api_key="k", page_input="image"), workdir)
    assert mime == "image/jpeg"
    assert list(workdir.glob("render-*")) == []


def test_image_file_stays_for_local_ocr(tmp_path, workdir):
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    pages = prepare_image(path, "image/jpeg", workdir)
    load_payload(pages[0], CFG, workdir)
    assert path.is_file()


def test_page_copy_too_large_to_send_is_not_kept(pdf_factory, workdir, monkeypatch):
    from paperless_gemini_ocr import pages as pages_module

    monkeypatch.setattr(pages_module, "MAX_INLINE_BYTES", 100)
    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    assert pages[0].kind == "gemini"
    assert pages[0].payload is None
    assert list(workdir.glob("page-*")) == []


needs_tesseract = pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None,
    reason="tesseract is not installed",
)


def written_page(lines):
    from PIL import ImageDraw
    from PIL import ImageFont

    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    for index, line in enumerate(lines):
        draw.text((150, 200 + 90 * index), line, fill="black", font=font)
    return image


LINES = [
    "Example Company Limited, 12 Sample Road",
    "Invoice number 2026-4711 for the delivery in October",
    "Total amount 486.12 Euro, payable within fourteen days",
    "With kind regards, the accounting department",
]


@needs_tesseract
def test_quick_look_finds_text_on_an_image(tmp_path, workdir):
    path = tmp_path / "letter.jpg"
    written_page(LINES).save(path, "JPEG")
    pages = prepare_image(path, "image/jpeg", workdir)
    seen = probe_text(pages[0], workdir, "eng")
    assert reads_like_text(seen)
    assert "Invoice" in seen
    assert list(workdir.glob("probe-*")) == []


@needs_tesseract
def test_quick_look_finds_nothing_on_a_speckled_blank_page(tmp_path, workdir):
    import random

    image = Image.new("L", (1654, 2339), 255)
    rng = random.Random(7)
    for _ in range(6000):
        image.putpixel((rng.randrange(1654), rng.randrange(2339)), rng.randrange(60, 160))
    path = tmp_path / "blank.png"
    image.save(path, "PNG")
    pages = prepare_image(path, "image/png", workdir)
    seen = probe_text(pages[0], workdir, "eng")
    assert seen is not None
    assert not reads_like_text(seen)


@needs_tesseract
def test_quick_look_reads_a_pdf_page(tmp_path, workdir):
    import io

    buffer = io.BytesIO()
    written_page(LINES).save(buffer, "JPEG")
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    pdf.add_blank_page(page_size=(595, 842))
    image = pikepdf.Stream(
        pdf,
        buffer.getvalue(),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=1654,
        Height=2339,
        ColorSpace=pikepdf.Name.DeviceRGB,
        BitsPerComponent=8,
        Filter=pikepdf.Name.DCTDecode,
    )
    page = pdf.pages[1]
    page.obj["/Resources"] = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image))
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 595 0 0 842 0 0 cm /Im0 Do Q")
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert reads_like_text(probe_text(pages[1], workdir, "eng"))
    # An odd language setting does not end up on the command line.
    assert reads_like_text(probe_text(pages[1], workdir, "eng; rm -rf /"))


def test_quick_look_that_cannot_run_gives_no_answer(tmp_path, workdir, monkeypatch):
    from paperless_gemini_ocr import pages as pages_module

    def missing(*args, **kwargs):
        raise FileNotFoundError("tesseract")

    monkeypatch.setattr(pages_module.subprocess, "run", missing)
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    pages = prepare_image(path, "image/jpeg", workdir)
    assert probe_text(pages[0], workdir, "eng") is None


def test_quick_look_without_a_requested_language_gives_no_answer(tmp_path, workdir, monkeypatch):
    """Tesseract exits with 0 after skipping a language it cannot load."""
    from types import SimpleNamespace

    from paperless_gemini_ocr import pages as pages_module

    def partly_loaded(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=b"",
            stderr=b"Failed loading language 'heb'\n",
        )

    monkeypatch.setattr(pages_module.subprocess, "run", partly_loaded)
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    pages = prepare_image(path, "image/jpeg", workdir)
    assert probe_text(pages[0], workdir, "eng+heb") is None


@needs_tesseract
def test_quick_look_keeps_small_print_on_a_large_page(tmp_path, workdir):
    """A3 at 300 dpi with print of about six points."""
    import io

    from PIL import ImageDraw
    from PIL import ImageFont

    image = Image.new("L", (3508, 4961), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25)
    for index, line in enumerate(LINES):
        draw.text((300, 400 + 60 * index), line, fill=0, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(842, 1191))
    stream = pikepdf.Stream(
        pdf,
        buffer.getvalue(),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=3508,
        Height=4961,
        ColorSpace=pikepdf.Name.DeviceGray,
        BitsPerComponent=8,
        Filter=pikepdf.Name.DCTDecode,
    )
    page = pdf.pages[0]
    page.obj["/Resources"] = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=stream))
    page.obj["/Contents"] = pikepdf.Stream(pdf, b"q 842 0 0 1191 0 0 cm /Im0 Do Q")
    path = tmp_path / "a3.pdf"
    pdf.save(path)
    pages = analyse_pdf(path, CFG, workdir)
    assert reads_like_text(probe_text(pages[0], workdir, "eng"))


def test_pdf_page_is_turned_by_its_rotate_entry(pdf_factory, workdir):
    import io

    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    data, mime = load_payload(pages[0], CFG, workdir)
    once, mime_once = turn_payload(data, mime, 90)
    twice, _ = turn_payload(once, mime_once, 270)
    assert mime_once == "application/pdf"
    with pikepdf.open(io.BytesIO(once)) as pdf:
        assert pdf.pages[0].obj["/Rotate"] == 90
    with pikepdf.open(io.BytesIO(twice)) as pdf:
        assert pdf.pages[0].obj["/Rotate"] == 0


@pytest.mark.parametrize(
    ("degrees", "size", "corner"),
    [
        # The dark corner starts top left; clockwise turns move it right, then down.
        (90, (80, 60), (79, 0)),
        (180, (60, 80), (59, 79)),
        (270, (80, 60), (0, 59)),
    ],
)
def test_image_is_turned_clockwise(degrees, size, corner):
    import io

    image = Image.new("RGB", (60, 80), "white")
    for x in range(12):
        for y in range(12):
            image.putpixel((x, y), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    data, mime = turn_payload(buffer.getvalue(), "image/png", degrees)
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(data)) as turned:
        assert turned.size == size
        assert sum(turned.convert("L").getpixel(corner) for _ in range(1)) < 60
        assert turned.convert("L").getpixel((size[0] // 2, size[1] // 2)) > 200


def test_turning_by_another_angle_is_refused():
    with pytest.raises(ValueError):
        turn_payload(b"", "application/pdf", 45)
