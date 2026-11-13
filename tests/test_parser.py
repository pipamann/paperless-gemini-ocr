"""The parser's decisions, against stand-ins for Paperless (see conftest)."""

from __future__ import annotations

import pikepdf
import pytest
from PIL import Image

from paperless_gemini_ocr import parser as plugin
from paperless_gemini_ocr import state
from paperless_gemini_ocr.gemini import PageResult
from paperless_gemini_ocr.pipeline import DocumentResult

from .conftest import ParseError


@pytest.fixture
def configured(monkeypatch, stub_parser):
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_API_KEY", "k")
    state.take_consume_flag(None)
    # The quick Tesseract look at pages the model calls empty: sees nothing
    # unless a test says otherwise, and never starts a real Tesseract here.
    monkeypatch.setattr(plugin, "probe_text", lambda page, workdir, languages: "")
    return stub_parser


def model_answers(monkeypatch, results=None, failures=None):
    calls = []

    def fake_read_pages(pages, cfg, workdir, client=None):
        calls.append([p.number for p in pages if p.kind == "gemini"])
        return DocumentResult(results=results or {}, failures=failures or {}, seconds=1.0)

    monkeypatch.setattr(plugin, "read_pages", fake_read_pages)
    return calls


def consume(path, mime="application/pdf", produce_archive=True):
    """What the Paperless consumer does around parse()."""
    from documents.signals import document_consumption_started

    document_consumption_started.send(sender=None, filename=path)
    parser = plugin.GeminiOcrParser()
    parser.parse(path, mime, produce_archive=produce_archive)
    return parser


def test_not_configured_means_invisible(stub_parser):
    assert plugin.GeminiOcrParser.score("application/pdf", "a.pdf") is None


def test_configured_outranks_builtin_parsers(configured):
    assert plugin.GeminiOcrParser.score("application/pdf", "a.pdf") == 30
    assert plugin.GeminiOcrParser.score("text/plain", "a.txt") is None
    assert plugin.GeminiOcrParser.uses_remote_service is True


def test_born_digital_document_never_reaches_the_model(configured, monkeypatch, pdf_factory):
    calls = model_answers(monkeypatch)
    parser = consume(pdf_factory(["text", "text"]))
    assert calls == []
    assert configured.calls == ["auto"]
    assert parser.get_text() == "local text (auto)"
    assert state.pop_report() is None


def test_scan_gets_model_text_without_running_tesseract(configured, monkeypatch, pdf_factory):
    calls = model_answers(
        monkeypatch,
        results={
            1: PageResult(text="Seite eins", rotation=0),
            2: PageResult(text="Seite zwei", rotation=0),
        },
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    assert calls == [[1, 2]]
    assert configured.calls == ["off"]
    assert parser.settings.mode == "auto"
    assert parser.get_text() == "Seite eins\n\nSeite zwei"
    assert parser.archive_path.is_file()
    assert state.pop_report() is None


def test_mixed_document_combines_embedded_and_model_text(configured, monkeypatch, pdf_factory):
    model_answers(monkeypatch, results={2: PageResult(text="Unterschriftsseite", rotation=0)})
    parser = consume(pdf_factory(["text", "scan"]))
    assert parser.get_text() == "Invoice 12345 total 99.50 EUR net\n\nUnterschriftsseite"


def test_local_ocr_always_keeps_tesseract_running(configured, monkeypatch, pdf_factory):
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    model_answers(monkeypatch, results={1: PageResult(text="Modelltext", rotation=180)})
    parser = consume(pdf_factory(["scan"]))
    assert configured.calls == ["auto"]
    assert parser.get_text() == "Modelltext"
    # Rotation is left to OCRmyPDF when it ran.
    with pikepdf.open(parser.archive_path) as archive:
        assert "/Rotate" not in archive.pages[0].obj


def test_upside_down_page_is_turned_in_the_archive_only(configured, monkeypatch, pdf_factory):
    source = pdf_factory(["scan", "scan"])
    original = source.read_bytes()
    model_answers(
        monkeypatch,
        results={
            1: PageResult(text="a", rotation=180),
            2: PageResult(text="b", rotation=0),
        },
    )
    parser = consume(source)
    with pikepdf.open(parser.archive_path) as archive:
        assert int(archive.pages[0].obj["/Rotate"]) == 180
        assert "/Rotate" not in archive.pages[1].obj
    assert source.read_bytes() == original


def test_rotation_adds_to_an_existing_rotation(configured, monkeypatch, pdf_factory, tmp_path):
    source = pdf_factory(["scan"])
    turned = tmp_path / "turned.pdf"
    with pikepdf.open(source) as pdf:
        pdf.pages[0].obj["/Rotate"] = 270
        pdf.save(turned)
    model_answers(monkeypatch, results={1: PageResult(text="a", rotation=180)})
    parser = consume(turned)
    with pikepdf.open(parser.archive_path) as archive:
        assert int(archive.pages[0].obj["/Rotate"]) == 90


def test_failed_page_falls_back_to_local_ocr_and_is_reported(configured, monkeypatch, pdf_factory):
    model_answers(
        monkeypatch,
        results={1: PageResult(text="gut gelesen", rotation=0)},
        failures={2: "HTTP 503 after 3 attempt(s)"},
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    # First the OCR-free run, then forced OCR for the failed page only.
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == "gut gelesen\n\nlocal text (force)"
    assert parser.settings.mode == "auto"
    # The archive of the import itself is still there and is the OCR-free one.
    with pikepdf.open(parser.archive_path) as archive:
        assert len(archive.pages) == 2
    report = state.pop_report()
    assert report.failures == {2: "HTTP 503 after 3 attempt(s)"}
    assert "page(s) 2 of 2" in report.note()


def test_several_failed_pages_get_their_own_ocr_text(configured, monkeypatch, pdf_factory):
    model_answers(
        monkeypatch,
        results={2: PageResult(text="Seite zwei vom Modell", rotation=0)},
        failures={1: "timeout", 3: "timeout"},
    )
    parser = consume(pdf_factory(["scan", "scan", "scan"]))
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == (
        "tesseract text of page 1 in this file\n\n"
        "Seite zwei vom Modell\n\n"
        "tesseract text of page 2 in this file"
    )


def test_fallback_ocr_runs_even_when_no_archive_is_wanted(configured, monkeypatch, pdf_factory):
    """Paperless calls a scan with a scanner text layer born-digital: no archive."""
    model_answers(monkeypatch, failures={1: "timeout", 2: "timeout"})
    parser = consume(pdf_factory(["scan_text", "scan_text"]), produce_archive=False)
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == (
        "tesseract text of page 1 in this file\n\ntesseract text of page 2 in this file"
    )
    assert parser.archive_path is None
    assert not (parser.tempdir / "archive.pdf").exists()


def test_import_survives_when_fallback_ocr_fails_too(configured, monkeypatch, pdf_factory):
    configured.fail_modes = {"force"}
    model_answers(
        monkeypatch,
        results={1: PageResult(text="gut gelesen", rotation=0)},
        failures={2: "timeout"},
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == "gut gelesen"
    assert parser.archive_path.is_file()
    assert parser.settings.mode == "auto"
    assert state.pop_report() is not None


def test_scanner_text_layer_is_used_when_ocr_yields_nothing(configured, monkeypatch, pdf_factory):
    configured.fail_modes = {"force"}
    model_answers(monkeypatch, failures={1: "timeout"})
    parser = consume(pdf_factory(["scan_text"]))
    assert parser.get_text() == "Invoice 12345 total 99.50 EUR net"


def test_no_fallback_ocr_when_ocr_is_switched_off(configured, monkeypatch, pdf_factory):
    configured.configured_mode = "off"
    model_answers(monkeypatch, failures={1: "timeout"})
    parser = consume(pdf_factory(["scan_text"]))
    assert configured.calls == ["off"]
    assert parser.get_text() == "Invoice 12345 total 99.50 EUR net"
    assert state.pop_report() is not None


def test_single_image_failure_uses_local_text(configured, monkeypatch, tmp_path):
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    model_answers(monkeypatch, failures={1: "empty answer after 3 attempt(s)"})
    parser = consume(path, "image/jpeg")
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == "local text (force)"
    assert parser.archive_path.is_file()
    assert state.pop_report().failures == {1: "empty answer after 3 attempt(s)"}


def test_crash_in_the_model_run_counts_as_all_pages_failed(configured, monkeypatch, pdf_factory):
    def explode(pages, cfg, workdir, client=None):
        raise RuntimeError("thread pool gone")

    monkeypatch.setattr(plugin, "read_pages", explode)
    parser = consume(pdf_factory(["scan", "scan"]))
    assert parser.get_text() == (
        "tesseract text of page 1 in this file\n\ntesseract text of page 2 in this file"
    )
    assert sorted(state.pop_report().failures) == [1, 2]


def test_no_second_rotation_when_ocrmypdf_made_the_archive(configured, monkeypatch, pdf_factory):
    """The OCR-free run failed, the normal one produced the archive and turns pages itself."""
    configured.fail_modes = {"off"}
    model_answers(monkeypatch, results={1: PageResult(text="a", rotation=180)})
    parser = consume(pdf_factory(["scan"]))
    assert configured.calls == ["off", "auto"]
    with pikepdf.open(parser.archive_path) as archive:
        assert "/Rotate" not in archive.pages[0].obj


def test_ocr_free_run_failing_retries_with_ocr(configured, monkeypatch, pdf_factory):
    configured.fail_modes = {"off"}
    model_answers(monkeypatch, results={1: PageResult(text="Modelltext", rotation=0)})
    parser = consume(pdf_factory(["scan"]))
    assert configured.calls == ["off", "auto"]
    assert parser.get_text() == "Modelltext"


def test_local_failure_without_plugin_equivalent_still_fails_the_import(
    configured,
    monkeypatch,
    pdf_factory,
):
    configured.fail_modes = {"off", "auto"}
    model_answers(monkeypatch, results={1: PageResult(text="Modelltext", rotation=0)})
    with pytest.raises(ParseError):
        consume(pdf_factory(["scan"]))


def test_reprocess_fails_loudly_instead_of_degrading(configured, monkeypatch, pdf_factory):
    model_answers(monkeypatch, failures={1: "HTTP 503 after 3 attempt(s)"})
    parser = plugin.GeminiOcrParser()
    with pytest.raises(ParseError, match="content left unchanged"):
        parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)
    assert state.pop_report() is None


def test_reprocess_succeeds_when_the_model_does(configured, monkeypatch, pdf_factory):
    model_answers(monkeypatch, results={1: PageResult(text="neu gelesen", rotation=0)})
    parser = plugin.GeminiOcrParser()
    parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)
    assert parser.get_text() == "neu gelesen"


def test_stale_consume_announcement_does_not_count(configured, monkeypatch, pdf_factory):
    """A consumption that died before parse() must not make a later reprocess lenient."""
    from documents.signals import document_consumption_started

    document_consumption_started.send(sender=None, filename="/tmp/some-other-file.pdf")
    model_answers(monkeypatch, failures={1: "timeout"})
    parser = plugin.GeminiOcrParser()
    with pytest.raises(ParseError):
        parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)


def test_unreadable_document_is_imported_locally_and_reported(configured, monkeypatch, tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    calls = model_answers(monkeypatch)
    parser = consume(path)
    assert calls == []
    assert configured.calls == ["auto"]
    assert parser.get_text() == "local text (auto)"
    assert "could not process this document" in state.pop_report().note()


def test_finished_signal_flags_the_document(configured, monkeypatch, pdf_factory):
    from documents.signals import document_consumption_finished

    flagged = []
    monkeypatch.setattr(plugin, "_flag_document", lambda doc, report: flagged.append((doc, report)))
    model_answers(monkeypatch, failures={1: "timeout"})
    consume(pdf_factory(["scan"]))

    class Doc:
        pk = 7

    document_consumption_finished.send(sender=None, document=Doc)
    assert len(flagged) == 1
    assert flagged[0][1].failures == {1: "timeout"}
    # Only once: the report is gone afterwards.
    document_consumption_finished.send(sender=None, document=Doc)
    assert len(flagged) == 1


def test_finished_signal_without_failures_does_nothing(configured, monkeypatch, pdf_factory):
    from documents.signals import document_consumption_finished

    flagged = []
    monkeypatch.setattr(plugin, "_flag_document", lambda doc, report: flagged.append(doc))
    model_answers(monkeypatch, results={1: PageResult(text="ok", rotation=0)})
    consume(pdf_factory(["scan"]))
    document_consumption_finished.send(sender=None, document=object())
    assert flagged == []


def test_flagging_errors_never_break_the_import(configured, monkeypatch, pdf_factory):
    from documents.signals import document_consumption_finished

    def explode(doc, report):
        raise RuntimeError("database down")

    monkeypatch.setattr(plugin, "_flag_document", explode)
    model_answers(monkeypatch, failures={1: "timeout"})
    consume(pdf_factory(["scan"]))

    class Doc:
        pk = 7

    document_consumption_finished.send(sender=None, document=Doc)


def test_sideways_pages_are_only_turned_when_asked(configured, monkeypatch, pdf_factory):
    answers = {1: PageResult(text="a", rotation=90)}
    model_answers(monkeypatch, results=answers)
    parser = consume(pdf_factory(["scan"]))
    with pikepdf.open(parser.archive_path) as archive:
        assert "/Rotate" not in archive.pages[0].obj

    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_ROTATE", "all")
    parser = consume(pdf_factory(["scan"], name="second.pdf"))
    with pikepdf.open(parser.archive_path) as archive:
        assert int(archive.pages[0].obj["/Rotate"]) == 90


def test_rotation_can_be_switched_off(configured, monkeypatch, pdf_factory):
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_ROTATE", "off")
    model_answers(monkeypatch, results={1: PageResult(text="a", rotation=180)})
    parser = consume(pdf_factory(["scan"]))
    with pikepdf.open(parser.archive_path) as archive:
        assert "/Rotate" not in archive.pages[0].obj


def test_first_ocr_run_is_kept_when_tesseract_always_runs(configured, monkeypatch, tmp_path):
    """LOCAL_OCR=always: the import's own OCR text is the fallback, no second pass."""
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    configured.fail_modes = {"force"}
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    model_answers(monkeypatch, failures={1: "timeout"})
    parser = consume(path, "image/jpeg")
    assert configured.calls == ["auto"]
    assert parser.get_text() == "local text (auto)"


def test_fallback_pass_cannot_touch_the_archive_of_the_import(configured, monkeypatch, pdf_factory):
    """The forced pass works in a parser of its own, with its own folder."""
    model_answers(monkeypatch, failures={1: "timeout"})
    source = pdf_factory(["scan", "scan"])
    parser = consume(source)
    with pikepdf.open(parser.archive_path) as archive:
        # Still the two-page archive of the import, not the one-page OCR output.
        assert len(archive.pages) == 2
    assert not (parser.tempdir / "gemini-ocr-first-pass-archive.pdf").exists()


def test_always_mode_still_forces_ocr_for_failed_pdf_pages(configured, monkeypatch, pdf_factory):
    """Text Paperless found somewhere in a PDF does not prove a page was OCRed."""
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    model_answers(monkeypatch, failures={2: "timeout"})
    parser = consume(pdf_factory(["text", "scan_text"]))
    assert configured.calls == ["auto", "force"]
    assert parser.get_text() == "Invoice 12345 total 99.50 EUR net\n\nlocal text (force)"


def test_model_rotation_applies_when_local_ocr_does_not_turn_pages(
    configured,
    monkeypatch,
    pdf_factory,
):
    """LOCAL_OCR=always with PAPERLESS_OCR_ROTATE_PAGES off: nobody else turns the page."""
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    configured.configured_rotate = False
    model_answers(monkeypatch, results={1: PageResult(text="a", rotation=180)})
    parser = consume(pdf_factory(["scan"]))
    with pikepdf.open(parser.archive_path) as archive:
        assert archive.pages[0].obj["/Rotate"] == 180


def tiff_with_frames(path, count):
    frames = [Image.new("RGB", (60, 80), "white") for _ in range(count)]
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:])
    return path


def test_failed_tiff_frame_gets_its_own_ocr_text(configured, monkeypatch, tmp_path):
    """Only the failed frame is read locally; the others keep the model's text, in order."""
    model_answers(
        monkeypatch,
        results={
            1: PageResult(text="eins", rotation=0),
            3: PageResult(text="drei", rotation=0),
        },
        failures={2: "timeout"},
    )
    parser = consume(tiff_with_frames(tmp_path / "fax.tif", 3), "image/tiff")
    assert parser.get_text() == "eins\n\ntesseract text of frame-00002.jpg\n\ndrei"
    assert [name for mode, name in configured.parsed if mode == "force"] == ["frame-00002.jpg"]


def test_several_failed_tiff_frames_keep_their_places(configured, monkeypatch, tmp_path):
    model_answers(
        monkeypatch,
        results={2: PageResult(text="zwei", rotation=0)},
        failures={1: "timeout", 3: "timeout"},
    )
    parser = consume(tiff_with_frames(tmp_path / "fax.tif", 3), "image/tiff")
    assert parser.get_text() == (
        "tesseract text of frame-00001.jpg\n\nzwei\n\ntesseract text of frame-00003.jpg"
    )


def test_foreign_exception_text_stays_out_of_the_report(configured, monkeypatch, tmp_path):
    """Only the kind of an unexpected error is kept, not what it says."""

    def explode(path, cfg, workdir):
        raise RuntimeError("secret detail from a library")

    monkeypatch.setattr(plugin, "analyse_pdf", explode)
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")
    consume(path)
    report = state.pop_report()
    assert report.document_error == "RuntimeError"
    assert "secret" not in report.note()


def test_rotation_setting_from_user_args_counts(configured, monkeypatch, pdf_factory):
    """PAPERLESS_OCR_USER_ARGS can switch OCRmyPDF's rotation on over the regular setting."""
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    configured.configured_rotate = False
    configured.configured_user_args = {"rotate_pages": True}
    model_answers(monkeypatch, results={1: PageResult(text="a", rotation=180)})
    parser = consume(pdf_factory(["scan"]))
    with pikepdf.open(parser.archive_path) as archive:
        assert "/Rotate" not in archive.pages[0].obj


LETTER = "Sehr geehrte Damen und Herren, anbei die Rechnung"
DOUBT = "the model found no text on the page, local OCR did"
UNCHECKED = "the model found no text on the page, local OCR could not check it"


def probe_sees(monkeypatch, answers):
    """Let the quick look return ``answers[page number]`` and record what it was asked."""
    asked = []

    def fake_probe(page, workdir, languages):
        asked.append((page.number, page.payload.name if page.payload else None, languages))
        return answers.get(page.number, "")

    monkeypatch.setattr(plugin, "probe_text", fake_probe)
    return asked


def test_page_called_empty_is_checked_with_local_ocr(configured, monkeypatch, pdf_factory):
    """A page full of text that the model calls empty must not be imported empty."""
    asked = probe_sees(monkeypatch, {2: LETTER})
    model_answers(
        monkeypatch,
        results={1: PageResult(text="eins", rotation=0), 2: PageResult(text="", rotation=0)},
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    assert [number for number, _, _ in asked] == [2]
    # The quick look only decides; the text comes from a proper OCR pass.
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == "eins\n\nlocal text (force)"
    assert state.pop_report().failures == {2: DOUBT}


def test_empty_page_stays_empty_without_any_ocr_run(configured, monkeypatch, pdf_factory):
    probe_sees(monkeypatch, {2: " . | — \n ,, 1"})
    model_answers(
        monkeypatch,
        results={1: PageResult(text="eins", rotation=0), 2: PageResult(text="", rotation=180)},
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    assert configured.calls == ["off"]
    assert parser.get_text() == "eins"
    assert state.pop_report() is None
    # The page is still the model's: its rotation counts.
    with pikepdf.open(parser.archive_path) as archive:
        assert archive.pages[1].obj["/Rotate"] == 180


def test_specks_on_a_blank_page_are_not_text(configured, monkeypatch, pdf_factory):
    """What OCR makes of scanner noise: stray marks with a few accidental words."""
    from pathlib import Path

    noise = (Path(__file__).parent / "noise_sample.txt").read_text(encoding="utf-8")
    probe_sees(monkeypatch, {1: noise})
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = consume(pdf_factory(["scan"]))
    assert parser.get_text() == ""
    assert state.pop_report() is None


def test_page_that_could_not_be_checked_is_flagged(configured, monkeypatch, pdf_factory):
    """No answer from the quick look says nothing about the page."""
    probe_sees(monkeypatch, {1: None})
    configured.force_text = ""
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = consume(pdf_factory(["scan"]))
    assert parser.get_text() == ""
    assert state.pop_report().failures == {1: UNCHECKED}


def test_empty_tiff_frame_is_checked_on_its_own(configured, monkeypatch, tmp_path):
    asked = probe_sees(monkeypatch, {2: LETTER})
    model_answers(
        monkeypatch,
        results={1: PageResult(text="eins", rotation=0), 2: PageResult(text="", rotation=0)},
    )
    parser = consume(tiff_with_frames(tmp_path / "fax.tif", 2), "image/tiff")
    assert [(number, name) for number, name, _ in asked] == [(2, "frame-00002.jpg")]
    assert [name for mode, name in configured.parsed if mode == "force"] == ["frame-00002.jpg"]
    assert parser.get_text() == "eins\n\ntesseract text of frame-00002.jpg"


def test_reprocess_does_not_replace_text_by_an_empty_page(configured, monkeypatch, pdf_factory):
    probe_sees(monkeypatch, {1: LETTER})
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = plugin.GeminiOcrParser()
    with pytest.raises(ParseError, match="content left unchanged"):
        parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)
    assert configured.calls == ["off"]
    assert state.pop_report() is None


def test_reprocess_stops_when_the_check_fails(configured, monkeypatch, pdf_factory):
    probe_sees(monkeypatch, {1: None})
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = plugin.GeminiOcrParser()
    with pytest.raises(ParseError, match="content left unchanged"):
        parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)


def test_reprocess_of_a_really_empty_page_works(configured, monkeypatch, pdf_factory):
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = plugin.GeminiOcrParser()
    parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)
    assert parser.get_text() == ""


def test_refused_reprocess_skips_the_empty_page_check(configured, monkeypatch, pdf_factory):
    asked = probe_sees(monkeypatch, {1: LETTER})
    model_answers(
        monkeypatch,
        results={1: PageResult(text="", rotation=0)},
        failures={2: "timeout"},
    )
    parser = plugin.GeminiOcrParser()
    with pytest.raises(ParseError, match=r"page\(s\) 2;"):
        parser.parse(pdf_factory(["scan", "scan"]), "application/pdf", produce_archive=True)
    assert asked == []
    assert configured.calls == ["off"]


def test_empty_page_is_believed_when_local_ocr_is_switched_off(configured, monkeypatch, pdf_factory):
    configured.configured_mode = "off"
    asked = probe_sees(monkeypatch, {1: LETTER})
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = consume(pdf_factory(["scan"]))
    assert asked == []
    assert configured.calls == ["off"]
    assert parser.get_text() == ""
    assert state.pop_report() is None


def test_quick_look_uses_the_ocr_languages_of_the_installation(configured, monkeypatch, pdf_factory):
    asked = probe_sees(monkeypatch, {})
    monkeypatch.setattr(plugin.GeminiOcrParser, "settings", None, raising=False)
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = plugin.GeminiOcrParser()
    parser.settings.language = "deu+eng"
    parser.parse(pdf_factory(["scan"]), "application/pdf", produce_archive=True)
    assert asked == [(1, "page-00001.pdf", "deu+eng")]


def test_text_of_the_first_ocr_run_is_not_replaced_by_noise(configured, monkeypatch, pdf_factory):
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    configured.force_text = " . | "
    monkeypatch.setattr(
        plugin.GeminiOcrParser,
        "_first_pass_texts",
        lambda self, pages, failed: {1: LETTER},
    )
    model_answers(monkeypatch, failures={1: "timeout"})
    parser = consume(pdf_factory(["scan"]))
    assert parser.get_text() == LETTER


def test_without_page_boundaries_every_page_is_read_alone(configured, monkeypatch, pdf_factory):
    """Otherwise the text of a later page would be taken for the first one's."""
    monkeypatch.setattr(plugin, "pdf_page_texts", lambda path: None)
    model_answers(
        monkeypatch,
        results={2: PageResult(text="zwei", rotation=0)},
        failures={1: "timeout", 3: "timeout"},
    )
    path = pdf_factory(["scan", "scan", "scan"])
    # Three widths, so that the pages can be told apart in what OCR is given.
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for index, page in enumerate(pdf.pages):
            page.obj["/MediaBox"] = [0, 0, 601 + index, 842]
        pdf.save(path)
    parser = consume(path)
    assert configured.calls == ["off", "force", "force", "force"]
    # Both failed pages together first, then each one alone: never page 2.
    assert configured.forced_widths == [[601, 603], [601], [603]]
    assert parser.get_text() == "local text (force)\n\nzwei\n\nlocal text (force)"


def test_too_many_pages_without_boundaries_keep_their_text_once(configured, monkeypatch, pdf_factory):
    monkeypatch.setattr(plugin, "pdf_page_texts", lambda path: None)
    monkeypatch.setattr(plugin, "_SINGLE_PAGE_PASSES", 1)
    model_answers(monkeypatch, failures={1: "timeout", 2: "timeout"})
    parser = consume(pdf_factory(["scan", "scan"]))
    assert configured.calls == ["off", "force"]
    assert parser.get_text() == "local text (force)"


def test_fallback_ocr_stops_at_the_page_limit_of_the_installation(configured, monkeypatch, pdf_factory):
    """PAPERLESS_OCR_PAGES=1: OCR reads one page, so only one is handed over."""
    configured.configured_pages = 1
    model_answers(monkeypatch, failures={1: "timeout", 2: "timeout"})
    parser = consume(pdf_factory(["scan", "scan"]))
    assert configured.forced_widths == [[595]]
    assert parser.get_text() == "local text (force)"


def test_short_text_layer_is_not_given_up_for_an_empty_answer(configured, monkeypatch, pdf_factory):
    """Below the size at which the layer alone decides, but plainly a sentence."""
    asked = probe_sees(monkeypatch, {})
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    configured.fail_modes = {"force"}
    parser = consume(pdf_factory(["scan_text"], text="Bitte senden Sie uns die Rechnung."))
    assert asked == []
    assert parser.get_text() == "Bitte senden Sie uns die Rechnung."
    assert state.pop_report().failures == {1: "the model found no text on a page that has some"}


def test_text_of_the_first_ocr_run_counts_against_an_empty_answer(
    configured,
    monkeypatch,
    pdf_factory,
):
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_LOCAL_OCR", "always")
    asked = probe_sees(monkeypatch, {})
    configured.force_text = ""
    monkeypatch.setattr(
        plugin.GeminiOcrParser,
        "_first_pass_texts",
        lambda self, pages, failed: {1: LETTER},
    )
    model_answers(monkeypatch, results={1: PageResult(text="", rotation=0)})
    parser = consume(pdf_factory(["scan"]))
    assert asked == []
    assert parser.get_text() == LETTER
    assert state.pop_report() is not None


def test_confirmed_sideways_rotation_is_applied_in_the_default_setting(
    configured,
    monkeypatch,
    pdf_factory,
):
    """Unconfirmed, 90 and 270 are left alone; a second reading settles which it is."""
    model_answers(
        monkeypatch,
        results={
            1: PageResult(text="quer, bestätigt", rotation=90, rotation_confirmed=True),
            2: PageResult(text="quer, unbestätigt", rotation=270),
        },
    )
    parser = consume(pdf_factory(["scan", "scan"]))
    with pikepdf.open(parser.archive_path) as archive:
        assert archive.pages[0].obj["/Rotate"] == 90
        assert "/Rotate" not in archive.pages[1].obj
