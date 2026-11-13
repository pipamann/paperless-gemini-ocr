from __future__ import annotations

import threading

from paperless_gemini_ocr.config import Config
from paperless_gemini_ocr.gemini import GeminiError
from paperless_gemini_ocr.gemini import PageResult
from paperless_gemini_ocr.pages import analyse_pdf
from paperless_gemini_ocr.pipeline import assemble
from paperless_gemini_ocr.pipeline import load_prompt
from paperless_gemini_ocr.pipeline import read_pages

CFG = Config(api_key="k", concurrency=4)


class FakeClient:
    """Answers per page number; an exception instance is raised instead."""

    def __init__(self, answers: dict[int, object]) -> None:
        self.answers = answers
        self.prompts: list[str] = []
        self.lock = threading.Lock()
        self.calls = 0

    def transcribe(self, data, mime_type, prompt, deadline=None):
        with self.lock:
            self.calls += 1
            self.prompts.append(prompt)
            call = self.calls
        answer = self.answers.get(call, self.answers.get("default"))
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_only_scan_pages_are_sent_and_text_stays_in_page_order(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["text", "scan", "blank", "scan"]), CFG, workdir)
    client = FakeClient({"default": PageResult(text="from the model", rotation=0)})
    outcome = read_pages(pages, Config(api_key="k", concurrency=1), workdir, client)
    assert client.calls == 2
    assert sorted(outcome.results) == [2, 4]
    assert outcome.failures == {}
    assert assemble(pages, outcome) == (
        "Invoice 12345 total 99.50 EUR net\n\nfrom the model\n\nfrom the model"
    )


def test_failed_page_is_reported_and_filled_from_local_text(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan", "scan"]), CFG, workdir)
    client = FakeClient(
        {
            1: PageResult(text="page one", rotation=0),
            2: GeminiError("HTTP 503", retryable=True, attempts=3),
        },
    )
    outcome = read_pages(pages, Config(api_key="k", concurrency=1), workdir, client)
    assert sorted(outcome.results) == [1]
    assert outcome.failures == {2: "HTTP 503 after 3 attempt(s)"}
    assert assemble(pages, outcome, {2: "tesseract text"}) == "page one\n\ntesseract text"
    assert assemble(pages, outcome) == "page one"


def test_unexpected_error_does_not_escape(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    outcome = read_pages(pages, CFG, workdir, FakeClient({"default": RuntimeError("boom")}))
    assert outcome.results == {}
    assert "RuntimeError" in outcome.failures[1]


def test_nothing_to_read(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["text"]), CFG, workdir)
    client = FakeClient({})
    outcome = read_pages(pages, CFG, workdir, client)
    assert client.calls == 0
    assert outcome.results == {}


def test_custom_prompt_file(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Mein Prompt\n", encoding="utf-8")
    assert load_prompt(Config(prompt_file=str(prompt))) == "Mein Prompt"


def test_missing_prompt_file_falls_back_to_builtin(tmp_path):
    builtin = load_prompt(Config())
    assert "faithful OCR transcription" in builtin
    assert load_prompt(Config(prompt_file=str(tmp_path / "missing.txt"))) == builtin


def test_setup_failure_counts_as_all_pages_failed(pdf_factory, workdir, monkeypatch):
    from paperless_gemini_ocr import pipeline

    def broken(cfg):
        raise RuntimeError("no certificates")

    monkeypatch.setattr(pipeline, "GeminiClient", broken)
    pages = analyse_pdf(pdf_factory(["scan", "text", "scan"]), CFG, workdir)
    outcome = read_pages(pages, CFG, workdir)
    assert outcome.results == {}
    assert sorted(outcome.failures) == [1, 3]
    assert "setup failed" in outcome.failures[1]


def test_prompt_file_with_invalid_encoding_falls_back(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"\xff\xfe broken")
    assert "faithful OCR transcription" in load_prompt(Config(prompt_file=str(prompt)))


def test_no_work_starts_after_the_document_time_limit(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan", "scan"]), CFG, workdir)
    client = FakeClient({"default": PageResult(text="late", rotation=0)})
    outcome = read_pages(pages, Config(api_key="k", document_timeout=0.5), workdir, client)
    assert client.calls == 0
    assert sorted(outcome.failures) == [1, 2]
    assert "time limit" in outcome.failures[1]


def test_empty_answer_is_not_believed_when_the_page_has_a_text_layer(pdf_factory, workdir):
    """A scan whose scanner found text cannot be an empty page."""
    text = "Invoice 12345 of 12 November, total 99.50 EUR net, payable within 14 days"
    pages = analyse_pdf(pdf_factory(["scan_text", "scan"], text=text), CFG, workdir)
    client = FakeClient({"default": PageResult(text="", rotation=0)})
    outcome = read_pages(pages, Config(api_key="k", concurrency=1), workdir, client)
    assert sorted(outcome.results) == [2]
    assert "no text" in outcome.failures[1]


def test_foreign_exception_text_stays_out_of_the_failures(pdf_factory, workdir):
    pages = analyse_pdf(pdf_factory(["scan"]), CFG, workdir)
    outcome = read_pages(pages, CFG, workdir, FakeClient({"default": RuntimeError("secret")}))
    assert outcome.failures == {1: "RuntimeError"}


def turn_of(data: bytes, mime_type: str) -> int:
    """How far a test page has been turned clockwise: /Rotate of a PDF, the dark corner of a picture."""
    import io

    if mime_type == "application/pdf":
        import pikepdf

        with pikepdf.open(io.BytesIO(data)) as pdf:
            return int(pdf.pages[0].obj.get("/Rotate", 0))
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        grey = image.convert("L")
        width, height = grey.size
        corners = {0: (20, 20), 90: (width - 20, 20), 180: (width - 20, height - 20), 270: (20, height - 20)}
        return min(corners, key=lambda turn: grey.getpixel(corners[turn]))


def one_picture(tmp_path, workdir):
    """A scan as an image file, with a dark corner top left to tell its orientation by."""
    from PIL import Image

    from paperless_gemini_ocr.pages import prepare_image

    image = Image.new("RGB", (600, 800), "white")
    image.paste((0, 0, 0), (0, 0, 120, 120))
    path = tmp_path / "scan.jpg"
    image.save(path, "JPEG", quality=95)
    return prepare_image(path, "image/jpeg", workdir)


class TurningClient:
    """Reports a page as rotated until it has been turned the right way round.

    ``needs`` is the turn (clockwise degrees) that brings the page upright;
    ``says`` maps the remaining turn to what the model reports for it.
    """

    def __init__(
        self,
        needs: int,
        says: dict[int, int] | None = None,
        fail_second: bool = False,
        refuses: tuple[int, ...] = (),
    ):
        self.needs = needs
        self.says = says or {}
        self.fail_second = fail_second
        # Turns at which the API refuses the page outright.
        self.refuses = refuses
        self.seen: list[int] = []

    def transcribe(self, data, mime_type, prompt, deadline=None):
        turned = turn_of(data, mime_type)
        self.seen.append(turned)
        if self.fail_second and len(self.seen) == 2:
            raise GeminiError("HTTP 503", retryable=True, attempts=3)
        if turned in self.refuses:
            raise GeminiError("request blocked (OTHER)", retryable=False, blocked=True)
        left = (self.needs - turned) % 360
        reported = self.says[left] if left in self.says else left
        return PageResult(text=f"read at {turned}", rotation=reported, input_tokens=10, output_tokens=5)


def one_scan(pdf_factory, workdir):
    return analyse_pdf(pdf_factory(["scan"]), CFG, workdir)


def test_upside_down_page_is_read_again_the_right_way_up(pdf_factory, workdir):
    client = TurningClient(needs=180)
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180]
    assert result.text == "read at 180"
    assert (result.rotation, result.rotation_confirmed) == (180, True)
    assert result.attempts == 2
    assert (result.input_tokens, result.output_tokens) == (20, 10)


def test_upright_page_is_read_once(pdf_factory, workdir):
    client = TurningClient(needs=0)
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    assert client.seen == [0]
    assert outcome.results[1].rotation_confirmed is False


def test_sideways_page_turned_the_wrong_way_first_is_turned_on(pdf_factory, workdir):
    """The model says 270 where 90 is needed; the second reading shows it, the third is upright."""
    client = TurningClient(needs=90, says={90: 270})
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 270, 90]
    assert result.text == "read at 90"
    assert (result.rotation, result.rotation_confirmed) == (90, True)


def test_first_reading_stays_when_the_second_fails(pdf_factory, workdir):
    client = TurningClient(needs=180, fail_second=True)
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert outcome.failures == {}
    assert result.text == "read at 0"
    assert (result.rotation, result.rotation_confirmed) == (180, False)
    # The failed view was tried three times; that counts.
    assert client.seen == [0, 180, 90, 270]
    assert result.attempts == 6


def test_first_reading_stays_when_the_page_never_comes_upright(pdf_factory, workdir):
    """A model that always says 180: every other way round is tried once, then it ends."""
    client = TurningClient(needs=180, says={0: 180, 180: 180, 90: 180, 270: 180})
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180, 90, 270]
    assert result.text == "read at 0"
    assert result.rotation_confirmed is False
    assert result.attempts == 4


def test_misleading_reports_still_end_at_the_upright_reading(pdf_factory, workdir):
    """Needs 270; the model says 180 first, then 270, then 180: the turn left over is the one."""
    client = TurningClient(needs=270, says={270: 180, 90: 270, 180: 180})
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180, 90, 270]
    assert result.text == "read at 270"
    assert (result.rotation, result.rotation_confirmed) == (270, True)


def test_second_reading_can_be_switched_off(pdf_factory, workdir):
    client = TurningClient(needs=180)
    cfg = Config(api_key="k", reread_rotated=False)
    outcome = read_pages(one_scan(pdf_factory, workdir), cfg, workdir, client)
    assert client.seen == [0]
    assert outcome.results[1].rotation == 180


def test_page_refused_as_stored_is_read_when_turned_upright(tmp_path, workdir):
    """Stored upside down and refused that way; turned, it is read and reported as upright."""
    client = TurningClient(needs=180, refuses=(0,))
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180]
    assert result.text == "read at 180"
    assert (result.rotation, result.rotation_confirmed) == (180, True)
    assert result.attempts == 2


def test_refused_upright_page_stays_a_failure(tmp_path, workdir):
    """Upright as stored and refused: the turned views are read, but none of them is upright."""
    client = TurningClient(needs=0, refuses=(0,))
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    assert client.seen == [0, 180, 90, 270]
    assert outcome.results == {}
    assert outcome.failures == {1: "request blocked (OTHER) after 4 attempt(s)"}


def test_refused_in_every_view_is_a_failure(tmp_path, workdir):
    client = TurningClient(needs=0, refuses=(0, 90, 180, 270))
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    assert client.seen == [0, 180, 90, 270]
    assert outcome.failures == {1: "request blocked (OTHER) after 4 attempt(s)"}


def test_refused_second_view_does_not_end_the_search(tmp_path, workdir):
    """Needs 270, the model says 180 first and that view is refused: the others are tried."""
    client = TurningClient(needs=270, says={270: 180}, refuses=(180,))
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180, 90, 270]
    assert result.text == "read at 270"
    assert (result.rotation, result.rotation_confirmed) == (270, True)


def test_other_errors_are_not_worked_around_by_turning(pdf_factory, workdir):
    client = TurningClient(needs=0, fail_second=False)
    client.transcribe = lambda *a, **k: (_ for _ in ()).throw(GeminiError("HTTP 503", retryable=True, attempts=3))
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    assert outcome.failures == {1: "HTTP 503 after 3 attempt(s)"}


def test_refused_page_is_searched_on_after_an_answer_without_rotation(tmp_path, workdir):
    """Refused as stored; the first turned view answers without a rotation line; the next is upright."""
    client = TurningClient(needs=90, says={270: None}, refuses=(0,))
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0, 180, 90]
    assert result.text == "read at 90"
    assert (result.rotation, result.rotation_confirmed) == (90, True)


def test_no_second_reading_once_the_document_time_is_up(pdf_factory, workdir):
    import time

    from paperless_gemini_ocr.pages import load_payload
    from paperless_gemini_ocr.pipeline import _read_page

    client = TurningClient(needs=180)
    page = one_scan(pdf_factory, workdir)[0]
    data, mime = load_payload(page, CFG, workdir)
    result = _read_page(client, data, mime, "prompt", time.monotonic())
    assert client.seen == [0]
    assert (result.text, result.rotation, result.rotation_confirmed) == ("read at 0", 180, False)


class PickyClient:
    """Refuses the page as a PDF and reads it as a picture."""

    def __init__(self, picture_rotation: int = 0, refuse_pictures: bool = False):
        self.picture_rotation = picture_rotation
        self.refuse_pictures = refuse_pictures
        self.seen: list[str] = []

    def transcribe(self, data, mime_type, prompt, deadline=None):
        self.seen.append(mime_type)
        if mime_type == "application/pdf" or self.refuse_pictures:
            raise GeminiError("request blocked (OTHER)", retryable=False, blocked=True)
        return PageResult(text="read as a picture", rotation=self.picture_rotation)


def test_pdf_page_refused_by_the_api_is_sent_as_a_picture(pdf_factory, workdir):
    client = PickyClient()
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    # One request with the page as PDF, no turned PDF views, then the picture.
    assert client.seen == ["application/pdf", "image/jpeg"]
    assert result.text == "read as a picture"
    assert result.attempts == 2


def test_page_refused_as_pdf_and_as_picture_is_a_failure(pdf_factory, workdir):
    client = PickyClient(refuse_pictures=True)
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    assert client.seen == ["application/pdf"] + ["image/jpeg"] * 4
    assert outcome.failures == {1: "request blocked (OTHER) after 5 attempt(s)"}


def test_refused_image_file_has_no_other_rendering_to_fall_back_on(tmp_path, workdir):
    from PIL import Image

    from paperless_gemini_ocr.pages import prepare_image

    path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 80), "white").save(path, "JPEG")
    client = PickyClient(refuse_pictures=True)
    outcome = read_pages(prepare_image(path, "image/jpeg", workdir), CFG, workdir, client)
    assert client.seen == ["image/jpeg"] * 4
    assert outcome.failures == {1: "request blocked (OTHER) after 4 attempt(s)"}


def test_refused_pdf_page_without_a_picture_is_tried_turned(pdf_factory, workdir, monkeypatch):
    """No picture can be made of the page: the PDF views turned the other ways round still are."""
    from paperless_gemini_ocr import pipeline as pipeline_module
    from paperless_gemini_ocr.pages import AnalysisError

    real_load = pipeline_module.load_payload

    def no_picture(page, cfg, workdir, deadline=None):
        if cfg.page_input == "image":
            raise AnalysisError("page too large to upload")
        return real_load(page, cfg, workdir, deadline)

    monkeypatch.setattr(pipeline_module, "load_payload", no_picture)
    client = TurningClient(needs=180, refuses=(0,))
    outcome = read_pages(one_scan(pdf_factory, workdir), CFG, workdir, client)
    result = outcome.results[1]
    # The page as stored is asked for once, not again for the turned views.
    assert client.seen == [0, 180]
    assert result.text == "read at 180"
    assert (result.rotation, result.rotation_confirmed, result.attempts) == (180, True, 2)


def test_turn_that_cannot_be_made_is_not_counted_as_a_request(tmp_path, workdir, monkeypatch):
    from paperless_gemini_ocr import pipeline as pipeline_module

    def cannot_turn(data, mime_type, degrees):
        raise OSError("no")

    monkeypatch.setattr(pipeline_module, "turn_payload", cannot_turn)
    client = TurningClient(needs=180)
    outcome = read_pages(one_picture(tmp_path, workdir), CFG, workdir, client)
    result = outcome.results[1]
    assert client.seen == [0]
    assert (result.text, result.attempts) == ("read at 0", 1)
