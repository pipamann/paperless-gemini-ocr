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
