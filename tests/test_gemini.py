from __future__ import annotations

import json

import httpx
import pytest

from paperless_gemini_ocr.config import Config
from paperless_gemini_ocr.gemini import PROTOCOL
from paperless_gemini_ocr.gemini import GeminiClient
from paperless_gemini_ocr.gemini import GeminiError
from paperless_gemini_ocr.gemini import build_request
from paperless_gemini_ocr.gemini import split_rotation

CFG = Config(api_key="secret-key", base_url="https://gemini.test/v1beta", max_attempts=3)


def answer(text: str, finish: str = "STOP", thought: str | None = None) -> dict:
    parts = []
    if thought:
        parts.append({"text": thought, "thought": True})
    parts.append({"text": text})
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 1100,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 7,
        },
    }


def client_for(responses: list, cfg: Config = CFG):
    """Client whose transport plays ``responses`` in order and records requests."""
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        return httpx.Response(status, json=payload)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiClient(cfg, http, sleep=sleeps.append), requests, sleeps


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ROTATION: 0\nHello\nWorld", (0, "Hello\nWorld")),
        ("ROTATION: 180\n\nText", (180, "Text")),
        ("`ROTATION: 90`\nText", (90, "Text")),
        ("rotation: 270°\nText", (270, "Text")),
        ("ROTATION: 0", (0, "")),
        ("ROTATION: 45\nText", (None, "Text")),
        ("Just text\nROTATION: 90", (None, "Just text\nROTATION: 90")),
        ("```\nROTATION: 0\nText\n```", (0, "Text")),
        ("ROTATION: 0\n```text\nText\n```", (0, "Text")),
    ],
)
def test_split_rotation(raw, expected):
    assert split_rotation(raw) == expected


def test_request_carries_page_prompt_and_settings():
    body = build_request(CFG, b"%PDF", "application/pdf", "PROMPT")
    part, text = body["contents"][0]["parts"]
    assert part["inline_data"] == {"mime_type": "application/pdf", "data": "JVBERg=="}
    assert text["text"] == "PROMPT" + PROTOCOL
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert body["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"


def test_thinking_budget_and_defaults():
    budget = build_request(Config(thinking="0"), b"x", "image/jpeg", "p")
    assert budget["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    plain = build_request(
        Config(thinking="default", media_resolution="default"),
        b"x",
        "image/jpeg",
        "p",
    )
    assert "thinkingConfig" not in plain["generationConfig"]
    assert "mediaResolution" not in plain["generationConfig"]


def test_successful_page():
    client, requests, sleeps = client_for([(200, answer("ROTATION: 180\nHallo", thought="hm"))])
    result = client.transcribe(b"data", "image/jpeg", "prompt")
    assert (result.text, result.rotation, result.attempts) == ("Hallo", 180, 1)
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == (1100, 40, 7)
    request = requests[0]
    assert str(request.url) == "https://gemini.test/v1beta/models/gemini-3.8-flash:generateContent"
    assert request.headers["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert json.loads(request.content)["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"
    assert sleeps == []


def test_blank_page_is_not_a_failure():
    client, _, _ = client_for([(200, answer("ROTATION: 0"))])
    result = client.transcribe(b"data", "image/jpeg", "prompt")
    assert result.text == ""
    assert result.rotation == 0


@pytest.mark.parametrize("status", [429, 500, 503])
def test_temporary_errors_are_retried(status):
    client, requests, sleeps = client_for(
        [(status, {"error": {"status": "UNAVAILABLE"}}), (200, answer("ROTATION: 0\nok"))],
    )
    result = client.transcribe(b"data", "image/jpeg", "prompt")
    assert result.text == "ok"
    assert result.attempts == 2
    assert len(requests) == 2
    assert len(sleeps) == 1


def test_empty_answer_is_retried_then_reported():
    empty = {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
    client, requests, _ = client_for([(200, empty), (200, {}), (200, empty)])
    with pytest.raises(GeminiError) as caught:
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 3
    assert caught.value.attempts == 3


def test_cut_off_answer_counts_as_failure():
    client, requests, _ = client_for(
        [(200, answer("ROTATION: 0\nloop loop", finish="MAX_TOKENS"))] * 3,
    )
    with pytest.raises(GeminiError, match="MAX_TOKENS"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 3


def test_permanent_error_is_not_retried():
    client, requests, sleeps = client_for([(400, {"error": {"status": "INVALID_ARGUMENT"}})])
    with pytest.raises(GeminiError, match="HTTP 400 INVALID_ARGUMENT") as caught:
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert not caught.value.retryable
    assert len(requests) == 1
    assert sleeps == []


def test_blocked_request_is_not_retried():
    client, requests, _ = client_for([(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(GeminiError, match="blocked"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 1


def test_timeout_and_connection_errors_are_retried():
    client, requests, _ = client_for(
        [
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("down"),
            (200, answer("ROTATION: 0\nfinally")),
        ],
    )
    result = client.transcribe(b"data", "image/jpeg", "prompt")
    assert result.text == "finally"
    assert result.attempts == 3
    assert len(requests) == 3


def test_document_deadline_stops_further_requests():
    now = [100.0]
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, json={})))
    client = GeminiClient(CFG, http, sleep=lambda s: None, clock=lambda: now[0])
    with pytest.raises(GeminiError, match="time limit"):
        client.transcribe(b"data", "image/jpeg", "prompt", deadline=100.5)


def test_error_text_never_contains_the_key():
    client, _, _ = client_for([(403, {"error": {"status": "PERMISSION_DENIED", "message": "key secret-key bad"}})])
    with pytest.raises(GeminiError) as caught:
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert "secret-key" not in str(caught.value)
