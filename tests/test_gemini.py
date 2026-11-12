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


class Body(httpx.SyncByteStream):
    """An answer that still has to be read, as it comes from a real connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data


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
        status, payload, *rest = item
        return httpx.Response(
            status,
            headers=rest[0] if rest else None,
            stream=Body(json.dumps(payload).encode()),
        )

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
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, stream=Body(b"{}"))),
    )
    client = GeminiClient(CFG, http, sleep=lambda s: None, clock=lambda: now[0])
    with pytest.raises(GeminiError, match="time limit"):
        client.transcribe(b"data", "image/jpeg", "prompt", deadline=100.5)


def test_error_text_never_contains_the_key():
    client, _, _ = client_for([(403, {"error": {"status": "PERMISSION_DENIED", "message": "key secret-key bad"}})])
    with pytest.raises(GeminiError) as caught:
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert "secret-key" not in str(caught.value)


@pytest.mark.parametrize("raw", ["ROTATION: 45", "\x00\x00", "ROTATION: 45\n\x00"])
def test_empty_text_needs_a_valid_protocol_line(raw):
    """Otherwise a garbled answer would pass as a blank page and wipe the content."""
    client, requests, _ = client_for([(200, answer(raw))] * 3)
    with pytest.raises(GeminiError, match="without text"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 3


@pytest.mark.parametrize("body", [[{"error": "x"}], "Service Unavailable", None, {"error": "rate limit"}])
def test_error_body_of_another_shape_is_still_retried(body):
    client, requests, _ = client_for([(503, body), (200, answer("ROTATION: 0\nok"))])
    assert client.transcribe(b"data", "image/jpeg", "prompt").text == "ok"
    assert len(requests) == 2


@pytest.mark.parametrize("body", [[1, 2], "text", {"candidates": "none"}, {"candidates": [5]}])
def test_answer_of_another_shape_counts_as_temporary_failure(body):
    client, requests, _ = client_for([(200, body)] * 3)
    with pytest.raises(GeminiError, match="unexpected structure|without candidates"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 3


def test_status_words_from_the_endpoint_are_not_passed_on_verbatim():
    """Whatever a broken endpoint echoes there, the key included, stays out."""
    echoes = [
        (403, {"error": {"status": "key secret-key"}}),
        (200, {"promptFeedback": {"blockReason": "secret-key"}}),
        (200, {"candidates": [{"finishReason": "secret-key", "content": {}}]}),
    ]
    for item in echoes:
        client, _, _ = client_for([item] * 3)
        with pytest.raises(GeminiError) as caught:
            client.transcribe(b"data", "image/jpeg", "prompt")
        assert "secret-key" not in str(caught.value)
        assert "UNKNOWN" in str(caught.value)


def test_slowly_dripping_answer_runs_into_the_request_time():
    """httpx only limits each read; the total is watched here."""
    now = [0.0]

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                now[0] += 10.0
                yield b" "

    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Drip())))
    client = GeminiClient(
        Config(api_key="k", base_url="https://gemini.test/v1beta", max_attempts=1, request_timeout=30),
        http,
        sleep=lambda s: None,
        clock=lambda: now[0],
    )
    with pytest.raises(GeminiError, match="no answer within 30 s"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert now[0] <= 50.0


def test_oversized_answer_is_refused():
    class Flood(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(40):
                yield b"x" * (1024 * 1024)

    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Flood())))
    client = GeminiClient(CFG, http, sleep=lambda s: None)
    with pytest.raises(GeminiError, match="too large"):
        client.transcribe(b"data", "image/jpeg", "prompt")


def test_compressed_answers_are_not_asked_for():
    client, requests, _ = client_for([(200, answer("ROTATION: 0\nok"))])
    client.transcribe(b"data", "image/jpeg", "prompt")
    assert requests[0].headers["accept-encoding"] == "identity"


def compressed_client(content: bytes, encoding: str = "gzip", **kwargs):
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                stream=Body(content),
                headers={"content-encoding": encoding},
            ),
        ),
    )
    return GeminiClient(CFG, http, sleep=lambda s: None, **kwargs)


def test_compressed_answer_is_read_all_the_same():
    import gzip

    body = json.dumps(answer("ROTATION: 0\nkomprimiert")).encode()
    assert compressed_client(gzip.compress(body)).transcribe(b"d", "image/jpeg", "p").text == "komprimiert"


def test_compressed_answer_cannot_unpack_past_the_size_limit():
    """40 kB that unpack to 40 MB: stopped at the limit, not after it."""
    import gzip

    with pytest.raises(GeminiError, match="too large"):
        compressed_client(gzip.compress(b" " * (40 * 1024 * 1024))).transcribe(b"d", "image/jpeg", "p")


def test_unknown_answer_encoding_is_refused():
    with pytest.raises(GeminiError, match="unexpected encoding"):
        compressed_client(b"x", encoding="gzip, gzip").transcribe(b"d", "image/jpeg", "p")


def test_request_time_counts_even_when_nothing_can_be_unpacked_yet():
    """A gzip header that never ends yields no data; the clock is still watched."""
    now = [0.0]

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            yield bytes([0x1F, 0x8B, 8, 0x08, 0, 0, 0, 0, 0, 3])  # header with a file name to follow
            for _ in range(1000):
                now[0] += 10.0
                yield b"a"

    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, stream=Drip(), headers={"content-encoding": "gzip"}),
        ),
    )
    client = GeminiClient(
        Config(api_key="k", base_url="https://gemini.test/v1beta", max_attempts=1, request_timeout=30),
        http,
        sleep=lambda s: None,
        clock=lambda: now[0],
    )
    with pytest.raises(GeminiError, match="no answer within 30 s"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert now[0] <= 50.0


RATE_LIMITED = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"},
        ],
    },
}


def test_rate_limit_waits_as_long_as_the_api_asks():
    """Coming back after two seconds would only use up the attempts."""
    client, requests, sleeps = client_for([(429, RATE_LIMITED), (200, answer("ROTATION: 0\nok"))])
    result = client.transcribe(b"data", "image/jpeg", "prompt")
    assert result.text == "ok"
    assert len(requests) == 2
    assert 37 <= sleeps[0] <= 40


def test_retry_after_header_counts_too():
    client, _, sleeps = client_for(
        [(503, {}, {"retry-after": "30"}), (200, answer("ROTATION: 0\nok"))],
    )
    assert client.transcribe(b"data", "image/jpeg", "prompt").text == "ok"
    assert 30 <= sleeps[0] <= 33


@pytest.mark.parametrize(
    "details",
    [None, "37s", [None, 5, {"retryDelay": 37}, {"retryDelay": "soon"}, {"retryDelay": "-5s"}]],
)
def test_odd_retry_hints_are_ignored(details):
    limited = {"error": {"status": "RESOURCE_EXHAUSTED", "details": details}}
    client, _, sleeps = client_for([(429, limited), (200, answer("ROTATION: 0\nok"))])
    assert client.transcribe(b"data", "image/jpeg", "prompt").text == "ok"
    assert sleeps[0] <= 3


def test_no_waiting_for_a_rate_limit_that_outlasts_the_document():
    now = [100.0]
    client, requests, sleeps = client_for([(429, RATE_LIMITED), (200, answer("ROTATION: 0\nok"))])
    client._clock = lambda: now[0]
    with pytest.raises(GeminiError, match="HTTP 429 RESOURCE_EXHAUSTED") as caught:
        client.transcribe(b"data", "image/jpeg", "prompt", deadline=120.0)
    assert caught.value.attempts == 1
    assert len(requests) == 1
    assert sleeps == []


def test_no_waiting_for_a_limit_that_lasts_hours():
    daily = {"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "40000s"}]}}
    client, requests, sleeps = client_for([(429, daily), (200, answer("ROTATION: 0\nok"))])
    with pytest.raises(GeminiError, match="HTTP 429"):
        client.transcribe(b"data", "image/jpeg", "prompt")
    assert len(requests) == 1
    assert sleeps == []


def test_retry_after_as_a_point_in_time():
    from datetime import datetime
    from datetime import timedelta
    from datetime import timezone
    from email.utils import format_datetime

    moment = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    client, _, sleeps = client_for(
        [(503, {}, {"retry-after": moment}), (200, answer("ROTATION: 0\nok"))],
    )
    assert client.transcribe(b"data", "image/jpeg", "prompt").text == "ok"
    assert 27 <= sleeps[0] <= 33


def test_unreadable_retry_after_is_ignored():
    client, _, sleeps = client_for(
        [(503, {}, {"retry-after": "next week"}), (200, answer("ROTATION: 0\nok"))],
    )
    assert client.transcribe(b"data", "image/jpeg", "prompt").text == "ok"
    assert sleeps[0] <= 3
