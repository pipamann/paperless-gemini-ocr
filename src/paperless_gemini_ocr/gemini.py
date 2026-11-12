"""Small Gemini client: one page in, its text and rotation out.

Talks to the ``generateContent`` REST endpoint with httpx, which Paperless-ngx
already ships, so the plugin needs no SDK. Every failure mode a page can hit
(timeout, rate limit, empty answer, cut-off answer) ends in ``GeminiError``;
callers decide what to do with a page that could not be read.
"""

from __future__ import annotations

import base64
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import Config

logger = logging.getLogger("paperless.parsing.gemini_ocr")

# Appended to every prompt, also to custom ones, because the plugin parses it.
# Asking for the line on every page also makes a blank page distinguishable
# from an empty (failed) answer.
PROTOCOL = (
    "\n\nOutput format: the very first line of your answer must be exactly "
    "`ROTATION: N`, where N is 0, 90, 180 or 270: the number of degrees the "
    "page as shown must be turned clockwise so that its text is upright "
    "(0 if it already is). The transcription starts on the next line. If the "
    "page contains no text at all, output only the ROTATION line."
)

_ROTATION_LINE = re.compile(r"^\s*`?\s*ROTATION:\s*(\d{1,3})\s*°?\s*`?\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.DOTALL)
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})


class GeminiError(Exception):
    """A page could not be read. ``retryable`` says whether trying again can help."""

    def __init__(self, message: str, *, retryable: bool, attempts: int = 1) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts


@dataclass
class PageResult:
    text: str
    rotation: int | None
    attempts: int = 1
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


def build_request(cfg: Config, data: bytes, mime_type: str, prompt: str) -> dict:
    generation: dict = {"maxOutputTokens": cfg.max_output_tokens}
    if cfg.thinking in _THINKING_LEVELS:
        generation["thinkingConfig"] = {"thinkingLevel": cfg.thinking}
    elif cfg.thinking.lstrip("-").isdigit():
        generation["thinkingConfig"] = {"thinkingBudget": int(cfg.thinking)}
    if cfg.media_resolution != "default":
        generation["mediaResolution"] = f"MEDIA_RESOLUTION_{cfg.media_resolution.upper()}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(data).decode("ascii"),
                        },
                    },
                    {"text": prompt + PROTOCOL},
                ],
            },
        ],
        "generationConfig": generation,
    }


def split_rotation(text: str) -> tuple[int | None, str]:
    """Separate the leading ``ROTATION: N`` line from the transcription."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    first, _, rest = stripped.partition("\n")
    match = _ROTATION_LINE.match(first)
    if not match:
        return None, stripped
    degrees = int(match.group(1))
    rest = rest.strip()
    fenced = _FENCE.match(rest)
    if fenced:
        rest = fenced.group(1).strip()
    return (degrees if degrees in (0, 90, 180, 270) else None), rest


def parse_response(payload: dict) -> PageResult:
    feedback = payload.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise GeminiError(
            f"request blocked ({feedback['blockReason']})",
            retryable=False,
        )
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiError("answer without candidates", retryable=True)
    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish not in (None, "STOP"):
        # MAX_TOKENS on a single page means the model looped or thought the
        # budget away; SAFETY and RECITATION are not deterministic either.
        raise GeminiError(f"answer ended with {finish}", retryable=True)
    parts = (candidate.get("content") or {}).get("parts") or []
    raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not raw.strip():
        raise GeminiError("empty answer", retryable=True)
    rotation, text = split_rotation(raw)
    usage = payload.get("usageMetadata") or {}
    return PageResult(
        text=text.replace("\x00", ""),
        rotation=rotation,
        input_tokens=int(usage.get("promptTokenCount") or 0),
        output_tokens=int(usage.get("candidatesTokenCount") or 0),
        thinking_tokens=int(usage.get("thoughtsTokenCount") or 0),
    )


def _http_error(response: httpx.Response) -> GeminiError:
    status = ""
    try:
        status = str((response.json().get("error") or {}).get("status") or "")
    except ValueError:
        pass
    return GeminiError(
        f"HTTP {response.status_code} {status}".strip(),
        retryable=response.status_code in _RETRY_STATUS,
    )


class GeminiClient:
    """Thread-safe: one instance serves all pages of a document."""

    def __init__(
        self,
        cfg: Config,
        http: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._owns_http = http is None
        self._http = http or httpx.Client(
            limits=httpx.Limits(
                max_connections=cfg.concurrency,
                max_keepalive_connections=cfg.concurrency,
            ),
        )
        self._sleep = sleep
        self._clock = clock

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def transcribe(
        self,
        data: bytes,
        mime_type: str,
        prompt: str,
        deadline: float | None = None,
    ) -> PageResult:
        """Read one page. ``deadline`` is a ``clock()`` value for the whole document."""
        cfg = self.cfg
        url = f"{cfg.base_url}/models/{cfg.model}:generateContent"
        headers = {"x-goog-api-key": cfg.api_key or ""}
        body = build_request(cfg, data, mime_type, prompt)
        started = self._clock()
        error = GeminiError("not attempted", retryable=False)

        for attempt in range(1, cfg.max_attempts + 1):
            timeout = cfg.request_timeout
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 1:
                    raise GeminiError(
                        f"document time limit reached ({error})",
                        retryable=False,
                        attempts=attempt - 1,
                    )
                timeout = min(timeout, remaining)
            try:
                response = self._http.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
                if response.status_code != 200:
                    raise _http_error(response)
                result = parse_response(response.json())
                result.attempts = attempt
                result.seconds = self._clock() - started
                return result
            except GeminiError as e:
                error = e
            except httpx.TimeoutException:
                error = GeminiError(f"no answer within {timeout:.0f} s", retryable=True)
            except httpx.TransportError as e:
                error = GeminiError(
                    f"connection failed ({type(e).__name__})",
                    retryable=True,
                )
            except ValueError:
                error = GeminiError("answer is not valid JSON", retryable=True)

            error.attempts = attempt
            if not error.retryable or attempt == cfg.max_attempts:
                break
            delay = min(2.0**attempt + random.uniform(0, 1), 20.0)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - self._clock() - 1))
            self._sleep(delay)

        raise error
