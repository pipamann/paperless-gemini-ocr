"""Small Gemini client: one page in, its text and rotation out.

Talks to the ``generateContent`` REST endpoint with httpx, which Paperless-ngx
already ships, so the plugin needs no SDK. Every failure mode a page can hit
(timeout, rate limit, empty answer, cut-off answer) ends in ``GeminiError``;
callers decide what to do with a page that could not be read.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from email.utils import parsedate_to_datetime
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

# Added to the request after the API has cut an answer off as RECITATION. It
# does that when the output repeats text it knows word for word (insurance
# terms, court forms, statutes), and it does so every time for the same page.
# With numbered lines the output is no longer that text, and the page is read.
NUMBERED_LINES = (
    "\n\nFor this page, start every line of the transcription with its running "
    "number and a vertical bar, like `1| first line`, `2| second line`. The "
    "ROTATION line stays as it is, without a number."
)
_LINE_NUMBER = re.compile(r"^\s*(\d{1,5})\s?\|\s?")

_ROTATION_LINE = re.compile(r"^\s*`?\s*ROTATION:\s*(\d{1,3})\s*°?\s*`?\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.DOTALL)
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
# Status words of the API end up in logs and document notes. Only plain ones
# are passed on, so nothing a broken endpoint echoes can travel that way.
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")
# One page of text is a few kilobytes; anything near this is not an answer.
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
# "37s" or "37.5s": how the API says when a rate limit lets requests in again.
_RETRY_DELAY = re.compile(r"^([0-9]{1,6}(?:\.[0-9]{1,9})?)s$")
_RETRY_HEADER = re.compile(r"^[0-9]{1,6}$")
# A longer wait than this is not a per-minute limit; the page is given up.
_MAX_RETRY_AFTER = 120.0


def _code(value: object) -> str:
    return value if isinstance(value, str) and _CODE.match(value) else "UNKNOWN"


class GeminiError(Exception):
    """A page could not be read. ``retryable`` says whether trying again can help."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        attempts: int = 1,
        retry_after: float | None = None,
        recitation: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts
        # Seconds the endpoint asked to wait before the next request.
        self.retry_after = retry_after
        # The answer was cut off for repeating known text.
        self.recitation = recitation


@dataclass
class PageResult:
    text: str
    rotation: int | None
    attempts: int = 1
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


def build_request(
    cfg: Config,
    data: bytes,
    mime_type: str,
    prompt: str,
    *,
    numbered: bool = False,
) -> dict:
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
                    {"text": prompt + PROTOCOL + (NUMBERED_LINES if numbered else "")},
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


def remove_line_numbers(raw: str) -> str:
    """Undo the numbering asked for with ``NUMBERED_LINES``, on the raw answer.

    Numbers are only taken off when they are plainly the ones asked for:
    every line carries one and they run 1, 2, 3 without a gap (the ROTATION
    line may be numbered along or not). An answer without a line numbered 1
    ignored the request and is left as it is, so table rows like
    ``12| 100 EUR`` keep their start. Anything in between, numbering begun
    and not kept, cannot be told apart from content and counts as a failed
    answer.
    """
    lines = raw.split("\n")
    filled = [i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("```")]
    if filled and _ROTATION_LINE.match(lines[filled[0]]):
        filled = filled[1:]
    found = [_LINE_NUMBER.match(lines[i]) for i in filled]
    numbers = [int(m.group(1)) if m else None for m in found]
    if 1 not in numbers:
        return raw
    if numbers != list(range(1, len(numbers) + 1)):
        raise GeminiError("answer did not keep to the numbered lines", retryable=True)
    for i, match in zip(filled, found, strict=True):
        lines[i] = lines[i][match.end() :]
    return "\n".join(lines)


def parse_response(payload: object, *, numbered: bool = False) -> PageResult:
    """Text and rotation of an answer; ``numbered`` when numbered lines were asked for."""
    try:
        return _parse_response(payload, numbered)
    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as e:
        # Valid JSON of another shape, e.g. from a gateway in front of the API.
        raise GeminiError("answer has an unexpected structure", retryable=True) from e


def _parse_response(payload: dict, numbered: bool = False) -> PageResult:
    feedback = payload.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise GeminiError(
            f"request blocked ({_code(feedback['blockReason'])})",
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
        raise GeminiError(
            f"answer ended with {_code(finish)}",
            retryable=True,
            recitation=finish == "RECITATION",
        )
    parts = (candidate.get("content") or {}).get("parts") or []
    raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not raw.strip():
        raise GeminiError("empty answer", retryable=True)
    raw = raw.replace("\x00", "")
    if numbered:
        raw = remove_line_numbers(raw)
    rotation, text = split_rotation(raw)
    if not text and rotation is None:
        # An empty page is only believed when the answer follows the protocol.
        raise GeminiError("answer without text and without a valid rotation line", retryable=True)
    usage = payload.get("usageMetadata") or {}
    return PageResult(
        text=text,
        rotation=rotation,
        input_tokens=int(usage.get("promptTokenCount") or 0),
        output_tokens=int(usage.get("candidatesTokenCount") or 0),
        thinking_tokens=int(usage.get("thoughtsTokenCount") or 0),
    )


def _retry_after(header: str | None, error: object) -> float | None:
    """Seconds to wait, from the Retry-After header or the API's RetryInfo."""
    waits: list[float] = []
    if header and _RETRY_HEADER.match(header.strip()):
        waits.append(float(header.strip()))
    elif header:
        # The other allowed form: a point in time.
        try:
            moment = parsedate_to_datetime(header.strip())
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            waits.append(max(0.0, (moment - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            pass
    details = error.get("details") if isinstance(error, dict) else None
    if isinstance(details, list):
        for detail in details:
            delay = detail.get("retryDelay") if isinstance(detail, dict) else None
            match = _RETRY_DELAY.match(delay.strip()) if isinstance(delay, str) else None
            if match:
                waits.append(float(match.group(1)))
    return max(waits) if waits else None


def _http_error(status_code: int, raw: bytes, retry_header: str | None = None) -> GeminiError:
    status = ""
    error = None
    try:
        body = json.loads(raw)
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("status") is not None:
            status = _code(error["status"])
    except ValueError:
        pass
    return GeminiError(
        f"HTTP {status_code} {status}".strip(),
        retryable=status_code in _RETRY_STATUS,
        retry_after=_retry_after(retry_header, error),
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

    def _post(
        self,
        url: str,
        headers: dict,
        body: dict,
        timeout: float,
    ) -> tuple[int, bytes, str | None]:
        """Send one request; the whole exchange gets ``timeout`` seconds.

        Returns status, body and the Retry-After header.

        httpx applies its timeout to every single read, so an endpoint that
        keeps sending a few bytes would never run into it. The answer is
        therefore read in pieces, with a look at the clock in between.
        """
        give_up_at = self._clock() + timeout
        chunks: list[bytes] = []
        size = 0
        with self._http.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=timeout,
        ) as response:
            # The bytes are taken as they come off the wire. httpx would
            # unpack a compressed answer without any limit and without
            # handing anything over while it does, so that is done here.
            encoding = response.headers.get("content-encoding", "").strip().lower()
            if encoding in ("", "identity"):
                inflater = None
            elif encoding in ("gzip", "x-gzip", "deflate"):
                inflater = zlib.decompressobj(47)
            else:
                raise GeminiError("answer in an unexpected encoding", retryable=False)
            for chunk in response.iter_raw():
                if self._clock() > give_up_at:
                    raise httpx.ReadTimeout("time for the request used up")
                while chunk:
                    if inflater is None:
                        piece, chunk = chunk, b""
                    else:
                        try:
                            piece = inflater.decompress(chunk, _MAX_RESPONSE_BYTES - size + 1)
                        except zlib.error as e:
                            raise GeminiError("answer cannot be unpacked", retryable=True) from e
                        chunk = inflater.unconsumed_tail
                    size += len(piece)
                    if size > _MAX_RESPONSE_BYTES:
                        raise GeminiError("answer too large", retryable=False)
                    chunks.append(piece)
            return response.status_code, b"".join(chunks), response.headers.get("retry-after")

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
        headers = {"x-goog-api-key": cfg.api_key or "", "accept-encoding": "identity"}
        body = build_request(cfg, data, mime_type, prompt)
        numbered = False
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
                status_code, raw, retry_header = self._post(url, headers, body, timeout)
                if status_code != 200:
                    raise _http_error(status_code, raw, retry_header)
                result = parse_response(json.loads(raw), numbered=numbered)
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
            if error.recitation and not numbered:
                # Asking the same again gets the same refusal; ask for
                # numbered lines instead, at once.
                numbered = True
                body = build_request(cfg, data, mime_type, prompt, numbered=True)
                continue
            delay = min(2.0**attempt + random.uniform(0, 1), 20.0)
            left = None if deadline is None else max(0.0, deadline - self._clock() - 1)
            if error.retry_after is not None:
                # A rate limit that says when it lets requests in again.
                # Asking earlier only uses up the attempts, and the pages of a
                # document would all come back at the same moment.
                wait = error.retry_after + random.uniform(0, 3)
                if error.retry_after > _MAX_RETRY_AFTER or (left is not None and wait > left):
                    break
                delay = max(delay, wait)
            if left is not None:
                delay = min(delay, left)
            self._sleep(delay)

        raise error
