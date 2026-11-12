"""Stand-in for the Gemini API, steerable over HTTP.

GET /control?mode=ok|fail|flaky|empty&delay=SECONDS&rotation=DEGREES&fail_first=N
    sets the behaviour and resets the counters.
GET /stats returns what was requested since.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs
from urllib.parse import urlparse

LOCK = threading.Lock()
STATE = {"mode": "ok", "delay": 0.0, "rotation": 0, "fail_first": 0}
STATS = {"requests": 0, "mimes": [], "max_parallel": 0, "active": 0, "key_seen": False}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/control":
            with LOCK:
                STATE.update(
                    mode=query.get("mode", ["ok"])[0],
                    delay=float(query.get("delay", ["0"])[0]),
                    rotation=int(query.get("rotation", ["0"])[0]),
                    fail_first=int(query.get("fail_first", ["0"])[0]),
                )
                STATS.update(requests=0, mimes=[], max_parallel=0, key_seen=False)
                self._json(200, dict(STATE))
            return
        if url.path == "/stats":
            with LOCK:
                self._json(200, dict(STATS))
            return
        self._json(404, {})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        part = body["contents"][0]["parts"][0]["inline_data"]
        with LOCK:
            STATS["requests"] += 1
            number = STATS["requests"]
            STATS["mimes"].append(part["mime_type"])
            STATS["active"] += 1
            STATS["max_parallel"] = max(STATS["max_parallel"], STATS["active"])
            STATS["key_seen"] = STATS["key_seen"] or bool(self.headers.get("x-goog-api-key"))
            state = dict(STATE)
        try:
            time.sleep(state["delay"])
            if state["mode"] == "fail" or (
                state["mode"] == "flaky" and number <= state["fail_first"]
            ):
                self._json(503, {"error": {"status": "UNAVAILABLE"}})
            elif state["mode"] == "empty":
                self._json(200, {"candidates": []})
            else:
                text = (
                    f"ROTATION: {state['rotation']}\n"
                    f"FAKE-OCR request {number} ({part['mime_type']})"
                )
                self._json(
                    200,
                    {
                        "candidates": [
                            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"},
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1100,
                            "candidatesTokenCount": 20,
                        },
                    },
                )
        finally:
            with LOCK:
                STATS["active"] -= 1


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
