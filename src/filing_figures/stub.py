"""A stand-in model server (OpenAI chat API) for testing the harness without a GPU.

It recognises each filing by its text and answers in one of three modes:
  gold    the answer key itself: must score 100%
  prior   the prior year's values from the same 10-K: must score as "wrong_period"
  garbage text that is not JSON: must score as "invalid_json"
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from filing_figures.labels import FIELD_NAMES, FIELDS, MAX_YEAR_DAYS, MIN_YEAR_DAYS

MODES = ("gold", "prior", "garbage")
KINDS = {f.name: f.kind for f in FIELDS}


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _part(value, scale, period_end, fiscal_year) -> dict:
    return {"value": value, "scale": scale, "period_end": period_end, "fiscal_year": fiscal_year}


def _prior(rec: dict, name: str) -> dict | None:
    """The same tag's value for the latest earlier period of the same kind, if the 10-K has one."""
    gold = rec["gold"][name]
    kind = KINDS[name]
    if kind == "cover":
        return None  # a 10-K files only this year's cover share count
    best = None
    for f in rec["context_facts"]:
        if f["tag"] != gold["tag"] or f["end"] >= gold["period_end"]:
            continue
        if kind == "instant" and f["start"] is not None:
            continue
        if kind == "year":
            if not f["start"]:
                continue
            days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days + 1
            if not MIN_YEAR_DAYS <= days <= MAX_YEAR_DAYS:
                continue
        if best is None or f["end"] > best["end"]:
            best = f
    return best


def answer(rec: dict, mode: str) -> str:
    if mode == "garbage":
        return "Revenue was strong this year."
    out = {}
    for name in FIELD_NAMES:
        g = rec["gold"][name]
        if g is None:
            out[name] = _part(None, None, None, None)
        elif mode == "gold":
            out[name] = _part(g["value"], g["scale"], g["period_end"], g["fiscal_year"])
        else:
            p = _prior(rec, name)
            out[name] = (
                _part(None, None, None, None)
                if p is None
                else _part(p["val"], g["scale"], p["end"], g["fiscal_year"] - 1)
            )
    return json.dumps(out)


class _Handler(BaseHTTPRequestHandler):
    server: StubServer

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        text = body["messages"][-1]["content"]
        rec = self.server.records.get(_key(text))
        content = answer(rec, self.server.mode) if rec else "unknown filing"
        resp = {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": len(content) // 4, "total_tokens": 0},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # quiet
        pass


class StubServer(ThreadingHTTPServer):
    def __init__(self, records: list[dict], mode: str = "gold", host: str = "127.0.0.1", port: int = 8001):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.records = {_key(r["input"]): r for r in records}
        self.mode = mode
        super().__init__((host, port), _Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"
