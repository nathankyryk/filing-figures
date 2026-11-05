"""A stand-in model server (OpenAI chat API) for testing the harness without a GPU.

It recognises each filing by its text and answers in one of three modes, for the record's own
task (v1: six fields, v2: twelve):
  gold    the answer key itself: must score 100%
  prior   a plausible mistake per field: the prior year's value from the same 10-K ("wrong_period");
          for last year's revenue, this year's ("wrong_period"); for free cash flow, operating cash
          flow with capex not subtracted ("wrong_line_item"). Cover-page shares are left blank.
  garbage text that is not JSON: must score as "invalid_json"
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from filing_figures.labels import FIELD_BY_NAME, MAX_YEAR_DAYS, MIN_YEAR_DAYS, field_names

MODES = ("gold", "prior", "garbage")
KINDS = {name: f.kind for name, f in FIELD_BY_NAME.items()}


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _part(value, scale, period_end, fiscal_year) -> dict:
    return {"value": value, "scale": scale, "period_end": period_end, "fiscal_year": fiscal_year}


def _is_year(f: dict) -> bool:
    if not f["start"]:
        return False
    days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days + 1
    return MIN_YEAR_DAYS <= days <= MAX_YEAR_DAYS


def _prior(rec: dict, name: str) -> dict | None:
    """The near miss `prior` mode gives for one field (a fact-shaped dict: val, end), or None."""
    gold = rec["gold"][name]
    kind = KINDS[name]
    if kind == "cover":
        return None  # a 10-K files only this year's cover share count
    if kind == "derived":  # free cash flow: the first part alone, the second not subtracted
        first = rec["gold"].get(FIELD_BY_NAME[name].parts[0])
        return first and {"val": first["value"], "end": first["period_end"]}
    if kind == "prior_year":  # last year's revenue: this year's figure instead
        return next(
            (
                f
                for f in rec["context_facts"]
                if f["tag"] == gold["tag"] and f["end"] == rec["period_end"] and _is_year(f)
            ),
            None,
        )
    best = None
    for f in rec["context_facts"]:
        if f["tag"] != gold["tag"] or f["end"] >= gold["period_end"]:
            continue
        if kind == "instant" and f["start"] is not None:
            continue
        if kind == "year" and not _is_year(f):
            continue
        if best is None or f["end"] > best["end"]:
            best = f
    return best


def answer(rec: dict, mode: str) -> str:
    if mode == "garbage":
        return "Revenue was strong this year."
    out = {}
    for name in field_names(rec.get("task", "v1")):
        g = rec["gold"][name]
        if g is None:
            out[name] = _part(None, None, None, None)
        elif mode == "gold":
            out[name] = _part(g["value"], g["scale"], g["period_end"], g["fiscal_year"])
        else:
            p = _prior(rec, name)
            # The year written for an earlier period's value is the earlier year (v1's behavior)
            fiscal_year = g["fiscal_year"] - 1 if KINDS[name] in ("year", "instant") else g["fiscal_year"]
            out[name] = (
                _part(None, None, None, None) if p is None else _part(p["val"], g["scale"], p["end"], fiscal_year)
            )
    return json.dumps(out)


class _Handler(BaseHTTPRequestHandler):
    server: StubServer

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        text = body["messages"][-1]["content"]
        rec = self.server.records.get(_key(text))
        content = answer(rec, self.server.mode) if rec else "unknown filing"
        usage = {"prompt_tokens": len(text) // 4, "completion_tokens": len(content) // 4, "total_tokens": 0}
        if body.get("stream"):
            self._stream(body, content, usage)
            return
        resp = {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage,
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, body: dict, content: str, usage: dict) -> None:
        """Server-sent events, as vLLM streams: content chunks, then a usage-only chunk."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "stub", "object": "chat.completion.chunk", "created": 0, "model": body.get("model", "stub")}
        pieces = [content[i : i + 40] for i in range(0, len(content), 40)] or [""]
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            choice = {"index": 0, "delta": {"content": piece}, "finish_reason": "stop" if last else None}
            self.wfile.write(f"data: {json.dumps(base | {'choices': [choice]})}\n\n".encode())
        if (body.get("stream_options") or {}).get("include_usage"):
            self.wfile.write(f"data: {json.dumps(base | {'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

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
