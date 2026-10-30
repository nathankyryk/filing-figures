"""One loop for every model. vLLM, OpenRouter and the stub server all speak the OpenAI chat API;
the rules baseline runs in-process. `predict` writes one JSON line per filing:
  {id, raw, error, finish_reason, usage, latency_s, model, endpoint, prompt_version, schema, ts}
Failures are written too (with `error`), never dropped; a re-run retries only those.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from filing_figures import prompts
from filing_figures.schema import response_format

ENDPOINTS = {  # name: (base URL, env var holding the API key)
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "vllm": ("http://localhost:8000/v1", None),
    "stub": ("http://127.0.0.1:8001/v1", None),
}


@dataclass
class Config:
    endpoint: str  # a name from ENDPOINTS, a base URL, or "rules"
    model: str = ""
    max_tokens: int = 2048
    temperature: float | None = 0.0  # None: not sent (some API models reject it)
    schema: bool = False  # constrained decoding with the JSON schema
    reasoning: str | None = None  # "off", or an effort level ("low", "medium", "high")
    workers: int = 8
    max_cost: float | None = None  # USD; no new requests once this much is spent
    timeout: float = 600.0
    extra_body: dict = field(default_factory=dict)

    def request_extras(self) -> dict:
        body = dict(self.extra_body)
        if self.endpoint == "openrouter":
            body.setdefault("usage", {"include": True})  # report the cost of each request
            if self.reasoning:
                body.setdefault(
                    "reasoning", {"enabled": False} if self.reasoning == "off" else {"effort": self.reasoning}
                )
            if self.schema:
                body.setdefault("provider", {"require_parameters": True})  # only providers that honour the schema
        elif self.reasoning == "off":  # vLLM: Qwen-style chat templates take this switch
            body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return body


def _client(cfg: Config):
    from openai import OpenAI

    base_url, key_env = ENDPOINTS.get(cfg.endpoint, (cfg.endpoint, None))
    api_key = os.environ.get(key_env, "") if key_env else "none"
    if key_env and not api_key:
        raise RuntimeError(f"{key_env} is not set (put it in .env)")
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=5, timeout=cfg.timeout)


def _ask_model(client, cfg: Config, rec: dict) -> dict:
    kwargs = {"model": cfg.model, "messages": prompts.messages(rec["input"]), "max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.schema:
        kwargs["response_format"] = response_format()
    extras = cfg.request_extras()
    if extras:
        kwargs["extra_body"] = extras
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return {
        "raw": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "usage": resp.usage.model_dump() if resp.usage else {},
    }


def _ask_rules(rec: dict) -> dict:
    from filing_figures import rules

    return {"raw": json.dumps(rules.extract(rec["input"])), "finish_reason": "stop", "usage": {}}


PROVIDER_TRIES = 3


def done_ids(out_path: Path) -> set[str]:
    """Ids already answered without an error (a re-run skips them)."""
    if not out_path.exists():
        return set()
    last = {}
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                last[p["id"]] = p
    return {i for i, p in last.items() if not p.get("error")}


def predict(records: list[dict], cfg: Config, out_path: Path, on_result=None) -> dict:
    """Ask the model about every record not yet answered; append one line per record."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in records if r["id"] not in done_ids(out_path)]
    client = None if cfg.endpoint == "rules" else _client(cfg)
    lock = threading.Lock()
    stats = {"asked": 0, "errors": 0, "skipped_for_cost": 0, "cost": 0.0, "todo": len(todo)}

    def one(rec: dict) -> None:
        with lock:
            if cfg.max_cost is not None and stats["cost"] >= cfg.max_cost:
                stats["skipped_for_cost"] += 1
                return
        t0 = time.perf_counter()
        cost = 0.0
        for _ in range(PROVIDER_TRIES):
            try:
                got = _ask_rules(rec) if client is None else _ask_model(client, cfg, rec)
                error = None
            except Exception as e:  # noqa: BLE001 - recorded in the output line, retried on the next run
                got, error = {"raw": "", "finish_reason": None, "usage": {}}, repr(e)
            cost += float(got["usage"].get("cost") or 0.0)
            # A provider can fail mid-answer and still return the partial text: not the model's error
            if got["finish_reason"] != "error":
                break
            error = "provider error mid-answer (finish_reason=error)"
        line = {
            "id": rec["id"],
            **got,
            "error": error,
            "latency_s": round(time.perf_counter() - t0, 3),
            "model": cfg.model or cfg.endpoint,
            "endpoint": cfg.endpoint,
            "prompt_version": prompts.PROMPT_VERSION,
            "schema": cfg.schema,
            "reasoning": cfg.reasoning,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with lock:
            stats["asked"] += 1
            stats["errors"] += error is not None
            stats["cost"] += cost
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            if on_result:
                on_result(line)

    with ThreadPoolExecutor(max(1, cfg.workers)) as pool:
        list(pool.map(one, todo))
    return stats
