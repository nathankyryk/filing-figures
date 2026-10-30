"""The whole harness against the stub server: client -> predictions file -> scorer."""

import copy
import threading

import pytest
from test_evaluate import END, FACTS, GOLD, PRIOR_END

from filing_figures.client import Config, predict
from filing_figures.evaluate import load_predictions, score, summarize
from filing_figures.stub import StubServer

PRIOR_FACTS = [
    *FACTS,
    {"tag": "us-gaap:NetIncomeLoss", "unit": "USD", "start": "2022-09-25", "end": PRIOR_END, "val": 96_995e6},
    {"tag": "us-gaap:NetIncomeLoss", "unit": "USD", "start": "2024-06-30", "end": END, "val": 14_736e6},  # Q4
    {
        "tag": "us-gaap:EarningsPerShareDiluted",
        "unit": "USD/shares",
        "start": "2022-09-25",
        "end": PRIOR_END,
        "val": 6.13,
    },
    {
        "tag": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "unit": "USD",
        "start": "2022-09-25",
        "end": PRIOR_END,
        "val": 110_543e6,
    },
]
RECORDS = [
    {"id": f"EXM-{i}", "cik": i, "input": f"filing text {i}", "gold": copy.deepcopy(GOLD), "context_facts": PRIOR_FACTS}
    for i in range(3)
]


def run(mode, tmp_path):
    server = StubServer(RECORDS, mode, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / f"{mode}.jsonl"
        cfg = Config(endpoint=server.url, model="stub", workers=2)
        stats = predict(RECORDS, cfg, out)
        again = predict(RECORDS, cfg, out)  # a re-run asks nothing that already has an answer
    finally:
        server.shutdown()
        server.server_close()
    assert stats["asked"] == 3 and stats["errors"] == 0 and again["todo"] == 0
    return score(RECORDS, load_predictions(out))


def test_gold_answers_score_100(tmp_path):
    rows = run("gold", tmp_path)
    assert summarize(rows)["overall"]["value"] == 1.0 and len(rows) == 18


def test_prior_year_answers_are_wrong_period(tmp_path):
    cats = {(r["id"], r["field"]): r["category"] for r in run("prior", tmp_path)}
    for (_, field), cat in cats.items():
        assert cat == ("missing" if field == "shares_outstanding" else "wrong_period"), field


def test_garbage_is_invalid_json(tmp_path):
    assert {r["category"] for r in run("garbage", tmp_path)} == {"invalid_json"}


def test_unreachable_server_is_recorded_not_dropped(tmp_path):
    out = tmp_path / "down.jsonl"
    cfg = Config(endpoint="http://127.0.0.1:9/v1", model="stub", workers=1, timeout=2)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "filing_figures.client._client",
            lambda c: __import__("openai").OpenAI(base_url=c.endpoint, api_key="none", max_retries=0, timeout=2),
        )
        stats = predict(RECORDS[:1], cfg, out)
    assert stats["errors"] == 1
    assert {r["category"] for r in score(RECORDS[:1], load_predictions(out))} == {"no_response"}


def test_provider_error_mid_answer_is_retried(tmp_path, monkeypatch):
    from filing_figures import client

    calls = []

    def flaky(_client, _cfg, rec):
        calls.append(rec["id"])
        reason = "error" if len(calls) == 1 else "stop"
        return {"raw": "{}", "finish_reason": reason, "usage": {"cost": 0.01}}

    monkeypatch.setattr(client, "_ask_model", flaky)
    monkeypatch.setattr(client, "_client", lambda cfg: object())
    out = tmp_path / "flaky.jsonl"
    stats = predict(RECORDS[:1], Config(endpoint="stub", model="m", workers=1), out)
    assert calls == ["EXM-0", "EXM-0"] and stats["errors"] == 0 and abs(stats["cost"] - 0.02) < 1e-9
