"""The hand-check page: pool drawing, verdict checks, and the server's guards."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from filing_figures.dataset import read_handcheck
from filing_figures.labels import FIELD_NAMES
from filing_figures.review import Review, build_pool, parse_printed, review_reasons, serve

END = "2024-12-31"


def record(i: int, flags: dict | None = None, profit_loss: float = 100e6) -> dict:
    gold = {n: None for n in FIELD_NAMES}
    gold["net_income"] = {
        "value": 100e6,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:NetIncomeLoss",
        "displayed": "100",
    }
    return {
        "id": f"EXM-{i}",
        "cik": i,
        "ticker": "EXM",
        "sector": "Industrials",
        "fiscal_year": 2024,
        "period_end": END,
        "filing_date": "2025-02-20",
        "accession": "0000000001-25-000001",
        "primary_document": "exm-20241231.htm",
        "record_flags": [],
        "input": "## Income statement\n(In millions)\nNet income | 100 | 90\nNet income incl. NCI | 1,100 | 95",
        "gold": gold,
        "flags": {n: list((flags or {}).get(n, [])) for n in FIELD_NAMES},
        "context_facts": [
            {"tag": "us-gaap:NetIncomeLoss", "unit": "USD", "start": "2024-01-01", "end": END, "val": 100e6},
            {"tag": "us-gaap:ProfitLoss", "unit": "USD", "start": "2024-01-01", "end": END, "val": profit_loss},
            {"tag": "us-gaap:ProfitLoss", "unit": "USD", "start": "2024-10-01", "end": END, "val": 30e6},  # Q4
        ],
    }


def test_multi_tag_counts_only_when_the_other_value_scores_differently():
    assert review_reasons(record(0, {"net_income": ["multi_tag"]}, profit_loss=1_100e6)) == ["net_income:multi_tag"]
    assert review_reasons(record(0, {"net_income": ["multi_tag"]}, profit_loss=100.2e6)) == []  # within 0.5%
    nci = record(0, {"net_income": ["tag_from_statement"]}, profit_loss=1_100e6)
    assert review_reasons(nci) == ["net_income:tag_from_statement"]
    assert review_reasons(record(0, {"net_income": ["tag_from_statement"]})) == []  # same value either way


def test_pool_is_seeded_and_flagged_filings_come_first():
    recs = [record(i) for i in range(20)] + [record(100 + i, {"revenue": ["mismatch"]}) for i in range(5)]
    pool = build_pool(recs, target=10, n_random=4, seed=0)
    assert pool == build_pool(recs, target=10, n_random=4, seed=0)
    assert len(pool) == 9  # 4 random + the 5 flagged (fewer flagged than the room left)
    assert sum(p["reason"].startswith("random") for p in pool) == 4
    assert {f"EXM-{100 + i}" for i in range(5)} <= {p["id"] for p in pool}


def test_parse_printed():
    assert parse_printed("(1,234.5)") == -1234.5 and parse_printed("$0.25") == 0.25
    with pytest.raises(ValueError):
        parse_printed("12 million")


def test_fix_is_stored_in_base_units(tmp_path):
    rv = Review([record(0)], [{"order": 0, "id": "EXM-0", "reason": "random"}], tmp_path / "hc.csv")
    rv.save({"id": "EXM-0", "field": "net_income", "verdict": "fix", "printed": "1,100.5", "note": "incl. NCI"})
    row = read_handcheck(tmp_path / "hc.csv")[("EXM-0", "net_income")]
    assert row["value"] == "1100500000" and row["scale"] == "millions" and row["period_end"] == END
    with pytest.raises(ValueError):
        rv.save({"id": "EXM-0", "field": "revenue", "verdict": "fix", "printed": "5"})  # no gold: scale needed
    with pytest.raises(ValueError):
        rv.save({"id": "EXM-9", "field": "revenue", "verdict": "ok"})
    rv.save({"id": "EXM-0", "field": "net_income", "verdict": ""})
    assert read_handcheck(tmp_path / "hc.csv") == {}


def test_server_guards(tmp_path):
    rv = Review([record(0)], [{"order": 0, "id": "EXM-0", "reason": "random"}], tmp_path / "hc.csv")
    server = serve(rv, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(headers, body):
        req = urllib.request.Request(base + "/api/check", json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        with urllib.request.urlopen(base + "/api/record?id=EXM-0") as r:
            view = json.loads(r.read())
        ni = next(f for f in view["fields"] if f["name"] == "net_income")
        assert ni["lines"] == [2]  # "100" on its own, not inside "1,100"
        assert [f["value"] for f in ni["related"]] == [100e6, 100e6]  # the quarter is left out
        ok = {"id": "EXM-0", "field": "net_income", "verdict": "ok"}
        assert post({"Content-Type": "application/json"}, ok) == 403  # no custom header
        assert post({"Content-Type": "text/plain", "X-Handcheck": "1"}, ok) == 403
        assert post({"Content-Type": "application/json", "X-Handcheck": "1", "Host": "evil.example"}, ok) == 403
        assert post({"Content-Type": "application/json", "X-Handcheck": "1"}, ok) == 200
        assert read_handcheck(tmp_path / "hc.csv")[("EXM-0", "net_income")]["verdict"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
