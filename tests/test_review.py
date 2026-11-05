"""The hand-check page: pool drawing, verdict checks, and the server's guards."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from filing_figures.dataset import read_handcheck
from filing_figures.labels import FIELD_NAMES, field_names
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


# ---- task v2: the ood-test hand-check --------------------------------------------------------

V2_NAMES = field_names("v2")


def gold(value, tag, scale="millions", end=END, displayed="1"):
    return {"value": value, "scale": scale, "period_end": end, "fiscal_year": 2024, "tag": tag, "displayed": displayed}


def record_v2(
    i: int, equity_tag="us-gaap:StockholdersEquity", capex_tag="us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
):
    rec = record(i)
    rec["gold"] = dict.fromkeys(V2_NAMES)
    rec["gold"] |= {
        "net_income": gold(100e6, "us-gaap:NetIncomeLoss", displayed="100"),
        "revenue_prior": gold(900e6, "us-gaap:Revenues", end="2023-12-31", displayed="900"),
        "eps_basic": gold(1.25, "us-gaap:EarningsPerShareBasic", scale="units", displayed="1.25"),
        "equity": gold(500e6, equity_tag, displayed="500"),
        "capex": gold(40e6, capex_tag, displayed="40"),
        "fcf": gold(60e6, "derived", displayed=None),
    }
    rec["flags"] = {n: [] for n in V2_NAMES}
    rec["context_facts"] += [
        {"tag": "us-gaap:Revenues", "unit": "USD", "start": "2023-01-01", "end": "2023-12-31", "val": 900e6},
        {"tag": "us-gaap:Revenues", "unit": "USD", "start": "2024-01-01", "end": END, "val": 1000e6},
    ]
    return rec


def test_v2_traps_flag_a_filing_and_v1_ignores_them():
    nci = "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    rec = record_v2(0, equity_tag=nci, capex_tag="us-gaap:PaymentsToAcquireProductiveAssets")
    assert review_reasons(rec, "v2") == ["equity:with_nci", "capex:fallback_tag"]
    assert review_reasons(record_v2(1), "v2") == []
    assert review_reasons(rec) == []  # v1: six fields, no traps


def test_v2_pool_is_25_random_and_25_flagged():
    nci = "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    recs = [record_v2(i) for i in range(60)] + [record_v2(100 + i, equity_tag=nci) for i in range(40)]
    pool = build_pool(recs, target=50, n_random=25, task="v2")
    assert pool == build_pool(recs, target=50, n_random=25, task="v2")
    assert len(pool) == len({p["id"] for p in pool}) == 50
    assert sum(p["reason"].startswith("random") for p in pool) == 25
    flagged = [p for p in pool if not p["reason"].startswith("random")]
    assert len(flagged) == 25 and all(p["reason"] == "equity:with_nci" for p in flagged)


def test_v2_page_checks_eleven_fields_and_fcf_follows_its_parts(tmp_path):
    rv = Review([record_v2(0)], [{"order": 0, "id": "EXM-0", "reason": "random"}], tmp_path / "hc_v2.csv", "v2")
    assert rv.pool_view()[0]["fields"] == 11
    view = rv.record_view("EXM-0")
    assert [f["name"] for f in view["fields"]] == list(V2_NAMES)
    by_name = {f["name"]: f for f in view["fields"]}
    assert by_name["eps_basic"]["per_share"] and not by_name["equity"]["per_share"]
    assert by_name["fcf"]["parts"] == ["operating_cash_flow", "capex"]
    assert [f["end"] for f in by_name["revenue_prior"]["related"]] == ["2023-12-31"]  # the year before
    with pytest.raises(ValueError, match="follows its parts"):
        rv.save({"id": "EXM-0", "field": "fcf", "verdict": "ok"})
    rv.save({"id": "EXM-0", "field": "eps_basic", "verdict": "fix", "printed": "1.27"})
    rv.save({"id": "EXM-0", "field": "revenue_prior", "verdict": "fix", "printed": "905", "period_end": "2023-12-31"})
    checks = read_handcheck(tmp_path / "hc_v2.csv")
    assert checks[("EXM-0", "eps_basic")]["value"] == "1.27" and checks[("EXM-0", "eps_basic")]["scale"] == "units"
    assert checks[("EXM-0", "revenue_prior")]["value"] == "905000000"
    assert rv.summary()["verdicts"] == {"fix": 2}
