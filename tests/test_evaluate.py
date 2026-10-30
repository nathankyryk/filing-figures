"""Scoring: tolerances, every error category, answer parsing and the seeded bootstrap."""

import copy
import json

import pytest

from filing_figures.evaluate import (
    close,
    compare,
    field_prediction,
    load_predictions,
    parse_answer,
    score,
    summarize,
)

END, PRIOR_END = "2024-09-28", "2023-09-30"
GOLD = {
    "revenue": {
        "value": 391_035e6,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:Revenues",
    },
    "net_income": {
        "value": 93_736e6,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:NetIncomeLoss",
    },
    "eps_diluted": {
        "value": 6.08,
        "scale": "units",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:EarningsPerShareDiluted",
    },
    "total_assets": {
        "value": 364_980e6,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:Assets",
    },
    "shares_outstanding": {
        "value": 15_115_823_000,
        "scale": "units",
        "period_end": "2024-10-18",
        "fiscal_year": 2024,
        "tag": "dei:EntityCommonStockSharesOutstanding",
    },
    "operating_cash_flow": {
        "value": 118_254e6,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    },
}
FACTS = [
    {"tag": "us-gaap:Revenues", "unit": "USD", "start": "2022-09-25", "end": PRIOR_END, "val": 383_285e6},
    {"tag": "us-gaap:Revenues", "unit": "USD", "start": "2023-10-01", "end": END, "val": 391_035e6},
    {"tag": "us-gaap:EarningsPerShareBasic", "unit": "USD/shares", "start": "2023-10-01", "end": END, "val": 6.11},
    {"tag": "us-gaap:Assets", "unit": "USD", "start": None, "end": PRIOR_END, "val": 352_583e6},
]


def record(rid="EXM-2024", cik=1):
    return {"id": rid, "cik": cik, "gold": copy.deepcopy(GOLD), "context_facts": FACTS}


def answer(**overrides) -> dict:
    out = {name: {k: g[k] for k in ("value", "scale", "period_end", "fiscal_year")} for name, g in GOLD.items()}
    for name, part in overrides.items():
        out[name] = part if part is None else {**out[name], **part}
    return out


def pred(raw, rid="EXM-2024", error=None):
    return {rid: {"id": rid, "raw": raw if isinstance(raw, str) else json.dumps(raw), "error": error}}


def categories(rows):
    return {r["field"]: r["category"] for r in rows}


def test_gold_answer_scores_everything_correct():
    rows = score([record()], pred(answer()))
    assert all(r["correct"] and r["scale_ok"] and r["fiscal_year_ok"] and r["valid_json"] for r in rows)
    s = summarize(rows)
    assert s["overall"]["value"] == 1.0 and s["all_right"]["value"] == 1.0 and s["valid_json"]["value"] == 1.0


def test_tolerances():
    assert close(391_035e6 * 1.0049, 391_035e6, "revenue")
    assert not close(391_035e6 * 1.006, 391_035e6, "revenue")
    assert close(6.084, 6.08, "eps_diluted") and close(6.085, 6.08, "eps_diluted")  # rounds to the cent
    assert not close(6.10, 6.08, "eps_diluted")  # within 0.5%, but a different number of cents
    assert close(0.0, 0.0, "net_income") and not close(1.0, 0.0, "net_income")


@pytest.mark.parametrize(
    ("field", "part", "expected"),
    [
        ("revenue", {"value": 391_035}, "scale"),
        ("revenue", {"value": -391_035e6}, "sign"),
        ("revenue", {"value": 383_285e6, "period_end": PRIOR_END}, "wrong_period"),
        ("total_assets", {"value": 352_583e6}, "wrong_period"),
        ("eps_diluted", {"value": 6.11}, "wrong_line_item"),
        ("revenue", {"period_end": "2024-09-30"}, "date"),
        ("revenue", {"value": None}, "missing"),
        ("net_income", None, "missing"),
        ("operating_cash_flow", {"value": 1.0}, "other"),
    ],
)
def test_error_categories(field, part, expected):
    rows = score([record()], pred(answer(**{field: part})))
    cats = categories(rows)
    assert cats[field] == expected
    assert all(c == "correct" for f, c in cats.items() if f != field)


def test_invalid_json_and_failed_requests():
    assert set(categories(score([record()], pred('{"revenue": 391'))).values()) == {"invalid_json"}
    assert set(categories(score([record()], pred("", error="timeout"))).values()) == {"no_response"}
    assert set(categories(score([record()], {})).values()) == {"no_response"}


def test_parse_answer_is_lenient_about_wrapping_only():
    good = json.dumps(answer())
    assert parse_answer(f"```json\n{good}\n```") == (answer(), True)
    assert parse_answer(f"<think>revenue is 391,035 {{guess}}</think>\n{good}")[1]
    assert parse_answer(f"<think>still thinking {good}") == (None, False)
    obj, valid = parse_answer(json.dumps({"revenue": {"value": "391,035"}}))
    assert obj is not None and not valid  # parsed, but not the schema
    assert field_prediction(obj, "revenue")["value"] == 391_035


def test_unscored_fields_are_skipped():
    rec = record()
    rec["gold"]["shares_outstanding"] = None
    rows = score([rec], pred(answer(shares_outstanding={"value": 1.0})))
    assert "shares_outstanding" not in categories(rows) and all(r["correct"] for r in rows)


def test_bootstrap_is_seeded_and_clusters_by_company():
    recs = [record(f"C{c}-{y}", cik=c) for c in range(1, 9) for y in range(3)]
    preds = {}
    for rec in recs:
        wrong = rec["cik"] % 3 == 0  # every filing of companies 3 and 6 gets revenue wrong
        preds |= pred(answer(revenue={"value": 1.0}) if wrong else answer(), rid=rec["id"])
    rows = score(recs, preds)
    a, b = summarize(rows), summarize(rows)
    assert a["fields"]["revenue"] == b["fields"]["revenue"]
    assert a["fields"]["revenue"]["value"] == pytest.approx(6 / 8)
    assert a["fields"]["revenue"]["lo"] < 0.75 < a["fields"]["revenue"]["hi"]
    assert a["n_companies"] == 8 and a["n_filings"] == 24
    gap = compare(rows, score(recs, {rec["id"]: pred(answer(), rid=rec["id"])[rec["id"]] for rec in recs}))
    # ~10% of resamples draw neither company 3 nor 6 ((6/8)^8), so B is not better there
    assert gap["diff"] == pytest.approx(6 / 144) and gap["lo"] >= 0 < gap["hi"]
    assert 0.05 < gap["p_b_not_better"] < 0.2


def test_later_prediction_line_replaces_earlier(tmp_path):
    path = tmp_path / "p.jsonl"
    lines = [{"id": "A", "raw": "", "error": "timeout"}, {"id": "A", "raw": "{}", "error": None}]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    assert load_predictions(path)["A"]["error"] is None
