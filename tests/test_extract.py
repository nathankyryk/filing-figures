"""`figures extract`: the answer-vs-XBRL table (the EDGAR and model calls are tested elsewhere)."""

import json

from filing_figures.extract import compare, table
from filing_figures.labels import FIELD_NAMES

END = "2024-09-29"


def gold(value, scale="millions", end=END):
    return {"value": value, "scale": scale, "period_end": end, "fiscal_year": 2024, "tag": "x", "displayed": ""}


RECORD = {
    "gold": {
        "revenue": gold(36_176.2e6),
        "net_income": gold(3_760.9e6),
        "eps_diluted": gold(3.31, "units"),
        "total_assets": gold(31_339.3e6),
        "shares_outstanding": None,  # e.g. several share classes: not scored
        "operating_cash_flow": gold(6_095.6e6),
    },
    "context_facts": [],
}


def answer(**override):
    out = {}
    for name in FIELD_NAMES:
        g = RECORD["gold"][name] or gold(1.0)
        out[name] = {k: g[k] for k in ("value", "scale", "period_end", "fiscal_year")}
    out.update(override)
    return json.dumps(out)


def test_all_right_and_unscored():
    rows = compare(RECORD, answer())
    assert [r["category"] for r in rows] == ["correct"] * 4 + ["not scored", "correct"]
    text = table(rows)
    assert text.endswith("5/5 right")
    assert "36,176,200,000" in text and "3.31" in text


def test_wrong_field_is_named():
    slip = {"value": 361_762e6, "scale": "millions", "period_end": END, "fiscal_year": 2024}
    rows = compare(RECORD, answer(revenue=slip))
    assert rows[0]["category"] == "off_by_10x"
    assert "WRONG: off_by_10x" in table(rows) and table(rows).endswith("4/5 right")


def test_unreadable_answer():
    rows = compare(RECORD, "not json")
    assert {r["category"] for r in rows} == {"invalid_json", "not scored"}
    assert table(rows).endswith("0/5 right")
