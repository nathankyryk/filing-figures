"""`figures extract`: one 10-K straight from EDGAR, through a model, next to the company's XBRL.

Any company and any fiscal year with an inline-XBRL 10-K works, not only the dataset's: the
filing is fetched, turned into the same model input as the dataset (cover page + three
statements), sent with the frozen prompt, and each answer is scored against the XBRL figures the
same way as in `figures score`.
"""

from __future__ import annotations

import json
from pathlib import Path

from filing_figures.dataset import build_record
from filing_figures.edgar import Edgar
from filing_figures.evaluate import categorize, field_prediction, parse_answer
from filing_figures.labels import FIELD_NAMES
from filing_figures.universe import TICKERS_URL, list_10ks


def find_cik(edgar: Edgar, ticker: str) -> int:
    data = json.loads(edgar.cached(TICKERS_URL, Path("company_tickers.json.gz")))
    want = ticker.upper().replace(".", "-")
    for row in data.values():
        if row["ticker"].upper() == want:
            return int(row["cik_str"])
    raise ValueError(f"no SEC company with ticker {ticker}")


def load_filing(edgar: Edgar, ticker: str, year: int) -> dict:
    """The dataset-style record (input text + XBRL answer key) of one company's 10-K for one year."""
    cik = find_cik(edgar, ticker)
    companyfacts = edgar.companyfacts(cik)
    filings = list_10ks(edgar.submissions(cik), companyfacts, first_year=year, last_year=year)
    if not filings:
        raise ValueError(f"{ticker}: no original inline-XBRL 10-K for fiscal year {year}")
    filing = {"cik": cik, "ticker": ticker.upper(), "sector": "", "split": None, **filings[0]}
    html = edgar.document(cik, filing["accession"], filing["primary_document"])
    return build_record(filing, companyfacts, html)


def compare(rec: dict, raw: str) -> list[dict]:
    """One row per field: the model's answer, the answer key's, and the score category."""
    obj, _ = parse_answer(raw)
    rows = []
    for name in FIELD_NAMES:
        pred = field_prediction(obj, name) if obj is not None else None
        gold = rec["gold"][name]
        if gold is None:
            category = "not scored"
        elif obj is None:
            category = "invalid_json"
        else:
            category = categorize(pred, gold, rec["context_facts"], name)
        rows.append({"field": name, "pred": pred, "gold": gold, "category": category})
    return rows


def _number(x: float | None, field: str) -> str:
    if x is None:
        return "-"
    return f"{x:,.2f}" if field == "eps_diluted" else f"{x:,.0f}"


def table(rows: list[dict]) -> str:
    """The comparison as plain text (ASCII only, so any console prints it)."""
    head = ("field", "model", "period end", "XBRL (answer key)", "period end", "")
    lines = []
    for r in rows:
        p, g = r["pred"] or {}, r["gold"] or {}
        verdict = "right" if r["category"] == "correct" else r["category"]
        if r["category"] not in ("correct", "not scored"):
            verdict = f"WRONG: {verdict}"
        lines.append(
            (
                r["field"],
                _number(p.get("value"), r["field"]),
                p.get("period_end") or "-",
                _number(g.get("value"), r["field"]),
                g.get("period_end") or "-",
                verdict,
            )
        )
    widths = [max(len(row[i]) for row in [head, *lines]) for i in range(len(head))]
    align = ("<", ">", "<", ">", "<", "<")

    def fmt(row) -> str:
        return "  ".join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths, strict=True)).rstrip()

    scored = [r for r in rows if r["category"] != "not scored"]
    right = sum(r["category"] == "correct" for r in scored)
    return "\n".join([fmt(head), *(fmt(x) for x in lines), "", f"{right}/{len(scored)} right"])
