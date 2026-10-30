"""Scoring against the XBRL answer key: field accuracy, error categories and 95% CIs.

A field is correct when its value is within 0.5% of the gold value AND its period end is exactly
the gold one. Diluted EPS is reported to the cent, so it must match to the cent (within $0.005):
0.5% of a $6.08 EPS would also accept $6.11, the basic EPS, a wrong line item. Scale and fiscal year are
reported as separate columns. Only fields with a gold value are scored.

Each wrong field gets the first category that fits, in this order:
  no_response (the request failed), invalid_json, missing (no value), date (right value, wrong
  period end), scale (off by 10^3, 10^6 or 10^9), sign, wrong_period (the same concept's value
  for another period, filed in the same 10-K), wrong_line_item (any other fact filed in the 10-K,
  e.g. basic EPS or net income including noncontrolling interests), other.

CIs come from a bootstrap over companies, not filings: one company's six 10-Ks share a format,
so they are not independent. `compare` is the paired version for the gap between two runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from filing_figures.labels import FIELD_NAMES, FIELDS
from filing_figures.schema import Answer

REL_TOL = 0.005
EPS_ABS_TOL = 0.005
SCALE_FACTORS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
CATEGORIES = (
    "correct",
    "no_response",
    "invalid_json",
    "missing",
    "date",
    "scale",
    "sign",
    "wrong_period",
    "wrong_line_item",
    "other",
)
TAGS = {f.name: f.tags for f in FIELDS}
N_BOOT = 2000

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?")


def parse_answer(raw: str | None) -> tuple[dict | None, bool]:
    """(the JSON object, whether it matches the schema exactly).

    Lenient about wrapping: drops <think> blocks and code fences and reads the outermost {...}.
    An unfinished <think> block (the model ran out of tokens) is no answer at all."""
    if not raw:
        return None, False
    text = _THINK.sub("", raw)
    if "<think>" in text:
        return None, False
    text = _FENCE.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None, False
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, False
    if not isinstance(obj, dict):
        return None, False
    try:
        Answer.model_validate(obj)
    except ValidationError:
        return obj, False
    return obj, True


def _number(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int | float):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").replace("$", "").strip()
        negative = s.startswith("(") and s.endswith(")")
        try:
            v = float(s.strip("()"))
        except ValueError:
            return None
        return -v if negative else v
    return None


def field_prediction(obj: dict, name: str) -> dict | None:
    """One field of a parsed answer, or None when it gives no usable value."""
    part = obj.get(name)
    if not isinstance(part, dict):
        return None
    value = _number(part.get("value"))
    if value is None:
        return None
    try:
        fiscal_year = int(part["fiscal_year"]) if part.get("fiscal_year") is not None else None
    except (TypeError, ValueError):
        fiscal_year = None
    period_end = part.get("period_end")
    return {
        "value": value,
        "scale": part.get("scale"),
        "period_end": period_end.strip() if isinstance(period_end, str) else None,
        "fiscal_year": fiscal_year,
    }


def close(pred: float, gold: float, field: str) -> bool:
    tol = EPS_ABS_TOL if field == "eps_diluted" else abs(gold) * REL_TOL
    return abs(pred - gold) <= tol + 1e-9


def categorize(pred: dict | None, gold: dict, facts: list[dict], field: str) -> str:
    """The category of one prediction (see the module docstring for the order)."""
    if pred is None:
        return "missing"
    v, g = pred["value"], gold["value"]
    if close(v, g, field):
        return "correct" if pred["period_end"] == gold["period_end"] else "date"
    if g != 0 and any(close(v, g * f, field) for f in SCALE_FACTORS):
        return "scale"
    if g != 0 and close(v, -g, field):
        return "sign"
    matches = [f for f in facts if close(v, f["val"], field) and not close(f["val"], g, field)]
    if any(f["tag"] in TAGS[field] and (f["end"] != gold["period_end"] or f["tag"] == gold["tag"]) for f in matches):
        return "wrong_period"
    if matches:
        return "wrong_line_item"
    return "other"


def load_predictions(path: Path) -> dict[str, dict]:
    """{record id: prediction}; a later line for the same id (a retry) replaces an earlier one."""
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out[p["id"]] = p
    return out


def score(records: list[dict], predictions: dict[str, dict]) -> list[dict]:
    """One row per scored field (fields with a gold value)."""
    rows = []
    for rec in records:
        p = predictions.get(rec["id"])
        failed = p is None or bool(p.get("error"))
        obj, valid = (None, False) if failed else parse_answer(p.get("raw"))
        for name in FIELD_NAMES:
            gold = rec["gold"][name]
            if gold is None:
                continue
            pred = None
            if failed:
                category = "no_response"
            elif obj is None:
                category = "invalid_json"
            else:
                pred = field_prediction(obj, name)
                category = categorize(pred, gold, rec["context_facts"], name)
            rows.append(
                {
                    "id": rec["id"],
                    "cik": rec["cik"],
                    "field": name,
                    "category": category,
                    "correct": category == "correct",
                    "scale_ok": pred is not None and pred["scale"] == gold["scale"],
                    "fiscal_year_ok": pred is not None and pred["fiscal_year"] == gold["fiscal_year"],
                    "valid_json": valid,
                    "pred": pred,
                    "gold": {k: gold[k] for k in ("value", "scale", "period_end", "fiscal_year")},
                }
            )
    return rows


class _Bootstrap:
    """Ratio metrics with a company-cluster bootstrap; one resample matrix serves every metric."""

    def __init__(self, ciks: list[int], n_boot: int = N_BOOT, seed: int = 0):
        self.pos = {c: i for i, c in enumerate(sorted(set(ciks)))}
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, len(self.pos), size=(n_boot, len(self.pos)))

    def sums(self, pairs) -> tuple[np.ndarray, np.ndarray]:
        num, den = np.zeros(len(self.pos)), np.zeros(len(self.pos))
        for cik, ok in pairs:
            num[self.pos[cik]] += ok
            den[self.pos[cik]] += 1
        return num, den

    def samples(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        n, d = num[self.idx].sum(1), den[self.idx].sum(1)
        return np.divide(n, d, out=np.full(len(n), np.nan), where=d > 0)

    def metric(self, pairs) -> dict:
        num, den = self.sums(pairs)
        boot = self.samples(num, den)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
        value = num.sum() / den.sum() if den.sum() else float("nan")
        return {"value": float(value), "lo": float(lo), "hi": float(hi), "n": int(den.sum())}


def _by_filing(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["id"]].append(r)
    return out


def summarize(rows: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    boot = _Bootstrap([r["cik"] for r in rows], n_boot, seed)
    filings = _by_filing(rows)
    return {
        "n_filings": len(filings),
        "n_companies": len(boot.pos),
        "n_fields": len(rows),
        "overall": boot.metric((r["cik"], r["correct"]) for r in rows),
        "fields": {
            name: boot.metric((r["cik"], r["correct"]) for r in rows if r["field"] == name) for name in FIELD_NAMES
        },
        "all_right": boot.metric((fr[0]["cik"], all(r["correct"] for r in fr)) for fr in filings.values()),
        "valid_json": boot.metric((fr[0]["cik"], fr[0]["valid_json"]) for fr in filings.values()),
        "scale": boot.metric((r["cik"], r["scale_ok"]) for r in rows),
        "fiscal_year": boot.metric((r["cik"], r["fiscal_year_ok"]) for r in rows),
        "categories": dict(Counter(r["category"] for r in rows)),
    }


def compare(rows_a: list[dict], rows_b: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Overall accuracy of B minus A, with a paired company bootstrap (both runs scored on the
    same filings). `p_b_not_better` is the share of resamples where B does not beat A."""
    key = lambda r: (r["id"], r["field"])
    a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    if a.keys() != b.keys():
        raise ValueError("the two runs were not scored on the same fields")
    boot = _Bootstrap([r["cik"] for r in rows_a], n_boot, seed)
    num_a, den = boot.sums((r["cik"], r["correct"]) for r in a.values())
    num_b, _ = boot.sums((b[k]["cik"], b[k]["correct"]) for k in a)
    diff = boot.samples(num_b, den) - boot.samples(num_a, den)
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return {
        "diff": float((num_b.sum() - num_a.sum()) / den.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "p_b_not_better": float(np.mean(diff <= 0)),
    }


def examples(rows: list[dict], per_category: int = 3) -> dict[str, list[dict]]:
    """A few wrong fields of each category, for the README."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"] != "correct" and len(out[r["category"]]) < per_category:
            out[r["category"]].append({k: r[k] for k in ("id", "field", "pred", "gold")})
    return dict(out)


def table(summary: dict) -> str:
    """A plain-text results table."""

    def cell(m: dict) -> str:
        return f"{100 * m['value']:5.1f}% [{100 * m['lo']:4.1f}, {100 * m['hi']:4.1f}]"

    lines = [
        f"{summary['n_filings']} filings, {summary['n_companies']} companies, {summary['n_fields']} scored fields",
        f"  {'overall':22s}{cell(summary['overall'])}",
    ]
    lines += [f"  {name:22s}{cell(m)}  (n={m['n']})" for name, m in summary["fields"].items()]
    lines += [
        f"  {'all fields right':22s}{cell(summary['all_right'])}",
        f"  {'valid JSON':22s}{cell(summary['valid_json'])}",
        f"  {'scale right':22s}{cell(summary['scale'])}",
        f"  {'fiscal year right':22s}{cell(summary['fiscal_year'])}",
        "  categories: "
        + ", ".join(f"{c}={summary['categories'][c]}" for c in CATEGORIES if c in summary["categories"]),
    ]
    return "\n".join(lines)
