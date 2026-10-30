"""The no-model baseline: regular expressions over the same input text the models read.

Per statement: find the row by its label, take the current year's column (the latest year in
the header), multiply by the scale in the caption. Shares and dates come from the cover page.
It answers in the models' JSON format and is scored the same way. Tuned on train/valid only.
"""

from __future__ import annotations

import re
from datetime import datetime

from filing_figures.inputs import date_near

I = re.IGNORECASE
SECTION = re.compile(r"^## (.+)$", re.MULTILINE)
SCALE = re.compile(r"(thousands|millions|billions)", I)  # captions are often run together: "IncomeIn millions"
SCALE_POWER = {"units": 0, "thousands": 3, "millions": 6, "billions": 9}
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", I)
# "year ended December 31, 2024", "Fiscal Year Ended: September 30, 2019", "fiscal year ended | October 3, 2020"
FISCAL_YEAR_END = re.compile(rf"year\s+ended[\s:|]*(?:on\s+)?(({MONTHS})\s+\d{{1,2}},?\s+\d{{4}})", I)
SHARE_COUNT = re.compile(r"(?<![\d$.,])\d{1,3}(?:,\d{3}){2,}(?![\d,]*\.\d)")
SHARE_WORDS = re.compile(r"(?<![\d$.,])(\d+(?:\.\d+)?)\s*(million|billion)\s+shares", I)

REVENUE = (
    re.compile(r"^total\s+(net\s+)?(revenues?|sales|net\s+sales|operating\s+revenues?)\b", I),
    re.compile(r"^(net\s+)?(revenues?|sales)(,?\s*net)?\b|^net\s+sales\b|^operating\s+revenues?\b", I),
    re.compile(r"revenue|sales", I),
)
REVENUE_HEADING = re.compile(r"^(total\s+)?(net\s+)?(revenues?|sales)\s*:?$", I)
NET_INCOME = (
    re.compile(r"^net\s+(income|earnings|loss|\(loss\)|income\s*\(loss\)).*attributable\s+to\s+(?!non)", I),
    re.compile(r"^net\s+(income|earnings)(\s*\(loss\))?$|^net\s+(\(loss\)\s+)?(income|loss)(\s*\(loss\))?$", I),
    re.compile(r"^net\s+(income|earnings|\(loss\)|loss)", I),
)
NOT_NET_INCOME = re.compile(r"per\s+share|comprehensive|non-?controlling|before|continuing|discontinued", I)
DILUTED = re.compile(r"diluted", I)
NOT_EPS = re.compile(r"shares|weighted|average|number", I)
TOTAL_ASSETS = re.compile(r"^total\s+assets$", I)
CURRENT_ASSETS = re.compile(r"^total\s+current\s+assets", I)
OPERATING_CASH = (
    re.compile(r"net\s+cash.*operat", I),
    re.compile(r"cash\s+(provided|generated)\s+(by|from)\s+operat", I),
    re.compile(r"^(total|net)\b.*operating\s+activities", I),
)


def cell_number(cell: str) -> float | None:
    s = cell.replace("$", "").replace(" ", "").strip()
    negative = s.startswith("(") or s.endswith(")")
    s = s.strip("()")
    if not re.fullmatch(r"-?\d[\d,]*(\.\d+)?", s):
        return None
    v = float(s.replace(",", ""))
    return -v if negative else v


def sections(text: str) -> dict[str, list[str]]:
    """{"cover page": lines, "income statement": lines, ...}"""
    marks = list(SECTION.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip().lower()] = text[m.end() : end].strip().split("\n")
    return out


class Statement:
    def __init__(self, lines: list[str]):
        self.rows: list[tuple[str, list[float]]] = []
        header_years: list[int] = []
        caption: list[str] = []  # the scale note sits in the first lines, above or below the years
        for i, line in enumerate(lines):
            cells = line.split(" | ")
            first = cell_number(cells[0])
            label = "" if first is not None else cells[0].strip()
            values = [v for v in (cell_number(c) for c in (cells if first is not None else cells[1:])) if v is not None]
            if not header_years:  # "2024 | 2023" or "September 28, 2024 | September 30, 2023"
                years = [int(y) for y in YEAR.findall(line)]
                if len(years) >= 2 and all(float(y) == v for y, v in zip(years, values, strict=False)):
                    header_years = years
                    caption.append(line)
                    continue
            if i < 10 and not values:  # not a data row: "Shares (in thousands) | 1,234" can't set it
                caption.append(line)
            self.rows.append((label, values))
        self.latest_last = len(header_years) >= 2 and header_years[-1] > header_years[0]
        m = SCALE.search(" ".join(caption))
        self.scale = m.group(1).lower() if m else "units"

    def current(self, values: list[float]) -> float | None:
        if not values:
            return None
        return values[-1] if self.latest_last else values[0]

    def find(self, patterns, exclude: re.Pattern | None = None) -> float | None:
        for p in patterns:
            for label, values in self.rows:
                if values and p.search(label) and not (exclude and exclude.search(label)):
                    return self.current(values)
        return None

    def revenue(self) -> float | None:
        v = self.find(REVENUE)
        if v is not None:
            return v
        heading = False  # "Revenue:" then its lines, then an unlabelled total
        for label, values in self.rows:
            if REVENUE_HEADING.search(label) and not values:
                heading = True
            elif heading and not label and values:
                return self.current(values)
        return None

    def eps(self) -> float | None:
        in_block, block_first = False, None
        for label, values in self.rows:
            per_share = bool(values) and all(abs(v) < 1000 for v in values)
            if DILUTED.search(label) and not NOT_EPS.search(label) and per_share:
                return self.current(values)
            if DILUTED.search(label) and not values:
                in_block = True  # "Diluted earnings per share:" heading; the figures follow
                continue
            if in_block and per_share:
                if re.match(r"(net|total)\b", label, I) or not label:
                    return self.current(values)
                block_first = block_first if block_first is not None else self.current(values)
        return block_first

    def total_assets(self) -> float | None:
        v = self.find([TOTAL_ASSETS])
        if v is not None:
            return v
        # Some balance sheets leave the total assets row unlabelled: take the largest unlabelled
        # row between "Total current assets" and the liabilities
        seen_current, best = False, None
        for label, values in self.rows:
            if CURRENT_ASSETS.search(label):
                seen_current = True
            elif seen_current and re.search(r"liabilit", label, I):
                break
            elif seen_current and not label and values:
                v = self.current(values)
                best = v if best is None or (v is not None and v > best) else best
        return best


def _iso(date_text: str) -> str | None:
    m = DATE.search(date_text)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date().isoformat()
    except ValueError:
        return None


def _shares(cover: str) -> tuple[float | None, str | None, str]:
    """(count, its "as of" date, scale) from the first sentence about shares outstanding."""
    for sentence in re.split(r"(?<=[.;])\s+|\n", cover):
        if not re.search(r"outstanding", sentence, I) or re.search(r"aggregate\s+market|affiliates", sentence, I):
            continue
        m = SHARE_COUNT.search(sentence)
        if m:
            return float(m.group(0).replace(",", "")), date_near(cover, m.group(0)), "units"
        m = SHARE_WORDS.search(sentence)
        if m:
            scale = m.group(2).lower() + "s"
            return float(m.group(1)) * 10 ** SCALE_POWER[scale], date_near(cover, m.group(0)), scale
    return None, None, "units"


def _part(value, scale, period_end, fiscal_year) -> dict:
    if value is None:
        return {"value": None, "scale": None, "period_end": None, "fiscal_year": None}
    return {"value": value, "scale": scale, "period_end": period_end, "fiscal_year": fiscal_year}


def extract(text: str) -> dict:
    secs = sections(text)
    cover = "\n".join(secs.get("cover page", []))
    m = FISCAL_YEAR_END.search(cover)
    period_end = _iso(m.group(1)) if m else None
    fiscal_year = int(period_end[:4]) if period_end else None
    stmts = {k: Statement(secs[k]) for k in ("income statement", "balance sheet", "cash flow statement") if k in secs}
    inc, bal, cash = (stmts.get(k) for k in ("income statement", "balance sheet", "cash flow statement"))

    def money(st: Statement | None, value: float | None) -> dict:
        if st is None or value is None:
            return _part(None, None, None, None)
        return _part(value * 10 ** SCALE_POWER[st.scale], st.scale, period_end, fiscal_year)

    shares, as_of, share_scale = _shares(cover)
    return {
        "revenue": money(inc, inc and inc.revenue()),
        "net_income": money(inc, inc and inc.find(NET_INCOME, NOT_NET_INCOME)),
        "eps_diluted": _part(inc.eps() if inc else None, "units", period_end, fiscal_year),
        "total_assets": money(bal, bal and bal.total_assets()),
        "shares_outstanding": _part(shares, share_scale, as_of, fiscal_year),
        "operating_cash_flow": money(cash, cash and cash.find(OPERATING_CASH)),
    }
