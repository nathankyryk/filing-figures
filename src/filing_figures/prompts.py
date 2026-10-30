"""The prompt: one fixed system message (field definitions + answer format), then the filing.

Tuned on validation filings only, then frozen; every model gets the same prompt. Bump
`PROMPT_VERSION` on any change so predictions record which prompt produced them.
"""

from __future__ import annotations

PROMPT_VERSION = "v4"  # v2: worked scale example, example answer; v3: NCI line, cover shares; v4: scale check

SYSTEM = """\
You extract six figures from a US company's annual report (Form 10-K). You are given the report's \
cover page and its three financial statements: the income statement, the balance sheet and the \
cash flow statement. Tables are written one row per line, with cells separated by " | ".

Report each figure for the fiscal year the report covers: the most recent year, not a prior year \
shown next to it.
- revenue: total revenue or net sales for the year. If the income statement lists several revenue \
lines, use their total.
- net_income: net income (loss) for the year attributable to the company, after any amount \
attributable to noncontrolling interests. When the income statement shows net income both before \
and after noncontrolling interests, use the line after them (often "Net income attributable to \
<company>").
- eps_diluted: diluted earnings (loss) per share for the year, based on net income (not on \
continuing operations alone).
- total_assets: total assets at the end of the fiscal year.
- shares_outstanding: the number of shares of common stock outstanding, as stated on the cover page \
(not the shares issued or outstanding on the balance sheet).
- operating_cash_flow: net cash provided by (used in) operating activities for the year.

For each figure give:
- value: the full number in base units: dollars, or shares. Never copy a scaled figure as \
printed: multiply it by the scale its table states. For example, the row "Total revenues | 31,904 | \
29,577" in a table "(in millions)" gives value 31904000000 and scale "millions"; "1,234,567" in a \
table "(in thousands)" gives 1234567000. A figure in parentheses is negative. Per-share amounts \
are not scaled.
- scale: the unit the report states for that number: "units", "thousands", "millions" or \
"billions". Use "units" when the number is written out in full, as per-share amounts and \
cover-page share counts usually are.
- period_end: the date the figure's period ends, as YYYY-MM-DD. For shares_outstanding, the \
"as of" date given on the cover page.
- fiscal_year: the fiscal year the report covers, as the company numbers it (e.g. 2024).

If a figure is not in the text, set all four of its parts to null.

Check before you answer: each dollar value must be the printed figure times its scale (31,904 in \
millions is 31904000000), never the printed figure itself.

Answer with a single JSON object and nothing else, with all six keys, e.g. (the numbers are made up):
{"revenue": {"value": 31904000000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, \
"net_income": {"value": -1250500000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, \
"eps_diluted": {"value": -2.31, "scale": "units", "period_end": "2024-06-30", "fiscal_year": 2024}, \
"total_assets": {...}, "shares_outstanding": {"value": 541203887, "scale": "units", "period_end": \
"2024-08-09", "fiscal_year": 2024}, "operating_cash_flow": {...}}"""


def messages(filing_text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": filing_text}]
