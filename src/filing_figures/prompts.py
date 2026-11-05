"""The prompt: one fixed system message per task (field definitions + answer format), then the filing.

Tuned on validation filings only, then frozen; every model gets the same prompt. Bump the task's
version on any change so predictions record which prompt produced them.
"""

from __future__ import annotations

from filing_figures.labels import field_names

PROMPT_VERSION = "v4"  # v2: worked scale example, example answer; v3: NCI line, cover shares; v4: scale check
# v5: task v2's twelve fields, tuned on valid (Nov 5: equity subtotal, one capex line, a decimal example), then frozen
PROMPT_VERSIONS = {"v1": PROMPT_VERSION, "v2": "v5"}

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


SYSTEM_V2 = """You extract twelve figures from a US company's annual report (Form 10-K). You are given the report's cover page and its three financial statements: the income statement, the balance sheet and the cash flow statement. Tables are written one row per line, with cells separated by " | ".

Report each figure for the fiscal year the report covers: the most recent year, not a prior year shown next to it. The one exception is revenue_prior.
- revenue: total revenue or net sales for the year. If the income statement lists several revenue lines, use their total.
- revenue_prior: the same total revenue for the fiscal year before, as this report's income statement shows it (if the report restates last year's figure, use the restated one). Its period_end is that earlier year's end.
- net_income: net income (loss) for the year attributable to the company, after any amount attributable to noncontrolling interests. When the income statement shows net income both before and after noncontrolling interests, use the line after them (often "Net income attributable to <company>").
- eps_basic: basic earnings (loss) per share for the year, based on net income (not on continuing operations alone).
- eps_diluted: diluted earnings (loss) per share for the year, based on net income (not on continuing operations alone).
- total_assets: total assets at the end of the fiscal year.
- cash: cash and cash equivalents at the end of the fiscal year, as the balance sheet's line shows it. Do not add restricted cash or short-term investments listed on other lines.
- equity: the stockholders' equity of the company's own shareholders at the end of the fiscal year. Many balance sheets print it as a subtotal (often "Total <company> stockholders' equity" or "Total stockholders' equity"), then a line for noncontrolling interests, then a larger "Total equity" that adds them in: equity is that subtotal, never the larger total after the noncontrolling-interests line. If the balance sheet prints only one total, use it as printed, even if it includes noncontrolling interests; don't subtract them yourself.
- shares_outstanding: the number of shares of common stock outstanding, as stated on the cover page (not the shares issued or outstanding on the balance sheet).
- operating_cash_flow: net cash provided by (used in) operating activities for the year.
- capex: capital expenditures for the year: the cash paid for property, plant and equipment, from the investing section of the cash flow statement (often "Purchases of property and equipment" or "Capital expenditures"), not acquisitions of businesses or purchases of investments. Use that one line as printed: don't add other lines to it, such as equipment leased to others, other kinds of assets, or changes in accrued capital expenditures. The statement shows it in parentheses as a cash outflow; report it as a positive number.
- fcf: free cash flow for the year: operating_cash_flow minus capex, computed from your two values above. Its scale and period_end are those of operating_cash_flow.

For each figure give:
- value: the full number in base units: dollars, or shares. Never copy a scaled figure as printed: multiply it by the scale its table states. For example, the row "Total revenues | 31,904 | 29,577" in a table "(in millions)" gives value 31904000000 and scale "millions"; "1,234,567" in a table "(in thousands)" gives 1234567000. A figure in parentheses is negative, except capex (see above). Per-share amounts are not scaled.
- scale: the unit the report states for that number: "units", "thousands", "millions" or "billions". Use "units" when the number is written out in full, as per-share amounts and cover-page share counts usually are.
- period_end: the date the figure's period ends, as YYYY-MM-DD. For shares_outstanding, the "as of" date given on the cover page; for revenue_prior, the end of the year before.
- fiscal_year: the fiscal year the report covers, as the company numbers it (e.g. 2024). It is the same for every figure, revenue_prior included.

If a figure is not in the text, set all four of its parts to null.

Check before you answer: each dollar value must be the printed figure times its scale (31,904 in millions is 31904000000, and 11,946.0 in millions is 11946000000), never the printed figure itself.

Answer with a single JSON object and nothing else, with all twelve keys in this order, e.g. (the numbers are made up):
{"revenue": {"value": 31904000000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, "revenue_prior": {"value": 29577000000, "scale": "millions", "period_end": "2023-06-30", "fiscal_year": 2024}, "net_income": {"value": -1250500000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, "eps_basic": {"value": -2.33, "scale": "units", "period_end": "2024-06-30", "fiscal_year": 2024}, "eps_diluted": {"value": -2.31, "scale": "units", "period_end": "2024-06-30", "fiscal_year": 2024}, "total_assets": {...}, "cash": {...}, "equity": {...}, "shares_outstanding": {"value": 541203887, "scale": "units", "period_end": "2024-08-09", "fiscal_year": 2024}, "operating_cash_flow": {"value": 4210000000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, "capex": {"value": 1385000000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}, "fcf": {"value": 2825000000, "scale": "millions", "period_end": "2024-06-30", "fiscal_year": 2024}}"""

SYSTEMS = {"v1": SYSTEM, "v2": SYSTEM_V2}


def system(task: str = "v1") -> str:
    field_names(task)  # raises for an unknown task
    return SYSTEMS[task]


def prompt_version(task: str = "v1") -> str:
    field_names(task)
    return PROMPT_VERSIONS[task]


def messages(filing_text: str, task: str = "v1") -> list[dict]:
    return [{"role": "system", "content": system(task)}, {"role": "user", "content": filing_text}]
