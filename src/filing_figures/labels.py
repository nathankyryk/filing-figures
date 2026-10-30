"""Gold labels for the six fields, from the company's own XBRL facts (SEC companyfacts).

A 10-K reports the current year next to one or two prior years, and companyfacts lists all of
them under the filing's accession number with the *filing's* fiscal year (`fy`). So a value
belongs to a filing only when its accession matches AND its period is the filing's current one:
  - instant facts (total assets): `end` is the period end;
  - year-long facts (revenue, net income, EPS, operating cash flow): `end` is the period end and
    the period lasts 350-380 days (52/53-week years run 364 or 371 days; quarters drop out);
  - shares outstanding come from the cover page, dated a few weeks after the year end.
Each field tries its XBRL tags in order and the first tag with a match wins (revenue alone has six).

Verified on Apple's FY2024 10-K (accession 0000320193-24-000123, 2026-10-28): its three revenue
values all carry fy=2024; only the one ending 2024-09-28 (363 days) is the FY2024 figure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

MIN_YEAR_DAYS, MAX_YEAR_DAYS = 350, 380
MAX_COVER_DAYS = 200  # the cover's shares date falls between the year end and the filing date


@dataclass(frozen=True)
class Field:
    name: str
    unit: str
    kind: str  # "year": a year-long period; "instant": at the period end; "cover": the cover-page date
    tags: tuple[str, ...]  # "taxonomy:Concept", tried in order


FIELDS: tuple[Field, ...] = (
    Field(
        "revenue",
        "USD",
        "year",
        (
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "us-gaap:SalesRevenueGoodsNet",
            "us-gaap:SalesRevenueServicesNet",
        ),
    ),
    # Net income attributable to the company. Some filers (e.g. American Tower) tag that line only
    # as "available to common stockholders"; ProfitLoss (including noncontrolling interests) is last.
    Field(
        "net_income",
        "USD",
        "year",
        ("us-gaap:NetIncomeLoss", "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic", "us-gaap:ProfitLoss"),
    ),
    Field(
        "eps_diluted",
        "USD/shares",
        "year",
        ("us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasicAndDiluted"),
    ),
    Field("total_assets", "USD", "instant", ("us-gaap:Assets",)),
    Field("shares_outstanding", "shares", "cover", ("dei:EntityCommonStockSharesOutstanding",)),
    Field(
        "operating_cash_flow",
        "USD",
        "year",
        (
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)

# Look-alike facts kept with each filing (every period), so the scorer can name an error:
# a prediction equal to one of these is a "wrong line item", one equal to a prior-year
# value of the right tag is a "wrong period".
LOOKALIKES: tuple[str, ...] = (
    "us-gaap:EarningsPerShareBasic",
    "us-gaap:NetIncomeLossAttributableToNoncontrollingInterest",
    "us-gaap:IncomeLossFromContinuingOperations",
    "us-gaap:ComprehensiveIncomeNetOfTax",
    "us-gaap:GrossProfit",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:CostOfRevenue",
    "us-gaap:AssetsCurrent",
    "us-gaap:Liabilities",
    "us-gaap:StockholdersEquity",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
    "us-gaap:CommonStockSharesOutstanding",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities",
    "dei:EntityPublicFloat",
)


def _days(start: str | None, end: str) -> int | None:
    if not start:
        return None
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def accession_facts(companyfacts: dict, accession: str, tags: tuple[str, ...]) -> list[dict]:
    """Every fact of these tags filed under one accession number (all periods), deduplicated."""
    out, seen = [], set()
    for tag in tags:
        taxonomy, concept = tag.split(":")
        units = companyfacts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
        for unit, rows in units.items():
            for r in rows:
                if r.get("accn") != accession:
                    continue
                key = (tag, unit, r.get("start"), r["end"], r["val"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"tag": tag, "unit": unit, "start": r.get("start"), "end": r["end"], "val": r["val"]})
    return out


def fiscal_years(companyfacts: dict) -> dict[str, int]:
    """Each accession's fiscal year (`fy`, from the filer's DocumentFiscalYearFocus): its most common one."""
    years: dict[str, Counter[int]] = {}
    for taxonomy in companyfacts.get("facts", {}).values():
        for concept in taxonomy.values():
            for rows in concept.get("units", {}).values():
                for r in rows:
                    if r.get("accn") and r.get("fy"):
                        years.setdefault(r["accn"], Counter())[r["fy"]] += 1
    return {accn: c.most_common(1)[0][0] for accn, c in years.items()}


def filing_fiscal_year(companyfacts: dict, accession: str) -> int | None:
    return fiscal_years(companyfacts).get(accession)


def current_period_end(facts: list[dict], report_date: str | None) -> tuple[str | None, list[str]]:
    """The filing's fiscal year end: the submissions' reportDate when the facts use it, else the
    latest end among the filing's year-long facts."""
    year_ends = [f["end"] for f in facts if (_days(f["start"], f["end"]) or 0) >= MIN_YEAR_DAYS]
    instants = [f["end"] for f in facts if f["start"] is None]
    if report_date and (report_date in year_ends or report_date in instants):
        return report_date, []
    if year_ends:
        return max(year_ends), ["period_from_facts"]
    return report_date, ["period_unconfirmed"]


def _matches(field: Field, f: dict, period_end: str) -> bool:
    if f["unit"] != field.unit:
        return False
    if field.kind == "instant":
        return f["start"] is None and f["end"] == period_end
    if field.kind == "year":
        days = _days(f["start"], f["end"])
        return f["end"] == period_end and days is not None and MIN_YEAR_DAYS <= days <= MAX_YEAR_DAYS
    # cover: an instant on or after the year end, before the filing could have been made
    lag = (date.fromisoformat(f["end"]) - date.fromisoformat(period_end)).days
    return f["start"] is None and 0 <= lag <= MAX_COVER_DAYS


def label_field(field: Field, facts: list[dict], period_end: str) -> tuple[dict | None, list[str]]:
    """The gold value of one field, and flags for the hand-check.

    Flags: `conflict` (one tag, one period, two values), `multi_tag` (a later tag in the
    fallback list also matches with a different value), `multi_class` (the cover lists several
    share counts, i.e. several share classes: left unscored), `missing`.
    The later tags' matches are kept as `alternatives`: when the statement the model reads shows
    one of those rather than the first choice, that is the figure to score (see dataset.py).
    """
    flags: list[str] = []
    chosen: dict | None = None
    alternatives: list[dict] = []
    for tag in field.tags:
        hits = [f for f in facts if f["tag"] == tag and _matches(field, f, period_end)]
        if not hits:
            continue
        values = {(f["end"], f["val"]) for f in hits}
        if chosen is not None:
            if {f["val"] for f in hits} != {chosen["value"]} and "multi_tag" not in flags:
                flags.append("multi_tag")
            if len(values) == 1:
                hit = hits[0]
                alternatives.append({"value": hit["val"], "start": hit["start"], "end": hit["end"], "tag": tag})
            continue
        if len(values) > 1:
            flags.append("multi_class" if field.kind == "cover" else "conflict")
            return None, flags
        hit = hits[0]
        chosen = {"value": hit["val"], "start": hit["start"], "end": hit["end"], "tag": tag}
    if chosen is None:
        flags.append("missing")
        return None, flags
    chosen["alternatives"] = alternatives
    return chosen, flags


def label_filing(companyfacts: dict, accession: str, report_date: str | None) -> dict:
    """Gold labels for one 10-K, plus the look-alike facts the scorer uses to name errors."""
    gold_tags = tuple(t for f in FIELDS for t in f.tags)
    facts = accession_facts(companyfacts, accession, gold_tags)
    period_end, period_flags = current_period_end(facts, report_date)
    labels, flags = {}, {}
    for field in FIELDS:
        if period_end is None:
            labels[field.name], flags[field.name] = None, ["missing"]
            continue
        labels[field.name], flags[field.name] = label_field(field, facts, period_end)
    return {
        "accession": accession,
        "period_end": period_end,
        "fiscal_year": filing_fiscal_year(companyfacts, accession),
        "labels": labels,
        "flags": flags,
        "period_flags": period_flags,
        "context_facts": facts + accession_facts(companyfacts, accession, LOOKALIKES),
    }
