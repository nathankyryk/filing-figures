"""Which companies and which filings.

Companies: the current S&P 500 (Wikipedia's constituents table, saved with its download date),
minus Financials: a bank's income statement has no single revenue line, so its "revenue" has no
clean answer. One row per CIK (GOOGL/GOOG and the like share one), then a seeded sample.

Filings: each company's original 10-Ks (no 10-K/A amendments or 10-KT transition reports) that
use inline XBRL, for fiscal years 2019-2024. The fiscal year is the filer's own
(DocumentFiscalYearFocus, carried as `fy` in companyfacts), so a retailer's year ending in
January keeps the name the company gives it.
"""

from __future__ import annotations

import io
import random
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from filing_figures.labels import fiscal_years

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
EXCLUDED_SECTORS = ("Financials",)
FIRST_YEAR, LAST_YEAR = 2019, 2024


def fetch_sp500(user_agent: str) -> pd.DataFrame:
    html = httpx.get(SP500_URL, headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True).text
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    df = table.rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"}
    )[["ticker", "name", "sector", "sub_industry", "CIK"]].rename(columns={"CIK": "cik"})
    df["cik"] = df["cik"].astype(int)
    df["retrieved"] = date.today().isoformat()
    return df


def choose_companies(sp500: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = sp500[~sp500["sector"].isin(EXCLUDED_SECTORS)]
    df = df.sort_values(["cik", "ticker"]).drop_duplicates("cik").reset_index(drop=True)
    if n >= len(df):
        return df
    picked = sorted(random.Random(seed).sample(range(len(df)), n))
    return df.iloc[picked].reset_index(drop=True)


def list_10ks(submissions: dict, companyfacts: dict) -> list[dict]:
    """The company's original inline-XBRL 10-Ks for fiscal years 2019-2024, one per year."""
    rec = submissions["filings"]["recent"]
    fy_of = fiscal_years(companyfacts)
    by_year: dict[int, dict] = {}
    for i, form in enumerate(rec["form"]):
        if form != "10-K" or not rec["isInlineXBRL"][i]:
            continue
        accession = rec["accessionNumber"][i]
        fy = fy_of.get(accession)
        if fy is None or not FIRST_YEAR <= fy <= LAST_YEAR:
            continue
        filing = {
            "accession": accession,
            "fiscal_year": fy,
            "filing_date": rec["filingDate"][i],
            "report_date": rec["reportDate"][i],
            "primary_document": rec["primaryDocument"][i],
        }
        # Two 10-Ks claiming one fiscal year (rare: a mis-set fiscal year focus): keep the earlier one.
        if fy not in by_year or filing["filing_date"] < by_year[fy]["filing_date"]:
            by_year[fy] = filing
    return [by_year[y] for y in sorted(by_year)]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
