"""The out-of-distribution set: reading the S&P 400 table, splitting by company, and keeping it out of v1."""

from collections import Counter

import pandas as pd

from filing_figures import cli
from filing_figures.dataset import assign_ood_splits
from filing_figures.universe import parse_sp400, ticker_ciks

TABLE = """<table class="wikitable" id="constituents"><tbody>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
<th>Headquarters Location</th><th>SEC filings</th></tr>
<tr><td><a href="https://www.nyse.com/quote/XNYS:AA">AA</a></td><td><a href="/wiki/Alcoa">Alcoa</a></td>
<td>Materials</td><td>Aluminum</td><td>Pittsburgh</td>
<td><a href="https://www.sec.gov/cgi-bin/browse-edgar?CIK=AA&amp;action=getcompany">reports</a></td></tr>
<tr><td><a href="https://www.nasdaq.com/aal">AAL</a></td><td>American Airlines Group</td>
<td>Industrials</td><td>Passenger Airlines</td><td>Fort Worth</td>
<td><a href="https://www.sec.gov/cgi-bin/browse-edgar?CIK=6201&amp;action=getcompany">reports</a></td></tr>
<tr><td>MOG.A</td><td>Moog</td><td>Industrials</td><td>Aerospace &amp; Defense</td><td>Elma</td>
<td><a href="https://www.sec.gov/cgi-bin/browse-edgar?CIK=MOG.A&amp;action=getcompany">reports</a></td></tr>
<tr><td>ZZZZ</td><td>Unknown Co</td><td>Energy</td><td>Oil</td><td>Nowhere</td><td>reports</td></tr>
</tbody></table>"""

SEC_TICKERS = {
    "0": {"cik_str": 1675149, "ticker": "AA", "title": "Alcoa Corp"},
    "1": {"cik_str": 6201, "ticker": "AAL", "title": "American Airlines Group Inc."},
    "2": {"cik_str": 67887, "ticker": "MOG-A", "title": "MOOG INC."},
}


def test_sp400_ciks_come_from_the_link_or_the_ticker_list():
    df = parse_sp400(TABLE, ticker_ciks(SEC_TICKERS)).set_index("ticker")
    assert df.loc["AAL", "cik"] == 6201 and df.loc["AAL", "cik_source"] == "wikipedia"
    assert df.loc["AA", "cik"] == 1675149 and df.loc["AA", "cik_source"] == "sec_tickers"
    assert df.loc["MOG.A", "cik"] == 67887  # Wikipedia's "." is the SEC's "-"
    assert pd.isna(df.loc["ZZZZ", "cik"]) and pd.isna(df.loc["ZZZZ", "cik_source"])
    assert df.loc["AAL", "name"] == "American Airlines Group" and df.loc["AAL", "sector"] == "Industrials"


def companies(n_per_sector=10, sectors=("Energy", "Health Care", "Industrials")):
    return [{"cik": 1000 * i + j, "sector": s} for i, s in enumerate(sectors, 1) for j in range(n_per_sector)]


def test_ood_splits_are_by_company_stratified_and_seeded():
    cs = companies()
    splits = assign_ood_splits(cs)
    assert set(splits) == {c["cik"] for c in cs}
    assert set(splits.values()) == {"ood-valid", "ood-test"}
    per_sector = Counter((c["sector"], splits[c["cik"]]) for c in cs)
    for s in ("Energy", "Health Care", "Industrials"):
        assert per_sector[s, "ood-valid"] == 4 and per_sector[s, "ood-test"] == 6
    assert assign_ood_splits(list(reversed(cs))) == splits  # input order doesn't matter


def write_universe(root, monkeypatch):
    uni, ood = root / "universe", root / "universe" / "ood"
    ood.mkdir(parents=True)
    row = {"sector": "Energy", "fiscal_year": 2024, "filing_date": "2025-02-01", "report_date": "2024-12-31"}
    pd.DataFrame(
        [{"cik": 1, "ticker": "OLD", "accession": "0000000001-25-000001", "primary_document": "a.htm", **row}]
    ).to_csv(uni / "filings.csv", index=False)
    pd.DataFrame([{"cik": 1, "ticker": "OLD", "sector": "Energy", "split": "train"}]).to_csv(
        uni / "splits.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "cik": 2,
                "ticker": "MID",
                "accession": "0000000002-26-000001",
                "primary_document": "b.htm",
                **row,
                "fiscal_year": 2025,
            }
        ]
    ).to_csv(ood / "filings.csv", index=False)
    pd.DataFrame([{"cik": 2, "ticker": "MID", "sector": "Energy", "split": "ood-test"}]).to_csv(
        ood / "splits.csv", index=False
    )
    monkeypatch.setattr(cli, "DATA", root)
    monkeypatch.setattr(cli, "OOD", ood)


def test_v1_never_sees_the_ood_filings(tmp_path, monkeypatch):
    write_universe(tmp_path, monkeypatch)
    assert cli._filings(None, None).ticker.tolist() == ["OLD"]
    both = cli._filings(None, None, ood=True)
    assert dict(zip(both.ticker, both.split, strict=True)) == {"OLD": "train", "MID": "ood-test"}
    assert both.accession.tolist()[1] == "0000000002-26-000001"  # kept as text
    assert cli._filings(None, None, split="ood-test", ood=True).ticker.tolist() == ["MID"]
