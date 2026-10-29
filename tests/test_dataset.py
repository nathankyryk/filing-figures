"""One dataset record from companyfacts + the hand-made 10-K in test_inputs."""

from test_inputs import DOC
from test_labels import ACC, END, base, row

from filing_figures.dataset import build_record

FILING = {
    "cik": 1,
    "ticker": "EXM",
    "sector": "Information Technology",
    "accession": ACC,
    "fiscal_year": 2024,
    "report_date": END,
    "filing_date": "2024-11-01",
    "primary_document": "exm-20240928.htm",
}
CONTRACT = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


def facts_matching_doc():
    cf = base()
    us = cf["facts"]["us-gaap"]
    # The fixture's statement starts its fiscal year on 2023-10-01 for every flow item
    us["NetIncomeLoss"]["units"]["USD"] = [row(93_736_000_000, END, "2023-10-01")]
    us["Revenues"]["units"]["USD"][0]["val"] = 391_035_000_000
    return cf


def test_record_scores_what_the_statement_shows():
    rec = build_record(FILING, facts_matching_doc(), DOC)
    assert rec["usable"] and rec["id"] == "EXM-2024" and rec["fiscal_year"] == 2024
    g = rec["gold"]
    assert g["revenue"] == {
        "value": 391_035_000_000,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "us-gaap:Revenues",
        "displayed": "391,035",
    }
    assert g["shares_outstanding"]["period_end"] == "2024-10-18" and g["shares_outstanding"]["scale"] == "units"
    assert "777,777" not in rec["input"]


def test_later_tag_used_when_the_statement_shows_it():
    """`Revenues` (first in the fallback list) is a different total, tagged only outside the
    statement; the statement shows contract revenue: that is the figure to score."""
    cf = facts_matching_doc()
    cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] = 396_035_000_000
    cf["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [row(391_035_000_000, END, "2023-10-01")]}
    }
    doc = DOC.replace(b'name="us-gaap:Revenues" contextRef="FY24"', f'name="{CONTRACT}" contextRef="FY24"'.encode())
    rec = build_record(FILING, cf, doc)
    assert rec["gold"]["revenue"]["tag"] == CONTRACT
    assert rec["gold"]["revenue"]["value"] == 391_035_000_000
    assert rec["flags"]["revenue"] == ["multi_tag", "tag_from_statement"]


def test_field_outside_the_input_is_unscored():
    doc = DOC.replace(b'name="us-gaap:Assets"', b'name="us-gaap:AssetsNoncurrent"')  # total assets not tagged there
    rec = build_record(FILING, facts_matching_doc(), doc)
    assert rec["gold"]["total_assets"] is None
    assert rec["flags"]["total_assets"] == ["not_in_input"]
    assert rec["usable"]  # one unscored field still leaves five
