from filing_figures.labels import FIELDS, accession_facts, current_period_end, filing_fiscal_year, label_filing

ACC = "0000000001-24-000010"
OTHER = "0000000001-24-000099"  # e.g. a later 10-K/A for the same year
END, PRIOR_END = "2024-09-28", "2023-09-30"


def row(val, end, start=None, accn=ACC, fy=2024, form="10-K"):
    r = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": "FY", "form": form, "filed": "2024-11-01"}
    if start:
        r["start"] = start
    return r


def facts(**concepts):
    """{"us-gaap:Revenues": {"USD": [rows]}} -> the companyfacts layout."""
    out: dict = {"facts": {}}
    for tag, units in concepts.items():
        taxonomy, concept = tag.split("__")
        out["facts"].setdefault(taxonomy.replace("_", "-"), {})[concept] = {"units": units}
    return out


def base(**extra):
    return facts(
        us_gaap__Revenues={
            "USD": [
                row(391_035_000_000, END, "2023-10-01"),  # current year: 364 days (a 52-week year)
                row(383_285_000_000, PRIOR_END, "2022-09-25"),  # prior year, same accession and fy
                row(94_930_000_000, END, "2024-06-30"),  # the fourth quarter: too short
                row(1.0, END, "2023-10-01", accn=OTHER),  # another filing's fact for the same period
            ]
        },
        us_gaap__NetIncomeLoss={"USD": [row(93_736_000_000, END, "2023-09-24")]},  # 371 days: a 53-week year
        us_gaap__EarningsPerShareDiluted={
            "USD/shares": [row(6.08, END, "2023-10-01"), row(6.13, PRIOR_END, "2022-09-25")]
        },
        us_gaap__Assets={"USD": [row(364_980_000_000, END), row(352_583_000_000, PRIOR_END)]},
        dei__EntityCommonStockSharesOutstanding={"shares": [row(15_115_823_000, "2024-10-18")]},
        us_gaap__NetCashProvidedByUsedInOperatingActivities={"USD": [row(118_254_000_000, END, "2023-10-01")]},
        **extra,
    )


def test_current_period_only():
    out = label_filing(base(), ACC, END)
    values = {k: v["value"] for k, v in out["labels"].items()}
    assert values == {
        "revenue": 391_035_000_000,  # not the prior year, the quarter, or the other filing's value
        "net_income": 93_736_000_000,
        "eps_diluted": 6.08,
        "total_assets": 364_980_000_000,
        "shares_outstanding": 15_115_823_000,
        "operating_cash_flow": 118_254_000_000,
    }
    assert out["labels"]["shares_outstanding"]["end"] == "2024-10-18"  # the cover date, not the year end
    assert out["fiscal_year"] == 2024 and out["period_end"] == END
    assert all(not f for f in out["flags"].values())


def test_prior_year_kept_as_context():
    ctx = label_filing(base(), ACC, END)["context_facts"]
    prior = [f for f in ctx if f["tag"] == "us-gaap:Revenues" and f["end"] == PRIOR_END]
    assert [f["val"] for f in prior] == [383_285_000_000]
    assert all(f["val"] != 1.0 for f in ctx)  # nothing from the other accession


def test_tag_fallback_order_and_multi_tag():
    cf = base()
    del cf["facts"]["us-gaap"]["Revenues"]
    cf["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [row(390_000_000_000, END, "2023-10-01")]}
    }
    cf["facts"]["us-gaap"]["SalesRevenueNet"] = {"units": {"USD": [row(1_000, END, "2023-10-01")]}}
    out = label_filing(cf, ACC, END)
    rev = out["labels"]["revenue"]
    assert rev["tag"] == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert rev["value"] == 390_000_000_000
    assert out["flags"]["revenue"] == ["multi_tag"]


def test_same_value_under_two_tags_is_not_flagged():
    cf = base()
    cf["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [row(391_035_000_000, END, "2023-10-01")]}
    }
    assert label_filing(cf, ACC, END)["flags"]["revenue"] == []


def test_several_share_classes_are_unscored():
    cf = base()
    cf["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(row(300_000_000, "2024-10-18"))
    out = label_filing(cf, ACC, END)
    assert out["labels"]["shares_outstanding"] is None
    assert out["flags"]["shares_outstanding"] == ["multi_class"]


def test_conflicting_values_go_to_the_hand_check():
    cf = base()
    cf["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(row(364_000_000_000, END))
    out = label_filing(cf, ACC, END)
    assert out["labels"]["total_assets"] is None
    assert out["flags"]["total_assets"] == ["conflict"]


def test_missing_field():
    cf = base()
    del cf["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]
    out = label_filing(cf, ACC, END)
    assert out["labels"]["operating_cash_flow"] is None
    assert out["flags"]["operating_cash_flow"] == ["missing"]


def test_period_end_from_facts_when_report_date_is_off():
    cf = base()
    gold = next(f for f in FIELDS if f.name == "revenue")
    period, flags = current_period_end(accession_facts(cf, ACC, gold.tags), "2024-09-30")
    assert period == END and flags == ["period_from_facts"]


def test_fiscal_year_is_the_filings():
    cf = base()
    last_year = "0000000001-23-000050"
    cf["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(row(5, "2023-09-30", accn=last_year, fy=2023))
    assert filing_fiscal_year(cf, ACC) == 2024
    assert filing_fiscal_year(cf, last_year) == 2023
