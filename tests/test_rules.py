from filing_figures import rules


def _cover(line: str) -> str:
    return f"## Cover page\n{line}\n\n## Income statement\nRevenue | 1 | 2\n"


def test_fiscal_year_end_variants():
    for line, want in [
        ("For the fiscal year ended December 31, 2024", "2024-12-31"),
        ("Fiscal Year Ended: September 30, 2019", "2019-09-30"),
        ("For the fiscal year ended | September 28, 2019", "2019-09-28"),
        ("FOR THE YEAR ENDED ON JANUARY 31, 2021", "2021-01-31"),
    ]:
        out = rules.extract(_cover(line))
        assert out["revenue"]["period_end"] == want, line


def test_statement_reads_current_column_and_scale():
    text = (
        "## Cover page\nfor the fiscal year ended December 31, 2024\n"
        "123,456,789 shares of common stock outstanding as of February 1, 2025.\n\n"
        "## Income statement\n(In millions, except per share amounts)\nYear | 2024 | 2023\n"
        "Total revenues | 1,200 | 1,100\nNet income | (50) | 40\nDiluted | (0.25) | 0.20\n"
    )
    out = rules.extract(text)
    assert out["revenue"]["value"] == 1_200e6 and out["revenue"]["scale"] == "millions"
    assert out["net_income"]["value"] == -50e6
    assert out["eps_diluted"]["value"] == -0.25
    assert out["shares_outstanding"]["value"] == 123_456_789
    assert out["shares_outstanding"]["period_end"] == "2025-02-01"
