"""The input builder on a small hand-made inline-XBRL 10-K."""

from filing_figures import inputs, ixbrl
from filing_figures.dataset import assign_splits

HEAD = (
    '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
    'xmlns:us-gaap="http://fasb.org/us-gaap/2024" xmlns:dei="http://xbrl.sec.gov/dei/2024"><body>'
)


def ctx(cid, start=None, end=None, instant=None, dims=False):
    period = (
        f"<xbrli:instant>{instant}</xbrli:instant>"
        if instant
        else f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>"
    )
    seg = (
        '<xbrli:segment><xbrldi:explicitMember dimension="x">y</xbrldi:explicitMember></xbrli:segment>' if dims else ""
    )
    return (
        f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier>{seg}'
        f"</xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>"
    )


def nf(name, cid, text, scale=6, sign=None):
    s = f' sign="{sign}"' if sign else ""
    return f'<ix:nonFraction name="{name}" contextRef="{cid}" unitRef="u" scale="{scale}"{s}>{text}</ix:nonFraction>'


def table(*rows):
    """Rows of cells; a cell may already be HTML."""
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


HEADER = (
    '<div style="display:none"><ix:header><ix:hidden>'
    '<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="FY24">2024</ix:nonNumeric>'
    '<ix:nonNumeric name="dei:DocumentType" contextRef="FY24">10-K</ix:nonNumeric>'
    + nf("us-gaap:Revenues", "FY24", "777,777")  # a hidden fact: must never reach the model
    + "</ix:hidden><ix:resources>"
    + ctx("FY24", "2023-10-01", "2024-09-28")
    + ctx("FY23", "2022-10-02", "2023-09-30")
    + ctx("I24", instant="2024-09-28")
    + ctx("COVER", instant="2024-10-18")
    + ctx("SEG24", "2023-10-01", "2024-09-28", dims=True)
    + "</ix:resources></ix:header></div>"
)

DOC = (
    HEAD
    + HEADER
    + "<div>FORM 10-K</div><div>For the fiscal year ended September 28, 2024</div>"
    + "<div>Table of Contents</div>"  # a page-top link on the cover's second page: not the end of the cover
    + "<div>As of October 18, 2024, there were "
    + nf("dei:EntityCommonStockSharesOutstanding", "COVER", "15,115,823,000", scale=0)
    + " shares outstanding.</div>"
    + "<div>TABLE OF CONTENTS</div>"
    + table(
        ["Item 1A.", "Risk Factors", "5"],
        ["Item 7.", "MD&amp;A", "20"],
        ["Item 8.", "Financial Statements", "28"],
        ["", "Consolidated Statements of Cash Flows", "32"],
    )
    + "<div>Item 7. Management's Discussion and Analysis</div>"
    # An MD&A table titled like the statement: the real one comes after Item 8
    + "<div>Consolidated Statements of Cash Flows</div>"
    + table(
        ["", "2024", "2023"],
        ["Net cash provided by operating activities", "$", "99,999", "$", "88,888"],
        ["Net cash used in investing activities", "(", "1,000", ")", "(", "900", ")"],
        ["Net cash used in financing activities", "(", "2,000", ")", "(", "1,800", ")"],
    )
    + "<div>Item 8. Financial Statements and Supplementary Data</div>"
    # A statement of comprehensive income alone (no revenue line) is not the income statement
    + "<div>CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME</div>"
    + table(
        ["", "2024", "2023"],
        ["Net income", "93,736", "96,995"],
        ["Foreign currency translation", "(", "100", ")"],
        ["Comprehensive income", "93,636", "96,900"],
    )
    + "<div>Example Corp.</div><div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>"
    + "<div>(In millions, except per-share amounts)</div>"
    + table(
        ["", "2024", "2023"],
        [
            "Total net sales",
            "$",
            nf("us-gaap:Revenues", "FY24", "391,035"),
            "$",
            nf("us-gaap:Revenues", "FY23", "383,285"),
        ],
        ["Segment A sales", nf("us-gaap:Revenues", "SEG24", "100,000")],
        ["Other income/(expense), net", "(", nf("us-gaap:NonoperatingIncomeExpense", "FY24", "565", sign="-"), ")"],
        ["Net income", "$", nf("us-gaap:NetIncomeLoss", "FY24", "93,736"), "$", "96,995"],
        ["Diluted", "$", nf("us-gaap:EarningsPerShareDiluted", "FY24", "6.08", scale=0), "$", "6.13"],
    )
    + "<div>CONSOLIDATED BALANCE SHEETS</div>"
    + table(
        ["", "2024", "2023"],
        ["Cash and cash equivalents", "$", "29,943", "$", "29,965"],
        ["Total current assets", "152,987", "143,566"],
        ["Total assets", "$", nf("us-gaap:Assets", "I24", "364,980"), "$", "352,583"],
        ["Total liabilities", "308,030", "290,437"],
    )
    + table(["Example Corp. | 2024 Form 10-K | 31"])  # a page-footer table
    + table(["Consolidated Statements of Cash Flows"])  # a heading table (as Intel and Verizon do)
    + table(
        ["", "2024", "2023"],
        ["Net income", "93,736", "96,995"],
        ["Depreciation", "11,445", "11,519"],
        [
            "Cash generated by operating activities",
            nf("us-gaap:NetCashProvidedByUsedInOperatingActivities", "FY24", "118,254"),
            "110,543",
        ],
        ["Cash used in investing activities", "2,935", "3,705"],
    )
    + "</body></html>"
).encode()

GOLD = {
    "revenue": {"value": 391_035e6, "start": "2023-10-01", "end": "2024-09-28", "tag": "us-gaap:Revenues"},
    "net_income": {"value": 93_736e6, "start": "2023-10-01", "end": "2024-09-28", "tag": "us-gaap:NetIncomeLoss"},
    "eps_diluted": {
        "value": 6.08,
        "start": "2023-10-01",
        "end": "2024-09-28",
        "tag": "us-gaap:EarningsPerShareDiluted",
    },
    "total_assets": {"value": 364_980e6, "start": None, "end": "2024-09-28", "tag": "us-gaap:Assets"},
    "shares_outstanding": {
        "value": 15_115_823_000,
        "start": None,
        "end": "2024-10-18",
        "tag": "dei:EntityCommonStockSharesOutstanding",
    },
    "operating_cash_flow": {
        "value": 118_254e6,
        "start": "2023-10-01",
        "end": "2024-09-28",
        "tag": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    },
}


def test_hidden_header_never_reaches_the_model():
    built = inputs.build(DOC)
    assert "777,777" not in built.text
    assert built.dei["DocumentFiscalYearFocus"] == "2024"  # still read, before the removal
    assert built.dei["DocumentType"] == "10-K"


def test_statements_after_item8_not_the_mdna_table():
    built = inputs.build(DOC)
    assert sorted(built.selection.statements) == ["balance_sheet", "cash_flow_statement", "income_statement"]
    cash = built.text.split("## Cash flow statement")[1]
    assert "118,254" in cash and "99,999" not in built.text


def test_heading_table_titles_the_statement():
    built = inputs.build(DOC)
    assert built.selection.how["cash_flow_statement"] == "title"
    caption, _ = built.selection.statements["cash_flow_statement"]
    assert [b.text for b in caption] == ["Consolidated Statements of Cash Flows"]


def test_comprehensive_income_statement_is_skipped():
    built = inputs.build(DOC)
    caption, _ = built.selection.statements["income_statement"]
    assert any("STATEMENTS OF OPERATIONS" in b.text for b in caption)
    assert "Foreign currency" not in built.text


def test_table_text_keeps_signs_and_dollars():
    text = inputs.build(DOC).text
    assert "Total net sales | $ 391,035 | $ 383,285" in text
    assert "Other income/(expense), net | (565)" in text
    assert "(In millions, except per-share amounts)" in text
    assert (
        "Example Corp." not in text.split("## Income statement")[1].split("## Balance")[0]
    )  # caption starts at the title


def test_untitled_statement_found_by_its_key_lines():
    doc = DOC.replace(b"<div>CONSOLIDATED BALANCE SHEETS</div>", b"<div>(In millions)</div>")
    built = inputs.build(doc)
    assert built.selection.how["balance_sheet"] == "structure"
    assert "364,980" in built.text.split("## Balance sheet")[1]


def test_reference_to_a_statement_is_not_a_title():
    assert not inputs.is_title("Analysis of the Consolidated Statements of Income", inputs.TITLES["income_statement"])
    assert not inputs.is_title("Condensed Balance Sheets (Parent Company Only)", inputs.TITLES["balance_sheet"])
    assert inputs.is_title("Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS", inputs.TITLES["income_statement"])
    assert inputs.is_title("STATEMENTS OF CONSOLIDATED INCOME", inputs.TITLES["income_statement"])


def test_coverage_finds_each_field_with_its_scale():
    built = inputs.build(DOC)
    cov = inputs.coverage(built.selection, built.root, built.contexts, GOLD)
    assert all(c["in_input"] and not c["mismatch"] for c in cov.values())
    assert {k: c["scale"] for k, c in cov.items()} == {
        "revenue": "millions",
        "net_income": "millions",
        "eps_diluted": "units",
        "total_assets": "millions",
        "shares_outstanding": "units",
        "operating_cash_flow": "millions",
    }
    assert cov["revenue"]["text"] == "391,035"  # the no-dimension fact, not the segment's


def test_coverage_flags_a_value_mismatch_and_a_missing_fact():
    built = inputs.build(DOC)
    gold = {
        "revenue": {**GOLD["revenue"], "value": 390_000e6},
        "net_income": {**GOLD["net_income"], "end": "2022-12-31"},
    }
    cov = inputs.coverage(built.selection, built.root, built.contexts, gold)
    assert cov["revenue"]["mismatch"]
    assert cov["net_income"] == {"in_input": False, "scale": None, "text": None, "mismatch": False}


def test_facts_apply_scale_and_sign():
    root = ixbrl.parse(DOC)
    other = [f for f in ixbrl.facts(root) if f.name == "us-gaap:NonoperatingIncomeExpense"]
    assert [f.value for f in other] == [-565e6]


def test_displayed_number_formats():
    assert ixbrl.displayed_number("1,234.5", "ixt:num-dot-decimal") == 1234.5
    assert ixbrl.displayed_number("1.234,5", "ixt:num-comma-decimal") == 1234.5
    assert ixbrl.displayed_number("—", "ixt:fixed-zero") == 0.0
    assert ixbrl.displayed_number("None", "ixt-sec:numwordsen") == 0.0
    assert ixbrl.displayed_number("", None) is None


def test_zero_width_spaces_are_dropped():
    doc = DOC.replace(b"Total net sales", "Total\u200b net sales\u200b".encode())
    assert "Total net sales | $ 391,035" in inputs.build(doc).text


def test_splits_are_by_company_and_seeded():
    companies = [{"cik": i, "sector": "A" if i % 2 else "B"} for i in range(100)]
    a, b = assign_splits(companies), assign_splits(companies)
    assert a == b
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "valid", "test")}
    assert counts == {"train": 68, "valid": 16, "test": 16}  # 50 per sector: round(7.5) = 8 valid and 8 test


def test_cover_runs_past_page_top_links_and_stops_at_the_contents():
    sel = inputs.build(DOC).selection
    cover = " ".join(b.text for b in sel.cover)
    assert "15,115,823,000 shares outstanding" in cover
    assert "Risk Factors" not in cover and "Item 7. Management" not in cover


def test_cover_paragraph_wrapping_the_rest_of_the_document_is_cut_at_part_one():
    """Some filings wrap everything after the cover's last paragraph in one inline element, so
    that paragraph's own text is the whole document (270K characters in VTR-2019)."""
    wrap = (
        "<div>DOCUMENTS INCORPORATED BY REFERENCE <span>Portions of the proxy statement are incorporated "
        "into Part III. <div>PART I</div><div>Item 1. Business " + "We make things. " * 1000 + "</div></span></div>"
    )
    doc = DOC.replace(b"<div>TABLE OF CONTENTS</div>", wrap.encode() + b"<div>TABLE OF CONTENTS</div>")
    built = inputs.build(doc)
    cover = " ".join(b.text for b in built.selection.cover)
    assert "15,115,823,000 shares outstanding" in cover
    assert "We make things" not in cover and len(cover) < inputs.COVER_CHARS
    assert "## Income statement" in built.text  # the statements are still found


def test_date_near_prefers_the_counts_own_sentence():
    line = (
        "The aggregate market value held by non-affiliates on March 31, 2022 was approximately $32.5 billion. "
        "114,844,152 shares of Common Stock, par value $1 per share, were outstanding on October 31, 2022."
    )
    assert inputs.date_near(line, "114,844,152") == "2022-10-31"
    row = "Common Stock, $0.01 par value | 441,823,811 | September 28, 2021"
    assert inputs.date_near(row, "441,823,811") == "2021-09-28"
    assert inputs.date_near("As of the date below\n441,823,811 shares", "441,823,811") is None
