"""Task v2 (twelve fields): the new label kinds, the record builder, schema, prompt, scorer and
training answer. v1 must come out exactly as before."""

import copy
import hashlib
import json

import pytest
from test_dataset import FILING, facts_matching_doc
from test_evaluate import GOLD, record
from test_inputs import DOC, nf
from test_labels import ACC, END, PRIOR_END, base, row

from filing_figures import inputs, prompts, schema, sft
from filing_figures.dataset import apply_handcheck, build_record
from filing_figures.evaluate import close, parse_answer, score
from filing_figures.labels import FIELD_NAMES, field_names, label_filing, task_fields

V2 = (
    "revenue",
    "revenue_prior",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "total_assets",
    "cash",
    "equity",
    "shares_outstanding",
    "operating_cash_flow",
    "capex",
    "fcf",
)
PPE = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
CASH = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
CASH_RESTRICTED = "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
EQUITY_NCI = "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


def add(cf, tag, unit, *rows):
    taxonomy, concept = tag.split(":")
    cf["facts"].setdefault(taxonomy, {})[concept] = {"units": {unit: list(rows)}}
    return cf


def v2_facts():
    """The v1 fixture plus the new fields' facts."""
    cf = base()
    add(
        cf,
        "us-gaap:EarningsPerShareBasic",
        "USD/shares",
        row(6.11, END, "2023-10-01"),
        row(6.16, PRIOR_END, "2022-09-25"),
    )
    add(cf, CASH, "USD", row(29_943_000_000, END), row(29_965_000_000, PRIOR_END))
    add(cf, "us-gaap:StockholdersEquity", "USD", row(56_950_000_000, END))
    add(cf, PPE, "USD", row(9_447_000_000, END, "2023-10-01"), row(10_959_000_000, PRIOR_END, "2022-09-25"))
    return cf


# --- labels -------------------------------------------------------------------------------------


def test_task_field_lists():
    assert field_names("v1") == FIELD_NAMES == field_names()
    assert field_names("v2") == V2
    assert set(FIELD_NAMES) < set(V2)
    with pytest.raises(ValueError, match="unknown task"):
        task_fields("v3")


def test_v1_labels_are_unchanged_by_the_task_argument():
    cf = v2_facts()
    v1 = label_filing(cf, ACC, END)
    assert label_filing(cf, ACC, END, "v1") == v1
    assert set(v1["labels"]) == set(FIELD_NAMES)
    v2 = label_filing(cf, ACC, END, "v2")
    assert {k: v2["labels"][k] for k in FIELD_NAMES} == v1["labels"]
    assert {k: v2["flags"][k] for k in FIELD_NAMES} == v1["flags"]


def test_v2_labels():
    out = label_filing(v2_facts(), ACC, END, "v2")
    values = {k: (v["value"] if v else None) for k, v in out["labels"].items()}
    assert values == {
        "revenue": 391_035_000_000,
        "revenue_prior": 383_285_000_000,  # last year's, from this filing
        "net_income": 93_736_000_000,
        "eps_basic": 6.11,
        "eps_diluted": 6.08,
        "total_assets": 364_980_000_000,
        "cash": 29_943_000_000,
        "equity": 56_950_000_000,
        "shares_outstanding": 15_115_823_000,
        "operating_cash_flow": 118_254_000_000,
        "capex": 9_447_000_000,
        "fcf": 118_254_000_000 - 9_447_000_000,
    }
    assert out["labels"]["revenue_prior"]["end"] == PRIOR_END
    assert out["labels"]["fcf"]["tag"] == "derived"
    assert all(not f for f in out["flags"].values())


def test_prior_year_is_the_year_before_only():
    cf = v2_facts()
    revenues = cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    revenues.append(row(365_817_000_000, "2022-09-24", "2021-09-26"))  # two years back
    revenues.append(row(89_498_000_000, PRIOR_END, "2023-07-02"))  # last year's fourth quarter
    out = label_filing(cf, ACC, END, "v2")
    assert out["labels"]["revenue_prior"]["value"] == 383_285_000_000
    assert out["flags"]["revenue_prior"] == []
    # With no prior-year fact the field is missing, not filled with another period
    cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [r for r in revenues if r["end"] != PRIOR_END]
    out = label_filing(cf, ACC, END, "v2")
    assert out["labels"]["revenue_prior"] is None and out["flags"]["revenue_prior"] == ["missing"]


def test_fallback_tags_for_cash_equity_and_basic_eps():
    cf = v2_facts()
    us = cf["facts"]["us-gaap"]
    del us["CashAndCashEquivalentsAtCarryingValue"], us["StockholdersEquity"], us["EarningsPerShareBasic"]
    add(cf, CASH_RESTRICTED, "USD", row(30_737_000_000, END))
    add(cf, EQUITY_NCI, "USD", row(57_100_000_000, END))
    add(cf, "us-gaap:EarningsPerShareBasicAndDiluted", "USD/shares", row(6.08, END, "2023-10-01"))
    out = label_filing(cf, ACC, END, "v2")["labels"]
    assert (out["cash"]["tag"], out["cash"]["value"]) == (CASH_RESTRICTED, 30_737_000_000)
    assert (out["equity"]["tag"], out["equity"]["value"]) == (EQUITY_NCI, 57_100_000_000)
    assert out["eps_basic"]["tag"] == "us-gaap:EarningsPerShareBasicAndDiluted"


def test_first_tag_wins_and_the_restricted_total_is_an_alternative():
    cf = add(v2_facts(), CASH_RESTRICTED, "USD", row(30_737_000_000, END))
    out = label_filing(cf, ACC, END, "v2")
    assert out["labels"]["cash"]["tag"] == CASH and out["labels"]["cash"]["value"] == 29_943_000_000
    assert [a["tag"] for a in out["labels"]["cash"]["alternatives"]] == [CASH_RESTRICTED]
    assert out["flags"]["cash"] == ["multi_tag"]


def test_fcf_needs_both_parts():
    cf = v2_facts()
    del cf["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]
    out = label_filing(cf, ACC, END, "v2")
    assert out["labels"]["capex"] is None and out["labels"]["fcf"] is None
    assert out["flags"]["fcf"] == ["part_missing"]


def test_new_tags_cannot_move_the_period_end():
    """With no reportDate the period end is the latest year-long fact: v1's tags only, in every task."""
    cf = add(v2_facts(), PPE, "USD", row(9_447_000_000, END, "2023-10-01"), row(1, "2024-12-28", "2024-01-01"))
    assert label_filing(cf, ACC, None, "v2")["period_end"] == label_filing(cf, ACC, None)["period_end"] == END


def test_v2_look_alikes_name_the_new_near_misses():
    cf = add(v2_facts(), "us-gaap:PaymentsToAcquireBusinessesNetOfCashAcquired", "USD", row(5, END, "2023-10-01"))
    tags = lambda task: {f["tag"] for f in label_filing(cf, ACC, END, task)["context_facts"]}
    assert "us-gaap:PaymentsToAcquireBusinessesNetOfCashAcquired" in tags("v2")
    assert "us-gaap:PaymentsToAcquireBusinessesNetOfCashAcquired" not in tags("v1")


# --- records ------------------------------------------------------------------------------------

BASIC_ROW = "<tr><td>Basic</td><td>$</td><td>{}</td><td>$</td><td>6.16</td></tr>".format(
    nf("us-gaap:EarningsPerShareBasic", "FY24", "6.11", scale=0)
)
EQUITY_ROW = "<tr><td>Total shareholders' equity</td><td>{}</td><td>62,146</td></tr>".format(
    nf("us-gaap:StockholdersEquity", "I24", "56,950")
)
CAPEX_ROW = "<tr><td>Payments for property, plant and equipment</td><td>(</td><td>{}</td><td>)</td></tr>".format(
    nf(PPE, "FY24", "9,447")
)


def v2_doc(capex=True):
    """The v1 fixture 10-K with the new fields tagged on its statements."""
    doc = DOC.decode()
    for old, before in [
        ("<tr><td>Diluted</td>", BASIC_ROW),
        ("<tr><td>Total liabilities</td>", EQUITY_ROW),
        ("<tr><td>Cash used in investing activities</td>", CAPEX_ROW if capex else ""),
    ]:
        assert doc.count(old) == 1, old
        doc = doc.replace(old, before + old)
    assert doc.count("<td>29,943</td>") == 1
    doc = doc.replace("<td>29,943</td>", f"<td>{nf(CASH, 'I24', '29,943')}</td>")
    return doc.encode()


def v2_record_facts():
    cf = facts_matching_doc()
    for name, unit, rows in [
        ("EarningsPerShareBasic", "USD/shares", [row(6.11, END, "2023-10-01")]),
        ("CashAndCashEquivalentsAtCarryingValue", "USD", [row(29_943_000_000, END)]),
        ("StockholdersEquity", "USD", [row(56_950_000_000, END)]),
        ("PaymentsToAcquirePropertyPlantAndEquipment", "USD", [row(9_447_000_000, END, "2023-10-01")]),
    ]:
        add(cf, f"us-gaap:{name}", unit, *rows)
    cf["facts"]["us-gaap"]["Revenues"]["units"]["USD"][1]["start"] = "2022-10-02"  # the document's FY23 context
    return cf


def test_v2_record_scores_all_twelve():
    rec = build_record(FILING, v2_record_facts(), v2_doc(), "v2")
    g = rec["gold"]
    assert tuple(g) == V2 and rec["task"] == "v2" and rec["usable"]
    assert all(g[n] is not None for n in V2), {n: rec["flags"][n] for n in V2 if g[n] is None}
    assert g["revenue_prior"]["value"] == 383_285_000_000 and g["revenue_prior"]["period_end"] == PRIOR_END
    assert g["revenue_prior"]["displayed"] == "383,285"
    assert g["capex"]["value"] == 9_447_000_000 and g["capex"]["scale"] == "millions"  # positive, as tagged
    assert g["fcf"] == {
        "value": 118_254_000_000 - 9_447_000_000,
        "scale": "millions",
        "period_end": END,
        "fiscal_year": 2024,
        "tag": "derived",
        "displayed": None,
    }


def test_v1_record_has_its_original_shape():
    rec = build_record(FILING, v2_record_facts(), v2_doc())
    assert "task" not in rec and tuple(rec["gold"]) == FIELD_NAMES
    assert rec == build_record(FILING, v2_record_facts(), v2_doc(), "v1")


def test_fcf_unscored_when_capex_is_not_in_the_input():
    rec = build_record(FILING, v2_record_facts(), v2_doc(capex=False), "v2")
    assert rec["gold"]["capex"] is None and rec["flags"]["capex"] == ["not_in_input"]
    assert rec["gold"]["fcf"] is None and rec["flags"]["fcf"] == ["part_missing"]
    assert rec["usable"]  # two of twelve unscored


def test_handcheck_fix_of_a_part_recomputes_fcf():
    rec = build_record(FILING, v2_record_facts(), v2_doc(), "v2")
    fix = {"verdict": "fix", "value": "9500000000", "scale": "", "period_end": ""}
    fixed = apply_handcheck(rec, {(rec["id"], "capex"): fix})
    assert fixed["gold"]["fcf"]["value"] == 118_254_000_000 - 9_500_000_000
    assert fixed["flags"]["fcf"] == ["handcheck_parts"]
    dropped = apply_handcheck(rec, {(rec["id"], "capex"): {**fix, "verdict": "drop"}})
    assert dropped["gold"]["fcf"] is None
    # A check of fcf itself wins over its parts
    own = apply_handcheck(rec, {(rec["id"], "capex"): fix, (rec["id"], "fcf"): {**fix, "value": "1"}})
    assert own["gold"]["fcf"]["value"] == 1.0


# --- schema, prompt, scorer, training answer ----------------------------------------------------


def test_schema_per_task():
    assert schema.answer_model("v1") is schema.Answer and schema.JSON_SCHEMA == schema.json_schema("v1")
    s = schema.json_schema("v2")
    assert s["required"] == list(V2) and list(s["properties"]) == list(V2)
    assert schema.response_format("v2")["json_schema"]["schema"] == s
    with pytest.raises(ValueError):
        schema.json_schema("v3")


def test_v1_prompt_is_frozen():
    assert hashlib.sha256(prompts.SYSTEM.encode()).hexdigest()[:16] == "0536a92d5a5eb0a9"
    assert prompts.messages("x") == prompts.messages("x", "v1") and prompts.prompt_version("v1") == "v4"


def test_v2_prompt_defines_every_field_and_its_example_matches_the_schema():
    text = prompts.system("v2")
    assert all(f"- {name}:" in text for name in V2)
    example = json.loads(
        text[text.index('{"revenue"') :].replace(
            "{...}", '{"value": null, "scale": null, "period_end": null, "fiscal_year": null}'
        )
    )
    assert list(example) == list(V2)
    schema.answer_model("v2").model_validate(example)
    assert prompts.prompt_version("v2") == "v5"


def v2_gold():
    gold = copy.deepcopy(GOLD)
    extra = {
        "revenue_prior": (383_285e6, "millions", PRIOR_END, "us-gaap:Revenues"),
        "eps_basic": (6.11, "units", END, "us-gaap:EarningsPerShareBasic"),
        "cash": (29_943e6, "millions", END, CASH),
        "equity": (56_950e6, "millions", END, "us-gaap:StockholdersEquity"),
        "capex": (9_447e6, "millions", END, PPE),
        "fcf": (108_807e6, "millions", END, "derived"),
    }
    for name, (value, scale, period_end, tag) in extra.items():
        gold[name] = {"value": value, "scale": scale, "period_end": period_end, "fiscal_year": 2024, "tag": tag}
    return {n: gold[n] for n in V2}


def v2_record():
    rec = record()
    rec["gold"], rec["task"] = v2_gold(), "v2"
    return rec


def v2_answer(**overrides):
    out = {n: {k: g[k] for k in ("value", "scale", "period_end", "fiscal_year")} for n, g in v2_gold().items()}
    for name, part in overrides.items():
        out[name] = {**out[name], **part}
    return out


def pred(obj, rid="EXM-2024"):
    return {rid: {"id": rid, "raw": json.dumps(obj), "error": None}}


def test_v2_scoring():
    rows = score([v2_record()], pred(v2_answer()))
    assert [r["field"] for r in rows] == list(V2)
    assert all(r["correct"] and r["valid_json"] for r in rows)
    # A v1-shaped answer to a v2 filing is not the schema; its six fields still score
    v1_only = {n: v for n, v in v2_answer().items() if n in FIELD_NAMES}
    rows = score([v2_record()], pred(v1_only))
    assert not rows[0]["valid_json"]
    assert {r["field"] for r in rows if r["correct"]} == set(FIELD_NAMES)
    assert parse_answer(json.dumps(v1_only))[1] and not parse_answer(json.dumps(v1_only), "v2")[1]


@pytest.mark.parametrize(
    ("field", "part", "expected"),
    [
        ("revenue_prior", {"value": 391_035e6, "period_end": END}, "wrong_period"),  # this year's revenue
        ("eps_basic", {"value": 6.08}, "wrong_line_item"),  # the diluted figure (it's in the gold facts)
        ("capex", {"value": -9_447e6}, "sign"),  # copied the parentheses
        ("eps_basic", {"value": 6.12}, "other"),  # one cent off is wrong
    ],
)
def test_v2_error_categories(field, part, expected):
    rec = v2_record()
    rec["context_facts"] = [
        *rec["context_facts"],
        {
            "tag": "us-gaap:EarningsPerShareDiluted",
            "unit": "USD/shares",
            "start": "2023-10-01",
            "end": END,
            "val": 6.08,
        },
    ]
    rows = {r["field"]: r["category"] for r in score([rec], pred(v2_answer(**{field: part})))}
    assert rows[field] == expected


def test_basic_eps_has_the_cent_tolerance():
    assert close(6.114, 6.11, "eps_basic") and not close(6.12, 6.11, "eps_basic")


def test_answer_text_has_twelve_keys_in_order():
    rec = v2_record()
    rec["gold"]["fcf"] = None
    text, hidden = sft.answer_text(rec)
    obj = json.loads(text)
    assert list(obj) == list(V2) and obj["fcf"] == dict.fromkeys(sft.PARTS)
    assert [text[a:b] for a, b in hidden] == [json.dumps(dict.fromkeys(sft.PARTS))]
    assert parse_answer(text, "v2")[1]


# --- a balance sheet printed across two pages ---------------------------------------------------

PAGE_BREAK = (
    "</table><div>Example Corp. | 31</div><table><tr><td></td><td>2024</td><td>2023</td></tr>"
    "<tr><td>Accounts payable</td><td>68,960</td><td>62,611</td></tr>"
    "<tr><td>Total current liabilities</td><td>176,392</td><td>145,308</td></tr>"
)


def split_doc(between=""):
    """The v2 fixture with the balance sheet broken after total assets, as many 10-Ks print it."""
    doc = v2_doc().decode()
    assert doc.count(EQUITY_ROW) == 1
    return doc.replace(EQUITY_ROW, PAGE_BREAK.replace("<table>", between + "<table>", 1) + EQUITY_ROW).encode()


def test_v2_joins_a_balance_sheet_split_across_two_pages():
    assert "56,950" not in inputs.build(split_doc()).text  # v1: one table per statement
    built = inputs.build(split_doc(), join_halves=True)
    bs = built.text.split("## Balance sheet", 1)[1].split("\n## ", 1)[0]
    assert "364,980" in bs and "56,950" in bs
    assert inputs.build(v2_doc(), join_halves=True).text == inputs.build(v2_doc()).text  # one table: nothing to join
    rec = build_record(FILING, v2_record_facts(), split_doc(), "v2")
    assert rec["gold"]["equity"]["value"] == 56_950_000_000
    assert build_record(FILING, v2_record_facts(), split_doc())["gold"]["total_assets"] is not None


def test_the_next_statement_is_not_a_second_half():
    doc = split_doc(between="<div>CONSOLIDATED STATEMENTS OF SHAREHOLDERS' EQUITY</div>")
    assert "56,950" not in inputs.build(doc, join_halves=True).text
    paragraph = "<div>" + "The accompanying notes are an integral part of these statements. " * 8 + "</div>"
    assert "56,950" not in inputs.build(split_doc(between=paragraph), join_halves=True).text
