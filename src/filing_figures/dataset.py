"""One record per 10-K: the model input, the gold answer and everything needed to check and score it.

A field's gold answer is scored only when its tagged number is inside the input the model sees
(`in_input`), so a miss by the statement finder never counts against a model. A filing missing
three or more of its six fields is left out entirely (`usable` is False); the counts go in the README.
"""

from __future__ import annotations

import random

from filing_figures import inputs
from filing_figures.labels import FIELD_NAMES, label_filing

MAX_MISSING = 2  # a filing with more unscorable fields than this is dropped
SPLIT_SEED = 0


def assign_splits(companies: list[dict], valid: float = 0.15, test: float = 0.15, seed: int = SPLIT_SEED) -> dict:
    """{cik: "train" | "valid" | "test"}, by company (a 10-K repeats last year's numbers, so
    one company in two splits would leak), stratified by sector, seeded."""
    rng = random.Random(seed)
    by_sector: dict[str, list[int]] = {}
    for c in companies:
        by_sector.setdefault(c["sector"], []).append(int(c["cik"]))
    out = {}
    for sector in sorted(by_sector):
        ciks = sorted(by_sector[sector])
        rng.shuffle(ciks)
        n_test, n_valid = round(len(ciks) * test), round(len(ciks) * valid)
        for i, cik in enumerate(ciks):
            out[cik] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    return out


def build_record(filing: dict, companyfacts: dict, html: bytes) -> dict:
    """filing: a row of data/universe/filings.csv (cik, ticker, sector, accession, ...)."""
    labelled = label_filing(companyfacts, filing["accession"], filing["report_date"])
    built = inputs.build(html)
    sel, text, doc = built.selection, built.text, built.dei
    cov = inputs.coverage(sel, built.root, built.contexts, labelled["labels"])
    try:
        fiscal_year = int(doc.get("DocumentFiscalYearFocus", ""))
    except ValueError:
        fiscal_year = labelled["fiscal_year"]

    gold, flags = {}, {}
    for name in FIELD_NAMES:
        label, c = labelled["labels"][name], cov[name]
        flags[name] = list(labelled["flags"][name])
        if label is None:
            gold[name] = None
            continue
        if not c["in_input"]:
            # The statement may show a later tag of the fallback list (e.g. contract revenue rather
            # than `Revenues`): the figure the model reads is the one to score.
            for alt in label.get("alternatives", []):
                alt_c = inputs.coverage(sel, built.root, built.contexts, {name: alt})[name]
                if alt_c["in_input"]:
                    label, c = alt, alt_c
                    flags[name].append("tag_from_statement")
                    break
        if not c["in_input"]:
            flags[name].append("not_in_input")
            gold[name] = None
            continue
        if c["mismatch"]:
            flags[name].append("mismatch")
        gold[name] = {
            "value": label["value"],
            "scale": c["scale"],
            "period_end": label["end"],
            "fiscal_year": fiscal_year,
            "tag": label["tag"],
            "displayed": c["text"],
        }
    unscored = sum(g is None for g in gold.values())
    record_flags = list(labelled["period_flags"])
    if labelled["fiscal_year"] != fiscal_year:
        record_flags.append("fy_disagree")
    if doc.get("DocumentType", "10-K") != "10-K" or doc.get("AmendmentFlag", "false").lower() == "true":
        record_flags.append("not_original_10k")
    return {
        "id": f"{filing['ticker']}-{fiscal_year}",
        "cik": int(filing["cik"]),
        "ticker": filing["ticker"],
        "sector": filing["sector"],
        "split": filing.get("split"),
        "accession": filing["accession"],
        "fiscal_year": fiscal_year,
        "period_end": labelled["period_end"],
        "filing_date": filing["filing_date"],
        "primary_document": filing["primary_document"],
        "sections": sorted(sel.statements),
        "input": text,
        "input_chars": len(text),
        "gold": gold,
        "flags": flags,
        "record_flags": record_flags,
        "usable": unscored <= MAX_MISSING and "not_original_10k" not in record_flags,
        "context_facts": labelled["context_facts"],
    }
