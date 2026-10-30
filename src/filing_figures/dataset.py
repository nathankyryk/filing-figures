"""One record per 10-K: the model input, the gold answer and everything needed to check and score it.

A field's gold answer is scored only when its tagged number is inside the input the model sees
(`in_input`), so a miss by the statement finder never counts against a model. A filing missing
three or more of its six fields is left out entirely (`usable` is False); the counts go in the README.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from filing_figures import inputs
from filing_figures.labels import FIELD_NAMES, label_filing

MAX_MISSING = 2  # a filing with more unscorable fields than this is dropped
SPLIT_SEED = 0
SPLITS = ("train", "valid", "test")
HANDCHECK_COLUMNS = ("id", "field", "verdict", "value", "scale", "period_end", "note", "checked_at")
VERDICTS = ("ok", "fix", "drop")  # fix also covers adding a value where the key had none


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

    def covered(gold: dict) -> dict:
        # In the selected section AND its displayed number in the text (the cover can be cut short)
        out = inputs.coverage(sel, built.root, built.contexts, gold)
        for c in out.values():
            if c and c["in_input"] and " ".join((c["text"] or "").split()) not in text:
                c["in_input"] = False
        return out

    cov = covered(labelled["labels"])
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
                alt_c = covered({name: alt})[name]
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
        period_end = label["end"]
        if name == "shares_outstanding":
            # The date a reader sees: the one printed next to the count on the cover
            cover = text.split("\n\n## ", 1)[0]  # the input starts with the cover section
            printed = inputs.date_near(cover, " ".join((c["text"] or "").split()))
            # The count is dated after the year end; an earlier date is some other figure's
            if printed and printed != period_end and printed >= (labelled["period_end"] or ""):
                period_end = printed
                flags[name].append("date_from_text")
        gold[name] = {
            "value": label["value"],
            "scale": c["scale"],
            "period_end": period_end,
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


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def read_handcheck(path: Path) -> dict[tuple[str, str], dict]:
    """{(record id, field): the latest hand-check row}."""
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {(r["id"], r["field"]): r for r in csv.DictReader(f)}


def write_handcheck(rows: dict[tuple[str, str], dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, HANDCHECK_COLUMNS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({c: rows[key].get(c, "") for c in HANDCHECK_COLUMNS})
    tmp.replace(path)


def apply_handcheck(rec: dict, checks: dict[tuple[str, str], dict]) -> dict:
    """The record with the hand-check applied: `fix` replaces (or adds) the gold value, `drop`
    takes the field out of scoring, `ok` confirms it. Each leaves a `handcheck_*` flag."""
    rec = json.loads(json.dumps(rec))  # deep copy
    for name in FIELD_NAMES:
        row = checks.get((rec["id"], name))
        if not row:
            continue
        verdict = row["verdict"]
        rec["flags"][name].append(f"handcheck_{verdict}")
        if verdict == "drop":
            rec["gold"][name] = None
        elif verdict == "fix":
            old = rec["gold"][name] or {}
            rec["gold"][name] = {
                "value": float(row["value"]),
                "scale": row["scale"] or old.get("scale"),
                "period_end": row["period_end"] or old.get("period_end"),
                "fiscal_year": rec["fiscal_year"],
                "tag": old.get("tag", "handcheck"),
                "displayed": None,
            }
    return rec


def export_splits(records_path: Path, out_dir: Path, handcheck_path: Path) -> dict[str, Counter]:
    """data/export/{train,valid,test}.jsonl: the usable records, hand-check applied."""
    checks = read_handcheck(handcheck_path)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for rec in read_jsonl(records_path):
        if "error" in rec or rec.get("split") not in by_split:
            continue
        c = counts[rec["split"]]
        c["filings"] += 1
        if not rec["usable"]:
            c["not_usable"] += 1
            continue
        rec = apply_handcheck(rec, checks)
        c["exported"] += 1
        c["scored_fields"] += sum(g is not None for g in rec["gold"].values())
        c["handchecked_fields"] += sum(any(x.startswith("handcheck_") for x in fl) for fl in rec["flags"].values())
        by_split[rec["split"]].append(rec)
    for split, recs in by_split.items():
        write_jsonl(sorted(recs, key=lambda r: r["id"]), out_dir / f"{split}.jsonl")
    return counts


def push_dataset(folder: Path, repo_id: str, token: str | None, message: str) -> str:
    """Upload the export to a Hugging Face dataset repo, created private if it doesn't exist.
    An existing repo keeps its visibility (publishing is a separate, deliberate step)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        commit_message=message,
    )
    return str(info)
