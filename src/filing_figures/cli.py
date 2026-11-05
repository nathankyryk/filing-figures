"""The `figures` command line."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, add_completion=False)
DATA = Path("data")


OUT = Path(os.environ.get("FIGURES_OUT", "outputs"))  # the pod's check stage writes to outputs/check
HANDCHECK = Path("labels") / "handcheck.csv"
# v2 reuses v1's checks (same filings, same six fields) and adds its own for the new fields
HANDCHECKS = {"v1": [HANDCHECK], "v2": [HANDCHECK, Path("labels") / "handcheck_v2.csv"]}
OOD = DATA / "universe" / "ood"  # the out-of-distribution companies and filings (task v2)
TASK_HELP = "Task version: v1 (6 fields) or v2 (12)"
TASK_SPLITS = {"v1": ("train", "valid", "test"), "v2": ("train", "valid", "test", "ood-valid", "ood-test")}
# Scored once, at the end: a run on these needs --final and is logged
FINAL_SPLITS = ("test", "ood-test")
# The splits a checkpoint is picked on: v2 pools in-distribution and out-of-distribution validation
PICK_SPLITS = {"v1": ("valid",), "v2": ("valid", "ood-valid")}


@app.callback()
def main() -> None:
    """FilingFigures: six figures from SEC 10-K filings, scored against XBRL."""
    load_dotenv()


def _edgar():
    from filing_figures.edgar import Edgar

    load_dotenv()
    return Edgar(cache_dir=DATA / "raw")


@app.command()
def universe(
    n: int = typer.Option(250, help="Companies to sample"),
    seed: int = typer.Option(0, help="Sampling seed"),
    min_filings: int = typer.Option(4, help="Keep companies with at least this many usable 10-Ks"),
) -> None:
    """Sample the companies and list their FY2019-2024 inline-XBRL 10-Ks (data/universe/)."""
    from filing_figures.universe import choose_companies, fetch_sp500, list_10ks, save_csv

    edgar = _edgar()
    out = DATA / "universe"
    sp500_path = out / "sp500.csv"
    if sp500_path.exists():  # keep the first download: the list changes as the index does
        sp500 = pd.read_csv(sp500_path)
    else:
        sp500 = fetch_sp500(os.environ["SEC_USER_AGENT"])
        save_csv(sp500, sp500_path)
    companies = choose_companies(sp500, n, seed)
    rows, kept = [], []
    with typer.progressbar(companies.itertuples(), length=len(companies), label="companies") as bar:
        for c in bar:
            filings = list_10ks(edgar.submissions(c.cik), edgar.companyfacts(c.cik))
            if len(filings) < min_filings:
                continue
            kept.append(c.cik)
            rows += [{"cik": c.cik, "ticker": c.ticker, "sector": c.sector, **f} for f in filings]
    save_csv(companies[companies["cik"].isin(kept)], out / "companies.csv")
    save_csv(pd.DataFrame(rows), out / "filings.csv")
    years = pd.DataFrame(rows)["fiscal_year"].value_counts().sort_index().to_dict()
    typer.echo(
        f"{len(sp500)} S&P 500 rows -> {len(companies)} sampled -> {len(kept)} companies with >= {min_filings} "
        f"10-Ks -> {len(rows)} filings {years}"
    )


@app.command()
def split() -> None:
    """Assign each company to train/valid/test (data/universe/splits.csv): by company, by sector, seeded."""
    from filing_figures.dataset import assign_splits

    companies = pd.read_csv(DATA / "universe" / "companies.csv")
    splits = assign_splits(companies.to_dict("records"))
    companies["split"] = companies["cik"].map(splits)
    companies[["cik", "ticker", "sector", "split"]].to_csv(DATA / "universe" / "splits.csv", index=False)
    filings = pd.read_csv(DATA / "universe" / "filings.csv").merge(companies[["cik", "split"]], on="cik")
    typer.echo(companies["split"].value_counts().to_string())
    typer.echo(filings["split"].value_counts().to_string())


@app.command("ood-universe")
def ood_universe(
    n: int = typer.Option(200, help="Companies to sample"),
    seed: int = typer.Option(0, help="Sampling seed"),
) -> None:
    """The out-of-distribution set for task v2 (data/universe/ood/): FY2025 10-Ks of S&P MidCap 400
    companies outside v1's S&P 500 list, split by company into ood-valid and ood-test."""
    import json

    from filing_figures.dataset import assign_ood_splits
    from filing_figures.universe import (
        OOD_YEAR,
        TICKERS_URL,
        choose_companies,
        fetch_sp400,
        list_10ks,
        save_csv,
        ticker_ciks,
    )

    edgar = _edgar()
    sp400_path = OOD / "sp400.csv"
    if sp400_path.exists():  # keep the first download: the list changes as the index does
        sp400 = pd.read_csv(sp400_path, dtype={"cik": "Int64"})
    else:
        sp400 = fetch_sp400(ticker_ciks(json.loads(edgar.cached(TICKERS_URL, Path("company_tickers.json.gz")))))
        save_csv(sp400, sp400_path)
    unresolved = sp400[sp400["cik"].isna()]
    in_v1 = set(pd.read_csv(DATA / "universe" / "sp500.csv")["cik"])
    pool = sp400.dropna(subset=["cik"]).astype({"cik": int})
    pool = pool[~pool["cik"].isin(in_v1)]
    companies = choose_companies(pool, n, seed)
    rows, kept = [], []
    with typer.progressbar(companies.itertuples(), length=len(companies), label="companies") as bar:
        for c in bar:
            filings = list_10ks(edgar.submissions(c.cik), edgar.companyfacts(c.cik), OOD_YEAR, OOD_YEAR)
            if filings:
                kept.append(c.cik)
                rows += [{"cik": c.cik, "ticker": c.ticker, "sector": c.sector, **f} for f in filings]
    companies = companies[companies["cik"].isin(kept)].copy()
    companies["split"] = companies["cik"].map(assign_ood_splits(companies.to_dict("records")))
    save_csv(companies, OOD / "companies.csv")
    save_csv(companies[["cik", "ticker", "sector", "split"]], OOD / "splits.csv")
    save_csv(pd.DataFrame(rows), OOD / "filings.csv")
    typer.echo(
        f"{len(sp400)} S&P 400 rows ({len(unresolved)} without a CIK) -> {len(pool)} outside v1's list "
        f"-> {len(companies)} of {min(n, len(pool))} sampled have an FY{OOD_YEAR} inline-XBRL 10-K"
    )
    typer.echo(companies["split"].value_counts().to_string())
    typer.echo(companies.groupby(["sector", "split"]).size().unstack(fill_value=0).to_string())


def _filings(limit: int | None, tickers: str | None, split: str | None = None, ood: bool = False) -> pd.DataFrame:
    """The filings to work on. `ood` adds the out-of-distribution set (task v2 only, so v1 never changes)."""
    df = pd.read_csv(DATA / "universe" / "filings.csv", dtype={"accession": str})
    splits_path = DATA / "universe" / "splits.csv"
    if splits_path.exists():
        df = df.merge(pd.read_csv(splits_path)[["cik", "split"]], on="cik", how="left")
    if ood and (OOD / "filings.csv").exists():
        extra = pd.read_csv(OOD / "filings.csv", dtype={"accession": str})
        extra = extra.merge(pd.read_csv(OOD / "splits.csv")[["cik", "split"]], on="cik", how="left")
        df = pd.concat([df, extra], ignore_index=True)
    if split:
        df = df[df["split"].isin(split.split(","))]
    if tickers:
        df = df[df["ticker"].isin(tickers.split(","))]
    return df.head(limit) if limit else df


@app.command()
def download(
    limit: int = typer.Option(None, help="Only the first N filings"),
    split: str = typer.Option(None, help="Only these splits, e.g. train or train,valid"),
    tickers: str = typer.Option(None, help="Comma-separated tickers"),
    workers: int = typer.Option(4, help="Download threads (they share the SEC rate limit)"),
) -> None:
    """Download each filing's primary document into the cache (data/raw/docs)."""
    from concurrent.futures import ThreadPoolExecutor

    edgar = _edgar()
    df = _filings(limit, tickers, split, ood=True)
    failed = []

    def get(row):
        try:
            edgar.document(row.cik, row.accession, row.primary_document)
        except Exception as e:  # noqa: BLE001 - record and carry on; reported at the end
            failed.append((row.accession, repr(e)))

    with ThreadPoolExecutor(workers) as pool, typer.progressbar(length=len(df), label="documents") as bar:
        for _ in pool.map(get, df.itertuples()):
            bar.update(1)
    typer.echo(f"{len(df) - len(failed)}/{len(df)} documents cached")
    for accession, err in failed:
        typer.echo(f"  FAILED {accession}: {err}")


def _task_dir(task: str, root: Path) -> Path:
    """v1 keeps its original paths (data/build, data/export); later tasks get a subfolder."""
    return root if task == "v1" else root / task


def _build_one(row: dict, task: str = "v1") -> dict:
    """One record, from the cache only (run `download` first). Top-level so processes can run it."""
    import gzip
    import json

    from filing_figures.dataset import build_record
    from filing_figures.edgar import cik10

    raw = DATA / "raw"
    folder = row["accession"].replace("-", "")
    try:
        cf = json.loads(gzip.decompress((raw / "companyfacts" / f"{cik10(row['cik'])}.json.gz").read_bytes()))
        html = gzip.decompress((raw / "docs" / str(row["cik"]) / folder / f"{row['primary_document']}.gz").read_bytes())
        return build_record(row, cf, html, task)
    except Exception as e:  # noqa: BLE001 - one bad filing must not stop the build
        return {"id": f"{row['ticker']}-{row['fiscal_year']}", "accession": row["accession"], "error": repr(e)}


@app.command()
def build(
    limit: int = typer.Option(None, help="Only the first N filings"),
    split: str = typer.Option(None, help="Only these splits, e.g. train or train,valid"),
    tickers: str = typer.Option(None, help="Comma-separated tickers"),
    workers: int = typer.Option(4, help="Processes"),
    task: str = typer.Option("v1", help="Task version: v1 (6 fields) or v2 (12)"),
    out: Path = typer.Option(None, help="Output JSONL [default: data/build(/<task>)/records.jsonl]"),
) -> None:
    """Labels + model input + coverage for every cached filing, then a coverage summary."""
    import json
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from filing_figures.labels import FIELD_BY_NAME, field_names

    names = field_names(task)
    out = out or _task_dir(task, DATA / "build") / "records.jsonl"
    rows = _filings(limit, tickers, split, ood=task != "v1").to_dict("records")
    out.parent.mkdir(parents=True, exist_ok=True)
    errors, flag_counts = [], Counter()
    usable, total, in_input, labelled = Counter(), Counter(), Counter(), Counter()  # keyed by (split, field)
    with (
        ProcessPoolExecutor(workers) as pool,
        out.open("w", encoding="utf-8") as f,
        typer.progressbar(
            pool.map(partial(_build_one, task=task), rows, chunksize=4), length=len(rows), label="filings"
        ) as bar,
    ):
        for rec in bar:
            f.write(json.dumps(rec) + "\n")
            if "error" in rec:
                errors.append(rec)
                continue
            s = rec.get("split") or "-"
            total[s] += 1
            usable[s] += rec["usable"]

            def is_labelled(name: str, rec: dict = rec) -> bool:
                if FIELD_BY_NAME[name].kind == "derived":  # labelled when all its parts are
                    return all(is_labelled(p) for p in FIELD_BY_NAME[name].parts)
                return rec["gold"][name] is not None or "not_in_input" in rec["flags"][name]

            for name in names:
                flag_counts.update(f"{name}:{x}" for x in rec["flags"][name])
                labelled[s, name] += is_labelled(name)
                in_input[s, name] += rec["gold"][name] is not None
    typer.echo(f"\n{len(rows)} filings, {len(errors)} errors -> {out}")
    typer.echo("Field coverage: labelled values that are inside the model input")
    order = ("train", "valid", "test", "ood-valid", "ood-test")
    splits = sorted(total, key=lambda s: order.index(s) if s in order else 9)
    typer.echo(f"  {'':20s}" + "".join(f"{s:>18s}" for s in splits))
    for name in [*names, "ALL"]:
        cells = []
        for s in splits:
            n_in = sum(in_input[s, f] for f in names) if name == "ALL" else in_input[s, name]
            n_lab = sum(labelled[s, f] for f in names) if name == "ALL" else labelled[s, name]
            cells.append(f"{n_in:>6d}/{n_lab:<6d}{100 * n_in / max(n_lab, 1):5.1f}%")
        typer.echo(f"  {name:20s}" + "".join(f"{c:>18s}" for c in cells))
    typer.echo(f"  {'usable filings':20s}" + "".join(f"{f'{usable[s]}/{total[s]}':>18s}" for s in splits))
    typer.echo("flags: " + ", ".join(f"{k}={v}" for k, v in sorted(flag_counts.items())))
    for e in errors[:10]:
        typer.echo(f"  ERROR {e['id']}: {e['error']}")


@app.command()
def export(
    task: str = typer.Option("v1", help="Task version: v1 or v2"),
    records: Path = typer.Option(None, help="Output of `build` [default: data/build(/<task>)/records.jsonl]"),
    out: Path = typer.Option(None, help="Folder for <split>.jsonl [default: data/export(/<task>)]"),
) -> None:
    """Write the usable records of each split, with the hand-check (labels/handcheck*.csv) applied."""
    from filing_figures.dataset import export_splits

    records = records or _task_dir(task, DATA / "build") / "records.jsonl"
    out = out or _task_dir(task, DATA / "export")
    for split, c in export_splits(records, out, HANDCHECKS[task]).items():
        typer.echo(
            f"{split:9s} {c['exported']:4d} of {c['filings']:4d} filings exported ({c['not_usable']} not usable), "
            f"{c['scored_fields']} scored fields, {c['handchecked_fields']} hand-checked"
        )


@app.command()
def push(
    task: str = typer.Option("v1", help=TASK_HELP),
    repo: str = typer.Option(None, help="Hugging Face dataset repo [default: the task's, runs.DATASETS]"),
    folder: Path = typer.Option(None, help="[default: data/export(/<task>)]"),
    message: str = typer.Option("Update the dataset", help="Commit message"),
) -> None:
    """Upload the export to Hugging Face (created private; publishing is a separate step)."""
    from filing_figures.dataset import push_dataset
    from filing_figures.runs import DATASETS

    repo = repo or DATASETS[task]
    folder = folder or _task_dir(task, DATA / "export")
    typer.echo(push_dataset(folder, repo, os.environ.get("HF_TOKEN"), message))


def _records(
    split: str, sample: int | None = None, seed: int = 0, ids: str | None = None, task: str = "v1"
) -> list[dict]:
    import random

    from filing_figures.dataset import read_jsonl

    recs = read_jsonl(_task_dir(task, DATA / "export") / f"{split}.jsonl")
    if ids:
        keep = set(ids.split(","))
        recs = [r for r in recs if r["id"] in keep]
    if sample and sample < len(recs):
        recs = sorted(random.Random(seed).sample(recs, sample), key=lambda r: r["id"])
    return recs


def _check_split(split: str, final: bool) -> None:
    if split in FINAL_SPLITS and not final:
        raise typer.BadParameter(f"the {split} split is for the final runs only: pass --final")


def _preds_path(split: str, run: str, task: str = "v1") -> Path:
    return _task_dir(task, OUT / "preds") / split / f"{run}.jsonl"


def _scores_path(split: str, run: str, task: str = "v1") -> Path:
    return _task_dir(task, OUT / "scores") / split / f"{run}.json"


@app.command()
def predict(
    endpoint: str = typer.Option(..., help="openrouter, vllm, stub, rules, or a base URL"),
    model: str = typer.Option("", help="Model name at the endpoint"),
    split: str = typer.Option("valid", help="train, valid or test (test needs --final)"),
    run: str = typer.Option(None, help="Run name (default: the model name)"),
    sample: int = typer.Option(None, help="A seeded sample of N filings"),
    seed: int = typer.Option(0, help="Sampling seed"),
    ids: str = typer.Option(None, help="Comma-separated record ids"),
    workers: int = typer.Option(8, help="Requests in flight"),
    max_tokens: int = typer.Option(2048, help="Output token cap per filing"),
    schema: bool = typer.Option(False, "--schema/--no-schema", help="Constrained decoding with the JSON schema"),
    reasoning: str = typer.Option(None, help="off, or an effort level: low, medium, high"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    no_temperature: bool = typer.Option(False, help="Don't send a temperature (some API models reject it)"),
    max_cost: float = typer.Option(None, help="USD: no new requests once this much is spent"),
    extra: str = typer.Option(None, help="JSON merged into each request body"),
    final: bool = typer.Option(False, help="Allow the test and ood-test splits; the run is logged"),
    task: str = typer.Option("v1", help=TASK_HELP),
) -> None:
    """Ask a model about each filing (outputs/preds(/<task>)/<split>/<run>.jsonl), then score it."""
    import json

    from filing_figures.client import Config

    _check_split(split, final)
    run = run or (model or endpoint).replace("/", "_").replace(":", "_")
    cfg = Config(
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        temperature=None if no_temperature else temperature,
        schema=schema,
        reasoning=reasoning,
        workers=workers,
        max_cost=max_cost,
        extra_body=json.loads(extra) if extra else {},
    )
    _predict_and_score(cfg, split, run, sample, seed, ids, task)


def _predict_and_score(
    cfg, split: str, run: str, sample: int | None, seed: int, ids: str | None, task: str = "v1"
) -> None:
    import json
    from datetime import UTC, datetime

    from filing_figures.client import predict as run_predict

    recs = _records(split, sample, seed, ids, task)
    out = _preds_path(split, run, task)
    if split in FINAL_SPLITS:
        log = OUT / "logs" / "test_runs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "run": run, "n": len(recs)}
            if task != "v1":  # v1's log lines keep their shape
                entry |= {"task": task, "split": split}
            f.write(json.dumps(entry | {"config": cfg.__dict__}) + "\n")
    with typer.progressbar(length=len(recs), label=run) as bar:
        stats = run_predict(recs, cfg, out, on_result=lambda _: bar.update(1))
    typer.echo(f"{stats} -> {out}")
    score(split=split, run=run, sample=sample, seed=seed, ids=ids, task=task)


@app.command()
def score(
    run: str = typer.Option(..., help="Run name"),
    split: str = typer.Option("valid", help="train, valid, test, ood-valid or ood-test"),
    sample: int = typer.Option(None, help="The same seeded sample the run used"),
    seed: int = typer.Option(0, help="Sampling seed"),
    ids: str = typer.Option(None, help="Comma-separated record ids"),
    task: str = typer.Option("v1", help=TASK_HELP),
) -> None:
    """Score a run: field accuracy with 95% CIs (bootstrap over companies) and error categories."""
    import json

    from filing_figures.evaluate import examples, load_predictions, summarize, table
    from filing_figures.evaluate import score as score_rows

    recs = _records(split, sample, seed, ids, task)
    preds = load_predictions(_preds_path(split, run, task))
    missing = [r["id"] for r in recs if r["id"] not in preds]
    if missing:
        typer.echo(f"WARNING: {len(missing)} of {len(recs)} filings have no prediction (scored as no_response)")
    rows = score_rows(recs, preds)
    summary = summarize(rows)
    typer.echo(f"{run} on {split}: " + table(summary))
    path = _scores_path(split, run, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "examples": examples(rows)}, indent=1), encoding="utf-8")


@app.command()
def compare(
    a: str = typer.Option(..., help="Run A"),
    b: str = typer.Option(..., help="Run B"),
    split: str = typer.Option("valid", help="train, valid, test, ood-valid or ood-test"),
    task: str = typer.Option("v1", help=TASK_HELP),
) -> None:
    """Overall accuracy of B minus A on the filings both runs answered (paired company bootstrap)."""
    from filing_figures.evaluate import compare as compare_rows
    from filing_figures.evaluate import load_predictions
    from filing_figures.evaluate import score as score_rows

    pa, pb = load_predictions(_preds_path(split, a, task)), load_predictions(_preds_path(split, b, task))
    recs = [r for r in _records(split, task=task) if r["id"] in pa and r["id"] in pb]
    gap = compare_rows(score_rows(recs, pa), score_rows(recs, pb))
    typer.echo(
        f"{b} - {a} on {len(recs)} {split} filings: {100 * gap['diff']:+.1f} points "
        f"[{100 * gap['lo']:+.1f}, {100 * gap['hi']:+.1f}], P(B not better) = {gap['p_b_not_better']:.3f}"
    )
    _write_json(_task_dir(task, OUT / "compare") / split / f"{b}-vs-{a}.json", gap | {"a": a, "b": b, "n": len(recs)})


CANDIDATE_TOKENIZERS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "google/gemma-4-E4B-it", "Qwen/Qwen3-8B")
MAX_INPUT_TOKENS = 16_000


@app.command()
def tokens(
    repos: str = typer.Option(",".join(CANDIDATE_TOKENIZERS), help="Comma-separated HF model repos"),
) -> None:
    """Prompt + filing tokens per split for each candidate's tokenizer, and the answer's length."""
    import json

    import numpy as np
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from filing_figures import prompts
    from filing_figures.stub import answer

    splits = {s: _records(s) for s in ("train", "valid", "test")}
    for repo in repos.split(","):
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", token=os.environ.get("HF_TOKEN")))
        system = len(tok.encode(prompts.SYSTEM).ids)
        typer.echo(f"{repo}: system prompt {system} tokens")
        for split, recs in splits.items():
            n = np.array([system + len(tok.encode(r["input"]).ids) for r in recs])
            out = np.array([len(tok.encode(answer(r, "gold")).ids) for r in recs])
            over = int((n > MAX_INPUT_TOKENS).sum())
            typer.echo(
                f"  {split:6s} input median {int(np.median(n)):6d}  p95 {int(np.percentile(n, 95)):6d}  "
                f"max {n.max():6d}  over {MAX_INPUT_TOKENS}: {over}   answer median {int(np.median(out))} max {out.max()}"
            )
    _ = json  # (kept for symmetry with other commands)


HANDCHECK_POOL = Path("labels") / "handcheck_pool.csv"
# Per task: the split checked, the pool file, the verdicts file (the task's last hand-check file),
# and the pool's default size (filings, of which random)
REVIEWS = {
    "v1": ("test", HANDCHECK_POOL, HANDCHECKS["v1"][-1], 100, 40),
    "v2": ("ood-test", Path("labels") / "handcheck_v2_pool.csv", HANDCHECKS["v2"][-1], 50, 25),
}


@app.command()
def review(
    port: int = typer.Option(7862, help="Port on 127.0.0.1"),
    target: int = typer.Option(None, help="Filings in the pool (v1: 100, v2: 50)"),
    n_random: int = typer.Option(
        None, help="Of those, a random sample, whose fix rate is the answer key's error rate (v1: 40, v2: 25)"
    ),
    new_pool: bool = typer.Option(False, "--new-pool", help="Redraw the pool file"),
    report: bool = typer.Option(False, "--report", help="Print progress and exit"),
    task: str = typer.Option("v1", help="v1: the test split, labels/handcheck.csv; v2: ood-test, handcheck_v2.csv"),
) -> None:
    """The hand-check page for a final split's answer key."""
    from filing_figures.dataset import read_handcheck, read_jsonl
    from filing_figures.review import Review, build_pool, checked_fields, read_pool, serve, summary, write_pool

    split, pool_path, checks_path, default_target, default_random = REVIEWS[task]
    # The XBRL key as built, before any hand-check, so the page shows both side by side
    recs = [
        r
        for r in read_jsonl(_task_dir(task, DATA / "build") / "records.jsonl")
        if "error" not in r and r.get("split") == split and r["usable"]
    ]
    if new_pool or not pool_path.exists():
        n = default_random if n_random is None else n_random
        write_pool(build_pool(recs, target=target or default_target, n_random=n, task=task), pool_path)
    pool = read_pool(pool_path)
    s = summary(pool, read_handcheck(checks_path), checked_fields(task))
    typer.echo(
        f"{s['filings_done']} of {s['filings']} filings done; verdicts {dict(s['verdicts'])}; "
        f"random sample: {s['random_changed']} of {s['random_fields']} checked fields changed"
    )
    if report:
        return
    server = serve(Review(recs, pool, checks_path, task), port)
    then = "figures export" + ("" if task == "v1" else f" --task {task}")
    typer.echo(f"hand-check at http://127.0.0.1:{port}/ ; Ctrl+C to stop, then `{then}`")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


@app.command()
def stub(
    split: str = typer.Option("valid", help="Whose filings it recognises"),
    mode: str = typer.Option("gold", help="gold, prior or garbage"),
    port: int = typer.Option(8001, help="Port on 127.0.0.1"),
    task: str = typer.Option("v1", help=TASK_HELP),
) -> None:
    """A stand-in model server for testing the harness (see stub.py)."""
    from filing_figures.stub import StubServer

    server = StubServer(_records(split, task=task), mode, port=port)
    typer.echo(f"stub ({mode}) at {server.url}; Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


@app.command()
def extract(
    ticker: str = typer.Option(..., help="Company ticker, e.g. SBUX"),
    year: int = typer.Option(..., help="The 10-K's fiscal year (the company's own)"),
    endpoint: str = typer.Option("vllm", help="openrouter, vllm, rules, or a base URL"),
    model: str = typer.Option("", help="Model name at the endpoint"),
    reasoning: str = typer.Option("off", help="off, or an effort level: low, medium, high"),
    schema: bool = typer.Option(False, "--schema/--no-schema", help="Constrained decoding with the JSON schema"),
    max_tokens: int = typer.Option(2048, help="Output token cap"),
    as_json: bool = typer.Option(False, "--json", help="Print only the model's answer"),
) -> None:
    """Fetch one 10-K from EDGAR, ask a model for the six figures, and score them against XBRL."""
    from filing_figures.client import Config, ask
    from filing_figures.extract import compare, load_filing, table

    rec = load_filing(_edgar(), ticker, year)
    cfg = Config(
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        schema=schema,
        reasoning=None if endpoint == "rules" else reasoning,
    )
    if not as_json:  # show the filing before the model call, so the wait has context
        typer.echo(
            f"{rec['ticker']} FY{rec['fiscal_year']} 10-K (accession {rec['accession']}, filed {rec['filing_date']}): "
            f"{len(rec['input']):,} characters of statements"
        )
    got = ask(rec, cfg)
    if as_json:
        typer.echo(got["raw"])
        return
    typer.echo(f"{model or endpoint}: answered in {got['latency_s']:.1f} s\n")
    typer.echo(table(compare(rec, got["raw"])))


@app.command()
def space(out: Path = typer.Option(Path("space") / "data", help="The viewer's data folder")) -> None:
    """Build the results viewer's data (space/) from the saved test predictions and scores."""
    from filing_figures.space import build

    sp500 = pd.read_csv(DATA / "universe" / "sp500.csv")
    names = dict(zip(sp500["ticker"], sp500["name"], strict=True))
    typer.echo(build(_records("test"), OUT, out, names))


# --- GPU pod commands (scripts/run_gpu_jobs.sh runs these; each names the environment it needs) ---


def short_name(model: str) -> str:
    """Run-name form of a model id: Qwen/Qwen3.5-4B -> qwen3.5-4b."""
    return model.rstrip("/").split("/")[-1].lower()


def _write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _gpu_name() -> str:
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return ""
    out = subprocess.run(
        [smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


@app.command()
def doctor(task: str = typer.Option("v1", help=TASK_HELP)) -> None:
    """Check the machine before GPU work: GPU, CUDA, data, HF token, disk, installed packages."""
    import importlib.util
    import shutil
    import subprocess

    failures = 0

    def check(name: str, ok: bool, detail: str = "", critical: bool = True) -> None:
        nonlocal failures
        failures += (not ok) and critical
        typer.echo(f"{'OK ' if ok else '!! '} {name:20s} {detail}")

    smi = shutil.which("nvidia-smi")
    if smi:
        q = "--query-gpu=name,memory.total,driver_version"
        out = subprocess.run(
            [smi, q, "--format=csv,noheader"], capture_output=True, text=True, check=False
        ).stdout.strip()
        check("GPU", bool(out), out)
    else:
        check("GPU", False, "no nvidia-smi (fine on the laptop)", critical=False)
    if importlib.util.find_spec("torch"):
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, CUDA {torch.version.cuda}"
        if cuda:
            detail += f", {torch.cuda.get_device_name(0)}, bf16 {torch.cuda.is_bf16_supported()}"
        check("torch sees the GPU", cuda, detail, critical=bool(smi))
    for split in TASK_SPLITS[task]:
        p = _task_dir(task, DATA / "export") / f"{split}.jsonl"
        n = sum(1 for line in p.open(encoding="utf-8") if line.strip()) if p.exists() else 0
        check(f"data: {task} {split}", n > 0, f"{n} filings in {p.as_posix()}")
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            who = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
            check("HF token", who == "kyryk", f"account {who}")
        except Exception as e:  # noqa: BLE001 - reported, never printed with the token
            check("HF token", False, type(e).__name__)
    else:
        check("HF token", False, "HF_TOKEN is not set")
    for pkg in ("unsloth", "llmcompressor", "vllm", "trl", "peft"):
        found = importlib.util.find_spec(pkg) is not None
        check(f"package {pkg}", True, "installed" if found else "not in this environment", critical=False)
    free = shutil.disk_usage(Path.cwd()).free / 1e9
    check("disk", free > 60, f"{free:.0f} GB free")
    raise typer.Exit(1 if failures else 0)


@app.command("fetch-data")
def fetch_data(task: str = typer.Option("v1", help=TASK_HELP)) -> None:
    """Download the exported splits from the task's Hugging Face dataset (as last pushed)."""
    from filing_figures.runs import DATASETS, fetch_dataset

    for p in fetch_dataset(_task_dir(task, DATA / "export"), DATASETS[task]):
        typer.echo(f"{p.as_posix()}: {sum(1 for line in p.open(encoding='utf-8') if line.strip())} filings")


@app.command("vllm-predict")
def vllm_predict(
    model: str = typer.Option(..., help="HF repo id or a local model folder"),
    run: str = typer.Option(..., help="Run name (also the served model name)"),
    split: str = typer.Option("valid", help="train, valid or test (test needs --final)"),
    sample: int = typer.Option(None, help="A seeded sample of N filings"),
    seed: int = typer.Option(0, help="Sampling seed"),
    thinking: bool = typer.Option(False, help="Let the model reason before answering"),
    schema: bool = typer.Option(False, "--schema/--no-schema", help="Constrained decoding with the JSON schema"),
    max_tokens: int = typer.Option(None, help="Output cap (default 2048, or 16000 when thinking)"),
    workers: int = typer.Option(64, help="Requests in flight"),
    port: int = typer.Option(8000),
    final: bool = typer.Option(False, help="Allow the test and ood-test splits; the run is logged"),
    task: str = typer.Option("v1", help=TASK_HELP),
    splits: str = typer.Option(None, help="Several splits with one server start, e.g. valid,ood-valid"),
) -> None:
    """Serve a model with vLLM, ask it about each filing, score it, stop the server. (serve environment)"""
    import time

    from filing_figures.client import Config
    from filing_figures.vllm_server import MAX_MODEL_LEN, THINKING_MAX_MODEL_LEN, running

    todo = splits.split(",") if splits else [split]
    for s in todo:
        _check_split(s, final)
    t0 = time.time()
    max_len = THINKING_MAX_MODEL_LEN if thinking else MAX_MODEL_LEN
    log = _task_dir(task, OUT / "logs" / "vllm") / f"{run}.log"
    with running(model, run, log, port=port, max_model_len=max_len, thinking=thinking) as (url, stats):
        startup_s = time.time() - t0
        cfg = Config(
            endpoint=url,
            model=run,
            max_tokens=max_tokens or (16_000 if thinking else 2048),
            schema=schema,
            reasoning="on" if thinking else "off",
            workers=workers,
            timeout=1800.0,  # a long thinking answer, queued behind 63 others
        )
        t1 = time.time()
        for s in todo:
            _predict_and_score(cfg, s, run, sample, seed, None, task)
        info = {"model": model, "split": ",".join(todo), "gpu": _gpu_name(), "startup_s": round(startup_s, 1)}
        out = _task_dir(task, OUT / "serve") / f"{run}.json"
        _write_json(out, info | {"predict_s": round(time.time() - t1, 1), **stats})


@app.command()
def train(
    base: str = typer.Option(..., help="Base model repo, e.g. Qwen/Qwen3.5-4B"),
    name: str = typer.Option(None, help="Run name (default: ft-<base>)"),
    epochs: float = typer.Option(3.0),
    rank: int = typer.Option(16),
    alpha: int = typer.Option(32),
    lr: float = typer.Option(2e-4),
    grad_accum: int = typer.Option(16, help="Filings per optimizer step"),
    max_steps: int = typer.Option(-1, help="> 0: a timing run (check stage)"),
    max_seq_length: int = typer.Option(8192, help="Longer filings are dropped (v2: 9216 keeps every train filing)"),
    four_bit: bool | None = typer.Option(
        None, "--four-bit/--bf16", help="Default: QLoRA where supported (train.FOUR_BIT)"
    ),
    task: str = typer.Option("v1", help=TASK_HELP),
) -> None:
    """LoRA fine-tune on the train split; adapters every half epoch in outputs/train/<name>. (train environment)"""
    from filing_figures.train import TrainConfig
    from filing_figures.train import train as run_train

    name = name or f"ft-{short_name(base)}"
    cfg = TrainConfig(
        base=base,
        out=OUT / "train" / name,
        epochs=epochs,
        rank=rank,
        alpha=alpha,
        lr=lr,
        grad_accum=grad_accum,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        load_in_4bit=four_bit,
    )
    meta = run_train(cfg, _records("train", task=task), _records("valid", task=task))
    typer.echo({k: meta[k] for k in ("steps", "seconds_per_step", "peak_gpu_gib", "train_loss", "dropped_too_long")})


RL_TASK = typer.Option(..., help=TASK_HELP)


@app.command("rl-filter")
def rl_filter(
    init: str = typer.Option(..., help="Starting model: a repo or a merged folder"),
    name: str = typer.Option(..., help="Output folder: outputs/rl-filter/<name>"),
    task: str = RL_TASK,
    split: str = typer.Option("train"),
    sample: int = typer.Option(None, help="A seeded sample of N filings (check stage)"),
    seed: int = typer.Option(0),
    group: int = typer.Option(8, help="Answers per filing"),
    gpu_mem: float = typer.Option(0.9, help="vLLM's share of GPU memory"),
) -> None:
    """Sample answers at temperature 1 and score them: GRPO's difficulty filter. (rl environment)"""
    from filing_figures import reward
    from filing_figures.rl import sample_groups

    _check_split(split, final=False)  # test and ood-test are never sampled
    recs = _records(split, sample=sample, seed=seed, task=task)
    groups = sample_groups(init, recs, OUT / "rl-filter" / name, n=group, seed=seed, gpu_mem=gpu_mem)
    typer.echo(reward.filter_summary(groups))


@app.command("train-rl")
def train_rl(
    init: str = typer.Option(..., help="Starting model: Qwen/Qwen3.5-4B or a merged SFT folder"),
    name: str = typer.Option(..., help="Run name: adapters in outputs/train/<name>"),
    task: str = RL_TASK,
    keep: Path = typer.Option(None, help="An rl-filter filter.jsonl: train on the filings it keeps"),
    sample: int = typer.Option(None, help="A seeded sample of N train filings (check stage)"),
    seed: int = typer.Option(0),
    steps: int = typer.Option(150),
    saves: int = typer.Option(6, help="Adapters saved, evenly spaced"),
    stack: str = typer.Option("trl", help="trl (rl environment) or unsloth (rl-unsloth)"),
    lr: float = typer.Option(1e-5),
    vllm_mem: float = typer.Option(None, help="vLLM's share of GPU memory (default: rl.VLLM_MEM)"),
) -> None:
    """GRPO with the filing reward; adapters in outputs/train/<name>. (rl or rl-unsloth environment)"""
    from filing_figures import reward
    from filing_figures.rl import RLConfig, read_filter
    from filing_figures.rl import train_rl as run

    recs = _records("train", sample=sample, seed=seed, task=task)
    if keep:
        kept = set(reward.keep(read_filter(keep)))
        recs = [r for r in recs if r["id"] in kept]
    cfg = RLConfig(init=init, out=OUT / "train" / name, stack=stack, steps=steps, saves=saves, lr=lr, vllm_mem=vllm_mem)
    if len(recs) < cfg.prompts_per_step:
        raise typer.BadParameter(f"{len(recs)} filings to train on; a step needs {cfg.prompts_per_step}")
    meta = run(cfg, recs)
    shown = ("steps", "train_filings", "seconds_per_step", "gpu_used_gib_max", "dropped_too_long")
    typer.echo({k: meta[k] for k in shown})


@app.command("check-report")
def check_report(
    grpo_steps: int = typer.Option(150, help="GRPO steps planned for the real run"),
    price: float = typer.Option(None, help="The pod's $/hour, for the cost"),
) -> None:
    """Go/no-go for GRPO and the hours of sft2 and grpo, from the v2 check stage's runs."""
    from filing_figures.check_report import report, samples

    lines, ok = report(OUT, grpo_steps, price)
    for line in [*lines, "", *samples(OUT)]:
        typer.echo(line)
    raise typer.Exit(0 if ok else 1)


@app.command()
def merge(
    adapter: Path = typer.Option(..., help="A checkpoint folder of a `train` run"),
    out: Path = typer.Option(..., help="Folder for the merged bf16 model"),
) -> None:
    """Merge an adapter into its base model's weights (bf16), ready for vLLM. (train environment)"""
    from filing_figures.train import merge as run_merge

    typer.echo(run_merge(adapter, out))


@app.command()
def checkpoints(name: str = typer.Option(..., help="Run name of a `train` run")) -> None:
    """Print a training run's checkpoint folders, in step order (for the stage scripts)."""
    from filing_figures.train import checkpoints as find

    for p in find(OUT / "train" / name):
        typer.echo(p.as_posix())


def _summary(split: str, run: str, task: str = "v1") -> dict | None:
    import json

    p = _scores_path(split, run, task)
    return json.loads(p.read_text(encoding="utf-8"))["summary"] if p.exists() else None


@app.command("pick-base")
def pick_base(models: str = typer.Option(",".join(CANDIDATE_TOKENIZERS), help="Comma-separated candidates")) -> None:
    """The bake-off's winner from the validation scores of runs base-<model> (rule: picks.py)."""
    from filing_figures.picks import pick_base as choose

    scores = {m: s for m in models.split(",") if (s := _summary("valid", f"base-{short_name(m)}"))}
    winner = choose(scores)
    rows = {m: round(100 * s["overall"]["value"], 1) for m, s in scores.items()}
    _write_json(OUT / "bakeoff" / "winner.json", {"winner": winner, "overall_pct": rows})
    typer.echo(f"scores {rows}")
    typer.echo(winner)


@app.command("pick-checkpoint")
def pick_checkpoint(
    name: str = typer.Option(..., help="Run name of a `train` or `train-rl` run"),
    task: str = typer.Option("v1", help="v1 picks on valid; v2 on valid and ood-valid pooled"),
) -> None:
    """The best checkpoint on validation (runs <name>-step<N>); writes outputs/train/<name>/best.json."""
    from filing_figures.picks import pick_checkpoint as choose
    from filing_figures.picks import pooled
    from filing_figures.train import checkpoints as find

    paths = {int(p.name.split("-")[1]): p for p in find(OUT / "train" / name)}
    scores = {}
    for step in paths:
        parts = [_summary(s, f"{name}-step{step}", task) for s in PICK_SPLITS[task]]
        if all(parts):  # a step missing one split's score isn't compared
            scores[step] = parts[0] if len(parts) == 1 else pooled(parts)
    step = choose(scores)
    best = {
        "step": step,
        "checkpoint": paths[step].as_posix(),
        "run": f"{name}-step{step}",
        "splits": list(PICK_SPLITS[task]),
        "overall_pct": {s: round(100 * v["overall"]["value"], 2) for s, v in sorted(scores.items())},
    }
    if task == "v1":  # v1's best.json keeps its shape
        del best["splits"]
    _write_json(OUT / "train" / name / "best.json", best)
    typer.echo(best)


@app.command()
def quantize(
    model: Path = typer.Option(..., help="A merged model folder"),
    scheme: str = typer.Option(..., help="fp8, w4a16 or nvfp4"),
    out: Path = typer.Option(None, help="Default: outputs/quant/<model>-<scheme>"),
    calib: int = typer.Option(256, help="Training filings for calibration (w4a16, nvfp4)"),
    seed: int = typer.Option(0),
) -> None:
    """Quantize with llm-compressor. (quant environment)"""
    from filing_figures.quantize import quantize as run_quantize

    out = out or OUT / "quant" / f"{model.name}-{scheme}"
    typer.echo(run_quantize(model, scheme, out, _records("train", calib, seed)))


@app.command()
def bench(
    model: str = typer.Option(..., help="HF repo id or a local model folder"),
    run: str = typer.Option(..., help="Name for the results file"),
    gpu_price: float = typer.Option(None, help="GPU $/hour, for the cost per 1,000 filings"),
    port: int = typer.Option(8000),
    quick: bool = typer.Option(False, help="Two small levels only (the check stage)"),
) -> None:
    """Speed of a served model: filings/s, latency, time to first token. (serve environment)"""
    from filing_figures import bench as speed
    from filing_figures.vllm_server import running

    with running(model, run, OUT / "logs" / "vllm" / f"bench-{run}.log", port=port) as (url, stats):
        levels = speed.run(url, run, _records("train"), levels=((1, 4), (8, 16)) if quick else speed.LEVELS)
    best = max(lv["filings_per_s"] or 0 for lv in levels)
    result = {"model": model, "gpu": _gpu_name(), "vllm": stats, "levels": levels, "best_filings_per_s": best}
    if gpu_price and best:
        result |= {
            "gpu_price_per_hour": gpu_price,
            "cost_per_1000_filings": round(speed.cost_per_1000(gpu_price, best), 4),
        }
    _write_json(OUT / "bench" / f"{run}.json", result)
    typer.echo(
        f"{run}: best {best} filings/s"
        + (f", ${result['cost_per_1000_filings']} per 1,000" if "cost_per_1000_filings" in result else "")
    )


@app.command()
def upload(
    paths: list[Path] = typer.Argument(..., help="Files or folders, uploaded at the same paths"),
    message: str = typer.Option("Pod results", help="Commit message"),
) -> None:
    """Upload results to the private runs repo on Hugging Face."""
    from filing_figures.runs import RUNS_REPO
    from filing_figures.runs import upload as push_paths

    done = push_paths(paths, message)
    typer.echo(f"uploaded to {RUNS_REPO}: {done}")


@app.command("fetch-runs")
def fetch_runs(patterns: list[str] = typer.Argument(..., help="Glob patterns, e.g. 'outputs/train/ft-x/*'")) -> None:
    """Download results from the private runs repo into this folder."""
    from filing_figures.runs import download as fetch

    typer.echo(fetch(patterns))


if __name__ == "__main__":
    app()
