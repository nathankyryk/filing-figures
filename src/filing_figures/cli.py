"""The `figures` command line."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, add_completion=False)
DATA = Path("data")


OUT = Path("outputs")
HANDCHECK = Path("labels") / "handcheck.csv"
HF_DATASET = "kyryk/filing-figures"


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


def _filings(limit: int | None, tickers: str | None, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "universe" / "filings.csv", dtype={"accession": str})
    splits_path = DATA / "universe" / "splits.csv"
    if splits_path.exists():
        df = df.merge(pd.read_csv(splits_path)[["cik", "split"]], on="cik", how="left")
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
    df = _filings(limit, tickers, split)
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


def _build_one(row: dict) -> dict:
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
        return build_record(row, cf, html)
    except Exception as e:  # noqa: BLE001 - one bad filing must not stop the build
        return {"id": f"{row['ticker']}-{row['fiscal_year']}", "accession": row["accession"], "error": repr(e)}


@app.command()
def build(
    limit: int = typer.Option(None, help="Only the first N filings"),
    split: str = typer.Option(None, help="Only these splits, e.g. train or train,valid"),
    tickers: str = typer.Option(None, help="Comma-separated tickers"),
    workers: int = typer.Option(4, help="Processes"),
    out: Path = typer.Option(DATA / "build" / "records.jsonl", help="Output JSONL"),
) -> None:
    """Labels + model input + coverage for every cached filing, then a coverage summary."""
    import json
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor

    from filing_figures.labels import FIELD_NAMES

    rows = _filings(limit, tickers, split).to_dict("records")
    out.parent.mkdir(parents=True, exist_ok=True)
    errors, flag_counts = [], Counter()
    usable, total, in_input, labelled = Counter(), Counter(), Counter(), Counter()  # keyed by (split, field)
    with (
        ProcessPoolExecutor(workers) as pool,
        out.open("w", encoding="utf-8") as f,
        typer.progressbar(pool.map(_build_one, rows, chunksize=4), length=len(rows), label="filings") as bar,
    ):
        for rec in bar:
            f.write(json.dumps(rec) + "\n")
            if "error" in rec:
                errors.append(rec)
                continue
            s = rec.get("split") or "-"
            total[s] += 1
            usable[s] += rec["usable"]
            for name in FIELD_NAMES:
                flag_counts.update(f"{name}:{x}" for x in rec["flags"][name])
                labelled[s, name] += rec["gold"][name] is not None or "not_in_input" in rec["flags"][name]
                in_input[s, name] += rec["gold"][name] is not None
    typer.echo(f"\n{len(rows)} filings, {len(errors)} errors -> {out}")
    typer.echo("Field coverage: labelled values that are inside the model input")
    splits = sorted(total, key=lambda s: ("train", "valid", "test").index(s) if s in ("train", "valid", "test") else 9)
    typer.echo(f"  {'':20s}" + "".join(f"{s:>16s}" for s in splits))
    for name in [*FIELD_NAMES, "ALL"]:
        cells = []
        for s in splits:
            n_in = sum(in_input[s, f] for f in FIELD_NAMES) if name == "ALL" else in_input[s, name]
            n_lab = sum(labelled[s, f] for f in FIELD_NAMES) if name == "ALL" else labelled[s, name]
            cells.append(f"{n_in:>6d}/{n_lab:<5d}{100 * n_in / max(n_lab, 1):4.1f}%")
        typer.echo(f"  {name:20s}" + "".join(f"{c:>16s}" for c in cells))
    typer.echo(f"  {'usable filings':20s}" + "".join(f"{f'{usable[s]}/{total[s]}':>16s}" for s in splits))
    typer.echo("flags: " + ", ".join(f"{k}={v}" for k, v in sorted(flag_counts.items())))
    for e in errors[:10]:
        typer.echo(f"  ERROR {e['id']}: {e['error']}")


@app.command()
def export(
    records: Path = typer.Option(DATA / "build" / "records.jsonl", help="Output of `build`"),
    out: Path = typer.Option(DATA / "export", help="Folder for train/valid/test.jsonl"),
) -> None:
    """Write the usable records of each split, with the hand-check (labels/handcheck.csv) applied."""
    from filing_figures.dataset import export_splits

    for split, c in export_splits(records, out, HANDCHECK).items():
        typer.echo(
            f"{split:6s} {c['exported']:4d} of {c['filings']:4d} filings exported ({c['not_usable']} not usable), "
            f"{c['scored_fields']} scored fields, {c['handchecked_fields']} hand-checked"
        )


@app.command()
def push(
    repo: str = typer.Option(HF_DATASET, help="Hugging Face dataset repo"),
    folder: Path = typer.Option(DATA / "export"),
    message: str = typer.Option("Update the dataset", help="Commit message"),
) -> None:
    """Upload the export to Hugging Face (created private; publishing is a separate step)."""
    from filing_figures.dataset import push_dataset

    typer.echo(push_dataset(folder, repo, os.environ.get("HF_TOKEN"), message))


def _records(split: str, sample: int | None = None, seed: int = 0, ids: str | None = None) -> list[dict]:
    import random

    from filing_figures.dataset import read_jsonl

    recs = read_jsonl(DATA / "export" / f"{split}.jsonl")
    if ids:
        keep = set(ids.split(","))
        recs = [r for r in recs if r["id"] in keep]
    if sample and sample < len(recs):
        recs = sorted(random.Random(seed).sample(recs, sample), key=lambda r: r["id"])
    return recs


def _preds_path(split: str, run: str) -> Path:
    return OUT / "preds" / split / f"{run}.jsonl"


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
    final: bool = typer.Option(False, help="Allow the test split; the run is logged"),
) -> None:
    """Ask a model about each filing (outputs/preds/<split>/<run>.jsonl), then score it."""
    import json
    from datetime import UTC, datetime

    from filing_figures.client import Config
    from filing_figures.client import predict as run_predict

    if split == "test" and not final:
        raise typer.BadParameter("the test split is for the final runs only: pass --final")
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
    recs = _records(split, sample, seed, ids)
    out = _preds_path(split, run)
    if split == "test":
        log = OUT / "logs" / "test_runs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "run": run, "n": len(recs)}
            f.write(json.dumps(entry | {"config": cfg.__dict__}) + "\n")
    with typer.progressbar(length=len(recs), label=run) as bar:
        stats = run_predict(recs, cfg, out, on_result=lambda _: bar.update(1))
    typer.echo(f"{stats} -> {out}")
    score(split=split, run=run, sample=sample, seed=seed, ids=ids)


@app.command()
def score(
    run: str = typer.Option(..., help="Run name"),
    split: str = typer.Option("valid", help="train, valid or test"),
    sample: int = typer.Option(None, help="The same seeded sample the run used"),
    seed: int = typer.Option(0, help="Sampling seed"),
    ids: str = typer.Option(None, help="Comma-separated record ids"),
) -> None:
    """Score a run: field accuracy with 95% CIs (bootstrap over companies) and error categories."""
    import json

    from filing_figures.evaluate import examples, load_predictions, summarize, table
    from filing_figures.evaluate import score as score_rows

    recs = _records(split, sample, seed, ids)
    preds = load_predictions(_preds_path(split, run))
    missing = [r["id"] for r in recs if r["id"] not in preds]
    if missing:
        typer.echo(f"WARNING: {len(missing)} of {len(recs)} filings have no prediction (scored as no_response)")
    rows = score_rows(recs, preds)
    summary = summarize(rows)
    typer.echo(f"{run} on {split}: " + table(summary))
    path = OUT / "scores" / split / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "examples": examples(rows)}, indent=1), encoding="utf-8")


@app.command()
def compare(
    a: str = typer.Option(..., help="Run A"),
    b: str = typer.Option(..., help="Run B"),
    split: str = typer.Option("valid", help="train, valid or test"),
) -> None:
    """Overall accuracy of B minus A on the filings both runs answered (paired company bootstrap)."""
    from filing_figures.evaluate import compare as compare_rows
    from filing_figures.evaluate import load_predictions
    from filing_figures.evaluate import score as score_rows

    pa, pb = load_predictions(_preds_path(split, a)), load_predictions(_preds_path(split, b))
    recs = [r for r in _records(split) if r["id"] in pa and r["id"] in pb]
    gap = compare_rows(score_rows(recs, pa), score_rows(recs, pb))
    typer.echo(
        f"{b} - {a} on {len(recs)} {split} filings: {100 * gap['diff']:+.1f} points "
        f"[{100 * gap['lo']:+.1f}, {100 * gap['hi']:+.1f}], P(B not better) = {gap['p_b_not_better']:.3f}"
    )


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


@app.command()
def review(
    port: int = typer.Option(7862, help="Port on 127.0.0.1"),
    target: int = typer.Option(100, help="Filings in the pool"),
    n_random: int = typer.Option(40, help="Of those, a random sample: its fix rate is the answer key's error rate"),
    new_pool: bool = typer.Option(False, "--new-pool", help="Redraw labels/handcheck_pool.csv"),
    report: bool = typer.Option(False, "--report", help="Print progress and exit"),
) -> None:
    """The hand-check page for the test split's answer key (writes labels/handcheck.csv)."""
    from filing_figures.dataset import read_handcheck, read_jsonl
    from filing_figures.review import Review, build_pool, read_pool, serve, summary, write_pool

    # The XBRL key as built, before any hand-check, so the page shows both side by side
    recs = [
        r
        for r in read_jsonl(DATA / "build" / "records.jsonl")
        if "error" not in r and r.get("split") == "test" and r["usable"]
    ]
    if new_pool or not HANDCHECK_POOL.exists():
        write_pool(build_pool(recs, target=target, n_random=n_random), HANDCHECK_POOL)
    pool = read_pool(HANDCHECK_POOL)
    s = summary(pool, read_handcheck(HANDCHECK))
    typer.echo(
        f"{s['filings_done']} of {s['filings']} filings done; verdicts {dict(s['verdicts'])}; "
        f"random sample: {s['random_changed']} of {s['random_fields']} checked fields changed"
    )
    if report:
        return
    server = serve(Review(recs, pool, HANDCHECK), port)
    typer.echo(f"hand-check at http://127.0.0.1:{port}/ ; Ctrl+C to stop, then `figures export`")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


@app.command()
def stub(
    split: str = typer.Option("valid", help="Whose filings it recognises"),
    mode: str = typer.Option("gold", help="gold, prior or garbage"),
    port: int = typer.Option(8001, help="Port on 127.0.0.1"),
) -> None:
    """A stand-in model server for testing the harness (see stub.py)."""
    from filing_figures.stub import StubServer

    server = StubServer(_records(split), mode, port=port)
    typer.echo(f"stub ({mode}) at {server.url}; Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
