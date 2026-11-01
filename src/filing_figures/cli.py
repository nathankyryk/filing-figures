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

    from filing_figures.client import Config

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
    _predict_and_score(cfg, split, run, sample, seed, ids)


def _predict_and_score(cfg, split: str, run: str, sample: int | None, seed: int, ids: str | None) -> None:
    import json
    from datetime import UTC, datetime

    from filing_figures.client import predict as run_predict

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
def doctor() -> None:
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
    for split in ("train", "valid", "test"):
        p = DATA / "export" / f"{split}.jsonl"
        n = sum(1 for line in p.open(encoding="utf-8") if line.strip()) if p.exists() else 0
        check(f"data: {split}", n > 0, f"{n} filings in {p}")
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            who = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
            check("HF token", who == "kyryk", f"account {who}")
        except Exception as e:  # noqa: BLE001 - reported, never printed with the token
            check("HF token", False, type(e).__name__)
    else:
        check("HF token", False, "HF_TOKEN is not set")
    for pkg in ("unsloth", "llmcompressor", "vllm"):
        found = importlib.util.find_spec(pkg) is not None
        check(f"package {pkg}", True, "installed" if found else "not in this environment", critical=False)
    free = shutil.disk_usage(Path.cwd()).free / 1e9
    check("disk", free > 60, f"{free:.0f} GB free")
    raise typer.Exit(1 if failures else 0)


@app.command("fetch-data")
def fetch_data() -> None:
    """Download the exported splits from the Hugging Face dataset (as last pushed)."""
    from filing_figures.runs import fetch_dataset

    for p in fetch_dataset(DATA / "export"):
        typer.echo(f"{p}: {sum(1 for line in p.open(encoding='utf-8') if line.strip())} filings")


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
    final: bool = typer.Option(False, help="Allow the test split; the run is logged"),
) -> None:
    """Serve a model with vLLM, ask it about each filing, score it, stop the server. (serve environment)"""
    import time

    from filing_figures.client import Config
    from filing_figures.vllm_server import MAX_MODEL_LEN, THINKING_MAX_MODEL_LEN, running

    if split == "test" and not final:
        raise typer.BadParameter("the test split is for the final runs only: pass --final")
    t0 = time.time()
    max_len = THINKING_MAX_MODEL_LEN if thinking else MAX_MODEL_LEN
    with running(
        model, run, OUT / "logs" / "vllm" / f"{run}.log", port=port, max_model_len=max_len, thinking=thinking
    ) as (
        url,
        stats,
    ):
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
        _predict_and_score(cfg, split, run, sample, seed, None)
        info = {"model": model, "split": split, "gpu": _gpu_name(), "startup_s": round(startup_s, 1)}
        _write_json(OUT / "serve" / f"{run}.json", info | {"predict_s": round(time.time() - t1, 1), **stats})


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
    four_bit: bool | None = typer.Option(
        None, "--four-bit/--bf16", help="Default: QLoRA where supported (train.FOUR_BIT)"
    ),
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
        load_in_4bit=four_bit,
    )
    meta = run_train(cfg, _records("train"), _records("valid"))
    typer.echo({k: meta[k] for k in ("steps", "seconds_per_step", "peak_gpu_gib", "train_loss", "dropped_too_long")})


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


def _summary(split: str, run: str) -> dict | None:
    import json

    p = OUT / "scores" / split / f"{run}.json"
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
def pick_checkpoint(name: str = typer.Option(..., help="Run name of a `train` run")) -> None:
    """The best checkpoint on validation (runs <name>-step<N>); writes outputs/train/<name>/best.json."""
    from filing_figures.picks import pick_checkpoint as choose
    from filing_figures.train import checkpoints as find

    paths = {int(p.name.split("-")[1]): p for p in find(OUT / "train" / name)}
    scores = {step: s for step in paths if (s := _summary("valid", f"{name}-step{step}"))}
    step = choose(scores)
    best = {
        "step": step,
        "checkpoint": paths[step].as_posix(),
        "run": f"{name}-step{step}",
        "overall_pct": {s: round(100 * v["overall"]["value"], 2) for s, v in sorted(scores.items())},
    }
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
