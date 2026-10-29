"""The `figures` command line."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, add_completion=False)
DATA = Path("data")


@app.callback()
def main() -> None:
    """FilingFigures: six figures from SEC 10-K filings, scored against XBRL."""


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


if __name__ == "__main__":
    app()
