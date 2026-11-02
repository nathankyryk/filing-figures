"""The data behind the results viewer (space/, a static Hugging Face Space).

`build` writes space/data/results.json (every test filing's answer key and each model's six
answers, scored) and space/data/inputs/<id>.txt (the model input, loaded when a filing is
opened). Everything comes from the one test run's saved predictions; nothing is re-asked.
"""

from __future__ import annotations

import json
from pathlib import Path

from filing_figures.evaluate import field_prediction, load_predictions, parse_answer, score
from filing_figures.labels import FIELD_NAMES

# Run name: (label, how its cost per 1,000 filings is measured)
MODELS = {
    "rules": ("Rules (no model)", "free"),
    "base-qwen3.5-4b": ("Qwen3.5-4B, no fine-tuning", "bench"),
    "ft-qwen3.5-4b-bf16": ("Qwen3.5-4B, fine-tuned (bf16)", "bench"),
    "ft-qwen3.5-4b-nvfp4": ("Qwen3.5-4B, fine-tuned (NVFP4)", "bench"),
    "ref-qwen3.5-397b": ("Qwen3.5-397B", "api"),
    "ref-opus-5.5": ("Claude Opus 5.5", "api"),
}


def cost_per_1000(run: str, how: str, preds_path: Path, bench_dir: Path, n_filings: int) -> float:
    if how == "free":
        return 0.0
    if how == "bench":
        return json.loads((bench_dir / f"{run}.json").read_text(encoding="utf-8"))["cost_per_1000_filings"]
    total = 0.0  # API: every request billed, retries included
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += float((json.loads(line).get("usage") or {}).get("cost") or 0.0)
    return round(total / n_filings * 1000, 2)


def build(records: list[dict], outputs: Path, out_dir: Path, names: dict[str, str]) -> dict:
    """Write the viewer's data files; returns a short summary."""
    preds_dir, scores_dir = outputs / "preds" / "test", outputs / "scores" / "test"
    models, answers = [], {}
    for run, (label, how) in MODELS.items():
        preds = load_predictions(preds_dir / f"{run}.jsonl")
        summary = json.loads((scores_dir / f"{run}.json").read_text(encoding="utf-8"))["summary"]
        models.append(
            {
                "run": run,
                "label": label,
                "overall": summary["overall"],
                "all_right": summary["all_right"],
                "fields": {k: v["value"] for k, v in summary["fields"].items()},
                "categories": summary["categories"],
                "cost_per_1000": cost_per_1000(run, how, preds_dir / f"{run}.jsonl", outputs / "bench", len(records)),
            }
        )
        rows = {(r["id"], r["field"]): r for r in score(records, preds)}
        for rec in records:
            p = preds.get(rec["id"])
            obj = parse_answer(p.get("raw"))[0] if p and not p.get("error") else None
            for name in FIELD_NAMES:
                row = rows.get((rec["id"], name))
                pred = row["pred"] if row else (field_prediction(obj, name) if obj else None)
                answers[rec["id"], name, run] = {
                    "value": pred["value"] if pred else None,
                    "period_end": pred["period_end"] if pred else None,
                    "category": row["category"] if row else "not_scored",
                }

    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    filings = []
    for rec in records:
        fields = {}
        for name in FIELD_NAMES:
            g = rec["gold"][name]
            fields[name] = {
                "gold": None
                if g is None
                else {k: g[k] for k in ("value", "scale", "period_end", "displayed")} | {"tag": g["tag"]},
                "flags": rec["flags"][name],
                "answers": {run: answers[rec["id"], name, run] for run in MODELS},
            }
        folder = rec["accession"].replace("-", "")
        filings.append(
            {
                "id": rec["id"],
                "ticker": rec["ticker"],
                "name": names.get(rec["ticker"], rec["ticker"]),
                "sector": rec["sector"],
                "fiscal_year": rec["fiscal_year"],
                "period_end": rec["period_end"],
                "filing_date": rec["filing_date"],
                "url": f"https://www.sec.gov/Archives/edgar/data/{rec['cik']}/{folder}/{rec['primary_document']}",
                "fields": fields,
            }
        )
        (out_dir / "inputs" / f"{rec['id']}.txt").write_text(rec["input"], encoding="utf-8")
    data = {"fields": list(FIELD_NAMES), "models": models, "filings": filings}
    (out_dir / "results.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return {"filings": len(filings), "models": len(models), "bytes": (out_dir / "results.json").stat().st_size}
