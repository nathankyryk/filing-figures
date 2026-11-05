"""The --task plumbing: v1 keeps every path it had; v2 gets its own folders and its own pick rule."""

import json
from pathlib import Path

import pytest
import typer

from filing_figures import cli, picks, runs


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    monkeypatch.setattr(cli, "OUT", tmp_path / "outputs")
    return tmp_path


def test_v1_paths_are_unchanged(roots):
    assert cli._preds_path("valid", "r") == roots / "outputs" / "preds" / "valid" / "r.jsonl"
    assert cli._scores_path("valid", "r") == roots / "outputs" / "scores" / "valid" / "r.json"
    assert cli._preds_path("valid", "r", "v1") == cli._preds_path("valid", "r")


def test_v2_paths_get_a_task_folder(roots):
    assert cli._preds_path("ood-valid", "r", "v2") == roots / "outputs" / "preds" / "v2" / "ood-valid" / "r.jsonl"
    assert cli._scores_path("valid", "r", "v2") == roots / "outputs" / "scores" / "v2" / "valid" / "r.json"


def write_split(folder: Path, split: str, ids: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{split}.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids), encoding="utf-8")


def test_records_read_the_task_export(roots):
    write_split(roots / "data" / "export", "valid", ["V1-A"])
    write_split(roots / "data" / "export" / "v2", "valid", ["V2-A", "V2-B"])
    assert [r["id"] for r in cli._records("valid")] == ["V1-A"]
    assert [r["id"] for r in cli._records("valid", task="v2")] == ["V2-A", "V2-B"]


@pytest.mark.parametrize("split", ["test", "ood-test"])
def test_final_splits_need_final(split):
    with pytest.raises(typer.BadParameter):
        cli._check_split(split, final=False)
    cli._check_split(split, final=True)
    cli._check_split("ood-valid", final=False)


def summary(value: float, n: int) -> dict:
    return {"overall": {"value": value, "lo": value, "hi": value, "n": n}}


def test_pooled_is_correct_over_scored_fields():
    p = picks.pooled([summary(0.9, 100), summary(0.5, 300)])
    assert p["overall"]["n"] == 400 and p["overall"]["value"] == pytest.approx((90 + 150) / 400)


def test_v2_pick_pools_valid_and_ood_valid(roots):
    run = roots / "outputs" / "train" / "grpo"
    for step in (10, 20):
        (run / f"checkpoint-{step}").mkdir(parents=True)
        (run / f"checkpoint-{step}" / "adapter_config.json").write_text("{}", encoding="utf-8")
    # step 10 wins on valid alone, step 20 on the pool (ood-valid has more fields here)
    scores = {
        ("valid", 10): summary(0.95, 100),
        ("ood-valid", 10): summary(0.80, 300),
        ("valid", 20): summary(0.90, 100),
        ("ood-valid", 20): summary(0.85, 300),
    }
    for (split, step), s in scores.items():
        path = cli._scores_path(split, f"grpo-step{step}", "v2")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": s}), encoding="utf-8")
    cli.pick_checkpoint(name="grpo", task="v2")
    best = json.loads((run / "best.json").read_text(encoding="utf-8"))
    assert best["step"] == 20 and best["splits"] == ["valid", "ood-valid"]
    assert best["overall_pct"] == {"10": 83.75, "20": 86.25}


def test_fetch_data_uses_the_task_dataset(roots, monkeypatch):
    calls = []

    def fake_fetch(out, repo):
        calls.append((out, repo))
        return []

    monkeypatch.setattr(runs, "fetch_dataset", fake_fetch)
    cli.fetch_data(task="v2")
    cli.fetch_data(task="v1")
    assert calls == [
        (roots / "data" / "export" / "v2", "kyryk/filing-figures-v2"),
        (roots / "data" / "export", "kyryk/filing-figures"),
    ]
