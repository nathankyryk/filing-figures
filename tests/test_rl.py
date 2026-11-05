"""GRPO's settings, dataset, logging and CLI guards: the parts of rl.py that run without a GPU."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass

import pytest
import typer
from test_gpu_helpers import CharTokenizer
from test_reward import rec_v2

from filing_figures import cli, reward, rl, sft
from filing_figures.stub import answer


def test_grpo_args_are_the_planned_recipe(tmp_path):
    a = rl.grpo_args(rl.RLConfig(init="Qwen/Qwen3.5-4B", out=tmp_path))
    # 8 filings x 8 answers, 2 answers per pass: 32 passes per optimizer step, one sampling round each
    assert a["num_generations"] == 8 and a["per_device_train_batch_size"] == 2
    assert a["gradient_accumulation_steps"] == a["steps_per_generation"] == 32 and a["num_iterations"] == 1
    assert (a["learning_rate"], a["lr_scheduler_type"], a["warmup_steps"]) == (1e-5, "constant_with_warmup", 8)
    assert (a["beta"], a["loss_type"], a["epsilon"], a["epsilon_high"]) == (0.0, "dapo", 0.2, 0.28)
    assert (a["temperature"], a["top_p"], a["top_k"]) == (1.0, 1.0, 0)
    assert a["max_completion_length"] == 1024 and a["vllm_max_model_length"] == 12288
    assert a["vllm_mode"] == "colocate" and a["vllm_enable_sleep_mode"] and a["vllm_gpu_memory_utilization"] == 0.3
    assert a["save_steps"] == 25 and a["save_only_model"]  # 150 steps, 6 saves
    assert not a["remove_unused_columns"]  # the reward reads the `id` column
    assert not a["mask_truncated_completions"]


def test_config_checks_and_per_stack_memory(tmp_path):
    assert rl.RLConfig(init="m", out=tmp_path, stack="unsloth").gpu_mem() == 0.9
    assert rl.RLConfig(init="m", out=tmp_path, vllm_mem=0.45).gpu_mem() == 0.45
    assert rl.RLConfig(init="m", out=tmp_path, steps=8, saves=1).save_every() == 8
    with pytest.raises(ValueError):
        rl.RLConfig(init="m", out=tmp_path, stack="axolotl")
    with pytest.raises(ValueError):
        rl.RLConfig(init="m", out=tmp_path, micro_batch=3)  # 64 answers don't split into 3s


def test_unsupported_arguments_are_reported_not_passed():
    @dataclass
    class OldConfig:
        beta: float = 0.0
        num_generations: int = 8

    kept, dropped = rl.supported({"beta": 0.0, "num_generations": 8, "vllm_max_model_length": 1}, OldConfig)
    assert kept == {"beta": 0.0, "num_generations": 8} and dropped == ["vllm_max_model_length"]


def test_lora_targets_the_language_model_projections_only():
    hit = re.compile(rl.TARGET_REGEX).fullmatch
    assert hit("model.language_model.layers.3.self_attn.q_proj")
    assert hit("model.language_model.layers.0.mlp.down_proj")
    assert not hit("model.language_model.layers.0.linear_attn.in_proj_qkv")  # same modules as SFT
    assert not hit("model.language_model.layers.0.linear_attn.out_proj")
    assert not hit("model.visual.blocks.0.mlp.linear_fc1")
    assert not hit("model.language_model.layers.0.mlp.down_proj.lora_A")


def test_dataset_is_the_inference_prompt_and_drops_long_filings():
    tok, a, b = CharTokenizer(), rec_v2("A-2024"), rec_v2("B-2024")
    a["input"], b["input"] = "## Cover page\n\nfiling text", "x" * 50_000
    rows, dropped = rl.rl_dataset(tok, [a, b], max_prompt_tokens=20_000)
    assert rows == [{"prompt": sft.chat_prompt(tok, a), "id": "A-2024"}] and dropped == ["B-2024"]
    assert rows[0]["prompt"].endswith("<think>\n\n</think>\n\n")  # thinking off, as served


def test_logged_reward_scores_logs_fields_and_samples(tmp_path):
    a, b = rec_v2("A-2024"), rec_v2("B-2024")
    path, metrics = tmp_path / "samples.jsonl", {}

    class State:
        global_step = 10

    fn = rl.logged_reward([a, b], path, every=10)
    texts = [answer(a, "gold"), answer(b, "prior"), answer(a, "garbage"), answer(b, "gold")]
    ids = ["A-2024", "B-2024", "A-2024", "B-2024"]
    got = fn(prompts=[""] * 4, completions=texts, id=ids, trainer_state=State(), log_metric=metrics.__setitem__)
    assert got == reward.make_reward([a, b])(prompts=[""] * 4, completions=texts, id=ids)
    assert metrics["fields/revenue"] == 0.5 and metrics["fields/fcf"] == 0.5  # right in the 2 gold answers
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["best"] * 3 + ["worst"] * 3 and rows[0]["reward"] == 1.0
    worst = {r["text"]: r for r in rows[3:]}
    assert worst[texts[2]]["reward"] == 0.0 and "revenue" in worst[texts[2]]["wrong"]  # the garbage answer
    State.global_step = 11  # off the sampling grid: nothing more written
    fn(prompts=[""], completions=texts[:1], id=ids[:1], trainer_state=State())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 6


def test_steady_seconds_skip_the_warm_up():
    assert rl.steady_seconds([90.0, 70.0, 60.0, 62.0, 58.0]) == 60.0
    assert rl.steady_seconds([90.0]) == 90.0 and rl.steady_seconds([]) is None


def write_filter(path, groups):
    path.write_text("".join(json.dumps({"id": i, "scores": s}) + "\n" for i, s in groups.items()), encoding="utf-8")


def test_filter_file_round_trip(tmp_path):
    groups = {"A-2024": [1.0] * 8, "B-2024": [0.5, 1.0] * 4}
    write_filter(tmp_path / "filter.jsonl", groups)
    assert rl.read_filter(tmp_path / "filter.jsonl") == groups
    assert reward.keep(rl.read_filter(tmp_path / "filter.jsonl")) == ["B-2024"]


@pytest.fixture
def train_split(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    monkeypatch.setattr(cli, "OUT", tmp_path / "outputs")
    folder = tmp_path / "data" / "export" / "v2"
    folder.mkdir(parents=True)
    recs = [rec_v2(f"C{i:02d}-2024") for i in range(12)]
    (folder / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    seen = {}

    def fake_run(cfg, recs):  # the run itself needs a GPU
        seen.update(cfg=cfg, ids=[r["id"] for r in recs])
        return defaultdict(int)

    monkeypatch.setattr(rl, "train_rl", fake_run)
    return tmp_path, seen


def test_train_rl_trains_on_the_kept_filings(train_split):
    root, seen = train_split
    mixed = {f"C{i:02d}-2024": [0.5, 1.0] * 4 for i in range(9)}
    write_filter(root / "filter.jsonl", {**mixed, "C09-2024": [1.0] * 8})
    cli.train_rl(init="m", name="g", task="v2", keep=root / "filter.jsonl", sample=None, seed=0, steps=8,
                 saves=1, stack="trl", lr=1e-5, vllm_mem=None)  # fmt: skip
    assert seen["ids"] == sorted(mixed) and seen["cfg"].out == root / "outputs" / "train" / "g"


def test_train_rl_needs_a_full_step_of_filings(train_split):
    root, seen = train_split
    write_filter(root / "filter.jsonl", {"C00-2024": [0.5, 1.0] * 4, "C01-2024": [0.0] * 8})
    with pytest.raises(typer.BadParameter):
        cli.train_rl(init="m", name="g", task="v2", keep=root / "filter.jsonl", sample=None, seed=0, steps=8,
                     saves=1, stack="trl", lr=1e-5, vllm_mem=None)  # fmt: skip
    assert not seen


def test_rl_filter_never_samples_the_final_splits(train_split):
    with pytest.raises(typer.BadParameter):
        cli.rl_filter(init="m", name="f", task="v2", split="ood-test", sample=None, seed=0, group=8, gpu_mem=0.9)


def test_sleep_mode_drops_only_expandable_segments(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rl.allow_sleep_mode()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in rl.os.environ
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")
    rl.allow_sleep_mode()
    assert rl.os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
