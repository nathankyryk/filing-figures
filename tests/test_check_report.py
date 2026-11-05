"""The v2 check stage's go/no-go lines and its estimate of the real runs."""

import json

import pytest

from filing_figures import check_report as cr


def step(gap=0.02, ratio=1.0, zero=0.25, reward=0.8, **extra):
    return {
        "sampling/sampling_logp_difference/mean": gap,
        "sampling/importance_sampling_ratio/mean": ratio,
        "frac_reward_zero_std": zero,
        "reward": reward,
        "completions/clipped_ratio": 0.0,
        **extra,
    }


def test_gates_pass_on_a_healthy_run():
    history = [step(), step(gap=0.04, ratio=0.98, reward=0.85), {"train_runtime": 100.0}]
    assert [passed for _, passed, _ in cr.gates(history)] == [True, True, True]


@pytest.mark.parametrize(
    ("bad", "which"),
    [({"gap": 0.3}, 0), ({"ratio": 0.7}, 1), ({"ratio": 1.4}, 1), ({"zero": 0.9}, 2)],
)
def test_each_gate_fails_on_its_own_signal(bad, which):
    passed = [p for _, p, _ in cr.gates([step(**bad)])]
    assert passed[which] is False and sum(passed) == 2


def test_missing_metrics_fail_the_gates():
    assert not any(p for _, p, _ in cr.gates([{"loss": 0.0}]))


def test_estimate_adds_training_filtering_and_scoring():
    est = cr.estimate(sft_s_per_step=20.0, grpo_s_per_step=60.0, filter_s_per_filing=1.0, grpo_steps=150, price=2.5)
    # SFT: 59 steps/epoch x 3 x 20 s = 0.98 h, + 7 scorings x 5 min = 0.58 h
    assert est["sft2_hours"] == pytest.approx(0.98 + 7 * 5 / 60, abs=0.01)
    # GRPO: 934 filings x 1 s = 0.26 h, 150 x 60 s = 2.5 h, 6 scorings = 0.5 h
    assert est["grpo_hours"] == pytest.approx(0.26 + 2.5 + 0.5, abs=0.01)
    assert est["grpo_usd"] == pytest.approx(est["grpo_hours"] * 2.5, abs=0.02)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_report_reads_the_check_folder(tmp_path):
    meta = {"seconds_per_step": 60.0, "gpu_used_gib_max": 61.2, "peak_gpu_gib": 30.1, "gpu": "NVIDIA H100 80GB HBM3"}
    write(tmp_path / "train" / "check2-grpo" / "train_meta.json", meta)
    write(tmp_path / "train" / "check2-grpo" / "log_history.json", [step(), step(reward=0.9)])
    write(tmp_path / "train" / "check2-sft" / "train_meta.json", {"seconds_per_step": 20.0})
    filt = {"kept": 20, "filings": 32, "mean_reward": 0.81, "seconds_per_filing": 1.0}
    write(tmp_path / "rl-filter" / "check2-filter" / "filter_summary.json", filt)
    lines, ok = cr.report(tmp_path, grpo_steps=150, price=2.5)
    assert ok and sum(line.startswith("GO") for line in lines) == 4
    assert any("reward 0.800 -> 0.900" in line for line in lines)
    assert any("grpo (150 steps) ~3.26 h" in line for line in lines)


def test_report_is_no_go_without_a_finished_run(tmp_path):
    lines, ok = cr.report(tmp_path)
    assert not ok and lines[0].startswith("NO-GO") and "GRPO ran" in lines[0]
