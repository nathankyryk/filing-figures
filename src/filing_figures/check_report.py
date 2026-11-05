"""The v2 check stage's verdict: go/no-go lines for GRPO, and the hours and dollars of the real runs.

The gates, read from the 8-step GRPO run's log:
- vLLM's and the trainer's log-probabilities of the sampled tokens agree: a mean gap under 0.1.
  A larger gap means the weights vLLM samples from aren't the ones being trained (a sync or
  naming problem).
- The importance-sampling ratio (trainer / vLLM probability of each sampled answer) averages
  within 0.9-1.1, for the same reason.
- Most groups have a spread: a group whose 8 answers all score the same contributes nothing.
- The run finished: no out-of-memory error with vLLM and the trainer sharing the GPU.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

LOGP_GAP_MAX = 0.1
IS_RATIO = (0.9, 1.1)
ZERO_STD_MAX = 0.5  # share of groups whose answers all score the same
TRAIN_FILINGS = 934  # v2's usable train filings
SFT_FILINGS_PER_STEP = 16
SCORE_MINUTES = 5  # merge + vLLM start + valid and ood-valid, per checkpoint (v1's measured, with room)


def mean_of(history: list[dict], key: str) -> float | None:
    vals = [h[key] for h in history if isinstance(h.get(key), int | float) and not math.isnan(h[key])]
    return statistics.fmean(vals) if vals else None


def _fmt(x: float | None, spec: str) -> str:
    return "missing" if x is None else format(x, spec)


def gates(history: list[dict]) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for each gate."""
    gap = mean_of(history, "sampling/sampling_logp_difference/mean")
    ratio = mean_of(history, "sampling/importance_sampling_ratio/mean")
    zero = mean_of(history, "frac_reward_zero_std")
    return [
        ("vLLM and the trainer agree", gap is not None and gap < LOGP_GAP_MAX, f"logprob gap {_fmt(gap, '.4f')} < 0.1"),
        (
            "importance ratio near 1",
            ratio is not None and IS_RATIO[0] <= ratio <= IS_RATIO[1],
            f"mean {_fmt(ratio, '.3f')} in 0.9-1.1",
        ),
        (
            "most groups have a spread",
            zero is not None and zero <= ZERO_STD_MAX,
            f"{_fmt(None if zero is None else 1 - zero, '.0%')} of groups",
        ),
    ]


def estimate(
    sft_s_per_step: float,
    grpo_s_per_step: float,
    filter_s_per_filing: float,
    grpo_steps: int = 150,
    epochs: float = 3.0,
    saves: int = 6,
    price: float | None = None,
) -> dict:
    """Hours (and dollars at `price` $/h) for sft2 and one grpo run, from the check's timings."""
    sft_steps = math.ceil(TRAIN_FILINGS / SFT_FILINGS_PER_STEP) * epochs
    score_h = SCORE_MINUTES / 60
    sft2 = sft_steps * sft_s_per_step / 3600 + (round(epochs * 2) + 1) * score_h  # checkpoints + base zero-shot
    grpo = TRAIN_FILINGS * filter_s_per_filing / 3600 + grpo_steps * grpo_s_per_step / 3600 + saves * score_h
    out = {"sft2_hours": round(sft2, 2), "grpo_hours": round(grpo, 2)}
    if price:
        out |= {"sft2_usd": round(sft2 * price, 2), "grpo_usd": round(grpo * price, 2)}
    return out


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def report(out: Path, grpo_steps: int = 150, price: float | None = None) -> tuple[list[str], bool]:
    """The report's lines for the check folder `out`, and whether every gate passed."""
    sft = _read(out / "train" / "check2-sft" / "train_meta.json")
    grpo = _read(out / "train" / "check2-grpo" / "train_meta.json")
    history = _read(out / "train" / "check2-grpo" / "log_history.json") or []
    filt = _read(out / "rl-filter" / "check2-filter" / "filter_summary.json")
    lines, ok = [], True

    def line(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"{'GO   ' if passed else 'NO-GO'}  {name:28s} {detail}")

    line("GRPO ran (no OOM)", grpo is not None, "" if grpo else "no train_meta.json: read the log")
    for name, passed, detail in gates(history):
        line(name, passed, detail)
    if grpo:
        rewards = [h["reward"] for h in history if "reward" in h]
        lines.append(
            f"       reward {rewards[0]:.3f} -> {rewards[-1]:.3f} over {len(rewards)} steps"
            if rewards
            else "       no reward logged"
        )
        clipped = mean_of(history, "completions/clipped_ratio")
        lines.append(f"       answers cut off at the cap: {_fmt(clipped, '.1%')}")
        lines.append(
            f"       {grpo['seconds_per_step']} s/step after warm-up, GPU {grpo['gpu_used_gib_max']} GiB used at most "
            f"(trainer {grpo['peak_gpu_gib']} GiB), {grpo['gpu']}"
        )
    if filt:
        lines.append(
            f"       filter: kept {filt['kept']} of {filt['filings']} filings, mean reward {filt['mean_reward']}, "
            f"{filt['seconds_per_filing']} s/filing"
        )
    if sft and grpo and filt:
        est = estimate(
            sft["seconds_per_step"], grpo["seconds_per_step"], filt["seconds_per_filing"], grpo_steps, price=price
        )
        cost = f" (${est['sft2_usd']} / ${est['grpo_usd']} at ${price}/h)" if price else ""
        lines.append(f"       sft2 ~{est['sft2_hours']} h, grpo ({grpo_steps} steps) ~{est['grpo_hours']} h{cost}")
    return lines, ok


def samples(out: Path, n_chars: int = 400) -> list[str]:
    """The logged best and worst answers of the check's GRPO run, shortened, for reading by eye."""
    path = out / "train" / "check2-grpo" / "samples.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        f"step {r['step']} {r['kind']} {r['id']} reward {r['reward']:.2f} wrong {r['wrong']}: {r['text'][:n_chars]!r}"
        for r in rows
    ]
