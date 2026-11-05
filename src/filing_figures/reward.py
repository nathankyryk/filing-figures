"""The GRPO reward: the share of a filing's scored fields that one answer gets right.

It is the test metric applied to a single answer: the same parser (`evaluate.parse_answer`), the
same field reader (`field_prediction`) and the same rule for "correct" (`categorize`: the value
within tolerance AND the exact period end). An answer with no JSON object scores 0, as does one cut
off at the length limit. There is no bonus for format: every model here already writes valid JSON.
Fields the answer key leaves blank don't count, as in scoring.

The model never sees the answer key, which is the company's own XBRL, so the only way to a high
reward is the right figures.

The difficulty filter: GRPO learns from the differences between the answers in a group (several
samples for one filing). A filing whose samples all score the same teaches nothing, so filings are
sampled first and only those with a spread are kept for training.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from filing_figures.evaluate import categorize, field_prediction, parse_answer


def scored_fields(rec: dict) -> list[str]:
    return [name for name, gold in rec["gold"].items() if gold is not None]


def field_results(text: str | None, rec: dict) -> dict[str, bool]:
    """{field: correct} for every scored field of the record."""
    obj, _ = parse_answer(text, rec.get("task", "v1"))
    out = {}
    for name in scored_fields(rec):
        pred = None if obj is None else field_prediction(obj, name)
        out[name] = obj is not None and categorize(pred, rec["gold"][name], rec["context_facts"], name) == "correct"
    return out


def score_completion(text: str | None, rec: dict) -> float:
    """The reward of one answer: correct fields / scored fields (0 for a record with none)."""
    results = field_results(text, rec)
    return sum(results.values()) / len(results) if results else 0.0


def completion_text(completion) -> str:
    """TRL hands over a string for a plain-text prompt, a message list for a chat prompt."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"] if completion else ""


def make_reward(records: list[dict]) -> Callable[..., list[float]]:
    """A reward function for TRL's GRPOTrainer. TRL passes each dataset column (here `id`) as a
    keyword argument, one value per completion."""
    by_id = {r["id"]: r for r in records}

    def filing_reward(prompts, completions, id, **kwargs) -> list[float]:  # `id`: the dataset column's name
        return [score_completion(completion_text(c), by_id[rid]) for c, rid in zip(completions, id, strict=True)]

    return filing_reward


def group_stats(scores: list[float]) -> dict:
    """One filing's group of sampled answers."""
    return {
        "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores),
        "all_right": all(s == 1.0 for s in scores),
        "all_wrong": all(s == 0.0 for s in scores),
    }


def keep(groups: dict[str, list[float]]) -> list[str]:
    """The filings worth training on: their samples don't all score the same."""
    return sorted(rid for rid, scores in groups.items() if len(set(scores)) > 1)


def filter_summary(groups: dict[str, list[float]]) -> dict:
    """What the filter found, for the logs and the README."""
    stats = [group_stats(s) for s in groups.values()]
    n = len(stats)
    return {
        "filings": n,
        "kept": len(keep(groups)),
        "all_right": sum(s["all_right"] for s in stats),
        "all_wrong": sum(s["all_wrong"] for s in stats),
        "same_score": sum(s["std"] == 0 for s in stats),
        "mean_reward": round(statistics.fmean(s["mean"] for s in stats), 4) if n else None,
    }
