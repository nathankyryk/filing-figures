"""The two choices made on validation scores, written down before the runs.

- Base model: the best overall field accuracy. Any candidate whose accuracy is inside the best
  one's 95% interval counts as a tie, and the smallest such model wins (cheaper to serve).
- Checkpoint: the best overall field accuracy on all validation filings; a tie goes to the
  earlier checkpoint.
"""

from __future__ import annotations

# Smallest first, by the compute per token that serving costs. Gemma 4 E4B runs ~4B parameters
# per token (its per-layer embeddings are lookups), so it sits next to the 4B.
SIZE_ORDER = ("Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B")


def pick_base(summaries: dict[str, dict], size_order: tuple[str, ...] = SIZE_ORDER) -> str:
    """summaries: {model: score summary}. Models missing from size_order rank last."""
    if not summaries:
        raise ValueError("no bake-off scores")
    best = max(summaries, key=lambda m: summaries[m]["overall"]["value"])
    floor = summaries[best]["overall"]["lo"]
    tied = [m for m, s in summaries.items() if s["overall"]["value"] >= floor]
    rank = {m: i for i, m in enumerate(size_order)}
    return min(tied, key=lambda m: (rank.get(m, len(rank)), -summaries[m]["overall"]["value"]))


def pick_checkpoint(summaries: dict[int, dict]) -> int:
    """summaries: {step: score summary}."""
    if not summaries:
        raise ValueError("no checkpoint scores")
    return max(sorted(summaries), key=lambda s: summaries[s]["overall"]["value"])
