"""Speed of a served model on real filings: filings per second, latency and time to first token
at several concurrency levels (pod only; any OpenAI-compatible server).

Each level gets its own filings (training filings, seeded shuffle), so no level can reuse
another level's cached prompts; the shared system prompt is cached, as it would be in use.
Speed needs no answer key, and keeping the test filings out keeps them for the one final run.

Cost per 1,000 filings = GPU $/hour / (filings/s x 3,600) x 1,000, at the busiest level.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time

from filing_figures import prompts

# (requests in flight, filings): one at a time is the latency view, the last level is a batch job
LEVELS = ((1, 16), (8, 64), (32, 192), (256, 512))


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_level(concurrency: int, wall_s: float, results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    lat = [r["latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    out_tok = sum(r["completion_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "filings": len(results),
        "errors": len(results) - len(ok),
        "wall_s": round(wall_s, 2),
        "filings_per_s": round(len(ok) / wall_s, 3) if wall_s else None,
        "output_tokens_per_s": round(out_tok / wall_s, 1) if wall_s else None,
        "input_tokens_per_s": round(in_tok / wall_s, 1) if wall_s else None,
        "latency_p50_s": _r(percentile(lat, 0.5)),
        "latency_p95_s": _r(percentile(lat, 0.95)),
        "ttft_p50_s": _r(percentile(ttft, 0.5)),
        "ttft_p95_s": _r(percentile(ttft, 0.95)),
        "mean_output_tokens": round(statistics.mean(r["completion_tokens"] for r in ok), 1) if ok else None,
    }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def cost_per_1000(gpu_price_per_hour: float, filings_per_s: float) -> float:
    return gpu_price_per_hour / (filings_per_s * 3600) * 1000


async def _one(client, model: str, rec: dict, sem: asyncio.Semaphore, max_tokens: int) -> dict:
    async with sem:
        t0 = time.perf_counter()
        ttft, usage = None, None
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=prompts.messages(rec["input"]),
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                    ttft = time.perf_counter() - t0
                if chunk.usage:
                    usage = chunk.usage
        except Exception as e:  # noqa: BLE001 - counted as an error in the level's summary
            return {"error": repr(e), "latency_s": time.perf_counter() - t0, "ttft_s": None}
        return {
            "latency_s": time.perf_counter() - t0,
            "ttft_s": ttft,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }


async def _level(base_url: str, model: str, recs: list[dict], concurrency: int, max_tokens: int) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="none", timeout=1800, max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(_one(client, model, r, sem, max_tokens) for r in recs))
    wall = time.perf_counter() - t0
    await client.close()
    return summarize_level(concurrency, wall, list(results))


def run(
    base_url: str, model: str, records: list[dict], levels=LEVELS, max_tokens: int = 1024, seed: int = 0
) -> list[dict]:
    """One summary per level. A few warm-up requests first (CUDA graphs, caches)."""
    pool = sorted(records, key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    need = sum(n for _, n in levels) + 4
    if len(pool) < need:
        raise ValueError(f"need {need} filings for these levels, have {len(pool)}")
    asyncio.run(_level(base_url, model, pool[:4], 4, max_tokens))
    out, start = [], 4
    for concurrency, n in levels:
        summary = asyncio.run(_level(base_url, model, pool[start : start + n], concurrency, max_tokens))
        start += n
        print(summary, flush=True)
        out.append(summary)
    return out
