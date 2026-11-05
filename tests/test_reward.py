"""The GRPO reward is the test metric applied to one answer; the difficulty filter."""

import json

import pytest
from test_evaluate import END, FACTS, record
from test_task_v2 import v2_answer, v2_record

from filing_figures import reward
from filing_figures.evaluate import score
from filing_figures.stub import answer

OCF_FACT = {
    "tag": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "unit": "USD",
    "start": "2023-10-01",
    "end": END,
    "val": 118_254e6,
}


def rec_v2(rid="EXM-2024"):
    rec = v2_record()
    rec["id"], rec["period_end"], rec["context_facts"] = rid, END, [*FACTS, OCF_FACT]
    return rec


def test_gold_scores_one_and_no_json_scores_zero():
    rec = rec_v2()
    assert reward.score_completion(answer(rec, "gold"), rec) == 1.0
    assert reward.score_completion(answer(rec, "garbage"), rec) == 0.0
    assert reward.score_completion("", rec) == 0.0 and reward.score_completion(None, rec) == 0.0


def test_an_answer_cut_off_at_the_length_limit_scores_zero():
    rec = rec_v2()
    text = answer(rec, "gold")
    assert reward.score_completion(text[: len(text) // 2], rec) == 0.0
    assert reward.score_completion("<think>\nthe revenue is", rec) == 0.0


def test_blank_fields_dont_count():
    rec = rec_v2()
    rec["gold"]["fcf"] = None
    one_wrong = v2_answer(capex={"value": -9_447e6})  # copied the parentheses
    assert reward.score_completion(json.dumps(one_wrong), rec) == pytest.approx(10 / 11)


def variants(rec: dict) -> list[str]:
    gold = json.loads(answer(rec, "gold"))
    off = json.loads(json.dumps(gold))
    off["revenue"]["value"] /= 1e6  # scale
    off["net_income"]["period_end"] = "2024-09-30"  # date
    del off["total_assets"]  # missing
    return [
        answer(rec, "gold"),
        answer(rec, "prior"),
        answer(rec, "garbage"),
        json.dumps(off),
        "```json\n" + json.dumps(off) + "\n```",
        "<think>\nlet me see\n</think>\n\n" + answer(rec, "gold"),
        '{"revenue": "not a figure"}',
    ]


@pytest.mark.parametrize("make", [rec_v2, lambda: {**record(), "period_end": END}])
def test_reward_times_fields_is_the_scorers_correct_count(make):
    """For every answer: the reward is exactly what `figures score` counts for that filing."""
    rec = make()
    for text in variants(rec):
        rows = score([rec], {rec["id"]: {"id": rec["id"], "raw": text, "error": None}})
        assert reward.score_completion(text, rec) * len(rows) == pytest.approx(sum(r["correct"] for r in rows))
        assert reward.field_results(text, rec) == {r["field"]: r["correct"] for r in rows}


def test_trl_reward_function_reads_both_completion_shapes():
    a, b = rec_v2("A-2024"), rec_v2("B-2024")
    fn = reward.make_reward([a, b])
    texts = [answer(a, "gold"), answer(b, "garbage")]
    got = fn(prompts=["p", "p"], completions=texts, id=["A-2024", "B-2024"], completion_ids=[[1], [2]])
    assert got == [1.0, 0.0]
    chat = [[{"role": "assistant", "content": t}] for t in texts]
    assert fn(prompts=[[], []], completions=chat, id=["A-2024", "B-2024"]) == [1.0, 0.0]


def test_filter_keeps_filings_whose_samples_differ():
    groups = {"solved": [1.0] * 8, "hopeless": [0.0] * 8, "stuck": [0.75] * 8, "mixed": [1.0, 0.75] * 4}
    assert reward.keep(groups) == ["mixed"]
    s = reward.group_stats(groups["mixed"])
    assert s["mean"] == pytest.approx(0.875) and s["std"] == pytest.approx(0.125) and not s["all_right"]
    assert reward.filter_summary(groups) == {
        "filings": 4,
        "kept": 1,
        "all_right": 1,
        "all_wrong": 1,
        "same_score": 3,
        "mean_reward": pytest.approx((1 + 0 + 0.75 + 0.875) / 4, abs=1e-4),
    }
