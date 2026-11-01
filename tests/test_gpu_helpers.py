"""The GPU-stage helpers that run without a GPU: training examples, the two picking rules,
benchmark statistics and the vLLM command line."""

import json

import pytest
from test_evaluate import record

from filing_figures import bench, picks, sft
from filing_figures.stub import answer
from filing_figures.vllm_server import read_stats, serve_args


class CharTokenizer:
    """One token per character, with a Qwen-style chat template: enough to check the masks."""

    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return text

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def blank_record():
    rec = record()
    rec["input"] = "## Cover page\n\nfiling text"
    rec["gold"]["shares_outstanding"] = None  # several share classes: not scored
    return rec


def test_answer_text_is_the_stub_gold_answer_and_marks_blanks():
    rec = blank_record()
    text, hidden = sft.answer_text(rec)
    assert text == answer(rec, "gold")
    assert json.loads(text)["revenue"]["value"] == rec["gold"]["revenue"]["value"]
    assert [text[a:b] for a, b in hidden] == [json.dumps(dict.fromkeys(sft.PARTS))]


def test_end_of_turn_is_the_template_tail():
    assert sft.end_of_turn(CharTokenizer()) == "<|im_end|>"


def test_encode_trains_on_the_answer_only_and_skips_blank_values():
    tok, rec = CharTokenizer(), blank_record()
    row = sft.encode(tok, rec, "<|im_end|>")
    prompt = sft.chat_prompt(tok, rec)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # thinking off, as served
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    assert all(x == -100 for x in row["labels"][: len(prompt)])
    trained = "".join(chr(x) for x in row["labels"] if x != -100)
    text, _ = sft.answer_text(rec)
    assert trained == text.replace(json.dumps(dict.fromkeys(sft.PARTS)), "") + "<|im_end|>"
    assert '"shares_outstanding": ' in trained  # the key is still trained, only its value is not


def _summary(value, lo):
    return {"overall": {"value": value, "lo": lo, "hi": value + 0.01}}


def test_pick_base_takes_the_smallest_model_inside_the_best_interval():
    scores = {
        "Qwen/Qwen3.5-9B": _summary(0.965, 0.942),
        "Qwen/Qwen3.5-4B": _summary(0.950, 0.920),  # inside 9B's interval: tie, and smaller
        "Qwen/Qwen3-8B": _summary(0.900, 0.870),
    }
    assert picks.pick_base(scores) == "Qwen/Qwen3.5-4B"
    scores["Qwen/Qwen3.5-4B"] = _summary(0.930, 0.900)  # below 0.942: the 9B wins
    assert picks.pick_base(scores) == "Qwen/Qwen3.5-9B"


def test_pick_checkpoint_prefers_the_earlier_of_equals():
    assert picks.pick_checkpoint({60: _summary(0.97, 0), 30: _summary(0.95, 0), 90: _summary(0.97, 0)}) == 60
    with pytest.raises(ValueError):
        picks.pick_checkpoint({})


def test_bench_summary_and_cost():
    results = [
        {"latency_s": 1.0, "ttft_s": 0.2, "prompt_tokens": 5000, "completion_tokens": 300},
        {"latency_s": 3.0, "ttft_s": 0.4, "prompt_tokens": 5000, "completion_tokens": 320},
        {"error": "timeout", "latency_s": 9.0, "ttft_s": None},
    ]
    s = bench.summarize_level(8, 2.0, results)
    assert s["filings"] == 3 and s["errors"] == 1 and s["filings_per_s"] == 1.0
    assert s["latency_p50_s"] == 2.0 and s["ttft_p95_s"] == pytest.approx(0.39)
    assert s["output_tokens_per_s"] == 310.0
    # $0.90/h at 10 filings/s: 36,000 filings an hour, so $0.025 per 1,000
    assert bench.cost_per_1000(0.90, 10.0) == pytest.approx(0.025)


def test_vllm_args_for_text_only_and_thinking():
    qwen35 = {"model_type": "qwen3_5", "vision_config": {}}
    args = serve_args("Qwen/Qwen3.5-4B", "base-qwen3.5-4b", qwen35)
    assert "--language-model-only" in args and "--reasoning-parser" not in args
    assert json.loads(args[args.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert args[args.index("--max-model-len") + 1] == "12288"
    thinking = serve_args("Qwen/Qwen3.5-4B", "x", qwen35, thinking=True, max_model_len=32768)
    assert thinking[thinking.index("--reasoning-parser") + 1] == "qwen3"
    assert "--language-model-only" not in serve_args("Qwen/Qwen3-8B", "x", {"model_type": "qwen3"})


def test_vllm_startup_stats():
    log = "INFO Model loading took 8.61 GiB and 12.3 seconds\nINFO GPU KV cache size: 141,312 tokens\n"
    assert read_stats(log) == {"weights_gib": 8.61, "kv_cache_tokens": 141312}


def test_merge_matches_adapter_modules_to_language_model_weights():
    from filing_figures.merge import lora_modules, match_base_key, scale_of

    base = {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",  # same path from `layers.` on
        "mtp.layers.0.mlp.down_proj.weight",
    }
    adapter = [
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",  # trained on the text model alone
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",  # on the full model
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    mods = lora_modules(adapter)
    assert {m: match_base_key(m, base) for m in mods} == {
        "model.layers.0.mlp.down_proj": "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj": "model.language_model.layers.0.self_attn.q_proj.weight",
    }
    with pytest.raises(ValueError):
        lora_modules(adapter[:1])  # A without B
    with pytest.raises(ValueError):
        match_base_key("model.layers.1.mlp.down_proj", base)  # nothing to update
    assert scale_of({"r": 16, "lora_alpha": 32}) == 2.0
    assert scale_of({"r": 16, "lora_alpha": 32, "use_rslora": True}) == 8.0


def test_bench_end_to_end_against_the_streaming_stub():
    import threading

    from filing_figures.stub import StubServer

    recs = [{**record(f"EXM-{i}", i), "input": f"filing text {i}"} for i in range(30)]
    server = StubServer(recs, "gold", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        levels = bench.run(server.url, "stub", recs, levels=((1, 4), (8, 16)))
    finally:
        server.shutdown()
    assert [lv["concurrency"] for lv in levels] == [1, 8]
    assert all(lv["errors"] == 0 and lv["filings_per_s"] > 0 for lv in levels)
    assert levels[1]["filings"] == 16 and levels[0]["ttft_p50_s"] is not None
    assert levels[0]["mean_output_tokens"] == len(answer(recs[0], "gold")) // 4
