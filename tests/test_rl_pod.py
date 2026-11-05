"""Pod only (`rl` environment; skipped where torch and peft aren't installed): the TRL stack's
model, LoRA targets and weight sync, checked on CPU before any GPU run.

The full-size Qwen3.5-4B is built on the meta device (names and shapes, no memory), so its names
are compared with the real checkpoint's index. The numbers are checked on a tiny module with the
same naming.
"""

import dataclasses
import json
from importlib.metadata import version
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from test_reward import rec_v2

from filing_figures import rl, sft

REPO = "Qwen/Qwen3.5-4B"


def index_keys() -> set[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "model.safetensors.index.json")
    return set(json.loads(Path(path).read_text(encoding="utf-8"))["weight_map"])


def lora_config(**kw):
    from peft import LoraConfig

    return LoraConfig(lora_dropout=0.0, bias="none", target_modules=rl.TARGET_REGEX, **kw)


@pytest.fixture(scope="module")
def full_meta():
    """Qwen3.5-4B as the TRL stack loads it (the vision-language class), with GRPO's LoRA."""
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(REPO)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    return cfg, get_peft_model(model, lora_config(r=32, lora_alpha=64, task_type="CAUSAL_LM"))


def test_lora_lands_on_the_language_models_projections_only(full_meta):
    from peft.tuners.lora import LoraLayer

    cfg, model = full_meta
    names = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    layers = cfg.text_config.layer_types
    # q/k/v/o in each full-attention layer, gate/up/down in every layer: what SFT trains
    assert len(names) == 4 * layers.count("full_attention") + 3 * len(layers)
    assert all(".language_model.layers." in n for n in names)


def test_sync_sends_every_language_model_weight_under_its_checkpoint_name(full_meta):
    _, model = full_meta
    sent = [name for name, _ in rl.merged_params(model)]
    assert len(sent) == len(set(sent))
    keys = index_keys()
    assert any(k.startswith(rl.SKIP_SYNC) for k in keys)  # the vision tower's prefix is right
    assert set(sent) == {k for k in keys if k.startswith("model.language_model.")}  # mtp: not in HF or vLLM


def test_grpo_adapters_merge_onto_the_checkpoint(full_meta):
    from peft import get_peft_model_state_dict

    from filing_figures.merge import lora_modules, match_base_key

    _, model = full_meta
    mods = lora_modules(list(get_peft_model_state_dict(model)))
    base = index_keys()
    targets = {match_base_key(m, base) for m in mods}
    assert len(targets) == len(mods) and all(t.startswith("model.language_model.layers.") for t in targets)


class Tiny(torch.nn.Module):
    """The Qwen3.5 naming in miniature: one layer of the language model, plus a vision part."""

    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(16, 32, bias=False)
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(32, 16, bias=False)
        lm = torch.nn.Module()
        lm.layers = torch.nn.ModuleList([layer])
        lm.norm = torch.nn.LayerNorm(16)
        self.model = torch.nn.Module()
        self.model.language_model = lm
        self.model.visual = torch.nn.Linear(4, 4)


def tiny_peft():
    from peft import get_peft_model

    torch.manual_seed(0)
    model = get_peft_model(Tiny().to(torch.bfloat16), lora_config(r=4, lora_alpha=8))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)  # as after training: B is no longer zero
    return model


def down_proj(model):
    return model.base_model.model.model.language_model.layers[0].mlp.down_proj


def test_sync_sends_w_plus_delta_and_never_changes_the_base():
    model = tiny_peft()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(3):
        sent = dict(rl.merged_params(model))
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert set(sent) == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.norm.weight",
        "model.language_model.norm.bias",
    }
    m = down_proj(model)
    delta = m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float()
    want = (m.base_layer.weight.float() + 2.0 * delta).to(torch.bfloat16)  # scale = alpha / r
    got = sent["model.language_model.layers.0.mlp.down_proj.weight"]
    assert got.dtype == torch.bfloat16 and torch.equal(got, want) and not torch.equal(got, m.base_layer.weight)


def test_in_place_merge_and_unmerge_drift_the_bf16_base():
    """Why lora_sync exists: TRL 1.17.1 syncs by merging and unmerging in place."""
    model = tiny_peft()
    w = down_proj(model).base_layer.weight
    before = w.detach().clone()
    for _ in range(3):
        model.merge_adapter()
        model.unmerge_adapter()
    assert not torch.equal(w, before)


def test_trl_tokenizes_the_prompt_as_vllm_serves_it():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO)
    rec = rec_v2()
    rec["input"] = "## Cover page\n\nExample Inc. | Form 10-K"
    prompt = sft.chat_prompt(tok, rec)
    # TRL tokenizes text prompts with the defaults; vLLM's chat endpoint adds no special tokens
    assert tok(text=[prompt])["input_ids"][0] == tok(prompt, add_special_tokens=False)["input_ids"]


def test_trl_internals_that_lora_sync_replaces():
    vg = pytest.importorskip("trl.generation.vllm_generation")
    assert version("trl") == "1.17.1"
    assert callable(vg.VLLMGeneration._iter_named_params)
    assert callable(vg.VLLMGeneration._fix_param_name_to_vllm)


def test_trls_vllm_engine_starts_language_model_only(monkeypatch):
    vg = pytest.importorskip("trl.generation.vllm_generation")
    from vllm.engine.arg_utils import EngineArgs

    assert "language_model_only" in {f.name for f in dataclasses.fields(EngineArgs)}
    seen = {}
    monkeypatch.setattr(vg, "LLM", lambda *a, **k: seen.update(k))
    rl.vllm_language_model_only(REPO)
    vg.LLM(model=REPO)
    assert seen == {"model": REPO, "language_model_only": True}


def test_grpo_config_takes_every_argument(tmp_path):
    trl = pytest.importorskip("trl")
    kwargs, unsupported = rl.supported(rl.grpo_args(rl.RLConfig(init=REPO, out=tmp_path)), trl.GRPOConfig)
    assert unsupported == []
    if torch.cuda.is_available():
        assert trl.GRPOConfig(**kwargs).generation_batch_size == 64  # 8 filings x 8 answers per round


def test_merge_onto_a_local_folder(tmp_path):
    """GRPO from the SFT model: its checkpoints merge onto the merged SFT folder, not a repo."""
    from safetensors.torch import load_file, save_file

    from filing_figures.merge import merge_adapter

    key = "model.language_model.layers.0.mlp.down_proj"
    base, adapter = tmp_path / "sft-merged", tmp_path / "checkpoint-25"
    base.mkdir()
    adapter.mkdir()
    w = torch.randn(16, 32).to(torch.bfloat16)
    a, b = torch.randn(4, 32), torch.randn(16, 4)
    save_file({key + ".weight": w, "model.language_model.norm.weight": torch.ones(16)}, str(base / "model.safetensors"))
    (base / "config.json").write_text("{}", encoding="utf-8")
    lora = {f"base_model.model.{key}.lora_A.weight": a, f"base_model.model.{key}.lora_B.weight": b}
    save_file(lora, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")
    info = merge_adapter(str(base), adapter, tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))[key + ".weight"]
    assert info["merged_weights"] == 1 and (tmp_path / "out" / "config.json").exists()
    assert torch.equal(got, (w.float() + 2.0 * (b @ a)).to(torch.bfloat16))
