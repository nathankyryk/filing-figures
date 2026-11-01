"""Quantize a merged model with llm-compressor (pod only, `quant` environment).

- fp8: FP8 weights and activations, dynamic per-token activation scales. No calibration.
- w4a16: 4-bit integer weights (GPTQ, group size 128), 16-bit activations. Calibrated.
- nvfp4: 4-bit float weights and activations (NVIDIA's FP4 format). Needs a Blackwell GPU to
  run its fast kernels. Calibrated (activation scales).

Calibration uses training filings with their gold answers: the text the model actually reads
while answering. Left in bf16, as in llm-compressor's Qwen3.5 examples: the output layer, any
vision or audio encoder, and Qwen3.5's linear-attention (Gated DeltaNet) layers, whose fused
projections the 4-bit kernels don't support. The same list is used for every format, so the
formats differ only in how the remaining layers are stored.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from filing_figures import sft

SCHEMES = ("fp8", "w4a16", "nvfp4")
CALIBRATED = {"w4a16", "nvfp4"}
IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*vision.*",
    "re:.*audio.*",
    "re:.*multi_modal_projector.*",
    "re:.*linear_attn.*",
]
EXTRA_FILES = (
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def _model_class(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    archs = cfg.get("architectures") or []
    multimodal = any(a.endswith("ForConditionalGeneration") for a in archs)
    return AutoModelForImageTextToText if multimodal else AutoModelForCausalLM


def quantize(model_dir: Path, scheme: str, out_dir: Path, calib_recs: list[dict], max_seq_length: int = 8192) -> dict:
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoTokenizer

    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = _model_class(cfg).from_pretrained(str(model_dir), dtype="auto")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    elif scheme == "w4a16":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)
    else:
        recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=IGNORE)
    kwargs = {}
    if scheme in CALIBRATED:
        eot = sft.end_of_turn(tok)
        texts = [sft.calibration_text(tok, r, eot) for r in calib_recs]
        enc = [tok(t, add_special_tokens=False, truncation=True, max_length=max_seq_length) for t in texts]
        ds = Dataset.from_dict(
            {"input_ids": [e["input_ids"] for e in enc], "attention_mask": [e["attention_mask"] for e in enc]}
        )
        # Our tokenizer as the processor: otherwise llm-compressor loads the model's full
        # AutoProcessor, which for Qwen3.5 includes a video processor that needs torchvision.
        kwargs = {
            "dataset": ds,
            "processor": tok,
            "max_seq_length": max_seq_length,
            "num_calibration_samples": len(enc),
        }
    oneshot(model=model, recipe=recipe, **kwargs)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save_pretrained(str(out_dir), save_compressed=True)
    tok.save_pretrained(str(out_dir))
    for name in EXTRA_FILES:  # keep the exact chat template and processor files of the source
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, out_dir / name)
    if (model_dir / "generation_config.json").exists():
        shutil.copy2(model_dir / "generation_config.json", out_dir / "generation_config.json")
    info = {"source": str(model_dir), "scheme": scheme, "calibration_filings": len(calib_recs), "ignore": IGNORE}
    (out_dir / "filing_figures.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
