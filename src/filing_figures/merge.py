"""Merge a LoRA adapter into its base model, working on the weight files themselves.

The base checkpoint is copied as is (config, tokenizer, chat template, processor files, every
weight shard), and each weight the adapter trained gets W + scale x B @ A, computed in float32
and stored in the weight's own dtype. The result has the base model's exact layout and
architecture, so vLLM serves it the same way as the base: no class conversions, and no runtime
LoRA (which vLLM doesn't apply reliably to Qwen3.5's hybrid layers).

Adapter keys name a module the way the training library built the model (Unsloth may load a
vision-language checkpoint as its text model alone), so each one is matched to the base
checkpoint's weight by its path from `layers.` on, preferring the language model's copy.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

_LORA = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")
_NOT_TEXT = ("visual", "vision", "audio", "mtp")


def lora_modules(adapter_keys: list[str]) -> dict[str, dict[str, str]]:
    """{module path: {"A": key, "B": key}} for every LoRA pair in an adapter file."""
    out: dict[str, dict[str, str]] = {}
    for key in adapter_keys:
        m = _LORA.match(key)
        if not m:
            raise ValueError(f"not a LoRA weight: {key} (only lora_A/lora_B adapters are supported)")
        out.setdefault(m["module"], {})[m["ab"]] = key
    broken = [k for k, v in out.items() if set(v) != {"A", "B"}]
    if broken:
        raise ValueError(f"LoRA pairs missing A or B: {broken[:3]}")
    return out


def match_base_key(module: str, base_keys: set[str]) -> str:
    """The base checkpoint weight a LoRA module updates."""
    i = module.find("layers.")
    if i < 0:
        raise ValueError(f"no `layers.` in module path {module}")
    tail = module[i:] + ".weight"
    found = [k for k in base_keys if k == tail or k.endswith("." + tail)]
    if len(found) > 1:
        found = [k for k in found if not any(x in k for x in _NOT_TEXT)]
    if len(found) > 1:
        found = [k for k in found if "language_model" in k] or found
    if len(found) != 1:
        raise ValueError(f"{module}: {len(found)} matching base weights {found[:3]}")
    return found[0]


def scale_of(cfg: dict) -> float:
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def merge_adapter(base: str, adapter_dir: Path, out_dir: Path) -> dict:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("modules_to_save"):
        raise ValueError("DoRA and modules_to_save adapters are not supported")
    scale = scale_of(cfg)
    adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
    modules = lora_modules(list(adapter))

    src = Path(snapshot_download(base))
    shards = sorted(src.glob("*.safetensors"))
    where: dict[str, Path] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            where.update(dict.fromkeys(f.keys(), shard))
    targets = {match_base_key(m, set(where)): ab for m, ab in modules.items()}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for p in src.iterdir():  # everything but the weights, unchanged
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p.resolve(), out_dir / p.name)
    merged = 0
    for shard in shards:
        tensors = load_file(str(shard))
        for key in [k for k in tensors if k in targets]:
            ab = targets[key]
            a, b = adapter[ab["A"]].float(), adapter[ab["B"]].float()
            w = tensors[key]
            if tuple(w.shape) != (b.shape[0], a.shape[1]):
                raise ValueError(f"{key}: weight {tuple(w.shape)} vs LoRA {(b.shape[0], a.shape[1])}")
            tensors[key] = (w.float() + scale * (b @ a)).to(w.dtype)
            merged += 1
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
        del tensors
    if merged != len(targets):
        raise ValueError(f"merged {merged} of {len(targets)} LoRA weights")
    info = {"base": base, "adapter": str(adapter_dir), "scale": scale, "merged_weights": merged}
    (out_dir / "filing_figures.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
