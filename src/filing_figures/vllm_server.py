"""Start a vLLM server for one model, wait until it answers, and always shut it down (pod only,
`serve` environment). The rest of the code talks to it like any other OpenAI-compatible endpoint.

Settings, from vLLM's Qwen3.5 recipe and our token counts:
- `--language-model-only` for models that ship a vision (or audio) encoder: the encoder isn't
  loaded, which leaves more memory for the KV cache. Our inputs are text.
- `--max-model-len 12288`: the longest prompt is ~8.8K tokens and answers ~320 tokens.
  Thinking runs get 32768.
- Thinking is off by default, as for every model in this project. A thinking run adds vLLM's
  reasoning parser, so the answer comes back without the reasoning text.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

MAX_MODEL_LEN = 12_288
THINKING_MAX_MODEL_LEN = 32_768


def model_config(model: str) -> dict:
    """config.json of a local model folder or a Hugging Face repo."""
    path = Path(model) / "config.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model, "config.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def is_multimodal(cfg: dict) -> bool:
    return any(k in cfg for k in ("vision_config", "audio_config"))


def serve_args(
    model: str,
    name: str,
    cfg: dict,
    *,
    port: int = 8000,
    max_model_len: int = MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.90,
    thinking: bool = False,
    extra: tuple[str, ...] = (),
) -> list[str]:
    exe = shutil.which("vllm")
    args = [exe] if exe else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    args += [
        "serve",
        model,
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        "0",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": thinking}),
    ]
    if is_multimodal(cfg):
        args.append("--language-model-only")
    model_type = str(cfg.get("model_type", ""))
    parser = "qwen3" if model_type.startswith("qwen3") else "gemma4" if model_type.startswith("gemma4") else None
    if thinking and parser:
        args += ["--reasoning-parser", parser]
    return args + list(extra)


def read_stats(log_text: str) -> dict:
    """Memory figures vLLM logs at startup: weights in GiB and KV cache capacity in tokens."""
    out = {}
    if m := re.search(r"Model loading took ([\d.]+) GiB", log_text):
        out["weights_gib"] = float(m.group(1))
    if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", log_text):
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    return out


@contextmanager
def running(
    model: str,
    name: str,
    log_path: Path,
    *,
    port: int = 8000,
    max_model_len: int = MAX_MODEL_LEN,
    thinking: bool = False,
    extra: tuple[str, ...] = (),
    startup_timeout: float = 1800.0,
):
    """Yields (base_url, stats). Raises with the end of vLLM's log if the server doesn't come up."""
    cfg = model_config(model)
    args = serve_args(model, name, cfg, port=port, max_model_len=max_model_len, thinking=thinking, extra=extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    print("$ " + " ".join(args), flush=True)
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {proc.returncode}:\n{_tail(log_path)}")
            try:
                if httpx.get(base + "/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"vLLM not up after {startup_timeout:.0f} s:\n{_tail(log_path)}")
            time.sleep(5)
        stats = read_stats(log_path.read_text(encoding="utf-8", errors="replace"))
        print(f"vLLM up: {name} {stats}", flush=True)
        yield base + "/v1", stats
    finally:
        _stop(proc)
        log.close()


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def _tail(path: Path, n: int = 40) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])
