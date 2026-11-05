"""GRPO on the filing reward (pod only, `rl` environment; `rl-unsloth` for the fallback stack).

The method (DeepSeekMath §4.1, as TRL implements it): for each filing, sample a group of answers,
score each with the reward (reward.py: the test metric on one answer), and push up the answers
that beat their group's mean. There is no value model: the group's mean is the baseline.

Settings, and why:
- LoRA r32, alpha 64, on the same modules SFT trains (attention and MLP projections of the
  language model; the vision tower is untouched). lr 1e-5, constant after a short warm-up: LoRA
  wants about 10x the full fine-tuning learning rate (Thinking Machines, "LoRA Without Regret"),
  and DeepSeekMath's full fine-tune used 1e-6.
- 8 filings x 8 answers per step, one optimizer step per round of sampling, so training is
  on-policy. beta 0 (no KL term, TRL's default), loss "dapo" (each token weighted equally across
  the batch), clipping 0.2 / 0.28 (DAPO's asymmetric clip). Temperature 1.0, no top-p or top-k,
  so the sampled answers come from the model's own distribution.
- Answers are capped at 1,024 tokens (a full answer is ~630). One cut off at the cap has no
  complete JSON and scores 0.

The TRL stack: vLLM runs in the same process as the trainer ("colocate") and sleeps while the
trainer runs. Three things are set up here rather than left to TRL's defaults:
1. The trainer loads the checkpoint's own vision-language class, so its parameter names are the
   checkpoint's and vLLM's (`model.language_model.layers.N...`). Training the text-only class gives
   `model.layers.N...`, which vLLM's Qwen3.5 loader doesn't expect (vLLM issue #36275).
2. vLLM is started with `language_model_only=True`, as `figures serve` does, so it never builds
   the vision tower. TRL has no option for it, so its `LLM` constructor is wrapped.
3. The weight sync. Before each round of sampling TRL merges LoRA into the bf16 base weights in
   place, sends them, and unmerges, so the frozen base drifts a rounding error per step (TRL PR
   #7427, open in 1.17.1). `lora_sync` sends W + scale * B @ A computed in fp32 instead, and never
   writes it back.
TRL is pinned (==1.17.1) because 3 replaces one of its private methods; tests/test_rl_pod.py
checks it on the pod before any run.

vLLM's sampling and the trainer's forward pass differ slightly (kernels, bf16 rounding), so TRL
corrects each token by the ratio of the two probabilities (on by default). `check` logs both
(`sampling/sampling_logp_difference`, `sampling/importance_sampling_ratio`) and stops on a large
gap.

The difficulty filter (`sample_groups`) runs before training: 8 answers per train filing from
the starting model, kept only where the scores differ (reward.keep).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, fields
from importlib import metadata
from pathlib import Path

from filing_figures import reward, sft
from filing_figures.train import TARGET_MODULES
from filing_figures.vllm_server import MAX_MODEL_LEN, is_multimodal, model_config

STACKS = ("trl", "unsloth")
# PEFT matches a string target with re.fullmatch: the language model's projections only
TARGET_REGEX = r".*language_model.*\.(" + "|".join(TARGET_MODULES) + ")"
SKIP_SYNC = ("model.visual.",)  # not built by vLLM in language-model-only mode, and frozen anyway
VLLM_MEM = {"trl": 0.3, "unsloth": 0.9}  # Unsloth's standby mode shares the memory with training
PACKAGES = ("torch", "transformers", "trl", "peft", "vllm", "flash-linear-attention", "unsloth", "unsloth_zoo")


@dataclass
class RLConfig:
    init: str  # the starting model: Qwen/Qwen3.5-4B or a merged SFT folder
    out: Path
    stack: str = "trl"
    steps: int = 150
    rank: int = 32
    alpha: int = 64
    lr: float = 1e-5
    warmup: float = 0.05  # fraction of the steps
    prompts_per_step: int = 8
    group: int = 8  # answers per filing
    micro_batch: int = 2  # answers per forward/backward pass
    temperature: float = 1.0
    max_completion: int = 1024
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    saves: int = 6
    vllm_mem: float | None = None  # None: VLLM_MEM[stack]
    max_model_len: int = MAX_MODEL_LEN
    seed: int = 0
    log_samples_every: int = 10
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.stack not in STACKS:
            raise ValueError(f"stack must be one of {STACKS}, not {self.stack!r}")
        if (self.prompts_per_step * self.group) % self.micro_batch:
            raise ValueError("micro_batch must divide prompts_per_step * group")

    def gpu_mem(self) -> float:
        return VLLM_MEM[self.stack] if self.vllm_mem is None else self.vllm_mem

    def accum(self) -> int:
        """Forward/backward passes per optimizer step: one whole round of sampling."""
        return self.prompts_per_step * self.group // self.micro_batch

    def save_every(self) -> int:
        return max(1, math.ceil(self.steps / max(1, self.saves)))


def grpo_args(cfg: RLConfig) -> dict:
    """GRPOConfig's keyword arguments (see the module docstring for the choices)."""
    return {
        "output_dir": str(cfg.out),
        "max_steps": cfg.steps,
        "learning_rate": cfg.lr,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": max(1, round(cfg.warmup * cfg.steps)),
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "per_device_train_batch_size": cfg.micro_batch,
        "gradient_accumulation_steps": cfg.accum(),
        "steps_per_generation": cfg.accum(),  # one round of sampling per optimizer step: on-policy
        "num_iterations": 1,
        "num_generations": cfg.group,
        "temperature": cfg.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "max_completion_length": cfg.max_completion,
        "mask_truncated_completions": False,  # an answer cut off at the cap scores 0, and that's the lesson
        "beta": cfg.beta,
        "loss_type": cfg.loss_type,
        "epsilon": cfg.epsilon,
        "epsilon_high": cfg.epsilon_high,
        "scale_rewards": cfg.scale_rewards,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": cfg.gpu_mem(),
        "vllm_max_model_length": cfg.max_model_len,
        "vllm_enable_sleep_mode": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "disable_dropout": True,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": cfg.save_every(),
        "save_only_model": True,  # adapters only
        "report_to": "none",
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "remove_unused_columns": False,  # the reward needs the `id` column
        "shuffle_dataset": True,
        "log_completions": False,  # samples.jsonl instead
    }


def supported(kwargs: dict, config_cls) -> tuple[dict, list[str]]:
    """The arguments `config_cls` accepts, and the names it doesn't (Unsloth pins an older TRL)."""
    names = {f.name for f in fields(config_cls)}
    return {k: v for k, v in kwargs.items() if k in names}, sorted(set(kwargs) - names)


def rl_dataset(tokenizer, records: list[dict], max_prompt_tokens: int) -> tuple[list[dict], list[str]]:
    """Rows {prompt, id}: the exact inference prompt (thinking off) as plain text, which TRL
    tokenizes without a chat template. Filings whose prompt leaves no room for an answer are
    dropped and reported."""
    rows, dropped = [], []
    for rec in records:
        prompt = sft.chat_prompt(tokenizer, rec)
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        (dropped.append(rec["id"]) if n > max_prompt_tokens else rows.append({"prompt": prompt, "id": rec["id"]}))
    return rows, dropped


# ---- the TRL stack --------------------------------------------------------------------------


def merged_params(model, fix: Callable[[str], str] = lambda n: n, skip: tuple[str, ...] = SKIP_SYNC) -> Iterator:
    """(name, weight) for vLLM: every base weight under its checkpoint name, with each LoRA
    module's update added out of place in fp32. The model's own weights are never changed."""
    import torch
    from peft.tuners.lora import LoraLayer

    lora = {}  # the base weight's parameter name -> its LoRA module
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if module.merged or getattr(module, "fan_in_fan_out", False):
                raise RuntimeError(f"{name}: merged or fan_in_fan_out LoRA isn't supported here")
            if any(module.use_dora.values()):
                raise RuntimeError(f"{name}: DoRA isn't supported here")
            lora[name + ".base_layer.weight"] = module
    for name, param in model.named_parameters():
        if ".lora_" in name:  # LoRA's own A and B: folded into their base weight below
            continue
        clean = fix(name.removeprefix("base_model.model.").replace(".base_layer", ""))
        if clean.startswith(skip):
            continue
        module = lora.get(name)
        if module is None:
            yield clean, param.data
            continue
        with torch.no_grad():
            w = param.data.to(torch.float32, copy=True)  # a copy even if the weight is fp32
            for adapter in module.active_adapters:
                a = module.lora_A[adapter].weight.float()
                b = module.lora_B[adapter].weight.float()
                w += module.scaling[adapter] * (b @ a)
            merged = w.to(param.dtype)
        del w
        yield clean, merged


def lora_sync(trainer) -> None:
    """Swap TRL's in-place merge/unmerge weight sync for `merged_params` (one process only)."""
    gen = trainer.vllm_generation
    if trainer.accelerator.num_processes != 1:
        raise RuntimeError("lora_sync supports a single process")
    gen._iter_named_params = lambda: merged_params(gen.model, fix=gen._fix_param_name_to_vllm)


def vllm_language_model_only(init: str) -> None:
    """Make TRL start its colocated vLLM engine without the vision tower (see the docstring)."""
    if not is_multimodal(model_config(init)):
        return
    import trl.generation.vllm_generation as vg

    real = vg.LLM
    if getattr(real, "_language_model_only", False):
        return

    def LLM(*args, **kwargs):  # stands in for vllm.LLM
        kwargs.setdefault("language_model_only", True)
        return real(*args, **kwargs)

    LLM._language_model_only = True
    vg.LLM = LLM


def _load_trl(cfg: RLConfig):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(cfg.init, dtype=torch.bfloat16)
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_REGEX,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def _load_unsloth(cfg: RLConfig):
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name=cfg.init,
        max_seq_length=cfg.max_model_len,
        load_in_4bit=False,
        fast_inference=True,  # Unsloth's own vLLM engine, with LoRA applied by vLLM
        max_lora_rank=cfg.rank,
        gpu_memory_utilization=cfg.gpu_mem(),
    )
    return FastLanguageModel.get_peft_model(
        model,
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )


# ---- logging --------------------------------------------------------------------------------


def logged_reward(records: list[dict], path: Path, every: int) -> Callable[..., list[float]]:
    """reward.make_reward, plus per-field accuracy in TRL's logs and, every `every` steps, the 3
    best and 3 worst answers in `path` (read by eye for reward hacking)."""
    by_id = {r["id"]: r for r in records}

    def filing_reward(prompts, completions, id, **kwargs) -> list[float]:  # `id`: the dataset column
        texts = [reward.completion_text(c) for c in completions]
        results = [reward.field_results(t, by_id[rid]) for t, rid in zip(texts, id, strict=True)]
        scores = [sum(r.values()) / len(r) if r else 0.0 for r in results]
        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            for name in sorted({f for r in results for f in r}):
                got = [r[name] for r in results if name in r]
                log_metric(f"fields/{name}", sum(got) / len(got))
        state = kwargs.get("trainer_state")
        step = state.global_step if state is not None else None
        if step is not None and step % every == 0:
            order = sorted(range(len(texts)), key=lambda i: scores[i])
            picked = [*order[-3:][::-1], *order[:3]]
            with path.open("a", encoding="utf-8") as f:
                for rank, i in enumerate(picked):
                    wrong = [name for name, ok in results[i].items() if not ok]
                    row = {"step": step, "kind": "best" if rank < 3 else "worst", "id": id[i]}
                    f.write(json.dumps({**row, "reward": scores[i], "wrong": wrong, "text": texts[i]}) + "\n")
        return scores

    return filing_reward


def _step_timer():
    """A TrainerCallback recording each optimizer step's wall time and the GPU's used memory."""
    import torch
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.seconds, self.used_gib, self._t = [], 0.0, None

        def on_train_begin(self, args, state, control, **kwargs):
            self._t = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            self.seconds.append(round(now - self._t, 2))
            self._t = now
            free, total = torch.cuda.mem_get_info()
            self.used_gib = max(self.used_gib, (total - free) / 2**30)

    return StepTimer()


def steady_seconds(seconds: list[float], warmup: int = 2) -> float | None:
    """Median step time after the first `warmup` steps (vLLM start-up, first sync, compiles)."""
    steady = seconds[warmup:] or seconds
    return round(statistics.median(steady), 2) if steady else None


def allow_sleep_mode() -> None:
    """vLLM's sleep mode refuses PyTorch's expandable segments, which pod_env.sh turns on for
    the other stages. Must run before the first CUDA allocation."""
    keep = [p for p in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",") if p and "expandable_segments" not in p]
    if keep:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(keep)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# ---- training -------------------------------------------------------------------------------


def train_rl(cfg: RLConfig, records: list[dict]) -> dict:
    """GRPO from `cfg.init` on `records` (already filtered). Writes adapters every
    `save_every` steps, train_meta.json (in train.py's format, so merging finds the base),
    log_history.json and samples.jsonl."""
    allow_sleep_mode()
    if cfg.stack == "unsloth":
        os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
        import unsloth  # noqa: F401  - must come first: it patches transformers and TRL

    # isort: split
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "samples.jsonl").unlink(missing_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.init)
    rows, dropped = rl_dataset(tok, records, cfg.max_model_len - cfg.max_completion)
    if cfg.stack == "trl":
        vllm_language_model_only(cfg.init)
        model = _load_trl(cfg)
    else:
        model = _load_unsloth(cfg)
    kwargs, unsupported = supported(grpo_args(cfg), GRPOConfig)
    if unsupported:
        print(f"GRPOConfig {metadata.version('trl')} doesn't take: {', '.join(unsupported)}")
    timer = _step_timer()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[logged_reward(records, cfg.out / "samples.jsonl", cfg.log_samples_every)],
        args=GRPOConfig(**kwargs),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,  # a plain tokenizer: text prompts, no image processing
        callbacks=[timer],
    )
    if cfg.stack == "trl":
        lora_sync(trainer)
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    result = trainer.train()
    train_s = time.time() - t_train
    last = cfg.out / f"checkpoint-{trainer.state.global_step}"
    if not last.exists():
        trainer.save_model(str(last))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    meta = {
        "config": {**config, "base": cfg.init, "vllm_mem": cfg.gpu_mem()},  # "base": what merging reads
        "method": "grpo",
        "stack": cfg.stack,
        "versions": versions(),
        "grpo_args": kwargs,
        "grpo_args_unsupported": unsupported,
        "train_filings": len(rows),
        "dropped_too_long": dropped,
        "steps": trainer.state.global_step,
        "save_every": cfg.save_every(),
        "train_seconds": round(train_s, 1),
        "seconds_per_step": steady_seconds(timer.seconds),
        "step_seconds": timer.seconds,
        "total_seconds": round(time.time() - t0, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),  # the trainer's own
        "gpu_used_gib_max": round(timer.used_gib, 2),  # the device's, vLLM included
        "gpu": torch.cuda.get_device_name(0),
        "train_loss": result.training_loss,
    }
    (cfg.out / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (cfg.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    return meta


# ---- the difficulty filter ------------------------------------------------------------------


def sample_groups(
    model: str,
    records: list[dict],
    out: Path,
    n: int = 8,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 0,
    max_model_len: int = MAX_MODEL_LEN,
    gpu_mem: float = 0.9,
) -> dict[str, list[float]]:
    """`n` answers per filing from `model`, sampled as GRPO will sample them, each scored by the
    reward. Writes filter.jsonl ({id, scores, mean, std, truncated, worst}) and
    filter_summary.json; returns {id: scores}."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)
    rows, dropped = rl_dataset(tok, records, max_model_len - max_tokens)
    extra = {"language_model_only": True} if is_multimodal(model_config(model)) else {}
    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, seed=seed, **extra)
    prompts = [TokensPrompt(prompt_token_ids=tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in rows]
    params = [
        SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed + i)
        for i in range(len(rows))
    ]
    t = time.time()
    outputs = llm.generate(prompts, params)
    seconds = time.time() - t
    by_id = {r["id"]: r for r in records}
    groups = {}
    with (out / "filter.jsonl").open("w", encoding="utf-8") as f:
        for row, o in zip(rows, outputs, strict=True):
            texts = [c.text for c in o.outputs]
            scores = [reward.score_completion(t, by_id[row["id"]]) for t in texts]
            groups[row["id"]] = scores
            s = reward.group_stats(scores)
            truncated = sum(c.finish_reason == "length" for c in o.outputs)
            worst = texts[min(range(len(scores)), key=scores.__getitem__)]
            line = {"id": row["id"], "scores": scores, "mean": s["mean"], "std": s["std"], "truncated": truncated}
            f.write(json.dumps({**line, "worst": worst}) + "\n")
    summary = {
        **reward.filter_summary(groups),
        "model": model,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "dropped_too_long": dropped,
        "seconds": round(seconds, 1),
        "seconds_per_filing": round(seconds / max(1, len(rows)), 2),
    }
    (out / "filter_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return groups


def read_filter(path: Path) -> dict[str, list[float]]:
    """{id: scores} from a filter.jsonl."""
    with path.open(encoding="utf-8") as f:
        return {row["id"]: row["scores"] for row in map(json.loads, f)}
