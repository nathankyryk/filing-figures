"""LoRA fine-tuning with Unsloth + TRL, and merging a checkpoint into full weights (pod only,
`train` environment).

- Qwen3.5 trains as bf16 LoRA. Unsloth: "It is not recommended to do QLoRA (4-bit) training on
  the Qwen3.5 models". The other candidates train as QLoRA, which Unsloth supports for them.
- r=16, alpha=32, lr 2e-4 with cosine decay, an effective batch of 16 filings, 3 epochs.
  Adapters are saved every half epoch, and each is scored on all validation filings in vLLM;
  the best one is picked by field accuracy, not by loss.
- The prompts come from the base model's own tokenizer and chat template, the same ones vLLM
  uses for the base model, so base and fine-tuned models see identical inputs.
- vLLM's runtime LoRA isn't reliable for Qwen3.5's hybrid layers, so checkpoints are merged
  into the base model's weights before serving (merge.py).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filing_figures import sft

FOUR_BIT = {  # QLoRA where the model's maintainers support it
    "Qwen/Qwen3.5-4B": False,
    "Qwen/Qwen3.5-9B": False,
    "google/gemma-4-E4B-it": True,
    "Qwen/Qwen3-8B": True,
}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base: str
    out: Path
    epochs: float = 3.0
    rank: int = 16
    alpha: int = 32
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 16
    warmup: float = 0.05  # fraction of the steps
    max_seq_length: int = 8192
    saves_per_epoch: int = 2
    max_steps: int = -1  # > 0: a timing run (the check stage)
    eval_filings: int = 40  # validation filings for the loss curve (picking uses accuracy)
    load_in_4bit: bool | None = None  # None: FOUR_BIT, else bf16
    seed: int = 0
    notes: dict = field(default_factory=dict)

    def four_bit(self) -> bool:
        return FOUR_BIT.get(self.base, False) if self.load_in_4bit is None else self.load_in_4bit


def _is_gemma(base: str) -> bool:
    return "gemma" in base.lower()


def _load(model_name: str, base: str, max_seq_length: int, four_bit: bool):
    from unsloth import FastLanguageModel, FastModel

    loader = FastModel if _is_gemma(base) else FastLanguageModel
    return loader, *loader.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=four_bit,
        load_in_16bit=not four_bit,
        full_finetuning=False,
    )


def encode_records(tokenizer, records: list[dict], max_len: int) -> tuple[list[dict], list[str]]:
    """Encoded examples, and the ids of filings too long to train on (dropped, then reported)."""
    eot = sft.end_of_turn(tokenizer)
    rows, dropped = [], []
    for rec in records:
        row = sft.encode(tokenizer, rec, eot)
        (dropped.append(rec["id"]) if len(row["input_ids"]) > max_len else rows.append(row))
    return rows, dropped


def train(cfg: TrainConfig, train_recs: list[dict], valid_recs: list[dict]) -> dict:
    import unsloth  # noqa: F401  - must come first: it patches transformers and TRL

    # isort: split
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.base)  # the base model's own template (see above)
    rows, dropped = encode_records(tok, train_recs, cfg.max_seq_length)
    eval_rows, _ = encode_records(tok, valid_recs[: cfg.eval_filings], cfg.max_seq_length)
    four = cfg.four_bit()
    loader, model, _ = _load(cfg.base, cfg.base, cfg.max_seq_length, four)
    lora = {"r": cfg.rank, "lora_alpha": cfg.alpha, "lora_dropout": 0, "bias": "none", "random_state": cfg.seed}
    if _is_gemma(cfg.base):  # Unsloth's Gemma 4 recipe: language layers only
        model = loader.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth",
            **lora,
        )
    else:
        model = loader.get_peft_model(
            model, target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth", **lora
        )

    per_step = cfg.batch_size * cfg.grad_accum
    steps_per_epoch = math.ceil(len(rows) / per_step)
    total = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.epochs)
    timing = cfg.max_steps > 0
    # Evenly spaced saves; the last one lands on the final step (saved below if it's off the grid)
    save_every = total if timing else max(1, math.ceil(total / round(cfg.epochs * cfg.saves_per_epoch)))
    args = SFTConfig(
        output_dir=str(cfg.out),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(cfg.warmup * total)),
        optim="adamw_8bit",
        weight_decay=0.0,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_every,
        save_only_model=True,  # adapters only: no optimizer state to resume from, ~100x smaller
        eval_strategy="no" if timing else "steps",
        eval_steps=save_every,
        report_to="none",
        seed=cfg.seed,
        max_length=cfg.max_seq_length,
        packing=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},  # already tokenized, with our own labels
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        eval_dataset=None if timing else Dataset.from_list(eval_rows),
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    result = trainer.train()
    train_s = time.time() - t_train
    last = cfg.out / f"checkpoint-{trainer.state.global_step}"
    if not last.exists():
        trainer.save_model(str(last))
    meta = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "load_in_4bit": four,
        "train_filings": len(rows),
        "dropped_too_long": dropped,
        "steps": trainer.state.global_step,
        "steps_per_epoch": steps_per_epoch,
        "save_every": save_every,
        "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / max(1, trainer.state.global_step), 2),
        "total_seconds": round(time.time() - t0, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
        "train_loss": result.training_loss,
    }
    (cfg.out / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (cfg.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    return meta


def checkpoints(run_dir: Path) -> list[Path]:
    """The run's saved adapters, in step order."""
    found = [p for p in run_dir.glob("checkpoint-*") if (p / "adapter_config.json").exists()]
    return sorted(found, key=lambda p: int(p.name.split("-")[1]))


def merge(adapter_dir: Path, out_dir: Path) -> dict:
    """A checkpoint merged into its base model's weights (see merge.py), ready for vLLM."""
    from filing_figures.merge import merge_adapter

    meta = json.loads((adapter_dir.parent / "train_meta.json").read_text(encoding="utf-8"))
    return merge_adapter(meta["config"]["base"], adapter_dir, out_dir)
