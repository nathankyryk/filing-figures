---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
datasets:
- kyryk/filing-figures-v2
language:
- en
pipeline_tag: text-generation
library_name: peft
tags:
- finance
- sec
- 10-k
- xbrl
- information-extraction
- lora
---

# FilingFigures v2 Qwen3.5-4B, SFT (LoRA adapter)

**A LoRA adapter that teaches Qwen3.5-4B to read a 10-K and return twelve financial figures as JSON.** On mid-size companies from a year it never saw, it gets 99.0% of figures right (86.2% before fine-tuning) and beats Qwen3.5-397B.

This is the supervised half of a comparison with reinforcement learning: see the [GRPO adapter](https://huggingface.co/kyryk/filing-figures-v2-qwen3.5-4b-grpo-lora). Code, prompt and full results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures) (part 2).

## Results

Each test set run once.

| Model | Mid-cap test (119 filings) | S&P 500 test (207 filings) |
|---|---|---|
| Qwen3.5-4B, no fine-tuning | 86.2% | 95.4% |
| Qwen3.5-4B + GRPO adapter | 98.2% | 98.4% |
| Qwen3.5-397B | 98.3% | – |
| **Qwen3.5-4B + this adapter** | **99.0%** | **98.8%** |
| Claude Opus 5.5 | 99.3% | – |

All 12 figures right on 92% of mid-cap filings.

## Training

| Setting | Value |
|---|---|
| Method | bf16 LoRA with Unsloth + TRL |
| Rank, α, dropout | 16, 32, 0 |
| Target modules | Attention and MLP projections of the language model |
| Learning rate | 2e-4 |
| Batch | 16 filings per step, 3 epochs (177 steps) |
| Data | 934 training filings; loss on the answer tokens only; blank (unscored) fields not trained on |
| This checkpoint | Step 60, the best on validation and mid-cap validation together |
| Hardware | One H100, 25 minutes |

## Use it

Merge the adapter into the base model's weights, then serve with vLLM:

```bash
git clone https://github.com/nathankyryk/filing-figures && cd filing-figures
uv sync --extra train   # Linux + CUDA 13
uv run hf download kyryk/filing-figures-v2-qwen3.5-4b-sft-lora --local-dir adapter/
uv run figures merge --adapter adapter/ --out merged/
vllm serve merged/ --max-model-len 12288 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The model expects the part 2 prompt (`SYSTEM_V2` in `src/filing_figures/prompts.py`) and the filing as built by `src/filing_figures/inputs.py`.

## Limitations

- Trained on non-financial S&P 500 10-Ks from 2019–2024.
- 11 of its 14 mid-cap misses are revenue, capex or free cash flow.
