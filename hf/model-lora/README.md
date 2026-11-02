---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
datasets:
- kyryk/filing-figures
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
- unsloth
---

# FilingFigures Qwen3.5-4B (LoRA adapter)

**A LoRA adapter that teaches Qwen3.5-4B to read a 10-K and return six financial figures as JSON**: revenue, net income, diluted EPS, total assets, shares outstanding and operating cash flow. On held-out companies the fine-tuned model gets 98.4% of figures right in bf16 (95.4% before fine-tuning).

For serving, use the quantized model [kyryk/filing-figures-qwen3.5-4b-nvfp4](https://huggingface.co/kyryk/filing-figures-qwen3.5-4b-nvfp4) (98.8%, $0.07 per 1,000 filings). Code, prompt and full results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures).

## Results

Test split of [kyryk/filing-figures](https://huggingface.co/datasets/kyryk/filing-figures): 207 filings from 36 companies, 1,186 figures, run once.

| Model | Figures right | Filings with all 6 right |
|---|---|---|
| Qwen3.5-4B, no fine-tuning | 95.4% | 83% |
| **Qwen3.5-4B + this adapter (bf16)** | **98.4%** | **91%** |
| Same, quantized to NVFP4 | 98.8% | 93% |

## Training

| Setting | Value |
|---|---|
| Method | bf16 LoRA with Unsloth + TRL (Unsloth advises against 4-bit QLoRA for Qwen3.5) |
| Rank, α, dropout | 16, 32, 0 |
| Target modules | Attention and MLP projections of the language model |
| Learning rate | 2e-4 |
| Batch | 16 filings per step |
| Data | 940 training filings; loss on the answer tokens only; blank (unscored) fields not trained on |
| This checkpoint | Step 30 of 177 (half an epoch). All six checkpoints scored 100% on validation, so the earliest was kept |
| Hardware | One RTX 5090, 44 minutes for 3 epochs, 10.4 GiB peak |

## Use it

Merge the adapter into the base model's weights. The results used `figures merge`, which writes W + (α/r)·B·A into a copy of the base checkpoint, so vLLM serves it like the base model:

```bash
git clone https://github.com/nathankyryk/filing-figures && cd filing-figures
uv sync --extra train   # Linux + CUDA 13
uv run hf download kyryk/filing-figures-qwen3.5-4b-lora --local-dir adapter/
uv run figures merge --adapter adapter/ --out merged/
vllm serve merged/ --max-model-len 12288 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

vLLM's runtime LoRA didn't apply to Qwen3.5's hybrid layers in our tests, so merge first. The model expects the training prompt: the system message in `src/filing_figures/prompts.py`, then the filing as built by `src/filing_figures/inputs.py` (`figures extract` does both).

## Limitations

- Trained on non-financial S&P 500 10-Ks from 2019–2024.
- Its remaining test errors are mostly definitional: "Sales and other operating revenues" vs total revenues, and net income before vs after preferred dividends.
