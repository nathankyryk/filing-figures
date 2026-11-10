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
- grpo
- reinforcement-learning
---

# FilingFigures v2 Qwen3.5-4B, GRPO (LoRA adapter)

**A LoRA adapter trained with GRPO to read a 10-K and return twelve financial figures as JSON, without ever seeing a correct answer.** It learned only from a reward: the share of its figures that match the XBRL answer key. On mid-size companies from a year it never saw, it gets 98.2% right (86.2% before training), a tie with Qwen3.5-397B.

It's the reinforcement-learning half of a comparison: the [SFT adapter](https://huggingface.co/kyryk/filing-figures-v2-qwen3.5-4b-sft-lora), trained on the answers, does better (99.0%). Code and full results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures) (part 2).

## Results

Each test set run once.

| Model | Mid-cap test (119 filings) | S&P 500 test (207 filings) |
|---|---|---|
| Qwen3.5-4B, no fine-tuning | 86.2% | 95.4% |
| **Qwen3.5-4B + this adapter** | **98.2%** | **98.4%** |
| Qwen3.5-4B + SFT adapter | 99.0% | 98.8% |

GRPO minus SFT: −0.8 points on mid-caps [−1.5, −0.1] and −0.4 on the S&P 500 [−0.9, −0.04], 95% intervals from resampling companies.

## Training

| Setting | Value |
|---|---|
| Method | GRPO with TRL 1.17.1; vLLM 0.32 on the same GPU generates the answers |
| Reward | Correct figures ÷ scored figures, with the test's own scoring rule. A broken answer scores 0 |
| Rank, α | 32, 64, on the language model's attention and MLP projections |
| Learning rate | 1e-5, constant after warm-up |
| Per step | 8 filings × 8 answers at temperature 1.0, up to 1,024 tokens each |
| Loss | DAPO token-level loss, clipping 0.2 / 0.28, no KL penalty |
| Data | 606 of 934 training filings: those where the base model's 8 answers didn't all score the same |
| This checkpoint | Step 48 of 72, the best on validation and mid-cap validation together |
| Hardware | One H100, 2.4 hours |

GRPO only learns from filings where its 8 answers disagree: 78% of filings in the first 12 steps, 5% by the end. Nothing improved after step 48.

## Use it

Merge the adapter into the base model's weights, then serve with vLLM:

```bash
git clone https://github.com/nathankyryk/filing-figures && cd filing-figures
uv sync --extra train   # Linux + CUDA 13
uv run hf download kyryk/filing-figures-v2-qwen3.5-4b-grpo-lora --local-dir adapter/
uv run figures merge --adapter adapter/ --out merged/
vllm serve merged/ --max-model-len 12288 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The model expects the part 2 prompt (`SYSTEM_V2` in `src/filing_figures/prompts.py`) and the filing as built by `src/filing_figures/inputs.py`.

## Limitations

- Trained on non-financial S&P 500 10-Ks from 2019–2024.
- Weaker than SFT on figures that follow a convention. On mid-caps: prior-year revenue 95.8% vs 98.3%, capex 94.3% vs 96.2%, free cash flow 93.3% vs 97.1%.
