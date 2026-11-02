---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
datasets:
- kyryk/filing-figures
language:
- en
pipeline_tag: text-generation
library_name: vllm
tags:
- finance
- sec
- 10-k
- xbrl
- information-extraction
- nvfp4
- compressed-tensors
---

# FilingFigures Qwen3.5-4B (NVFP4)

**Qwen3.5-4B fine-tuned to read a 10-K and return six financial figures as JSON, quantized to 4-bit NVFP4 for vLLM.** On held-out companies it gets 98.8% of figures right, ties Qwen3.5-397B and costs $0.07 per 1,000 filings on one RTX 5090.

Code, prompt, data and full results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures). The LoRA adapter is at [kyryk/filing-figures-qwen3.5-4b-lora](https://huggingface.co/kyryk/filing-figures-qwen3.5-4b-lora).

## Results

Test split of [kyryk/filing-figures](https://huggingface.co/datasets/kyryk/filing-figures): 207 filings from 36 companies not seen in training, 1,186 figures, run once.

| Model | Figures right | Filings with all 6 right | $ per 1,000 filings |
|---|---|---|---|
| Qwen3.5-4B, no fine-tuning | 95.4% | 83% | $0.11 |
| **This model (fine-tuned, NVFP4)** | **98.8%** | **93%** | **$0.07** |
| Qwen3.5-397B | 98.7% | 93% | $3.66 |
| Claude Opus 5.5 | 100.0% | 100% | $29.84 |

A figure is right when it's within 0.5% (EPS: $0.005) and its period end date is exact.

## Use it

Needs an NVIDIA Blackwell GPU (RTX 50-series, B200) for NVFP4. Tested with vLLM 0.32.0 on an RTX 5090.

```bash
vllm serve kyryk/filing-figures-qwen3.5-4b-nvfp4 --served-model-name filing-figures \
  --max-model-len 12288 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Then send it a 10-K with the repo's CLI. This fetches the filing from EDGAR, builds the input and scores the answer against XBRL:

```bash
git clone https://github.com/nathankyryk/filing-figures && cd filing-figures && uv sync
uv run figures extract --ticker SBUX --year 2025 --endpoint http://localhost:8000/v1 --model filing-figures
```

The model expects the exact prompt it was trained with: the system message in `src/filing_figures/prompts.py`, then the filing as built by `src/filing_figures/inputs.py`. It answers with one JSON object:

```json
{"revenue": {"value": 36176200000, "scale": "millions", "period_end": "2024-09-29", "fiscal_year": 2024}, "net_income": {...}, "eps_diluted": {...}, "total_assets": {...}, "shares_outstanding": {...}, "operating_cash_flow": {...}}
```

## How it was made

- **Fine-tuning:** bf16 LoRA (rank 16, α 32, lr 2e-4) with Unsloth on 940 training filings. The loss covers the answer only. The checkpoint after half an epoch was kept: every checkpoint scored 100% on validation, so the earliest won.
- **Quantization:** llm-compressor NVFP4, calibrated on 256 training filings. `lm_head`, the vision tower and Qwen3.5's linear-attention layers stay in bf16. Validation accuracy: 99.8% vs 100% in bf16.
- **Speed:** 4.6 filings/s at 32 in flight on one RTX 5090 (~5,000 tokens in and ~300 out per filing).

## Limitations

- Trained and tested on non-financial S&P 500 10-Ks from 2019–2024. Banks and insurers weren't included.
- Its remaining test errors are mostly definitional: "Sales and other operating revenues" vs total revenues, and net income before vs after preferred dividends.
- Answers only from the text it's given. It wasn't tested on PDFs, scanned filings or other report types.
