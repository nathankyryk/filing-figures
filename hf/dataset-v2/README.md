---
license: cc-by-4.0
language:
- en
pretty_name: FilingFigures v2
task_categories:
- text-generation
- table-question-answering
tags:
- finance
- sec
- 10-k
- xbrl
- information-extraction
- out-of-distribution
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: valid.jsonl
  - split: test
    path: test.jsonl
  - split: midcap_validation
    path: ood-valid.jsonl
  - split: midcap_test
    path: ood-test.jsonl
---

# FilingFigures v2

**1,549 SEC 10-K filings with twelve financial figures each, labelled from the companies' own XBRL data, plus an out-of-distribution test set of mid-size companies.** Built to compare supervised fine-tuning and reinforcement learning (GRPO) on companies unlike the training set.

Code, prompt and results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures) (part 2). Part 1, with six figures: [kyryk/filing-figures](https://huggingface.co/datasets/kyryk/filing-figures).

## What's in it

| Split | Filings | Companies | Scored figures |
|---|---|---|---|
| train | 934 | 162 | 10,788 |
| validation | 214 | 37 | 2,433 |
| test | 207 | 36 | 2,324 |
| midcap_validation | 75 | 75 | 865 |
| midcap_test | 119 | 119 | 1,372 |

- **train, validation, test:** non-financial S&P 500 companies, 10-Ks for fiscal years 2019–2024, with part 1's split by company.
- **midcap_validation, midcap_test:** one FY2025 10-K each from S&P MidCap 400 companies, none of them on part 1's S&P 500 list. Smaller companies, a year after all the training data.

## The twelve figures

Revenue, prior-year revenue, net income, basic and diluted EPS, total assets, cash and cash equivalents, stockholders' equity, shares outstanding, operating cash flow, capital expenditures, and free cash flow (operating cash flow minus capex).

Records have the same keys as [part 1](https://huggingface.co/datasets/kyryk/filing-figures) (`id`, `input`, `gold`, `flags`, `context_facts`, …). In `gold`, `null` means not scored. Free cash flow is scored only when both its parts are.

## How the labels were made

Same method as part 1: values from SEC companyfacts matched to the filing and period, the printed scale from inline XBRL, and a figure scored only if its printed number is in the model input.

**Hand-check:** 50 midcap_test filings (550 figures) checked against the filings. No value was wrong; 10 figures the key had left blank were filled in. In the 25 random filings, 4 of 275 figures changed, all blanks filled in.

## Scoring

A figure is right when its value is within 0.5% (EPS: within $0.005) and its period end is exact. See `evaluate.py` in the code repo.

## Source and license

- Built from public SEC EDGAR filings and XBRL data.
- Labels, inputs and structure: CC-BY-4.0.
