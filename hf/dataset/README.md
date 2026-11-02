---
license: cc-by-4.0
language:
- en
pretty_name: FilingFigures
task_categories:
- text-generation
- table-question-answering
tags:
- finance
- sec
- 10-k
- xbrl
- information-extraction
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
---

# FilingFigures

**1,361 SEC 10-K filings with six financial figures each, labelled from the companies' own XBRL data.** Built to train and score small models that read a 10-K and return revenue, net income, diluted EPS, total assets, shares outstanding and operating cash flow as JSON.

Code, prompt and results: [github.com/nathankyryk/filing-figures](https://github.com/nathankyryk/filing-figures).

## What's in it

| Split | Filings | Companies | Scored figures |
|---|---|---|---|
| train | 940 | 163 | 5,487 |
| validation | 214 | 37 | 1,237 |
| test | 207 | 36 | 1,186 |

- **Companies:** a seeded sample of 250 S&P 500 companies, without Financials (a bank has no single revenue line). 236 of them had at least four inline-XBRL 10-Ks.
- **Filings:** original 10-Ks (no amendments) for fiscal years 2019–2024.
- **Split by company,** so no company appears in two splits. A 10-K repeats the prior year's numbers, so a split by filing would leak answers.

## One record

| Key | What it is |
|---|---|
| `id` | Ticker and fiscal year, e.g. `SBUX-2024` |
| `cik`, `ticker`, `sector`, `accession`, `filing_date`, `fiscal_year`, `period_end` | The filing |
| `input` | The model input: the cover page and the three statements as plain text, tables one row per line with ` \| ` between cells. Median ~11K characters (~4.5K tokens) |
| `gold` | The answer key: per field, `value` (full number, in dollars or shares), `scale` (as printed: units, thousands, millions, billions), `period_end`, `fiscal_year`, the XBRL `tag` used and the number as `displayed`. `null` means not scored |
| `flags` | Per field, how its label was made, e.g. `tag_from_statement`, `not_in_input`, `handcheck_ok` |
| `context_facts` | Every other XBRL fact for these concepts in the filing (prior years, other tags), used to name error types |

## How the labels were made

1. **Values** come from SEC companyfacts, matched on the filing's accession number and its period end. Year-long figures must also cover 350–380 days, so quarters and prior years drop out.
2. **Tags** are tried in a fixed order per field, e.g. net income: `NetIncomeLoss` → `NetIncomeLossAvailableToCommonStockholdersBasic` → `ProfitLoss`. If the statement shows a later tag's figure, that one is used.
3. **The printed scale** comes from the filing's inline XBRL (`scale="6"` = millions).
4. **Coverage check:** a figure is scored only if its printed number appears in the model input. Filings with more than two unscorable figures were dropped (35 of 1,396).
5. **Shares outstanding** are left unscored when the cover lists several share classes.
6. **Hand-check:** 100 test filings (600 figures) were checked by hand against the filings: 1 fix. In the random 40 of them, 1 of 240 figures was wrong, an error rate of ~0.4% (95% upper bound ~2%).

## Scoring

A figure is right when its value is within 0.5% (EPS: within $0.005) and its period end is exact. See `evaluate.py` in the code repo.

## Source and license

- Built from public SEC EDGAR filings and XBRL data.
- Labels, inputs and structure: CC-BY-4.0.
