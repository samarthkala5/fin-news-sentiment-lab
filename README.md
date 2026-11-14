# Financial News Sentiment Lab

Compare classical ML, Keras, PyTorch and LoRA-fine-tuned LLM approaches on financial
news sentiment with proper statistical testing, plus a small LangChain RAG over public
central-bank text.

> **Every number between the `<!-- BEGIN:... -->` / `<!-- END:... -->` markers is
> generated** from files in `results/` and `data/splits/` by
> `scripts/make_results_table.py --update-readme`; nothing between the markers is written
> by hand. Models that have not been trained yet are shown as *not run yet*.

## Problem

Classify short English financial news posts (headline-style tweets such as
`$CX - Cemex cut at Credit Suisse, J.P. Morgan on weak building outlook`) as
**Bearish**, **Bullish** or **Neutral**, and find out whether bigger models are
*significantly* better than a strong bag-of-words baseline on the same fixed test set.

## Dataset

[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment),
English finance tweets labelled Bearish / Bullish / Neutral.

- **License:** MIT (stated on the dataset card).
- **Version:** Hub revision `ccbe24de388e287beb92dd393a335c376b350ac3`, pinned in `src/fnsl/config.py`.
- **Note:** the row counts on the dataset card do not match the pinned revision; the
  counts in the generated block below are what that revision actually contains
  (`scripts/prepare_data.py` downloads it).
- **Cleaning** (`src/fnsl/data.py`): normalise whitespace, drop empty and URL-only texts,
  drop near-duplicates (same text after lower-casing and removing links), and drop
  groups of near-duplicates with conflicting labels. See `DEBUG_LOG.md`, Day 1.
- **Split:** the two Hub splits are pooled and re-split once, seed 42, stratified
  70 / 15 / 15 into `data/splits/{train,val,test}.csv`. The test set is used only for
  final evaluation. Because of this re-split, scores here are **not directly comparable**
  to numbers reported on the Hub "validation" split.
- **Fallback** (not used): the Financial PhraseBank parquet mirror
  `atrost/financial_phrasebank` (CC BY-NC-SA 3.0), selectable with
  `python scripts/prepare_data.py --source fallback`.

<!-- BEGIN:DATASET -->
- **Dataset:** `zeroshot/twitter-financial-news-sentiment`, Hub revision `ccbe24de388e287beb92dd393a335c376b350ac3`, licence: MIT
- **Cleaning:** 11,931 raw rows; dropped 0 empty, 3 URL-only, 6 with conflicting labels, 147 near-duplicates; 11,775 rows left
- **Split** (stratified 70/15/15, seed 42): train 8,241, val 1,767, test 1,767

| Split | Rows | Bearish | Bullish | Neutral |
|---|---|---|---|---|
| all | 11,775 | 1,765 (15.0%) | 2,380 (20.2%) | 7,630 (64.8%) |
| train | 8,241 | 1,235 (15.0%) | 1,666 (20.2%) | 5,340 (64.8%) |
| val | 1,767 | 265 (15.0%) | 357 (20.2%) | 1,145 (64.8%) |
| test | 1,767 | 265 (15.0%) | 357 (20.2%) | 1,145 (64.8%) |

_Source: `data/splits/split_report.json` (written by `scripts/prepare_data.py`)._
<!-- END:DATASET -->

## Results

All models are scored on the same fixed test split. Macro-F1 is the main metric (each
class counts equally; Neutral is about two thirds of the data). 95% confidence intervals
come from bootstrap resamples of the test set (`scripts/compare_models.py`; the number of
resamples is in the note under the table).

<!-- BEGIN:RESULTS -->
| Model | Macro-F1 | 95% CI | Accuracy | F1 Bearish | F1 Bullish | F1 Neutral | n_test | Train time (s) | Commit | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + logistic regression (`tfidf_logreg`) | 0.739 | [0.715, 0.762] | 0.804 | 0.633 | 0.708 | 0.876 | 1767 | 3.7 | `18e5768` |  |
| Keras BiLSTM (`keras_bilstm`) | not run yet |  |  |  |  |  |  |  |  | run `python scripts/train_keras.py` |
| DistilBERT fine-tuned in PyTorch (`distilbert`) | not run yet |  |  |  |  |  |  |  |  | run `python scripts/train_distilbert.py` on a GPU (notebooks/colab_runner.ipynb) |
| Small LLM, zero-shot (`llm_zeroshot`) | not run yet |  |  |  |  |  |  |  |  | run `python scripts/train_unsloth.py --mode zero-shot` on a GPU |
| Small LLM + LoRA via Unsloth (`llm_lora`) | not run yet |  |  |  |  |  |  |  |  | run `python scripts/train_unsloth.py --mode train`, then `--mode eval`, on a GPU |

_95% CI: percentile bootstrap over 2,000 resamples of the test set (`results/metrics/comparison.json`). Per-model scores: `results/metrics/<model>.json`._
<!-- END:RESULTS -->

### Are the differences significant?

McNemar's test on paired test-set predictions for every pair of models
(`results/metrics/comparison.json`).

<!-- BEGIN:COMPARISON -->
_No pairs compared yet. Models with test predictions: `tfidf_logreg`. A pair needs two models with predictions on the same test ids._
<!-- END:COMPARISON -->

### Clustering sentence embeddings (unsupervised)

<!-- BEGIN:KMEANS -->
Embeddings `sentence-transformers/all-MiniLM-L6-v2`, split: train, n = 8,241 (`results/metrics/kmeans.json`).

| k | Silhouette | ARI | NMI |
|---|---|---|---|
| 3 | 0.012 | 0.050 | 0.051 |
| 5 | 0.011 | 0.043 | 0.054 |

_Silhouette: how well separated the clusters are (-1 to 1). ARI: agreement between clusters and sentiment labels, corrected for chance (0 = chance level, 1 = identical). NMI: information the clusters share with the labels (0 = none, 1 = identical; not corrected for chance, so random clusters score slightly above 0)._
<!-- END:KMEANS -->

### RAG retrieval over central-bank text

`data/rag_corpus/` holds 10 public statements: 7 Federal Reserve FOMC statements (public
domain) and 3 ECB monetary-policy press releases (reproduction allowed with the source
acknowledged). URLs, dates and licence notes are in `data/rag_corpus/SOURCES.md`.
The pipeline (`src/fnsl/rag/`) moves each file's Source/Title/Date/Publisher lines into
metadata, splits the text with LangChain's `RecursiveCharacterTextSplitter`, embeds the
chunks with `sentence-transformers/all-MiniLM-L6-v2` and searches a FAISS index. The
default answer is retrieval-only (best chunk plus its source file); `--llm-model` adds a
small local generator through a LangChain chain. **hit@k** is the share of hand-written
questions in `data/rag_eval.csv` whose source file is among the top-k retrieved chunks;
it is reported for chunk sizes 300 and 800.

<!-- BEGIN:RAG -->
_Not available yet: write the evaluation questions in `data/rag_eval.csv`, run `python scripts/eval_rag.py`, then `python scripts/make_results_table.py --update-readme`._
<!-- END:RAG -->

### Findings

_To be written after the comparison is complete (Day 10): which model wins, whether the
gap is statistically significant, and what each extra point of macro-F1 costs._

## How to reproduce

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # light: CI, data, TF-IDF baseline, stats, tables
pip install -e . --no-deps               # makes `import fnsl` work in scripts
```

For embeddings, RAG and `--smoke` runs of the deep-learning scripts on a laptop CPU:

```bash
pip install "torch>=2.4,<2.15" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-local.txt
```

On Colab / Kaggle (GPU) use `requirements-gpu.txt` via `notebooks/colab_runner.ipynb`.

### Local (CPU)

```bash
python scripts/prepare_data.py     # once: the fixed split (already committed, read-only)
python scripts/run_baseline.py     # TF-IDF + logistic regression (Day 2)
python scripts/run_kmeans.py       # K-Means on MiniLM embeddings + NumPy PCA (Day 3)
python scripts/train_keras.py      # Keras BiLSTM (Day 4); a laptop CPU is enough
python scripts/compare_models.py   # bootstrap CIs + McNemar for every pair of models
python scripts/build_rag.py --question "How did the Federal Reserve respond to the coronavirus in March 2020?"
python scripts/eval_rag.py         # hit@1 / hit@3; needs >= 5 real questions in data/rag_eval.csv
python scripts/make_results_table.py --update-readme --count-tests   # tables + RESUME_NUMBERS.md
python scripts/push_to_hub.py      # dry run: writes the model card to results/model_card/
pytest -m "not heavy" && ruff check .
pytest -m heavy                    # needs requirements-local.txt; downloads small models
```

`data/rag_eval.csv` has two columns, `question,source_file` (a file name in
`data/rag_corpus/`). Put a question that contains a comma in double quotes. The three rows
whose question starts with `EXAMPLE` only show the format: real runs skip them, and only
the pipeline check `--smoke --include-examples` (written to `results/smoke/`) uses them.

Every training script also takes `--smoke` (tiny data, 1 epoch, CPU). Smoke outputs go to
`results/smoke/` and `checkpoints/smoke/` (both gitignored) and are never used as results:

```bash
python scripts/run_baseline.py --smoke
python scripts/run_kmeans.py --smoke
python scripts/train_keras.py --smoke
python scripts/train_distilbert.py --smoke
python scripts/train_unsloth.py --smoke --mode all   # tiny SmolLM2-135M via the plain HF backend
python scripts/eval_rag.py --smoke --include-examples
python scripts/compare_models.py --smoke
```

### GPU (Colab / Kaggle)

[`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb)
([open in Colab](https://colab.research.google.com/github/samarthkala5/fin-news-sentiment-lab/blob/main/notebooks/colab_runner.ipynb),
which works once this code is on `main` and the repository is public) is a thin runner.
Until this code is merged into `main`, set `BRANCH = "claude/project-skeleton-build-dj1aj6"`
in its clone step. It checks the GPU,
clones the repo (`REPO_URL` / `BRANCH`; never paste a token), installs
`requirements-gpu.txt`, runs the scripts below and zips `results/metrics`,
`results/predictions` and `results/figures` for download. On Colab pick
*Runtime > Change runtime type > T4 GPU*; on Kaggle pick *GPU T4 x2* with Internet on
(Unsloth does not support the P100).

| Day | Command | Writes |
|---|---|---|
| 5-6 | `python scripts/train_distilbert.py --smoke`, then `python scripts/train_distilbert.py` | `distilbert` metrics, predictions and confusion matrix; best checkpoint in `checkpoints/distilbert/best` |
| 7 (optional) | `python scripts/train_distilbert.py --seed 43`, then `--seed 44` | `distilbert_seed43`, `distilbert_seed44`; `compare_models.py` reports mean ± std |
| 8 | `python scripts/train_unsloth.py --mode zero-shot` | `llm_zeroshot` on 50 test headlines (marked *not comparable*) |
| 9 | `python scripts/train_unsloth.py --mode train` | LoRA adapter in `checkpoints/llm_lora_adapter` + `llm_lora_train.json` (loss curves, no test scores) |
| 10 | `python scripts/train_unsloth.py --mode eval` | `llm_lora` on the full test split |
| 10 (optional) | `python scripts/train_unsloth.py --mode zero-shot --n-test 0` | zero-shot on the full test split, so McNemar can pair it |

Run Days 9 and 10 in the same Colab session: a new session starts without `checkpoints/`,
so the adapter would have to be uploaded again (a flag in the last cell adds it to the zip
to keep a copy). Back on the laptop, unzip into
the repo, run `python scripts/compare_models.py` and
`python scripts/make_results_table.py --update-readme --count-tests`, run the tests, and
commit the new files in `results/`.

[`notebooks/kaggle_report.ipynb`](notebooks/kaggle_report.ipynb) is the public report. It
trains nothing; it reads only committed files (`data/splits/`, `results/`) and plots them.

## Repo layout

```
fin-news-sentiment-lab/
├── CLAUDE.md, PROJECT_PLAN.md       # working rules and the 14-day plan
├── README.md, DEBUG_LOG.md, reading_log.md
├── RESUME_NUMBERS.md                # generated by make_results_table.py, never hand-edited
├── requirements.txt                 # light deps (CI)
├── requirements-local.txt           # + torch CPU, transformers, TF/Keras, LangChain HF
├── requirements-gpu.txt             # Colab/Kaggle GPU, adds Unsloth
├── pyproject.toml, pytest.ini       # package + ruff config; pytest `heavy` marker
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                         # download cache (gitignored)
│   ├── splits/                      # fixed train/val/test CSVs + split_report.json
│   ├── rag_corpus/                  # public central-bank texts + SOURCES.md
│   └── rag_eval.csv                 # hand-written retrieval questions
├── src/fnsl/                        # reusable modules: data, models, metrics, stats, RAG
├── scripts/                         # command-line entry points (every training script has --smoke)
├── notebooks/                       # colab_runner.ipynb, kaggle_report.ipynb
├── results/                         # metrics/*.json, predictions/*.csv, figures/
└── tests/
```

## Limitations

- **Small, single dataset:** short English tweets from one source (sizes in the Dataset
  section); results
  may not transfer to longer news articles, other languages or other periods.
- **Short texts:** most posts are one headline, so context that would disambiguate
  sentiment is often missing.
- **Single fixed split:** one train/val/test split (seed 42). Bootstrap CIs capture
  test-set sampling noise but not variation from a different split; seed variation is
  only measured where several seeds were run.
- **Label quality:** labels come from the dataset authors; near-duplicates with
  conflicting labels were removed, but other label noise remains.
- **No market validation:** sentiment labels are not checked against actual price moves,
  so nothing here says the predictions are useful for trading.
- **RAG evaluation measures retrieval only:** hit@k checks whether the right source file
  is retrieved, not whether the generated answer is correct.

## Logs

- `DEBUG_LOG.md`: one real bug and fix per day.
- `reading_log.md`: my own summaries of the FinBERT, LoRA and RAG papers (template, to be
  filled in as the plan reaches each paper).
