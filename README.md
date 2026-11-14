# Financial News Sentiment Lab

Compare classical ML, Keras, PyTorch and LoRA-fine-tuned LLM approaches on financial
news sentiment with proper statistical testing, plus a small LangChain RAG over public
central-bank text.

> **Every number in this README is generated** from files in `results/` and `data/splits/`
> by `scripts/make_results_table.py --update-readme`. Nothing between the
> `<!-- BEGIN:... -->` / `<!-- END:... -->` markers is written by hand. Models that have
> not been trained yet are shown as *not run yet*.

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
- **Note:** the dataset card lists 9,938 / 2,486 rows, but the pinned revision actually
  contains 9,543 (train) + 2,388 (validation) rows.
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
_Not generated yet: run `python scripts/make_results_table.py --update-readme`._
<!-- END:DATASET -->

## Results

All models are scored on the same fixed test split. Macro-F1 is the main metric (each
class counts equally; Neutral is about two thirds of the data). 95% confidence intervals
come from 1,000+ bootstrap resamples of the test set (`scripts/compare_models.py`).

<!-- BEGIN:RESULTS -->
_Not generated yet: run `python scripts/make_results_table.py --update-readme`._
<!-- END:RESULTS -->

### Are the differences significant?

McNemar's test on paired test-set predictions for every pair of models
(`results/metrics/comparison.json`).

<!-- BEGIN:COMPARISON -->
_Not generated yet: run `python scripts/compare_models.py`, then `make_results_table.py`._
<!-- END:COMPARISON -->

### Clustering sentence embeddings (unsupervised)

<!-- BEGIN:KMEANS -->
_Not generated yet: run `python scripts/run_kmeans.py`, then `make_results_table.py`._
<!-- END:KMEANS -->

### RAG retrieval over central-bank text

<!-- BEGIN:RAG -->
_Not generated yet: write `data/rag_eval.csv`, run `python scripts/eval_rag.py`, then `make_results_table.py`._
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
python scripts/prepare_data.py            # once: fixed split (already committed)
python scripts/run_baseline.py            # TF-IDF + logistic regression
python scripts/run_kmeans.py              # K-Means on sentence embeddings + NumPy PCA
python scripts/compare_models.py          # McNemar + bootstrap CIs for all models
python scripts/build_rag.py               # FAISS index over data/rag_corpus/
python scripts/eval_rag.py                # hit@1 / hit@3 on data/rag_eval.csv
python scripts/make_results_table.py --update-readme   # tables + RESUME_NUMBERS.md
pytest -m "not heavy" && ruff check .
```

Every training script also takes `--smoke` (tiny data, 1 epoch, CPU); smoke outputs go
to `results/smoke/` (gitignored) and are never used as results.

### GPU (Colab / Kaggle)

Open `notebooks/colab_runner.ipynb`, pick a GPU runtime, and run the cells for
DistilBERT, Keras or Unsloth. The notebook clones this repo, installs
`requirements-gpu.txt`, runs the script in `scripts/` and zips `results/` for download.
Copy the files back into `results/` locally and commit them.

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
├── src/fnsl/                        # all logic lives here
├── scripts/                         # thin CLI wrappers (every training script has --smoke)
├── notebooks/                       # colab_runner.ipynb, kaggle_report.ipynb
├── results/                         # metrics/*.json, predictions/*.csv, figures/
└── tests/
```

## Limitations

- **Small, single dataset:** about 11.8k short English tweets from one source; results
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
- `reading_log.md`: FinBERT, LoRA and RAG papers, summarised in my own words.
