# PROJECT_PLAN.md — Financial News Sentiment Lab

> **How to use this file.** Paste it into a new Claude Code session (or save it as `PROJECT_PLAN.md` in an empty repo and say: *"Read PROJECT_PLAN.md fully. Create CLAUDE.md from the section 'CLAUDE.md contents', then start Day 1."*). Work one day at a time. After each day, stop, show me the results and the commit, and wait for me.

---

## 0. Context (read first)

- **Owner:** Samarth Kala, final-year B.Tech ECE at SVNIT Surat (2027). Python is my primary language. I already know pytest, GitHub Actions CI, Docker, FastAPI, MLflow and Sentence Transformers.
- **Goal:** build one portfolio project that gives honest, measurable evidence for a *Data Science Intern, Global Markets Data & AI Lab* application (BNP Paribas). The job post asks for: Python, ML classification/prediction/clustering, statistics and linear algebra, neural network architectures, PyTorch, TensorFlow, Keras, scikit-learn, pandas, NumPy, Git, LangChain, Unsloth, Hugging Face, hypothesis testing and careful experiments, alternative data (news), involvement in the data science community (Kaggle etc.), and awareness of current ML literature.
- **Time budget:** 14 days, about 1 hour per day from me. Claude Code writes most code; my hour is for running it, reading results, debugging and understanding.
- **Hard constraints:**
  1. Do **not** touch my existing projects (AI Job Application Prioritizer, Crypto Order Book Monitor, Task Manager).
  2. **Never invent numbers.** Every metric on the README or resume must come from a results file produced by a script in this repo.
  3. I must be able to explain every model, metric and test in an interview. Keep code clear and commented where the logic is non-obvious. Prefer simple over clever.
  4. Do not claim a skill (LangChain, Unsloth, etc.) in any resume text until the matching day is complete and verified.

## 1. Project summary

**Name:** `fin-news-sentiment-lab`

**One line:** Compare classical ML, Keras, PyTorch and LoRA-fine-tuned LLM approaches on financial news sentiment with proper statistical testing, plus a small LangChain RAG over public central-bank text.

**Deliverables**
1. GitHub repo with clean structure, tests and CI.
2. Results table of 4 models on one fixed test split, with bootstrap 95% CIs and McNemar tests.
3. Hugging Face model card for the fine-tuned DistilBERT.
4. Public Kaggle notebook summarising data, models and results (a notebook, not a competition).
5. `reading_log.md` with three paper summaries, `DEBUG_LOG.md` with one real bug per day.
6. `RESUME_NUMBERS.md` generated from results files, used to fill the resume template.

## 2. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, NumPy, `datasets` (Hugging Face) |
| Classical ML | scikit-learn (TF-IDF, LogisticRegression, KMeans, PCA for checking) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Keras | TensorFlow / Keras (BiLSTM or 1D-CNN) |
| PyTorch | PyTorch + Hugging Face Transformers, `distilbert-base-uncased` |
| LLM LoRA | Unsloth on a small instruct model (1B–3B), run on Colab/Kaggle GPU |
| RAG | LangChain + Hugging Face embeddings + FAISS (or Chroma) |
| Stats | statsmodels (McNemar), NumPy bootstrap |
| Tracking | MLflow (local `mlruns/`, not committed) |
| Quality | pytest, ruff, GitHub Actions |

**Version caution:** LangChain, Unsloth, TensorFlow and `datasets` change APIs often. Before writing code for each, check the installed version and the current official docs. Do not rely on memory of old APIs.

## 3. Where things run

Claude Code runs locally and usually has **no GPU**. So:

- **Local (CPU):** data loading, splits, TF-IDF baseline, K-Means, PCA, statistics, RAG indexing, tests, CI, and `--smoke` runs of every training script.
- **Colab / Kaggle (GPU):** full DistilBERT fine-tune (Day 5–6), Unsloth LoRA (Day 8–10), and the Keras run if local TensorFlow install fails.
- **Pattern:** every training script is a normal `scripts/*.py` with CLI args and a `--smoke` flag (tiny subset, 1 epoch, CPU, finishes in about a minute). `notebooks/colab_runner.ipynb` is a thin wrapper: clone repo, install `requirements-gpu.txt`, run the script, copy `results/` back (Drive or download), then I commit it. Never put logic only in a notebook.

## 4. Repo layout

```
fin-news-sentiment-lab/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── README.md
├── DEBUG_LOG.md
├── reading_log.md
├── RESUME_NUMBERS.md            # generated, never hand-edited
├── requirements.txt             # light deps, enough for CI
├── requirements-gpu.txt         # torch, transformers, unsloth etc. (Colab)
├── pytest.ini                   # defines marker: heavy
├── .gitignore                   # mlruns/, data/raw/, *.pt, checkpoints/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                     # gitignored
│   ├── splits/                  # train/val/test CSVs (small, committed)
│   └── rag_corpus/              # 10 public .txt docs + SOURCES.md
├── src/fnsl/
│   ├── __init__.py
│   ├── config.py                # seeds, paths, label map
│   ├── data.py                  # load, clean, split
│   ├── metrics.py               # macro-F1, confusion matrix, save_results()
│   ├── baselines.py             # TF-IDF + LogReg
│   ├── embeddings.py            # sentence-transformers wrapper
│   ├── pca.py                   # PCA via SVD in NumPy
│   ├── keras_model.py
│   ├── torch_model.py           # DistilBERT fine-tune
│   ├── llm_lora.py              # Unsloth LoRA
│   ├── stats.py                 # McNemar + bootstrap CI
│   └── rag/
│       ├── ingest.py
│       ├── retriever.py
│       └── evaluate.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_baseline.py
│   ├── run_kmeans.py
│   ├── train_keras.py
│   ├── train_distilbert.py
│   ├── train_unsloth.py
│   ├── compare_models.py
│   ├── build_rag.py
│   ├── eval_rag.py
│   └── make_results_table.py    # results/*.json -> README tables + RESUME_NUMBERS.md
├── notebooks/
│   ├── colab_runner.ipynb
│   └── kaggle_report.ipynb
├── results/
│   ├── metrics/                 # one JSON per model run
│   ├── predictions/             # one CSV per model
│   └── figures/
└── tests/
```

## 5. Conventions (apply every day)

- **Dataset (verify on Day 1):** primary `zeroshot/twitter-financial-news-sentiment` (3 labels: Bearish, Bullish, Neutral). Fallback `takala/financial_phrasebank` or a parquet mirror of it. Check that it loads with the current `datasets` library and read its license/terms. Record the dataset name, version and license in the README. If the primary dataset fails, pick the fallback and note why in `DEBUG_LOG.md`.
- **Fixed split:** created once on Day 1 with seed 42, saved to `data/splits/{train,val,test}.csv`. Every model reads these files. Nobody re-splits. The test set is used only for final evaluation, never for tuning.
- **Predictions schema** (`results/predictions/<model>.csv`): `id,text,label,pred`. The `id` matches the split file so McNemar can pair predictions.
- **Metrics schema** (`results/metrics/<model>.json`): `model, dataset, n_test, macro_f1, accuracy, per_class_f1, seed, train_time_s, params, git_commit, timestamp`.
- **Seeds:** set Python, NumPy and framework seeds in one `set_seed()` helper in `config.py`.
- **Commits:** one focused commit per day (more are fine), message format `<verb> <what>`, e.g. `add tfidf logreg baseline with mlflow`.
- **Tests:** CPU-only tests run in CI. Anything needing torch/tensorflow/unsloth is marked `@pytest.mark.heavy` and skipped in CI.
- **Daily close-out (last 10 minutes):** run `pytest -m "not heavy"` and `ruff check .`, add one bug and its fix to `DEBUG_LOG.md`, commit and push.

## 6. Day-by-day plan

Each day lists: **Goal → Build → Done when → Commit → I should be able to explain.** Claude Code: do the build, then stop and report.

### Week 1 — Data, classical ML, linear algebra, Keras, PyTorch, statistics

#### Day 1 — Scaffold, data, fixed split
- **Build:** create the layout above, `requirements.txt`, `pytest.ini`, `.gitignore`, a minimal `ci.yml` (install light deps, run `pytest -m "not heavy"` and `ruff`). Implement `data.py` (load, drop empty/duplicate texts, normalise whitespace) and `scripts/prepare_data.py` (seed-42 stratified train/val/test split, saved to `data/splits/`). Print class balance. Write tests: split sizes add up, no text overlap between splits, label set is as expected.
- **Done when:** split files exist, tests pass, CI is green.
- **Commit:** `add data loading and fixed split`
- **Explain:** why stratify, why a fixed split, why the test set stays untouched.

#### Day 2 — TF-IDF + logistic regression baseline
- **Build:** `baselines.py` and `scripts/run_baseline.py`. Fit on train, tune one or two settings on val only, evaluate on test. Implement `metrics.save_results()` producing the JSON and predictions CSV per the schemas. Log the run to MLflow. Save a confusion matrix figure.
- **Done when:** `results/metrics/tfidf_logreg.json` and its predictions CSV exist.
- **Commit:** `add tfidf logreg baseline with mlflow`
- **Explain:** what TF-IDF does, what macro-F1 measures and why it beats accuracy for imbalanced classes.

#### Day 3 — Clustering and PCA via SVD
- **Build:** `embeddings.py` (sentence-transformers, cache embeddings to disk). `scripts/run_kmeans.py`: K-Means with k=3 and one other k, report silhouette score and cluster-vs-label agreement (e.g. adjusted Rand index), save a 2-D plot. `pca.py`: PCA implemented from scratch with NumPy SVD (centre data, SVD, project, explained variance). Test it against `sklearn.decomposition.PCA` using `np.allclose` (allow for sign flips by comparing absolute values or explained variance).
- **Done when:** the PCA test passes and a cluster plot is saved.
- **Commit:** `add kmeans clustering and numpy pca with test`
- **Explain:** how SVD relates to PCA, and what clustering agreement says (and does not say) about sentiment.

#### Day 4 — Keras baseline
- **Build:** `keras_model.py` and `scripts/train_keras.py`: tokenisation or `TextVectorization`, embedding, BiLSTM (or 1D-CNN), softmax. Early stopping on val. `--smoke` flag. Save metrics and predictions in the standard schema. If local TensorFlow install fails, document it in `DEBUG_LOG.md` and run this script in Colab.
- **Done when:** a test-set predictions CSV exists for the Keras model.
- **Commit:** `add keras baseline`
- **Explain:** what the BiLSTM layers do, what early stopping prevents.

#### Day 5 — PyTorch DistilBERT training script
- **Build:** `torch_model.py` and `scripts/train_distilbert.py`: tokenizer, `Dataset`/`DataLoader`, explicit PyTorch training loop (not the `Trainer` API, so I can explain it), AdamW, linear LR schedule, mixed precision if a GPU is present, val evaluation each epoch, best-checkpoint saving. `--smoke` run on CPU must work. Write `notebooks/colab_runner.ipynb` and run the full training on Colab.
- **Done when:** the smoke run passes locally and the Colab run shows falling training loss.
- **Commit:** `add pytorch distilbert training script`
- **Explain:** each step of the training loop, what the learning-rate schedule does.

#### Day 6 — Evaluate DistilBERT, consolidate results, reading log
- **Build:** finish evaluation, save metrics and predictions, log to MLflow. `scripts/make_results_table.py` reads `results/metrics/*.json` and prints one markdown table of all models so far. Write the first `reading_log.md` entry: FinBERT (Araci, 2019), 5–8 sentences in my own words, including what it implies for this project.
- **Done when:** the table shows TF-IDF, Keras and DistilBERT on the same test set.
- **Commit:** `evaluate distilbert, save predictions, reading log`
- **Explain:** why fine-tuned transformers usually beat bag-of-words, and any case here where they did not.

#### Day 7 — Statistical comparison
- **Build:** `stats.py` with (a) McNemar's test between two models' paired predictions (use `statsmodels`, report the contingency counts and p-value) and (b) a bootstrap 95% CI for macro-F1 (resample the test set, 1,000+ resamples, fixed seed). `scripts/compare_models.py` runs all model pairs and writes `results/metrics/comparison.json`. If time allows, rerun DistilBERT with 3 seeds in Colab and report mean ± std. Tests: bootstrap CI contains the point estimate on a toy case; McNemar returns p=1 for identical predictions.
- **Done when:** the comparison file lists, for every pair, whether the difference is significant at 0.05.
- **Commit:** `add mcnemar and bootstrap comparison`
- **Explain:** what a p-value means here, why McNemar suits paired classifiers, why a CI is better than a single F1 number. Report non-significant differences honestly.

### Week 2 — LLM fine-tuning, RAG, publishing

#### Day 8 — Unsloth setup and zero-shot baseline
- **Build:** in Colab only. Check the current Unsloth install instructions and model list. Load a small instruct model (1B–3B, 4-bit). `llm_lora.py` with a prompt template (headline in, one label word out) and an output parser that maps text to a label (unparseable output counts as wrong, and is logged). Run zero-shot on 50 test headlines. Budget the whole hour for install friction.
- **Done when:** the model loads and produces parseable labels. If install fails twice, log it and move on.
- **Commit:** `add unsloth setup and zero-shot eval`
- **Explain:** what 4-bit quantisation is, and why the prompt template must match between training and inference.

#### Day 9 — LoRA fine-tune
- **Build:** `scripts/train_unsloth.py`: convert train split to instruction format, attach LoRA adapters (record rank, alpha, target modules in the metrics JSON), short training run (a few hundred steps is enough), save the adapter. Use val to monitor loss.
- **Done when:** the adapter is saved and the loss falls.
- **Commit:** `add unsloth lora finetune`
- **Explain:** what LoRA trains and what it freezes, why it is cheaper than full fine-tuning.

#### Day 10 — Evaluate LoRA, complete the comparison
- **Build:** run the full test set through the LoRA model, save predictions and metrics in the standard schema, add it to `compare_models.py` and rerun. Add the second `reading_log.md` entry: LoRA (Hu et al., 2021).
- **Done when:** the table and comparison cover four models.
- **Commit:** `add lora eval and comparison`
- **Explain:** whether the LLM beat DistilBERT, whether the difference is significant, and the cost trade-off.

#### Day 11 — LangChain RAG pipeline
- **Build:** put about 10 public central-bank or market commentary documents in `data/rag_corpus/` as `.txt` (for example FOMC statements, RBI monetary policy statements, ECB press releases; copy text manually) and list every source URL in `SOURCES.md`. Check current LangChain docs, then implement `rag/ingest.py` (chunk, embed with a Hugging Face model, build a FAISS index), `rag/retriever.py`, and a basic question-answering chain. A small open model or the retrieval-only output is fine as the generator; retrieval quality is what I will evaluate.
- **Done when:** a sample question returns relevant chunks and an answer that cites its source file.
- **Commit:** `add langchain rag pipeline`
- **Explain:** chunking, embeddings, vector search, and why retrieval can fail.

#### Day 12 — RAG evaluation
- **Build:** I write 20–30 questions by hand in `data/rag_eval.csv` (`question,source_file`), each answerable from exactly one document. `rag/evaluate.py` computes hit@1 and hit@3 (is the correct source file among the top-k retrieved chunks). Compare two chunk sizes and keep the better one. Save to `results/metrics/rag_eval.json`.
- **Done when:** hit@3 is reported for two chunk sizes.
- **Commit:** `add rag eval hit@3`
- **Explain:** what hit@k measures and what it does not (answer correctness).

#### Day 13 — README, Kaggle notebook, reading log
- **Build:** README with: problem, dataset and license, repo layout, how to reproduce (local and Colab), results table (auto-generated), comparison findings, a **Limitations** section (small dataset, short texts, single split, no market-return validation). `notebooks/kaggle_report.ipynb`: data overview, model results, comparison plot; no secrets, no private paths. Publish it publicly on Kaggle. Add the third reading-log entry: the original RAG paper (Lewis et al., 2020).
- **Done when:** the notebook is public and the README renders properly on GitHub.
- **Commit:** `add readme, results, reading log`
- **Explain:** every limitation I list, because interviewers will probe them.

#### Day 14 — Publish model, CI, resume numbers
- **Build:** push the DistilBERT model and a model card (task, data, metrics with CI, limitations, intended use) to the Hugging Face Hub. Make sure CI is green. Extend `make_results_table.py` to write `RESUME_NUMBERS.md` from the results JSONs. Then fill the commented project block in my LaTeX resume using only those numbers.
- **Done when:** CI is green, the model card is live, and `RESUME_NUMBERS.md` exists.
- **Commit:** `add ci, model card link, final cleanup`

## 7. CLAUDE.md contents

Create `CLAUDE.md` in the repo root with this text:

```
# CLAUDE.md
Project: fin-news-sentiment-lab. Full spec in PROJECT_PLAN.md.

Rules
- Work on one day of the plan at a time. At the end, report what was built, the test results, and the commit hash, then wait.
- Never invent metrics. All numbers come from results/metrics/*.json written by scripts.
- Do not modify files outside this repo. Do not touch my other projects.
- Check current library docs/versions before using LangChain, Unsloth, TensorFlow, datasets, or transformers APIs.
- Every training script needs a --smoke flag (tiny data, 1 epoch, CPU).
- Fixed splits in data/splits/ are read-only after Day 1. Test set is for final evaluation only.
- Keep code simple and commented; I must be able to explain it.
- Run `pytest -m "not heavy"` and `ruff check .` before every commit.
- GPU work goes through notebooks/colab_runner.ipynb; logic lives in scripts/ and src/.
- One focused commit per day, message "<verb> <what>".
```

## 8. Supporting templates

**DEBUG_LOG.md entry**
```
## Day N — <date>
- Symptom:
- Cause:
- Fix:
- What I learned:
```

**reading_log.md entry**
```
## <Paper title> (<authors>, <year>)
- Problem it solves:
- Core idea (my words):
- Result that matters:
- How it relates to this project:
```

**CI sketch (`.github/workflows/ci.yml`)**
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt ruff pytest
      - run: ruff check .
      - run: pytest -m "not heavy"
```

**Hugging Face model card outline:** task, base model, training data and license, hyperparameters, test macro-F1 with 95% CI, comparison to baselines, limitations (short English financial text, 3 classes, not for trading decisions), how to load and run.

## 9. Resume mapping (fill only after Day 14)

Use only numbers from `RESUME_NUMBERS.md`. Delete any bullet whose work was not done.

```
Financial News Sentiment and Market-Comment Lab | GitHub | Kaggle/HF | Nov 2026 – Present
Python, PyTorch, TensorFlow/Keras, scikit-learn, Hugging Face, Unsloth, LangChain, MLflow
- Fine-tuned DistilBERT in PyTorch for financial news sentiment (macro-F1 [X] vs [X] for a TF-IDF + logistic regression baseline); compared four models with McNemar's test and bootstrap confidence intervals.
- LoRA fine-tuned a small LLM with Unsloth for the same task ([X] macro-F1); built a LangChain RAG Q&A over public central bank text (hit@3 of [X] on [X] labeled questions).
- Published a Kaggle notebook and a Hugging Face model card; [X] pytest tests, CI on every push.
```

Skills line rules: add **PyTorch, TensorFlow/Keras** after Days 4–6, **Hugging Face** after Day 5–6, **Unsloth** after Day 10, **LangChain** after Day 12. Add "Hypothesis testing (McNemar, bootstrap CI)" after Day 7.

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Local TensorFlow install fails | Run `train_keras.py` in Colab; log in `DEBUG_LOG.md`. |
| Colab GPU disconnects | Save checkpoints and prediction CSVs after every run; keep runs under 30 minutes. |
| Unsloth install or model load fails twice | Stop at Day 8, skip Days 9–10, do not claim Unsloth. Use the freed time for a second seed run on Day 7 or a longer README. |
| Dataset fails to load | Use the fallback dataset; note it in the README. |
| Time runs short | Minimum viable path: Days 1–7, 13, 14. This still covers scikit-learn, pandas, NumPy, Keras, PyTorch, Hugging Face, hypothesis testing, Git and a published notebook. |
| Small test set gives wide CIs | Report them. Do not hide them or tune on the test set. |

## 11. Kickoff prompt (paste after this file)

> Read PROJECT_PLAN.md completely. Create the repo scaffold and CLAUDE.md as specified, then do Day 1 only. Verify the dataset loads and note its license. When done, run the tests, commit, and report: files created, test results, class balance, split sizes, commit hash. Then wait for me.

For each later day, I will say: *"Start Day N."*
