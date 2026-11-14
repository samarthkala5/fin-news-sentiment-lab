# DEBUG_LOG.md

One real bug (or surprise) per day: what broke, why, how it was fixed.
Entries marked *(skeleton build)* were hit while the project skeleton was generated;
add your own entries as you run and extend the code.

Template:

```
## Day N — <date>
- Symptom:
- Cause:
- Fix:
- What I learned:
```

---

## Day 1 — 2026-11-14 — near-duplicate tweets would leak across splits *(skeleton build)*
- Symptom: the raw data (11,931 rows) has **no exact duplicate texts**, but after
  lower-casing and removing `https://t.co/...` links, 152 rows repeat an earlier headline.
  3 of those groups even carry different labels, and 3 tweets are nothing but a URL.
- Cause: the same headline is often tweeted several times with a different short link,
  so exact-match de-duplication misses it. A random split would put copies of the same
  headline in both train and test, which leaks information and inflates test scores.
- Fix: `data.dedup_key()` (lowercase, strip URLs, collapse whitespace). `data.clean()` drops
  URL-only rows (3), every row of a conflicting-label group (6) and later copies of
  duplicates (147), leaving 11,775 rows (counts in `data/splits/split_report.json`).
  `tests/test_data.py::test_no_overlap_between_splits` checks that splits share no ids,
  texts or dedup keys.
- What I learned: check for *near*-duplicates before splitting; leakage can hide behind
  differences that look harmless, such as a URL.

## Day 1 — 2026-11-14 — fallback dataset no longer loads *(skeleton build)*
- Symptom: `load_dataset("takala/financial_phrasebank", "sentences_allagree")` raises
  `RuntimeError: Dataset scripts are no longer supported, but found financial_phrasebank.py`.
  Trying `revision="refs/convert/parquet"` fails too (that branch does not exist).
- Cause: `datasets` 4.x removed support for Python loading scripts, and this dataset
  is script-based.
- Fix: the fallback in `config.py` is now the parquet mirror `atrost/financial_phrasebank`
  (pinned revision), with its labels remapped (negative→Bearish, positive→Bullish,
  neutral→Neutral) in `data.FALLBACK_LABEL_MAP`. The primary dataset loads fine, so the
  fallback is not used.
- What I learned: check that "plan B" actually works with today's library versions;
  pin dataset revisions so a silent upstream change cannot alter the data.

## Day 3 — 2026-11-14 — embedding cache crashed under concurrent runs *(skeleton build)*
- Symptom: when several processes embedded the same texts at once, some crashed at the end
  of `embed_texts` with `FileNotFoundError` on `<name>.tmp.npy` (3 of 18 runs in a
  concurrency test).
- Cause: the cache is saved "atomically" (write a temp file, then rename it into place),
  but every process used the same temp file name. One process renamed the shared temp
  file away while another was still writing it or about to rename it.
- Fix: each writer uses its own temp file, `<stem>.<pid>.tmp.npy`, before
  `Path.replace()` (`src/fnsl/embeddings.py`). Afterwards 18 of 18 and 32 of 32
  concurrent runs succeeded.
- What I learned: write-then-rename is only atomic if every writer has its own temp name.

## Day 4 — 2026-11-14 — t.co links flooded the Keras vocabulary *(skeleton build)*
- Symptom: with Keras's default standardisation the train vocabulary had 21,034 distinct
  tokens, 4,292 of them t.co links such as `httpstcobd0xbfgjkt` (each seen about once).
  That is more than `max_tokens=20000`, so the cap would drop real rare words while
  keeping link junk.
- Cause: `lower_and_strip_punctuation` deletes `://` and `/`, so every URL becomes one
  unique "word"; 46% of train tweets contain a URL.
- Fix: a custom `standardize_text` (registered with
  `keras.saving.register_keras_serializable`) removes URLs before stripping punctuation.
  The vocabulary drops to 16,742 words. Because the step lives inside the model, the saved
  `.keras` file applies the same cleaning at prediction time.
- What I learned: look at the learned vocabulary; preprocessing inside the model keeps
  training and inference identical.

## Day 5 — 2026-11-14 — stale checkpoint metadata after an interrupted run *(skeleton build)*
- Symptom: after a simulated interrupted training run, `train_distilbert.py --eval-only`
  wrote metrics claiming lr 0.001 and 1 epoch for weights that were trained with lr 5e-3.
- Cause: `checkpoints/<run>/best/` is reused across runs. The new run saved its epoch-1
  weights there, but `training_info.json` was only rewritten after the whole training
  loop, so the previous run's metadata stayed next to the new weights.
- Fix: `train()` in `scripts/train_distilbert.py` deletes `best/training_info.json`
  before training starts, and
  `push_to_hub.py --push` refuses a checkpoint whose `training_info.json` does not match
  `results/metrics/distilbert.json`. Regression tests simulate the disconnect and the
  mismatched push.
- What I learned: files that describe each other (weights and their metadata) must be
  invalidated together; test the interrupted path, not only the happy path.

## Day 7 — 2026-11-14 — McNemar p-value was not 1 for an even split *(skeleton build)*
- Symptom: in a smoke-run comparison (a pipeline check, not a result) a pair with perfectly
  balanced disagreements, b = 15 and c = 15, got McNemar p = 0.855 instead of 1.
- Cause: statsmodels 0.15 `mcnemar(exact=False, correction=True)` applies the continuity
  correction (|b − c| − 1)² / (b + c) even when b = c, which gives 1/30 instead of 0.
  With b = c = 0 it is (0 − 1)² / 0 = 1/0 = infinity, so statsmodels returns p = 0.0:
  identical predictions would look maximally significant.
- Fix: `fnsl.stats.mcnemar_test` returns p = 1.0 directly when b == c and uses
  statsmodels otherwise. Tests cover 3/3 (exact test), 15/15 (chi-square) and identical
  predictions.
- What I learned: check a library's edge cases (b = c, b + c = 0) against the formula.

## Day 8 — 2026-11-14 — "greedy" decoding still applied a repetition penalty *(skeleton build)*
- Symptom: simulating how transformers 5.17 merges generation settings for
  Qwen2.5-1.5B-Instruct showed `repetition_penalty=1.1` (plus temperature, top_p, top_k)
  still active after passing only `do_sample=False`.
- Cause: transformers fills every unset generation field from the model's
  `generation_config.json`, and Qwen2.5 sets `repetition_penalty=1.1`. The penalty applies
  even to greedy decoding, and the label words appear in the prompt, so the answers
  Bearish / Bullish / Neutral would be penalised.
- Fix: `llm_lora.generate_labels` passes `repetition_penalty=1.0` and
  `temperature=None, top_p=None, top_k=None` together with `do_sample=False`. Found before
  any GPU run, so no result was affected.
- What I learned: greedy decoding is more than `do_sample=False`; read the model's own
  generation config.

## Day 9 — 2026-11-14 — TRL's SFTTrainer crashed on CPU *(skeleton build)*
- Symptom: the first CPU run of `train_lora` crashed in SFTTrainer's first evaluation
  pass with `AttributeError: 'NoneType' object has no attribute 'apply'`
  (`trl/trainer/utils.py`, `_fused_forward`).
- Cause: TRL 1.15 always computes the SFT loss with a Triton GPU kernel. The CPU build of
  torch ships no Triton, so TRL's guarded import set that function to `None`, and
  `SFTConfig` has no switch to turn it off.
- Fix: `llm_lora.sft_trainer_class(use_cuda)` uses TRL's SFTTrainer unchanged on a GPU.
  On CPU (smoke runs) it uses a 3-line subclass whose `compute_loss` calls the plain
  `transformers.Trainer.compute_loss`: the same cross-entropy on the same labels, with
  prompt tokens already set to -100 by TRL.
- What I learned: a "CPU smoke test" can hit code paths that only exist for the GPU; keep
  the workaround tiny and limited to the CPU.

## Day 11 — 2026-11-14 — header-only chunks beat the real passages *(skeleton build)*
- Symptom: asking "What did the Federal Reserve decide on September 18, 2024?" returned
  only `Title: Federal Reserve issues FOMC statement Date: March 15, 2020 Publisher: ...`,
  the wrong statement and no content.
- Cause: each corpus file starts with Title / Date / Publisher lines. `load_corpus` kept
  them in the indexed text, so the splitter made header-only chunks that look almost the
  same for every statement and match "Federal Reserve ... decide ... <date>" questions
  better than real passages.
- Fix: `load_corpus` moves those lines into metadata, so only statement text is chunked
  and embedded (93 instead of 100 chunks at chunk size 500). On 7 date questions the
  header chunk now ranks first 0 times (was 4). A regression test was added.
- What I learned: print and read your chunks; boilerplate can dominate similarity search.

## Day 12 — 2026-11-14 — an unquoted comma silently corrupted an eval question *(skeleton build)*
- Symptom: the hand-written question "What did the ECB decide on 6 June 2024, and why?"
  was loaded as the question "and why?", with no error or warning.
- Cause: the unquoted comma gave the first data row 3 fields under a 2-column header, so
  `pandas.read_csv` silently used the extra leading field as the row index.
- Fix: `rag.evaluate.load_eval_questions` reads the file with the `csv` module and
  requires exactly two filled-in cells per row; otherwise it raises a `ValueError` with
  the line number and a hint to put questions with commas in double quotes.
- What I learned: validate the shape of human-written input instead of trusting a parser
  that guesses.

## Day 13 — 2026-11-14 — a public notebook leaked a local path *(skeleton build)*
- Symptom: running `kaggle_report.ipynb` with no split files printed the absolute path
  of the missing `data/splits/train.csv` in the notebook output.
- Cause: the "not run yet" branch printed the exception text, and `fnsl.data` puts the
  full path into that text.
- Fix: print a repo-relative message instead. Notebooks are committed without outputs,
  and `tests/test_notebooks.py` rejects machine-specific paths (`/home/`, `/root/`,
  `/Users/`, `C:\`) and token-like strings anywhere in a notebook.
- What I learned: a notebook can leak machine details through its outputs, not only its
  code.
