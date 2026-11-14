# Mathematics, part 1: Data, the baseline and the statistics (section 8.1–8.3)

Part of the [project guide](../PROJECT_GUIDE.md#8-the-mathematics-part-by-part) for fin-news-sentiment-lab, Days 1, 2, 7 of `PROJECT_PLAN.md`. Checked against the code at commit `98a9d42`; the one later code change (`aaf142e`) makes the Unsloth resume claim require an adapter trained with `--backend unsloth`.

> **Numbers rule.** A *result* is copied from the results file named next to it. Every other number is a setting read from the code, a check you can rerun (named as such), or arithmetic shown in full. Models that have only had a `--smoke` run have no results, and none are stated.

Each concept follows the same pattern: the intuition, the mathematics (every symbol defined), how this code implements it, and why it matters or where it can go wrong. Formulas use GitHub's math rendering.

## Contents

- [8.1 Data, splits and evaluation metrics](#81-data-splits-and-evaluation-metrics)
  - [8.1.1 Cleaning and near-duplicate leakage (dedup_key)](#811-cleaning-and-near-duplicate-leakage-dedup_key)
  - [8.1.2 Stratified two-stage 70/15/15 split](#812-stratified-two-stage-701515-split)
  - [8.1.3 A fixed split and an untouched test set](#813-a-fixed-split-and-an-untouched-test-set)
  - [8.1.4 Confusion matrix](#814-confusion-matrix)
  - [8.1.5 Precision, recall and F1 per class](#815-precision-recall-and-f1-per-class)
  - [8.1.6 Macro-F1 versus accuracy under class imbalance](#816-macro-f1-versus-accuracy-under-class-imbalance)
  - [8.1.7 Unparseable predictions (UNPARSEABLE = -1)](#817-unparseable-predictions-unparseable---1)
  - [8.1.8 Answers to the plan's "Explain" questions (Days 1 and 2)](#818-answers-to-the-plans-explain-questions-days-1-and-2)
- [8.2 TF-IDF + logistic regression baseline](#82-tf-idf--logistic-regression-baseline)
  - [8.2.1 Tokens, n-grams and the vocabulary](#821-tokens-n-grams-and-the-vocabulary)
  - [8.2.2 Term frequency with sublinear scaling](#822-term-frequency-with-sublinear-scaling)
  - [8.2.3 Inverse document frequency, exactly as scikit-learn computes it](#823-inverse-document-frequency-exactly-as-scikit-learn-computes-it)
  - [8.2.4 L2 normalisation and the final feature vector](#824-l2-normalisation-and-the-final-feature-vector)
  - [8.2.5 Multinomial logistic regression: scores and softmax](#825-multinomial-logistic-regression-scores-and-softmax)
  - [8.2.6 Training objective: cross-entropy, the L2 penalty and the role of C](#826-training-objective-cross-entropy-the-l2-penalty-and-the-role-of-c)
  - [8.2.7 class_weight="balanced"](#827-class_weightbalanced)
  - [8.2.8 Hyperparameter selection on the validation split only](#828-hyperparameter-selection-on-the-validation-split-only)
  - [8.2.9 Results of the real run](#829-results-of-the-real-run)
  - [8.2.10 Answer to the plan's "Explain" question (Day 2)](#8210-answer-to-the-plans-explain-question-day-2)
- [8.3 Statistical comparison: bootstrap, McNemar, Holm, seed variation](#83-statistical-comparison-bootstrap-mcnemar-holm-seed-variation)
  - [8.3.1 Three kinds of uncertainty](#831-three-kinds-of-uncertainty)
  - [8.3.2 Percentile bootstrap confidence interval](#832-percentile-bootstrap-confidence-interval)
  - [8.3.3 Paired bootstrap of the macro-F1 difference](#833-paired-bootstrap-of-the-macro-f1-difference)
  - [8.3.4 McNemar's test](#834-mcnemars-test)
  - [8.3.5 The b = c rule (a statsmodels edge case)](#835-the-b--c-rule-a-statsmodels-edge-case)
  - [8.3.6 What a p-value means here](#836-what-a-p-value-means-here)
  - [8.3.7 Holm step-down correction](#837-holm-step-down-correction)
  - [8.3.8 Seed reruns: mean ± sample standard deviation](#838-seed-reruns-mean--sample-standard-deviation)
  - [8.3.9 From results files to README and RESUME_NUMBERS.md: the gates](#839-from-results-files-to-readme-and-resume_numbersmd-the-gates)
  - [8.3.10 Provenance: git_commit and the -dirty flag](#8310-provenance-git_commit-and-the--dirty-flag)
  - [8.3.11 Experiment tracking with MLflow (SQLite backend)](#8311-experiment-tracking-with-mlflow-sqlite-backend)
  - [8.3.12 Answers to the plan's "Explain" questions (Day 7)](#8312-answers-to-the-plans-explain-questions-day-7)
  - [8.3.13 What is left to do for this part](#8313-what-is-left-to-do-for-this-part)
- [Files this part reads and writes](#files-this-part-reads-and-writes)

---

## 8.1 Data, splits and evaluation metrics

> **Status at HEAD `98a9d42`: done, real run.** `scripts/prepare_data.py` built the fixed split once (commit `9c85ee4`), and `data/splits/` has not changed since (`git log -- data/splits` lists only that commit). Nothing is left to do here. Every number in this section is copied from `data/splits/split_report.json` or `results/metrics/tfidf_logreg.json`, is arithmetic on those numbers shown in full, or is a toy example labelled as one.

| Fact (from `data/splits/split_report.json`) | Value |
|---|---|
| Dataset | `zeroshot/twitter-financial-news-sentiment`, Hub revision `ccbe24de388e287beb92dd393a335c376b350ac3`, licence MIT |
| Raw rows (Hub `train` and `validation` splits pooled) | 11,931 |
| Rows dropped by cleaning | 0 empty, 3 URL-only, 6 in conflicting-label groups, 147 near-duplicates |
| Clean rows | 11,775 |
| Split (stratified, seed 42) | train 8,241 / val 1,767 / test 1,767 |
| Class balance (the same in every split, to one decimal) | Bearish 15.0%, Bullish 20.2%, Neutral 64.8% |
| Train class counts | Bearish 1,235, Bullish 1,666, Neutral 5,340 |
| Test class counts | Bearish 265, Bullish 357, Neutral 1,145 |

Label ids are fixed in `src/fnsl/config.py` → `LABELS`: 0 = Bearish, 1 = Bullish, 2 = Neutral. The dataset revision is pinned in `src/fnsl/config.py` → `DATASET_REVISION`, and `src/fnsl/data.py` → `load_raw` caches the download in the gitignored `data/raw/`. The Hub dataset has only a `train` and a `validation` split, no test split. Because the two are pooled and re-split, scores here cannot be compared directly with numbers reported on the Hub "validation" split (README, Dataset).

### 8.1.1 Cleaning and near-duplicate leakage (`dedup_key`)

**Intuition.** Many tweets are the same headline posted again with a different `t.co` short link. If one copy lands in train and another in test, the model is partly graded on sentences it has already seen, and the test score is inflated. So copies are found with a normalised key before splitting, and only one copy is kept.

**Math.** For a text $`t`$, define the key

```math
k(t) = \text{collapse}\Big(\text{lower}\big(\text{strip-urls}(t)\big)\Big)
```

where strip-urls replaces every match of the regular expression `https?://\S+` with a space, lower converts to lower case, and collapse turns every run of whitespace into one space and trims the ends. Two rows are near-duplicates when their keys are equal. Exact-key leakage is ruled out when no key appears in two splits: for any two different splits $`A`$ and $`B`$,

```math
\{\, k(t) : t \in A \,\} \cap \{\, k(t) : t \in B \,\} = \varnothing
```

**In this code.** `src/fnsl/data.py` → `normalize_whitespace`, `dedup_key`, `clean`. `clean` applies, in this order:

1. normalise whitespace and drop empty texts (0 rows);
2. drop rows whose key is empty, i.e. tweets that are only a URL (3 rows);
3. drop every row of a key group that carries more than one label, because the right label cannot be known (6 rows);
4. within each remaining key group, keep the first row in pooled order (Hub `train` rows come before `validation` rows) and drop the rest (147 rows).

It then numbers the 11,775 surviving rows 0 to 11,774 as `id`. The stored `text` is the whitespace-normalised original: case and URLs are kept, and the key is used only to find copies. `tests/test_data.py` → `test_clean_drops_empty_url_only_conflicts_and_duplicates` checks each rule on a toy frame, and `test_no_overlap_between_splits` checks that the committed splits share no `id`, no exact text and no dedup key. The bug story is in `DEBUG_LOG.md`, Day 1.

**Pitfall.** The key catches only copies that become identical after lower-casing and link removal. A reworded version of the same news (made-up example: "Credit Suisse cuts Cemex" vs "Cemex cut at Credit Suisse") is not caught, so the same story can still appear in two splits. Fuzzy matching (MinHash or embedding similarity) would be the next step.

### 8.1.2 Stratified two-stage 70/15/15 split

**Intuition.** Stratifying gives every split the same class proportions as the full data. Bearish is only 15% of the rows, so an unlucky random split could leave the test set with noticeably fewer Bearish rows. That would make the Bearish F1, and so macro-F1, noisier, and val and test less alike.

**Math.** Let $`n`$ be the number of clean rows and $`n_k`$ the number in class $`k`$. With a fractional `test_size`, scikit-learn's `train_test_split` takes $`n_{\text{test}} = \lceil 0.15\, n \rceil`$ rows (the ceiling rounds up) and spreads them over the classes so that $`n_{k,\text{test}} \approx n_{\text{test}}\, n_k / n`$. The second stage splits the remaining rows with a rescaled fraction, because 15% of the whole is a larger share of what is left:

```math
\begin{aligned}
\text{val share of the rest} &= \frac{0.15}{1 - 0.15} \approx 0.1765 \\
n_{\text{val}} &= \big\lceil 0.1765\,(n - n_{\text{test}}) \big\rceil
\end{aligned}
```

With $`n = 11{,}775`$: $`n_{\text{test}} = \lceil 1766.25 \rceil = 1767`$, 10,008 rows remain, $`n_{\text{val}} = \lceil 1766.1 \rceil = 1767`$, and train gets 8,241. These are exactly the sizes in `split_report.json`.

What stratification buys, as a calculation (not a measured number): in an unstratified random test set of $`n_{\text{test}} = 1767`$ rows drawn from 11,775 rows of which 1,765 are Bearish, the Bearish count follows a hypergeometric distribution with mean $`n_{\text{test}}\, p \approx 264.9`$ and standard deviation

```math
\sigma = \sqrt{n_{\text{test}}\; p\,(1-p)\; \frac{n - n_{\text{test}}}{n - 1}} \approx 13.8, \qquad p = \frac{1765}{11775}
```

So test sets with anywhere from about 237 to 293 Bearish rows ($`\pm 2\sigma`$) would be unremarkable. Stratification fixes the count at 265.

**In this code.** `src/fnsl/data.py` → `make_splits(df, fractions=None, seed=config.SEED)`, where `fractions=None` means `config.SPLIT_FRACTIONS` (0.70 / 0.15 / 0.15): first `train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)`, then `train_test_split(train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=42)`. It raises a `ValueError` if the fractions do not sum to 1 and sorts each split by `id`. `scripts/prepare_data.py` → `main` runs download → clean → split → save and writes `split_report.json`. The split's randomness comes only from `random_state=42`; `src/fnsl/config.py` → `set_seed` also seeds Python and NumPy globally (its `PYTHONHASHSEED` line only affects child processes, which does not matter here). Tests in `tests/test_data.py`: `test_splits_are_stratified` (each class share within 1 percentage point of the overall share), `test_split_sizes_add_up` (sizes match the report, sum to `n_clean`, each within 2 rows of its target), `test_make_splits_is_deterministic_and_complete`.

**Why it matters here.** The test set has only 265 Bearish rows, so the minority-class F1 scores are already the noisiest part of macro-F1. Stratification removes one avoidable source of that noise and keeps val and test alike, so a setting chosen on val behaves the same way on test.

### 8.1.3 A fixed split and an untouched test set

**Intuition.** All models are scored on the same 1,767 test rows, so differences come from the models and not from the luck of the split, and paired tests (8.3) can match predictions row by row. The test set is used once, at the end. If it is used to make choices, its score stops being an honest estimate.

**Math.** Suppose you try $`M`$ configurations and see test scores $`\hat F_m = F_m + \varepsilon_m`$, where $`F_m`$ is the true score of configuration $`m`$ and $`\varepsilon_m`$ is zero-mean sampling noise. The maximum is a convex function, so by Jensen's inequality

```math
\mathbb{E}\Big[\max_{m} \hat F_m\Big] \;\ge\; \max_{m}\, \mathbb{E}\big[\hat F_m\big] \;=\; \max_m F_m
```

The test score of the configuration that won on the test set is biased upwards (the "winner's curse"), and the bias grows with $`M`$ and with the noise. If the choice is made on a separate validation set, the test score of the chosen configuration is not inflated by the choice.

**In this code.** `src/fnsl/data.py` → `save_splits` raises `FileExistsError` if split files exist, unless `overwrite=True` (`scripts/prepare_data.py --overwrite`, whose help text says "avoid!"). `load_split` checks that the columns are exactly `id,text,label` and that the labels are in {0, 1, 2}. Every model reads the same files through `load_split` / `load_run_splits` and writes the split `id` next to each prediction, so `scripts/compare_models.py` → `align_pair` can pair models by id. The baseline tunes on val only (`src/fnsl/baselines.py` → `tune_on_val`) and each run predicts the test split once, after tuning (`scripts/run_baseline.py` → `main`). `CLAUDE.md` states the rule: the splits are read-only after Day 1, and the test set is for final evaluation only. Smoke runs use fixed small subsets of the same files (`src/fnsl/data.py` → `smoke_subset`; 200 / 64 / 64 rows from `config.SMOKE_SIZES`) and write to the gitignored `results/smoke/`.

**Why it matters here.** The baseline's best validation macro-F1 (0.733, `tfidf_logreg.json` → `params.val_macro_f1`) is the maximum of 10 noisy numbers, so it is optimistic in expectation; the test macro-F1 (0.739) is the number to report. Here the test score happens to be the higher of the two: the bias is an average effect, and a single run can land on either side. One fixed split cannot show how much scores move with a different split; the README Limitations section says so.

### 8.1.4 Confusion matrix

**Intuition.** A 3×3 table of counts. Rows are the true class and columns the predicted class. The diagonal holds the correct predictions; each off-diagonal cell is one specific kind of mistake.

**Math.** For $`n`$ test rows with true labels $`y_i`$ and predictions $`\hat y_i`$,

```math
M_{jk} = \#\{\, i : y_i = j \ \text{and}\ \hat y_i = k \,\}, \qquad j, k \in \{0, 1, 2\}
```

where $`\#`$ counts the rows. Row sums are the true class counts, column sums are how often each class was predicted, and $`\operatorname{trace}(M) = \sum_k M_{kk}`$ is the number of correct predictions.

**In this code.** `src/fnsl/metrics.py` → `compute_metrics` calls `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2])` and stores the result as `confusion_matrix` in the metrics JSON; `save_results` passes it to `save_confusion_matrix`, which draws `results/figures/tfidf_logreg_confusion.png`. The real matrix of the baseline (`tfidf_logreg.json`):

| true \ predicted | Bearish | Bullish | Neutral | row sum |
|---|---|---|---|---|
| Bearish | 177 | 21 | 67 | 265 |
| Bullish | 42 | 247 | 68 | 357 |
| Neutral | 75 | 73 | 997 | 1,145 |

The row sums equal the test class counts in `split_report.json`, which is a quick consistency check. The trace is 177 + 247 + 997 = 1,421 correct rows, and 1,421 / 1,767 gives the stored accuracy 0.804.

**Why it matters here.** Of the 346 wrong predictions, 283 involve Neutral (a Neutral headline called Bearish or Bullish, or the reverse) and only 63 confuse Bearish with Bullish (arithmetic on the matrix above). Most errors are about whether a headline carries sentiment at all, not about its direction.

### 8.1.5 Precision, recall and F1 per class

**Intuition.** Precision: when the model says "Bearish", how often is it right? Recall: of all truly Bearish headlines, how many did it find? F1 is the harmonic mean of the two, so it is high only when both are high.

**Math.** For class $`k`$:

```math
\begin{aligned}
TP_k &= M_{kk}, \qquad FP_k = \sum_{j \ne k} M_{jk}, \qquad FN_k = \sum_{j \ne k} M_{kj} \\[4pt]
P_k &= \frac{TP_k}{TP_k + FP_k}, \qquad R_k = \frac{TP_k}{TP_k + FN_k} \\[4pt]
F1_k &= \frac{2\,P_k R_k}{P_k + R_k} = \frac{2\,TP_k}{2\,TP_k + FP_k + FN_k}
\end{aligned}
```

$`TP_k`$ (true positives) is the diagonal cell. $`FP_k`$ (false positives) is the rest of column $`k`$: other classes predicted as $`k`$. $`FN_k`$ (false negatives) is the rest of row $`k`$: class $`k`$ predicted as something else. $`P_k`$ is precision and $`R_k`$ is recall.

Worked check on the real matrix (hand arithmetic; the JSON stores only the F1 column):

| Class | TP | FP | FN | Precision | Recall | F1 (= `per_class_f1`) |
|---|---|---|---|---|---|---|
| Bearish | 177 | 42 + 75 = 117 | 21 + 67 = 88 | 177 / 294 = 0.602 | 177 / 265 = 0.668 | 354 / 559 = 0.633 |
| Bullish | 247 | 21 + 73 = 94 | 42 + 68 = 110 | 247 / 341 = 0.724 | 247 / 357 = 0.692 | 494 / 698 = 0.708 |
| Neutral | 997 | 67 + 68 = 135 | 75 + 73 = 148 | 997 / 1132 = 0.881 | 997 / 1145 = 0.871 | 1994 / 2277 = 0.876 |

**In this code.** `src/fnsl/metrics.py` → `compute_metrics` uses `f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)` and stores `per_class_f1` by class name. scikit-learn computes F1 as $`2TP_k / (2TP_k + FP_k + FN_k)`$, so a class that is present but never predicted simply gets F1 = 0 (scikit-learn 1.9.1 raises no warning in that case). `zero_division=0` matters only when a class has no true rows and no predicted rows at all, which makes the F1 formula $`0/0`$: its F1 is then set to 0 silently, and because `labels=[0, 1, 2]` is fixed, that 0 still counts in the macro average. Toy example: true labels (1, 2, 2) with identical predictions give per-class F1 (0, 1, 1) and macro-F1 0.667, although every prediction is right. This cannot happen on the full test split, where all three classes are present, and `smoke_subset` keeps every label in the smoke subsets. `tests/test_metrics.py` → `test_compute_metrics_perfect_and_known_case` checks a hand-computed 6-row example.

**Pitfall.** Precision and recall trade off through the decision rule: predicting a minority class more often raises its recall and lowers its precision. `class_weight="balanced"` (8.2) does exactly this, so per-class precision and recall shift when it is switched on.

### 8.1.6 Macro-F1 versus accuracy under class imbalance

**Intuition.** Accuracy counts every row equally, so the majority class dominates it. Macro-F1 averages the three per-class F1 scores, so the 265 Bearish rows count as much as the 1,145 Neutral rows.

**Math.**

```math
\begin{aligned}
\text{accuracy} &= \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[\hat y_i = y_i] = \frac{\operatorname{trace}(M)}{n} \\
\text{macro-F1} &= \frac{1}{K}\sum_{k=1}^{K} F1_k, \qquad K = 3
\end{aligned}
```

$`\mathbf{1}[\cdot]`$ is 1 when the condition holds and 0 otherwise. Here $`n`$ is the number of test rows; if some predictions are unparseable (next part), the entries of $`M`$ add up to less than $`n`$, but the trace form still holds because those rows are never correct.

Calculation with the real test balance (`split_report.json`: 265 / 357 / 1,145 of 1,767) for a hypothetical classifier that always answers Neutral (hand arithmetic, not a run):

```math
\begin{aligned}
\text{accuracy} &= \frac{1145}{1767} \approx 0.648 \\
F1_{\text{Neutral}} &= \frac{2 \cdot 1145}{2 \cdot 1145 + 622 + 0} \approx 0.786 \\
\text{macro-F1} &= \frac{0 + 0 + 0.786}{3} \approx 0.262
\end{aligned}
```

Here 622 = 265 + 357 is the number of false positives for Neutral; Bearish and Bullish are never predicted, so their F1 is 0. The real baseline has accuracy 0.804 and macro-F1 0.739 (`tfidf_logreg.json`). Accuracy puts the useless always-Neutral model only 0.156 below the baseline (0.648 vs 0.804); macro-F1 puts it 0.477 below (0.262 vs 0.739), which matches how useless it is.

**In this code.** `src/fnsl/metrics.py` → `macro_f1` is `f1_score(..., labels=[0, 1, 2], average="macro", zero_division=0)`; `compute_metrics` computes accuracy as `np.mean(y_true == y_pred)`. Both go into every metrics JSON. Macro-F1 is the headline metric of the README table and the selection score in `tune_on_val`.

**Pitfall.** "Macro-F1" has two definitions in the literature: the mean of the per-class F1 scores (scikit-learn and this repo) and the F1 of macro-averaged precision and recall. They give different numbers, so say which one you mean. Macro-F1 is also not an average over rows, so it has no simple formula for its standard error; that is one reason the CIs in 8.3 use the bootstrap.

### 8.1.7 Unparseable predictions (`UNPARSEABLE = -1`)

**Intuition.** An LLM can answer something that is none of the three labels. Dropping those rows would flatter the LLM and break row-by-row pairing with the other models, so such answers are kept as a special prediction, −1, that is always wrong.

**Math.** If $`\hat y_i = -1`$, then $`\mathbf{1}[\hat y_i = y_i] = 0`$ in accuracy. In the per-class counts the row adds 1 to $`FN_{y_i}`$ (its true class was missed) but adds to no $`FP_k`$, because −1 is not one of the scored labels {0, 1, 2}.

**In this code.** `src/fnsl/config.py` → `UNPARSEABLE = -1`. `src/fnsl/metrics.py` → `compute_metrics` reports `n_unparseable`; the confusion matrix uses `labels=[0, 1, 2]`, so −1 rows are left out of it, and `save_results` says so in the figure title ("N unparseable outputs not shown"). `src/fnsl/stats.py` → `mcnemar_test` counts −1 as wrong because it never equals a label. `scripts/compare_models.py` → `load_all_predictions` accepts predictions in {0, 1, 2, −1} only, and `src/fnsl/reporting.py` → `results_section` adds the note "N unparseable outputs counted as wrong". Tests: `tests/test_metrics.py` → `test_unparseable_predictions_count_as_wrong`, `tests/test_stats.py` → `test_mcnemar_counts_unparseable_as_wrong`. The baseline cannot produce −1 (`n_unparseable` is 0 in `tfidf_logreg.json`); this path matters for the LLM models, which so far exist only as smoke runs.

**Pitfall.** In macro-F1 an unparseable answer is charged as a false negative only, while a wrong real label is charged as a false negative for the true class and a false positive for the predicted class. So −1 costs slightly less macro-F1 than a confident wrong answer. Toy example: true labels (0, 1, 2, 2) with predictions (0, 1, 2, −1) give macro-F1 0.889; predictions (0, 1, 2, 0) give 0.778. Accuracy (0.75 in both cases) and McNemar treat both simply as wrong.

### 8.1.8 Answers to the plan's "Explain" questions (Days 1 and 2)

- **Why stratify?** To give train, val and test the same class mix (15.0 / 20.2 / 64.8%, `split_report.json`). With only 15% Bearish, an unstratified test set of 1,767 rows would typically be off by about 14 Bearish rows (one standard deviation) and plausibly by up to about 28 (two), which changes the minority-class F1 and makes val a worse guide to test.
- **Why a fixed split?** So every model is scored on exactly the same rows. Differences are then due to the models, paired tests (McNemar, paired bootstrap) become possible because predictions match by `id`, and anyone can reproduce the numbers. The cost is that split-to-split variation is not measured.
- **Why does the test set stay untouched?** Any choice made by looking at test scores (a hyperparameter, a model, a prompt) turns the test score into the maximum of several noisy numbers, which is biased upwards. Choices are made on val; the test set is scored once per final model.
- **What does macro-F1 measure, and why is it better than accuracy for imbalanced classes?** It is the unweighted mean of the per-class F1 scores, so each class counts equally however rare it is. With Neutral at 64.8% of the test set, accuracy can look decent for a model that mostly predicts Neutral (always-Neutral: 0.648), while macro-F1 exposes it (0.262).

What TF-IDF does is answered at the end of 8.2.

## 8.2 TF-IDF + logistic regression baseline

> **Status at HEAD `98a9d42`: done, real run.** `scripts/run_baseline.py` wrote `results/metrics/tfidf_logreg.json`, `results/predictions/tfidf_logreg.csv` and `results/figures/tfidf_logreg_confusion.png` (code stamp `18e5768`); its bootstrap CI is in `results/metrics/comparison.json`. The baseline code (`src/fnsl/baselines.py`, `scripts/run_baseline.py`, `src/fnsl/data.py`, `src/fnsl/config.py`, `src/fnsl/tracking.py`) is unchanged from `18e5768` to HEAD (`src/fnsl/metrics.py` changed only inside `git_commit`), and refitting the chosen setting on train at HEAD gives the same 16,172-term vocabulary. Nothing is left to do here.

**The pipeline.** text → lower-case and tokenise → unigrams and bigrams → keep terms seen in at least 2 training texts → TF-IDF weights → scale each row to length 1 → multinomial logistic regression → softmax probabilities → predicted class = highest probability.

| Setting | Value | Source |
|---|---|---|
| `lowercase` | `True` | `src/fnsl/baselines.py` → `build_pipeline` |
| token pattern | `(?u)\b\w\w+\b` (scikit-learn default) | default |
| `ngram_range` | `(1, 2)` | `build_pipeline` |
| `min_df` | `2` (an integer, so an absolute count of documents) | `build_pipeline` |
| `sublinear_tf` | `True` | `build_pipeline` |
| `use_idf`, `smooth_idf`, `norm` | `True`, `True`, `"l2"` (defaults) | default |
| stop words, `max_df`, `max_features` | none, `1.0`, no cap (defaults) | default |
| classifier | `LogisticRegression`, solver `lbfgs`, L2 penalty via `l1_ratio=0.0` (defaults) | `build_pipeline` |
| `C` | grid 0.1, 0.3, 1.0, 3.0, 10.0; chosen **3.0** | `scripts/run_baseline.py` `--c-grid`; `tfidf_logreg.json` |
| `class_weight` | grid `None`, `"balanced"`; chosen **`"balanced"`** | `scripts/run_baseline.py`; `tfidf_logreg.json` |
| `max_iter` | `2000` (scikit-learn default is 100) | `build_pipeline` |
| `tol` | `1e-4` (default) | default |
| `random_state` | `42` (no effect with `lbfgs`) | `build_pipeline` |

Installed in the project venv: scikit-learn 1.9.1 (`requirements.txt` allows `>=1.5,<2`).

### 8.2.1 Tokens, n-grams and the vocabulary

**Intuition.** Each text is cut into tokens (words). Single tokens (unigrams) and pairs of neighbouring tokens (bigrams) become candidate features. Terms that occur in only one training text are dropped, because a weight learned from one example is mostly noise.

**Math.** Let $`D_{\text{train}}`$ be the $`N = 8{,}241`$ training texts. A text with tokens $`(w_1, \dots, w_L)`$ has the terms $`w_j`$ (unigrams, $`j = 1, \dots, L`$) and $`(w_j, w_{j+1})`$ (bigrams, $`j = 1, \dots, L-1`$). The document frequency of a term $`t`$ is

```math
\text{df}(t) = \#\{\, d \in D_{\text{train}} : t \text{ occurs in } d \,\}
```

and the vocabulary is $`V = \{\, t : \text{df}(t) \ge 2 \,\}`$.

**In this code.** `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)` in `src/fnsl/baselines.py` → `build_pipeline`. The default token pattern keeps runs of two or more letters, digits or underscores; bigrams are formed from the tokens that survive. Toy headline (checked with the vectoriser's analyser):

```text
$AAPL beats by 2.5% - I upgrade https://t.co/AbC123
→ aapl, beats, by, upgrade, https, co, abc123,
  aapl beats, beats by, by upgrade, upgrade https, https co, co abc123
```

The dollar sign of the cashtag, single characters (I, t, 2, 5) and punctuation disappear. The fitted vocabulary has 16,172 terms (`tfidf_logreg.json` → `n_features`).

**Pitfall.** The split files keep URLs (only the dedup key strips them), so `https`, `co` and the bigram `https co` are in the fitted vocabulary (checked by refitting on train). `min_df=2` removes almost all of the random link ids, but a few survive because the same short link appears in two training tweets, and "this tweet contains a link" survives as a feature. The Keras model strips URLs in its own preprocessing (`src/fnsl/keras_model.py` → `standardize_text`; `DEBUG_LOG.md`, Day 4), so the two models do not see exactly the same text.

### 8.2.2 Term frequency with sublinear scaling

**Intuition.** A word used three times in a text matters more than a word used once, but not three times as much. A logarithm dampens repetition.

**Math.** With $`c(t,d)`$ the raw count of term $`t`$ in text $`d`$,

```math
\text{tf}(t,d) = \begin{cases} 1 + \ln c(t,d) & \text{if } c(t,d) > 0 \\ 0 & \text{if } c(t,d) = 0 \end{cases}
```

Counts 1, 2, 3 become 1, 1.69, 2.10.

**In this code.** `sublinear_tf=True` in `build_pipeline`; scikit-learn takes the natural log of the non-zero counts and adds 1. Without it, tf would be the raw count.

**Why it matters here.** Tweets are short, so most counts are 0 or 1 and the effect is small; it mainly stops a repeated word or cashtag from dominating a row.

### 8.2.3 Inverse document frequency, exactly as scikit-learn computes it

**Intuition.** Terms that occur in many texts carry little information about any one text; rare terms carry more. idf multiplies each term by a factor that is larger the rarer the term is.

**Math.** scikit-learn's default (`smooth_idf=True`) is

```math
\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1
```

with $`N`$ the number of training texts and $`\text{df}(t)`$ the number of training texts that contain $`t`$. The $`+1`$ inside the fraction acts as if one extra document contained every term once, so there is never a division by zero. The $`+1`$ outside means a term that occurs in every document still gets weight 1 instead of 0. This differs from the textbook $`\ln\big(N / \text{df}(t)\big)`$.

Toy example: 4 texts, "stocks fall fall fall", "stocks rise", "bank cuts rates", "stocks fall", so $`N = 4`$. "stocks" is in 3 texts: $`\ln(5/4) + 1 \approx 1.223`$. "fall" is in 2: $`\ln(5/3) + 1 \approx 1.511`$. "rise" is in 1: $`\ln(5/2) + 1 \approx 1.916`$.

**In this code.** `use_idf=True` and `smooth_idf=True` are `TfidfVectorizer` defaults that `build_pipeline` does not change. The idf vector is learned in `fit` from the **training texts only** and reused unchanged on the validation and test texts; terms that are not in the training vocabulary are ignored. The formula was checked against the installed scikit-learn with this snippet (toy corpus, run outside the repo):

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["stocks fall fall fall", "stocks rise", "bank cuts rates", "stocks fall"]
vec = TfidfVectorizer(sublinear_tf=True).fit(docs)
n = len(docs)
df = np.array([sum(t in d.split() for d in docs) for t in vec.get_feature_names_out()])
assert np.allclose(vec.idf_, np.log((1 + n) / (1 + df)) + 1)  # passes
```

**Why it matters here.** idf is a statistic of the data. Fitting it on train only is part of keeping the test set untouched: computing it on train plus test would leak test information into the features.

### 8.2.4 L2 normalisation and the final feature vector

**Intuition.** After weighting, each text's vector is scaled to length 1, so that only the relative weights of its terms matter, not how long the text is.

**Math.**

```math
\begin{aligned}
v_t(d) &= \text{tf}(t,d)\,\text{idf}(t) \\
x(d) &= \frac{v(d)}{\lVert v(d) \rVert_2}, \qquad \lVert v \rVert_2 = \sqrt{\textstyle\sum_{t \in V} v_t^2}
\end{aligned}
```

$`x(d)`$ has one entry per vocabulary term (16,172 here) and is sparse: only the terms that occur in the text are non-zero.

Toy example continued, for "stocks fall fall fall": $`v = (1 \times 1.223,\ (1 + \ln 3) \times 1.511) = (1.223,\ 3.171)`$ for (stocks, fall), $`\lVert v \rVert_2 \approx 3.398`$, so $`x \approx (0.360,\ 0.933)`$ (matches the vectoriser's output).

**In this code.** `norm="l2"` is the `TfidfVectorizer` default. The vectoriser and the classifier sit in one `sklearn.pipeline.Pipeline` (`build_pipeline`), so `fit` learns the vocabulary and idf from the training texts and `predict` applies the same transformation to new texts.

**Why it matters here.** With unit-length rows, the class score $`w_k^\top x`$ in the next part is a weighted vote of the text's terms that does not grow with text length, and the dot product of two rows is their cosine similarity.

### 8.2.5 Multinomial logistic regression: scores and softmax

**Intuition.** The model keeps one weight per (term, class) pair. For a text it adds up the weights of the text's terms for each class, which gives three scores, and the softmax turns the scores into three probabilities that sum to 1.

**Math.** With $`K = 3`$ classes, a weight matrix $`W \in \mathbb{R}^{K \times |V|}`$ whose row $`w_k`$ belongs to class $`k`$, and intercepts $`b \in \mathbb{R}^{K}`$:

```math
\begin{aligned}
z_k(x) &= w_k^\top x + b_k \\
p(y = k \mid x) &= \frac{e^{z_k(x)}}{\sum_{j=1}^{K} e^{z_j(x)}} \\
\hat y &= \arg\max_k\, p(y = k \mid x) = \arg\max_k\, z_k(x)
\end{aligned}
```

A positive weight $`W_{kt}`$ means that term $`t`$ pushes the text towards class $`k`$. Adding the same vector to every $`w_k`$ (or the same constant to every $`b_k`$) does not change the probabilities, so only differences between classes are identified.

**In this code.** `LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=seed)` in `build_pipeline`. scikit-learn 1.9.1 has no `multi_class` option any more: with three classes and the default `lbfgs` solver it fits the full multinomial (softmax) model, not one-vs-rest. The fitted `coef_` has shape (3, 16172) and `intercept_` has shape (3,). Because of the L2 penalty, the fitted class weight vectors sum to zero across the three classes (checked on the refit, up to rounding error). `predict` returns the class with the highest score.

**Pitfall.** The probabilities of a regularised, class-weighted model are not calibrated: balanced class weights in particular push probability towards the minority classes. Use them to rank texts, not as "the chance this tweet is Bearish", unless you calibrate them first.

### 8.2.6 Training objective: cross-entropy, the L2 penalty and the role of C

**Intuition.** Training chooses the weights that give the true labels high probability (low cross-entropy), while a penalty on large weights stops the model from memorising rare n-grams. $`C`$ sets the balance: small $`C`$ means a strong penalty and a simpler model; large $`C`$ means a weak penalty and a closer fit to the training data.

**Math.** scikit-learn's L2-penalised multinomial objective, with a weight $`s_i`$ for each training row $`i = 1, \dots, N`$:

```math
J(W, b) = C \sum_{i=1}^{N} s_i \big(-\ln p(y_i \mid x_i)\big) \;+\; \frac{1}{2}\lVert W \rVert_F^2,
\qquad
\lVert W \rVert_F^2 = \sum_{k=1}^{K}\sum_{t \in V} W_{kt}^2
```

and training finds $`\min_{W,b} J(W,b)`$. $`-\ln p(y_i \mid x_i)`$ is the cross-entropy of row $`i`$: 0 if the model gives the true class probability 1, and large if it gives it almost 0. The intercepts $`b`$ are not penalised. The gradient with respect to $`w_k`$ is

```math
\frac{\partial J}{\partial w_k} = C \sum_{i=1}^{N} s_i \big(p(y = k \mid x_i) - \mathbf{1}[y_i = k]\big)\, x_i \;+\; w_k
```

that is, "predicted minus actual, times the input", plus a pull of every weight towards zero. The objective is convex, so every local minimum is a global minimum.

**In this code.** `class_weight=None` means $`s_i = 1`$; `"balanced"` sets $`s_i`$ by class (next part). Internally scikit-learn minimises the same objective divided by $`C \sum_i s_i`$, that is a weighted mean loss plus $`\lVert W \rVert_F^2 / (2C\sum_i s_i)`$ (`sklearn/linear_model/_logistic.py`: `l2_reg_strength = 1.0 / (C * sw_sum)`, with `sw_sum` computed after the class weights are multiplied in), which has the same minimiser. The solver is L-BFGS, a quasi-Newton method, with `max_iter=2000` and the default tolerance `1e-4`. It is deterministic, so refitting on the same data gives the same model. From scikit-learn 1.8 the `penalty` argument is deprecated (to be removed in 1.10) and L2 comes from the default `l1_ratio=0.0`, which is why `build_pipeline` does not pass `penalty`. `random_state=42` is passed but unused: the scikit-learn docs say it is used only by the `sag`, `saga` and `liblinear` solvers, and refitting with another seed gives identical coefficients.

**Why it matters here.** The validation grid shows the strong-penalty failure mode: at $`C = 0.1`$ without class weights, val macro-F1 is only 0.392, against 0.717 at $`C = 3`$ (`tfidf_logreg.json` → `val_grid`). With a strong penalty all weights shrink towards zero, so every text gets nearly the same scores; the unpenalised intercepts, which encode how common each class is, then decide, and the model predicts Neutral for most rows. (Confirmed by refitting that setting on train and predicting the validation split; the check is not saved in any results file, so no count is quoted.)

### 8.2.7 `class_weight="balanced"`

**Intuition.** Neutral has about 4.3 times as many training rows as Bearish, so an unweighted loss is dominated by Neutral. Balanced weights make every class contribute the same total weight to the loss, as if the classes were equally common.

**Math.** For class $`k`$ with $`N_k`$ of the $`N`$ training rows, every row of that class gets the weight

```math
s_k = \frac{N}{K\,N_k},
\qquad\text{so}\qquad
N_k\, s_k = \frac{N}{K}\ \text{for every class, and}\ \sum_{k} N_k\, s_k = N
```

(row $`i`$ gets $`s_i = s_{y_i}`$). With the train counts in `split_report.json` (1,235 Bearish, 1,666 Bullish, 5,340 Neutral; $`N = 8{,}241`$):

```math
\begin{aligned}
s_{\text{Bearish}} &= \frac{8241}{3 \cdot 1235} \approx 2.224 \\
s_{\text{Bullish}} &= \frac{8241}{3 \cdot 1666} \approx 1.649 \\
s_{\text{Neutral}} &= \frac{8241}{3 \cdot 5340} \approx 0.514
\end{aligned}
```

A mistake on a Bearish row costs about 4.3 times as much as a mistake on a Neutral row (2.224 / 0.514 = 5,340 / 1,235).

**In this code.** `LogisticRegression(class_weight="balanced")` computes `n_samples / (n_classes * np.bincount(y))` from the labels passed to `fit` (the train split) and multiplies each row's loss by the weight of its class. Because the balanced weights add up to $`N`$ over the rows, the internal penalty strength $`1/(C\sum_i s_i)`$ equals $`1/(CN)`$, exactly as without weights, so a given $`C`$ means the same penalty strength in both columns of the grid. Whether to use the weights is tuned: `class_weight_grid=[None, "balanced"]` in `scripts/run_baseline.py` → `main`.

**Why it matters here.** In the real grid, balanced beats unweighted at every $`C`$, for example 0.673 vs 0.392 at $`C = 0.1`$ and 0.733 vs 0.717 at $`C = 3`$ (`tfidf_logreg.json` → `val_grid`). That is expected when the metric is macro-F1 and one class is 65% of the data; the price is more false minority predictions and lower Neutral recall, which can cost accuracy.

### 8.2.8 Hyperparameter selection on the validation split only

**Intuition.** Try a small grid of settings, train each on train, score each on val, keep the best, and only then use the test set, once.

**Math.** With the grid $`\mathcal{G} = \{0.1, 0.3, 1, 3, 10\} \times \{\text{none}, \text{balanced}\}`$ (10 settings),

```math
(C^\star, \omega^\star) = \arg\max_{(C,\,\omega) \in \mathcal{G}}\ \text{macro-F1}\big(y^{\text{val}},\ f_{C,\omega}(X^{\text{val}})\big)
```

where $`f_{C,\omega}`$ is the pipeline trained on the training split with penalty parameter $`C`$ and class weighting $`\omega`$, and $`X^{\text{val}}`$, $`y^{\text{val}}`$ are the validation texts and labels.

**In this code.** `src/fnsl/baselines.py` → `tune_on_val(train_df, val_df, c_grid, class_weight_grid)` fits every setting on train and scores `macro_f1` on val; Python's `max` keeps the first best setting if there is a tie. `scripts/run_baseline.py` → `main` uses `--c-grid` (default `[0.1, 0.3, 1.0, 3.0, 10.0]`) and `class_weight_grid=[None, "balanced"]`, refits the chosen setting **on train only** (not train + val), times that refit (`train_time_s`), predicts the test split once and calls `src/fnsl/metrics.py` → `save_results`. The n-gram range, `min_df`, `sublinear_tf` and `max_iter` are fixed, not tuned. The full grid is saved in `tfidf_logreg.json` → `val_grid`:

| C | val macro-F1, no class weights | val macro-F1, balanced |
|---|---|---|
| 0.1 | 0.392 | 0.673 |
| 0.3 | 0.563 | 0.691 |
| 1.0 | 0.676 | 0.725 |
| 3.0 | 0.717 | **0.733** (chosen) |
| 10.0 | 0.725 | 0.728 |

**Pitfall.** The overall best value lies inside the grid ($`C = 3`$, not the edge value 10), which is what you want; a best value at an edge means the grid should be extended. (Without class weights the score is still rising at the edge, $`C = 10`$, but that column does not hold the overall best.) Refitting on train only keeps the final model exactly the one that was validated, but leaves the 1,767 validation rows out of the final fit; refitting on train + val is a common alternative.

### 8.2.9 Results of the real run

From `results/metrics/tfidf_logreg.json`, with the CIs from `results/metrics/comparison.json` (2,000 bootstrap resamples):

| Test set, n = 1,767 | Value |
|---|---|
| Macro-F1 | 0.739 (95% CI 0.715 to 0.762) |
| Accuracy | 0.804 (95% CI 0.786 to 0.822) |
| F1 Bearish / Bullish / Neutral | 0.633 / 0.708 / 0.876 |
| Unparseable predictions | 0 |
| Vocabulary size | 16,172 |
| Refit time on train | 3.67 s (on the machine that ran it) |
| Code commit | `18e5768` |

Reading: Neutral, the largest class, is easiest; Bearish, the smallest, is hardest. The validation score of the chosen setting (0.733) lies inside the test CI, so the small val–test gap is consistent with ordinary sampling noise. Most errors involve Neutral (confusion matrix in 8.1).

**Why it matters here.** This is the bar every larger model must clear, and clear *significantly* (8.3). It trains in seconds on a CPU, it is deterministic, and its weights can be inspected term by term.

### 8.2.10 Answer to the plan's "Explain" question (Day 2)

- **What does TF-IDF do?** It turns each tweet into a fixed-length, sparse vector with one entry per vocabulary term (16,172 here). A term's entry is large when the term occurs in this tweet (tf, log-dampened) and is rare across the training tweets (idf); each vector is then scaled to length 1. A linear classifier on top learns which terms point to which class.
- **Macro-F1 versus accuracy:** answered at the end of 8.1.

## 8.3 Statistical comparison: bootstrap, McNemar, Holm, seed variation

> **Status at HEAD `98a9d42`: partly real.** `scripts/compare_models.py` has run on the real predictions, but only `tfidf_logreg` has real predictions, so `results/metrics/comparison.json` holds its bootstrap CIs and empty `pairs`, `skipped_pairs`, `seed_runs` and `skipped_models`. McNemar's test, the paired bootstrap and the Holm correction have run only in unit tests on synthetic data (`tests/test_stats.py`, `tests/test_compare_models.py`) and on smoke-run predictions (gitignored `results/smoke/`, never results). There is no real pairwise result yet, so no claim of the form "model X is significantly better" can be made.

Defaults (in `scripts/compare_models.py` → `parse_args`, and recorded in `comparison.json`): `--n-resamples 2000`, `--alpha 0.05`, `--seed 42`.

### 8.3.1 Three kinds of uncertainty

| Source of variation | Question it answers | Tool in this repo | State |
|---|---|---|---|
| Test-set sampling | How much would the score move on another 1,767 headlines from the same source? | Bootstrap CI; paired bootstrap; McNemar | CI real for `tfidf_logreg`; pairwise tools tested, not yet used on real data |
| Training randomness | How much does it move if the same model is retrained with another seed? | Seed reruns, mean ± sample std | code only, no reruns |
| Choice of split | How much does it move with a different train/val/test split? | Not measured (one fixed split) | README limitation |

### 8.3.2 Percentile bootstrap confidence interval

**Intuition.** There is only one test set, but we want to know how much the score would wobble on other test sets like it. The bootstrap treats the test set as the population: draw 1,767 rows from it with replacement (some rows twice, some not at all), rescore the fixed predictions, repeat 2,000 times, and read off the middle 95% of the scores.

**Math.** Test rows $`i = 1, \dots, n`$ have labels $`y_i`$ and fixed predictions $`\hat y_i`$; $`m(\cdot,\cdot)`$ is the metric (macro-F1 or accuracy); the point estimate is $`\hat\theta = m(y, \hat y)`$ on all rows. For $`b = 1, \dots, B`$:

```math
I_b = (i_{b,1}, \dots, i_{b,n}),\quad i_{b,j} \overset{\text{iid}}{\sim} \text{Uniform}\{1, \dots, n\},
\qquad
\hat\theta^{*}_b = m\big(y_{I_b},\, \hat y_{I_b}\big)
```

```math
\text{CI}_{1-\alpha} = \Big[\, Q_{\alpha/2}\big(\hat\theta^*_1, \dots, \hat\theta^*_B\big),\ \ Q_{1-\alpha/2}\big(\hat\theta^*_1, \dots, \hat\theta^*_B\big) \,\Big]
```

$`I_b`$ is the list of row numbers in resample $`b`$, $`\hat\theta^*_b`$ the metric on that resample, and $`Q_q`$ the empirical $`q`$-quantile. With $`\alpha = 0.05`$ these are the 2.5th and 97.5th percentiles. The model is **not** retrained; only the evaluation rows are resampled.

**In this code.** `src/fnsl/stats.py` → `bootstrap_ci(y_true, y_pred, metric=macro_f1, n_resamples=2000, alpha=0.05, seed=42)`: `rng = np.random.default_rng(seed)`, one `rng.integers(0, n, size=n)` per resample, quantiles from `np.quantile` (default linear interpolation). It returns `point` (the metric on all rows, not the bootstrap mean), `low`, `high`, `n_resamples`, `alpha` and `method="percentile"`. `scripts/compare_models.py` → `summarize_model` calls it for macro-F1 and for accuracy with the same seed. Real result (`comparison.json`): `tfidf_logreg` macro-F1 0.739, 95% CI [0.715, 0.762]; accuracy 0.804, 95% CI [0.786, 0.822]. Because the seed is fixed, rerunning `bootstrap_ci` on the committed `results/predictions/tfidf_logreg.csv` reproduces these values exactly. Tests in `tests/test_stats.py`: `test_bootstrap_ci_contains_point_estimate`, `test_bootstrap_ci_narrows_with_more_data` (ten times the rows gives a CI less than half as wide, as $`1/\sqrt{n}`$ scaling predicts), `test_bootstrap_is_reproducible_with_a_seed_and_takes_any_metric`.

**What it means, and what it does not.**

- It means: a range of plausible values for this trained model's macro-F1 on headlines like these. If the whole procedure were repeated on fresh test sets, about 95% of the intervals would contain the true value (approximately: the bootstrap is an approximation).
- It does not mean "there is a 95% probability that the true macro-F1 is between 0.715 and 0.762". In the frequentist view the true value is fixed and this interval either contains it or not; the 95% describes the procedure.
- It does not include training randomness (another seed), the effect of the hyperparameter search, or a different split.
- It is not a test between two models. Two models' CIs can overlap while a paired test still finds a significant difference, because separate CIs ignore that both models were scored on the same rows.
- It assumes the test rows are independent draws. Remaining near-copies of the same story would make it too narrow.

**Why it matters here.** The CI half-width is about 0.023 macro-F1 ((0.7619 − 0.7152) / 2, from the unrounded values in `comparison.json`), so the single number 0.739 really means "somewhere around 0.715 to 0.762". Toy example: if another model scored 0.750, its single number would look better, but a 0.011 gap cannot be called from per-model CIs of this width; the paired methods below are needed, and they are more sensitive because they cancel the difficulty of each row.

### 8.3.3 Paired bootstrap of the macro-F1 difference

**Intuition.** To compare models A and B, resample the test rows as before, but use the **same** resampled rows for both models and record the difference. A resample that happens to contain many easy rows helps both models, so that luck cancels in the difference.

**Math.** The observed difference is $`\hat D = m(y, \hat y^{A}) - m(y, \hat y^{B})`$. For each resample $`I_b`$:

```math
D^*_b = m\big(y_{I_b},\, \hat y^{A}_{I_b}\big) - m\big(y_{I_b},\, \hat y^{B}_{I_b}\big)
```

The CI is the percentile interval of $`D^*_1, \dots, D^*_B`$. Pairing helps because

```math
\operatorname{Var}(\hat D) = \operatorname{Var}(\hat\theta_A) + \operatorname{Var}(\hat\theta_B) - 2\,\operatorname{Cov}(\hat\theta_A, \hat\theta_B)
```

and the covariance is positive when both models find the same rows easy or hard. The two-sided p-value, exactly as the code computes it:

```math
p = \min\Big(1,\ 2\,\min\Big(\frac{1 + \#\{b : D^*_b \le 0\}}{B + 1},\ \frac{1 + \#\{b : D^*_b \ge 0\}}{B + 1}\Big)\Big)
```

It measures how often the resampled difference lands on the other side of zero. The $`+1`$ terms keep $`p`$ above zero; with $`B = 2000`$ the smallest possible value is $`2/2001 \approx 0.001`$, so this method cannot report anything smaller.

Toy example with $`B = 9`$ resampled differences +0.02, +0.01, −0.01, +0.03, 0.00, +0.02, +0.01, +0.04, +0.02: two are $`\le 0`$ and eight are $`\ge 0`$ (0.00 counts on both sides), so $`p = \min\big(1,\ 2 \min(3/10,\ 9/10)\big) = 0.6`$.

**In this code.** `src/fnsl/stats.py` → `paired_bootstrap_diff(y_true, pred_a, pred_b, metric=macro_f1, n_resamples=2000, alpha=0.05, seed=42)` returns `diff`, `low`, `high`, `p_value`. `scripts/compare_models.py` → `compare_pair` stores it as `macro_f1_diff` for each pair, after `align_pair` has matched the two files by test `id` (never by row order) and checked that their labels agree. Tests: `tests/test_stats.py` → `test_paired_diff_of_model_with_itself_is_zero` (diff, low, high, p = 0, 0, 0, 1) and `test_paired_diff_detects_a_clearly_better_model` (p equals the floor $`2/501`$ for 500 resamples), plus `tests/test_compare_models.py` → `test_rows_are_aligned_by_id_not_by_file_order`.

**Pitfall.** This p-value is an approximation, not an exact test. The resamples are centred on the observed difference, not on "no difference", so it is essentially "does the CI exclude zero?" written as a p-value, and values near 0.05 can flip with another seed. It is stored in `comparison.json` and printed by the script, but it is not Holm-corrected and it does not decide the "significant" column, which comes from McNemar plus Holm.

### 8.3.4 McNemar's test

**Intuition.** Look only at the rows where exactly one of the two models is right. If the models are equally good, "A right, B wrong" and "A wrong, B right" should be about equally common, like heads and tails. A lopsided split is evidence that one model has the lower error rate.

**Math.** For row $`i`$ let $`r^A_i = \mathbf{1}[\hat y^A_i = y_i]`$ and $`r^B_i = \mathbf{1}[\hat y^B_i = y_i]`$ (1 = right, 0 = wrong). The 2×2 table is

```math
\begin{array}{c|cc}
 & B\ \text{right} & B\ \text{wrong} \\ \hline
A\ \text{right} & n_{11} & b \\
A\ \text{wrong} & c & n_{22}
\end{array}
\qquad
b = \sum_{i} r^A_i\,(1 - r^B_i), \qquad c = \sum_{i} (1 - r^A_i)\, r^B_i
```

$`n_{11}`$ and $`n_{22}`$ count rows where both are right or both are wrong; $`b`$ and $`c`$ are the discordant rows. The accuracy difference depends only on them: $`\text{acc}_A - \text{acc}_B = (b - c)/n`$. The null hypothesis $`H_0`$ is that the two models have the same error rate on the population of headlines, equivalently $`P(A\ \text{right}, B\ \text{wrong}) = P(A\ \text{wrong}, B\ \text{right})`$. Given $`b + c`$ discordant rows, under $`H_0`$

```math
b \sim \text{Binomial}\Big(b + c,\ \frac{1}{2}\Big)
```

The exact two-sided test is

```math
p_{\text{exact}} = \min\Big(1,\ 2\sum_{j=0}^{\min(b,c)} \binom{b+c}{j} \Big(\frac{1}{2}\Big)^{b+c}\Big)
```

and the large-sample version, with continuity correction (the $`-1`$ compensates for approximating a whole-number count with a continuous distribution), is

```math
\chi^2 = \frac{\big(|b - c| - 1\big)^2}{b + c},
\qquad
p_{\chi^2} = P\big(\chi^2_1 \ge \chi^2\big)
```

where $`\chi^2_1`$ is a chi-square random variable with 1 degree of freedom.

Toy examples: $`b = 7, c = 2`$ gives $`p_{\text{exact}} = 2\,(1 + 9 + 36)/512 \approx 0.180`$, not significant. $`b = 30, c = 12`$ gives $`\chi^2 = 17^2/42 \approx 6.88`$ and $`p \approx 0.0087`$.

**In this code.** `src/fnsl/stats.py` → `mcnemar_test(y_true, pred_a, pred_b, exact=None)` builds the table and, when $`b \ne c`$, calls `statsmodels.stats.contingency_tables.mcnemar(table, exact=exact, correction=True)` (statsmodels 0.15.0 in the venv). With `exact=None` it uses the exact binomial test when $`b + c < 25`$ (`EXACT_BELOW = 25`, a rule of thumb) and the corrected chi-square otherwise. It returns `table`, `b`, `c`, `statistic` ($`\min(b,c)`$ for the exact test, $`\chi^2`$ otherwise), `p_value` and `method`. `scripts/compare_models.py` → `build_comparison` runs it (through `compare_pair`) for every pair of distinct models with identical test ids and labels; pairs that cannot be aligned go to `skipped_pairs` with the reason. Tests in `tests/test_stats.py`: `test_mcnemar_known_table_matches_binomial_test` (against `scipy.stats.binomtest`), `test_mcnemar_uses_chi_square_for_many_discordant_rows` (against the formula and `scipy.stats.chi2`), `test_mcnemar_is_symmetric_in_the_two_models`.

**Why McNemar suits paired classifiers.** Both models label the same rows, so their right/wrong outcomes come in pairs and are correlated: hard headlines tend to defeat both. A test that treats the two accuracies as independent samples (the two-proportion z-test) ignores the pairing and gets the variance wrong. McNemar conditions on the discordant rows, where the difficulty of each row cancels out; it needs one test set and no retraining, and Dietterich (1998) found it to have an acceptable false-positive rate in exactly this setting, where each model can be trained only once.

**Pitfall.** McNemar compares error rates (accuracy), not macro-F1. A model can win on accuracy by doing better on Neutral while losing macro-F1 on the rare classes, so the paired macro-F1 bootstrap is reported next to it, and the README notes that the two can disagree. McNemar is also conditional on the two trained models: it says nothing about what happens if they are retrained.

### 8.3.5 The b = c rule (a statsmodels edge case)

**Intuition.** If the disagreements split exactly evenly, the data look exactly like "equally good", so the p-value must be 1.

**Math.** Under $`H_0`$ the expected value of $`b`$ is $`(b + c)/2`$, so $`b = c`$ is the most likely outcome and the two-sided p-value is 1. The corrected formula, however, gives $`(|b - c| - 1)^2/(b + c) = 1/(b + c) > 0`$ when $`b = c`$. Toy example: $`b = c = 15`$ gives $`\chi^2 = 1/30 \approx 0.033`$ and $`p \approx 0.855`$ instead of 1. For identical predictions, $`b = c = 0`$, it gives $`1/0 = \infty`$ and $`p = 0`$: two identical models would look as different as possible.

**In this code.** `mcnemar_test` sets `p_value = 1.0` directly when `b == c` (statistic `b` for the exact test, `0.0` for the chi-square) and calls statsmodels only otherwise. statsmodels' exact test already returns 1 when $`b = c`$, so the guard matters for the chi-square path: with the automatic choice that path is taken only when $`b + c \ge 25`$, i.e. for an even split from $`b = c = 13`$ upwards, while $`b = c = 0`$ goes to the exact test and gives 1 anyway. A caller who forces `exact=False` would hit the problem for every $`b = c`$, including 0. R's `mcnemar.test` likewise skips the correction when $`b = c`$. The problem was found in a smoke-run comparison (`DEBUG_LOG.md`, Day 7). Tests: `test_mcnemar_identical_predictions_give_p_one` (for `exact` = None, True, False) and `test_mcnemar_even_split_of_discordant_rows_gives_p_one` (3/3 with the exact test, 15/15 with the chi-square).

**Why it matters here.** Two models with equally many mistakes have $`b = c`$ (because $`\text{acc}_A - \text{acc}_B = (b - c)/n`$). Without the rule, such a pair would get a p-value below 1 (0.855 at 15/15), which misstates "no evidence of a difference", and with `exact=False` a duplicated predictions file would get $`p = 0`$ and look as significant as possible. "I checked the library's edge cases against the formula" is a good interview story.

### 8.3.6 What a p-value means here

**Intuition.** The p-value says how surprising the observed pattern of disagreements would be if the two models were truly equally accurate.

**Math.**

```math
p = P\big(\text{a split of the discordant rows at least as lopsided as the observed } (b, c) \;\big|\; H_0\big)
```

If $`H_0`$ is true, "significant at 0.05" happens in at most about 5% of such tests by chance alone (the exact binomial test and the continuity-corrected chi-square test are slightly conservative, so usually a little less).

**In this code.** Each entry of `pairs` in `comparison.json` stores the raw McNemar p-value (`mcnemar.p_value`), `significant_at_0.05` (raw $`p < 0.05`$; the level is fixed in the key name), the Holm-adjusted `p_holm`, and `significant_holm` ($`p_{\text{holm}} < \alpha`$ with `--alpha`, default 0.05). The README column "Significant? (Holm)" shows "yes" or "no" from `significant_holm`, and the RESUME_NUMBERS.md significance lines print "significant" or "NOT significant" from the same field; `src/fnsl/reporting.py` → `fmt_p` prints p-values below 0.001 as `<0.001`.

**Pitfall.** A p-value is not the probability that $`H_0`$ is true, not "the probability that the result is due to chance", and not the size of the improvement. With 1,767 rows a tiny difference can be significant and a useful one can miss, so always give the effect size with its CI (the macro-F1 difference) next to the p-value, and report non-significant differences as non-significant (`PROJECT_PLAN.md`, Day 7).

### 8.3.7 Holm step-down correction

**Intuition.** Testing many pairs at the 0.05 level gives many chances of a false alarm. Holm's method raises the bar most for the smallest p-value and less for the others, so the chance of even one false "significant" across all pairs stays at or below 5%, while being less strict than multiplying every p-value by the number of tests (Bonferroni).

**Math.** With $`m`$ tests and raw p-values sorted as $`p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}`$, the Holm-adjusted p-values are

```math
\tilde p_{(i)} = \min\Big(1,\ \max_{j \le i}\ (m - j + 1)\, p_{(j)}\Big)
```

and a test is significant if $`\tilde p < \alpha`$. Equivalently: compare $`p_{(1)}`$ with $`\alpha/m`$, $`p_{(2)}`$ with $`\alpha/(m-1)`$, and so on, and stop at the first failure. Holm controls the family-wise error rate

```math
\text{FWER} = P(\text{at least one true null hypothesis is rejected}) \le \alpha
```

whatever the dependence between the tests, which matters here because pairs that share a model are not independent. With $`k`$ models on the same test ids there are $`m = k(k-1)/2`$ pairs (4 models give 6 pairs).

Toy example ($`m = 4`$, $`\alpha = 0.05`$), raw p-values in input order 0.010, 0.040, 0.012, 0.030:

| Step | Values |
|---|---|
| Sort ascending | 0.010, 0.012, 0.030, 0.040 |
| Multiply by 4, 3, 2, 1 | 0.040, 0.036, 0.060, 0.040 |
| Running maximum, cap at 1 | 0.040, 0.040, 0.060, 0.060 |
| Back in input order | 0.040, 0.060, 0.040, 0.060 |

All four raw p-values are below 0.05; after Holm only two pairs are significant. The running maximum lifts 0.036 to 0.040 so that a larger raw p-value never gets a smaller adjusted one. A second toy example shows why Holm beats Bonferroni: raw p-values 0.020, 0.010, 0.015 ($`m = 3`$) become 0.030, 0.030, 0.030 under Holm (all significant) but 0.060, 0.030, 0.045 under Bonferroni (two significant). (Both toy examples give the same output from `holm_adjust`.)

**In this code.** `src/fnsl/stats.py` → `holm_adjust(p_values)`: stable `argsort`, multiply by `m, m-1, ..., 1`, `np.maximum.accumulate`, cap at 1, and put the values back in input order; it rejects NaN and values outside [0, 1]. `scripts/compare_models.py` → `build_comparison` applies it to the McNemar p-values of all compared pairs; seed reruns are not in the family. Tests: `tests/test_stats.py` → `test_holm_on_known_values`, `test_holm_matches_statsmodels` (against `statsmodels.stats.multitest.multipletests(method="holm")`), and `tests/test_compare_models.py` → `test_holm_is_applied_across_all_pairs`.

**Why it matters here.** Day 7 of the plan asks for "significant at 0.05" for every pair. With 4 or 5 models (6 or 10 pairs), raw p-values would carry a sizeable chance of at least one false claim. As a rough calculation: if every null were true and the tests were independent, that chance would be $`1 - 0.95^{6} \approx 0.26`$ for 6 pairs and $`1 - 0.95^{10} \approx 0.40`$ for 10 (pairs that share a model are not independent, so this is only a guide). The Holm column is the honest verdict to put in the README.

### 8.3.8 Seed reruns: mean ± sample standard deviation

**Intuition.** A neural network gets a slightly different score each time it is retrained with a different random seed (weight initialisation, data order). Rerunning with a few seeds shows how large that wobble is. The reruns are summarised, not tested against each other.

**Math.** For $`R`$ runs with test macro-F1 $`f_1, \dots, f_R`$:

```math
\bar f = \frac{1}{R}\sum_{r=1}^{R} f_r,
\qquad
s = \sqrt{\frac{1}{R - 1}\sum_{r=1}^{R} \big(f_r - \bar f\big)^2}
```

Dividing by $`R - 1`$ rather than $`R`$ makes $`s^2`$ an unbiased estimate of the variance. With $`R = 3`$ the estimate $`s`$ is itself very uncertain (for normal data its relative standard error is roughly $`1/\sqrt{2(R-1)}`$, about 50% when $`R = 3`$), so treat it as a rough guide.

**In this code.** `scripts/compare_models.py` → `model_and_seed`, `find_seed_reruns`, `seed_summary`. A rerun is a predictions file named `<model>_seed<N>` (regular expression `^(?P<base>.+)_seed(?P<seed>\d+)$`); the training scripts name a run that way when `--seed` is not 42 (for example `scripts/train_distilbert.py --seed 43` writes `distilbert_seed43`), and the plain name is the seed-42 run. A rerun is counted only if it covers the same test ids with the same labels as the kept run; otherwise it stays an ordinary model so the mismatch is reported under `skipped_pairs`. Every run still gets its own bootstrap CI, but only the kept run enters the pairwise tests. `seed_summary` uses `statistics.mean` and `statistics.stdev` (the $`R - 1`$ version) and writes `seed_runs`; `src/fnsl/reporting.py` → `seed_runs_note` prints "mean ± sample std over R runs, seeds …" under the results table, but only while every run still matches its metrics JSON. Tests: `tests/test_compare_models.py` → `test_seed_reruns_are_summarised_not_paired`, `test_seed_reruns_without_base_or_not_lining_up`. Current state: `seed_runs` is empty in `comparison.json`; the planned reruns are DistilBERT with `--seed 43` and `--seed 44` (README, GPU table).

**Why they are not tested pairwise.** "Is seed 43 better than seed 42?" is not a meaningful question: both runs are draws from the same training procedure. Adding such pairs would also enlarge the Holm family $`m`$ and make the real comparisons stricter; the useful summary of training randomness is its spread. For the TF-IDF baseline, seed reruns would be pointless: its objective is convex and `lbfgs` ignores `random_state`, so every seed gives the same model.

### 8.3.9 From results files to README and RESUME_NUMBERS.md: the gates

**Intuition.** Numbers reach the README and the resume only through a script, and the script refuses to show a number that is missing, stale, from a smoke run or not comparable. It shows the command to run instead.

**Math.** No real mathematics: the "stale" rule is an equality check. A model's entry in `comparison.json` is current only if

```math
n_{\text{comparison}} = n_{\text{test}}
\quad\text{and}\quad
\big|\,\text{macro-F1}_{\text{comparison}} - \text{macro-F1}_{\text{metrics JSON}}\,\big| \le 10^{-9}
```

and a pair is current only if both of its models have a real metrics JSON and are current.

**In this code.** `scripts/make_results_table.py` → `main`, `load_result`, `build_sections`, `update_readme`, and `src/fnsl/reporting.py`:

- **Smoke files ignored.** `load_result` returns nothing for a JSON with `"smoke": true`; smoke per-model records are dropped and listed as "Ignored smoke-run files (never results)".
- **Only real model records.** `src/fnsl/metrics.py` → `load_metrics` keeps only JSONs with all 11 required keys (`is_model_metrics`), so `comparison.json` or `kmeans.json` can never become a model row.
- **Stale comparison.** `consistent_ci` applies the rule above; a failing CI cell reads "missing or stale: rerun `python scripts/compare_models.py`". `split_pairs` and `stale_pairs_note` hide stale pairs and name them without numbers.
- **Not-comparable subsets.** `comparable_note` flags a model whose `n_test` differs from the full test split ("evaluated on X of Y test rows: not comparable"); in RESUME_NUMBERS.md such a line ends with "WARNING … do not use this number yet".
- **Not run yet.** Expected models without a metrics JSON show "not run yet" and the command to run; tests assert that such rows contain no digits.
- **"Compared four models".** `_bullet_1` counts distinct models in current pairs (seed reruns merged by stripping `_seed<N>`); with fewer than four it says to reword the bullet.
- **Skills.** `_skills` says YES only if a real (non-smoke) results file backs the skill: scikit-learn needs `tfidf_logreg.json`; PyTorch and Hugging Face need `distilbert.json`; TensorFlow/Keras needs `keras_bilstm.json`; Unsloth needs `llm_lora.json` with `params.train_backend` = `unsloth` (an adapter trained with the `--backend hf` fallback does not count); LangChain needs `rag_eval.json`; hypothesis testing needs `comparison.json` with at least one current pair. Today only scikit-learn is YES (`RESUME_NUMBERS.md`).
- **Safe README edits.** `replace_section` rewrites only the text between `<!-- BEGIN:NAME -->` and `<!-- END:NAME -->` (DATASET, RESULTS, COMPARISON, KMEANS, RAG); if any marker is missing, duplicated or out of order, nothing is written.

Tests: `tests/test_results_table.py` (full chain, stale pairs, not-comparable subsets, seed-run notes, missing files, broken markers).

**Why it matters here.** The project's hard rule, never invent numbers, is enforced by code rather than by memory. That is a strong answer to "how do you know your README matches your experiments?", and it is why the README currently shows "not run yet" for four models and "No pairs compared yet".

### 8.3.10 Provenance: `git_commit` and the `-dirty` flag

**Intuition.** Every results file records which version of the code produced it, so "which code produced 0.739?" has an answer in the file itself: commit `18e5768`. If the code had uncommitted changes, the stamp says so.

**Math.** None; it is a rule. The stamp is the short hash of HEAD; `-dirty` is appended if anything in `src/` or `scripts/` differs from HEAD (modified files or new untracked files); it is `unknown` outside a git checkout.

**In this code.** `src/fnsl/metrics.py` → `git_commit` runs `git rev-parse --short HEAD` and `git status --porcelain --untracked-files=normal -- src scripts`. Ignored files (`__pycache__`, `*.egg-info`, the editor temp files listed in `.gitignore`) do not count. `save_results` writes the stamp into every metrics JSON; `scripts/compare_models.py` and `scripts/make_results_table.py` stamp `comparison.json` and RESUME_NUMBERS.md; the README results table shows each model's commit. Test: `tests/test_metrics.py` → `test_git_commit_marks_uncommitted_code_dirty` (uses a temporary repository). Current stamps: `tfidf_logreg.json` `18e5768`, `comparison.json` `cc1abac`.

**Pitfall.** The stamp covers only `src/` and `scripts/`; it does not pin library versions (`requirements.txt` gives ranges such as `scikit-learn>=1.5,<2`). Also, `tfidf_logreg.json` was written at `18e5768`, before commit `3ea10c9` made untracked files count as dirty (at `18e5768` only modified tracked files did). All the baseline code was committed in `18e5768`, the JSON's timestamp is later than that commit, and `git diff 18e5768 HEAD` shows the baseline code unchanged since (in `src/fnsl/metrics.py` only `git_commit` itself changed), so the stamp still identifies the code correctly.

### 8.3.11 Experiment tracking with MLflow (SQLite backend)

**Intuition.** MLflow keeps a browsable log of runs: parameters, metrics and files. Here it is a convenience; the JSON files remain the source of truth for every number.

**Math.** None.

**In this code.** `src/fnsl/tracking.py` → `mlflow_run(run_name, params, enabled, tags)` points MLflow at `sqlite:///` plus the absolute path of `mlruns/mlflow.db` in the repo (`tracking_uri`), creates the experiment `fin-news-sentiment-lab` with artifacts under `mlruns/artifacts/` if it does not exist yet, flattens nested parameters into strings of at most 500 characters (`_flatten`), and yields a run with `log_metrics` and `log_artifact`. It becomes a no-op for `--smoke`, for `--no-mlflow`, or if MLflow cannot be imported or set up ("tracking must never break a training run"). `scripts/run_baseline.py` logs the chosen parameters (`C`, `class_weight`, `ngram_range`, `min_df`, `sublinear_tf`, `max_iter`, `val_macro_f1`), the metrics `val_macro_f1`, `test_macro_f1`, `test_accuracy` and `test_f1_<class>`, and the confusion-matrix PNG. The grid search runs before the MLflow run opens, so the grid is saved only in the JSON (`val_grid`). Browse with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` from the repo root; `mlruns/` is gitignored. Installed: MLflow 3.18.0 (`requirements.txt`: `mlflow>=3.0,<4`).

**Why it matters here.** The installed MLflow defaults to a SQLite file (`mlflow/store/tracking/__init__.py` → `DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"`, relative to the current folder) and keeps the old `./mlruns` file store only for folders that already hold one; the `tracking.py` docstring notes that the file store is in maintenance mode. Setting an absolute SQLite path means runs always land in the repo's `mlruns/`, whatever folder a script is started from. A store can hold several runs with the same name (every rerun adds one), so quote numbers from the JSON files, never from the MLflow UI.

### 8.3.12 Answers to the plan's "Explain" questions (Day 7)

- **What does a p-value mean here?** Assume the two models have the same error rate on headlines like these. The p-value is the probability, under that assumption, of a split of the discordant rows ($`b`$ vs $`c`$) at least as lopsided as the one observed. Small means "hard to explain as luck"; it is not the probability that the models are equal, and it is not the size of the gain.
- **Why does McNemar suit paired classifiers?** Both models are scored on the same rows, so their errors are paired and correlated. McNemar uses only the rows where they disagree, which cancels how hard each row is, needs no retraining, and has an exact binomial form for small counts. Tests that treat the two accuracies as independent samples get the variance wrong.
- **Why is a CI better than a single F1 number?** 0.739 alone hides how much it would move on another test sample; the 95% CI [0.715, 0.762] (`comparison.json`) shows it. That tells you which differences are within noise and stops you over-reading small gaps. To compare two models, use the paired tests rather than eyeballing two CIs.
- **Reporting non-significant differences honestly.** The README and RESUME_NUMBERS.md print the Holm verdict for every current pair, including "no" / "NOT significant", and whenever RESUME_NUMBERS.md lists pairs it adds the reminder "Report non-significant differences as non-significant." (`src/fnsl/reporting.py` → `_significance`).

### 8.3.13 What is left to do for this part

1. Produce full-test predictions for at least one more model: `python scripts/train_keras.py` (a laptop CPU is enough) and/or `python scripts/train_distilbert.py` on a Colab or Kaggle GPU through `notebooks/colab_runner.ipynb`.
2. Run `python scripts/compare_models.py`, then `python scripts/make_results_table.py --update-readme --count-tests`, run `pytest -m "not heavy"` and `ruff check .`, and commit the new files in `results/`.
3. Optional (Day 7): rerun DistilBERT with `--seed 43` and `--seed 44` to get mean ± std.
4. To include the zero-shot LLM in McNemar, run it on the full test split (`python scripts/train_unsloth.py --mode zero-shot --n-test 0`); the default run uses 50 test rows (`ZEROSHOT_N_TEST` in `scripts/train_unsloth.py`), has different test ids, and is skipped.
5. Write up every pair's verdict, including the non-significant ones.

---

## Files this part reads and writes

**`data/splits/{train,val,test}.csv`** — written once by `scripts/prepare_data.py` (`data.save_splits`, which refuses to overwrite unless `overwrite=True` / `--overwrite`); read by every script through `data.load_split` / `load_run_splits`, which check columns and labels; committed, read-only.
- `id` (int): row number after cleaning, 0 … n_clean − 1 in pooled Hub order (all `train` rows, then `validation`); the pairing key for every prediction file.
- `text` (str): whitespace-normalised original tweet (case and URLs kept); read with `keep_default_na=False` so "NA"/"null" stay strings.
- `label` (int): 0 = Bearish, 1 = Bullish, 2 = Neutral. Rows sorted by `id`.

**`data/splits/split_report.json`** — written by `scripts/prepare_data.py`; read by `scripts/make_results_table.py` (README DATASET block via `reporting.dataset_section`), `data.current_dataset_name` (the `dataset` field stamped into every metrics JSON) and `tests/test_data.py`.
- `dataset`, `revision`, `license`, `seed`, `fractions` {train, val, test}
- `cleaning` {n_raw, dropped_empty, dropped_url_only, dropped_conflicting_labels, dropped_duplicates, n_clean}
- `sizes` {train, val, test}
- `class_balance` {all | train | val | test: {n, Bearish, Bearish_%, Bullish, Bullish_%, Neutral, Neutral_%}} (percentages rounded to one decimal)
- `labels` {"0": "Bearish", "1": "Bullish", "2": "Neutral"}

**`results/predictions/<model>.csv`** — written by `metrics.save_results`; read by `compare_models.load_all_predictions` (via `metrics.load_predictions`).
- Columns exactly `id,text,label,pred`, rows in test-split order; `pred` in {0, 1, 2, −1} (−1 = `UNPARSEABLE`); ids unique; pairs are aligned by `id`, never by row order. Seed reruns are named `<model>_seed<N>.csv` (the training scripts use that name when `--seed` is not 42).

**`results/metrics/<model>.json`** — written by `metrics.save_results`; read by `metrics.load_metrics`, `scripts/make_results_table.py`, `src/fnsl/reporting.py`.
- Required (`REQUIRED_METRIC_KEYS`, 11 keys): `model, dataset, n_test, macro_f1, accuracy, per_class_f1 {Bearish, Bullish, Neutral}, seed, train_time_s (rounded to 2 decimals), params, git_commit (short hash, with "-dirty", or "unknown"), timestamp (UTC, ISO 8601 to the second)`.
- Extras: `smoke` (bool), `confusion_matrix` (3×3, rows = true, columns = predicted, labels 0..2, −1 predictions excluded), `n_unparseable`, plus any `extra` keys. For `tfidf_logreg`: `params` = {C, class_weight, ngram_range, min_df, sublinear_tf, max_iter, val_macro_f1}; extras `val_grid` [{C, class_weight, val_macro_f1}] and `n_features`.

**`results/figures/<model>_confusion.png`** — drawn by `metrics.save_confusion_matrix`; the title (model, n, " [SMOKE]" for smoke runs, "N unparseable outputs not shown" when there are any) is built in `metrics.save_results`.

**`results/metrics/comparison.json`** — written by `scripts/compare_models.py` (smoke inputs always go to `results/smoke/metrics/comparison.json` instead); read by `src/fnsl/reporting.py` via `make_results_table.load_result`.
- Top level: `kind` = "comparison", `n_resamples`, `alpha`, `seed`, `smoke`, `predictions_dir`, `git_commit`, `timestamp`.
- `models` {name: {n, macro_f1, ci_low, ci_high, accuracy, acc_ci_low, acc_ci_high}} (every model, including seed reruns).
- `pairs` [{model_a, model_b, n, mcnemar {table [[n11, b], [c, n22]], b, c, statistic, p_value, method}, macro_f1_diff {diff, low, high, p_value}, significant_at_0.05, p_holm, significant_holm}].
- `skipped_pairs` [{model_a, model_b, reason}]; `seed_runs` {base: {runs [{model, seed, n, macro_f1}], n_runs, macro_f1_mean, macro_f1_std}}; `skipped_models` [{model, reason}].
- At HEAD: `models` has only `tfidf_logreg`; `pairs`, `skipped_pairs`, `seed_runs`, `skipped_models` are empty.

**Smoke variants** — same schemas under `results/smoke/{metrics,predictions,figures}/` with `"smoke": true`; gitignored; ignored by `make_results_table.py` (and dropped even if a smoke JSON is copied into `results/metrics/`).

**`README.md`** — only the text between `<!-- BEGIN:X -->` and `<!-- END:X -->` for X in DATASET, RESULTS, COMPARISON, KMEANS, RAG is rewritten by `make_results_table.py --update-readme` (all sections or nothing). **`RESUME_NUMBERS.md`** — fully regenerated: header with source commit and timestamp, Bullet 1–3 values each with its source file (or NOT AVAILABLE / WARNING), bootstrap CI table, Holm significance lines, skills YES/NO list.

**`mlruns/mlflow.db`** (SQLite, absolute `sqlite:///` URI from `tracking.tracking_uri`) + `mlruns/artifacts/` — gitignored; experiment `fin-news-sentiment-lab`; run name = model name; params flattened to strings (≤ 500 chars); baseline metrics `val_macro_f1, test_macro_f1, test_accuracy, test_f1_<class>`; artifact = confusion PNG. Never a source for README numbers.
