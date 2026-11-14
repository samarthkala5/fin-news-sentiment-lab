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
