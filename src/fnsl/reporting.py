"""Turn results files into markdown for README.md and RESUME_NUMBERS.md.

Every number shown here is read from a file a script wrote (results/metrics/*.json and
data/splits/split_report.json); nothing is typed in by hand. When a file is missing, the
text names the script to run instead of showing a number. scripts/make_results_table.py
loads the files, calls these functions, and pastes the output between the
<!-- BEGIN:NAME --> and <!-- END:NAME --> markers of README.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fnsl import config

# README sections that make_results_table.py rewrites, in README order.
SECTIONS = ("DATASET", "RESULTS", "COMPARISON", "KMEANS", "RAG")

# Models the plan expects, in table order, with how to produce each one ("run ...").
EXPECTED_MODELS = {
    "tfidf_logreg": "`python scripts/run_baseline.py`",
    "keras_bilstm": "`python scripts/train_keras.py`",
    "distilbert": "`python scripts/train_distilbert.py` on a GPU (notebooks/colab_runner.ipynb)",
    "llm_zeroshot": "`python scripts/train_unsloth.py --mode zero-shot` on a GPU",
    "llm_lora": "`python scripts/train_unsloth.py --mode train`, then `--mode eval`, on a GPU",
}
MODEL_DESCRIPTIONS = {
    "tfidf_logreg": "TF-IDF + logistic regression",
    "keras_bilstm": "Keras BiLSTM",
    "distilbert": "DistilBERT fine-tuned in PyTorch",
    "llm_zeroshot": "Small LLM, zero-shot",
    "llm_lora": "Small LLM + LoRA via Unsloth",
}
# Readable names for the cleaning counts in split_report.json (other keys: key name).
CLEANING_LABELS = {
    "dropped_empty": "empty",
    "dropped_url_only": "URL-only",
    "dropped_conflicting_labels": "with conflicting labels",
    "dropped_duplicates": "near-duplicates",
}
COMPARE_CMD = "python scripts/compare_models.py"
RAG_CMD = "python scripts/eval_rag.py"
TABLE_CMD = "python scripts/make_results_table.py --update-readme"

# Resume template from PROJECT_PLAN.md section 9; each [X] is a number to fill in.
BULLET_1 = (
    "Fine-tuned DistilBERT in PyTorch for financial news sentiment (macro-F1 [X] vs [X] for a "
    "TF-IDF + logistic regression baseline); compared four models with McNemar's test and "
    "bootstrap confidence intervals."
)
BULLET_2 = (
    "LoRA fine-tuned a small LLM with Unsloth for the same task ([X] macro-F1); built a "
    "LangChain RAG Q&A over public central bank text (hit@3 of [X] on [X] labeled questions)."
)
BULLET_3 = (
    "Published a Kaggle notebook and a Hugging Face model card; [X] pytest tests, CI on every push."
)


# --- small formatting helpers ------------------------------------------------------------
def fmt(x: Any, digits: int = 3) -> str:
    """A number with `digits` decimals, '-' when missing (anything else is shown as is)."""
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_int(x: Any) -> str:
    """A count with thousands separators, '-' when missing."""
    return "-" if x is None else f"{int(x):,}"


def fmt_p(p: float | None) -> str:
    """A p-value with 3 decimals; very small values are shown as <0.001."""
    if p is None:
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def display_path(path: Path | str) -> str:
    """Path relative to the repo root when possible (no machine-specific prefixes)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def model_label(name: str) -> str:
    """'TF-IDF + logistic regression (`tfidf_logreg`)', or just the code name."""
    description = MODEL_DESCRIPTIONS.get(name)
    return f"{description} (`{name}`)" if description else f"`{name}`"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub markdown table. A '|' inside a cell is escaped so it cannot split cells."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def replace_section(text: str, name: str, content: str) -> str:
    """Replace whatever is between <!-- BEGIN:name --> and <!-- END:name --> with content.

    The markers must already exist exactly once each; everything outside them is kept
    byte for byte. Raises ValueError otherwise, so a broken README is never half-written.
    """
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one {begin} and one {end}")
    start, stop = text.index(begin) + len(begin), text.index(end)
    if stop < start:
        raise ValueError(f"{end} comes before {begin}")
    return text[:start] + "\n" + content.strip() + "\n" + text[stop:]


def consistent_ci(name: str, record: dict, comparison: dict | None) -> dict | None:
    """The model's entry in comparison.json if it was computed from the same predictions
    as its metrics JSON (same n and macro-F1), else None: a CI from an older run must not
    be shown next to a newer score."""
    entry = (comparison or {}).get("models", {}).get(name)
    if entry is None or entry.get("n") != record["n_test"]:
        return None
    if abs(entry.get("macro_f1", -1.0) - record["macro_f1"]) > 1e-9:
        return None
    return entry


def split_pairs(comparison: dict, records: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split the pairs of comparison.json into (current, stale).

    A pair is current when both models still have a real metrics JSON that matches their
    entry in comparison.json (consistent_ci), so the pair was computed from the same
    predictions as the scores in the RESULTS table. Otherwise it is stale (a model was
    retrained after compare_models.py ran, or has no real metrics JSON, e.g. smoke files)
    and its numbers must not be shown or claimed.
    """
    current, stale = [], []
    for pair in comparison["pairs"]:
        both_match = all(
            name in records and consistent_ci(name, records[name], comparison) is not None
            for name in (pair["model_a"], pair["model_b"])
        )
        if both_match:
            current.append(pair)
        else:
            stale.append(pair)
    return current, stale


def stale_pairs_note(stale: list[dict]) -> str:
    """One line naming the stale pairs and how to refresh them, '' if there are none."""
    if not stale:
        return ""
    names = ", ".join(f"`{p['model_a']}` vs `{p['model_b']}`" for p in stale)
    return (
        f"_Not shown: {names}. comparison.json no longer matches the metrics files (a model "
        f"was retrained, or has no real metrics JSON); rerun `{COMPARE_CMD}`._"
    )


def seed_runs_note(comparison: dict | None, records: dict[str, dict]) -> str:
    """Mean ± std of test macro-F1 over seed reruns ("seed_runs" in comparison.json).

    A model's line is shown only if every one of its runs still matches its metrics JSON
    (consistent_ci); otherwise it says to rerun compare_models.py. '' when there are none.
    The seeds shown are the ones saved in each run's metrics JSON.
    """
    lines = []
    for base, group in (comparison or {}).get("seed_runs", {}).items():
        names = [run["model"] for run in group["runs"]]
        if not all(n in records and consistent_ci(n, records[n], comparison) for n in names):
            lines.append(
                f"- `{base}`: runs changed since the last comparison, or a run has no real "
                f"metrics JSON; rerun `{COMPARE_CMD}`"
            )
            continue
        seeds = ", ".join(
            str(records[run["model"]].get("seed", run["seed"])) for run in group["runs"]
        )
        lines.append(
            f"- `{base}`: macro-F1 {fmt(group['macro_f1_mean'])} ± {fmt(group['macro_f1_std'])} "
            f"(mean ± sample std over {group['n_runs']} runs, seeds {seeds})"
        )
    if not lines:
        return ""
    return "Seed variation (the same model retrained with different random seeds):\n\n" + "\n".join(
        lines
    )


def ci_level(comparison: dict | None) -> int:
    """Confidence level in percent used by comparison.json (alpha 0.05 -> 95)."""
    return round(100 * (1 - comparison["alpha"])) if comparison else 95


def comparable_note(record: dict, n_test_full: int | None) -> str:
    """Warning for a model scored on fewer rows than the full test split, else ''."""
    if n_test_full is None or record["n_test"] == n_test_full:
        return ""
    return f"evaluated on {record['n_test']} of {n_test_full} test rows: not comparable"


# --- README sections --------------------------------------------------------------------
def dataset_section(report: dict | None) -> str:
    """Dataset, licence, cleaning counts, split sizes and class balance (split_report.json)."""
    if report is None:
        return "_Not available: run `python scripts/prepare_data.py` to create `data/splits/`._"
    cleaning = report.get("cleaning", {})
    drops = ", ".join(
        f"{fmt_int(count)} {CLEANING_LABELS.get(key, key)}"
        for key, count in cleaning.items()
        if key.startswith("dropped_")
    )
    sizes = ", ".join(f"{name} {fmt_int(n)}" for name, n in report.get("sizes", {}).items())
    fractions = "/".join(str(round(100 * f)) for f in report.get("fractions", {}).values())
    rows = []
    for split, balance in report.get("class_balance", {}).items():
        cells = [split, fmt_int(balance.get("n"))]
        for label in config.LABELS:
            count, pct = balance.get(label), balance.get(f"{label}_%")
            if count is None:
                cells.append("-")
            else:
                cells.append(fmt_int(count) + (f" ({pct:.1f}%)" if pct is not None else ""))
        rows.append(cells)
    lines = [
        f"- **Dataset:** `{report.get('dataset', '-')}`, Hub revision "
        f"`{report.get('revision', '-')}`, licence: {report.get('license', '-')}",
        f"- **Cleaning:** {fmt_int(cleaning.get('n_raw'))} raw rows; dropped {drops or '-'}; "
        f"{fmt_int(cleaning.get('n_clean'))} rows left",
        f"- **Split** (stratified {fractions or '-'}, seed {report.get('seed', '-')}): {sizes}",
        "",
        markdown_table(["Split", "Rows", *config.LABELS], rows),
        "",
        "_Source: `data/splits/split_report.json` (written by `scripts/prepare_data.py`)._",
    ]
    return "\n".join(lines)


def results_section(
    records: list[dict],
    comparison: dict | None = None,
    n_test_full: int | None = None,
    skipped_smoke: list[str] | None = None,
) -> str:
    """One row per model metrics JSON; expected models without a file say 'not run yet'."""
    by_name = {r["model"]: r for r in records}
    names = list(EXPECTED_MODELS) + sorted(set(by_name) - set(EXPECTED_MODELS))
    level = ci_level(comparison)
    headers = [
        "Model",
        "Macro-F1",
        f"{level}% CI",
        "Accuracy",
        *(f"F1 {label}" for label in config.LABELS),
        "n_test",
        "Train time (s)",
        "Commit",
        "Note",
    ]
    rows = []
    for name in names:
        record = by_name.get(name)
        if record is None:
            empty = [""] * (len(headers) - 3)
            rows.append([model_label(name), "not run yet", *empty, f"run {EXPECTED_MODELS[name]}"])
            continue
        entry = consistent_ci(name, record, comparison)
        if comparison is None:
            ci = "-"
        elif entry is None:
            ci = f"missing or stale: rerun `{COMPARE_CMD}`"
        else:
            ci = fmt_ci(entry["ci_low"], entry["ci_high"])
        notes = [comparable_note(record, n_test_full)]
        if record.get("n_unparseable"):
            notes.append(f"{record['n_unparseable']} unparseable outputs counted as wrong")
        per_class = record.get("per_class_f1", {})
        rows.append(
            [
                model_label(name),
                fmt(record["macro_f1"]),
                ci,
                fmt(record["accuracy"]),
                *(fmt(per_class.get(label)) for label in config.LABELS),
                str(record["n_test"]),
                fmt(record.get("train_time_s"), 1),
                f"`{record['git_commit']}`",
                "; ".join(note for note in notes if note),
            ]
        )

    parts = [markdown_table(headers, rows)]
    seed_note = seed_runs_note(comparison, by_name)
    if seed_note:
        parts.append(seed_note)
    if comparison is None:
        parts.append(f"_Confidence intervals: not computed yet, run `{COMPARE_CMD}`._")
    else:
        parts.append(
            f"_{level}% CI: percentile bootstrap over {comparison['n_resamples']:,} resamples of "
            "the test set (`results/metrics/comparison.json`). Per-model scores: "
            "`results/metrics/<model>.json`._"
        )
    if skipped_smoke:
        names_md = ", ".join(f"`{name}`" for name in skipped_smoke)
        parts.append(f"_Ignored smoke-run files (never results): {names_md}._")
    return "\n\n".join(parts)


def comparison_section(comparison: dict | None, records: list[dict]) -> str:
    """Pairs table (McNemar, Holm, paired bootstrap diff) plus skipped pairs with reasons.

    records: the real per-model metrics; pairs that no longer match them are not shown.
    """
    if comparison is None:
        return f"_Not available yet: run `{COMPARE_CMD}`, then `{TABLE_CMD}`._"
    level = ci_level(comparison)
    pairs, stale = split_pairs(comparison, {r["model"]: r for r in records})
    parts = []
    if pairs:
        rows = []
        for pair in pairs:
            mc, diff = pair["mcnemar"], pair["macro_f1_diff"]
            rows.append(
                [
                    f"`{pair['model_a']}`",
                    f"`{pair['model_b']}`",
                    str(pair["n"]),
                    f"{mc['b']} / {mc['c']}",
                    fmt_p(mc["p_value"]),
                    fmt_p(pair["p_holm"]),
                    f"{diff['diff']:+.3f} {fmt_ci(diff['low'], diff['high'])}",
                    "yes" if pair["significant_holm"] else "no",
                ]
            )
        headers = [
            "A",
            "B",
            "n",
            "b / c",
            "McNemar p",
            "Holm p",
            f"Macro-F1 diff A - B [{level}% CI]",
            "Significant? (Holm)",
        ]
        parts.append(markdown_table(headers, rows))
        parts.append(
            "_b = rows only A got right, c = rows only B got right. McNemar's test asks whether "
            "the two error rates differ; Holm adjusts its p-values for testing several pairs "
            f"(significant if Holm p < {comparison['alpha']}). The macro-F1 difference and its "
            f"CI come from a paired bootstrap ({comparison['n_resamples']:,} resamples). The two "
            "can disagree: macro-F1 weights the rare classes more than the error rate does._"
        )
    elif not stale:
        names = ", ".join(f"`{m}`" for m in comparison["models"]) or "none"
        parts.append(
            f"_No pairs compared yet. Models with test predictions: {names}. A pair needs two "
            "different models with predictions on the same test ids (seed reruns of one model "
            "are summarised under the results table, not paired)._"
        )
    if stale:
        parts.append(stale_pairs_note(stale))
    skipped = [
        f"- Skipped `{p['model_a']}` vs `{p['model_b']}`: {p['reason']}"
        for p in comparison.get("skipped_pairs", [])
    ] + [
        f"- Skipped predictions file `{m['model']}`: {m['reason']}"
        for m in comparison.get("skipped_models", [])
    ]
    if skipped:
        parts.append("\n".join(skipped))
    return "\n\n".join(parts)


def kmeans_section(record: dict | None) -> str:
    """Silhouette, ARI and NMI for each k from results/metrics/kmeans.json."""
    if record is None:
        return f"_Not available yet: run `python scripts/run_kmeans.py`, then `{TABLE_CMD}`._"
    rows = [
        [
            str(run.get("k", "-")),
            fmt(run.get("silhouette")),
            fmt(run.get("ari")),
            fmt(run.get("nmi")),
        ]
        for run in record.get("runs", [])
    ]
    n = record.get("n")
    info = (
        f"Embeddings `{record.get('embedding_model', '-')}`, split: {record.get('split', '-')}, "
        f"n = {fmt_int(n) if isinstance(n, int) else '-'} (`results/metrics/kmeans.json`)."
    )
    table = markdown_table(["k", "Silhouette", "ARI", "NMI"], rows) if rows else "_No runs._"
    legend = (
        "_Silhouette: how well separated the clusters are (-1 to 1). ARI: agreement between "
        "clusters and sentiment labels, corrected for chance (0 = chance level, 1 = identical). "
        "NMI: information the clusters share with the labels (0 = none, 1 = identical; not "
        "corrected for chance, so random clusters score slightly above 0)._"
    )
    return "\n\n".join([info, table, legend])


def rag_section(record: dict | None) -> str:
    """hit@k per chunk size from results/metrics/rag_eval.json."""
    if record is None:
        return (
            "_Not available yet: write the evaluation questions in `data/rag_eval.csv`, run "
            f"`{RAG_CMD}`, then `{TABLE_CMD}`._"
        )
    ks = record.get("k_values") or [1, 3]
    headers = ["Chunk size", "Chunks", *(f"hit@{k}" for k in ks), "Questions"]
    rows = [
        [
            str(cfg.get("chunk_size", "-")),
            str(cfg.get("n_chunks", "-")),
            *(fmt(cfg.get(f"hit@{k}")) for k in ks),
            str(record.get("n_questions", "-")),
        ]
        for cfg in record.get("configs", [])
    ]
    info = (
        f"{record.get('n_questions', '-')} hand-written questions, embeddings "
        f"`{record.get('embedding_model', '-')}`, chunk overlap "
        f"{record.get('chunk_overlap', '-')}. Best chunk size: "
        f"{record.get('best_chunk_size', '-')} (picked on the same questions, so its score is "
        "slightly optimistic). Source: `results/metrics/rag_eval.json`."
    )
    legend = (
        "_hit@k: share of questions whose source file is among the top-k retrieved chunks. "
        "It measures retrieval only, not whether a generated answer is correct._"
    )
    return "\n\n".join([info, markdown_table(headers, rows), legend])


# --- RESUME_NUMBERS.md --------------------------------------------------------------------
def _macro_f1_line(
    label: str,
    name: str,
    records: dict[str, dict],
    comparison: dict | None,
    n_test_full: int | None,
    paths: dict[str, str],
) -> tuple[str, bool]:
    """Resume line for one model's test macro-F1, and whether the number may be used."""
    record = records.get(name)
    if record is None:
        return f"- {label}: NOT AVAILABLE - run {EXPECTED_MODELS[name]}", False
    line = f"- {label}: {record['macro_f1']:.3f} ({paths['metrics_dir']}/{name}.json)"
    entry = consistent_ci(name, record, comparison)
    if entry is not None:
        ci = fmt_ci(entry["ci_low"], entry["ci_high"])
        line += f", {ci_level(comparison)}% CI {ci} ({paths['comparison']})"
    warning = comparable_note(record, n_test_full)
    if warning:
        return f"{line} - WARNING: {warning}; do not use this number yet", False
    return line, True


def _bullet_1(
    records: dict[str, dict],
    comparison: dict | None,
    n_test_full: int | None,
    paths: dict[str, str],
) -> str:
    distil, distil_ok = _macro_f1_line(
        "DistilBERT test macro-F1", "distilbert", records, comparison, n_test_full, paths
    )
    tfidf, tfidf_ok = _macro_f1_line(
        "TF-IDF + logistic regression baseline test macro-F1",
        "tfidf_logreg",
        records,
        comparison,
        n_test_full,
        paths,
    )
    four_compared = False
    if comparison is None:
        compared = f"- Models compared: NOT AVAILABLE - run `{COMPARE_CMD}`"
    else:
        pairs, _ = split_pairs(comparison, records)
        names = sorted({p["model_a"] for p in pairs} | {p["model_b"] for p in pairs})
        # distilbert and distilbert_seed7 are one model trained twice: count it once.
        distinct = {re.sub(r"_seed\d+$", "", name) for name in names}
        four_compared = len(distinct) >= 4
        listed = ", ".join(f"`{name}`" for name in names) or "none"
        compared = (
            f"- Models compared with McNemar + bootstrap: {len(distinct)} distinct models "
            f"({listed}) in {len(pairs)} pairs ({paths['comparison']})"
        )
        if not four_compared:
            compared += ' - fewer than four: do not write "compared four models"'
    if not (distil_ok and tfidf_ok):
        verdict = "Do not use bullet 1 yet."
    elif not four_compared:
        verdict = 'Bullet 1 macro-F1 numbers are available; reword "compared four models".'
    else:
        verdict = "Bullet 1 numbers are available."
    return "\n".join(["## Bullet 1", "", f"> {BULLET_1}", "", distil, tfidf, compared, "", verdict])


def _bullet_2(
    records: dict[str, dict],
    comparison: dict | None,
    rag: dict | None,
    n_test_full: int | None,
    paths: dict[str, str],
) -> str:
    lora, lora_ok = _macro_f1_line(
        "LoRA fine-tuned LLM test macro-F1", "llm_lora", records, comparison, n_test_full, paths
    )
    lines = ["## Bullet 2", "", f"> {BULLET_2}", "", lora]
    rag_ok = False
    if rag is None:
        lines += [
            f"- RAG hit@3: NOT AVAILABLE - write questions in data/rag_eval.csv, run `{RAG_CMD}`",
            f"- Labelled RAG questions: NOT AVAILABLE - run `{RAG_CMD}`",
        ]
    else:
        best = next(
            (
                c
                for c in rag.get("configs", [])
                if c.get("chunk_size") == rag.get("best_chunk_size")
            ),
            {},
        )
        if best.get("hit@3") is None:
            lines.append(
                f"- RAG hit@3: NOT AVAILABLE - no hit@3 in {paths['rag']}; rerun `{RAG_CMD}`"
            )
        else:
            rag_ok = True
            lines.append(
                f"- RAG hit@3: {best['hit@3']:.3f} at chunk size {best['chunk_size']} (best of "
                f"{len(rag['configs'])} chunk sizes, picked on the same questions) ({paths['rag']})"
            )
        lines.append(f"- Labelled RAG questions: {rag.get('n_questions', '-')} ({paths['rag']})")
    if not lora_ok:
        unsloth_part = "do not use yet."
    elif trained_with_unsloth(records):
        unsloth_part = "numbers available."
    else:
        unsloth_part = (
            "numbers available, but the adapter was not trained with --backend unsloth: "
            'replace "with Unsloth" by "with Hugging Face PEFT".'
        )
    lines += [
        "",
        "Unsloth part: " + unsloth_part,
        "RAG part: " + ("numbers available." if rag_ok else "do not use yet."),
    ]
    return "\n".join(lines)


def trained_with_unsloth(records: dict[str, dict]) -> bool:
    """True only if llm_lora.json says its adapter was trained by the Unsloth backend.

    train_unsloth.py also has a plain Hugging Face fallback (--backend hf). A model trained
    that way is still a LoRA model, but it is no evidence for the Unsloth skill.
    """
    params = records.get("llm_lora", {}).get("params") or {}
    return params.get("train_backend") == "unsloth"


def _bullet_3(test_counts: dict[str, Any] | None) -> str:
    if test_counts is None:
        tests = f"- pytest tests: NOT AVAILABLE - run `{TABLE_CMD} --count-tests`"
    elif "error" in test_counts:
        tests = f"- pytest tests: NOT AVAILABLE - {test_counts['error']}"
    else:
        tests = (
            f"- pytest tests: {test_counts['all']} collected, {test_counts['not_heavy']} of them "
            'run in CI with -m "not heavy" (pytest --collect-only)'
        )
    manual = (
        "- Kaggle notebook, Hugging Face model card and green CI are not recorded in results "
        "files: check by hand that they are live before using this bullet."
    )
    return "\n".join(["## Bullet 3", "", f"> {BULLET_3}", "", tests, manual])


def _confidence_intervals(
    comparison: dict | None, records: dict[str, dict], paths: dict[str, str]
) -> str:
    lines = ["## Bootstrap confidence intervals", ""]
    if comparison is None:
        return "\n".join([*lines, f"NOT AVAILABLE - run `{COMPARE_CMD}`."])
    level = ci_level(comparison)
    rows, stale = [], []
    for name, m in comparison["models"].items():
        # Only CIs computed from the same predictions as the model's real metrics JSON.
        if name not in records or consistent_ci(name, records[name], comparison) is None:
            stale.append(f"`{name}`")
            continue
        rows.append(
            [
                f"`{name}`",
                str(m["n"]),
                f"{m['macro_f1']:.3f} {fmt_ci(m['ci_low'], m['ci_high'])}",
                f"{m['accuracy']:.3f} {fmt_ci(m['acc_ci_low'], m['acc_ci_high'])}",
            ]
        )
    headers = ["Model", "n", f"Macro-F1 [{level}% CI]", f"Accuracy [{level}% CI]"]
    lines += [
        f"Percentile bootstrap, {comparison['n_resamples']:,} resamples ({paths['comparison']}).",
        "",
        markdown_table(headers, rows) if rows else "No current CIs.",
    ]
    if stale:
        lines += [
            "",
            f"Not shown (stale or no real metrics JSON): {', '.join(stale)}; rerun "
            f"`{COMPARE_CMD}`.",
        ]
    return "\n".join(lines)


def _significance(comparison: dict | None, records: dict[str, dict]) -> str:
    lines = ["## Significance (McNemar, Holm-adjusted)", ""]
    if comparison is None:
        return "\n".join([*lines, f"NOT AVAILABLE - run `{COMPARE_CMD}`."])
    pairs, stale = split_pairs(comparison, records)
    if stale:
        lines += [stale_pairs_note(stale), ""]
    if not pairs:
        return "\n".join(
            [*lines, "No current pairs: needs two different models on the same test ids."]
        )
    for p in pairs:
        diff = p["macro_f1_diff"]
        verdict = "significant" if p["significant_holm"] else "NOT significant"
        lines.append(
            f"- `{p['model_a']}` vs `{p['model_b']}`: macro-F1 diff {diff['diff']:+.3f} "
            f"{fmt_ci(diff['low'], diff['high'])}, McNemar p: {fmt_p(p['mcnemar']['p_value'])}, "
            f"Holm p: {fmt_p(p['p_holm'])} -> {verdict} at alpha {comparison['alpha']}"
        )
    return "\n".join([*lines, "", "Report non-significant differences as non-significant."])


def _skills(
    records: dict[str, dict], comparison: dict | None, rag: dict | None, paths: dict[str, str]
) -> str:
    """Skills allowed by the PROJECT_PLAN.md section 9 rules, judged only by which results
    files exist."""
    metrics_dir = paths["metrics_dir"]
    has_pairs = comparison is not None and bool(split_pairs(comparison, records)[0])
    checks = [
        ("scikit-learn", "tfidf_logreg" in records, f"{metrics_dir}/tfidf_logreg.json"),
        ("PyTorch", "distilbert" in records, f"{metrics_dir}/distilbert.json"),
        ("Hugging Face", "distilbert" in records, f"{metrics_dir}/distilbert.json"),
        ("TensorFlow/Keras", "keras_bilstm" in records, f"{metrics_dir}/keras_bilstm.json"),
        (
            "Unsloth",
            trained_with_unsloth(records),
            f"{metrics_dir}/llm_lora.json with params.train_backend = unsloth",
        ),
        ("LangChain", rag is not None, paths["rag"]),
        (
            "Hypothesis testing (McNemar, bootstrap CI)",
            has_pairs,
            f"{paths['comparison']} with at least one current pair",
        ),
    ]
    lines = ["## Skills that may be claimed (PROJECT_PLAN.md section 9)", ""]
    for skill, ok, source in checks:
        lines.append(
            f"- {skill}: {'YES' if ok else 'NO'} - {'backed by' if ok else 'needs'} {source}"
        )
    lines.append("- Python, MLflow: not tied to a results file by the section-9 rules.")
    return "\n".join(lines)


def resume_numbers_md(
    records: dict[str, dict],
    comparison: dict | None,
    rag: dict | None,
    n_test_full: int | None,
    test_counts: dict[str, Any] | None,
    paths: dict[str, str],
    commit: str,
    timestamp: str,
) -> str:
    """Text of RESUME_NUMBERS.md: every [X] of the resume template (PROJECT_PLAN.md
    section 9) with its value and source file, the CIs, a significance summary and the
    skills that may be claimed.

    records: real (non-smoke) per-model metrics by model name. test_counts: None (not
    counted), {"all": n, "not_heavy": m} or {"error": message}. paths: display paths for
    "metrics_dir", "comparison" and "rag".
    """
    header = (
        "# RESUME_NUMBERS\n\n"
        f"GENERATED by scripts/make_results_table.py, do not edit by hand. Source commit "
        f"{commit}, generated {timestamp}.\n\n"
        "Fill the resume template in PROJECT_PLAN.md section 9 with these numbers only. A "
        "value marked NOT AVAILABLE or WARNING means the matching bullet, or that part of it, "
        "must not be used yet."
    )
    blocks = [
        header,
        _bullet_1(records, comparison, n_test_full, paths),
        _bullet_2(records, comparison, rag, n_test_full, paths),
        _bullet_3(test_counts),
        _confidence_intervals(comparison, records, paths),
        _significance(comparison, records),
        _skills(records, comparison, rag, paths),
    ]
    return "\n\n".join(blocks) + "\n"
