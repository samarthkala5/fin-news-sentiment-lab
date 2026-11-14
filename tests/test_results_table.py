"""Tests for scripts/make_results_table.py and fnsl.reporting on fixture files in tmp."""

from __future__ import annotations

import importlib.util
import itertools
import json
import re

import pytest

from fnsl import config, metrics, reporting


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_results_table = _load_script("make_results_table")
compare_models = _load_script("compare_models")

README = """# Fixture README

Hand-written intro with a number 0.123 that must never change.

<!-- BEGIN:DATASET -->
old dataset text
<!-- END:DATASET -->

## Results
<!-- BEGIN:RESULTS -->
old results 0.999
<!-- END:RESULTS -->

Hand-written text between sections.
<!-- BEGIN:COMPARISON -->
<!-- END:COMPARISON -->
<!-- BEGIN:KMEANS -->
old
<!-- END:KMEANS -->
<!-- BEGIN:RAG -->
old
<!-- END:RAG -->

## Limitations
Hand-written footer.
"""

SPLIT_REPORT = {
    "dataset": "toy/dataset",
    "revision": "abc123",
    "license": "MIT",
    "seed": 42,
    "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
    "cleaning": {"n_raw": 70, "dropped_empty": 0, "dropped_duplicates": 10, "n_clean": 60},
    "sizes": {"train": 0, "val": 0, "test": 60},
    "class_balance": {
        "test": {
            "n": 60,
            "Bearish": 20,
            "Bearish_%": 33.3,
            "Bullish": 20,
            "Bullish_%": 33.3,
            "Neutral": 20,
            "Neutral_%": 33.3,
        }
    },
}

KMEANS = {
    "model": "kmeans",
    "kind": "clustering",
    "split": "train",
    "n": 8241,
    "embedding_model": "toy-embedder",
    "runs": [{"k": 3, "silhouette": 0.0612, "ari": 0.0123, "nmi": 0.0456}, {"k": 8}],
}

RAG = {
    "kind": "rag_eval",
    "n_questions": 24,
    "embedding_model": "toy-embedder",
    "chunk_overlap": 50,
    "k_values": [1, 3],
    "configs": [
        {"chunk_size": 300, "n_chunks": 155, "hit@1": 0.5, "hit@3": 0.75},
        {"chunk_size": 800, "n_chunks": 63, "hit@1": 0.625, "hit@3": 0.875},
    ],
    "best_chunk_size": 800,
}


def _section(text: str, name: str) -> str:
    """The text between one pair of markers."""
    return text.split(f"<!-- BEGIN:{name} -->")[1].split(f"<!-- END:{name} -->")[0]


def _outside_markers(text: str) -> str:
    """The README with every marked section emptied, i.e. only the hand-written parts."""
    return re.sub(r"(<!-- BEGIN:(\w+) -->).*?(<!-- END:\2 -->)", r"\1\3", text, flags=re.S)


def _row(section: str, model: str) -> str:
    """The table row of a model in the RESULTS section."""
    return next(line for line in section.splitlines() if f"`{model}`" in line)


@pytest.fixture
def splits(monkeypatch, tmp_path, toy_df):
    """Fixture split folder: split_report.json and a 60-row test.csv."""
    folder = tmp_path / "splits"
    folder.mkdir()
    (folder / "split_report.json").write_text(json.dumps(SPLIT_REPORT))
    toy_df.to_csv(folder / "test.csv", index=False)
    monkeypatch.setattr(config, "SPLITS_DIR", folder)
    return folder


def _fake_model(toy_df, model: str, n_wrong: int, smoke: bool = False) -> dict:
    """Write predictions + metrics for a fake model that gets n_wrong rows wrong."""
    pred = toy_df["label"].to_numpy().copy()
    pred[:n_wrong] = (pred[:n_wrong] + 1) % 3
    return metrics.save_results(
        model, toy_df, pred, params={}, train_time_s=1.5, smoke=smoke, make_figure=False
    )


def test_full_chain_updates_only_marked_sections(tmp_results, splits, tmp_path, toy_df):
    tfidf = _fake_model(toy_df, "tfidf_logreg", n_wrong=15)
    distil = _fake_model(toy_df, "distilbert", n_wrong=5)
    compare_models.main(["--n-resamples", "100"])
    # A smoke record copied next to the real ones must be ignored, never shown as a result.
    smoke = _fake_model(toy_df, "keras_bilstm", n_wrong=1, smoke=True)
    (tmp_results / "metrics" / "keras_bilstm.json").write_text(json.dumps(smoke))
    (tmp_results / "metrics" / "kmeans.json").write_text(json.dumps(KMEANS))
    readme, resume = tmp_path / "README.md", tmp_path / "RESUME_NUMBERS.md"
    readme.write_text(README)

    result = make_results_table.main(
        ["--update-readme", "--readme", str(readme), "--resume-out", str(resume)]
    )

    new = readme.read_text()
    assert result["readme_updated"] is True
    assert _outside_markers(new) == _outside_markers(README), "hand-written text changed"
    assert "old results 0.999" not in new

    dataset = _section(new, "DATASET")
    assert "`toy/dataset`" in dataset and "abc123" in dataset and "MIT" in dataset
    assert "10 near-duplicates" in dataset and "20 (33.3%)" in dataset

    results = _section(new, "RESULTS")
    comparison = json.loads((tmp_results / "metrics" / "comparison.json").read_text())
    ci = comparison["models"]["distilbert"]
    assert f"{distil['macro_f1']:.3f}" in _row(results, "distilbert")
    assert f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]" in _row(results, "distilbert")
    assert f"{tfidf['macro_f1']:.3f}" in _row(results, "tfidf_logreg")
    for missing in ("keras_bilstm", "llm_zeroshot", "llm_lora"):
        row = _row(results, missing)
        assert "not run yet" in row
        assert not re.search(r"\d", row), f"a model that was not run shows a number: {row}"
    assert "Ignored smoke-run files" in results and "`keras_bilstm`" in results

    pairs = _section(new, "COMPARISON")
    assert "| `distilbert` | `tfidf_logreg` | 60 |" in pairs
    kmeans = _section(new, "KMEANS")
    assert "| 3 | 0.061 | 0.012 | 0.046 |" in kmeans and "| 8 | - | - | - |" in kmeans
    rag = _section(new, "RAG")
    assert "python scripts/eval_rag.py" in rag and not re.search(r"\d", rag)

    text = resume.read_text()
    assert text == result["resume"]
    assert "GENERATED by scripts/make_results_table.py, do not edit by hand." in text
    assert f"DistilBERT test macro-F1: {distil['macro_f1']:.3f} (" in text
    assert "distilbert.json" in text and "comparison.json" in text
    assert f"baseline test macro-F1: {tfidf['macro_f1']:.3f}" in text
    assert "LoRA fine-tuned LLM test macro-F1: NOT AVAILABLE - run" in text
    assert "RAG hit@3: NOT AVAILABLE" in text
    assert "pytest tests: NOT AVAILABLE" in text  # only counted with --count-tests
    assert 'do not write "compared four models"' in text  # only 2 models compared
    assert "Do not use bullet 1 yet." not in text
    assert 'reword "compared four models"' in text
    for skill, allowed in [
        ("PyTorch", "YES"),
        ("Hugging Face", "YES"),
        ("TensorFlow/Keras", "NO"),  # only a smoke file exists
        ("Unsloth", "NO"),
        ("LangChain", "NO"),
        ("Hypothesis testing (McNemar, bootstrap CI)", "YES"),
    ]:
        assert f"- {skill}: {allowed}" in text, skill


def test_print_only_writes_nothing(tmp_results, splits, tmp_path, toy_df):
    _fake_model(toy_df, "tfidf_logreg", n_wrong=10)
    readme, resume = tmp_path / "README.md", tmp_path / "RESUME_NUMBERS.md"
    readme.write_text(README)
    result = make_results_table.main(["--readme", str(readme), "--resume-out", str(resume)])
    assert readme.read_text() == README
    assert not resume.exists()
    assert set(result["sections"]) == set(reporting.SECTIONS)


def test_missing_files_say_which_script_to_run(tmp_results, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPLITS_DIR", tmp_path / "no_splits")
    sections = make_results_table.main([])["sections"]
    expected_script = {
        "DATASET": "scripts/prepare_data.py",
        "COMPARISON": "scripts/compare_models.py",
        "KMEANS": "scripts/run_kmeans.py",
        "RAG": "scripts/eval_rag.py",
    }
    for name, script in expected_script.items():
        assert script in sections[name]
        assert not re.search(r"\d", sections[name]), f"{name} shows a number without a file"
    for model in reporting.EXPECTED_MODELS:
        row = _row(sections["RESULTS"], model)
        assert "not run yet" in row and not re.search(r"\d", row)


def test_missing_marker_leaves_readme_untouched(tmp_results, splits, tmp_path):
    readme = tmp_path / "README.md"
    broken = README.replace("<!-- END:RAG -->", "")
    readme.write_text(broken)
    with pytest.raises(SystemExit):
        make_results_table.main(["--update-readme", "--readme", str(readme), "--no-resume"])
    assert readme.read_text() == broken


def test_replace_section_keeps_markers_and_rejects_duplicates():
    text = "a\n<!-- BEGIN:X -->\nold\n<!-- END:X -->\nb"
    assert reporting.replace_section(text, "X", "new") == (
        "a\n<!-- BEGIN:X -->\nnew\n<!-- END:X -->\nb"
    )
    with pytest.raises(ValueError):
        reporting.replace_section(text + text, "X", "new")
    with pytest.raises(ValueError):
        reporting.replace_section("<!-- END:X --><!-- BEGIN:X -->", "X", "new")


def _record(model: str, macro_f1: float, n_test: int) -> dict:
    return {
        "model": model,
        "macro_f1": macro_f1,
        "accuracy": 0.8,
        "per_class_f1": {"Bearish": 0.5, "Bullish": 0.6, "Neutral": 0.9},
        "n_test": n_test,
        "train_time_s": 2.0,
        "git_commit": "abc1234",
        "n_unparseable": 0,
    }


def test_results_flags_subsets_and_stale_cis():
    records = [_record("llm_lora", 0.61, 50), _record("distilbert", 0.8, 60)]
    comparison = {
        "n_resamples": 100,
        "alpha": 0.05,
        "models": {"distilbert": {"n": 60, "macro_f1": 0.75, "ci_low": 0.7, "ci_high": 0.8}},
    }
    section = reporting.results_section(records, comparison, n_test_full=60)
    assert "evaluated on 50 of 60 test rows: not comparable" in _row(section, "llm_lora")
    assert "missing or stale" in _row(section, "distilbert")  # macro-F1 0.8 vs 0.75


def test_resume_with_rag_subset_model_and_test_counts():
    paths = {
        "metrics_dir": "results/metrics",
        "comparison": "results/metrics/comparison.json",
        "rag": "results/metrics/rag_eval.json",
    }
    text = reporting.resume_numbers_md(
        records={"llm_lora": _record("llm_lora", 0.61, 50)},
        comparison=None,
        rag=RAG,
        n_test_full=60,
        test_counts={"all": 96, "not_heavy": 81},
        paths=paths,
        commit="abc1234",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert "Source commit abc1234, generated 2026-01-01T00:00:00+00:00." in text
    assert "RAG hit@3: 0.875 at chunk size 800" in text
    assert "Labelled RAG questions: 24 (results/metrics/rag_eval.json)" in text
    assert "0.610 (results/metrics/llm_lora.json) - WARNING: evaluated on 50 of 60" in text
    assert "Unsloth part: do not use yet." in text and "RAG part: numbers available." in text
    assert "pytest tests: 96 collected, 81 of them run in CI" in text
    assert "- LangChain: YES" in text and "- Unsloth: YES" in text
    assert "- Hypothesis testing (McNemar, bootstrap CI): NO" in text
    assert "Bootstrap confidence intervals\n\nNOT AVAILABLE - run" in text


PATHS = {
    "metrics_dir": "results/metrics",
    "comparison": "results/metrics/comparison.json",
    "rag": "results/metrics/rag_eval.json",
}


def _ci_entry(macro_f1: float, n: int = 60) -> dict:
    """A comparison.json "models" entry."""
    return {
        "n": n,
        "macro_f1": macro_f1,
        "ci_low": macro_f1 - 0.05,
        "ci_high": macro_f1 + 0.05,
        "accuracy": 0.8,
        "acc_ci_low": 0.75,
        "acc_ci_high": 0.85,
    }


def _pair(model_a: str, model_b: str) -> dict:
    """A comparison.json "pairs" entry for a clearly significant difference."""
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n": 60,
        "mcnemar": {"b": 12, "c": 1, "p_value": 0.0034, "method": "exact binomial"},
        "macro_f1_diff": {"diff": 0.2, "low": 0.1, "high": 0.3, "p_value": 0.002},
        "significant_at_0.05": True,
        "p_holm": 0.0034,
        "significant_holm": True,
    }


def _resume(records: dict, comparison: dict) -> str:
    return reporting.resume_numbers_md(
        records=records,
        comparison=comparison,
        rag=None,
        n_test_full=60,
        test_counts=None,
        paths=PATHS,
        commit="abc1234",
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_stale_pairs_are_not_shown_or_claimed():
    # distilbert was retrained after compare_models.py ran: its metrics JSON now says 0.6,
    # but comparison.json still holds the old run (0.9) and its "significant" pair.
    records = [_record("distilbert", 0.6, 60), _record("tfidf_logreg", 0.7, 60)]
    comparison = {
        "n_resamples": 100,
        "alpha": 0.05,
        "models": {"distilbert": _ci_entry(0.9), "tfidf_logreg": _ci_entry(0.7)},
        "pairs": [_pair("distilbert", "tfidf_logreg")],
        "skipped_pairs": [],
    }
    section = reporting.comparison_section(comparison, records)
    assert "Not shown: `distilbert` vs `tfidf_logreg`" in section
    assert "rerun `python scripts/compare_models.py`" in section
    assert not re.search(r"\d", section), f"stale numbers are shown: {section}"

    text = _resume({r["model"]: r for r in records}, comparison)
    assert "- Hypothesis testing (McNemar, bootstrap CI): NO" in text
    assert "0 distinct models (none) in 0 pairs" in text
    assert "significant at alpha" not in text and "0.900" not in text
    assert "| `tfidf_logreg` | 60 | 0.700 [0.650, 0.750]" in text  # still current
    assert "Not shown (stale or no real metrics JSON): `distilbert`" in text

    # A pair whose model has no real metrics JSON at all (e.g. smoke files) is not shown.
    only_tfidf = [records[1]]
    assert "Not shown:" in reporting.comparison_section(comparison, only_tfidf)

    # After rerunning compare_models.py the scores match again and the pair is shown.
    comparison["models"]["distilbert"] = _ci_entry(0.6)
    fresh = reporting.comparison_section(comparison, records)
    assert "| `distilbert` | `tfidf_logreg` | 60 | 12 / 1 | 0.003 | 0.003 |" in fresh
    assert "- Hypothesis testing (McNemar, bootstrap CI): YES" in _resume(
        {r["model"]: r for r in records}, comparison
    )


def _all_pairs_comparison(names: list[str]) -> dict:
    return {
        "n_resamples": 100,
        "alpha": 0.05,
        "models": {name: _ci_entry(0.7) for name in names},
        "pairs": [_pair(a, b) for a, b in itertools.combinations(names, 2)],
        "skipped_pairs": [],
    }


def test_seed_reruns_count_once_toward_four_models():
    names = ["distilbert", "distilbert_seed7", "keras_bilstm", "tfidf_logreg"]
    records = {name: _record(name, 0.7, 60) for name in names}
    text = _resume(records, _all_pairs_comparison(names))
    assert "3 distinct models (`distilbert`, `distilbert_seed7`, `keras_bilstm`, " in text
    assert "in 6 pairs" in text
    assert 'do not write "compared four models"' in text
    assert 'Bullet 1 macro-F1 numbers are available; reword "compared four models".' in text

    names.append("llm_lora")
    records["llm_lora"] = _record("llm_lora", 0.7, 60)
    text = _resume(records, _all_pairs_comparison(names))
    assert "4 distinct models" in text and "in 10 pairs" in text
    assert "fewer than four" not in text and "Bullet 1 numbers are available." in text


def test_seed_runs_note_shows_mean_and_std_only_while_current():
    records = [_record("distilbert", 0.80, 60), _record("distilbert_seed43", 0.76, 60)]
    comparison = {
        "n_resamples": 100,
        "alpha": 0.05,
        "models": {"distilbert": _ci_entry(0.80), "distilbert_seed43": _ci_entry(0.76)},
        "pairs": [],
        "skipped_pairs": [],
        "seed_runs": {
            "distilbert": {
                "runs": [
                    {"model": "distilbert", "seed": 42, "n": 60, "macro_f1": 0.80},
                    {"model": "distilbert_seed43", "seed": 43, "n": 60, "macro_f1": 0.76},
                ],
                "n_runs": 2,
                "macro_f1_mean": 0.78,
                "macro_f1_std": 0.0283,
            }
        },
    }
    section = reporting.results_section(records, comparison, n_test_full=60)
    expected = "`distilbert`: macro-F1 0.780 ± 0.028 (mean ± sample std over 2 runs, seeds 42, 43)"
    assert expected in section

    # The seeds shown are the ones saved in the metrics JSONs, not parsed from file names.
    records[1]["seed"] = 44  # e.g. trained with --seed 44 but named distilbert_seed43
    assert "seeds 42, 44)" in reporting.results_section(records, comparison, n_test_full=60)

    # A seed run retrained after compare_models.py ran: the old mean must not be shown.
    records[1] = _record("distilbert_seed43", 0.70, 60)
    section = reporting.results_section(records, comparison, n_test_full=60)
    assert "0.780" not in section and "runs changed since the last comparison" in section


def test_rag_section_shows_every_chunk_size():
    section = reporting.rag_section(RAG)
    assert "| 300 | 155 | 0.500 | 0.750 | 24 |" in section
    assert "| 800 | 63 | 0.625 | 0.875 | 24 |" in section
    assert "Best chunk size: 800" in section


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("tests/test_a.py::test_x\n\n96 tests collected in 5.09s\n", 96),
        ("81/96 tests collected (15 deselected) in 5.28s", 81),
        ("1 test collected in 0.10s", 1),
        ("no tests ran in 0.01s", None),
    ],
)
def test_parse_collected(output, expected):
    assert make_results_table.parse_collected(output) == expected
