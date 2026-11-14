"""Tests for scripts/compare_models.py on synthetic prediction files (no real results)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fnsl import config, metrics


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_models = _load_script("compare_models")


def _write_predictions(
    folder: Path, model: str, ids, p_correct: float, seed: int, labels=None
) -> pd.DataFrame:
    """A predictions CSV in the standard schema; labels default to id % 3."""
    ids = np.asarray(list(ids))
    y_true = ids % 3 if labels is None else np.asarray(labels)
    rng = np.random.default_rng(seed)
    pred = np.where(rng.random(len(ids)) < p_correct, y_true, (y_true + 1) % 3)
    df = pd.DataFrame({"id": ids, "text": [f"headline {i}" for i in ids], "label": y_true})
    df["pred"] = pred
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{model}.csv", index=False)
    return df


def test_compare_models_pairs_matching_ids_and_skips_others(tmp_results):
    folder = tmp_results / "predictions"
    a = _write_predictions(folder, "model_a", range(120), 0.9, seed=1)
    _write_predictions(folder, "model_b", range(120), 0.6, seed=2)
    _write_predictions(folder, "model_c", range(60, 180), 0.8, seed=3)  # other ids

    result = compare_models.main(["--n-resamples", "200"])

    out = tmp_results / "metrics" / "comparison.json"
    on_disk = json.loads(out.read_text())
    assert on_disk == json.loads(json.dumps(result))
    assert on_disk["kind"] == "comparison" and on_disk["smoke"] is False
    for key in ("n_resamples", "alpha", "seed", "git_commit", "timestamp"):
        assert key in on_disk
    assert set(on_disk["models"]) == {"model_a", "model_b", "model_c"}
    for m in on_disk["models"].values():
        assert m["ci_low"] <= m["macro_f1"] <= m["ci_high"]
        assert m["acc_ci_low"] <= m["accuracy"] <= m["acc_ci_high"]
    assert on_disk["models"]["model_a"]["macro_f1"] == pytest.approx(
        metrics.macro_f1(a["label"], a["pred"])
    )

    assert len(on_disk["pairs"]) == 1
    pair = on_disk["pairs"][0]
    assert (pair["model_a"], pair["model_b"], pair["n"]) == ("model_a", "model_b", 120)
    assert pair["mcnemar"]["b"] > pair["mcnemar"]["c"]  # model_a is the better one
    assert pair["macro_f1_diff"]["diff"] > 0
    assert pair["p_holm"] == pytest.approx(pair["mcnemar"]["p_value"])  # only one test
    assert pair["significant_at_0.05"] is (pair["mcnemar"]["p_value"] < 0.05)
    assert pair["significant_holm"] is (pair["p_holm"] < 0.05)

    skipped = {(s["model_a"], s["model_b"]): s["reason"] for s in on_disk["skipped_pairs"]}
    assert set(skipped) == {("model_a", "model_c"), ("model_b", "model_c")}
    assert all("different test ids" in reason for reason in skipped.values())


def test_holm_is_applied_across_all_pairs(tmp_results):
    folder = tmp_results / "predictions"
    for i, p_correct in enumerate([0.9, 0.7, 0.5]):
        _write_predictions(folder, f"m{i}", range(150), p_correct, seed=10 + i)
    result = compare_models.main(["--n-resamples", "100"])
    raw = [p["mcnemar"]["p_value"] for p in result["pairs"]]
    assert len(raw) == 3
    for pair in result["pairs"]:
        assert pair["p_holm"] >= pair["mcnemar"]["p_value"]
    assert min(p["p_holm"] for p in result["pairs"]) == pytest.approx(min(1.0, 3 * min(raw)))


def test_rows_are_aligned_by_id_not_by_file_order(tmp_results):
    folder = tmp_results / "predictions"
    df = _write_predictions(folder, "model_a", range(90), 0.8, seed=4)
    df.sample(frac=1.0, random_state=0).to_csv(folder / "model_a_shuffled.csv", index=False)
    result = compare_models.main(["--n-resamples", "50"])
    pair = result["pairs"][0]
    assert pair["mcnemar"]["b"] == pair["mcnemar"]["c"] == 0  # same predictions per id
    assert pair["mcnemar"]["p_value"] == 1.0
    assert pair["macro_f1_diff"]["diff"] == 0.0


def test_pair_with_disagreeing_labels_is_skipped_with_reason(tmp_results):
    folder = tmp_results / "predictions"
    _write_predictions(folder, "model_a", range(30), 0.8, seed=5)
    _write_predictions(folder, "model_b", range(30), 0.8, seed=6, labels=[0] * 30)
    result = compare_models.main(["--n-resamples", "50"])
    assert result["pairs"] == []
    assert "labels disagree" in result["skipped_pairs"][0]["reason"]


def test_seed_reruns_are_summarised_not_paired(tmp_results):
    folder = tmp_results / "predictions"
    runs = {"distilbert": 1, "distilbert_seed43": 2, "distilbert_seed44": 3}
    for name, seed in runs.items():
        _write_predictions(folder, name, range(90), 0.8, seed=seed)
    _write_predictions(folder, "tfidf_logreg", range(90), 0.7, seed=4)
    result = compare_models.main(["--n-resamples", "50"])

    # Every run gets its own CI, but only distinct models are tested against each other.
    assert set(result["models"]) == {*runs, "tfidf_logreg"}
    assert [(p["model_a"], p["model_b"]) for p in result["pairs"]] == [
        ("distilbert", "tfidf_logreg")
    ]
    group = result["seed_runs"]["distilbert"]
    assert [(r["model"], r["seed"]) for r in group["runs"]] == [
        ("distilbert", 42),
        ("distilbert_seed43", 43),
        ("distilbert_seed44", 44),
    ]
    scores = [result["models"][name]["macro_f1"] for name in runs]
    assert group["n_runs"] == 3
    assert group["macro_f1_mean"] == pytest.approx(np.mean(scores))
    assert group["macro_f1_std"] == pytest.approx(np.std(scores, ddof=1))
    assert set(result["seed_runs"]) == {"distilbert"}

    # Without its base model in the comparison, a "_seed" file is just another model.
    subset = ["--n-resamples", "50", "--models", "distilbert_seed43", "tfidf_logreg"]
    result = compare_models.main(subset)
    assert len(result["pairs"]) == 1 and result["seed_runs"] == {}


def test_single_model_gives_cis_and_no_pairs(tmp_results):
    _write_predictions(tmp_results / "predictions", "only_model", range(60), 0.8, seed=7)
    result = compare_models.main(["--n-resamples", "50"])
    assert list(result["models"]) == ["only_model"]
    assert result["pairs"] == [] and result["skipped_pairs"] == []


def test_bad_files_are_reported_not_silently_dropped(tmp_results):
    folder = tmp_results / "predictions"
    _write_predictions(folder, "good", range(30), 0.8, seed=8)
    pd.DataFrame({"id": [1], "prediction": [0]}).to_csv(folder / "wrong_columns.csv", index=False)
    dup = _write_predictions(folder, "dup_ids", range(30), 0.8, seed=9)
    pd.concat([dup, dup.head(2)]).to_csv(folder / "dup_ids.csv", index=False)
    # Values that are not class ids must be reported too, not crash the whole run or be
    # silently truncated to an int (1.5 -> 1).
    text_preds = dup.assign(pred=["Bullish"] * 29 + [""])
    text_preds.to_csv(folder / "text_preds.csv", index=False)
    dup.assign(pred=dup["pred"] + 0.5).to_csv(folder / "float_preds.csv", index=False)
    _write_predictions(folder, "label_out_of_range", range(30), 0.8, seed=10, labels=[5] * 30)
    result = compare_models.main(["--n-resamples", "50"])
    assert list(result["models"]) == ["good"]
    reasons = {s["model"]: s["reason"] for s in result["skipped_models"]}
    assert set(reasons) == {
        "wrong_columns",
        "dup_ids",
        "text_preds",
        "float_preds",
        "label_out_of_range",
    }
    assert "repeated ids" in reasons["dup_ids"]
    assert reasons["wrong_columns"].startswith("wrong_columns.csv has columns")
    assert str(tmp_results) not in reasons["wrong_columns"]
    for name in ("text_preds", "float_preds", "label_out_of_range"):
        assert "label is not 0/1/2 or pred is not 0/1/2/-1" in reasons[name]


def test_models_option_selects_a_subset(tmp_results):
    folder = tmp_results / "predictions"
    for name in ("x", "y", "z"):
        _write_predictions(folder, name, range(30), 0.8, seed=len(name))
    result = compare_models.main(["--n-resamples", "50", "--models", "x", "z"])
    assert set(result["models"]) == {"x", "z"}
    with pytest.raises(SystemExit):
        compare_models.main(["--models", "x", "not_there"])


def test_smoke_reads_and_writes_only_smoke_folders(tmp_results):
    _write_predictions(tmp_results / "smoke" / "predictions", "m", range(64), 0.8, seed=11)
    result = compare_models.main(["--smoke", "--n-resamples", "50"])
    assert result["smoke"] is True
    assert (tmp_results / "smoke" / "metrics" / "comparison.json").exists()
    assert not (tmp_results / "metrics").exists(), "a smoke run must not write real results"


def test_smoke_predictions_dir_without_smoke_flag_is_still_a_smoke_run(tmp_results):
    smoke_predictions = tmp_results / "smoke" / "predictions"
    _write_predictions(smoke_predictions, "m1", range(64), 0.8, seed=12)
    _write_predictions(smoke_predictions, "m2", range(64), 0.6, seed=13)
    out = tmp_results / "metrics" / "comparison.json"
    result = compare_models.main(
        ["--predictions-dir", str(smoke_predictions), "--out", str(out), "--n-resamples", "50"]
    )
    assert result["smoke"] is True and len(result["pairs"]) == 1
    assert (tmp_results / "smoke" / "metrics" / "comparison.json").exists()
    assert not out.exists(), "smoke predictions must never write the real comparison.json"


def test_no_predictions_exits_with_a_message(tmp_results):
    (tmp_results / "predictions").mkdir(parents=True)
    with pytest.raises(SystemExit):
        compare_models.main([])
    assert not (tmp_results / "metrics" / "comparison.json").exists()
