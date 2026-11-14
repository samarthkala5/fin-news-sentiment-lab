"""Tests for metric computation and the standard results-file schema."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fnsl import config, metrics


def test_compute_metrics_perfect_and_known_case():
    perfect = metrics.compute_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["accuracy"] == pytest.approx(1.0)

    # true:  0 0 1 1 2 2 ; pred: 0 1 1 1 2 0
    scores = metrics.compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    # class 0: P=1/2 R=1/2 F1=0.5 ; class 1: P=2/3 R=1 F1=0.8 ; class 2: P=1 R=1/2 F1=2/3
    assert scores["per_class_f1"]["Bearish"] == pytest.approx(0.5)
    assert scores["per_class_f1"]["Bullish"] == pytest.approx(0.8)
    assert scores["per_class_f1"]["Neutral"] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert scores["accuracy"] == pytest.approx(4 / 6)
    assert np.array(scores["confusion_matrix"]).sum() == 6


def test_unparseable_predictions_count_as_wrong():
    u = config.UNPARSEABLE
    scores = metrics.compute_metrics([0, 1, 2, 2], [0, 1, 2, u])
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["n_unparseable"] == 1
    assert scores["per_class_f1"]["Neutral"] < 1.0
    assert scores["macro_f1"] < 1.0


def test_save_results_follows_schema(tmp_results, toy_df):
    y_pred = toy_df["label"].to_numpy().copy()
    y_pred[:5] = (y_pred[:5] + 1) % 3  # make a few mistakes
    record = metrics.save_results(
        "toy_model", toy_df, y_pred, params={"C": 1.0, "ngram_range": (1, 2)}, train_time_s=1.234
    )

    metrics_file = tmp_results / "metrics" / "toy_model.json"
    preds_file = tmp_results / "predictions" / "toy_model.csv"
    assert metrics_file.exists() and preds_file.exists()
    assert (tmp_results / "figures" / "toy_model_confusion.png").exists()

    on_disk = json.loads(metrics_file.read_text())
    assert on_disk == json.loads(json.dumps(record))
    for key in metrics.REQUIRED_METRIC_KEYS:
        assert key in on_disk, key
    assert on_disk["n_test"] == len(toy_df)
    assert set(on_disk["per_class_f1"]) == set(config.LABELS)
    assert 0.0 <= on_disk["macro_f1"] <= 1.0
    assert on_disk["params"]["ngram_range"] == [1, 2]
    assert on_disk["smoke"] is False
    assert isinstance(on_disk["git_commit"], str) and on_disk["git_commit"]

    preds = metrics.load_predictions(preds_file)
    assert list(preds.columns) == metrics.PREDICTION_COLUMNS
    assert list(preds["id"]) == list(toy_df["id"])
    assert (preds["pred"] != preds["label"]).sum() == 5


def test_smoke_results_go_to_smoke_folder(tmp_results, toy_df):
    metrics.save_results(
        "toy_smoke", toy_df, toy_df["label"], params={}, train_time_s=0, smoke=True
    )
    assert (tmp_results / "smoke" / "metrics" / "toy_smoke.json").exists()
    assert not (tmp_results / "metrics" / "toy_smoke.json").exists()


def test_save_results_rejects_length_mismatch(tmp_results, toy_df):
    with pytest.raises(ValueError):
        metrics.save_results("bad", toy_df, [0, 1], params={}, train_time_s=0)


def test_load_metrics_skips_other_json(tmp_results, toy_df):
    metrics.save_results("m1", toy_df, toy_df["label"], params={}, train_time_s=0)
    (tmp_results / "metrics" / "comparison.json").write_text(json.dumps({"pairs": []}))
    loaded = metrics.load_metrics(tmp_results / "metrics")
    assert [r["model"] for r in loaded] == ["m1"]


def test_load_predictions_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1], "pred": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        metrics.load_predictions(path)
