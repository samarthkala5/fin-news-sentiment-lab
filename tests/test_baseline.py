"""Tests for the TF-IDF + logistic regression baseline."""

from __future__ import annotations

import importlib.util
import json

import pytest

from fnsl import config
from fnsl.baselines import build_pipeline, tune_on_val


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_learns_toy_data(toy_df):
    model = build_pipeline(C=10.0, min_df=1)
    model.fit(toy_df["text"], toy_df["label"])
    assert (model.predict(toy_df["text"]) == toy_df["label"]).mean() > 0.9


def test_tune_on_val_returns_best_of_grid(toy_df):
    train, val = toy_df.iloc[:45], toy_df.iloc[45:]
    best, grid = tune_on_val(train, val, c_grid=[0.01, 10.0], class_weight_grid=[None])
    assert len(grid) == 2
    best_score = max(r["val_macro_f1"] for r in grid)
    assert any(r["C"] == best["C"] and r["val_macro_f1"] == best_score for r in grid), (
        "best must be the grid point with the highest val macro-F1"
    )


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_run_baseline_smoke_writes_schema(tmp_results):
    script = _load_script("run_baseline")
    record = script.main(["--smoke", "--no-mlflow", "--c-grid", "1.0"])
    path = tmp_results / "smoke" / "metrics" / "tfidf_logreg.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["smoke"] is True
    assert on_disk["n_test"] == config.SMOKE_SIZES["test"] == record["n_test"]
    assert not (tmp_results / "metrics").exists(), "smoke run must not write real results"
