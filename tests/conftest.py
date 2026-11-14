"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from fnsl import config


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    """Redirect every results folder to a temp dir so tests never touch results/."""
    results = tmp_path / "results"
    monkeypatch.setattr(config, "RESULTS_DIR", results)
    monkeypatch.setattr(config, "METRICS_DIR", results / "metrics")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", results / "predictions")
    monkeypatch.setattr(config, "FIGURES_DIR", results / "figures")
    monkeypatch.setattr(config, "SMOKE_DIR", results / "smoke")
    return results


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """60 short labelled 'headlines' (20 per class) in the split-file format."""
    words = {0: "falls plunges cut", 1: "rises beats upgrade", 2: "announces meeting update"}
    rows = []
    for i in range(60):
        label = i % 3
        rows.append({"id": i, "text": f"company {i} {words[label]} item{i}", "label": label})
    return pd.DataFrame(rows)
