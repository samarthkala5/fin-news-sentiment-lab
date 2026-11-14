"""Tests for scripts/run_kmeans.py. CPU only: embeddings are faked, no model is loaded."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from fnsl import config, metrics

REQUIRED_KEYS = {
    "model",
    "kind",
    "dataset",
    "split",
    "n",
    "embedding_model",
    "seed",
    "git_commit",
    "timestamp",
    "pca_explained_variance_ratio_2d",
    "runs",
}
REQUIRED_RUN_KEYS = {
    "k",
    "silhouette",
    "ari",
    "nmi",
    "contingency",
    "cluster_majority_label",
    "cluster_purity",
}


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its functions."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script("run_kmeans")


def test_contingency_table_counts_cluster_label_pairs(script):
    clusters = np.array([0, 0, 1, 1, 1, 2])
    labels = np.array([0, 0, 1, 1, 2, 2])
    table = script.contingency_table(clusters, labels, k=3)
    assert table.tolist() == [[2, 0, 0], [0, 2, 1], [0, 0, 1]]


def test_separated_groups_with_matching_labels_score_perfectly(script):
    rng = np.random.default_rng(0)
    centres = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    labels = np.repeat([0, 1, 2], 40)
    X = centres[labels] + rng.normal(scale=0.3, size=(120, 2))  # one tight blob per label

    run, clusters = script.evaluate_kmeans(X, labels, k=3, seed=0)
    assert run["ari"] == pytest.approx(1.0)
    assert run["nmi"] == pytest.approx(1.0)
    assert run["silhouette"] > 0.8
    assert run["cluster_purity"] == [1.0, 1.0, 1.0]
    assert sorted(run["cluster_majority_label"]) == sorted(config.LABELS)
    assert np.asarray(run["contingency"]).sum() == len(clusters) == 120


def test_purity_is_high_for_free_when_one_label_dominates(script):
    """Labels unrelated to the clusters: ARI is ~0, yet purity is at least the share of
    the biggest class. That is why purity must be compared with that share."""
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(loc=c, scale=0.3, size=(100, 2)) for c in (-5.0, 0.0, 5.0)])
    labels = rng.choice(3, size=300, p=[0.15, 0.15, 0.70])

    run, _ = script.evaluate_kmeans(X, labels, k=3, seed=0)
    assert abs(run["ari"]) < 0.05
    assert run["overall_purity"] >= np.mean(labels == 2)


def test_parse_args_rejects_bad_k_and_the_test_split(script):
    assert script.parse_args(["--ks", "5", "3", "3"]).ks == [3, 5]
    with pytest.raises(SystemExit):
        script.parse_args(["--ks", "1"])
    with pytest.raises(SystemExit):
        script.parse_args(["--split", "test"])  # test split = final evaluation only


@pytest.mark.skipif(not (config.SPLITS_DIR / "train.csv").exists(), reason="no split files")
def test_main_smoke_writes_kmeans_json_and_figure(script, tmp_results, monkeypatch):
    calls = {}

    def fake_embed(texts, model_name, batch_size, normalize, **kwargs):
        """Random unit vectors instead of a real model (no torch, no download)."""
        calls.update(n=len(texts), model_name=model_name, batch_size=batch_size, norm=normalize)
        X = np.random.default_rng(0).normal(size=(len(texts), 16)).astype(np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    monkeypatch.setattr(script, "embed_texts", fake_embed)
    record = script.main(["--smoke", "--no-mlflow", "--batch-size", "8", "--model-name", "fake/m"])
    # The silhouette distance (Euclidean on unit vectors) needs normalised embeddings.
    assert calls == {"n": script.SMOKE_N, "model_name": "fake/m", "batch_size": 8, "norm": True}

    path = tmp_results / "smoke" / "metrics" / "kmeans.json"
    on_disk = json.loads(path.read_text())
    assert on_disk == record
    assert REQUIRED_KEYS <= on_disk.keys()
    assert on_disk["model"] == "kmeans_minilm" and on_disk["kind"] == "clustering"
    assert on_disk["n"] == script.SMOKE_N and on_disk["smoke"] is True
    # No macro_f1: results tables and load_metrics() must skip this file.
    assert "macro_f1" not in on_disk and not metrics.is_model_metrics(on_disk)

    assert [run["k"] for run in on_disk["runs"]] == [3, 5]
    for run in on_disk["runs"]:
        assert REQUIRED_RUN_KEYS <= run.keys()
        table = np.asarray(run["contingency"])
        assert table.shape == (run["k"], config.NUM_LABELS)
        assert table.sum() == script.SMOKE_N
        assert table.sum(axis=0).tolist() == list(on_disk["label_counts"].values())
        assert len(run["cluster_purity"]) == len(run["cluster_majority_label"]) == run["k"]
        assert -1.0 <= run["silhouette"] <= 1.0

    ratio = on_disk["pca_explained_variance_ratio_2d"]
    assert len(ratio) == 2 and 1.0 >= ratio[0] >= ratio[1] > 0.0
    assert (tmp_results / "smoke" / "figures" / "kmeans_pca.png").exists()
    assert not (tmp_results / "metrics").exists(), "smoke run must not write real results"
