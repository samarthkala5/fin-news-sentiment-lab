"""Tests for McNemar's test, bootstrap CIs and the Holm correction (fnsl.stats)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest, chi2
from statsmodels.stats.multitest import multipletests

from fnsl import config
from fnsl.metrics import macro_f1
from fnsl.stats import accuracy, bootstrap_ci, holm_adjust, mcnemar_test, paired_bootstrap_diff


def _noisy_predictions(y_true: np.ndarray, p_correct: float, seed: int) -> np.ndarray:
    """Predictions that are right with probability p_correct, else the next label."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(len(y_true)) < p_correct, y_true, (y_true + 1) % 3)


def _pair_with_table(n11: int, b: int, c: int, n22: int):
    """Labels and two prediction vectors whose right/wrong table is [[n11, b], [c, n22]]."""
    y_true = np.zeros(n11 + b + c + n22, dtype=int)
    pred_a = np.zeros_like(y_true)
    pred_b = np.zeros_like(y_true)
    pred_b[n11 : n11 + b] = 1  # A right, B wrong
    pred_a[n11 + b : n11 + b + c] = 1  # A wrong, B right
    pred_a[n11 + b + c :] = 2  # both wrong
    pred_b[n11 + b + c :] = 1
    return y_true, pred_a, pred_b


# --- McNemar --------------------------------------------------------------------------
@pytest.mark.parametrize("exact", [None, True, False])
def test_mcnemar_identical_predictions_give_p_one(exact):
    y_true = np.array([0, 1, 2] * 10)
    pred = _noisy_predictions(y_true, 0.7, seed=1)
    result = mcnemar_test(y_true, pred, pred.copy(), exact=exact)
    assert result["b"] == result["c"] == 0
    assert result["p_value"] == 1.0 and not math.isnan(result["p_value"])


def test_mcnemar_known_table_matches_binomial_test():
    y_true, pred_a, pred_b = _pair_with_table(n11=50, b=7, c=2, n22=5)
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["table"] == [[50, 7], [2, 5]]
    assert (result["b"], result["c"]) == (7, 2)
    assert result["method"] == "exact binomial"  # b + c = 9 < 25
    # Under "equally good", the 9 discordant rows are fair coin flips.
    assert result["p_value"] == pytest.approx(binomtest(2, 9, 0.5).pvalue)
    assert result["statistic"] == 2  # min(b, c) for the exact test


def test_mcnemar_uses_chi_square_for_many_discordant_rows():
    y_true, pred_a, pred_b = _pair_with_table(n11=40, b=40, c=20, n22=0)
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["method"] == "chi-square with continuity correction"
    statistic = (abs(40 - 20) - 1) ** 2 / (40 + 20)
    assert result["statistic"] == pytest.approx(statistic)
    assert result["p_value"] == pytest.approx(chi2.sf(statistic, df=1))


@pytest.mark.parametrize("b", [3, 15])  # 3+3 -> exact test, 15+15 -> chi-square
def test_mcnemar_even_split_of_discordant_rows_gives_p_one(b):
    # Found in a smoke run: b = c = 15 gave p = 0.855 because the continuity correction
    # (|b - c| - 1)^2 / (b + c) is 1/30, not 0, when b = c.
    y_true, pred_a, pred_b = _pair_with_table(n11=10, b=b, c=b, n22=2)
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["p_value"] == 1.0


def test_mcnemar_is_symmetric_in_the_two_models():
    y_true, pred_a, pred_b = _pair_with_table(n11=10, b=8, c=1, n22=3)
    ab, ba = mcnemar_test(y_true, pred_a, pred_b), mcnemar_test(y_true, pred_b, pred_a)
    assert (ab["b"], ab["c"]) == (ba["c"], ba["b"])
    assert ab["p_value"] == pytest.approx(ba["p_value"])


def test_mcnemar_counts_unparseable_as_wrong():
    y_true = np.array([0, 1, 2, 2])
    result = mcnemar_test(y_true, y_true, np.array([0, 1, 2, config.UNPARSEABLE]))
    assert (result["b"], result["c"]) == (1, 0)


def test_mismatched_lengths_and_empty_inputs_raise():
    with pytest.raises(ValueError):
        mcnemar_test([0, 1], [0, 1], [0])
    with pytest.raises(ValueError):
        bootstrap_ci([], [])


# --- bootstrap CI ---------------------------------------------------------------------
def test_bootstrap_ci_contains_point_estimate():
    y_true = np.array([0, 1, 2] * 100)
    pred = _noisy_predictions(y_true, 0.8, seed=2)
    ci = bootstrap_ci(y_true, pred, n_resamples=500, seed=0)
    assert ci["point"] == pytest.approx(macro_f1(y_true, pred))
    assert ci["low"] < ci["point"] < ci["high"]
    assert ci["method"] == "percentile" and ci["n_resamples"] == 500 and ci["alpha"] == 0.05


def test_bootstrap_ci_narrows_with_more_data():
    y_small = np.array([0, 1, 2] * 50)
    pred_small = _noisy_predictions(y_small, 0.75, seed=3)
    # Same rows repeated 10 times: same point estimate, 10x the data.
    y_big, pred_big = np.tile(y_small, 10), np.tile(pred_small, 10)
    small = bootstrap_ci(y_small, pred_small, n_resamples=400, seed=0)
    big = bootstrap_ci(y_big, pred_big, n_resamples=400, seed=0)
    assert big["point"] == pytest.approx(small["point"])
    assert big["high"] - big["low"] < (small["high"] - small["low"]) / 2


def test_bootstrap_is_reproducible_with_a_seed_and_takes_any_metric():
    y_true = np.array([0, 1, 2] * 40)
    pred = _noisy_predictions(y_true, 0.7, seed=4)
    first = bootstrap_ci(y_true, pred, metric=accuracy, n_resamples=200, seed=7)
    second = bootstrap_ci(y_true, pred, metric=accuracy, n_resamples=200, seed=7)
    assert first == second
    assert first["point"] == pytest.approx(np.mean(y_true == pred))


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], [0, 1], n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], [0, 1], alpha=1.5)


# --- paired bootstrap difference ---------------------------------------------------
def test_paired_diff_of_model_with_itself_is_zero():
    y_true = np.array([0, 1, 2] * 30)
    pred = _noisy_predictions(y_true, 0.7, seed=5)
    result = paired_bootstrap_diff(y_true, pred, pred, n_resamples=300)
    assert result == {"diff": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}


def test_paired_diff_detects_a_clearly_better_model():
    y_true = np.array([0, 1, 2] * 100)
    good = _noisy_predictions(y_true, 0.9, seed=6)
    bad = _noisy_predictions(y_true, 0.5, seed=7)
    result = paired_bootstrap_diff(y_true, good, bad, n_resamples=500)
    assert result["diff"] == pytest.approx(macro_f1(y_true, good) - macro_f1(y_true, bad))
    assert 0 < result["low"] <= result["diff"] <= result["high"]
    # No resample crossed zero, so p is the smallest value 500 resamples can give.
    assert result["p_value"] == pytest.approx(2 / 501)
    flipped = paired_bootstrap_diff(y_true, bad, good, n_resamples=500)
    assert flipped["diff"] == pytest.approx(-result["diff"])
    assert flipped["p_value"] == pytest.approx(result["p_value"])


# --- Holm ---------------------------------------------------------------------------
def test_holm_on_known_values():
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> running max 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm_adjust([0.02]) == pytest.approx([0.02])  # one test: no correction
    assert holm_adjust([]) == []


def test_holm_matches_statsmodels():
    p_values = [0.001, 0.2, 0.013, 0.04, 0.5, 0.0101]
    expected = multipletests(p_values, method="holm")[1]
    assert holm_adjust(p_values) == pytest.approx(list(expected))


def test_holm_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        holm_adjust([0.1, float("nan")])
    with pytest.raises(ValueError):
        holm_adjust([1.2])
