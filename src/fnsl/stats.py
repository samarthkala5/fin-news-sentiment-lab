"""Statistics for comparing classifiers that were scored on the same test set.

A single macro-F1 number hides how much it would move on another sample of headlines.
The bootstrap estimates that: re-draw the test rows with replacement many times, rescore
the model each time, and read a confidence interval off the spread of the scores.
To compare two models we use PAIRED methods, because both models labelled exactly the
same rows: McNemar's test looks only at rows where one model is right and the other is
wrong, and the paired bootstrap re-draws the same rows for both models and looks at the
difference in macro-F1. Holm's method corrects the p-values when many pairs are tested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from statsmodels.stats.contingency_tables import mcnemar

from fnsl import config
from fnsl.metrics import macro_f1

# A metric takes (y_true, y_pred) and returns one number, e.g. fnsl.metrics.macro_f1.
Metric = Callable[[np.ndarray, np.ndarray], float]

# Below this many discordant rows the chi-square approximation is poor, so McNemar uses
# the exact binomial test instead (a common rule of thumb).
EXACT_BELOW = 25


def accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Share of rows where the prediction equals the label (UNPARSEABLE counts as wrong)."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def _as_int_arrays(*arrays: npt.ArrayLike) -> list[np.ndarray]:
    """Convert label/prediction vectors to 1-D int arrays and check they line up."""
    out = [np.asarray(a, dtype=int) for a in arrays]
    if any(a.ndim != 1 for a in out):
        raise ValueError("labels and predictions must be 1-D")
    lengths = {len(a) for a in out}
    if len(lengths) != 1:
        raise ValueError(f"labels and predictions have different lengths: {sorted(lengths)}")
    if lengths == {0}:
        raise ValueError("labels and predictions are empty")
    return out


def _check_bootstrap_args(n_resamples: int, alpha: float) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def mcnemar_test(
    y_true: npt.ArrayLike,
    pred_a: npt.ArrayLike,
    pred_b: npt.ArrayLike,
    exact: bool | None = None,
) -> dict[str, Any]:
    """McNemar's test: do models A and B have the same error rate on data like these rows?

    Every test row is right or wrong for each model, which gives a 2x2 table:

                     B right   B wrong
        A right   [[   n11,       b   ],
        A wrong    [    c,       n22  ]]

    Rows where both models agree (n11 and n22) say nothing about which model is better,
    so the test uses only the discordant counts b (A right, B wrong) and c (A wrong,
    B right). If the two models were equally good, each discordant row would be like a
    fair coin flip, so b should be close to (b + c) / 2; the p-value says how surprising
    the observed split is under that assumption. The test is paired: both models are
    scored on the same rows, so how hard each row is cancels out. That makes it the right
    test here, and more powerful than comparing two accuracies as independent samples.

    exact=None picks the method automatically: the exact binomial test when b + c < 25,
    otherwise the chi-square approximation with continuity correction (statsmodels).
    Returns the table, b, c, statistic (min(b, c) for the exact test, the chi-square value
    otherwise), p_value and method.
    """
    y_true, pred_a, pred_b = _as_int_arrays(y_true, pred_a, pred_b)
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    table = [
        [int(np.sum(a_right & b_right)), int(np.sum(a_right & ~b_right))],
        [int(np.sum(~a_right & b_right)), int(np.sum(~a_right & ~b_right))],
    ]
    b, c = table[0][1], table[1][0]
    if exact is None:
        exact = b + c < EXACT_BELOW
    method = "exact binomial" if exact else "chi-square with continuity correction"

    if b == c:
        # The discordant rows split exactly evenly (also when b = c = 0, e.g. identical
        # predictions), which is exactly what "equally good" predicts, so p = 1. Set
        # directly because statsmodels' corrected chi-square gives (|b - c| - 1)^2 / (b + c)
        # = 1 / (b + c) > 0 here, so p < 1 (and 1 / 0 = inf, so p = 0, when b = c = 0).
        # R's mcnemar.test skips the correction when b = c; the exact test already gives 1.
        statistic = float(b) if exact else 0.0
        p_value = 1.0
    else:
        result = mcnemar(np.array(table), exact=exact, correction=True)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "table": table,
        "b": b,
        "c": c,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
    }


def bootstrap_ci(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    metric: Metric = macro_f1,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval of a metric on one test set.

    Treat the test set as if it were the whole population: draw n rows WITH replacement
    (some rows appear twice, some not at all), score the model on that resample, and
    repeat n_resamples times. The middle (1 - alpha) share of the scores is the interval:
    with alpha=0.05, the 2.5th and 97.5th percentiles give a 95% CI.
    Meaning: a range of plausible values for the model's true score on headlines like
    these. If the whole experiment were repeated with fresh test sets, about 95% (for
    alpha=0.05) of the intervals built this way would contain the true score. It captures
    test-set sampling noise only, not the effect of retraining with another seed or using
    another split.
    """
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    _check_bootstrap_args(n_resamples, alpha)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = rng.integers(0, n, size=n)  # row numbers drawn with replacement
        scores[i] = metric(y_true[rows], y_pred[rows])
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(metric(y_true, y_pred)),
        "low": float(low),
        "high": float(high),
        "n_resamples": n_resamples,
        "alpha": alpha,
        "method": "percentile",
    }


def paired_bootstrap_diff(
    y_true: npt.ArrayLike,
    pred_a: npt.ArrayLike,
    pred_b: npt.ArrayLike,
    metric: Metric = macro_f1,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> dict[str, float]:
    """Paired bootstrap of the difference metric(A) - metric(B), with CI and p-value.

    In every resample the SAME rows are drawn for both models, so the luck of the draw
    (a resample with many easy or many hard rows) hits both models alike and cancels out
    in the difference. The CI is the percentile interval of the resampled differences.

    p-value (two-sided): how often does the resampled difference land on the other side
    of zero? Count the resamples with difference <= 0 and those with difference >= 0,
    take the smaller share and double it (two-sided), capped at 1:
        p = min(1, 2 * min(share(diff <= 0), share(diff >= 0)))
    Each share is (count + 1) / (n_resamples + 1), so p is never exactly 0: with 2000
    resamples we cannot resolve p-values below about 1/1000. p < alpha roughly matches
    "the (1 - alpha) CI excludes 0".
    Caveats: this is an approximation, not an exact test. The resamples are centred on
    the observed difference, not on "no difference"; test rows are assumed to be
    independent draws; p-values close to alpha can flip with another seed. It
    complements McNemar, which tests error rates (right/wrong per row), not macro-F1.
    """
    y_true, pred_a, pred_b = _as_int_arrays(y_true, pred_a, pred_b)
    _check_bootstrap_args(n_resamples, alpha)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = rng.integers(0, n, size=n)  # one draw, used for BOTH models
        diffs[i] = metric(y_true[rows], pred_a[rows]) - metric(y_true[rows], pred_b[rows])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    share_at_or_below_zero = (np.sum(diffs <= 0) + 1) / (n_resamples + 1)
    share_at_or_above_zero = (np.sum(diffs >= 0) + 1) / (n_resamples + 1)
    p_value = min(1.0, 2 * min(share_at_or_below_zero, share_at_or_above_zero))
    return {
        "diff": float(metric(y_true, pred_a) - metric(y_true, pred_b)),
        "low": float(low),
        "high": float(high),
        "p_value": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the same order as the input.

    Testing m model pairs at the 0.05 level gives m chances of a false positive. Holm's
    method keeps the chance of even one false positive across all m tests at or below
    alpha, while being less strict than plain Bonferroni (which multiplies every p-value
    by m):
      1. sort the p-values from smallest to largest;
      2. multiply the smallest by m, the next by m - 1, ..., the largest by 1;
      3. walk up the sorted list keeping a running maximum (an adjusted p-value may not be
         smaller than the one before it) and cap at 1.
    A pair is significant after correction if its adjusted p-value is below alpha.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a flat list")
    if np.any(np.isnan(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError(f"p-values must be between 0 and 1, got {p.tolist()}")
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")  # positions of the p-values, smallest first
    scaled = p[order] * np.arange(m, 0, -1)  # multiply by m, m-1, ..., 1
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted  # put them back in the input order
    return [float(x) for x in adjusted]
