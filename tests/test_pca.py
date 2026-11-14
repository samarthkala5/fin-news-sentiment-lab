"""Tests for the NumPy PCA (src/fnsl/pca.py), checked against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from fnsl.pca import PCA, flip_signs


def _random_data(n: int = 200, d: int = 10, seed: int = 0) -> np.ndarray:
    """Correlated features with a non-zero mean, so centring actually matters."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + rng.normal(scale=5.0, size=d)


def _low_rank_data(n: int = 150, d: int = 12, rank: int = 3, seed: int = 1) -> np.ndarray:
    """n points that lie exactly on a `rank`-dimensional (shifted) plane inside d dims."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d)) + rng.normal(size=d)


def _sign_alignment(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """+1/-1 per component that makes our direction point the same way as sklearn's.

    SVD signs are arbitrary, so two correct implementations may disagree by a sign.
    """
    signs = np.sign(np.sum(ours * theirs, axis=1))
    assert np.all(signs != 0), "components are orthogonal to sklearn's, not just flipped"
    return signs


def _assert_matches_sklearn(X: np.ndarray, k: int) -> None:
    ours = PCA(n_components=k).fit(X)
    # svd_solver="full" is exact; "auto" may pick an approximate or covariance solver.
    theirs = SklearnPCA(n_components=k, svd_solver="full").fit(X)

    np.testing.assert_allclose(ours.mean_, theirs.mean_)
    np.testing.assert_allclose(ours.explained_variance_, theirs.explained_variance_)
    np.testing.assert_allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_)
    np.testing.assert_allclose(ours.singular_values_, theirs.singular_values_)

    signs = _sign_alignment(ours.components_, theirs.components_)
    np.testing.assert_allclose(ours.components_ * signs[:, None], theirs.components_, atol=1e-10)
    # Projections flip together with their component, column by column.
    np.testing.assert_allclose(ours.transform(X) * signs, theirs.transform(X), atol=1e-8)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_matches_sklearn_on_random_data(k):
    _assert_matches_sklearn(_random_data(), k)


def test_matches_sklearn_on_low_rank_data():
    X = _low_rank_data(rank=3)
    _assert_matches_sklearn(X, 3)
    # Three directions hold all the variance of rank-3 data ...
    ratio = PCA(n_components=3).fit(X).explained_variance_ratio_
    assert ratio.sum() == pytest.approx(1.0)
    # ... and every direction after that holds (numerically) none.
    assert PCA(n_components=4).fit(X).explained_variance_ratio_[3] < 1e-12


def test_reconstruction_with_all_components_recovers_x():
    X = _random_data(n=50, d=8)
    pca = PCA(n_components=8)
    np.testing.assert_allclose(pca.inverse_transform(pca.fit_transform(X)), X, atol=1e-10)


def test_rank_k_data_is_recovered_from_k_components():
    X = _low_rank_data(rank=3)
    pca = PCA(n_components=3)
    np.testing.assert_allclose(pca.inverse_transform(pca.fit_transform(X)), X, atol=1e-10)


def test_projection_is_u_times_s_with_the_explained_variances():
    X = _random_data()
    pca = PCA(n_components=4)
    Z = pca.fit_transform(X)
    # Z = U_k S_k: column i has length S_i and the columns are orthogonal ...
    np.testing.assert_allclose(np.linalg.norm(Z, axis=0), pca.singular_values_)
    # ... so the projections are uncorrelated, with variance S_i^2 / (n - 1).
    np.testing.assert_allclose(np.cov(Z, rowvar=False), np.diag(pca.explained_variance_), atol=1e-8)


def test_components_are_orthonormal_and_signs_are_fixed():
    pca = PCA(n_components=5).fit(_random_data())
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(5), atol=1e-12)
    rows = np.arange(5)
    largest = np.argmax(np.abs(pca.components_), axis=1)
    assert np.all(pca.components_[rows, largest] > 0)


def test_flip_signs_makes_largest_entry_positive():
    flipped = flip_signs(np.array([[0.1, -0.9, 0.2], [0.5, 0.3, -0.1]]))
    np.testing.assert_allclose(flipped, [[-0.1, 0.9, -0.2], [0.5, 0.3, -0.1]])


def test_fit_transform_equals_fit_then_transform():
    X = _random_data()
    np.testing.assert_allclose(
        PCA(n_components=2).fit_transform(X), PCA(n_components=2).fit(X).transform(X)
    )


def test_invalid_use_raises():
    X = _random_data(n=20, d=5)
    with pytest.raises(ValueError):
        PCA(n_components=6).fit(X)  # more components than features
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(RuntimeError):
        PCA(n_components=2).transform(X)  # not fitted
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(X).transform(X[:, :4])  # wrong number of features
