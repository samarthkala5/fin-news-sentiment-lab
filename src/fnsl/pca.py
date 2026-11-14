"""Principal component analysis (PCA) from scratch with NumPy's SVD.

PCA finds the directions along which the data varies most, so that a few numbers per
row (here 2) can stand in for many (here 384 embedding dimensions). Subtract the mean
of every feature, take the singular value decomposition (SVD) of the centred data, and
the right singular vectors are those directions, ordered by how much variance each
explains. Projecting onto the first k of them gives the k-D coordinates used for plots.
tests/test_pca.py checks this class against sklearn.decomposition.PCA.
"""

from __future__ import annotations

import numpy as np


def flip_signs(components: np.ndarray) -> np.ndarray:
    """Flip rows so that the entry with the largest absolute value in each row is positive.

    SVD defines every singular vector only up to its sign: if (u_i, v_i) is a solution,
    (-u_i, -v_i) is one too, and which one LAPACK returns can change between machines
    or library versions. Without a fixed rule the same data could give a mirrored PCA
    plot on another computer. scikit-learn 1.9 uses this same rule for its PCA.
    """
    rows = np.arange(components.shape[0])
    largest = np.argmax(np.abs(components), axis=1)  # column of the largest |value| per row
    signs = np.sign(components[rows, largest])
    signs[signs == 0] = 1.0  # only happens for an all-zero row; leave it unchanged
    return components * signs[:, None]


def _as_2d_float(X) -> np.ndarray:
    """Convert input to a 2-D float64 array (float64 keeps the SVD accurate)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array (n_samples, n_features), got shape {X.shape}")
    return X


class PCA:
    """PCA via the SVD of the centred data matrix.

    How it works, for X with n rows (samples) and d columns (features):
      1. Centre:  X_c = X - mean_  (mean_ = average of every column).
      2. SVD:     X_c = U S V^T, with r = min(n, d), U (n x r) and V (d x r) having
                  orthonormal columns and S = the r singular values, largest first.
      3. The rows of V^T are the principal directions (unit length, at right angles).
         They are also the eigenvectors of the covariance matrix
         C = X_c^T X_c / (n - 1) = V (S^2 / (n - 1)) V^T, but the SVD finds them without
         forming C, which is more accurate.
      4. Projection onto the first k directions: Z = X_c V_k, which equals U_k S_k
         because V^T V = I.
      5. Column i of Z has mean 0 and squared length S_i^2, so the variance of the data
         along direction i is S_i^2 / (n - 1).

    Attributes (set by fit):
        mean_: (d,) column means used for centring.
        components_: (k, d) principal directions, one per row, by decreasing variance.
        singular_values_: (k,) the k largest singular values S_i.
        explained_variance_: (k,) variance along each direction, S_i^2 / (n - 1).
        explained_variance_ratio_: (k,) that variance divided by the total variance.
    """

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X) -> PCA:
        """Learn the mean and the principal directions of X (n_samples x n_features)."""
        X = _as_2d_float(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples")
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} is larger than "
                f"min(n_samples, n_features)={min(n_samples, n_features)}"
            )

        self.mean_ = X.mean(axis=0)
        X_centred = X - self.mean_
        # full_matrices=False is the "thin" SVD: U is n x r instead of n x n, which is
        # all PCA needs and much cheaper when n is large. U itself is not needed.
        _, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
        Vt = flip_signs(Vt)

        variance = S**2 / (n_samples - 1)  # variance along every one of the r directions
        total_variance = variance.sum()  # = sum of the per-column variances of X
        if total_variance == 0:
            raise ValueError("all samples are identical, so there is no variance to explain")

        k = self.n_components
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / total_variance
        return self

    def transform(self, X) -> np.ndarray:
        """Coordinates of X along the principal directions: Z = (X - mean_) V_k, (n x k)."""
        self._check_fitted()
        X = _as_2d_float(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, PCA was fit on {self.mean_.shape[0]}")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        """Fit on X, then return its k-D coordinates."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        """Map k-D coordinates back to the original space: X_hat = Z V_k^T + mean_.

        With k < d every row lands on the k-dimensional plane through mean_ spanned by
        the components; for the data PCA was fit on, that is the best rank-k
        least-squares approximation of the centred data (plus the mean). With every
        component kept it gives back X up to rounding error.
        """
        self._check_fitted()
        Z = _as_2d_float(Z)
        if Z.shape[1] != self.components_.shape[0]:
            raise ValueError(f"Z has {Z.shape[1]} columns, expected {self.components_.shape[0]}")
        return Z @ self.components_ + self.mean_

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("this PCA is not fitted yet; call fit() first")
