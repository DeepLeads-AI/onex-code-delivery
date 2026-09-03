# Vendored from the oneX platform's OOD detector, byte for byte, so the gate here
# scores exactly the way the client's OOD panel does. Do not reformat: the shipped
# manifold_qwen_stg_layer14_v1.npz was written by this file's save().

"""
Distance-to-Manifold (Mahalanobis) when you ONLY have embeddings.

You already have:
  - Z_train: (N_train, D) embedding vectors for the intended-purpose (in-domain) data
  - Z_live:  (N_live,  D) embedding vectors for production/live data

We fit a Gaussian manifold on Z_train:
  mu = mean(Z_train)
  Sigma = cov(Z_train)  (optionally after PCA whitening / dimensionality reduction)
Then score each live vector with Mahalanobis distance:
  d(z) = sqrt( (z-mu)^T Sigma^{-1} (z-mu) )

This detects out-of-manifold usage (likely wrong purpose).

Dependencies:
  pip install numpy
Optional (recommended for stability when D is large):
  pip install scikit-learn
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
from scipy.stats import chi2

# -------------------------
# Core math
# -------------------------

def _cov_shrinkage(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """
    Shrink covariance towards scaled identity for numerical stability:
      cov' = (1 - s)*cov + s*I*avg_var
    """
    d = cov.shape[0]
    avg_var = float(np.trace(cov) / d)
    return (1.0 - shrinkage) * cov + shrinkage * np.eye(d, dtype=cov.dtype) * avg_var


def fit_gaussian_manifold(
    Z_train: np.ndarray,
    *,
    shrinkage: float = 1e-2,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mu and Sigma^{-1} on training embeddings.

    Args:
      Z_train: (N, D) float array
      shrinkage: covariance shrinkage coefficient
      center: whether to center embeddings before covariance

    Returns:
      mu: (D,)
      inv_cov: (D, D)
    """
    if Z_train.ndim != 2:
        raise ValueError("Z_train must be shape (N, D)")
    if Z_train.shape[0] < 2:
        raise ValueError("Need at least 2 training samples to fit covariance.")

    Z_train = np.asarray(Z_train, dtype=np.float64)

    mu = Z_train.mean(axis=0) if center else np.zeros((Z_train.shape[1],), dtype=np.float64)
    X = Z_train - mu if center else Z_train.copy()

    cov = np.cov(X, rowvar=False, bias=False).astype(np.float64)
    cov_reg = _cov_shrinkage(cov, shrinkage=shrinkage).astype(np.float64)

    # Invert; if singular, raise a clearer error.
    try:
        inv_cov = np.linalg.inv(cov_reg).astype(np.float64)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Covariance inversion failed. Increase shrinkage, reduce dimension (PCA), "
            "or use diagonal covariance."
        ) from e

    return mu, inv_cov


def mahalanobis_distance(
    Z: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    *,
    squared: bool = False
) -> np.ndarray:
    """
    Compute Mahalanobis distances for each row of Z.

    Args:
      Z: (N, D) or (D,)
      mu: (D,)
      inv_cov: (D, D)

    Returns:
      d: (N,)
    """
    Z = np.asarray(Z, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)

    Z2 = Z[None, :] if Z.ndim == 1 else Z
    if Z2.shape[1] != mu.shape[0]:
        raise ValueError(f"Dim mismatch: Z has D={Z2.shape[1]} but mu has D={mu.shape[0]}")

    X = Z2 - mu
    d2 = np.einsum("nd,dd,nd->n", X, inv_cov, X).astype(np.float64)
    return d2 if squared else np.sqrt(np.maximum(d2, 0.0))


def pick_threshold_from_training(d_train: np.ndarray, *, quantile: float = 0.99) -> float:
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(np.asarray(d_train, dtype=np.float64), quantile))


# -------------------------
# Optional: PCA reduction (recommended if N_train is not huge)
# -------------------------

@dataclass
class PCAReducer:
    """
    Minimal PCA reducer using SVD in numpy (no sklearn).
    Reduces D -> k, then you fit the manifold in reduced space.

    This is strongly recommended when:
      - D is large (e.g., 768), and/or
      - N_train is not >> D

    Using k in [64, 256] often works well.
    """
    k: int
    mean_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None  # (k, D)

    def fit(self, Z_train: np.ndarray) -> "PCAReducer":
        Z_train = np.asarray(Z_train, dtype=np.float64)
        if Z_train.ndim != 2:
            raise ValueError("Z_train must be shape (N, D)")
        if self.k <= 0 or self.k > Z_train.shape[1]:
            raise ValueError("k must be in [1, D]")

        self.mean_ = Z_train.mean(axis=0)
        X = Z_train - self.mean_

        # SVD: X = U S Vt; principal axes are rows of Vt
        # Vt shape: (D, D) if full_matrices=True; use False for efficiency
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[: self.k, :]  # (k, D)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit() first.")
        Z = np.asarray(Z, dtype=np.float64)
        X = Z - self.mean_
        return (X @ self.components_.T).astype(np.float64)  # (N, k) or (k,)

    def fit_transform(self, Z_train: np.ndarray) -> np.ndarray:
        self.fit(Z_train)
        return self.transform(Z_train)


# -------------------------
# Detector wrapper
# -------------------------

@dataclass
class DistanceToManifoldEmbeddings:
    shrinkage: float = 1e-2
    threshold_quantile: float = 0.99
    use_pca: bool = True
    pca_k: int = 128

    mu: Optional[np.ndarray] = None
    inv_cov: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    pca: Optional[PCAReducer] = None

    def fit(self, Z_train: np.ndarray) -> Dict[str, float]:
        Z_train = np.asarray(Z_train, dtype=np.float64)
        if Z_train.ndim != 2:
            raise ValueError("Z_train must be shape (N, D)")

        Z_fit = Z_train
        if self.use_pca:
            self.pca = PCAReducer(k=min(self.pca_k, Z_train.shape[1])).fit(Z_train)
            Z_fit = self.pca.transform(Z_train)

        self.mu, self.inv_cov = fit_gaussian_manifold(Z_fit, shrinkage=self.shrinkage, center=True)

        d_train = mahalanobis_distance(Z_fit, self.mu, self.inv_cov)
        self.threshold = pick_threshold_from_training(d_train, quantile=self.threshold_quantile)

        return {
            "n_train": float(Z_train.shape[0]),
            "orig_dim": float(Z_train.shape[1]),
            "fit_dim": float(Z_fit.shape[1]),
            "use_pca": float(int(self.use_pca)),
            "pca_k": float(self.pca.k) if self.pca is not None else float("nan"),
            "shrinkage": float(self.shrinkage),
            "threshold_quantile": float(self.threshold_quantile),
            "threshold": float(self.threshold),
            "train_dist_p50": float(np.quantile(d_train, 0.50)),
            "train_dist_p95": float(np.quantile(d_train, 0.95)),
            "train_dist_p99": float(np.quantile(d_train, 0.99)),
        }

    def score(self, Z: np.ndarray) -> np.ndarray:
        if self.mu is None or self.inv_cov is None:
            raise RuntimeError("Call fit() first.")
        Z = np.asarray(Z, dtype=np.float64)
        Zs = self.pca.transform(Z) if self.pca is not None else Z
        return mahalanobis_distance(Zs, self.mu, self.inv_cov)

    def predict(self, Z_live: np.ndarray) -> Dict[str, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Call fit() first.")
        dist = self.score(Z_live)
        is_ood = dist > self.threshold
        df = self.pca.k if self.pca is not None else Z_live.shape[1]
        ood_prob = chi2.cdf(np.square(dist), df=df)
        return {"dist": dist, "is_ood": is_ood, "ood_prob": ood_prob}

    def save(self, path: str) -> None:
        if self.mu is None or self.inv_cov is None or self.threshold is None:
            raise RuntimeError("Nothing to save; call fit() first.")

        payload = {
            "mu": self.mu.astype(np.float64),
            "inv_cov": self.inv_cov.astype(np.float64),
            "threshold": np.array([self.threshold], dtype=np.float64),
            "shrinkage": np.array([self.shrinkage], dtype=np.float64),
            "threshold_quantile": np.array([self.threshold_quantile], dtype=np.float64),
            "use_pca": np.array([int(self.use_pca)], dtype=np.int32),
        }

        if self.pca is not None:
            payload.update({
                "pca_k": np.array([self.pca.k], dtype=np.int32),
                "pca_mean": self.pca.mean_.astype(np.float64),
                "pca_components": self.pca.components_.astype(np.float64),
            })

        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "DistanceToManifoldEmbeddings":
        data = np.load(path, allow_pickle=True)

        det = cls(
            shrinkage=float(data["shrinkage"][0]),
            threshold_quantile=float(data["threshold_quantile"][0]),
            use_pca=bool(int(data["use_pca"][0])),
            pca_k=int(data["pca_k"][0]) if "pca_k" in data else 0,
        )

        det.mu = data["mu"].astype(np.float64)
        det.inv_cov = data["inv_cov"].astype(np.float64)
        det.threshold = float(data["threshold"][0])

        if "pca_components" in data and "pca_mean" in data:
            det.pca = PCAReducer(k=int(data["pca_k"][0]))
            det.pca.mean_ = data["pca_mean"].astype(np.float64)
            det.pca.components_ = data["pca_components"].astype(np.float64)

        return det
