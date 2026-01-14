import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def set_seed(seed: int = 42):
    """Set numpy's RNG seed."""
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def clamp_var(x: float, eps: float = 1e-6) -> float:
    return float(max(x, eps))


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Elementwise log N(x | mean, var). All arrays broadcastable."""
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def gaussian2_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log pdf for 2D Gaussian. x: (...,2), mean: (...,2), cov: (...,2,2).
    Supports broadcasting across leading dimensions.
    """
    # Compute inverse and logdet for 2x2 covariances efficiently
    a = cov[..., 0, 0]
    b = cov[..., 0, 1]
    c = cov[..., 1, 0]
    d = cov[..., 1, 1]
    det = a * d - b * c
    det = np.maximum(det, 1e-18)
    inv = np.empty_like(cov)
    inv[..., 0, 0] = d / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -c / det
    inv[..., 1, 1] = a / det
    diff = x - mean
    # quadratic form
    q0 = inv[..., 0, 0] * diff[..., 0] + inv[..., 0, 1] * diff[..., 1]
    q1 = inv[..., 1, 0] * diff[..., 0] + inv[..., 1, 1] * diff[..., 1]
    quad = diff[..., 0] * q0 + diff[..., 1] * q1
    logdet = np.log(det)
    return -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)


def logsumexp(a: np.ndarray, axis: int = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    if axis is None:
        return s.squeeze()
    return np.squeeze(s, axis=axis)


@dataclass
class OneDNormal:
    mean: float
    var: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_logpdf(x, self.mean, self.var)

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mean, math.sqrt(self.var), size=(n,))


@dataclass
class OneDMixture2:
    """Two-component Gaussian mixture in 1D with shared mean for convenience."""
    pi: float  # probability of component 0
    mean: float
    var0: float
    var1: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        l0 = np.log(self.pi + 1e-12) + gaussian_logpdf(x, self.mean, self.var0)
        l1 = np.log(1 - self.pi + 1e-12) + gaussian_logpdf(x, self.mean, self.var1)
        return logsumexp(np.stack([l0, l1], axis=-1), axis=-1)

    def sample(self, n: int) -> np.ndarray:
        z = (np.random.rand(n) < self.pi).astype(np.int32)
        std0 = math.sqrt(self.var0)
        std1 = math.sqrt(self.var1)
        noise = np.where(z == 1, np.random.normal(0, std0, size=n), 0.0)
        noise1 = np.where(z == 0, np.random.normal(0, std1, size=n), 0.0)
        return self.mean + noise + noise1


def crps_mc_1d(dist, y: np.ndarray, S: int = 64) -> np.ndarray:
    """Monte Carlo CRPS for 1D predictive dist vs scalar y.
    dist must provide sample(n) that returns (n,) array.
    y: (N,) values. Returns (N,) CRPS estimates.
    """
    N = y.shape[0]
    # Sample S for each row independently
    # We'll sample N*S and reshape for vectorized computation
    xs = np.array([dist.sample(S) for _ in range(N)])  # (N, S)
    # E|X - y|
    e1 = np.mean(np.abs(xs - y[:, None]), axis=1)
    # 0.5 E|X - X'|
    # Approximate with V-statistic over pairs within samples
    # Compute pairwise differences along axis 1
    # Efficient approx: sample another set X'
    xs2 = np.array([dist.sample(S) for _ in range(N)])
    e2 = 0.5 * np.mean(np.abs(xs - xs2), axis=1)
    return e1 - e2
