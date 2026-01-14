import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions import (
    Normal as TorchNormal,
    MultivariateNormal as TorchMVN,
    Categorical,
    MixtureSameFamily,
)


def set_seed(seed: int = 42):
    """Set RNG seeds for numpy and torch."""
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass


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


def _to_tensor(x, dtype=torch.float64) -> Tensor:
    """Convert input (numpy, list, scalar, tensor) to a torch Tensor on CPU with dtype."""
    if isinstance(x, Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Elementwise log N(x | mean, var) using torch distributions (zuko-compatible backend).
    Accepts numpy arrays or scalars, returns numpy array; supports broadcasting.
    """
    # Clamp variance for stability
    var = np.maximum(var, 1e-12)
    tx = _to_tensor(x)
    tmean = _to_tensor(mean)
    tscale = torch.sqrt(_to_tensor(var))
    dist = TorchNormal(loc=tmean, scale=tscale)
    lp = dist.log_prob(tx)
    return _to_numpy(lp)


def gaussian2_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log pdf for 2D Gaussian using torch.distributions.MultivariateNormal.
    x: (..., 2), mean: (..., 2), cov: (..., 2, 2). Supports broadcasting.
    Returns numpy array.
    """
    tx = _to_tensor(x)
    tmean = _to_tensor(mean)
    tcov = _to_tensor(cov)
    # Ensure positive definiteness with tiny jitter if needed (numerical safety)
    # Only add jitter when covariance is 2x2 matrix (no batch) or batched square mats
    eye = torch.eye(2, dtype=tcov.dtype)
    # Add a very small jitter to the diagonal to avoid singularities
    tcov = tcov + 1e-12 * eye if tcov.shape[-2:] == (2, 2) else tcov
    dist = TorchMVN(loc=tmean, covariance_matrix=tcov)
    lp = dist.log_prob(tx)
    return _to_numpy(lp)


def logsumexp(a: np.ndarray, axis: int = None) -> np.ndarray:
    """Torch-based logsumexp that accepts numpy and returns numpy."""
    ta = _to_tensor(a)
    if axis is None:
        ta = ta.reshape(-1)
        out = torch.logsumexp(ta, dim=0)
    else:
        out = torch.logsumexp(ta, dim=axis)
    return _to_numpy(out)


@dataclass
class OneDNormal:
    mean: float
    var: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_logpdf(x, self.mean, self.var)

    def sample(self, n: int) -> np.ndarray:
        # Use torch Normal for sampling
        dist = TorchNormal(loc=_to_tensor(self.mean), scale=torch.sqrt(_to_tensor(self.var)))
        return _to_numpy(dist.sample((n,)))


@dataclass
class OneDMixture2:
    """Two-component Gaussian mixture in 1D with shared mean for convenience."""
    pi: float  # probability of component 0
    mean: float
    var0: float
    var1: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        # MixtureSameFamily with two Normal components at the same mean
        probs = _to_tensor([self.pi, 1 - self.pi])
        cat = Categorical(probs=probs)
        comp = TorchNormal(
            loc=_to_tensor([self.mean, self.mean]),
            scale=torch.sqrt(_to_tensor([self.var0, self.var1])),
        )
        mix = MixtureSameFamily(cat, comp)
        lx = _to_tensor(x)
        lp = mix.log_prob(lx)
        return _to_numpy(lp)

    def sample(self, n: int) -> np.ndarray:
        probs = _to_tensor([self.pi, 1 - self.pi])
        cat = Categorical(probs=probs)
        comp = TorchNormal(
            loc=_to_tensor([self.mean, self.mean]),
            scale=torch.sqrt(_to_tensor([self.var0, self.var1])),
        )
        mix = MixtureSameFamily(cat, comp)
        return _to_numpy(mix.sample((n,)))


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