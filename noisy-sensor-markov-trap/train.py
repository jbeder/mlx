#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from zuko.distributions import MultivariateNormal as ZMVN  # type: ignore

# Use zuko distributions for Gaussians as requested
from zuko.distributions import Normal as ZNormal  # type: ignore


class SpeedDataset(Dataset):
    """Torch dataset returning (source, upstream, downstream).

    - source: LongTensor [N]
    - upstream: FloatTensor [N]
    - downstream: FloatTensor [N]
    """

    def __init__(self, df: pd.DataFrame):
        df = df[["id", "source", "upstream_speed", "downstream_speed"]].copy()
        df["source"] = df["source"].astype(int)
        # Store as tensors
        self.src = torch.as_tensor(df["source"].to_numpy().astype(np.int64))
        self.up = torch.as_tensor(df["upstream_speed"].to_numpy(), dtype=torch.float32)
        self.down = torch.as_tensor(df["downstream_speed"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore
        return int(self.src.numel())

    def __getitem__(self, idx: int):  # type: ignore
        return int(self.src[idx].item()), float(self.up[idx].item()), float(self.down[idx].item())


def load_data(path: str) -> SpeedDataset:
    """Read parquet and return a torch Dataset instead of a DataFrame."""
    df = pd.read_parquet(path)
    return SpeedDataset(df)


############################################
# GMM (2D Gaussian per source)
############################################


def fit_gmm(dataset: SpeedDataset, epochs: int = 300, lr: float = 0.05) -> Dict:
    """Train a per-source 2D Gaussian with MLE using zuko MultivariateNormal.

    We maintain learnable per-source mean (2,) and scale_tril (2x2) parameters and
    optimize average NLL over the dataset.
    """
    src = dataset.src
    up = dataset.up
    down = dataset.down
    x = torch.stack([up, down], dim=1)  # (N, 2)
    S = int(src.max().item()) + 1

    # Initialize from empirical stats
    with torch.no_grad():
        mean0 = torch.zeros(S, 2, dtype=torch.float32)
        L0 = torch.zeros(S, 2, 2, dtype=torch.float32)
        for s in range(S):
            idx = src == s
            xs = x[idx]
            if xs.numel() == 0:
                # Safe default
                mean0[s] = 0.0
                L0[s] = torch.eye(2)
            else:
                m = xs.mean(dim=0)
                # unbiased covariance
                xm = xs - m
                cov = (xm.t() @ xm) / max(xs.shape[0] - 1, 1)
                cov = cov + 1e-4 * torch.eye(2)
                L = torch.linalg.cholesky(cov)
                mean0[s] = m
                L0[s] = L

    mean = nn.Parameter(mean0.clone())
    raw_L = nn.Parameter(L0.clone())  # We'll enforce lower-tri with positive diag using softplus

    opt = torch.optim.Adam([mean, raw_L], lr=lr)
    sp = torch.nn.Softplus()

    for _ in range(epochs):
        # Build batched scale_tril with positive diagonal
        L = torch.tril(raw_L)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = sp(diag) + 1e-6
        L = L.clone()
        for i in range(S):
            L[i, 0, 0] = diag[i, 0]
            L[i, 1, 1] = diag[i, 1]

        m_b = mean[src]
        L_b = L[src]
        dist = ZMVN(m_b, scale_tril=L_b)
        loss = -dist.log_prob(x).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Export learned parameters to JSON-friendly dict
    with torch.no_grad():
        L = torch.tril(raw_L)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = sp(diag) + 1e-6
        for i in range(S):
            L[i, 0, 0] = diag[i, 0]
            L[i, 1, 1] = diag[i, 1]
        cov = L @ L.transpose(-1, -2)
        params: Dict[int, Dict] = {}
        for s in range(S):
            params[s] = {
                "mean": mean[s].detach().cpu().numpy().astype(float).tolist(),
                "cov": cov[s].detach().cpu().numpy().astype(float).tolist(),
            }

    return {"config": "gmm", "params": params}


############################################
# Markov (upstream ~ N; downstream | upstream ~ N(a*up+b, var))
############################################


def fit_markov(dataset: SpeedDataset, epochs: int = 300, lr: float = 0.05) -> Dict:
    """Train the Markov model (up ~ N; down|up ~ N(a*up+b, var)) via MLE using zuko Normals."""
    src = dataset.src
    up = dataset.up
    down = dataset.down
    S = int(src.max().item()) + 1

    # Initialize with simple statistics / least squares
    with torch.no_grad():
        mu0 = torch.zeros(S, dtype=torch.float32)
        log_std_u0 = torch.zeros(S, dtype=torch.float32)
        a0 = torch.zeros(S, dtype=torch.float32)
        b0 = torch.zeros(S, dtype=torch.float32)
        log_std_d0 = torch.zeros(S, dtype=torch.float32)
        for s in range(S):
            idx = src == s
            u = up[idx]
            d = down[idx]
            if u.numel() == 0:
                continue
            mu = u.mean()
            var_u = u.var(unbiased=True) + 1e-6
            X = torch.stack([u, torch.ones_like(u)], dim=1)
            sol = torch.linalg.lstsq(X, d).solution
            a = sol[0]
            b = sol[1]
            resid = d - (a * u + b)
            var_d = resid.var(unbiased=True) + 1e-6
            mu0[s] = mu
            log_std_u0[s] = 0.5 * torch.log(var_u)
            a0[s] = a
            b0[s] = b
            log_std_d0[s] = 0.5 * torch.log(var_d)

    mu_u = nn.Parameter(mu0)
    log_std_u = nn.Parameter(log_std_u0)
    a = nn.Parameter(a0)
    b = nn.Parameter(b0)
    log_std_d = nn.Parameter(log_std_d0)
    opt = torch.optim.Adam([mu_u, log_std_u, a, b, log_std_d], lr=lr)

    for _ in range(epochs):
        std_u = torch.exp(log_std_u)
        std_d = torch.exp(log_std_d)
        mu_u_b = mu_u[src]
        std_u_b = std_u[src]
        a_b = a[src]
        b_b = b[src]
        std_d_b = std_d[src]

        up_dist = ZNormal(mu_u_b, std_u_b)
        down_mu = a_b * up + b_b
        down_dist = ZNormal(down_mu, std_d_b)
        loss = -(up_dist.log_prob(up) + down_dist.log_prob(down)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Export
    with torch.no_grad():
        std_u = torch.exp(log_std_u)
        std_d = torch.exp(log_std_d)
        params: Dict[int, Dict] = {}
        for s in range(S):
            params[s] = {
                "up_mean": float(mu_u[s].item()),
                "up_var": float((std_u[s] ** 2).item()),
                "a": float(a[s].item()),
                "b": float(b[s].item()),
                "down_var": float((std_d[s] ** 2).item()),
            }

    return {"config": "markov", "params": params}


############################################
# Latent (u prior, v|u process, sensor mixture)
############################################


def _prior_stats(mu_u: float, var_u: float, a: float, b: float, var_v: float):
    """Prior stats as torch tensors (float64)."""
    m = torch.as_tensor([mu_u, a * mu_u + b], dtype=torch.float64)
    P = torch.as_tensor(
        [
            [var_u, a * var_u],
            [a * var_u, a * a * var_u + var_v],
        ],
        dtype=torch.float64,
    )
    return m, P


def _posterior_uv_given_xy(m: torch.Tensor, P: torch.Tensor, x: float, y: float, r_eff: float):
    """Posterior for linear-Gaussian model with identity observation and noise diag(r_eff, r_eff).
    Returns torch tensors (mu, Sigma).
    """
    R = torch.diag(torch.as_tensor([r_eff, r_eff], dtype=torch.float64))
    Pinv = torch.linalg.inv(P)
    Rinv = torch.linalg.inv(R)
    Sigma = torch.linalg.inv(Pinv + Rinv)
    yv = torch.as_tensor([x, y], dtype=torch.float64)
    mu = Sigma @ (Pinv @ m + Rinv @ yv)
    return mu, Sigma


def fit_latent(dataset: SpeedDataset, epochs: int = 200, lr: float = 0.05) -> Dict:
    """Train the latent model via direct MLE of the marginal mixture using zuko MVNs.

    Marginal for (x, y) given source s is a mixture of two Gaussians with means
    m_s = [mu_u, a*mu_u + b] and covariances P_s + sigma_k^2 I, where
    P_s = [[var_u, a var_u], [a var_u, a^2 var_u + var_v]].
    """
    src = dataset.src
    up = dataset.up
    down = dataset.down
    xy = torch.stack([up, down], dim=1)
    S = int(src.max().item()) + 1

    # Initialize from simple heuristics
    with torch.no_grad():
        mu0 = torch.zeros(S, dtype=torch.float32)
        log_var_u0 = torch.zeros(S, dtype=torch.float32)
        a0 = torch.zeros(S, dtype=torch.float32)
        b0 = torch.zeros(S, dtype=torch.float32)
        log_var_v0 = torch.zeros(S, dtype=torch.float32)
        raw_pi0 = torch.zeros(S, dtype=torch.float32)
        for s in range(S):
            idx = src == s
            u = up[idx]
            d = down[idx]
            if u.numel() == 0:
                continue
            mu = u.mean()
            var_u = u.var(unbiased=True) + 1e-3
            X = torch.stack([u, torch.ones_like(u)], dim=1)
            sol = torch.linalg.lstsq(X, d).solution
            aa = sol[0]
            bb = sol[1]
            resid = d - (aa * u + bb)
            var_v = resid.var(unbiased=True) + 1e-3
            mu0[s] = mu
            log_var_u0[s] = torch.log(var_u)
            a0[s] = aa
            b0[s] = bb
            log_var_v0[s] = torch.log(var_v)
            raw_pi0[s] = 0.0  # sigmoid -> 0.5

    mu_u = nn.Parameter(mu0)
    log_var_u = nn.Parameter(log_var_u0)
    a = nn.Parameter(a0)
    b = nn.Parameter(b0)
    log_var_v = nn.Parameter(log_var_v0)
    raw_pi = nn.Parameter(raw_pi0)
    raw_sigma0 = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))  # ~0.3 after softplus
    raw_delta_sigma = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))  # positive increment

    opt = torch.optim.Adam([mu_u, log_var_u, a, b, log_var_v, raw_pi, raw_sigma0, raw_delta_sigma], lr=lr)
    sp = torch.nn.Softplus()

    I = torch.eye(2, dtype=torch.float32)
    for _ in range(epochs):
        # Parameters with constraints
        var_u = torch.exp(log_var_u).clamp_min(1e-6)
        var_v = torch.exp(log_var_v).clamp_min(1e-6)
        pi = torch.sigmoid(raw_pi)  # (S,)
        sigma0 = sp(raw_sigma0) + 1e-6
        sigma1 = sigma0 + sp(raw_delta_sigma)  # enforce sigma1 >= sigma0

        # Build per-source prior mean and covariance P
        m_u = mu_u
        m_v = a * mu_u + b
        m = torch.stack([m_u, m_v], dim=1)  # (S, 2)
        # P = [[var_u, a var_u], [a var_u, a^2 var_u + var_v]]
        P = torch.zeros(S, 2, 2, dtype=torch.float32)
        P[:, 0, 0] = var_u
        P[:, 0, 1] = a * var_u
        P[:, 1, 0] = a * var_u
        P[:, 1, 1] = a * a * var_u + var_v

        m_b = m[src]
        P_b = P[src]
        pi_b = pi[src]

        cov0 = P_b + (sigma0**2) * I
        cov1 = P_b + (sigma1**2) * I

        dist0 = ZMVN(m_b, covariance_matrix=cov0)
        dist1 = ZMVN(m_b, covariance_matrix=cov1)
        l0 = torch.log(pi_b + 1e-12) + dist0.log_prob(xy)
        l1 = torch.log(1.0 - pi_b + 1e-12) + dist1.log_prob(xy)
        ll = torch.logsumexp(torch.stack([l0, l1], dim=1), dim=1)  # (N,)
        loss = -ll.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        var_u = torch.exp(log_var_u).clamp_min(1e-6)
        var_v = torch.exp(log_var_v).clamp_min(1e-6)
        pi = torch.sigmoid(raw_pi)
        sigma0 = float((sp(raw_sigma0) + 1e-6).item())
        sigma1 = float((sp(raw_sigma0) + sp(raw_delta_sigma) + 1e-6).item())
        params: Dict[int, Dict] = {}
        for s in range(S):
            params[s] = {
                "mu_u": float(mu_u[s].item()),
                "var_u": float(var_u[s].item()),
                "a": float(a[s].item()),
                "b": float(b[s].item()),
                "var_v": float(var_v[s].item()),
                "pi": float(pi[s].item()),
            }

    return {
        "config": "latent",
        "params": params,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "iters": epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["gmm", "markov", "latent"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = load_data(args.data)

    os.makedirs(args.out, exist_ok=True)

    if args.config == "gmm":
        model = fit_gmm(dataset)
    elif args.config == "markov":
        model = fit_markov(dataset)
    else:
        model = fit_latent(dataset)

    with open(os.path.join(args.out, "model.json"), "w") as f:
        json.dump(model, f, indent=2)

    print(f"Saved model to {os.path.join(args.out, 'model.json')}")


if __name__ == "__main__":
    main()
