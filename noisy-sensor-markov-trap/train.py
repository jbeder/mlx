#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .utils import ensure_dir, gaussian2_logpdf, gaussian_logpdf, logsumexp, save_json  # type: ignore
except Exception:
    import sys
    import os as _os
    sys.path.append(_os.path.dirname(__file__))
    from utils import ensure_dir, gaussian2_logpdf, gaussian_logpdf, logsumexp, save_json  # type: ignore


def read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e1:
        # Fallback to duckdb python if available
        try:
            import duckdb  # type: ignore

            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM read_parquet('{path}')").df()
            con.close()
            return df
        except Exception as e2:
            raise RuntimeError(
                f"Failed reading parquet {path}. Install pyarrow/fastparquet or duckdb. Errors: {e1} / {e2}"
            )


############################################
# GMM (2D Gaussian per source)
############################################


def fit_gmm(df: pd.DataFrame) -> Dict:
    params = {}
    for s, g in df.groupby("source"):
        x = g[["upstream_speed", "downstream_speed"]].to_numpy()
        mean = x.mean(axis=0)
        cov = np.cov(x.T, bias=False)
        # Regularize covariance for stability
        cov = cov + 1e-6 * np.eye(2)
        params[int(s)] = {
            "mean": mean.tolist(),
            "cov": cov.tolist(),
        }
    return {
        "config": "gmm",
        "params": params,
    }


############################################
# Markov (upstream ~ N; downstream | upstream ~ N(a*up+b, var))
############################################


def fit_markov(df: pd.DataFrame) -> Dict:
    params = {}
    for s, g in df.groupby("source"):
        up = g["upstream_speed"].to_numpy()
        down = g["downstream_speed"].to_numpy()
        mu = float(up.mean())
        var = float(np.var(up, ddof=1))
        X = np.stack([up, np.ones_like(up)], axis=1)
        # Least squares for down ~ a*up + b
        ab, _, _, _ = np.linalg.lstsq(X, down, rcond=None)
        a = float(ab[0])
        b = float(ab[1])
        resid = down - (a * up + b)
        var_d = float(np.var(resid, ddof=1) + 1e-6)
        params[int(s)] = {
            "up_mean": mu,
            "up_var": max(var, 1e-6),
            "a": a,
            "b": b,
            "down_var": var_d,
        }
    return {
        "config": "markov",
        "params": params,
    }


############################################
# Latent (u prior, v|u process, sensor mixture)
############################################


def _prior_stats(mu_u: float, var_u: float, a: float, b: float, var_v: float):
    m = np.array([mu_u, a * mu_u + b])
    P = np.array(
        [
            [var_u, a * var_u],
            [a * var_u, a * a * var_u + var_v],
        ]
    )
    return m, P


def _posterior_uv_given_xy(m: np.ndarray, P: np.ndarray, x: float, y: float, r_eff: float):
    # Observation is identity with noise R = diag(r_eff, r_eff)
    R = np.diag([r_eff, r_eff])
    # Posterior covariance: (P^{-1} + R^{-1})^{-1}
    Pinv = np.linalg.inv(P)
    Rinv = np.linalg.inv(R)
    Sigma = np.linalg.inv(Pinv + Rinv)
    # Posterior mean: Sigma * (P^{-1} m + R^{-1} y)
    yv = np.array([x, y])
    mu = Sigma @ (Pinv @ m + Rinv @ yv)
    return mu, Sigma


def fit_latent(df: pd.DataFrame, iters: int = 10) -> Dict:
    S = int(df["source"].max()) + 1
    # Initialize per-source parameters from observed heuristics
    params = {}
    for s, g in df.groupby("source"):
        up = g["upstream_speed"].to_numpy()
        down = g["downstream_speed"].to_numpy()
        mu_u = float(up.mean())
        var_u = float(np.var(up, ddof=1))
        X = np.stack([up, np.ones_like(up)], axis=1)
        ab, _, _, _ = np.linalg.lstsq(X, down, rcond=None)
        a = float(ab[0])
        b = float(ab[1])
        resid = down - (a * up + b)
        var_v = max(float(np.var(resid, ddof=1) - 0.1), 1e-3)
        params[int(s)] = {
            "mu_u": mu_u,
            "var_u": max(var_u, 1e-3),
            "a": a,
            "b": b,
            "var_v": var_v,
            "pi": 0.5,
        }
    # Global sensor variances (initialized near the spec)
    sigma0 = 0.3
    sigma1 = 1.2

    # Pre-cache arrays
    x_all = df["upstream_speed"].to_numpy()
    y_all = df["downstream_speed"].to_numpy()
    s_all = df["source"].to_numpy().astype(int)
    N = len(df)

    # Storage for responsibilities and posteriors
    r0 = np.full(N, 0.5)
    r1 = np.full(N, 0.5)
    Eu = np.zeros(N)
    Ev = np.zeros(N)
    Eu2 = np.zeros(N)
    Ev2 = np.zeros(N)
    Euv = np.zeros(N)

    for it in range(iters):
        # E-step: responsibilities and posterior moments
        logw0 = np.zeros(N)
        logw1 = np.zeros(N)
        for s in range(S):
            idx = np.where(s_all == s)[0]
            if len(idx) == 0:
                continue
            p = params[s]
            m, P = _prior_stats(p["mu_u"], p["var_u"], p["a"], p["b"], p["var_v"])
            # Likelihood for z=0 and z=1 marginals
            r0_s = sigma0 ** 2
            r1_s = sigma1 ** 2
            cov0 = P + np.diag([r0_s, r0_s])
            cov1 = P + np.diag([r1_s, r1_s])
            xy = np.stack([x_all[idx], y_all[idx]], axis=1)
            l0 = gaussian2_logpdf(xy, m, cov0) + np.log(p["pi"] + 1e-12)
            l1 = gaussian2_logpdf(xy, m, cov1) + np.log(1 - p["pi"] + 1e-12)
            norm = logsumexp(np.stack([l0, l1], axis=1), axis=1)
            r0[idx] = np.exp(l0 - norm)
            r1[idx] = 1.0 - r0[idx]

            # Effective measurement variance per row (mixing responsibilities)
            r_eff = r0[idx] * r0_s + r1[idx] * r1_s
            # Posterior moments
            for j, n in enumerate(idx):
                mu_post, Sigma_post = _posterior_uv_given_xy(m, P, x_all[n], y_all[n], r_eff[j])
                Eu[n] = mu_post[0]
                Ev[n] = mu_post[1]
                Eu2[n] = Sigma_post[0, 0] + mu_post[0] ** 2
                Ev2[n] = Sigma_post[1, 1] + mu_post[1] ** 2
                Euv[n] = Sigma_post[0, 1] + mu_post[0] * mu_post[1]

        # M-step: update per-source process params
        for s in range(S):
            idx = np.where(s_all == s)[0]
            if len(idx) == 0:
                continue
            mu_u = float(Eu[idx].mean())
            # var_u as E[(u - mu)^2]
            var_u = float((Eu2[idx].mean() - 2 * mu_u * Eu[idx].mean() + mu_u ** 2))
            # Linear regression v ~ a u + b using expected sufficient stats
            Eu_s = Eu[idx].mean()
            Ev_s = Ev[idx].mean()
            cov_uu = float(Eu2[idx].mean() - Eu_s ** 2)
            cov_uv = float(Euv[idx].mean() - Eu_s * Ev_s)
            a = 0.0
            if cov_uu > 1e-12:
                a = cov_uv / cov_uu
            b = Ev_s - a * Eu_s
            # var_v as E[(v - a u - b)^2]
            Ev2m = Ev2[idx]
            term = Ev2m - 2 * a * Euv[idx] - 2 * b * Ev[idx] + (a ** 2) * Eu2[idx] + 2 * a * b * Eu[idx] + b ** 2
            var_v = float(max(term.mean(), 1e-6))
            params[s].update({
                "mu_u": mu_u,
                "var_u": max(var_u, 1e-6),
                "a": float(a),
                "b": float(b),
                "var_v": var_v,
                "pi": float(r0[idx].mean()),
            })

        # Update global sensor variances using expected residuals
        # E[(x - u)^2] = (x - E[u])^2 + Var(u|data), similarly for y-v
        # Approximate Var(u|data) and Var(v|data) from Eu2 - Eu^2 etc.
        var_u_post = np.maximum(Eu2 - Eu ** 2, 0.0)
        var_v_post = np.maximum(Ev2 - Ev ** 2, 0.0)
        exu2 = (x_all - Eu) ** 2 + var_u_post
        evv2 = (y_all - Ev) ** 2 + var_v_post
        num0 = np.sum(r0 * (exu2 + evv2) * 0.5)
        den0 = np.sum(r0) + 1e-12
        num1 = np.sum(r1 * (exu2 + evv2) * 0.5)
        den1 = np.sum(r1) + 1e-12
        sigma0 = float(np.sqrt(max(num0 / den0, 1e-6)))
        sigma1 = float(np.sqrt(max(num1 / den1, 1e-6)))
        # Enforce ordering sigma0 <= sigma1 for identifiability
        if sigma0 > sigma1:
            sigma0, sigma1 = sigma1, sigma0

    return {
        "config": "latent",
        "params": params,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["gmm", "markov", "latent"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = read_parquet(args.data)
    # Ensure correct dtypes
    df = df[["id", "source", "upstream_speed", "downstream_speed"]].copy()
    df["source"] = df["source"].astype(int)

    ensure_dir(args.out)

    if args.config == "gmm":
        model = fit_gmm(df)
    elif args.config == "markov":
        model = fit_markov(df)
    else:
        model = fit_latent(df)

    save_json(os.path.join(args.out, "model.json"), model)
    print(f"Saved model to {os.path.join(args.out, 'model.json')}")


if __name__ == "__main__":
    main()