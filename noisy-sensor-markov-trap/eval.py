#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import (
    Normal as TorchNormal,
    MultivariateNormal as TorchMVN,
    Categorical,
    MixtureSameFamily,
)

try:
    from .utils import (
        ensure_dir,
        gaussian2_logpdf,
        gaussian_logpdf,
        logsumexp,
        save_json,
        OneDMixture2,
    )  # type: ignore
except Exception:
    import sys
    import os as _os
    sys.path.append(_os.path.dirname(__file__))
    from utils import (
        ensure_dir,
        gaussian2_logpdf,
        gaussian_logpdf,
        logsumexp,
        save_json,
        OneDMixture2,
    )  # type: ignore


def read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e1:
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


def load_model(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, "model.json"), "r") as f:
        return json.load(f)


def nll_gaussian(x, mean, var):
    # Use torch.distributions via utils.gaussian_logpdf
    return -gaussian_logpdf(np.array(x), np.array(mean), np.array(var))


def nll_mixture2(x, mean, var0, var1, pi):
    # Use OneDMixture2 implemented on torch distributions
    mix = OneDMixture2(pi=float(pi), mean=float(mean), var0=float(var0), var1=float(var1))
    return -mix.logpdf(np.array(x))


def crps_mc_perrow(sample_fn_list, y: np.ndarray, S: int = 64) -> np.ndarray:
    N = y.shape[0]
    crps = np.zeros(N)
    for i in range(N):
        xs = sample_fn_list[i](S)
        xs2 = sample_fn_list[i](S)
        e1 = np.mean(np.abs(xs - y[i]))
        e2 = 0.5 * np.mean(np.abs(xs - xs2))
        crps[i] = e1 - e2
    return crps


def evaluate(model: Dict, df: pd.DataFrame, crps_S: int = 64, energy_M: int = 2000) -> Dict:
    cfg = model["config"]
    df = df[["id", "source", "upstream_speed", "downstream_speed"]].copy()
    src = df["source"].to_numpy().astype(int)
    up = df["upstream_speed"].to_numpy()
    down = df["downstream_speed"].to_numpy()
    N = len(df)

    # Per-speed NLL and CRPS
    nll_up = np.zeros(N)
    nll_down = np.zeros(N)
    up_samplers = []
    down_samplers = []

    if cfg == "gmm":
        params = model["params"]
        for i in range(N):
            p = params[str(src[i])] if str(src[i]) in params else params[int(src[i])]
            mean = np.array(p["mean"])  # (2,)
            cov = np.array(p["cov"])   # (2,2)
            # Marginals
            mu_u, var_u = mean[0], cov[0, 0]
            mu_d, var_d = mean[1], cov[1, 1]
            nll_up[i] = nll_gaussian(up[i], mu_u, var_u)
            nll_down[i] = nll_gaussian(down[i], mu_d, var_d)

            def make_sampler(mu, var):
                def samp(S):
                    dist = TorchNormal(torch.as_tensor(mu, dtype=torch.float64), torch.sqrt(torch.as_tensor(var, dtype=torch.float64)))
                    return dist.sample((S,)).numpy()
                return samp

            up_samplers.append(make_sampler(mu_u, var_u))
            down_samplers.append(make_sampler(mu_d, var_d))

    elif cfg == "markov":
        params = model["params"]
        for i in range(N):
            p = params[str(src[i])] if str(src[i]) in params else params[int(src[i])]
            mu_u = p["up_mean"]
            var_u = p["up_var"]
            a = p["a"]
            b = p["b"]
            var_d = p["down_var"]
            nll_up[i] = nll_gaussian(up[i], mu_u, var_u)
            mu_d_obs = a * up[i] + b
            nll_down[i] = nll_gaussian(down[i], mu_d_obs, var_d)

            up_samplers.append(lambda S, mu=mu_u, var=var_u: TorchNormal(torch.as_tensor(mu, dtype=torch.float64), torch.sqrt(torch.as_tensor(var, dtype=torch.float64))).sample((S,)).numpy())
            down_samplers.append(lambda S, mu=mu_d_obs, var=var_d: TorchNormal(torch.as_tensor(mu, dtype=torch.float64), torch.sqrt(torch.as_tensor(var, dtype=torch.float64))).sample((S,)).numpy())

    elif cfg == "latent":
        params = model["params"]
        sigma0 = model["sigma0"]
        sigma1 = model["sigma1"]
        var0 = sigma0 ** 2
        var1 = sigma1 ** 2
        for i in range(N):
            p = params[str(src[i])] if str(src[i]) in params else params[int(src[i])]
            mu_u = p["mu_u"]
            var_u = p["var_u"]
            a = p["a"]
            b = p["b"]
            var_v = p["var_v"]
            pi = p["pi"]
            # Upstream predictive: mixture of two Normals centered at mu_u with variances var_u + var_k
            nll_up[i] = nll_mixture2(up[i], mu_u, var_u + var0, var_u + var1, pi)
            # Downstream predictive: y ~ mixture N(a*mu_u + b, a^2 var_u + var_v + var_k)
            mu_d = a * mu_u + b
            base_var = a * a * var_u + var_v
            nll_down[i] = nll_mixture2(down[i], mu_d, base_var + var0, base_var + var1, pi)

            def make_mix_sampler(mu, v0, v1, pi):
                def samp(S):
                    cat = Categorical(probs=torch.as_tensor([pi, 1 - pi], dtype=torch.float64))
                    comp = TorchNormal(
                        loc=torch.as_tensor([mu, mu], dtype=torch.float64),
                        scale=torch.sqrt(torch.as_tensor([v0, v1], dtype=torch.float64)),
                    )
                    mix = MixtureSameFamily(cat, comp)
                    return mix.sample((S,)).numpy()
                return samp

            up_samplers.append(make_mix_sampler(mu_u, var_u + var0, var_u + var1, pi))
            down_samplers.append(make_mix_sampler(mu_d, base_var + var0, base_var + var1, pi))
    else:
        raise ValueError(f"Unknown config {cfg}")

    # CRPS per speed
    crps_up = crps_mc_perrow(up_samplers, up, S=crps_S)
    crps_down = crps_mc_perrow(down_samplers, down, S=crps_S)

    # Rollout: one sample per row
    up_hat = np.zeros(N)
    down_hat = np.zeros(N)
    if cfg == "gmm":
        for i in range(N):
            p = model["params"][str(src[i])] if str(src[i]) in model["params"] else model["params"][int(src[i])]
            mean = torch.as_tensor(p["mean"], dtype=torch.float64)
            cov = torch.as_tensor(p["cov"], dtype=torch.float64)
            dist = TorchMVN(loc=mean, covariance_matrix=cov + 1e-12 * torch.eye(2, dtype=cov.dtype))
            sample = dist.sample()
            up_hat[i], down_hat[i] = float(sample[0].item()), float(sample[1].item())
    elif cfg == "markov":
        for i in range(N):
            p = model["params"][str(src[i])] if str(src[i]) in model["params"] else model["params"][int(src[i])]
            up_dist = TorchNormal(torch.as_tensor(p["up_mean"], dtype=torch.float64), torch.sqrt(torch.as_tensor(p["up_var"], dtype=torch.float64)))
            up_s = float(up_dist.sample().item())
            down_dist = TorchNormal(torch.as_tensor(p["a"] * up_s + p["b"], dtype=torch.float64), torch.sqrt(torch.as_tensor(p["down_var"], dtype=torch.float64)))
            down_s = float(down_dist.sample().item())
            up_hat[i], down_hat[i] = up_s, down_s
    elif cfg == "latent":
        sigma0 = model["sigma0"]
        sigma1 = model["sigma1"]
        for i in range(N):
            p = model["params"][str(src[i])] if str(src[i]) in model["params"] else model["params"][int(src[i])]
            u = float(TorchNormal(torch.as_tensor(p["mu_u"], dtype=torch.float64), torch.sqrt(torch.as_tensor(p["var_u"], dtype=torch.float64))).sample().item())
            v = float(TorchNormal(torch.as_tensor(p["a"] * u + p["b"], dtype=torch.float64), torch.sqrt(torch.as_tensor(p["var_v"], dtype=torch.float64))).sample().item())
            z = bool(torch.rand(()) < torch.as_tensor(p["pi"], dtype=torch.float64))
            sig = sigma0 if z else sigma1
            up_hat[i] = float(TorchNormal(torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(sig, dtype=torch.float64)).sample().item())
            down_hat[i] = float(TorchNormal(torch.as_tensor(v, dtype=torch.float64), torch.as_tensor(sig, dtype=torch.float64)).sample().item())

    # Joint rollout metrics
    def energy_distance(xy: np.ndarray, uv: np.ndarray, M: int = 2000) -> float:
        n = xy.shape[0]
        m = uv.shape[0]
        M1 = min(n, M)
        M2 = min(m, M)
        idx1 = np.random.choice(n, M1, replace=False)
        idx2 = np.random.choice(m, M2, replace=False)
        A = xy[idx1]
        B = uv[idx2]
        # 2 E||A - B||
        d_ab = 0.0
        for a in A:
            d_ab += np.mean(np.sqrt(np.sum((B - a) ** 2, axis=1)))
        d_ab = 2.0 * (d_ab / M1)
        # E||A - A'||
        d_aa = 0.0
        for i in range(M1):
            a = A[i]
            others = np.concatenate([A[:i], A[i + 1 :]], axis=0)
            if len(others) == 0:
                continue
            d_aa += np.mean(np.sqrt(np.sum((others - a) ** 2, axis=1)))
        d_aa = d_aa / max(M1, 1)
        # E||B - B'||
        d_bb = 0.0
        for i in range(M2):
            b = B[i]
            others = np.concatenate([B[:i], B[i + 1 :]], axis=0)
            if len(others) == 0:
                continue
            d_bb += np.mean(np.sqrt(np.sum((others - b) ** 2, axis=1)))
        d_bb = d_bb / max(M2, 1)
        return float(d_ab - d_aa - d_bb)

    corr_data = np.corrcoef(up, down)[0, 1]
    corr_hat = np.corrcoef(up_hat, down_hat)[0, 1]
    corr_err = abs(float(corr_data - corr_hat))
    joint_energy = energy_distance(np.stack([up, down], axis=1), np.stack([up_hat, down_hat], axis=1), M=energy_M)
    q90_data = float(np.percentile(down, 90.0))
    q90_hat = float(np.percentile(down_hat, 90.0))
    q90_err = abs(q90_data - q90_hat)
    var_ratio = float(np.var(down_hat, ddof=1) / (np.var(down, ddof=1) + 1e-12))

    def summarize(arr: np.ndarray) -> Tuple[float, float, float]:
        return float(arr.mean()), float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9))

    nll_up_mean, nll_up_q50, nll_up_q90 = summarize(nll_up)
    nll_down_mean, nll_down_q50, nll_down_q90 = summarize(nll_down)
    crps_up_mean = float(crps_up.mean())
    crps_down_mean = float(crps_down.mean())

    return {
        "upstream": {
            "nll_mean": nll_up_mean,
            "nll_q50": nll_up_q50,
            "nll_q90": nll_up_q90,
            "crps_mean": crps_up_mean,
        },
        "downstream": {
            "nll_mean": nll_down_mean,
            "nll_q50": nll_down_q50,
            "nll_q90": nll_down_q90,
            "crps_mean": crps_down_mean,
        },
        "rollout": {
            "corr_err": corr_err,
            "joint_energy": joint_energy,
            "downstream_q90_err": q90_err,
            "downstream_var_ratio": var_ratio,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crps_samples", type=int, default=64)
    ap.add_argument("--energy_M", type=int, default=2000)
    args = ap.parse_args()

    df = read_parquet(args.data)
    model = load_model(args.model)
    metrics = evaluate(model, df, crps_S=args.crps_samples, energy_M=args.energy_M)

    ensure_dir(os.path.dirname(args.out) or ".")
    save_json(args.out, metrics)
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()