from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import JointGaussianModel, LatentModel, MarkovModel


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = torch.load(model_dir / "model.pt", map_location=device)
    kind = payload.get("model_kind")
    if kind is None:
        raise RuntimeError("model.pt missing 'model_kind'")

    num_sources = int(payload["num_sources"])

    # Recreate architecture to load state_dict
    if kind == "gmm":
        model: nn.Module = JointGaussianModel(num_sources=num_sources)
    elif kind == "markov":
        model = MarkovModel(num_sources=num_sources)
    elif kind == "latent":
        model = LatentModel(num_sources=num_sources)
    else:
        raise ValueError(f"Unknown model_kind: {kind}")

    model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
    model.to(device)
    model.eval()
    return model, payload


def _to_tensor(x: pd.Series, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x.to_numpy(), dtype=dtype)


def _crps_mc(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Monte Carlo CRPS for 1D predictive distribution.

    Args:
      samples: (S, B) tensor of samples
      target: (B,) tensor of observed values

    Returns:
      (B,) tensor with CRPS per row
    """
    # E|S - x|
    mean_abs = (samples - target.unsqueeze(0)).abs().mean(dim=0)  # (B,)

    # 0.5 * E|S - S'| using sorted-sample formula for efficiency
    # pairwise_mean = (1/S^2) sum_{i,j} |s_i - s_j|
    S = samples.size(0)
    s_sorted, _ = samples.sort(dim=0)
    # weights w_i = 2*i - S - 1 for i=1..S
    idx = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype).unsqueeze(1)  # (S,1)
    w = 2 * idx - (S + 1)
    pairwise_mean = (2.0 / (S * S)) * (w * s_sorted).sum(dim=0)  # (B,)
    crps = mean_abs - 0.5 * pairwise_mean
    return crps


@torch.no_grad()
def _gmm_1d_log_prob(component_dist, mixture_dist, x: torch.Tensor, dim_index: int) -> torch.Tensor:
    """
    Compute 1D marginal log-prob for a mixture of multivariate normals along a given dimension.
    Assumes component_dist is MultivariateNormal with batch shape (B, K) and event dim 2.
    mixture_dist is Categorical with logits shape (B, K).

    Args:
      component_dist: distribution with attributes .loc (B,K,D), .covariance_matrix (B,K,D,D)
      mixture_dist: distribution with .logits (B,K)
      x: (B,) observed values
      dim_index: 0 for upstream, 1 for downstream
    Returns: (B,) log probability
    """
    logits = mixture_dist.logits  # (B,K)
    log_w = torch.log_softmax(logits, dim=-1)  # (B,K)

    loc = getattr(component_dist, "loc")  # (B,K,2)
    # Prefer covariance_matrix if available
    if hasattr(component_dist, "covariance_matrix"):
        cov = component_dist.covariance_matrix  # (B,K,2,2)
        var = cov[..., dim_index, dim_index]
    elif hasattr(component_dist, "scale_tril"):
        L = component_dist.scale_tril  # (B,K,2,2)
        var = (L[..., dim_index, :] ** 2).sum(dim=-1)
    else:
        raise RuntimeError("Unsupported component distribution for GMM marginals")

    mu = loc[..., dim_index]  # (B,K)
    std = (var.clamp_min(1e-12)).sqrt()

    x_ = x.unsqueeze(-1)  # (B,1)
    lp_comp = torch.distributions.Normal(mu, std).log_prob(x_)  # (B,K)
    lp = torch.logsumexp(log_w + lp_comp, dim=-1)  # (B,)
    return lp


@torch.no_grad()
def _compute_metrics_gmm(model: JointGaussianModel, df: pd.DataFrame, crps_samples: int = 64) -> Dict:
    device = next(model.parameters()).device

    source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
    upstream = _to_tensor(df["upstream_speed"]).to(device)
    downstream = _to_tensor(df["downstream_speed"]).to(device)

    dist = model(source)
    mix = dist.mixture_distribution
    comp = dist.component_distribution

    # Per-dim NLLs
    lp_up = _gmm_1d_log_prob(comp, mix, upstream, dim_index=0)
    lp_down = _gmm_1d_log_prob(comp, mix, downstream, dim_index=1)
    nll_up = (-lp_up).cpu().numpy()
    nll_down = (-lp_down).cpu().numpy()

    # CRPS via MC from joint samples
    samples = dist.sample((crps_samples,))  # (S,B,2)
    up_s = samples[..., 0]  # (S,B)
    down_s = samples[..., 1]
    crps_up = _crps_mc(up_s, upstream).cpu().numpy()
    crps_down = _crps_mc(down_s, downstream).cpu().numpy()

    # Rollout one sample per row
    y_hat = dist.sample()  # (B,2)
    up_hat = y_hat[..., 0].cpu().numpy()
    down_hat = y_hat[..., 1].cpu().numpy()

    return _aggregate_metrics(df, nll_up, nll_down, crps_up, crps_down, up_hat, down_hat)


@torch.no_grad()
def _compute_metrics_markov(model: MarkovModel, df: pd.DataFrame, crps_samples: int = 64) -> Dict:
    device = next(model.parameters()).device

    source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
    upstream = _to_tensor(df["upstream_speed"]).to(device)
    downstream = _to_tensor(df["downstream_speed"]).to(device)

    up_dist, down_dist = model(source, upstream)

    # NLLs (teacher-forced)
    nll_up = (-up_dist.log_prob(upstream.unsqueeze(-1)).squeeze(-1)).cpu().numpy()
    nll_down = (-down_dist.log_prob(downstream.unsqueeze(-1)).squeeze(-1)).cpu().numpy()

    # CRPS via MC from 1D distributions
    up_s = up_dist.sample((crps_samples,)).squeeze(-1)  # (S,B)
    down_s = down_dist.sample((crps_samples,)).squeeze(-1)
    crps_up = _crps_mc(up_s, upstream).cpu().numpy()
    crps_down = _crps_mc(down_s, downstream).cpu().numpy()

    # Rollout: one sample per row
    ro = model.rollout(source)
    up_hat = ro.upstream_sample.squeeze(-1).cpu().numpy()
    down_hat = ro.downstream_sample.squeeze(-1).cpu().numpy()

    return _aggregate_metrics(df, nll_up, nll_down, crps_up, crps_down, up_hat, down_hat)


@torch.no_grad()
def _latent_log_prob_up(
    model: LatentModel, source: torch.Tensor, xu: torch.Tensor, num_samples: int = 64
) -> torch.Tensor:
    """Approximate log p(xu | source) via Monte Carlo over u and sensor mix."""
    device = xu.device
    ctx = model.source_emb(source)
    p_u = model.prior_u(ctx)

    # Sample u: (S,B,1)
    u = p_u.sample((num_samples,))
    if u.dim() == 2:
        u = u.unsqueeze(-1)
    # Sensor mixture weights
    logit = model.sensor_logit(source).squeeze(-1)  # (B,)
    log_w0 = torch.nn.functional.logsigmoid(logit)
    log_w1 = torch.nn.functional.logsigmoid(-logit)
    std = torch.exp(model.sensor_log_std.to(device))  # (2,)
    s0, s1 = std[0], std[1]

    xu_ = xu.unsqueeze(0).unsqueeze(-1)  # (1,B,1)
    lp0 = torch.distributions.Normal(u, s0).log_prob(xu_)  # (S,B,1)
    lp1 = torch.distributions.Normal(u, s1).log_prob(xu_)
    lp_mix = torch.logsumexp(torch.stack([lp0.squeeze(-1) + log_w0, lp1.squeeze(-1) + log_w1], dim=-1), dim=-1)
    # IWAE-style average over u samples
    return torch.logsumexp(lp_mix, dim=0) - torch.log(torch.tensor(float(num_samples), device=device))


@torch.no_grad()
def _latent_log_prob_down(
    model: LatentModel, source: torch.Tensor, xd: torch.Tensor, num_samples: int = 64
) -> torch.Tensor:
    """Approximate log p(xd | source) via Monte Carlo over u, v and sensor mix."""
    device = xd.device
    ctx = model.source_emb(source)
    p_u = model.prior_u(ctx)

    # Sample u: (S,B,1)
    u = p_u.sample((num_samples,))
    if u.dim() == 2:
        u = u.unsqueeze(-1)

    # Build batched contexts for v prior
    S, B = u.shape[0], u.shape[1]
    ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
    pv_ctx = torch.cat([ctx_rep, u], dim=-1)  # (S,B,E+1)
    pv_ctx_flat = pv_ctx.reshape(S * B, -1)
    p_v = model.prior_v(pv_ctx_flat)
    v = p_v.sample().reshape(S, B, 1)

    # Sensor mix
    logit = model.sensor_logit(source).squeeze(-1)
    log_w0 = torch.nn.functional.logsigmoid(logit)
    log_w1 = torch.nn.functional.logsigmoid(-logit)
    std = torch.exp(model.sensor_log_std.to(device))
    s0, s1 = std[0], std[1]

    xd_ = xd.unsqueeze(0).unsqueeze(-1)
    lp0 = torch.distributions.Normal(v, s0).log_prob(xd_)
    lp1 = torch.distributions.Normal(v, s1).log_prob(xd_)
    lp_mix = torch.logsumexp(torch.stack([lp0.squeeze(-1) + log_w0, lp1.squeeze(-1) + log_w1], dim=-1), dim=-1)
    return torch.logsumexp(lp_mix, dim=0) - torch.log(torch.tensor(float(num_samples), device=device))


@torch.no_grad()
def _latent_samples_up(model: LatentModel, source: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Samples from p(xu | source). Returns (S,B)."""
    device = source.device
    ctx = model.source_emb(source)
    p_u = model.prior_u(ctx)
    u = p_u.sample((num_samples,))  # (S,B,1)
    if u.dim() == 2:
        u = u.unsqueeze(-1)

    # Sensor noise
    logit = model.sensor_logit(source).squeeze(-1)  # (B,)
    pi = torch.sigmoid(logit)
    k = torch.bernoulli(pi.unsqueeze(0).expand(num_samples, -1)).long()  # (S,B)
    std = torch.exp(model.sensor_log_std.to(device))  # (2,)
    sigma = std[k]  # (S,B)
    eps = torch.randn_like(sigma)
    x = u.squeeze(-1) + sigma * eps  # (S,B)
    return x


@torch.no_grad()
def _latent_samples_down(model: LatentModel, source: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Samples from p(xd | source). Returns (S,B)."""
    device = source.device
    ctx = model.source_emb(source)
    p_u = model.prior_u(ctx)
    u = p_u.sample((num_samples,))  # (S,B,1)
    if u.dim() == 2:
        u = u.unsqueeze(-1)

    S, B = u.shape[0], u.shape[1]
    ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
    pv_ctx = torch.cat([ctx_rep, u], dim=-1)  # (S,B,E+1)
    pv_ctx_flat = pv_ctx.reshape(S * B, -1)
    p_v = model.prior_v(pv_ctx_flat)
    v = p_v.sample().reshape(S, B)  # (S,B)

    logit = model.sensor_logit(source).squeeze(-1)  # (B,)
    pi = torch.sigmoid(logit)
    k = torch.bernoulli(pi.unsqueeze(0).expand(num_samples, -1)).long()  # (S,B)
    std = torch.exp(model.sensor_log_std.to(device))
    sigma = std[k]
    eps = torch.randn_like(sigma)
    x = v + sigma * eps  # (S,B)
    return x


@torch.no_grad()
def _compute_metrics_latent(model: LatentModel, df: pd.DataFrame, crps_samples: int = 64) -> Dict:
    device = next(model.parameters()).device

    source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
    upstream = _to_tensor(df["upstream_speed"]).to(device)
    downstream = _to_tensor(df["downstream_speed"]).to(device)

    # NLLs for 1D predictive marginals
    lp_up = _latent_log_prob_up(model, source, upstream, num_samples=crps_samples)
    lp_down = _latent_log_prob_down(model, source, downstream, num_samples=crps_samples)
    nll_up = (-lp_up).cpu().numpy()
    nll_down = (-lp_down).cpu().numpy()

    # CRPS via MC
    up_s = _latent_samples_up(model, source, num_samples=crps_samples)
    down_s = _latent_samples_down(model, source, num_samples=crps_samples)
    crps_up = _crps_mc(up_s, upstream).cpu().numpy()
    crps_down = _crps_mc(down_s, downstream).cpu().numpy()

    # Rollout one sample per row
    ro = model.sample(source)
    up_hat = ro.upstream.cpu().numpy()
    down_hat = ro.downstream.cpu().numpy()

    return _aggregate_metrics(df, nll_up, nll_down, crps_up, crps_down, up_hat, down_hat)


def _aggregate_metrics(
    df: pd.DataFrame,
    nll_up: np.ndarray,
    nll_down: np.ndarray,
    crps_up: np.ndarray,
    crps_down: np.ndarray,
    up_hat: np.ndarray,
    down_hat: np.ndarray,
    energy_k: int = 2000,
) -> Dict:
    # Per-speed metrics
    upstream_metrics = {
        "nll_mean": float(np.mean(nll_up)),
        "nll_q90": float(np.quantile(nll_up, 0.9)),
        "crps_mean": float(np.mean(crps_up)),
    }
    downstream_metrics = {
        "nll_mean": float(np.mean(nll_down)),
        "nll_q90": float(np.quantile(nll_down, 0.9)),
        "crps_mean": float(np.mean(crps_down)),
    }

    # Rollout metrics
    obs_up = df["upstream_speed"].to_numpy()
    obs_down = df["downstream_speed"].to_numpy()

    # Energy distance on random subset of size m = min(N, energy_k)
    N = len(df)
    m = min(N, energy_k)
    idx = np.random.choice(N, size=m, replace=False)
    X = np.stack([obs_up[idx], obs_down[idx]], axis=1)  # (m,2)
    Y = np.stack([up_hat[idx], down_hat[idx]], axis=1)

    def pairwise_mean_norm(a: np.ndarray, b: np.ndarray) -> float:
        # Mean Euclidean distance between all pairs from a and b
        diff = a[:, None, :] - b[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        return float(d.mean())

    cross = pairwise_mean_norm(X, Y)
    within_x = pairwise_mean_norm(X, X)
    within_y = pairwise_mean_norm(Y, Y)
    joint_energy = 2.0 * cross - within_x - within_y

    # Other rollout stats over full population
    down_q90_err = float(abs(np.quantile(obs_down, 0.9) - np.quantile(down_hat, 0.9)))
    down_var_ratio = float(np.var(down_hat, ddof=0) / max(1e-12, np.var(obs_down, ddof=0)))
    down_mean_err = float(abs(np.mean(obs_down) - np.mean(down_hat)))
    up_var_ratio = float(np.var(up_hat, ddof=0) / max(1e-12, np.var(obs_up, ddof=0)))

    rollout_metrics = {
        "joint_energy": float(joint_energy),
        "downstream_q90_err": down_q90_err,
        "downstream_var_ratio": down_var_ratio,
        "downstream_mean_err": down_mean_err,
        "upstream_var_ratio": up_var_ratio,
    }

    return {
        "upstream": upstream_metrics,
        "downstream": downstream_metrics,
        "rollout": rollout_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Input parquet file path")
    ap.add_argument("--model", type=str, required=True, help="Model directory containing model.pt")
    ap.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation (default: cpu)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--crps_samples", type=int, default=64, help="Number of MC samples for CRPS/NLL approx")
    ap.add_argument(
        "--energy_k",
        type=int,
        default=2000,
        help="Use a random subset of this size for energy distance (default: 2000)",
    )
    args = ap.parse_args()

    _seed_all(args.seed)
    device = torch.device(args.device)

    df = pd.read_parquet(args.data)
    model_dir = Path(args.model)
    out_dir = model_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, payload = _load_model(model_dir, device)
    kind = payload["model_kind"]

    if kind == "gmm":
        metrics = _compute_metrics_gmm(model, df, crps_samples=args.crps_samples)
    elif kind == "markov":
        metrics = _compute_metrics_markov(model, df, crps_samples=args.crps_samples)
    elif kind == "latent":
        metrics = _compute_metrics_latent(model, df, crps_samples=args.crps_samples)
    else:
        raise ValueError(kind)

    # Energy distance subset parameter
    # Re-aggregate rollout metrics with chosen energy subset size if needed
    # The per-kind compute already used default; re-run aggregation only for energy
    # Here we recompute rollout metrics using the stored per-kind outputs would require refactor.
    # Simpler: recompute energy here from df and last rollout samples via per-kind recomputation.
    # To avoid additional sampling noise, we recompute metrics dict to include energy with specific k.
    # We'll patch-in energy using aggregation util.
    # Extract rollout samples again for energy with args.energy_k.
    # This requires mild duplication; acceptable for clarity.

    # Recompute rollout metrics portion using fresh samples for consistency with requested 2k subset
    # Note: Other metrics remain unchanged.
    if kind == "gmm":
        source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
        dist = model(source)
        y_hat = dist.sample()
        up_hat = y_hat[..., 0].cpu().numpy()
        down_hat = y_hat[..., 1].cpu().numpy()
    elif kind == "markov":
        source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
        ro = model.rollout(source)
        up_hat = ro.upstream_sample.squeeze(-1).cpu().numpy()
        down_hat = ro.downstream_sample.squeeze(-1).cpu().numpy()
    else:  # latent
        source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
        ro = model.sample(source)
        up_hat = ro.upstream.cpu().numpy()
        down_hat = ro.downstream.cpu().numpy()

    # Compute energy with requested subset
    N = len(df)
    m = min(N, int(args.energy_k))
    idx = np.random.choice(N, size=m, replace=False)
    X = np.stack([df["upstream_speed"].to_numpy()[idx], df["downstream_speed"].to_numpy()[idx]], axis=1)
    Y = np.stack([up_hat[idx], down_hat[idx]], axis=1)
    diff_xy = X[:, None, :] - Y[None, :, :]
    cross = float(np.sqrt((diff_xy * diff_xy).sum(axis=-1)).mean())
    diff_xx = X[:, None, :] - X[None, :, :]
    within_x = float(np.sqrt((diff_xx * diff_xx).sum(axis=-1)).mean())
    diff_yy = Y[:, None, :] - Y[None, :, :]
    within_y = float(np.sqrt((diff_yy * diff_yy).sum(axis=-1)).mean())
    joint_energy = 2.0 * cross - within_x - within_y

    # Patch rollout metrics
    obs_up = df["upstream_speed"].to_numpy()
    obs_down = df["downstream_speed"].to_numpy()
    metrics["rollout"] = {
        "joint_energy": float(joint_energy),
        "downstream_q90_err": float(abs(np.quantile(obs_down, 0.9) - np.quantile(down_hat, 0.9))),
        "downstream_var_ratio": float(np.var(down_hat, ddof=0) / max(1e-12, np.var(obs_down, ddof=0))),
        "downstream_mean_err": float(abs(np.mean(obs_down) - np.mean(down_hat))),
        "upstream_var_ratio": float(np.var(up_hat, ddof=0) / max(1e-12, np.var(obs_up, ddof=0))),
    }

    out_path = out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
