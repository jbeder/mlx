from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import AppConfig, decode_config
from .model import JointGaussianModel, LatentModel, MarkovModel


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _load_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = torch.load(model_dir / "model.pt", map_location=device)
    kind = payload.get("model_kind")
    if kind is None:
        raise RuntimeError("model.pt missing 'model_kind'")

    num_sources = int(payload["num_sources"])

    cfg_path = model_dir / "config.resolved.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = decode_config(AppConfig, json.load(f))

    if kind == "gmm":
        model: nn.Module = JointGaussianModel(num_sources=num_sources)
    elif kind == "markov":
        model = MarkovModel(
            num_sources=num_sources,
            emb_dim=cfg.model.markov.emb_dim,
            num_components=cfg.model.markov.num_components,
        )
    elif kind == "latent":
        model = LatentModel(
            num_sources=num_sources,
            emb_dim=cfg.model.latent.emb_dim,
            num_components=cfg.model.latent.num_components,
            encoder_hidden=cfg.model.latent.encoder_hidden,
        )
    else:
        raise ValueError(f"Unknown model_kind: {kind}")

    model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
    model.to(device)
    model.eval()
    return model, payload


def _crps_mc(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Monte Carlo CRPS for 1D predictive distribution.

    Args:
      samples: (S, B) array of rollout samples
      target: (B,) array of observed values

    Returns:
      (B,) array with CRPS per row
    """
    S = samples.shape[0]
    mean_abs = np.mean(np.abs(samples - target[None, :]), axis=0)  # (B,)

    # Pairwise term using the equivalent sorted-sum trick to avoid O(S^2) memory
    s_sorted = np.sort(samples, axis=0)
    idx = np.arange(1, S + 1, dtype=s_sorted.dtype)[:, None]
    w = 2.0 * idx - (S + 1)
    pairwise_mean = (2.0 / (S * S)) * np.sum(w * s_sorted, axis=0)
    crps = mean_abs - 0.5 * pairwise_mean
    return crps


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx * vx).sum() * (vy * vy).sum())
    if denom <= 0:
        return 0.0
    return float((vx * vy).sum() / denom)


def _energy_distance_2d(obs: np.ndarray, pred: np.ndarray) -> float:
    """Energy distance between two 2D point clouds.

    obs: (N,2), pred: (N,2) ideally same N (we'll assume same length)
    """
    if len(obs) == 0 or len(pred) == 0:
        return 0.0

    def pair_mean(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None, :] - b[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        return float(d.mean())

    cross = pair_mean(obs, pred)
    within_obs = 0.0
    within_pred = 0.0
    if obs.shape[0] >= 2:
        Dxx = np.sqrt(((obs[:, None, :] - obs[None, :, :]) ** 2).sum(axis=-1))
        within_obs = float(Dxx[~np.eye(obs.shape[0], dtype=bool)].mean())
    if pred.shape[0] >= 2:
        Dyy = np.sqrt(((pred[:, None, :] - pred[None, :, :]) ** 2).sum(axis=-1))
        within_pred = float(Dyy[~np.eye(pred.shape[0], dtype=bool)].mean())
    return 2.0 * cross - within_obs - within_pred


def _aggregate_rollout_metrics(
    df: pd.DataFrame,
    u_s: np.ndarray,  # (S,B)
    d_s: np.ndarray,  # (S,B)
    *,
    take_first_sample: bool = True,
) -> Dict:
    # Observed
    obs_up = df["upstream_speed"].to_numpy()
    obs_down = df["downstream_speed"].to_numpy()
    sources = df["source"].to_numpy()

    # CRPS (rollout) upstream/downstream
    crps_up = _crps_mc(u_s, obs_up).mean()
    crps_down = _crps_mc(d_s, obs_down).mean()

    # Per-source downstream mean MAE and std log-error using all rollout samples
    # Collapse samples across S for mean/std aggregation
    S, B = d_s.shape
    d_flat = d_s.reshape(S * B)
    src_rep = np.repeat(sources, S)

    downstream_mean_errs = []
    downstream_sd_logerrs = []
    for k in np.unique(sources):
        mask_data = sources == k
        mask_roll = src_rep == k
        if mask_data.sum() == 0 or mask_roll.sum() == 0:
            continue
        mu_data = float(obs_down[mask_data].mean())
        mu_roll = float(d_flat[mask_roll].mean())
        downstream_mean_errs.append(abs(mu_roll - mu_data))

        sd_data = float(obs_down[mask_data].std(ddof=0))
        sd_roll = float(d_flat[mask_roll].std(ddof=0))
        if sd_data <= 0 or sd_roll <= 0:
            downstream_sd_logerrs.append(0.0)
        else:
            downstream_sd_logerrs.append(abs(np.log(sd_roll / sd_data)))

    mean_mae_by_source = float(np.mean(downstream_mean_errs)) if downstream_mean_errs else 0.0
    sd_logerr_by_source = float(np.mean(downstream_sd_logerrs)) if downstream_sd_logerrs else 0.0

    # Within-source correlation error (use one rollout sample per row to avoid overweighting rows)
    if take_first_sample:
        u_one = u_s[0]
        d_one = d_s[0]
    else:
        # Random single sample per row
        idx = np.random.randint(0, u_s.shape[0], size=u_s.shape[1])
        u_one = u_s[idx, np.arange(u_s.shape[1])]
        d_one = d_s[idx, np.arange(d_s.shape[1])]

    corr_errs = []
    for k in np.unique(sources):
        mask = sources == k
        if mask.sum() < 2:
            continue
        rho_data = _pearson_corr(obs_up[mask], obs_down[mask])
        rho_roll = _pearson_corr(u_one[mask], d_one[mask])
        corr_errs.append(abs(rho_roll - rho_data))
    corr_err_by_source = float(np.mean(corr_errs)) if corr_errs else 0.0

    # Joint energy distance (2D) using one sample per row
    obs_pairs = np.stack([obs_up, obs_down], axis=1)  # (B,2)
    roll_pairs = np.stack([u_one, d_one], axis=1)  # (B,2)
    energy_2d = float(_energy_distance_2d(obs_pairs, roll_pairs))

    return {
        "crps_upstream": float(crps_up),
        "crps_downstream": float(crps_down),
        "downstream_mean_mae_by_source": mean_mae_by_source,
        "downstream_std_logerr_by_source": sd_logerr_by_source,
        "corr_err_by_source": corr_err_by_source,
        "energy_2d": energy_2d,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # Data location: allow either explicit --data or (--data_dir + --mode)
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    ap.add_argument("--data", type=str, default=None, help="Input parquet file path (overrides --data_dir/--mode)")
    ap.add_argument("--data_dir", type=str, default=default_data_dir, help=f"Directory containing parquet data (default: {default_data_dir})")
    ap.add_argument("--mode", type=str, choices=["clean", "noisy"], default="clean", help="Dataset variant when using --data_dir (default: clean)")
    ap.add_argument("--model", type=str, required=True, help="Model directory containing model.pt")
    ap.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation (default: cpu)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--rollout_samples", type=int, default=16, help="Number of rollout samples per row (default: 16)")
    args = ap.parse_args()

    _seed_all(args.seed)
    device = torch.device(args.device)

    data_path = args.data or os.path.join(args.data_dir, f"{args.mode}.parquet")
    df = pd.read_parquet(data_path)
    model_dir = Path(args.model)
    out_dir = model_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, payload = _load_model(model_dir, device)

    source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
    with torch.no_grad():
        u_s, d_s = model.sample(source.to(device), num_samples=args.rollout_samples)
        # Convert to numpy for metric computations
        u_s_np = u_s.detach().cpu().numpy()
        d_s_np = d_s.detach().cpu().numpy()

    rollout_metrics = _aggregate_rollout_metrics(df, u_s_np, d_s_np)

    metrics = {
        "meta": {
            "num_samples": int(args.rollout_samples),
            "num_rows": int(len(df)),
        },
        "rollout": rollout_metrics,
    }

    out_path = out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()