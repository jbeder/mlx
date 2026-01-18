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
    cfg = None
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    if kind == "gmm":
        model: nn.Module = JointGaussianModel(num_sources=num_sources)
    elif kind == "markov":
        if cfg is not None:
            emb_dim = int(cfg["model"]["markov"]["emb_dim"])  # type: ignore[index]
            num_components = int(cfg["model"]["markov"]["num_components"])  # type: ignore[index]
            model = MarkovModel(num_sources=num_sources, emb_dim=emb_dim, num_components=num_components)
        else:
            model = MarkovModel(num_sources=num_sources)
    elif kind == "latent":
        if cfg is not None:
            emb_dim = int(cfg["model"]["latent"]["emb_dim"])  # type: ignore[index]
            num_components = int(cfg["model"]["latent"]["num_components"])  # type: ignore[index]
            encoder_hidden = int(cfg["model"]["latent"]["encoder_hidden"])  # type: ignore[index]
            model = LatentModel(
                num_sources=num_sources,
                emb_dim=emb_dim,
                num_components=num_components,
                encoder_hidden=encoder_hidden,
            )
        else:
            model = LatentModel(num_sources=num_sources)
    else:
        raise ValueError(f"Unknown model_kind: {kind}")

    model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
    model.to(device)
    model.eval()
    return model, payload


def _to_tensor(x: pd.Series, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x.to_numpy(), dtype=dtype)


def _aggregate_metrics(
    df: pd.DataFrame,
    *,
    nll_up: np.ndarray,
    crps_up: np.ndarray,
    nll_up_name: str,
    crps_up_name: str,
    nll_down_tf: np.ndarray,
    crps_down_tf: np.ndarray,
    nll_down_tf_name: str,
    crps_down_tf_name: str,
    up_hat: np.ndarray,
    down_hat: np.ndarray,
    energy_k: int = 2000,
) -> Dict:
    # Teacher-forced fit metrics
    upstream_fit = {
        "nll_name": nll_up_name,
        "nll_mean": float(np.mean(nll_up)),
        "nll_q90": float(np.quantile(nll_up, 0.9)),
        "crps_name": crps_up_name,
        "crps_mean": float(np.mean(crps_up)),
    }
    downstream_tf_fit = {
        "nll_name": nll_down_tf_name,
        "nll_mean": float(np.mean(nll_down_tf)),
        "nll_q90": float(np.quantile(nll_down_tf, 0.9)),
        "crps_name": crps_down_tf_name,
        "crps_mean": float(np.mean(crps_down_tf)),
    }

    # Rollout metrics (no teacher forcing)
    obs_up = df["upstream_speed"].to_numpy()
    obs_down = df["downstream_speed"].to_numpy()

    N = len(df)
    m = min(N, energy_k)
    idx = np.random.choice(N, size=m, replace=False)
    X = np.stack([obs_up[idx], obs_down[idx]], axis=1)  # (m,2)
    Y = np.stack([up_hat[idx], down_hat[idx]], axis=1)

    def pairwise_mean_norm(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None, :] - b[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        return float(d.mean())

    cross = pairwise_mean_norm(X, Y)

    if m >= 2:
        diff_xx = X[:, None, :] - X[None, :, :]
        Dxx = np.sqrt((diff_xx * diff_xx).sum(axis=-1))
        mask = ~np.eye(m, dtype=bool)
        within_x = float(Dxx[mask].mean())

        diff_yy = Y[:, None, :] - Y[None, :, :]
        Dyy = np.sqrt((diff_yy * diff_yy).sum(axis=-1))
        within_y = float(Dyy[mask].mean())
    else:
        within_x = 0.0
        within_y = 0.0
    joint_energy = 2.0 * cross - within_x - within_y

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
        "fit": {
            "upstream": upstream_fit,
            "downstream_tf": downstream_tf_fit,
        },
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

    source = torch.as_tensor(df["source"].to_numpy(), dtype=torch.long, device=device)
    upstream = _to_tensor(df["upstream_speed"]).to(device)
    downstream = _to_tensor(df["downstream_speed"]).to(device)

    with torch.no_grad():
        fit = model.eval_fit_arrays(
            source,
            upstream,
            downstream,
            crps_samples=args.crps_samples,
        )
        ro = model.eval_rollout_arrays(
            source,
        )

    metrics = _aggregate_metrics(
        df,
        nll_up=fit.nll_up.cpu().numpy(),
        crps_up=fit.crps_up.cpu().numpy(),
        nll_up_name=getattr(fit, "nll_up_name", "nll"),
        crps_up_name=getattr(fit, "crps_up_name", "crps"),
        nll_down_tf=fit.nll_down_tf.cpu().numpy(),
        crps_down_tf=fit.crps_down_tf.cpu().numpy(),
        nll_down_tf_name=getattr(fit, "nll_down_tf_name", "nll"),
        crps_down_tf_name=getattr(fit, "crps_down_tf_name", "crps"),
        up_hat=ro.up_hat.cpu().numpy(),
        down_hat=ro.down_hat.cpu().numpy(),
        energy_k=args.energy_k,
    )

    out_path = out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
