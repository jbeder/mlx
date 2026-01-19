from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TypeVar

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import AppConfig, load_config
from .model import JointGaussianModel, LatentModel, MarkovModel, SpeedDataset

T = TypeVar("T")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _infer_num_sources(df: pd.DataFrame) -> int:
    return int(df["source"].max()) + 1


def _make_model(cfg: AppConfig, num_sources: int, kind: str) -> nn.Module:
    if kind == "gmm":
        return JointGaussianModel(num_sources=num_sources)

    if kind == "markov":
        return MarkovModel(
            num_sources=num_sources,
            emb_dim=cfg.model.markov.emb_dim,
            num_components=cfg.model.markov.num_components,
        )

    if kind == "latent":
        return LatentModel(
            num_sources=num_sources,
            emb_dim=cfg.model.latent.emb_dim,
            num_components_u=cfg.model.latent.num_components_u,
            num_components_v=cfg.model.latent.num_components_v,
            encoder_hidden=cfg.model.latent.encoder_hidden,
            pv_ctx_hidden=cfg.model.latent.pv_ctx_hidden,
        )

    raise ValueError(f"Unknown model kind: {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to model/config YAML (no run info)")
    ap.add_argument("--model", type=str, required=True, choices=["gmm", "markov", "latent"], help="Model kind")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--device", type=str, default="auto", help="torch device, e.g. cpu|cuda|auto (default: auto)")
    # Data location: allow either explicit --data or (--data_dir + --mode)
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    ap.add_argument("--data", type=str, default=None, help="Input parquet file path (overrides --data_dir/--mode)")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help=f"Directory containing parquet data (default: {default_data_dir})",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["clean", "noisy"],
        default="clean",
        help="Dataset variant when using --data_dir (default: clean)",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for model + metrics")
    args = ap.parse_args()

    cfg = load_config(args.config, AppConfig)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_all(args.seed)
    device = _get_device(args.device)

    data_path = args.data or os.path.join(args.data_dir, f"{args.mode}.parquet")
    df = pd.read_parquet(data_path)
    num_sources = _infer_num_sources(df)

    ds: Dataset = SpeedDataset(df)
    dl = DataLoader(
        ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        drop_last=cfg.dataloader.drop_last,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=bool(cfg.dataloader.pin_memory) and (device.type == "cuda"),
    )

    model = _make_model(cfg, num_sources=num_sources, kind=args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in dl:
            source = batch.source.to(device)
            upstream_speed = batch.upstream_speed.to(device)
            downstream_speed = batch.downstream_speed.to(device)

            opt.zero_grad(set_to_none=True)

            if args.model == "gmm":
                dist = model(source)
                y = torch.stack([upstream_speed, downstream_speed], dim=-1)
                loss = (-dist.log_prob(y)).mean()

            elif args.model == "markov":
                up_dist, down_dist = model(source, upstream_speed)
                u = upstream_speed.unsqueeze(-1)
                d = downstream_speed.unsqueeze(-1)
                loss = ((-up_dist.log_prob(u)) + (-down_dist.log_prob(d))).mean()

            elif args.model == "latent":
                # KL annealing: scale KL term by beta in [start,end] over kl_anneal_steps
                if cfg.model.latent.kl_anneal_steps > 0:
                    t = min(1.0, global_step / float(cfg.model.latent.kl_anneal_steps))
                else:
                    t = 1.0
                beta = (1.0 - t) * cfg.model.latent.kl_beta_start + t * cfg.model.latent.kl_beta_end

                log_p, log_q, log_px = model.elbo_terms(
                    source,
                    upstream_speed,
                    downstream_speed,
                    num_samples=cfg.model.latent.elbo_samples,
                )
                # Negative ELBO with KL scaled by beta
                loss = -(log_px + beta * (log_p - log_q))
            else:
                raise ValueError(args.model)

            loss.backward()

            if cfg.train.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip))

            opt.step()

            running += float(loss.detach().item())
            n_batches += 1
            global_step += 1

        avg = running / max(1, n_batches)
        print(f"epoch={epoch + 1}/{cfg.train.epochs} loss={avg:.6f}")

    payload = {
        "config_path": args.config,
        "model_kind": args.model,
        "seed": args.seed,
        "device": str(device),
        "data": data_path,
        "out_dir": str(out_dir),
        "num_sources": num_sources,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out_dir / "model.pt")

    with (out_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, default=lambda o: o.__dict__, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
