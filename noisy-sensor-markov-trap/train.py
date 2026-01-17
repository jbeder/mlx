# train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

import pandas as pd
import torch
import torch.nn as nn
from config import AppConfig, load_config
from model import JointGaussianModel, LatentModel, MarkovModel, SpeedDataset
from torch.utils.data import DataLoader, Dataset

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


def _make_model(cfg: AppConfig, num_sources: int) -> nn.Module:
    kind = cfg.model.kind

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
            num_components=cfg.model.latent.num_components,
            encoder_hidden=cfg.model.latent.encoder_hidden,
        )

    raise ValueError(f"Unknown model kind: {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, AppConfig)

    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_all(cfg.run.seed)
    device = _get_device(cfg.run.device)

    df = pd.read_parquet(cfg.run.data)
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

    model = _make_model(cfg, num_sources=num_sources).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in dl:
            source = batch.source.to(device)
            upstream_speed = batch.upstream_speed.to(device)
            downstream_speed = batch.downstream_speed.to(device)

            opt.zero_grad(set_to_none=True)

            if cfg.model.kind == "gmm":
                dist = model(source)
                y = torch.stack([upstream_speed, downstream_speed], dim=-1)
                loss = (-dist.log_prob(y)).mean()

            elif cfg.model.kind == "markov":
                up_dist, down_dist = model(source, upstream_speed)
                u = upstream_speed.unsqueeze(-1)
                d = downstream_speed.unsqueeze(-1)
                loss = ((-up_dist.log_prob(u)) + (-down_dist.log_prob(d))).mean()

            elif cfg.model.kind == "latent":
                loss = model.loss(
                    source,
                    upstream_speed,
                    downstream_speed,
                    num_samples=cfg.model.latent.elbo_samples,
                )
            else:
                raise ValueError(cfg.model.kind)

            loss.backward()

            if cfg.train.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip))

            opt.step()

            running += float(loss.detach().item())
            n_batches += 1

        avg = running / max(1, n_batches)
        print(f"epoch={epoch + 1}/{cfg.train.epochs} loss={avg:.6f}")

    payload = {
        "config_path": args.config,
        "num_sources": num_sources,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out_dir / "model.pt")

    with (out_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, default=lambda o: o.__dict__, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
