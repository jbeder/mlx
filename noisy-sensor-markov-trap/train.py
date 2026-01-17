# train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import JointGaussianModel, LatentModel, MarkovModel, SpeedDataset  # type: ignore


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _infer_num_sources(df: pd.DataFrame) -> int:
    mx = int(df["source"].max())
    return mx + 1


def _make_model(config: str, num_sources: int, emb_dim: int, num_components: int, latent_samples: int) -> nn.Module:
    if config == "gmm":
        # joint 2D model conditioned on source
        # (your JointGaussianModel should accept num_sources and expose forward(source) -> dist over (u,d))
        return JointGaussianModel(num_sources=num_sources)
    if config == "markov":
        return MarkovModel(num_sources=num_sources, emb_dim=emb_dim, num_components=num_components)
    if config == "latent":
        # latent model uses its own internal encoder hidden size; keep defaults unless you add args
        return LatentModel(num_sources=num_sources, emb_dim=emb_dim, num_components=num_components)
    raise ValueError(f"Unknown config: {config!r}")


def _batch_to_device(batch, device: torch.device):
    return batch.source.to(device), batch.upstream_speed.to(device), batch.downstream_speed.to(device)


def _train_step(
    model: nn.Module,
    config: str,
    source: torch.Tensor,
    upstream_speed: torch.Tensor,
    downstream_speed: torch.Tensor,
    latent_samples: int,
) -> torch.Tensor:
    if config == "gmm":
        dist = model(source)
        y = torch.stack([upstream_speed, downstream_speed], dim=-1)  # (B, 2)
        nll = -dist.log_prob(y)  # (B,)
        return nll.mean()

    if config == "markov":
        up_dist, down_dist = model(source, upstream_speed)
        u = upstream_speed.unsqueeze(-1)  # (B, 1)
        d = downstream_speed.unsqueeze(-1)  # (B, 1)
        nll_u = -up_dist.log_prob(u)  # (B,)
        nll_d = -down_dist.log_prob(d)  # (B,)
        return (nll_u + nll_d).mean()

    if config == "latent":
        # LatentModel.loss returns scalar
        return model.loss(source, upstream_speed, downstream_speed, num_samples=latent_samples)

    raise ValueError(f"Unknown config: {config!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["gmm", "markov", "latent"], required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--emb_dim", type=int, default=8)
    p.add_argument("--num_components", type=int, default=2)
    p.add_argument("--latent_samples", type=int, default=1)  # only used for latent

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_all(args.seed)
    device = _get_device(args.device)

    df = pd.read_parquet(args.data)
    num_sources = _infer_num_sources(df)

    ds: Dataset = SpeedDataset(df)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory) and (device.type == "cuda"),
    )

    model = _make_model(
        config=args.config,
        num_sources=num_sources,
        emb_dim=args.emb_dim,
        num_components=args.num_components,
        latent_samples=args.latent_samples,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in dl:
            source, upstream_speed, downstream_speed = _batch_to_device(batch, device)

            opt.zero_grad(set_to_none=True)
            loss = _train_step(
                model=model,
                config=args.config,
                source=source,
                upstream_speed=upstream_speed,
                downstream_speed=downstream_speed,
                latent_samples=args.latent_samples,
            )
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            opt.step()

            running += float(loss.detach().item())
            n_batches += 1
            step += 1

        avg = running / max(1, n_batches)
        print(f"epoch={epoch + 1}/{args.epochs} loss={avg:.6f}")

    payload = {
        "config": args.config,
        "num_sources": num_sources,
        "emb_dim": args.emb_dim,
        "num_components": args.num_components,
        "latent_samples": args.latent_samples,
        "state_dict": model.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, out_dir / "model.pt")

    with (out_dir / "args.json").open("w") as f:
        json.dump(payload["args"], f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
