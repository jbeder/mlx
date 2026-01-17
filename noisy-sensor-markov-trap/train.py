# train.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast

import pandas as pd
import torch
import torch.nn as nn
import yaml

# Assumes these are defined in model.py:
# - SpeedDataset
# - JointGaussianModel
# - MarkovModel
# - LatentModel
from model import JointGaussianModel, LatentModel, MarkovModel, SpeedDataset  # type: ignore
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
    mx = int(df["source"].max())
    return mx + 1


def _as_mapping(x: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(x).__name__}")
    return cast(Mapping[str, Any], x)


def _get(m: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in m:
        raise KeyError(f"Missing required key: {path}.{key}")
    return m[key]


def _get_opt(m: Mapping[str, Any], key: str, default: Any) -> Any:
    return m[key] if key in m else default


def _expect_str(x: Any, *, path: str) -> str:
    if not isinstance(x, str):
        raise TypeError(f"{path} must be str, got {type(x).__name__}")
    return x


def _expect_int(x: Any, *, path: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{path} must be int, got {type(x).__name__}")
    return int(x)


def _expect_float(x: Any, *, path: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{path} must be float, got {type(x).__name__}")
    return float(x)


def _expect_bool(x: Any, *, path: str) -> bool:
    if not isinstance(x, bool):
        raise TypeError(f"{path} must be bool, got {type(x).__name__}")
    return bool(x)


@dataclass(frozen=True)
class RunConfig:
    seed: int
    device: str
    data: str
    out_dir: str

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "RunConfig":
        return RunConfig(
            seed=_expect_int(_get(d, "seed", path=path), path=f"{path}.seed"),
            device=_expect_str(_get(d, "device", path=path), path=f"{path}.device"),
            data=_expect_str(_get(d, "data", path=path), path=f"{path}.data"),
            out_dir=_expect_str(_get(d, "out_dir", path=path), path=f"{path}.out_dir"),
        )


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int
    pin_memory: bool

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "DataLoaderConfig":
        return DataLoaderConfig(
            batch_size=_expect_int(_get(d, "batch_size", path=path), path=f"{path}.batch_size"),
            shuffle=_expect_bool(_get(d, "shuffle", path=path), path=f"{path}.shuffle"),
            drop_last=_expect_bool(_get(d, "drop_last", path=path), path=f"{path}.drop_last"),
            num_workers=_expect_int(_get(d, "num_workers", path=path), path=f"{path}.num_workers"),
            pin_memory=_expect_bool(_get(d, "pin_memory", path=path), path=f"{path}.pin_memory"),
        )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    grad_clip: float

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "TrainConfig":
        return TrainConfig(
            epochs=_expect_int(_get(d, "epochs", path=path), path=f"{path}.epochs"),
            grad_clip=_expect_float(_get(d, "grad_clip", path=path), path=f"{path}.grad_clip"),
        )


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "OptimConfig":
        return OptimConfig(
            lr=_expect_float(_get(d, "lr", path=path), path=f"{path}.lr"),
            weight_decay=_expect_float(_get(d, "weight_decay", path=path), path=f"{path}.weight_decay"),
        )


@dataclass(frozen=True)
class GMMModelConfig:
    # Placeholder for future knobs; keep typed for stability
    dummy: int = 0

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "GMMModelConfig":
        dummy = _expect_int(_get_opt(d, "dummy", 0), path=f"{path}.dummy")
        return GMMModelConfig(dummy=dummy)


@dataclass(frozen=True)
class MarkovModelConfig:
    emb_dim: int
    num_components: int

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "MarkovModelConfig":
        return MarkovModelConfig(
            emb_dim=_expect_int(_get(d, "emb_dim", path=path), path=f"{path}.emb_dim"),
            num_components=_expect_int(_get(d, "num_components", path=path), path=f"{path}.num_components"),
        )


@dataclass(frozen=True)
class LatentModelConfig:
    emb_dim: int
    num_components: int
    encoder_hidden: int
    elbo_samples: int

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "LatentModelConfig":
        return LatentModelConfig(
            emb_dim=_expect_int(_get(d, "emb_dim", path=path), path=f"{path}.emb_dim"),
            num_components=_expect_int(_get(d, "num_components", path=path), path=f"{path}.num_components"),
            encoder_hidden=_expect_int(_get(d, "encoder_hidden", path=path), path=f"{path}.encoder_hidden"),
            elbo_samples=_expect_int(_get(d, "elbo_samples", path=path), path=f"{path}.elbo_samples"),
        )


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    gmm: GMMModelConfig
    markov: MarkovModelConfig
    latent: LatentModelConfig

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, path: str) -> "ModelConfig":
        kind = _expect_str(_get(d, "kind", path=path), path=f"{path}.kind")
        if kind not in ("gmm", "markov", "latent"):
            raise ValueError(f"{path}.kind must be one of gmm|markov|latent, got {kind!r}")

        gmm_d = _as_mapping(_get(d, "gmm", path=path), path=f"{path}.gmm")
        markov_d = _as_mapping(_get(d, "markov", path=path), path=f"{path}.markov")
        latent_d = _as_mapping(_get(d, "latent", path=path), path=f"{path}.latent")

        return ModelConfig(
            kind=kind,
            gmm=GMMModelConfig.from_dict(gmm_d, path=f"{path}.gmm"),
            markov=MarkovModelConfig.from_dict(markov_d, path=f"{path}.markov"),
            latent=LatentModelConfig.from_dict(latent_d, path=f"{path}.latent"),
        )


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    dataloader: DataLoaderConfig
    train: TrainConfig
    optim: OptimConfig
    model: ModelConfig

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AppConfig":
        run_d = _as_mapping(_get(d, "run", path="config"), path="config.run")
        dl_d = _as_mapping(_get(d, "dataloader", path="config"), path="config.dataloader")
        train_d = _as_mapping(_get(d, "train", path="config"), path="config.train")
        optim_d = _as_mapping(_get(d, "optim", path="config"), path="config.optim")
        model_d = _as_mapping(_get(d, "model", path="config"), path="config.model")

        return AppConfig(
            run=RunConfig.from_dict(run_d, path="config.run"),
            dataloader=DataLoaderConfig.from_dict(dl_d, path="config.dataloader"),
            train=TrainConfig.from_dict(train_d, path="config.train"),
            optim=OptimConfig.from_dict(optim_d, path="config.optim"),
            model=ModelConfig.from_dict(model_d, path="config.model"),
        )


def _load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    d = _as_mapping(raw, path="config")
    return AppConfig.from_dict(d)


def _make_model(cfg: ModelConfig, num_sources: int) -> nn.Module:
    if cfg.kind == "gmm":
        return JointGaussianModel(num_sources=num_sources)

    if cfg.kind == "markov":
        return MarkovModel(
            num_sources=num_sources,
            emb_dim=cfg.markov.emb_dim,
            num_components=cfg.markov.num_components,
        )

    if cfg.kind == "latent":
        # Assumes your LatentModel __init__ supports encoder_hidden; if not, remove it.
        return LatentModel(
            num_sources=num_sources,
            emb_dim=cfg.latent.emb_dim,
            num_components=cfg.latent.num_components,
            encoder_hidden=cfg.latent.encoder_hidden,
        )

    raise ValueError(f"Unknown model kind: {cfg.kind!r}")


def _train_step(
    model: nn.Module,
    kind: str,
    source: torch.Tensor,
    upstream_speed: torch.Tensor,
    downstream_speed: torch.Tensor,
    elbo_samples: int,
) -> torch.Tensor:
    if kind == "gmm":
        dist = model(source)
        y = torch.stack([upstream_speed, downstream_speed], dim=-1)  # (B, 2)
        nll = -dist.log_prob(y)  # (B,)
        return nll.mean()

    if kind == "markov":
        up_dist, down_dist = model(source, upstream_speed)
        u = upstream_speed.unsqueeze(-1)  # (B, 1)
        d = downstream_speed.unsqueeze(-1)  # (B, 1)
        nll_u = -up_dist.log_prob(u)  # (B,)
        nll_d = -down_dist.log_prob(d)  # (B,)
        return (nll_u + nll_d).mean()

    if kind == "latent":
        return model.loss(source, upstream_speed, downstream_speed, num_samples=elbo_samples)

    raise ValueError(f"Unknown kind: {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_config(args.config)

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

    model = _make_model(cfg.model, num_sources=num_sources).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in dl:
            source = batch.source.to(device)
            upstream_speed = batch.upstream_speed.to(device)
            downstream_speed = batch.downstream_speed.to(device)

            opt.zero_grad(set_to_none=True)
            loss = _train_step(
                model=model,
                kind=cfg.model.kind,
                source=source,
                upstream_speed=upstream_speed,
                downstream_speed=downstream_speed,
                elbo_samples=cfg.model.latent.elbo_samples,
            )
            loss.backward()

            if cfg.train.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip))

            opt.step()

            running += float(loss.detach().item())
            n_batches += 1
            step += 1

        avg = running / max(1, n_batches)
        print(f"epoch={epoch + 1}/{cfg.train.epochs} loss={avg:.6f}")

    payload = {
        "config": {
            "run": cfg.run.__dict__,
            "dataloader": cfg.dataloader.__dict__,
            "train": cfg.train.__dict__,
            "optim": cfg.optim.__dict__,
            "model": {
                "kind": cfg.model.kind,
                "gmm": cfg.model.gmm.__dict__,
                "markov": cfg.model.markov.__dict__,
                "latent": cfg.model.latent.__dict__,
            },
        },
        "num_sources": num_sources,
        "state_dict": model.state_dict(),
    }

    torch.save(payload, out_dir / "model.pt")
    with (out_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(payload["config"], f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
