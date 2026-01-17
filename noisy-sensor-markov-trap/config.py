# configs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunConfig:
    seed: int
    device: str
    data: str
    out_dir: str


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    grad_clip: float


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class GMMModelConfig:
    dummy: int


@dataclass(frozen=True)
class MarkovModelConfig:
    emb_dim: int
    num_components: int


@dataclass(frozen=True)
class LatentModelConfig:
    emb_dim: int
    num_components: int
    encoder_hidden: int
    elbo_samples: int


@dataclass(frozen=True)
class ModelConfig:
    kind: Literal["gmm", "markov", "latent"]
    gmm: GMMModelConfig
    markov: MarkovModelConfig
    latent: LatentModelConfig


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    dataloader: DataLoaderConfig
    train: TrainConfig
    optim: OptimConfig
    model: ModelConfig
