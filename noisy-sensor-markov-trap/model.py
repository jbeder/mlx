from typing import NamedTuple

import pandas as pd
import torch
import zuko
from torch import nn
from torch.distributions import Distribution
from torch.utils.data import Dataset


class SpeedBatch(NamedTuple):
    source: torch.Tensor
    upstream_speed: torch.Tensor
    downstream_speed: torch.Tensor


class SpeedDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.source = torch.as_tensor(data["source"].to_numpy(), dtype=torch.long)
        self.upstream_speed = torch.as_tensor(data["upstream_speed"].to_numpy(), dtype=torch.float32)
        self.downstream_speed = torch.as_tensor(data["downstream_speed"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx) -> SpeedBatch:
        return SpeedBatch(
            source=self.source[idx],
            upstream_speed=self.upstream_speed[idx],
            downstream_speed=self.downstream_speed[idx],
        )


class JointGaussianModel(nn.Module):
    def __init__(self, num_sources: int, emb_dim: int = 8, num_components: int = 2):
        super().__init__()
        self.source_emb = nn.Embedding(num_sources, emb_dim)
        self.gmm = zuko.GMM(
            features=2,
            context=emb_dim,
            components=num_components,
            covariance_type="full",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor) -> Distribution:
        ctx = self.source_emb(source)  # (B, emb_dim)
        return self.gmm(ctx)


class MarkovModel(nn.Module):
    def __init__(self, num_sources: int, emb_dim: int = 8, num_components: int = 2):
        super().__init__()
        self.num_sources = num_sources
        self.source_emb = nn.Embedding(num_sources, emb_dim)
        self.upstream = zuko.GMM(
            features=1,
            context=emb_dim,
            components=num_components,
            covariance_type="full",
            epsilon=1e-6,
        )
        self.downstream = zuko.GMM(
            features=1,
            context=emb_dim + 1,
            components=num_components,
            covariance_type="full",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor) -> Distribution:
        source_feat = self.source_emb(source)  # (B, emb_dim)
        upstream = self.upstream(context=source_feat)
        upstream_sample = upstream.rsample()  # Use rsample to allow gradients to flow
        downstream = self.downstream(context=torch.cat([source_feat, upstream_sample], dim=-1))
        return upstream, downstream  # TODO
