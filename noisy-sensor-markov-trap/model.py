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
    def __init__(self, num_sources: int):
        super().__init__()
        self.num_sources = num_sources
        self.gmm = zuko.GMM(
            features=2,
            context=num_sources,
            components=1,
            covariance_type="full",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor) -> Distribution:
        return self.gmm(context=nn.functional.one_hot(source, num_classes=self.num_sources).float())


class MarkovModel(nn.Module):
    def __init__(self, num_sources: int):
        super().__init__()
        self.num_sources = num_sources
        self.upstream = zuko.GMM(
            features=1,
            context=num_sources,
            components=1,
            covariance_type="full",
            epsilon=1e-6,
        )
        self.downstream = zuko.GMM(
            features=1,
            context=num_sources + 1,
            components=1,
            covariance_type="full",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor) -> Distribution:
        source_feat = nn.functional.one_hot(source, num_classes=self.num_sources).float()
        upstream = self.upstream(context=source_feat)
        upstream_sample = upstream.rsample()  # Use rsample to allow gradients to flow
        downstream = self.downstream(context=torch.cat([source_feat, upstream_sample], dim=-1))
        return upstream, downstream  # TODO
