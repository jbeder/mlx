from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpeedBatch(NamedTuple):
    source: torch.Tensor
    upstream_speed: torch.Tensor
    downstream_speed: torch.Tensor


class SpeedDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.source = torch.as_tensor(data["source"].to_numpy(), dtype=torch.long)
        self.upstream_speed = torch.as_tensor(data["upstream_speed"].to_numpy(), dtype=torch.float)
        self.downstream_speed = torch.as_tensor(data["downstream_speed"].to_numpy(), dtype=torch.float)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return SpeedBatch(
            source=self.source[idx],
            upstream_speed=self.upstream_speed[idx],
            downstream_speed=self.downstream_speed[idx],
        )
