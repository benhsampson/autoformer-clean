from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from dataset.time_series import TimeSeriesDataset, TimeSeriesDatasetConfig


@dataclass
class ETTSmall_DatasetConfig(TimeSeriesDatasetConfig):
    root_path: Union[str, Path]
    data_path: Union[str, Path]


class ETTSmall_Dataset(TimeSeriesDataset):
    def __init__(
        self,
        cfg: ETTSmall_DatasetConfig,
    ):
        root_path = Path(cfg.root_path)
        data_path = Path(cfg.data_path)

        df = pd.read_csv(root_path / data_path)

        super().__init__(cfg, df, date_col='date')

        self.df = df
        self.cfg = cfg

    @property
    def num_features(self) -> int:
        return self.data.shape[1]

    @property
    def target_index(self) -> Optional[int]:
        col2index = {col: i for i, col in enumerate(self.df.columns)}
        return col2index(self.cfg.target) if self.cfg.target is not None else None
