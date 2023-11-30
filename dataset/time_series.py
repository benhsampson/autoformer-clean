from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class TimeSeriesDatasetConfig:
    seq_len: int
    label_len: int
    pred_len: int
    scale: bool  # whether to scale the data to be zero mean and unit variance
    highest_freq: Literal['hour', 'minute']
    task: Literal[
        'S', 'M', 'MS'
    ]  # S: uni-to-univariate, M: multi-to-multivariate, MS: multi-to-univariate
    target: Optional[str]  # name of the target column in S or MS tasks


def parse_config(cfg: TimeSeriesDatasetConfig):
    assert cfg.seq_len > 0
    assert cfg.label_len > 0
    assert cfg.pred_len > 0
    assert cfg.label_len <= cfg.seq_len
    if cfg.task in {'S', 'MS'}:
        assert cfg.target is not None
    else:
        assert cfg.target is None


Period = Literal['month', 'day', 'weekday', 'hour', 'minute']
PERIODS: list[Period] = ['month', 'day', 'weekday', 'hour', 'minute']


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        cfg: TimeSeriesDatasetConfig,
        df: pd.DataFrame,
        date_col: str = 'date',
    ):
        """Initialize a dataset for time series forecasting.

        Args:
            df: A pandas dataset with columns [date_col, ..]
                where can be converted to datetime using
                `pd.to_datetime(df[date_col])`.
            cfg: Config class for the dataset.
            date_col: The name of the date column in df.
        """
        parse_config(cfg)

        # split date into periodic intervals
        df.loc[:, '_date'] = pd.to_datetime(df[date_col])
        dt = df['_date'].dt
        df.loc[:, 'month'] = dt.month
        df.loc[:, 'day'] = dt.day
        df.loc[:, 'weekday'] = dt.weekday
        df.loc[:, 'hour'] = dt.hour
        df.loc[:, 'minute'] = dt.minute
        df.drop(columns=[date_col, '_date'], inplace=True)

        # split into into temporal and non-temporal (main) features
        df_main = df.drop(columns=PERIODS)
        periods = PERIODS[: PERIODS.index(cfg.highest_freq) + 1]
        df_time = df[periods]

        data = df_main.values.astype(np.float32)
        data_time = df_time.values

        if cfg.scale:
            scaler = StandardScaler()
            scaler.fit(data)
            data = scaler.transform(data)

        self.data = data
        self.data_time = data_time
        self.cfg = cfg

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gets the input and output sequences from an index.

        Args:
            index: Index the input sequence starts at.

        Returns:
            (x, y, x_time, y_time) where
                x is the input sequence w/ shape (seq_len, C),
                y is the output sequence w/ shape (label_len+pred_len, C),
                x_time is the time features for x w/ shape (seq_len, P),
                y_time is the time features for y w/ shape (label_len+pred_len, P),
                with C being the number of channels and P being the number of periods.

        * The last label_len elements of x == the first label_len elements of y.
        i.e. x[-label_len:] == y[:label_len].
        A prediction task may involve a decoder predicting y[-pred_len:] given x[-label_len:]
        but the encoder may have access to all of x for greater context.
        """
        x_begin = index
        x_end = x_begin + self.cfg.seq_len
        y_begin = x_end - self.cfg.label_len
        y_end = y_begin + self.cfg.label_len + self.cfg.pred_len

        seq_x = self.data[x_begin:x_end]
        seq_y = self.data[y_begin:y_end]
        seq_x_time = self.data_time[x_begin:x_end]
        seq_y_time = self.data_time[y_begin:y_end]

        return seq_x, seq_y, seq_x_time, seq_y_time

    def __len__(self) -> int:
        return len(self.data) - self.cfg.seq_len - self.cfg.pred_len + 1
