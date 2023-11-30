import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, random_split

from dataset.ett_small import ETTSmall_Dataset, ETTSmall_DatasetConfig
from dataset.time_series import TimeSeriesDatasetConfig
from models.Autoformer import AutoformerConfig
from models.Autoformer import Model as Autoformer
from models.model import ModelConfig
from utils.early_stopping import Checkpoint, EarlyStopping


@dataclass
class TrainConfig:
    model: Literal['autoformer']
    model_cfg: ModelConfig
    dataset: ETTSmall_Dataset
    splits: tuple[float, float, float]  # % train, % test, % val
    loss_fn: Literal['mse']
    optimizer: Literal['adam']
    lradj: Literal['type1', 'type2']
    use_gpu: bool
    checkpoints_path: Union[str, Path]
    train_epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    delta: float


def parse_config(cfg: TrainConfig):
    assert cfg.train_epochs > 0
    assert cfg.learning_rate > 0.0
    assert cfg.batch_size > 0
    assert cfg.patience > 0
    assert cfg.delta >= 0.0


def calc_metrics(preds: Tensor, trues: Tensor):
    return {
        'mse': nn.MSELoss()(preds, trues).item(),
        'mae': nn.L1Loss()(preds, trues).item(),
    }


class Trainer:
    def __init__(self, cfg: TrainConfig):
        parse_config(cfg)
        self.cfg = cfg
        self.device = self.create_device()

        if cfg.model == 'autoformer':
            assert isinstance(cfg.model_cfg, AutoformerConfig)
            self.model = Autoformer(cfg.model_cfg)
        assert self.model is not None
        self.model = self.model.to(self.device)

        self.dataset = cfg.dataset

        if cfg.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        assert self.loss_fn is not None

        if cfg.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), cfg.learning_rate)
        assert self.optimizer is not None

        self.train_loader, self.test_loader, self.val_loader = self.prepare_loaders()

        # TODO: add setting to path
        self.checkpoints_path = Path(self.cfg.checkpoints_path)

    def create_device(self) -> torch.device:
        return torch.device(
            'cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu'
        )

    def prepare_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train, test, val = random_split(self.dataset, self.cfg.splits)
        train_loader = DataLoader(
            train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(test, batch_size=self.cfg.batch_size, shuffle=False)
        val_loader = DataLoader(
            val, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        print(f'train: {len(train)} | test: {len(test)} | val: {len(val)}')
        return train_loader, test_loader, val_loader

    def predict(
        self, x: Tensor, y: Tensor, x_time: Tensor, y_time: Tensor
    ) -> tuple[Tensor, Tensor]:
        label_len = self.cfg.model_cfg.label_len
        pred_len = self.cfg.model_cfg.pred_len
        target = self.dataset.target_index

        B, len_x, C = x.shape
        _, len_y, _ = y.shape
        _, _, P = x_time.shape

        # assert B == y.shape[0] == x_time.shape[0] == y_time.shape[0]
        # assert len_x == x_time.shape[1]
        # assert len_y == y_time.shape[1] == label_len + pred_len
        # assert P == y_time.shape[2]
        # if target is not None:
        #     assert 0 <= target < y.shape[2]

        zeros = torch.zeros((B, pred_len, C)).to(y.device)
        dec_in = torch.cat(
            [y[:, :label_len, :], zeros], dim=1
        )  # (B, label_len + pred_len, C)

        y_pred, _ = self.model(x, x_time, dec_in, y_time)
        target_slice = slice(None) if target is None else target
        y_pred = y_pred[:, -pred_len:, target_slice]  # (B, pred_len, C')
        y_true = y[:, -pred_len:, target_slice]  # (B, pred_len, C')

        return y_pred, y_true

    def adjust_learning_rate(self, epoch: int):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if self.cfg.lradj == 'type1':
            lr_adjust = {epoch: self.cfg.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif self.cfg.lradj == 'type2':
            lr_adjust = {
                2: 5e-5,
                4: 1e-5,
                6: 5e-6,
                8: 1e-6,
                10: 5e-7,
                15: 1e-7,
                20: 5e-8,
            }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('updating learning rate to {}'.format(lr))

    @torch.no_grad()
    def eval(self, loader: DataLoader):
        self.model.eval()

        losses = []

        for _, (x, y, x_mark, y_mark) in enumerate(loader):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            x_mark = x_mark.float().to(self.device)
            y_mark = y_mark.float().to(self.device)

            y_pred, y_true = self.predict(x, y, x_mark, y_mark)
            loss = self.loss_fn(y_pred, y_true)
            losses.append(loss.item())

        loss_avg = np.mean(losses)

        self.model.train()
        return loss_avg

    def train(self):
        if not self.checkpoints_path.exists():
            print('checkpoints dir does not exist, creating...')
            self.checkpoints_path.mkdir(parents=True)

        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(
            self.model,
            self.optimizer,
            checkpoints_dir=str(self.checkpoints_path),
            patience=self.cfg.patience,
            delta=self.cfg.delta,
            verbose=True,
        )

        for epoch in range(self.cfg.train_epochs):
            self.model.train()

            iter_count = 0
            epoch_time = time.time()
            train_losses = []
            for i, (x, y, x_mark, y_mark) in enumerate(self.train_loader):
                iter_count += 1

                self.optimizer.zero_grad()

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                y_pred, y_true = self.predict(x, y, x_mark, y_mark)

                loss = self.loss_fn(y_pred, y_true)
                train_losses.append(loss.item())

                if i == 0 or (i + 1) % 100 == 0:
                    print(
                        f'iter: {i + 1} | epoch: {epoch + 1} | loss {loss.item():.4f}'
                    )
                    speed = (time.time() - time_now) / iter_count
                    time_left = speed * (
                        (self.cfg.train_epochs - epoch) * train_steps - i
                    )
                    print(f'speed: {speed:.2f} s/iter | left: {time_left:.2f} s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            print(f'epoch: {epoch + 1} | took {time.time() - epoch_time:.2f} s')
            train_loss = np.mean(train_losses)
            val_loss = self.eval(self.val_loader)
            test_loss = self.eval(self.test_loader)
            print(
                f'epoch: {epoch + 1} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | test loss: {test_loss:.4f}'
            )
            early_stopping(epoch + 1, val_loss)
            if early_stopping.early_stop:
                print('! early stopping')
                break
            self.adjust_learning_rate(epoch + 1)

    @torch.no_grad()
    def test(self):
        best_model_path = self.checkpoints_path / 'checkpoint.pt'
        best_model: Checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(best_model['model_state_dict'])

        self.model.eval()

        preds, trues = [], []

        for _, (x, y, x_time, y_time) in enumerate(self.test_loader):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            x_time = x_time.float().to(self.device)
            y_time = y_time.float().to(self.device)

            y_pred, y_true = self.predict(x, y, x_time, y_time)
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()
            preds.append(y_pred)
            trues.append(y_true)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        results = calc_metrics(preds, trues)
        print(results)
