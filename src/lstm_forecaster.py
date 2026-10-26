"""Small CPU LSTM for sales-only, recursive benchmark (not dashboard model)."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 28


def calendar_values(dates) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    return np.column_stack([
        dates.dayofweek / 6, (dates.dayofweek >= 5).astype(float),
        (dates.day - 1) / 30,
    ]).astype("float32")


def sales_matrix(daily: pd.DataFrame, ids=None):
    table = daily.pivot(index="tanggal", columns="id_obat", values="qty_keluar").sort_index()
    if ids is not None:
        table = table.reindex(columns=ids)
    if table.isna().any().any() or not table.index.equals(
        pd.date_range(table.index.min(), table.index.max())
    ):
        raise ValueError("LSTM membutuhkan grid harian lengkap tanpa nilai kosong.")
    return table.to_numpy(dtype="float32").T, table.index, table.columns.tolist()


def sequence_arrays(values, dates, scale, start=LOOKBACK):
    """Target t only sees quantities t-28..t-1; current calendar is known."""
    if len(dates) <= start or start < LOOKBACK:
        raise ValueError("Histori terlalu pendek untuk sequence LSTM.")
    normalized = values / scale[:, None]
    windows = np.lib.stride_tricks.sliding_window_view(normalized, LOOKBACK, axis=1)
    x_qty = windows[:, start - LOOKBACK:len(dates) - LOOKBACK, :]
    cal = calendar_values(dates)
    x_cal = np.stack([cal[t - LOOKBACK:t] for t in range(start, len(dates))])
    x_cal = np.broadcast_to(x_cal, (len(values),) + x_cal.shape)
    x = np.concatenate([x_qty[..., None], x_cal], axis=-1)
    current = np.broadcast_to(cal[start:], (len(values), len(dates) - start, 3))
    ids = np.repeat(np.arange(len(values)), len(dates) - start)
    return (x.reshape(-1, LOOKBACK, 4).copy(), current.reshape(-1, 3).copy(),
            ids, normalized[:, start:].reshape(-1).copy())


class SalesLSTM(nn.Module):
    def __init__(self, n_ids, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(n_ids, 8)
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 11, 32), nn.ReLU(),
                                  nn.Linear(32, 1), nn.Softplus())

    def forward(self, sequence, calendar, ids):
        _, (hidden, _) = self.lstm(sequence)
        return self.head(torch.cat([hidden[-1], calendar, self.embedding(ids)], dim=1)).squeeze(1)


class LSTMForecaster:
    def __init__(self, category, seed=42, hidden_size=32, batch_size=512):
        self.category = category
        self.seed = seed
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def loss(self, prediction, actual):
        if self.category == "A":
            return torch.mean((prediction - actual) ** 2)
        # Tweedie p=1.5, same family as XGBoost categories B/C; constants omitted.
        prediction = prediction.clamp_min(1e-6)
        return torch.mean(2 * actual / prediction.sqrt() + 2 * prediction.sqrt())

    def fit(self, daily, epochs=20, validation_days=0, patience=4):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        values, dates, self.ids = sales_matrix(daily)
        fit_end = len(dates) - validation_days
        self.scale = np.maximum(values[:, :fit_end].mean(axis=1), 1).astype("float32")
        arrays = sequence_arrays(values, dates, self.scale)
        target_dates = np.tile(np.arange(LOOKBACK, len(dates)), len(self.ids))
        train_mask = target_dates < fit_end
        tensors = [torch.from_numpy(a) for a in arrays]
        train_set = TensorDataset(*(t[train_mask] for t in tensors))
        if not len(train_set):
            raise ValueError("Training LSTM kosong.")
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        validation = TensorDataset(*(t[~train_mask] for t in tensors))
        val_loader = DataLoader(validation, batch_size=self.batch_size)
        self.network = SalesLSTM(len(self.ids), self.hidden_size)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        best_loss, stale, self.best_epoch = float("inf"), 0, epochs
        self.validation_history = []
        for epoch in range(1, epochs + 1):
            self.network.train()
            for x, calendar, ids, y in loader:
                optimizer.zero_grad()
                loss = self.loss(self.network(x, calendar, ids), y)
                if not torch.isfinite(loss):
                    raise ValueError("Loss LSTM tidak finite.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
            if validation_days:
                self.network.eval()
                total = 0.0
                with torch.no_grad():
                    for x, calendar, ids, y in val_loader:
                        total += self.loss(self.network(x, calendar, ids), y).item() * len(y)
                score = total / len(validation)
                self.validation_history.append(score)
                if score < best_loss - 1e-6:
                    best_loss, stale, self.best_epoch = score, 0, epoch
                else:
                    stale += 1
                if stale >= patience:
                    break
        return self

    def forecast(self, daily, horizon):
        values, dates, _ = sales_matrix(daily, self.ids)
        history = values / self.scale[:, None]
        metadata = daily.drop_duplicates("id_obat").set_index("id_obat").loc[self.ids]
        rows = []
        self.network.eval()
        for step in range(1, horizon + 1):
            day = dates[-1] + pd.Timedelta(days=step)
            past_dates = pd.date_range(day - pd.Timedelta(days=LOOKBACK), periods=LOOKBACK)
            cal = np.broadcast_to(calendar_values(past_dates), (len(self.ids), LOOKBACK, 3))
            x = np.concatenate([history[:, -LOOKBACK:, None], cal], axis=2).astype("float32")
            current = np.repeat(calendar_values([day]), len(self.ids), axis=0)
            with torch.no_grad():
                normalized = self.network(torch.from_numpy(x), torch.from_numpy(current),
                                          torch.arange(len(self.ids))).numpy()
            prediction = normalized * self.scale
            prediction = np.where(prediction < 0.01, 0.0, prediction)
            history = np.column_stack([history, prediction / self.scale])
            rows.append(pd.DataFrame({
                "tanggal": day, "id_obat": self.ids,
                "kategori_pareto": metadata["kategori_pareto"].to_numpy(),
                "prediksi": prediction,
            }))
        return pd.concat(rows, ignore_index=True)
