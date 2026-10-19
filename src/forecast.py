"""Recursive multi-step forecasting (7-14 hari)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import FEATURES, build_features


def recursive_forecast(model, daily: pd.DataFrame, horizon: int, id_map: dict) -> pd.DataFrame:
    """Prediksi `horizon` hari setelah tanggal terakhir di `daily`.

    Tiap hari diprediksi, lalu hasilnya dipakai sebagai lag/rolling hari berikutnya.
    """
    hist = daily[["tanggal", "id_obat", "nama_obat", "kategori_pareto", "qty_keluar"]].copy()
    meta = hist.drop_duplicates("id_obat")[["id_obat", "nama_obat", "kategori_pareto"]]
    last = hist["tanggal"].max()
    out = []

    for d in range(1, horizon + 1):
        day = last + pd.Timedelta(days=d)
        with warnings.catch_warnings():  # kolom qty baris baru sengaja NaN (diisi prediksi)
            warnings.simplefilter("ignore", FutureWarning)
            hist = pd.concat(
                [hist, meta.assign(tanggal=day, qty_keluar=np.nan)], ignore_index=True
            )
        feat = build_features(hist, id_map)
        row = feat[feat["tanggal"] == day]
        pred = np.clip(model.predict(row[FEATURES]), 0, None)
        pred = np.where(pred < 0.01, 0.0, pred)
        pmap = dict(zip(row["id_obat"], pred))
        mask = hist["tanggal"] == day
        hist.loc[mask, "qty_keluar"] = hist.loc[mask, "id_obat"].map(pmap).astype("float32")
        out.append(
            meta.assign(tanggal=day, prediksi=meta["id_obat"].map(pmap).values)
        )

    res = pd.concat(out, ignore_index=True)
    return res[["tanggal", "id_obat", "nama_obat", "kategori_pareto", "prediksi"]]


def baseline_forecast(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Baseline: rata-rata 7 hari terakhir dipakai datar untuk seluruh horizon."""
    last = daily["tanggal"].max()
    cut = daily[daily["tanggal"] > last - pd.Timedelta(days=7)]
    m = cut.groupby("id_obat")["qty_keluar"].mean().rename("prediksi").reset_index()
    meta = daily.drop_duplicates("id_obat")[["id_obat", "nama_obat", "kategori_pareto"]]
    rows = [
        meta.merge(m, on="id_obat").assign(tanggal=last + pd.Timedelta(days=d))
        for d in range(1, horizon + 1)
    ]
    return pd.concat(rows, ignore_index=True)
