"""Recursive multi-step forecasting (7-14 hari)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import CONTEXT_COLS, FEATURES, build_features
from .schedule import attach_schedule, require_schedule_dates
from .calendar_features import CALENDAR_FEATURES, OPTIONAL_FEATURES


def recursive_forecast(model, daily: pd.DataFrame, horizon: int, id_map: dict, schedule=None,
                       calendar_overrides=None) -> pd.DataFrame:
    """Prediksi `horizon` hari setelah tanggal terakhir di `daily`.

    Tiap hari diprediksi, lalu hasilnya dipakai sebagai lag/rolling hari berikutnya.
    """
    keep = ["tanggal", "id_obat", "nama_obat", "kategori_pareto", "qty_keluar"] + CONTEXT_COLS
    hist = daily.copy()
    for col in CONTEXT_COLS:
        if col not in hist.columns:
            hist[col] = 0.0
    hist = hist[keep].copy()
    meta = hist.drop_duplicates("id_obat")[["id_obat", "nama_obat", "kategori_pareto"]]
    last = hist["tanggal"].max()
    feature_names = getattr(model, "feature_names_in_", FEATURES)
    if any(c.startswith("skor_jadwal_") for c in feature_names):
        if schedule is None:
            raise ValueError("Model memakai skor jadwal; tabel jadwal wajib diberikan saat forecast.")
        require_schedule_dates(schedule, pd.date_range(last + pd.Timedelta(days=1), periods=horizon))
    out = []

    # Spesialis/rawat untuk tanggal mendatang belum diketahui. Gunakan
    # estimasi naive berupa rata-rata tujuh hari terakhir per obat agar fitur
    # konteks tidak tiba-tiba berubah menjadi nol pada hari forecast.
    recent = hist[hist["tanggal"] > last - pd.Timedelta(days=7)]
    context_forecast = {
        col: recent.groupby("id_obat")[col].mean()
        for col in CONTEXT_COLS
    }

    for d in range(1, horizon + 1):
        day = last + pd.Timedelta(days=d)
        with warnings.catch_warnings():  # kolom qty baris baru sengaja NaN (diisi prediksi)
            warnings.simplefilter("ignore", FutureWarning)
            future = meta.assign(tanggal=day, qty_keluar=np.nan)
            for col in CONTEXT_COLS:
                future[col] = future["id_obat"].map(context_forecast[col]).fillna(0.0)
            hist = pd.concat([hist, future[keep]], ignore_index=True)
        feat = build_features(hist, id_map,
                              include_calendar=bool(set(feature_names) & set(CALENDAR_FEATURES + OPTIONAL_FEATURES)),
                              calendar_overrides=calendar_overrides)
        row = feat[feat["tanggal"] == day]
        if schedule is not None:
            row = attach_schedule(row, schedule)
        pred = np.clip(model.predict(row[list(feature_names)]), 0, None)
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
