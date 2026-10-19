"""Feature engineering (Rule 3 & 4) - batas lag/rolling maksimal 28 hari."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 21]
PARETO_MAP = {"A": 2, "B": 1, "C": 0}

FEATURES = (
    [f"lag_{l}" for l in LAGS]
    + [
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_max_14",
        "rolling_mean_28",
    ]
    + ["day_of_week", "is_weekend", "day_of_month"]
    + ["id_obat_encoded", "kategori_pareto_encoded"]
)


def make_id_map(ids) -> dict:
    return {i: n for n, i in enumerate(sorted(set(ids)))}


def build_features(df: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    """df: kolom tanggal, id_obat, kategori_pareto, qty_keluar (boleh NaN di masa depan)."""
    df = df.sort_values(["id_obat", "tanggal"]).reset_index(drop=True).copy()
    g = df.groupby("id_obat")["qty_keluar"]

    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)

    # Anti-leakage: .shift(1) SEBELUM .rolling()
    def roll(fn, w, mp):
        return g.transform(lambda x: getattr(x.shift(1).rolling(w, min_periods=mp), fn)())

    df["rolling_mean_7"] = roll("mean", 7, 3)
    df["rolling_std_7"] = roll("std", 7, 3)
    df["rolling_mean_14"] = roll("mean", 14, 3)
    df["rolling_max_14"] = roll("max", 14, 3)
    df["rolling_mean_28"] = roll("mean", 28, 3)

    df["day_of_week"] = df["tanggal"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["tanggal"].dt.day

    df["id_obat_encoded"] = df["id_obat"].map(id_map)
    df["kategori_pareto_encoded"] = df["kategori_pareto"].map(PARETO_MAP).fillna(0).astype(int)
    return df
