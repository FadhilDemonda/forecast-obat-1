"""Load, sanitize, and zero-fill hospital drug-expenditure CSV (Rule 1 & 2)."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

# Nama kolom standar internal -> kemungkinan nama di CSV (setelah dinormalisasi).
COLUMN_ALIASES = {
    "tanggal": ["tanggal", "date", "tgl"],
    "id_obat": ["id_obat", "id_barang", "idobat", "kode_obat", "kode_barang"],
    "nama_obat": ["nama_obat", "barang", "nama_barang", "nama"],
    "qty_keluar": ["qty_keluar", "qty_total", "qty", "jumlah_keluar"],
    "kategori_pareto": ["kategori_pareto", "kategori_abc", "pareto", "abc"],
}
STD_COLS = list(COLUMN_ALIASES)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def base_name(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name).strip().upper())
    stripped = re.sub(r"\s*\([^()]*\)\s*$", "", n).strip()
    return stripped or n


def _read_csv(source) -> pd.DataFrame:
    if isinstance(source, bytes):
        raw = source.decode("utf-8-sig", errors="replace")
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
    else:
        with open(source, encoding="utf-8-sig", errors="replace") as f:
            raw = f.read()
    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)

    # Rule 1.2: file terbaca sebagai satu kolom teks -> pecah dengan koma.
    if df.shape[1] == 1:
        lines = [df.columns[0]] + df.iloc[:, 0].tolist()
        parts = pd.Series(lines).str.split(",", expand=True)
        df = parts.iloc[1:].copy()
        df.columns = parts.iloc[0].tolist()
        df = df.reset_index(drop=True)
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normed = {_norm(c): c for c in df.columns}
    rename, missing = {}, []
    for std, aliases in COLUMN_ALIASES.items():
        hit = next((normed[a] for a in aliases if a in normed), None)
        if hit is None:
            if std != "nama_obat":
                missing.append(std)
        else:
            rename[hit] = std
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}. Kolom di file: {list(df.columns)}"
        )
    df = df.rename(columns=rename)
    if "nama_obat" not in df.columns:
        df["nama_obat"] = df["id_obat"]
    return df[STD_COLS]


def clean_raw(df: pd.DataFrame, group_supplier: bool = True) -> pd.DataFrame:
    """Sanitasi: buang header duplikat, standarisasi tipe data."""
    df = _standardize_columns(df)

    # Rule 1.1: buang baris header yang berulang di tengah data.
    is_header = df["tanggal"].astype(str).str.strip().str.lower() == "tanggal"
    df = df[~is_header].copy()

    df["tanggal"] = pd.to_datetime(df["tanggal"].str.strip(), errors="coerce")
    df.dropna(subset=["tanggal"], inplace=True)
    df["id_obat"] = df["id_obat"].astype(str).str.strip()
    df["nama_obat"] = df["nama_obat"].astype(str).str.strip()
    df["kategori_pareto"] = df["kategori_pareto"].astype(str).str.strip().str.upper()
    qty = pd.to_numeric(df["qty_keluar"], errors="coerce").fillna(0.0)
    df["qty_keluar"] = qty.clip(lower=0).astype("float32")  # retur negatif -> 0

    df["id_asli"] = df["id_obat"]
    df["nama_asli"] = df["nama_obat"]
    if group_supplier:
        df["nama_obat"] = df["nama_obat"].map(base_name)
        df["id_obat"] = df["nama_obat"]

    return df.sort_values(["tanggal", "id_obat"]).reset_index(drop=True)


def build_daily(clean: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: lookup Pareto, agregasi harian, complete date grid (zero-fill)."""
    # 2.1 Label Pareto dari varian dengan total volume terbesar per obat
    ref = (
        clean.groupby(["id_obat", "nama_obat", "kategori_pareto"], as_index=False)["qty_keluar"].sum()
        .sort_values("qty_keluar")
        .drop_duplicates("id_obat", keep="last")[["id_obat", "nama_obat", "kategori_pareto"]]
    )

    # 2.2 Agregasi harian.
    agg = clean.groupby(["tanggal", "id_obat"], as_index=False)["qty_keluar"].sum()

    # 2.3 Grid tanggal lengkap per obat, isi 0, lalu merge ulang label Pareto.
    dates = pd.date_range(agg["tanggal"].min(), agg["tanggal"].max(), freq="D")
    idx = pd.MultiIndex.from_product([dates, ref["id_obat"].tolist()], names=["tanggal", "id_obat"])
    full = (
        agg.set_index(["tanggal", "id_obat"])
        .reindex(idx, fill_value=0.0)
        .reset_index()
        .merge(ref, on="id_obat", how="left")
    )
    full["qty_keluar"] = full["qty_keluar"].astype("float32")
    full = full[STD_COLS].sort_values(["id_obat", "tanggal"]).reset_index(drop=True)
    return full


def load_daily(source, group_supplier: bool = True) -> pd.DataFrame:
    """Pipeline lengkap: CSV -> data harian lengkap per obat."""
    return build_daily(clean_raw(_read_csv(source), group_supplier=group_supplier))


def group_mapping(source) -> pd.DataFrame:
    """Tabel pengecekan penggabungan varian supplier."""
    clean = clean_raw(_read_csv(source), group_supplier=True)
    mapping = (
        clean.groupby(["id_obat", "id_asli", "nama_asli", "kategori_pareto"], as_index=False)["qty_keluar"]
        .sum()
        .rename(columns={"id_obat": "obat_gabungan", "qty_keluar": "total_qty"})
    )
    mapping["total_qty"] = mapping["total_qty"].round(2)
    cols = ["obat_gabungan", "id_asli", "nama_asli", "kategori_pareto", "total_qty"]
    return mapping[cols].sort_values(["obat_gabungan", "total_qty"], ascending=[True, False]).reset_index(drop=True)
