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
    "dokter": ["dokter", "nama_dokter", "nama dokter", "doctor"],
    "spesialis": ["spesialis", "spesialist", "specialist", "dokter_spesialis"],
    "jenis_rawat": ["jenis_rawat", "jenis_perawatan", "tipe_rawat", "care_type"],
    "qty_keluar": ["qty_keluar", "qty_total", "qty", "jumlah_keluar"],
    "kategori_pareto": ["kategori_pareto", "kategori_abc", "pareto", "abc"],
}
STD_COLS = list(COLUMN_ALIASES)
REQUIRED_COLS = {"tanggal", "id_obat", "qty_keluar", "kategori_pareto"}


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
            if std in REQUIRED_COLS:
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
    # Konteks klinis opsional supaya dataset lama tetap dapat digunakan.
    for optional in ("dokter", "spesialis", "jenis_rawat"):
        if optional not in df.columns:
            df[optional] = ""
    return df[STD_COLS]


def clean_raw(df: pd.DataFrame, group_supplier: bool = True) -> pd.DataFrame:
    """Sanitasi: buang header duplikat, standarisasi tipe data, proteksi ID & Pareto NULL."""
    df = _standardize_columns(df)

    # Rule 1.1: buang baris header yang berulang di tengah data.
    is_header = df["tanggal"].astype(str).str.strip().str.lower() == "tanggal"
    df = df[~is_header].copy()

    df["tanggal"] = pd.to_datetime(df["tanggal"].str.strip(), errors="coerce")
    df.dropna(subset=["tanggal"], inplace=True)

    # Standarisasi nama obat (identitas utama)
    df["nama_obat"] = df["nama_obat"].astype(str).str.strip()
    df["dokter"] = df["dokter"].astype(str).str.strip().str.upper()
    df["spesialis"] = df["spesialis"].astype(str).str.strip().str.upper()
    df["jenis_rawat"] = df["jenis_rawat"].astype(str).str.strip().str.upper()

    # Simpan identitas asli sebelum penggabungan
    df["id_asli"] = df["id_obat"]
    df["nama_asli"] = df["nama_obat"]

    # Proteksi ID obat: jika ID kosong/NULL/NaN, gunakan nama_obat
    id_str = df["id_obat"].astype(str).str.strip()
    id_empty = df["id_obat"].isna() | (id_str == "") | (id_str.str.lower().isin(["nan", "none", "null"]))
    df.loc[id_empty, "id_obat"] = df.loc[id_empty, "nama_obat"]

    # Identitas obat diutamakan nama gabungan (BARANG)
    if group_supplier:
        df["nama_obat"] = df["nama_obat"].map(base_name)
        df["id_obat"] = df["nama_obat"]

    # Proteksi kategori_pareto kosong: isi 'C' & catat jumlah obat terdampak
    raw_pareto = df["kategori_pareto"].astype(str).str.strip().str.upper()
    pareto_empty = (
        df["kategori_pareto"].isna()
        | (raw_pareto == "")
        | (raw_pareto.isin(["NAN", "NONE", "NULL"]))
    )
    n_empty_pareto = int(pareto_empty.sum())
    empty_drugs_count = int(df.loc[pareto_empty, "nama_obat"].nunique()) if n_empty_pareto > 0 else 0
    if n_empty_pareto > 0:
        import warnings
        warnings.warn(
            f"Terdapat {n_empty_pareto} baris ({empty_drugs_count} obat) dengan kategori Pareto kosong/NULL. Otomatis diisi 'C'.",
            UserWarning,
        )
        raw_pareto = raw_pareto.mask(pareto_empty, "C")
    df["kategori_pareto"] = raw_pareto

    qty = pd.to_numeric(df["qty_keluar"], errors="coerce").fillna(0.0)
    df["qty_keluar"] = qty.clip(lower=0).astype("float32")  # retur negatif -> 0

    res = df.sort_values(["tanggal", "id_obat"]).reset_index(drop=True)
    res.attrs["empty_pareto_drugs"] = empty_drugs_count
    res.attrs["empty_pareto_rows"] = n_empty_pareto
    return res


def build_daily(clean: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: lookup Pareto, agregasi harian, complete date grid (zero-fill)."""
    # 2.1 Label Pareto dari varian dengan total volume terbesar per obat
    vol_grouped = (
        clean.groupby(["id_obat", "nama_obat", "kategori_pareto"], as_index=False)["qty_keluar"]
        .sum()
        .sort_values("qty_keluar")
    )

    # Deteksi apakah ada obat yang memiliki >1 label Pareto
    pareto_counts = vol_grouped.groupby("id_obat")["kategori_pareto"].nunique()
    n_multi_pareto = int((pareto_counts > 1).sum())
    if n_multi_pareto > 0:
        import warnings
        warnings.warn(
            f"Terdapat {n_multi_pareto} obat dengan >1 label kategori Pareto. Menggunakan label dengan volume terbesar.",
            UserWarning,
        )

    ref = vol_grouped.drop_duplicates("id_obat", keep="last")[["id_obat", "nama_obat", "kategori_pareto"]]

    # 2.2 Agregasi harian. Konteks klinis dipertahankan sebagai sinyal
    # historis; fitur lag/rolling-nya dibentuk di src/features.py.
    keys = ["tanggal", "id_obat"]
    agg = clean.groupby(keys, as_index=False)["qty_keluar"].sum()

    ctx = clean.groupby(keys, as_index=False).agg(
        ctx_n_dokter=("dokter", lambda x: x[x.ne("")].nunique()),
        ctx_n_spesialis=("spesialis", lambda x: x[x.ne("")].nunique()),
        ctx_n_jenis_rawat=("jenis_rawat", lambda x: x[x.ne("")].nunique()),
    )

    # Jumlah dokter/spesialis aktif pada seluruh data tanggal tersebut.
    # Nilai ini diulang untuk setiap obat pada tanggal yang sama, lalu hanya
    # versi lag/rolling-nya yang digunakan model.
    daily_activity = clean.groupby("tanggal", as_index=False).agg(
        ctx_daily_n_dokter=("dokter", lambda x: x[x.ne("")].nunique()),
        ctx_daily_n_spesialis=("spesialis", lambda x: x[x.ne("")].nunique()),
    )
    care_qty = clean.pivot_table(
        index=keys, columns="jenis_rawat", values="qty_keluar", aggfunc="sum", fill_value=0
    ).reset_index()
    care_qty = care_qty.rename(columns={
        "INSTALASI RAWAT INAP": "ctx_qty_rawat_inap",
        "INSTALASI RAWAT JALAN": "ctx_qty_rawat_jalan",
        "INSTALASI GAWAT DARURAT": "ctx_qty_gawat_darurat",
    })
    ctx = ctx.merge(care_qty, on=keys, how="left")

    specialist_qty = (
        clean[clean["spesialis"].ne("")]
        .groupby(keys + ["spesialis"], as_index=False)["qty_keluar"].sum()
    )
    top_specialist = specialist_qty.groupby(keys, as_index=False)["qty_keluar"].max()
    top_specialist = top_specialist.rename(columns={"qty_keluar": "ctx_qty_top_spesialis"})
    ctx = ctx.merge(top_specialist, on=keys, how="left")

    # 2.3 Grid tanggal lengkap per obat, isi 0, lalu merge ulang label Pareto.
    dates = pd.date_range(agg["tanggal"].min(), agg["tanggal"].max(), freq="D")
    idx = pd.MultiIndex.from_product([dates, ref["id_obat"].tolist()], names=["tanggal", "id_obat"])
    full = (
        agg.set_index(keys)
        .reindex(idx, fill_value=0.0)
        .reset_index()
        .merge(ref, on="id_obat", how="left")
        .merge(ctx, on=keys, how="left")
        .merge(daily_activity, on="tanggal", how="left")
    )
    full["qty_keluar"] = full["qty_keluar"].astype("float32")
    context_cols = [
        "ctx_n_dokter", "ctx_n_spesialis", "ctx_n_jenis_rawat",
        "ctx_daily_n_dokter", "ctx_daily_n_spesialis", "ctx_qty_rawat_inap",
        "ctx_qty_rawat_jalan", "ctx_qty_gawat_darurat", "ctx_qty_top_spesialis",
    ]
    for col in context_cols:
        if col not in full:
            full[col] = 0.0
    full[context_cols] = full[context_cols].fillna(0).astype("float32")
    # Date grid berada pada level obat, jadi kategori teks tidak lagi unik per
    # baris. Simpan kolomnya kosong dan gunakan agregat numerik di atas.
    full["dokter"] = ""
    full["spesialis"] = ""
    full["jenis_rawat"] = ""
    full = full[STD_COLS + context_cols].sort_values(["id_obat", "tanggal"]).reset_index(drop=True)

    # Teruskan metadata
    full.attrs["empty_pareto_drugs"] = clean.attrs.get("empty_pareto_drugs", 0)
    full.attrs["empty_pareto_rows"] = clean.attrs.get("empty_pareto_rows", 0)
    full.attrs["multi_pareto_drugs"] = n_multi_pareto
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
