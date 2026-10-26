"""Ringkasan numerik untuk visualisasi evaluasi tanpa melatih ulang model."""
import pandas as pd

# Daftar label milik visualisasi, bukan daftar input model. Jangan mengimpor
# src.features di sini: heatmap hanya perlu membaca kolom yang tersedia dan
# harus tetap dapat dibuka ketika modul model lama masih dimuat Streamlit.
CONTEXT_LABELS = {
    "ctx_n_dokter": "Dokter per obat",
    "ctx_n_spesialis": "Spesialis per obat",
    "ctx_n_jenis_rawat": "Jenis rawat per obat",
    "ctx_daily_n_dokter": "Dokter harian (dataset)",
    "ctx_daily_n_spesialis": "Spesialis harian (dataset)",
    "ctx_qty_rawat_inap": "Qty rawat inap",
    "ctx_qty_rawat_jalan": "Qty rawat jalan",
    "ctx_qty_gawat_darurat": "Qty gawat darurat",
    "ctx_qty_top_spesialis": "Qty spesialis terbesar",
}


def wape_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """WAPE dari hasil evaluasi yang sama, bukan perbandingan antar-CSV."""
    matrix = metrics.pivot(index="model", columns="kategori", values="WAPE (%)")
    order = [c for c in ("Overall", "A", "B", "C") if c in matrix.columns]
    order += [c for c in matrix.columns if c not in order]
    return matrix.reindex(columns=order)


def context_correlations(
    frame: pd.DataFrame, suffix: str = "_lag_7", method: str = "pearson"
) -> tuple[pd.DataFrame, list[str]]:
    """Korelasi pada level obat-hari; konstanta tidak memiliki korelasi.

    Jumlah dokter harian sudah berulang per obat: jangan dijumlahkan kembali
    atau ditafsirkan sebagai jumlah dokter unik per kategori.
    """
    names = {"qty_keluar": "Qty obat"}
    names.update({col + suffix: label for col, label in CONTEXT_LABELS.items()})
    values = frame[[c for c in names if c in frame]].rename(columns=names)
    constant = values.columns[values.nunique(dropna=True) < 2].tolist()
    if "Qty obat" not in values or "Qty obat" in constant:
        return pd.DataFrame(), constant
    values = values.drop(columns=constant)
    if len(values.columns) < 2:
        return pd.DataFrame(), constant
    return values.corr(method=method, min_periods=3), constant
