"""Dashboard Streamlit: forecasting pengeluaran obat harian (XGBoost)."""
import hashlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import group_mapping, load_daily
from src.explain import feature_impact
from src.evaluation import context_correlations, wape_matrix
from src.train_xgb import run_pipeline, save_model
from src.schedule import (DEFAULT_SCHEDULE_CSV, DEFAULT_DOCTOR_PARETO_CSV,
                          DEFAULT_DOCTOR_FORECAST_CSV, load_schedule)

# ---------------------------------------------------------------- dark theme for plots
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "text.color": "#f0f6fc",
    "legend.facecolor": "#0d1117",
    "legend.edgecolor": "#30363d",
})

DEFAULT_CSV = "data/raw/groubynama+spesialis+dokter.csv"
st.set_page_config(page_title="Forecast Pengeluaran Obat", layout="wide", page_icon="💊")

# Custom Dark Mode styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22 0%, #1f242c 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00d4b2;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        color: #8b949e;
        transition: color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4b2 !important;
        border-bottom: 2px solid #00d4b2 !important;
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


CACHE_VERSION = "v11_calendar_opt_in"


@st.cache_data(show_spinner="Membersihkan data & zero-filling...")
def get_daily(file_bytes: bytes, version: str = CACHE_VERSION) -> pd.DataFrame:
    import io
    return load_daily(io.BytesIO(file_bytes), group_supplier=True)


@st.cache_resource(show_spinner="Melatih XGBoost per kategori & menghitung forecast...")
def get_results(key: str, daily: pd.DataFrame, horizon: int, schedule_bytes: bytes,
                doctor_pareto_bytes=None, doctor_practice_bytes=None, include_calendar=False,
                calendar_bytes=None, version: str = CACHE_VERSION):
    res = run_pipeline(daily, horizon, schedule=io.BytesIO(schedule_bytes),
                       doctor_pareto=io.BytesIO(doctor_pareto_bytes) if doctor_pareto_bytes is not None else None,
                       doctor_practice=io.BytesIO(doctor_practice_bytes) if doctor_practice_bytes is not None else None,
                       include_calendar=include_calendar,
                       calendar_overrides=io.BytesIO(calendar_bytes) if calendar_bytes is not None else None)
    save_model(res)
    return res


def show_heatmap(matrix, *, correlation=False):
    """Tampilkan nilai tidak terdefinisi sebagai N/A, bukan angka nol."""
    fig, ax = plt.subplots(figsize=(max(7, len(matrix.columns) * 0.9),
                                    max(2.5, len(matrix.index) * 0.55)))
    cmap = plt.get_cmap("RdBu_r" if correlation else "YlOrRd").copy()
    cmap.set_bad("#374151")
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    vmax = 1 if correlation else max(10.0, float(np.nanmax(values)) if np.isfinite(values).any() else 10.0)
    image = ax.imshow(masked, cmap=cmap, vmin=-1 if correlation else 0, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            value = values[row, col]
            text = "N/A" if not np.isfinite(value) else f"{value:.2f}" if correlation else f"{value:.2f}%"
            dark = abs(value) > 0.6 if correlation else value > vmax * 0.55
            ax.text(col, row, text, ha="center", va="center", fontsize=9,
                    color="white" if dark or not np.isfinite(value) else "#111827")
    fig.colorbar(image, ax=ax, label="Koefisien korelasi" if correlation else "WAPE (%) — lebih kecil lebih baik")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.title("💊 Forecast Pengeluaran Obat Harian (XGBoost)")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pengaturan")
    up = st.file_uploader("Upload CSV (kolom: tanggal, ID obat, qty, kategori ABC)", type="csv")
    schedule_up = st.file_uploader("Upload skor jadwal spesialis (opsional, mengganti default)", type="csv")
    use_doctor_forecast = st.checkbox("Pakai jadwal per dokter untuk forecast", value=True)
    doctor_pareto_up = st.file_uploader("Master Pareto dokter (opsional, mengganti default)", type="csv")
    doctor_practice_up = st.file_uploader("Jadwal per dokter (opsional, mengganti default)", type="csv")
    include_calendar = st.checkbox("Pakai fitur libur dan Ramadan", value=False,
                                   key="include_calendar_opt_in")
    calendar_up = st.file_uploader("Koreksi kalender / tutup klinik (opsional)", type="csv")
    st.caption("Libur nasional/cuti bersama dari holidays. Ramadan estimasi Umm al-Qura; "
               "unggah koreksi sesuai keputusan Indonesia. Libur tidak berarti RS tutup.")
    horizon = st.slider("Horizon forecast (hari)", 7, 14, 14)
    if st.button("🔄 Reset Cache & Latih Ulang"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if up is not None:
    data_bytes = up.getvalue()
elif os.path.exists(DEFAULT_CSV):
    data_bytes = open(DEFAULT_CSV, "rb").read()
    st.sidebar.caption(f"Memakai file default: `{DEFAULT_CSV}`")
else:
    st.info("Upload file CSV untuk memulai.")
    st.stop()

try:
    daily = get_daily(data_bytes, version=CACHE_VERSION)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

with st.expander("Penggabungan varian supplier (cek di sini)"):
    st.dataframe(group_mapping(data_bytes), use_container_width=True, hide_index=True)

empty_pareto = daily.attrs.get("empty_pareto_drugs", 0)
if empty_pareto > 0:
    st.warning(f"⚠️ Perhatian: Terdapat {empty_pareto} obat dengan kategori Pareto kosong/NULL di dataset. Kategori otomatis diisi 'C'.")
multi_pareto = daily.attrs.get("multi_pareto_drugs", 0)
if multi_pareto > 0:
    st.info(f"ℹ️ Info: Terdapat {multi_pareto} obat dengan >1 label kategori Pareto antar varian (diambil label bervolume terbesar).")

n_days = daily["tanggal"].nunique()
if n_days < 45:
    st.warning(f"Data hanya {n_days} hari; hasil model kemungkinan kurang andal (disarankan ≥ 45 hari).")

try:
    if schedule_up is not None:
        schedule_bytes = schedule_up.getvalue()
    else:
        with open(DEFAULT_SCHEDULE_CSV, "rb") as schedule_file:
            schedule_bytes = schedule_file.read()
        st.sidebar.caption(f"Jadwal default: `{DEFAULT_SCHEDULE_CSV}`")
    schedule_table = load_schedule(io.BytesIO(schedule_bytes))
    st.sidebar.caption(f"Cakupan jadwal: {schedule_table['tanggal'].min().date()} s/d "
                       f"{schedule_table['tanggal'].max().date()}")
    key = hashlib.md5(data_bytes).hexdigest()
    doctor_pareto_bytes = doctor_practice_bytes = None
    if use_doctor_forecast:
        if doctor_pareto_up is not None:
            doctor_pareto_bytes = doctor_pareto_up.getvalue()
        else:
            with open(DEFAULT_DOCTOR_PARETO_CSV, "rb") as f:
                doctor_pareto_bytes = f.read()
        if doctor_practice_up is not None:
            doctor_practice_bytes = doctor_practice_up.getvalue()
        else:
            with open(DEFAULT_DOCTOR_FORECAST_CSV, "rb") as f:
                doctor_practice_bytes = f.read()
    res = get_results(key, daily, horizon, schedule_bytes, doctor_pareto_bytes,
                      doctor_practice_bytes, include_calendar,
                      calendar_up.getvalue() if calendar_up is not None else None, version=CACHE_VERSION)
except (ValueError, OSError) as e:
    st.error(f"Gagal menyiapkan training/forecast: {e}")
    st.stop()

if res.get("calendar") is not None:
    with st.expander("Kalender yang dipakai model"):
        st.caption("Jarak libur dihitung terhadap libur nasional atau cuti bersama, maksimal 28 hari. "
                   "is_clinic_closed hanya digunakan jika diunggah dan seluruh tanggal tercakup. "
                   "Kalender pustaka bukan arsip versi keputusan yang tersedia pada tanggal forecast lama.")
        st.dataframe(res["calendar"], hide_index=True, use_container_width=True)
        st.download_button("Download kalender model", res["calendar"].to_csv(index=False).encode("utf-8"),
                           "kalender_model.csv", "text/csv")

if res.get("doctor_master") is not None:
    st.sidebar.caption("Forecast memakai skor dari jadwal dokter A/B/C. Training dan holdout tetap memakai skor historis.")
    with st.expander("ID dokter dan jadwal forecast"):
        st.warning("ID berasal dari pasangan nama tersensor + spesialis, bukan identitas terverifikasi. "
                   "Roster dianggap lengkap pada setiap tanggal tercantum: spesialis tanpa baris dihitung nol. "
                   "Tanggal tanpa baris tetap dianggap belum tersedia. Periksa kelengkapan roster sebelum dipakai.")
        st.dataframe(res["doctor_master"], hide_index=True, use_container_width=True)
        st.download_button("Download master ID dokter", res["doctor_master"].to_csv(index=False).encode("utf-8"),
                           "master_dokter_id.csv", "text/csv")
        st.download_button("Download jadwal dengan ID", res["doctor_roster"].to_csv(index=False).encode("utf-8"),
                           "jadwal_dokter_id.csv", "text/csv")

cats = sorted(daily["kategori_pareto"].unique())
with st.sidebar:
    sel_cats = st.multiselect("Kategori Pareto", cats, default=cats)
    names = (daily[daily["kategori_pareto"].isin(sel_cats)]
             .drop_duplicates("id_obat")[["id_obat", "nama_obat"]])
    label = {
        r.id_obat: r.nama_obat if r.id_obat == r.nama_obat else f"{r.id_obat} - {r.nama_obat}"
        for r in names.itertuples()
    }
    sel_ids = st.multiselect("ID Obat (kosong = semua)", list(label), format_func=label.get)

# ---------------------------------------------------------------- KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah obat", daily["id_obat"].nunique())
c2.metric("Rentang data", f"{n_days} hari")
c3.metric("Objective (per kat)", res["objective"])
c4.metric("Jumlah pohon", res["n_trees"])

fc = res["forecast"]
fc = fc[fc["kategori_pareto"].isin(sel_cats)]
hist = daily[daily["kategori_pareto"].isin(sel_cats)]
if sel_ids:
    fc, hist = fc[fc["id_obat"].isin(sel_ids)], hist[hist["id_obat"].isin(sel_ids)]

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🎯 Evaluasi", "🔍 Interpretabilitas", "🗂 Data"])

# ---------------------------------------------------------------- Forecast
with tab1:
    st.subheader(f"Prediksi {horizon} hari ke depan")
    ids = sel_ids or list(hist["id_obat"].unique())
    show = st.selectbox("Obat untuk grafik", ids, format_func=lambda i: label.get(i, i)) if ids else None
    if show:
        h = hist[hist["id_obat"] == show].tail(60)
        f = fc[fc["id_obat"] == show]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")
        ax.plot(h["tanggal"], h["qty_keluar"], label="Aktual", color="#38bdf8", lw=1.8)
        ax.plot(f["tanggal"], f["prediksi"], label="Forecast", color="#fb7185", marker="o", ms=4, lw=2)
        ax.axvline(res["max_date"], color="#94a3b8", ls=":", lw=1.5, label="Batas Forecast")
        ymax = max(float(h["qty_keluar"].max() if len(h) else 0), float(f["prediksi"].max() if len(f) else 0))
        ax.set_ylim(bottom=0, top=max(1.0, ymax * 1.15))
        ax.set_ylabel("Qty keluar", color="#c9d1d9", fontsize=10)
        ax.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d")
        fig.autofmt_xdate()
        st.pyplot(fig)

    pivot = fc.pivot_table(index=["id_obat", "nama_obat", "kategori_pareto"],
                           columns=fc["tanggal"].dt.strftime("%d-%m"), values="prediksi").round(1)
    pivot["TOTAL"] = pivot.sum(axis=1)
    st.dataframe(pivot, use_container_width=True)
    st.download_button("⬇️ Unduh forecast (CSV)", fc.to_csv(index=False).encode(),
                       "forecast.csv", "text/csv")

# ---------------------------------------------------------------- Evaluasi
with tab2:
    st.caption("Skor jadwal spesialis aktif pada training dan forecast. Evaluasi mengasumsikan jadwal "
               "serta Pareto dokter sudah diketahui sebelum forecast dibuat. Pareto yang dihitung "
               "dari kunjungan masa depan dapat menyebabkan kebocoran data.")
    st.caption(f"Time-based split: train s/d {res['split_date'].date()}, uji = 14 hari terakhir. "
               "Model dilatih mandiri per kategori Pareto (A: squarederror, B/C: tweedie).")
    st.subheader("Recursive 14 hari (kondisi nyata forecasting)")
    st.dataframe(res["metrics_recursive"].round(2), use_container_width=True, hide_index=True)
    st.subheader("One-step-ahead (fitur dari data aktual)")
    st.dataframe(res["metrics_onestep"].round(2), use_container_width=True, hide_index=True)
    st.info("WAPE sangat besar pada kategori dengan total aktual kecil/jarang terpakai "
             "adalah normal: penyebutnya mendekati 0. Lihat MAE untuk perbandingan yang lebih stabil.")

    st.subheader("Heatmap WAPE per kategori")
    metric_view = st.radio("Evaluasi untuk heatmap WAPE", ["Recursive 14 hari", "One-step-ahead"],
                           horizontal=True)
    metric_key = "metrics_recursive" if metric_view == "Recursive 14 hari" else "metrics_onestep"
    show_heatmap(wape_matrix(res[metric_key]))
    st.caption("WAPE memakai seluruh obat pada CSV aktif, tidak mengikuti filter obat sidebar. "
               "Warna lebih gelap berarti error lebih besar. Ini membandingkan XGBoost dengan SMA-7, "
               "bukan model dengan vs tanpa fitur dokter.")

    st.subheader("Heatmap korelasi dokter, spesialis, dan jenis rawat")
    col_scope, col_period = st.columns(2)
    with col_scope:
        corr_cat = st.selectbox("Kategori untuk korelasi", ["Semua"] + cats)
        corr_method = st.selectbox("Metode korelasi", ["Pearson", "Spearman"],
                                   help="Pearson: hubungan linear. Spearman: hubungan berdasarkan peringkat.")
    with col_period:
        corr_period = st.selectbox("Periode korelasi", ["Data latih", "Holdout 14 hari", "Data latih + holdout"])
        context_mode = st.selectbox("Versi fitur konteks", ["Lag 7 hari", "Rolling mean 7 hari", "Hari yang sama"])
    if corr_period == "Data latih":
        corr_frame = res["train_features"]
    elif corr_period == "Holdout 14 hari":
        corr_frame = res["test_features"]
    else:
        corr_frame = pd.concat([res["train_features"], res["test_features"]], ignore_index=True)
    if corr_cat != "Semua":
        corr_frame = corr_frame[corr_frame["kategori_pareto"] == corr_cat]
    suffix = {"Lag 7 hari": "_lag_7", "Rolling mean 7 hari": "_rolling_mean_7", "Hari yang sama": ""}[context_mode]
    corr, constants = context_correlations(corr_frame, suffix, corr_method.lower())
    st.caption(f"{len(corr_frame):,} observasi obat-hari pada {corr_frame['tanggal'].nunique()} tanggal. "
               "Menggunakan seluruh obat dalam kategori pilihan, tidak mengikuti filter sidebar. "
               "Dokter/spesialis harian dihitung dari transaksi pada CSV aktif, bukan jadwal praktik; "
               "nilainya berulang untuk setiap obat pada tanggal yang sama.")
    if corr.empty:
        st.info("Korelasi belum dapat dihitung: data kurang, qty konstan, atau fitur konteks tidak tersedia/konstan. "
                "Upload CSV dengan DOKTER, SPESIALIST, dan JENIS RAWAT untuk analisis konteks.")
    else:
        show_heatmap(corr, correlation=True)
        st.download_button("⬇️ Unduh korelasi (CSV)", corr.to_csv().encode("utf-8-sig"),
                           "korelasi_konteks.csv", "text/csv")
    if constants:
        st.caption("Dikeluarkan karena konstan/kosong (korelasi tidak terdefinisi): " + ", ".join(constants))
    st.warning("Korelasi bukan bukti bahwa fitur dokter menurunkan WAPE. Korelasi gabungan dapat dipengaruhi "
               "perbedaan skala antarobat dan pola hari pelayanan. Qty per jenis rawat/spesialis pada hari "
               "yang sama merupakan bagian dari target qty obat, sehingga hubungannya bersifat mekanis "
               "dan tidak boleh digunakan sebagai input forecast hari tersebut. Untuk mengukur manfaat "
               "dokter, bandingkan model dengan/tanpa fitur dokter pada obat, periode, dan split yang sama.")

    ho = res["holdout"]
    rid = st.selectbox("Aktual vs prediksi (holdout)", sorted(ho["id_obat"].unique()),
                       format_func=lambda i: label.get(i, i))
    g = ho[ho["id_obat"] == rid].sort_values("tanggal")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")
    ax.plot(g["tanggal"], g["qty_keluar"], label="Aktual", marker="o", ms=4, color="#38bdf8", lw=1.8)
    ax.plot(g["tanggal"], g["pred_xgb"], label="XGBoost", marker="o", ms=4, color="#fb7185", lw=2)
    ax.plot(g["tanggal"], g["pred_sma7"], label="Baseline SMA-7", ls="--", color="#facc15", lw=1.6)
    ymax_ho = max(float(g["qty_keluar"].max() if len(g) else 0), float(g["pred_xgb"].max() if len(g) else 0), float(g["pred_sma7"].max() if len(g) else 0))
    ax.set_ylim(bottom=0, top=max(1.0, ymax_ho * 1.15))
    ax.set_ylabel("Qty keluar", color="#c9d1d9", fontsize=10)
    ax.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d")
    fig.autofmt_xdate()
    st.pyplot(fig)

# ---------------------------------------------------------------- SHAP
with tab3:
    eval_models = res.get("eval_models", {})
    if isinstance(eval_models, dict) and len(eval_models) > 0:
        cat_choices = sorted(eval_models.keys())
        cat_sel = st.selectbox("Pilih Model Kategori Pareto untuk Dianalisis", cat_choices)
        selected_model = eval_models[cat_sel]
        cat_train = res["train_features"][res["train_features"]["kategori_pareto"] == cat_sel]
        sample_df = cat_train.sample(min(500, len(cat_train)), random_state=0) if len(cat_train) > 0 else cat_train
        imp = feature_impact(selected_model, sample_df)
        st.caption(f"Model Kategori: **{cat_sel}** | Objective: `{res['objectives'].get(cat_sel, '')}` | Sumber: {imp.attrs.get('source', '')}")
    else:
        eval_m = res["eval_model"]
        sample_df = res["train_features"].sample(min(500, len(res["train_features"])), random_state=0)
        imp = feature_impact(eval_m, sample_df)
        st.caption(f"Sumber: {imp.attrs.get('source', '')}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#30363d")
    imp.iloc[::-1].plot.barh(x="fitur", y="dampak", ax=ax, legend=False, color="#00d4b2", edgecolor="#161b22")
    ax.set_xlabel("Dampak rata-rata terhadap prediksi", color="#c9d1d9", fontsize=10)
    st.pyplot(fig)

# ---------------------------------------------------------------- Data
with tab4:
    st.write("Data harian setelah sanitasi & zero-filling:")
    st.dataframe(hist.tail(500), use_container_width=True)
    st.download_button("⬇️ Unduh data bersih (CSV)", daily.to_csv(index=False).encode(),
                       "daily_zero_filled.csv", "text/csv")
