"""Dashboard Streamlit: forecasting pengeluaran obat harian (XGBoost)."""
import hashlib
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import group_mapping, load_daily
from src.explain import feature_impact
from src.train_xgb import run_pipeline, save_model

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

DEFAULT_CSV = "data/raw/test1.csv"
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


CACHE_VERSION = "v4_dark_mode"


@st.cache_data(show_spinner="Membersihkan data & zero-filling...")
def get_daily(file_bytes: bytes, version: str = CACHE_VERSION) -> pd.DataFrame:
    import io
    return load_daily(io.BytesIO(file_bytes), group_supplier=True)


@st.cache_resource(show_spinner="Melatih XGBoost per kategori & menghitung forecast...")
def get_results(key: str, daily: pd.DataFrame, horizon: int, version: str = CACHE_VERSION):
    res = run_pipeline(daily, horizon)
    save_model(res)
    return res


st.title("💊 Forecast Pengeluaran Obat Harian (XGBoost)")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pengaturan")
    up = st.file_uploader("Upload CSV (kolom: tanggal, ID obat, qty, kategori ABC)", type="csv")
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

n_days = daily["tanggal"].nunique()
if n_days < 45:
    st.warning(f"Data hanya {n_days} hari; hasil model kemungkinan kurang andal (disarankan ≥ 45 hari).")

key = hashlib.md5(data_bytes).hexdigest()
res = get_results(key, daily, horizon, version=CACHE_VERSION)

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
    st.caption(f"Time-based split: train s/d {res['split_date'].date()}, uji = 14 hari terakhir. "
               "Model dilatih mandiri per kategori Pareto (A: squarederror, B/C: tweedie).")
    st.subheader("Recursive 14 hari (kondisi nyata forecasting)")
    st.dataframe(res["metrics_recursive"].round(2), use_container_width=True, hide_index=True)
    st.subheader("One-step-ahead (fitur dari data aktual)")
    st.dataframe(res["metrics_onestep"].round(2), use_container_width=True, hide_index=True)
    st.info("WAPE sangat besar pada kategori dengan total aktual kecil/jarang terpakai "
            "adalah normal: penyebutnya mendekati 0. Lihat MAE untuk perbandingan yang lebih stabil.")

    ho = res["holdout"]
    rid = st.selectbox("Aktual vs prediksi (holdout)", sorted(ho["id_obat"].unique()),
                       format_func=lambda i: label.get(i, i))
    g = ho[ho["id_obat"] == rid].sort_values("tanggal")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")
    ax.plot(g["tanggal"], g["qty_keluar"], label="Aktual", marker="o", ms=4, color="#38bdf8", lw=1.8)
    ax.plot(g["tanggal"], g["pred_xgb"], label="XGBoost", marker="o", ms=4, color="#fb7185", lw=2)
    ax.plot(g["tanggal"], g["pred_sma7"], label="Baseline SMA-7", ls="--", color="#facc15", lw=1.6)
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
