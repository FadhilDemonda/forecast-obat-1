# 💊 Drug Forecasting XGBoost

Sistem peramalan (*forecasting*) kebutuhan pengeluaran obat harian berbasis **Machine Learning (XGBoost)** yang dirancang khusus untuk manajemen rantai pasok dan farmasi rumah sakit. 

Sistem ini menerapkan prinsip segmentasi Pareto (ABC), penanganan data terputus (*zero-filling*), penggabungan otomatis varian antar-supplier, serta pelatihan model terpisah per kategori untuk menghasilkan akurasi optimal pada pola obat *fast-moving* maupun *slow-moving / intermittent*.

---

## 🚀 Panduan Cepat

### 1. Instalasi Dependensi
Pastikan Python 3.9+ telah terpasang:
```bash
pip install -r requirements.txt
```

### 2. Pelatihan & Evaluasi Model via CLI
Jalankan pipeline penuh (sanitasi, zero-filling, training per kategori, evaluasi 14 hari, dan recursive forecast):
```bash
python -m src.train_xgb --csv data/raw/test1.csv --horizon 14
```
Hasil model akan disimpan di `models/xgb_forecaster.joblib` dan data forecast di `data/processed/forecast.csv`.

### 3. Menjalankan Dashboard Interaktif
Aplikasi berbasis Streamlit untuk visualisasi data, simulasi forecast, holdout test, dan interpretabilitas fitur:
```bash
streamlit run app.py
```

---

## 🧠 Arsitektur & Aturan Sistem

### 1. Sanitasi & Penggabungan Varian Supplier (`src/data_loader.py`)
* **Pemetaan Kolom Otomatis**: Mendeteksi berbagai variasi penamaan kolom CSV (misal `ID BARANG` → `id_obat`, `qty_total` → `qty_keluar`, `kategori_abc` → `kategori_pareto`) via `COLUMN_ALIASES`.
* **Penggabungan Varian Supplier**: Fungsi `base_name` memotong sufiks kurung supplier di akhir nama (contoh `ACARBOSE 100 MG TAB (DEXA)` dan `ACARBOSE 100 MG TAB (NOVELL)` otomatis disatukan menjadi `ACARBOSE 100 MG TAB`).
* **Agregasi Kuantitas Antar-Supplier**: Kuantitas keluar dari seluruh supplier untuk obat yang sama dijumlahkan per tanggal.
* **Resolusi Label Pareto**: Jika varian obat memiliki label ABC berbeda antar supplier, label yang diadopsi diambil dari varian dengan total volume terbesar.
* **Zero-Filling Deret Waktu**: Tanggal tanpa mutasi/transaksi diisi nilai `0.0` melalui pembuatan *date grid* lengkap per obat untuk mencegah data deret waktu bolong (*intermittent gap*).
* **Tabel Pemetaan**: Fungsi `group_mapping(source)` menyediakan tabel pengecekan transparan antara varian asli dan obat gabungan.

### 2. Rekayasa Fitur (*Feature Engineering*) (`src/features.py`)
Membangun fitur deret waktu tanpa kebocoran data (*data leakage*):
* **Lags**: $t-1, t-2, t-3, t-7, t-14, t-21$.
* **Rolling Statistics**: Rata-rata (*mean*), deviasi standar (*std*), dan nilai maksimum (*max*) untuk jendela 7, 14, dan 28 hari. Seluruh fitur *rolling* menerapkan `.shift(1)` sebelum kalkulasi untuk mencegah *lookahead bias*.
* **Fitur Kalender**: Hari dalam minggu (`day_of_week`), indikator akhir pekan (`is_weekend`), dan tanggal dalam bulan (`day_of_month`).
* **Identitas Obat & Pareto**: Encoding integer untuk identitas unik obat dan bobot kategori Pareto.

### 3. Pelatihan Terpisah Per Kategori Pareto (`src/train_xgb.py`)
Model **tidak dicampur menjadi satu**, melainkan dilatih secara mandiri untuk masing-masing kategori Pareto:
* **Kategori A (Volume Besar / Fast-Moving)**:
  * Menggunakan objective `reg:squarederror` untuk memprediksi kuantitas kontinu dengan akurasi tinggi tanpa terdistorsi oleh angka 0 dari obat jarang terpakai.
* **Kategori B & C (Volume Sedang-Kecil / Intermittent / Slow-Moving)**:
  * Menggunakan objective `reg:tweedie` (`tweedie_variance_power=1.5`) yang dirancang khusus untuk distribusi data *zero-inflated* (banyak nilai 0).
* **Early Stopping & Batas Minimal Pohon**:
  * Menggunakan 14 hari terakhir dari data latih sebagai *validation set* untuk menentukan iterasi optimal.
  * Menerapkan batas minimal `MIN_TREES = 150` untuk mencegah model terlalu dangkal akibat *noise* validasi kecil.

### 4. Peramalan Multi-Langkah (*Recursive Forecasting*) (`src/forecast.py`)
* Prediksi dilakukan hari demi hari sepanjang horizon (misal 14 hari ke depan).
* Hasil prediksi hari ke-$t$ diumpankan kembali ke data historis untuk menghitung fitur lag dan rolling hari ke-$t+1$.
* Menggunakan `warnings.catch_warnings()` untuk menangani placeholder baris masa depan secara bersih.

### 5. Evaluasi Dua Tingkat & Metrik Bisnis
* **Time-Based Split**: Memisahkan 14 hari terakhir data aktual sebagai *holdout test* (bukan acak, sesuai urutan waktu riil).
* **Evaluasi One-Step-Ahead**: Mengukur akurasi model ketika fitur lag berasal dari data aktual.
* **Evaluasi Recursive 14 Hari**: Mengukur akurasi realistis saat model beroperasi di lapangan (memprediksi 14 hari ke depan secara beruntun).
* **Baseline SMA-7**: Model dibandingkan langsung dengan rata-rata bergerak 7 hari terakhir (*Simple Moving Average*).
* **Metrik yang Digunakan**:
  * **MAE** (*Mean Absolute Error*)
  * **RMSE** (*Root Mean Squared Error*)
  * **WAPE (%)** (*Weighted Absolute Percentage Error*)
  * *Catatan: MAPE sengaja dihindari karena aktual pengeluaran obat bernilai 0 akan menyebabkan pembagian dengan nol.*

---

## 📊 Hasil Evaluasi Terkini (Dataset 214 Hari)

Dengan pemisahan model per kategori dan penggabungan varian supplier, XGBoost mengungguli Baseline SMA-7 di semua kategori:

| Model | Kategori | Objective | MAE | RMSE | WAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Per Kat)** | **Overall** | **Campuran** | **111.04** | **244.30** | **37.13%** |
| **XGBoost (Per Kat)** | **A** | `squarederror` | **184.37** | **315.38** | **37.11%** |
| **XGBoost (Per Kat)** | **B** | `tweedie` | **1.05** | **1.85** | **44.35%** |
| Baseline SMA-7 | Overall | - | 151.58 | 310.23 | 50.69% |
| Baseline SMA-7 | A | - | 251.74 | 400.50 | 50.67% |
| Baseline SMA-7 | B | - | 1.34 | 2.15 | 56.71% |

---

## 🖥️ Fitur Dashboard Streamlit (`app.py`)

1. **Panel Penggabungan Varian Supplier**: Menampilkan ringkasan tabel mapping supplier, ID asli, dan kategori Pareto terpilih.
2. **KPI Header**: Memantau jumlah obat, rentang data, objective per kategori, dan jumlah pohon XGBoost.
3. **Tab 📈 Forecast**: Grafik tren harian per obat (aktual 60 hari terakhir + forecast 14 hari), tabel pivot harian, serta tombol unduh `forecast.csv`.
4. **Tab 🎯 Evaluasi**: Tabel perbandingan metrik recursive & one-step-ahead vs SMA-7, serta grafik perbandingan aktual vs holdout per obat.
5. **Tab 🔍 Interpretabilitas (SHAP / Gain)**: Analisis fitur paling berpengaruh dengan opsi pemilihan model per kategori Pareto (Model Kategori A vs Model Kategori B).
6. **Tab 🗂 Data**: Peninjauan data bersih hasil zero-filling dan tombol unduh `daily_zero_filled.csv`.
7. **Tombol Sidebar**: Fitur `🔄 Reset Cache & Latih Ulang` untuk memastikan perubahan data/model langsung teraplikasi tanpa residu cache browser.

---

## 📁 Struktur Direktori

```text
drug_forecasting_xgb/
├── app.py                      # Aplikasi dashboard interaktif Streamlit
├── requirements.txt            # Daftar pustaka Python (xgboost, streamlit, dll)
├── README.md                   # Dokumentasi teknis proyek
├── data/
│   ├── raw/
│   │   └── test1.csv           # File data mentah input
│   └── processed/
│       ├── daily_zero_filled.csv  # Data hasil sanitasi & zero-filling
│       └── forecast.csv           # Hasil peramalan masa depan
├── models/
│   └── xgb_forecaster.joblib   # Artefak model terlatih (per kategori)
└── src/
    ├── __init__.py
    ├── data_loader.py          # Pembersihan data, penggabungan supplier, zero-filling
    ├── features.py             # Feature engineering (lag, rolling, calendar)
    ├── forecast.py             # Recursive multi-step & baseline forecasting
    ├── train_xgb.py            # Pelatihan per kategori Pareto & evaluasi metrik
    └── explain.py              # Interpretabilitas fitur (SHAP / feature importance)
```

---

## 📋 Spesifikasi Format Data Input (CSV)

Sistem menerima file CSV dengan kolom berikut (dikenali otomatis melalui alias):
* **Tanggal**: `tanggal`, `date`, `tgl` (Format: `YYYY-MM-DD`).
* **ID Obat**: `id_obat`, `id_barang`, `kode_obat`, `kode_barang`.
* **Nama Obat**: `nama_obat`, `barang`, `nama_barang`, `nama`.
* **Kuantitas Keluar**: `qty_keluar`, `qty_total`, `qty`, `jumlah_keluar`.
* **Kategori Pareto**: `kategori_pareto`, `kategori_abc`, `pareto`, `abc` (`A`, `B`, atau `C`).
