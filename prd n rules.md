# PRD & AI Agent Rules: Drug Expenditure Forecasting (< 6 Months Data)

## 1. Product Requirements Document (PRD)

### 1.1 Ringkasan Proyek (Project Overview)
Sistem prediksi (*forecasting*) pengeluaran obat harian di Rumah Sakit menggunakan Python dan **XGBoost** untuk merencanakan kebutuhan stok jangka pendek (7–14 hari ke depan). Sistem ini mengolah dataset historis berdurasi **kurang dari 6 bulan (< 180 hari)** dan langsung memanfaatkan kolom **Kategori Pareto (A, B, C)** yang sudah tersedia di dalam dataset.
    
### 1.2 Tujuan & Target (Objectives & Key Results)
* **Tujuan Utama:** Menghasilkan prediksi jumlah pengeluaran harian per ID Obat untuk horizon 7 hingga 14 hari ke depan menggunakan algoritma XGBoost.
* **Pemanfaatan Kategori Pareto Bawaan:** Menggunakan kolom Pareto yang sudah ada di dataset sebagai fitur prediktor sekaligus filter evaluasi (memfokuskan akurasi harian pada Kategori A dan B).
* **Target Evaluasi:** Mencapai skor *Mean Absolute Error* (MAE) dan *Weighted Absolute Percentage Error* (WAPE) yang lebih rendah dibandingkan *Baseline Model* (*Simple Moving Average* 7 hari).

### 1.3 Tech Stack (Python Ecosystem)
* **Data Manipulation & Preprocessing:** `pandas`, `numpy`
* **Machine Learning (Core Model):** `xgboost` (`XGBRegressor`), `scikit-learn` (untuk *preprocessing* & *metrics*)
* **Model Interpretability:** `shap` (untuk analisis *feature importance* prediksi obat)
* **Model Persistence:** `joblib`
* **Dashboard & Visualisasi:** `streamlit`, `matplotlib`, `seaborn`

---

## 2. Aturan Bisnis & Teknis (System & Modeling Rules)

### Rule 1: Data Cleaning & Sanitasi CSV
1. **Hapus Header Duplikat:** Dataset mentah memiliki baris *header* yang berulang di tengah data (contoh: muncul baris `tanggal,ID...` di tengah transaksi). Sistem wajib membuang baris di mana kolom `tanggal == 'tanggal'` sebelum konversi tipe data.
2. **Pemisahan Kolom:** Jika file CSV terbaca sebagai satu kolom teks utuh yang dipisahkan koma, sistem harus memecahnya sesuai *delimiter* koma (`,`).
3. **Standarisasi Tipe Data:**
   * `tanggal` $\rightarrow$ `datetime64[ns]` (diurutkan dari terlama ke terbaru).
   * `id_obat` $\rightarrow$ `string` / `category`.
   * `kategori_pareto` $\rightarrow$ `string` / `category` (langsung diambil dari dataset asli, bersih dari *whitespace* seperti `' A '` menjadi `'A'`).
   * `qty_keluar` $\rightarrow$ `float32` / `int32` (nilai negatif akibat retur di-*clip* menjadi `0`).

### Rule 2: Pemetaan Pareto Bawaan & Zero-Filling (Krusial)
1. **Lookup Tabel Obat & Pareto:** Sebelum melakukan *zero-filling*, ekstrak tabel referensi unik berisi pasangan `(id_obat, kategori_pareto)` dari dataset asli. Jika satu obat memiliki lebih dari satu label Pareto karena kesalahan input masa lalu, ambil label Pareto dari tanggal paling baru (*most recent*).
2. **Agregasi Harian:** Jumlahkan (`SUM`) kolom `qty_keluar` per `tanggal` dan `id_obat`.
3. **Complete Date Grid (Zero-Filling):** Buat rentang tanggal harian penuh (`min_date` s/d `max_date`) untuk setiap `id_obat`. Tanggal tanpa transaksi wajib diisi `qty_keluar = 0`, lalu gabungkan kembali (*merge*) dengan tabel referensi `kategori_pareto` bawaan tadi agar kolom Pareto tidak ikut menjadi `NaN`/`0`.

### Rule 3: Pemanfaatan Kategori Pareto Bawaan di XGBoost
Karena kategori Pareto (A, B, C) sudah tersedia dari dataset:
1. **Jangan Hitung Ulang Pareto:** Gunakan langsung nilai dari kolom dataset.
2. **Sebagai Fitur Model:** Lakukan *Ordinal Encoding* (`A: 2, B: 1, C: 0`) atau *One-Hot Encoding* pada kolom `kategori_pareto` agar pohon keputusan (*trees*) di XGBoost langsung mengenali skala perputaran obat.
3. **Strategi Loss Function XGBoost:**
   * Gunakan `objective='reg:squarederror'` jika melatih khusus obat **Kategori A**.
   * Gunakan `objective='reg:tweedie'` (dengan `tweedie_variance_power=1.5`) jika melatih **Kategori A, B, dan C sekaligus** dalam satu *Global Model*, karena distribusi *Tweedie* sangat tahan terhadap data yang memiliki banyak angka `0` (*zero-inflated*).

### Rule 4: Pembatasan Feature Engineering (< 6 Bulan Data)
* **DILARANG** membuat fitur musiman tahunan (`month`, `year`, `lag_90`, `lag_180`) karena durasi data < 6 bulan.
* **WAJIB** membatasi *Lag* dan *Rolling Window* maksimal **21–28 hari**:
  * **Lag Features (per `id_obat`):** `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_21`.
  * **Rolling Features (Wajib `.shift(1)` sebelum `.rolling()` agar bebas *data leakage*):**
    * `rolling_mean_7`, `rolling_std_7`
    * `rolling_mean_14`, `rolling_max_14`
    * `rolling_mean_28`
  * **Calendar Features:** `day_of_week` (0–6), `is_weekend` (0/1), `day_of_month` (1–31).
  * **Static Features (dari Dataset):** `id_obat_encoded`, `kategori_pareto_encoded`.

### Rule 5: Validasi, Splitting, & Metrik Evaluasi
1. **DILARANG Random Split:** Wajib menggunakan **Time-Based Split**.
   * **Train Set:** Data awal s/d `max_date - 14 hari`.
   * **Validation/Test Set:** 14 hari terakhir dari dataset.
2. **Metrik Evaluasi (Terpisah per Kategori Pareto):**
   * Hitung **MAE**, **RMSE**, dan **WAPE** secara keseluruhan, **DAN** tampilkan tabel evaluasi terpisah untuk **Kategori A**, **Kategori B**, dan **Kategori C**.
   * **DILARANG** menggunakan MAPE standar karena adanya nilai aktual `0` pada pengeluaran harian.

---

## 3. AI Coding Agent Prompt (`AGENT_INSTRUCTIONS.md`)

> **Cara Pakai:** Simpan blok di bawah ini sebagai file `.cursorrules`, `CLAUDE.md`, atau berikan sebagai instruksi pembuka (*System Prompt*) ke AI Coding Agent kamu.

### System Prompt untuk Agent

**Role:**
Kamu adalah Senior Data Scientist dan Machine Learning Engineer spesialis *Time-Series Forecasting* menggunakan **XGBoost** dan Python.

**Project Context:**
Kita membangun sistem prediksi pengeluaran obat harian di Rumah Sakit dengan data historis **kurang dari 6 bulan (< 180 hari)**.
* Dataset sudah memiliki kolom **Kategori Pareto (`A`, `B`, `C`)** bawaan untuk setiap obat. **JANGAN membuat perhitungan Pareto baru**, langsung ambil dan pertahankan kolom Pareto dari dataset tersebut.
* Algoritma utama yang digunakan adalah **XGBoost (`xgboost.XGBRegressor`)**.

**Strict Coding Rules (WAJIB DIPATUHI AGENT):**
1. **Sanitasi CSV & Header Berulang:** Pada `data_loader.py`, buat fungsi yang membersihkan baris *header* duplikat di tengah data (contoh: membuang baris yang nilai kolom tanggalnya adalah string `'tanggal'`).
2. **Preservasi Kategori Pareto Saat Zero-Filling:**
   * Sebelum membuat *complete date grid* (`pd.MultiIndex.from_product`), simpan mapping unik `id_obat -> kategori_pareto` dari dataframe asli.
   * Setelah mengisi tanggal kosong dengan `qty_keluar = 0`, petakan kembali (`map`/`merge`) `kategori_pareto` ke seluruh baris tanggal agar setiap obat tetap memiliki label A/B/C yang utuh.
3. **Anti-Data Leakage pada Rolling Window:**
   * Saat menghitung fitur *rolling* (`mean`, `std`, `max`), kamu **WAJIB** menerapkan `.shift(1)` terlebih dahulu di dalam `groupby('id_obat')`.
   * Contoh wajib: `df.groupby('id_obat')['qty_keluar'].transform(lambda x: x.shift(1).rolling(7).mean())`.
4. **Batas Horizon Fitur (< 6 Bulan):** Jangan membuat fitur *lag* atau *rolling* lebih dari 28 hari agar data *training* tidak habis terpotong `NaN`.
5. **Konfigurasi Model XGBoost:**
   * Gunakan `XGBRegressor` dari library `xgboost`.
   * Mendukung `enable_categorical=True` atau lakukan encoding eksplisit pada `id_obat` dan `kategori_pareto`.
   * Gunakan `objective='reg:tweedie'` (default untuk data gabungan A, B, C yang memiliki banyak nilai 0) atau `reg:squarederror`, dengan `early_stopping_rounds` pada data validasi.
6. **Evaluasi Per Kategori Pareto:**
   * Jangan pernah menggunakan MAPE standar (`mean_absolute_percentage_error`).
   * Buat fungsi evaluasi yang menghitung **MAE**, **RMSE**, dan **WAPE**, lalu cetak hasilnya secara grup berdasarkan `kategori_pareto` (A, B, dan C) dan bandingkan dengan *Baseline Moving Average 7 Hari*.
7. **Interpretabilitas Model:** Sediakan modul opsional menggunakan `shap.TreeExplainer` atau *built-in feature importance* XGBoost untuk menjelaskan fitur apa yang paling memengaruhi prediksi pengeluaran obat.

**Struktur Direktori Proyek:**
```text
drug_forecasting_xgb/
├── data/
│   ├── raw/                  # CSV mentah dari RS (sudah ada kolom Pareto)
│   └── processed/            # Data bersih hasil zero-filling & fitur lag
├── src/
│   ├── data_loader.py        # Clean header ganda, ekstrak mapping Pareto, & zero-filling
│   ├── features.py           # Lag (1-21), Rolling shifted (7-28), Calendar, & Pareto encoding
│   ├── train_xgb.py          # Time-based split, Baseline vs XGBRegressor, evaluasi per Pareto
│   └── forecast.py           # Recursive forecasting 7-14 hari ke depan
├── models/
│   └── xgb_forecaster.joblib # Model XGBoost & encoder tersimpan
├── app.py                    # Dashboard Streamlit (filter per Kategori Pareto A/B/C & ID Obat)
└── requirements.txt          # pandas, numpy, scikit-learn, xgboost, shap, joblib, streamlit