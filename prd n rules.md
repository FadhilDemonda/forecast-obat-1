# PRD & AI Agent Rules: Drug Expenditure Forecasting (< 6 Months Data)

## 1. Product Requirements Document (PRD)

### 1.1 Ringkasan Proyek (Project Overview)
Sistem prediksi (*forecasting*) pengeluaran obat harian di Rumah Sakit menggunakan Python dan **XGBoost** untuk merencanakan kebutuhan stok jangka pendek (7–14 hari ke depan). Sistem ini mengolah dataset historis berdurasi **kurang dari 6 bulan (< 180 hari)**, menggabungkan varian supplier untuk obat yang sama secara otomatis, dan melatih **model terpisah per Kategori Pareto (A, B, C)** agar akurasi optimal pada obat *fast-moving* maupun *slow-moving / intermittent*.
    
### 1.2 Tujuan & Target (Objectives & Key Results)
* **Tujuan Utama:** Menghasilkan prediksi jumlah pengeluaran harian per obat untuk horizon 7 hingga 14 hari ke depan menggunakan algoritma XGBoost.
* **Konsolidasi Multi-Supplier:** Menggabungkan varian nama obat yang berbeda supplier (misalnya varian `(DEXA)` dan `(NOVELL)`) menjadi satu entitas obat gabungan dan menjumlahkan kuantitas hariannya.
* **Pelatihan Model Terpisah Per Kategori Pareto:** Melatih model mandiri untuk setiap Kategori Pareto (Kategori A terpisah dari Kategori B/C) guna mencegah distorsi pola antara obat volume besar dan obat terputus-putus (*zero-inflated*).
* **Target Evaluasi:** Mencapai skor *Mean Absolute Error* (MAE), *Root Mean Squared Error* (RMSE), dan *Weighted Absolute Percentage Error* (WAPE) yang lebih rendah dibandingkan *Baseline Model* (*Simple Moving Average* 7 hari).

### 1.3 Tech Stack (Python Ecosystem)
* **Data Manipulation & Preprocessing:** `pandas`, `numpy`
* **Machine Learning (Core Model):** `xgboost` (`XGBRegressor`), `scikit-learn` (untuk *preprocessing* & *metrics*)
* **Model Interpretability:** `shap` & *feature importance* bawaan XGBoost (dapat dipilih per model kategori)
* **Model Persistence:** `joblib`
* **Dashboard & Visualisasi:** `streamlit`, `matplotlib`

---

## 2. Aturan Bisnis & Teknis (System & Modeling Rules)

### Rule 1: Data Cleaning, Sanitasi CSV, & Penggabungan Supplier
1. **Hapus Header Duplikat:** Dataset mentah memiliki baris *header* yang berulang di tengah data (contoh: muncul baris `tanggal,ID...` di tengah transaksi). Sistem wajib membuang baris di mana kolom `tanggal == 'tanggal'` sebelum konversi tipe data.
2. **Pemisahan Kolom:** Jika file CSV terbaca sebagai satu kolom teks utuh yang dipisahkan koma, sistem harus memecahnya sesuai *delimiter* koma (`,`).
3. **Standarisasi Kolom:** Menggunakan kamus alias (`COLUMN_ALIASES`) untuk memetakan nama kolom CSV mentah ke standar internal (`tanggal`, `id_obat`, `nama_obat`, `qty_keluar`, `kategori_pareto`).
4. **Standarisasi Tipe Data:**
   * `tanggal` $\rightarrow$ `datetime64[ns]` (diurutkan dari terlama ke terbaru).
   * `id_obat` & `nama_obat` $\rightarrow$ `string`.
   * `kategori_pareto` $\rightarrow$ `string` (bersih dari spasi, huruf kapital `A`, `B`, `C`).
   * `qty_keluar` $\rightarrow$ `float32` (nilai negatif akibat retur di-*clip* menjadi `0`).
5. **Penggabungan Varian Supplier (`base_name`):**
   * Fungsi `base_name` memotong kurung sufiks nama supplier di akhir nama obat (contoh: `ACARBOSE 100 MG TAB (DEXA)` dan `ACARBOSE 100 MG TAB (NOVELL)` menjadi `ACARBOSE 100 MG TAB`).
   * Awalan kurung seperti `(N5) NACL ...` atau `(OTSUSALIN 3) ...` tidak dipotong.
   * Nilai ID dan Nama asli sebelum penggabungan disimpan di kolom `id_asli` dan `nama_asli`.
   * Nama gabungan dijadikan sebagai `id_obat` dan `nama_obat` aktif.
   * Sediakan fungsi `group_mapping(source)` untuk menampilkan tabel verifikasi pemetaan supplier ke obat gabungan.

### Rule 2: Pemetaan Pareto Bawaan, Agregasi Harian, & Zero-Filling
1. **Resolusi Label Pareto Berdasarkan Volume Terbesar:** Jika setelah penggabungan varian obat terdapat beberapa label Pareto (misal DEXA berlabel B, NOVELL berlabel A), sistem **WAJIB** menetapkan label Pareto dari varian dengan **total volume transaksi terbesar** (`total_qty`), bukan tanggal terbaru.
2. **Agregasi Harian:** Jumlahkan (`SUM`) kolom `qty_keluar` per `tanggal` dan `id_obat`. Penjumlahan transaksi antar-supplier terjadi otomatis pada tahap ini.
3. **Complete Date Grid (Zero-Filling):** Buat rentang tanggal harian penuh (`min_date` s/d `max_date`) untuk setiap `id_obat`. Tanggal tanpa transaksi wajib diisi `qty_keluar = 0`, lalu gabungkan kembali (*merge*) dengan tabel referensi Pareto hasil resolusi volume terbesar.

### Rule 3: Pelatihan Terpisah Per Kategori Pareto di XGBoost (Krusial)
Model **DILARANG dicampur menjadi satu model tunggal untuk seluruh kategori**. Pelatihan model XGBoost dilakukan secara mandiri per Kategori Pareto:
1. **Model Kategori A (Volume Besar / Fast-Moving):**
   * Dilatih khusus menggunakan subset data obat Kategori A.
   * Menggunakan `objective='reg:squarederror'`.
2. **Model Kategori B & C (Volume Kecil / Slow-Moving / Intermittent):**
   * Dilatih khusus menggunakan subset data obat Kategori B (dan C).
   * Menggunakan `objective='reg:tweedie'` dengan `tweedie_variance_power=1.5` yang dirancang khusus untuk distribusi data banyak nilai nol (*zero-inflated*).
3. **Batas Minimal Pohon (*Tree Constraint*):**
   * Pada tahap penentuan pohon optimal via *early stopping* (`fit_with_early_stopping`), tetapkan batas minimal:
     ```python
     MIN_TREES = 150
     best = int(getattr(m, "best_iteration", m.n_estimators - 1)) + 1
     return max(best, MIN_TREES)
     ```
   * Hal ini mencegah model berhenti terlalu dini pada pohon dangkal (misal 30-40 pohon) akibat *noise* pada validasi time-series berdurasi pendek.
4. **Penyimpanan Multi-Model:** Objek model, objective, dan jumlah pohon disimpan dalam dictionary per kategori (misal `models={"A": model_A, "B": model_B}`) di dalam file `models/xgb_forecaster.joblib`.

### Rule 4: Pembatasan Feature Engineering (< 6 Bulan Data)
* **DILARANG** membuat fitur musiman tahunan (`month`, `year`, `lag_90`, `lag_180`) karena durasi data < 6 bulan.
* **WAJIB** membatasi *Lag* dan *Rolling Window* maksimal **21–28 hari**:
  * **Lag Features (per `id_obat`):** `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_21`.
  * **Rolling Features (Wajib `.shift(1)` sebelum `.rolling()` agar bebas *data leakage*):**
    * `rolling_mean_7`, `rolling_std_7`
    * `rolling_mean_14`, `rolling_max_14`
    * `rolling_mean_28`
  * **Calendar Features:** `day_of_week` (0–6), `is_weekend` (0/1), `day_of_month` (1–31).
  * **Static Features:** `id_obat_encoded`, `kategori_pareto_encoded`.

### Rule 5: Validasi, Splitting, & Metrik Evaluasi
1. **DILARANG Random Split:** Wajib menggunakan **Time-Based Split**.
   * **Train Set:** Data awal s/d `max_date - 14 hari`.
   * **Validation/Test Set:** 14 hari terakhir dari dataset.
2. **Dua Tingkat Evaluasi:**
   * **Evaluasi One-Step-Ahead:** Mengukur performa model jika seluruh lag berasal dari data aktual.
   * **Evaluasi Recursive 14 Hari:** Mengukur performa simulasi lapangan saat model meramal 14 hari ke depan secara berantai.
3. **Metrik Evaluasi:**
   * Hitung **MAE**, **RMSE**, dan **WAPE (%)** secara keseluruhan (**Overall**), serta per masing-masing **Kategori A**, **Kategori B**, dan **Kategori C**.
   * Bandingkan langsung terhadap **Baseline SMA-7** (*Simple Moving Average* 7 hari terakhir).
   * **DILARANG** menggunakan MAPE standar karena adanya nilai aktual `0` pada pengeluaran harian.

### Rule 6: Peramalan Multi-Langkah (*Recursive Forecasting*)
1. Prediksi dilakukan berurutan hari demi hari sepanjang horizon (misal 14 hari ke depan).
2. Hasil prediksi hari $t$ dimasukkan kembali ke deret historis untuk membentuk lag dan rolling hari $t+1$.
3. **Bungkam Warning Concat:** Pembentukan baris tanggal masa depan dengan nilai `qty_keluar = NaN` wajib dibungkus dengan `warnings.catch_warnings()` agar bersih dari pesan *FutureWarning* pandas:
   ```python
   with warnings.catch_warnings():
       warnings.simplefilter("ignore", FutureWarning)
       hist = pd.concat([hist, meta.assign(tanggal=day, qty_keluar=np.nan)], ignore_index=True)
   ```

### Rule 7: Dashboard Interaktif (`app.py`)
1. **Inspeksi Mapping Supplier:** Sediakan panel `st.expander` yang menampilkan tabel mapping dari `group_mapping` (nama asli, supplier, obat gabungan, dan volume total).
2. **Interpretabilitas Terpisah (SHAP / Feature Importance):** Karena model dipisah per kategori, sediakan seleksi model pada Tab Interpretabilitas agar pengguna dapat menganalisis *feature importance* khusus Model Kategori A atau Model Kategori B.
3. **Format Label Dropdown Bersih:** Jika `id_obat == nama_obat` (hasil penggabungan supplier), tampilkan nama obat secara bersih tanpa pengulangan kode/ID.
4. **Cache Management:** Gunakan kunci cache versi (misal `version=CACHE_VERSION`) dan tombol `🔄 Reset Cache & Latih Ulang` di sidebar untuk menghindari pembacaan data/model lama dari memori Streamlit.

---

## 3. AI Coding Agent Prompt (`AGENT_INSTRUCTIONS.md`)

> **Cara Pakai:** Berikan blok di bawah ini sebagai instruksi pembuka (*System Prompt*) ke AI Coding Agent kamu.

### System Prompt untuk Agent

**Role:**
Kamu adalah Senior Data Scientist dan Machine Learning Engineer spesialis *Time-Series Forecasting* menggunakan **XGBoost** dan Python.

**Project Context:**
Kita membangun sistem prediksi pengeluaran obat harian di Rumah Sakit dengan data historis **kurang dari 6 bulan (< 180 hari)**.
* Dataset memiliki kolom **Kategori Pareto (`A`, `B`, `C`)** bawaan.
* Nama obat sering kali memiliki sufiks supplier di dalam tanda kurung di akhir nama (contoh `ACARBOSE 100 MG TAB (DEXA)` dan `(NOVELL)`). Sistem wajib menggabungkan varian ini menjadi satu nama obat inti (`base_name`) dan menjumlahkan kuantitas hariannya.
* Model **XGBoost dilatih terpisah per Kategori Pareto**: Kategori A menggunakan `reg:squarederror`, sedangkan Kategori B dan C menggunakan `reg:tweedie`.

**Strict Coding Rules (WAJIB DIPATUHI AGENT):**
1. **Penggabungan Varian Supplier & Sanitasi Data:**
   * Di `src/data_loader.py`, gunakan `base_name` untuk memangkas kurung nama supplier di akhir nama obat.
   * Simpan ID dan nama asli di `id_asli` dan `nama_asli`, serta jadikan nama gabungan sebagai `id_obat` dan `nama_obat`.
   * Sediakan fungsi `group_mapping(source)` untuk audit tabel varian ke obat gabungan.
2. **Resolusi Label Pareto Berdasarkan Total Volume:**
   * Jika satu obat gabungan memiliki lebih dari satu label Pareto antar varian supplier, ambil label Pareto dari varian yang memiliki total volume transaksi (`total_qty`) terbesar.
3. **Preservasi Kategori Pareto Saat Zero-Filling:**
   * Bentuk *complete date grid* (`pd.MultiIndex.from_product`), isi tanggal tanpa transaksi dengan `qty_keluar = 0`, lalu *merge* kembali label Pareto hasil resolusi volume terbesar.
4. **Anti-Data Leakage pada Rolling Window:**
   * Wajib menerapkan `.shift(1)` terlebih dahulu di dalam `groupby('id_obat')` sebelum memanggil `.rolling()`.
5. **Batas Horizon Fitur (< 6 Bulan Data):**
   * Jangan membuat fitur *lag* atau *rolling* lebih dari 28 hari agar data latih tidak habis terpotong `NaN`.
6. **Pelatihan Model Mandiri Per Kategori Pareto:**
   * Di `src/train_xgb.py`, lakukan iterasi per kategori Pareto (`for cat in categories:`).
   * Latih model terpisah untuk masing-masing kategori:
     * Kategori A $\rightarrow$ `objective='reg:squarederror'`.
     * Kategori B/C $\rightarrow$ `objective='reg:tweedie'` (`tweedie_variance_power=1.5`).
   * Terapkan batas minimal `MIN_TREES = 150` pada fungsi `fit_with_early_stopping`.
   * Simpan seluruh model dalam satu dictionary pada file model joblib.
7. **Recursive Multi-Step Forecasting (7–14 Hari):**
   * Gunakan peramalan berantai (*recursive*) di `src/forecast.py` dan bungkus penambahan baris tanggal masa depan dengan `warnings.catch_warnings()` untuk mengabaikan `FutureWarning`.
8. **Evaluasi Per Kategori Pareto:**
   * Jangan pernah menggunakan MAPE standar.
   * Hitung dan tampilkan **MAE**, **RMSE**, dan **WAPE (%)** untuk **Overall** dan per masing-masing **Kategori A**, **B**, dan **C** pada evaluasi One-Step-Ahead dan Recursive 14 Hari, lalu bandingkan dengan Baseline SMA-7.
9. **Dashboard Streamlit (`app.py`):**
   * Sediakan panel expander `group_mapping`.
   * Sediakan seleksi model pada Tab Interpretabilitas (SHAP/Gain) per kategori Pareto.
   * Tambahkan tombol reset cache di sidebar dan terapkan penamaan cache versi agar bebas dari *stale memory cache*.

**Struktur Direktori Proyek:**
```text
drug_forecasting_xgb/
├── data/
│   ├── raw/                  # CSV mentah dari RS
│   └── processed/            # Data bersih zero-filled & forecast CSV
├── src/
│   ├── data_loader.py        # Pembersihan header, penggabungan varian supplier, & zero-filling
│   ├── features.py           # Lag (1-21), Rolling shifted (7-28), Calendar, & Pareto encoding
│   ├── train_xgb.py          # Pelatihan mandiri per kategori Pareto & evaluasi metrik
│   ├── forecast.py           # Recursive multi-step & baseline forecasting
│   └── explain.py            # Interpretabilitas fitur (SHAP / feature importance)
├── models/
│   └── xgb_forecaster.joblib # Dictionary model XGBoost per kategori & parameter
├── app.py                    # Dashboard Streamlit (visualisasi, evaluasi, SHAP per kategori)
├── README.md                 # Dokumentasi teknis proyek
├── prd n rules.md            # PRD, Aturan Sistem, & Agent System Prompt
└── requirements.txt          # Daftar dependensi library Python
```
