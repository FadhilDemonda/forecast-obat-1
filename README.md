# 💊 Drug Forecasting XGBoost

## Benchmark jendela training 90/120/180 hari

```bash
python -m src.benchmark_windows --windows 90 120 180 --folds 2
```

Benchmark ini menggunakan **dataset dashboard default**
`data/raw/groubynama+spesialis+dokter.csv`, bukan file satu tahun pada benchmark
LSTM di bawah. Skor jadwal historis tetap dipakai; fitur libur tambahan tidak
dipakai. Hanya jumlah hari training yang berubah. Tuning memakai 14 hari akhir
training masing-masing varian; prediksi holdout recursive 14 hari. Histori
pemanasan maksimal 28 hari tidak ikut training, tetapi dipakai menghitung fitur.

Pada histori 214 hari, dua holdout yang sama dapat diuji: 3–16 September dan
17–30 September 2026. Training 180 hari hanya mempunyai pemanasan 6/20 hari;
training 90/120 hari mempunyai 28 hari. Sebagian fitur awal varian 180 hari
berbasis histori parsial atau kosong. Jadi perbandingan ini bersifat eksploratif,
belum benchmark dengan pemanasan penuh yang setara. Lebih banyak histori perlu
untuk beberapa fold dengan pemanasan lengkap.

Keluaran: `data/processed/window_benchmark.csv` (per fold, termasuk batas training
dan panjang warmup) dan `data/processed/window_benchmark_summary.csv` (WAPE pooled,
rasio total error absolut terhadap total aktual, bukan rata-rata WAPE fold).
Label Pareto dan universe obat mengikuti loader dashboard; belum dibekukan ulang
per cutoff. Keterbatasan ini sama untuk seluruh varian. Memilih jendela berdasarkan
holdout ini memerlukan validasi lanjutan pada periode baru. Benchmark tidak
mengganti default dashboard, model tersimpan, atau forecast aktif.

## Benchmark XGBoost vs LSTM: data penjualan satu tahun

Benchmark terpisah memakai `data/raw/penjualan-10-2025.csv`, tanpa mengubah model
dashboard, `models/xgb_forecaster.joblib`, atau `data/processed/forecast.csv`.

```bash
python -m pip install -r requirements-benchmark.txt
python -m src.benchmark_models --folds 3 --horizon 14 --epochs 20
python -m unittest discover -s tests -v
```

Jika pemasangan PyTorch pada Windows bermasalah, gunakan wheel CPU:
`python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.

Protokol perbandingan:

* Tiga holdout berurutan, masing-masing 14 hari, expanding training window.
  Semua prediksi recursive: aktual holdout tidak diumpankan kembali ke model.
* Obat, label Pareto dominan menurut volume, dan lookup ditetapkan dari data
  training setiap fold; obat yang baru muncul pada test dikecualikan dan dicatat.
  Label A/B/C bawaan tidak dihitung ulang. Asal waktu label sumber tetap perlu diverifikasi.
* Model terpisah per A/B/C; XGBoost memakai konfigurasi proyek tanpa jadwal,
  konteks klinis, atau tambahan libur. Kalender dasar tersedia bagi kedua model.
  XGBoost memakai lag/rolling maksimal 28 hari; LSTM memakai sequence 28 hari,
  embedding obat, kalender historis dan kalender hari target.
* Warmup 28 hari sama. Pemilihan jumlah pohon/epoch memakai 14 hari terakhir
  **training**, evaluasi one-step, kemudian refit seluruh training. Test tidak
  digunakan memilih konfigurasi. Validasi ini bukan optimasi recursive loss.
* LSTM: hidden size 32, embedding 8, Adam 0.001, batch 512, maksimal 20 epoch,
  patience 4, seed 42. Kuantitas dibagi rata-rata per obat dari training saja
  (minimal skala 1). Loss MSE untuk A dan Tweedie p=1.5 untuk B/C; normalisasi
  LSTM membuat bobot relatif antarobat berbeda dari XGBoost. Ini perbandingan
  pipeline praktis, bukan eksperimen arsitektur dengan loss identik.
* Baseline SMA-7 tetap datar sepanjang horizon. MAE, RMSE, WAPE dihitung Overall
  dan A/B/C. Ringkasan pooled memakai seluruh prediksi, bukan rata-rata WAPE fold.
  Satu seed dan konfigurasi kecil belum mewakili seluruh potensi LSTM/XGBoost.

Keluaran di `data/processed/benchmark_xgb_lstm/`: `metrics_folds.csv`,
`metrics_pooled.csv`, `predictions.csv`, `training.csv`, dan `metadata.json`
(konfigurasi, versi pustaka, hash sumber, obat yang dikecualikan, batasan).

**Validasi sumber:** file mencakup 1 Oktober 2025–30 September 2026, tepat
100.000 baris dan 590 ID mentah terisi. Ada 473 baris tanpa ID/Pareto (26 nama
obat); loader memakai nama sebagai ID dan mengisi kategori C. Total menjadi
616 obat setelah normalisasi pada file ini. Akhir file berada pada Januari 2026 untuk obat
terakhir; ini indikasi kemungkinan ekspor terpotong, bukan bukti pasti. Konfirmasi
ekspor lengkap sebelum memakai hasil untuk keputusan stok. Zero-fill berasumsi
tanggal tanpa transaksi berarti nol, bukan transaksi yang hilang. Nama supplier
dengan kurung tidak ditutup dipertahankan oleh loader lama; jumlah obat hasil
normalisasi dicatat terpisah. Tiga fold berdekatan dan satu tahun histori tidak
membuktikan generalisasi musiman tahunan.

### Hasil benchmark penjualan satu tahun

Hasil pooled tiga holdout recursive (20 Agustus–30 September 2026), seed 42:

| Model | MAE Overall | RMSE Overall | WAPE Overall | WAPE A | WAPE B | WAPE C |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 36.63 | 140.86 | 57.75% | 51.61% | 53.04% | 72.59% |
| LSTM | 39.06 | 143.55 | 61.57% | 52.23% | 59.62% | 77.36% |
| Baseline SMA-7 | 40.95 | 156.28 | 64.55% | 57.95% | 62.55% | 76.50% |

| Periode holdout | XGBoost WAPE | LSTM WAPE | SMA-7 WAPE |
|---|---:|---:|---:|
| 20 Agustus–2 September 2026 | 54.99% | 62.01% | 63.86% |
| 3–16 September 2026 | 60.29% | 62.33% | 65.16% |
| 17–30 September 2026 | 58.02% | 60.49% | 64.64% |

XGBoost unggul pada WAPE Overall di ketiga fold dan pada WAPE pooled tiap
kategori. LSTM tidak selalu kalah pada setiap metrik/subset: RMSE Overall fold
2/3 dan WAPE A fold 2/3 lebih rendah. WAPE pooled LSTM kategori C bahkan lebih
tinggi daripada SMA-7. Belum ada model yang mencapai WAPE 15%; hasil masih
sementara karena kelengkapan ekspor belum terverifikasi. Angka ini tidak
sebanding langsung dengan benchmark lama yang memakai dataset dan fitur berbeda.

## Default data dan skor jadwal spesialis

### Fitur kalender Indonesia

Fitur tambahan **nonaktif secara default** (opsional): `is_public_holiday`, `is_collective_leave`,
`days_until_holiday`, `days_since_holiday`, dan `is_ramadan`. Fitur lama
`day_of_week`, `is_weekend`, dan `day_of_month` tetap tersedia. Tidak menambah
fitur musiman tahunan atau lag lebih dari 28 hari. Jarak dihitung terhadap libur
nasional/cuti bersama, termasuk hari libur itu sendiri (jarak 0), dibatasi 28 hari.

Sumber kalender: `holidays==0.107`, Indonesia kategori PUBLIC dan GOVERNMENT.
Ramadan adalah **estimasi Umm al-Qura** dari `hijridate==2.6.0`, bukan keputusan
sidang isbat Indonesia. Kalender pustaka bukan arsip informasi yang tersedia
pada setiap tanggal forecast historis; hasil backtest harus ditafsirkan dengan
batasan ini. Koreksi sesuai keputusan resmi dapat diunggah melalui sidebar atau
CLI. Semua fitur tersedia pada training, recursive forecast, dan SHAP.

CSV koreksi opsional (flag 0/1; sel kosong mempertahankan default):

```csv
tanggal,is_public_holiday,is_collective_leave,is_ramadan
2026-03-01,,,1
```

Kolom opsional `is_clinic_closed` hanya ditambahkan jika disuplai; harus lengkap
untuk **seluruh tanggal training, holdout, dan horizon forecast**. Tidak ada
asumsi bahwa libur nasional berarti RS/klinik tutup. Jika status berbeda antar
klinik, jangan memakai satu flag global tanpa agregasi operasional yang jelas.
Unduhan kalender efektif tersedia di dashboard; konten upload ikut kunci cache.

```bash
python -m pip install -r requirements.txt
python -m src.train_xgb
python -m src.train_xgb --with-calendar
python -m src.train_xgb --without-calendar
python -m src.train_xgb --with-calendar --calendar-overrides data/raw/kalender_rs.csv
python -m src.benchmark_calendar --folds 3
```

Benchmark memakai data obat, fitur lain, split temporal dan konfigurasi model
yang sama, dengan tuning early stopping yang sama untuk masing-masing varian.
Tidak mengubah model/forecast aktif. Hasil WAPE recursive Overall:

| Akhir holdout 14 hari | Tanpa tambahan kalender | Dengan tambahan kalender |
|---|---:|---:|
| 2026-09-02 | 32.49% | 31.74% |
| 2026-09-16 | 28.01% | 28.13% |
| 2026-09-30 | 28.66% | 29.07% |

Kalender belum menunjukkan peningkatan konsisten dan belum mencapai 15%.
Holdout terakhir memburuk sedikit. Tabel hasil lengkap disimpan di
`data/processed/calendar_benchmark.csv`. Tiga fold ini masih mencakup periode
berdekatan, bukan validasi seluruh musim. Feature importance tidak membuktikan
peningkatan akurasi; gunakan ablation dan lebih banyak periode sebelum memilih
konfigurasi produksi. Checkbox kalender default mati; aktifkan untuk eksperimen.
`day_of_week`, `is_weekend`, `day_of_month`, dan skor jadwal dokter tetap aktif.

### Jadwal per dokter untuk forecast

Streamlit dan CLI kini secara default membaca `data/raw/kategoriparetodokter.csv`
dan `data/raw/jadwalpraktekdokterforecast.csv`. Pasangan nama tersensor dan
spesialis dinormalisasi, lalu diberi ID deterministik berbasis SHA-256. ID tetap
sama meskipun urutan baris berubah; perubahan nama atau spesialis menghasilkan ID
berbeda. ID ini bukan verifikasi identitas asli dan tidak dipakai sebagai fitur numerik.

Kategori dokter A/B/C dibobot 3/2/1 dan dijumlahkan per tanggal/spesialis. Skor ini
hanya mengganti jadwal **setelah tanggal penjualan terakhir**. Training dan holdout
tetap memakai skor historis, sehingga WAPE holdout tidak berubah. Belum ada fitur
jumlah dokter A/B/C karena histori terperinci belum tersedia.

Roster diasumsikan lengkap untuk setiap tanggal tercantum: spesialis tanpa baris
pada tanggal tersebut dihitung nol. Tanggal yang sama sekali tidak tercantum tidak
dianggap nol dan tidak diganti diam-diam dengan skor lama. Spesialis baru yang tidak
ada dalam skema training ditolak. Periksa kelengkapan roster sebelum forecast.
Tanggal 30 September 2026 yang tidak tercantum pada roster tetap memakai skor
historis karena termasuk periode aktual. Roster default mencakup forecast 1–14
Oktober 2026. Jadwal dan Pareto dokter harus sudah diketahui sebelum forecast dibuat.

CLI menulis `data/processed/master_dokter_id.csv` dan
`data/processed/jadwal_dokter_id.csv`; Streamlit menyediakan unduhan keduanya.
Upload kedua sumber pengganti tersedia di sidebar. Nonaktifkan checkbox jadwal
dokter atau gunakan `--without-doctor-forecast` untuk memakai skor jadwal lama saja.

Streamlit dan CLI memakai `data/raw/groubynama+spesialis+dokter.csv` sebagai
data penjualan default, serta `data/raw/skorspesialis2025.csv` sebagai jadwal.
Skor per spesialis (bobot dokter A/B/C = 3/2/1) dan `total_skor` digabung
berdasarkan tanggal tanpa menggandakan kuantitas. Kolom model memakai awalan
`skor_jadwal_`, termasuk pada SHAP. Skor hari prediksi dipakai langsung karena
jadwal diasumsikan sudah diketahui sebelum forecast, bukan berasal dari transaksi
hari tersebut. Kolom spesialis yang tidak ada tidak dibuat menjadi skor nol.

Jadwal harus mencakup seluruh tanggal training, holdout, dan horizon forecast.
Tanggal hilang, tanggal duplikat, skor negatif/kosong, serta total yang tidak
sesuai akan ditolak. Streamlit menyediakan upload jadwal pengganti dan menampilkan
rentang tanggalnya. Perubahan isi jadwal ikut membatalkan cache hasil training.

```bash
python -m src.train_xgb
python -m src.train_xgb --schedule data/raw/skorspesialis2025.csv
python -m src.train_xgb --without-schedule
```

Perbandingan satu holdout recursive 14 hari pada file penjualan yang sama
(187 obat, split 2026-09-16):

| Kategori | Tanpa skor jadwal, WAPE (%) | Dengan skor jadwal, WAPE (%) |
|---|---:|---:|
| Overall | 33.88 | 28.66 |
| A | 31.37 | 25.74 |
| B | 31.72 | 25.01 |
| C | 47.39 | 46.88 |

Hasil ini belum merupakan rolling backtest. Validitasnya bergantung pada jadwal
dan kategori Pareto dokter yang memang tersedia pada tanggal asal forecast.
Kategori dokter berdasarkan kunjungan masa depan akan menyebabkan leakage.
Menjalankan CLI menyimpan ulang artefak model dan CSV keluaran sesuai dataset aktif.

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
python -m src.train_xgb --csv data/raw/groubynama.csv --horizon 14
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

## 📊 Hasil Evaluasi Terkini (Dataset 214 Hari - `groubynama.csv`)

Dengan pemisahan model per kategori dan penggabungan varian supplier, XGBoost konsisten mengungguli Baseline SMA-7 di seluruh kategori (A, B, C):

| Model | Kategori | Objective | MAE | RMSE | WAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recursive 14d)** | **Overall** | **Campuran** | **34.63** | **148.76** | **39.64%** |
| **XGBoost (Recursive 14d)** | **A** | `squarederror` | **116.66** | **334.06** | **30.90%** |
| **XGBoost (Recursive 14d)** | **B** | `tweedie` | **56.89** | **163.70** | **39.23%** |
| **XGBoost (Recursive 14d)** | **C** | `tweedie` | **14.42** | **72.22** | **67.49%** |
| Baseline SMA-7 | Overall | - | 41.30 | 182.89 | 47.27% |
| Baseline SMA-7 | A | - | 150.65 | 439.54 | 39.90% |
| Baseline SMA-7 | B | - | 66.62 | 175.65 | 45.93% |
| Baseline SMA-7 | C | - | 15.49 | 78.23 | 72.52% |

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
