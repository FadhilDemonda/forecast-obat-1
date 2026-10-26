"""Validasi dan penggabungan skor jadwal yang diketahui sebelum forecast."""
import re
import hashlib

import numpy as np
import pandas as pd

DEFAULT_SCHEDULE_CSV = "data/raw/skorspesialis2025.csv"
DEFAULT_DOCTOR_PARETO_CSV = "data/raw/kategoriparetodokter.csv"
DEFAULT_DOCTOR_FORECAST_CSV = "data/raw/jadwalpraktekdokterforecast.csv"
SCHEDULE_PREFIX = "skor_jadwal_"


def load_schedule(source=DEFAULT_SCHEDULE_CSV) -> pd.DataFrame:
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if "tanggal" not in frame or "total_skor" not in frame:
        raise ValueError("Jadwal wajib memiliki kolom tanggal dan total_skor.")
    frame["tanggal"] = pd.to_datetime(frame["tanggal"], errors="coerce")
    if frame.empty or frame["tanggal"].isna().any() or frame["tanggal"].duplicated().any():
        raise ValueError("Tanggal jadwal kosong, tidak valid, atau duplikat.")
    names = {}
    for col in frame.columns.drop("tanggal"):
        name = re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        names[col] = SCHEDULE_PREFIX + name
    if len(set(names.values())) != len(names):
        raise ValueError("Nama spesialis jadwal bertabrakan setelah normalisasi.")
    values = frame[list(names)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Skor jadwal wajib berupa angka non-negatif dan tidak kosong.")
    specialties = [c for c in values if c != "total_skor"]
    if not specialties or not np.allclose(values[specialties].sum(axis=1), values["total_skor"]):
        raise ValueError("total_skor tidak sama dengan jumlah skor seluruh spesialis.")
    frame[list(names)] = values.astype("float32")
    return frame.rename(columns=names).sort_values("tanggal").reset_index(drop=True)


def _doctor_keys(frame):
    frame = frame.copy()
    for col in ("SPESIALIS", "dokter_sensor"):
        if col not in frame or frame[col].isna().any():
            raise ValueError(f"Kolom dokter wajib tersedia dan tidak kosong: {col}")
        frame[col] = frame[col].astype(str).str.normalize("NFKC").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        if frame[col].eq("").any():
            raise ValueError(f"Kolom dokter kosong: {col}")
    return frame


def build_doctor_schedule(pareto_source, practice_source):
    """ID pseudonim deterministik; pasangan tersensor bukan identitas terverifikasi."""
    master = _doctor_keys(pd.read_csv(pareto_source))
    keys = ["SPESIALIS", "dokter_sensor"]
    if master.empty or master.duplicated(keys).any():
        raise ValueError("Master dokter kosong atau pasangan nama dan spesialis duplikat.")
    if "kategori_pareto" not in master:
        raise ValueError("Master dokter wajib memiliki kategori_pareto.")
    master["kategori_pareto"] = master["kategori_pareto"].astype(str).str.strip().str.upper()
    if not master["kategori_pareto"].isin(["A", "B", "C"]).all():
        raise ValueError("Pareto dokter wajib A, B, atau C.")
    master["id_dokter"] = [
        "D_" + hashlib.sha256((s + "\x1f" + n).encode()).hexdigest()[:20]
        for s, n in master[keys].itertuples(index=False, name=None)
    ]
    if master.id_dokter.duplicated().any():
        raise ValueError("ID dokter bertabrakan.")
    practice = _doctor_keys(pd.read_csv(practice_source))
    if "tanggal" not in practice:
        raise ValueError("Jadwal dokter wajib memiliki tanggal.")
    practice["tanggal"] = pd.to_datetime(practice["tanggal"], errors="coerce")
    if practice.empty or practice.tanggal.isna().any() or practice.duplicated(["tanggal"] + keys).any():
        raise ValueError("Jadwal dokter kosong, tanggal tidak valid, atau dokter duplikat per hari.")
    practice = practice.merge(master[keys + ["id_dokter", "kategori_pareto"]], on=keys,
                              how="left", validate="many_to_one")
    if practice.id_dokter.isna().any():
        raise ValueError("Ada dokter jadwal yang tidak cocok dengan master Pareto.")
    practice["bobot"] = practice.kategori_pareto.map({"A": 3, "B": 2, "C": 1})
    scores = practice.pivot_table(index="tanggal", columns="SPESIALIS", values="bobot",
                                  aggfunc="sum", fill_value=0).reset_index()
    scores["total_skor"] = scores.drop(columns="tanggal").sum(axis=1)
    return master, practice, load_schedule(scores)


def combine_forecast_schedule(historical, doctor_scores, last_actual_date):
    """Historis tak berubah; roster lengkap hanya mengganti tanggal setelah aktual."""
    future = doctor_scores[doctor_scores.tanggal > pd.Timestamp(last_actual_date)].copy()
    cols = schedule_columns(historical)
    unknown = set(schedule_columns(future)) - set(cols)
    if unknown:
        raise ValueError(f"Spesialis jadwal forecast tidak dikenal model: {sorted(unknown)}")
    if future.empty:
        return historical.copy()
    # Roster dianggap lengkap pada tanggal yang tercantum; tanggal hilang tetap hilang.
    future = future.reindex(columns=["tanggal"] + cols, fill_value=0)
    future["skor_jadwal_total_skor"] = future[[c for c in cols if c != "skor_jadwal_total_skor"]].sum(axis=1)
    future[cols] = future[cols].astype(historical[cols].dtypes.to_dict())
    # Jangan gunakan skor lama sebagai pengganti tanggal hilang dalam horizon baru.
    past = historical[historical.tanggal <= pd.Timestamp(last_actual_date)]
    return pd.concat([past, future], ignore_index=True).sort_values("tanggal").reset_index(drop=True)


def schedule_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if c.startswith(SCHEDULE_PREFIX)]


def require_schedule_dates(schedule: pd.DataFrame, dates) -> None:
    missing = pd.DatetimeIndex(dates).unique().difference(schedule["tanggal"])
    if len(missing):
        preview = ", ".join(missing.strftime("%Y-%m-%d")[:5])
        raise ValueError(f"Jadwal belum tersedia untuk {len(missing)} tanggal: {preview}. "
                         "Lengkapi CSV jadwal; tanggal kosong tidak dianggap skor nol.")


def attach_schedule(daily: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    require_schedule_dates(schedule, daily["tanggal"])
    attrs = daily.attrs.copy()
    # Jadwal hanya satu baris per tanggal: join tidak boleh menggandakan qty.
    result = daily.drop(columns=schedule_columns(daily)).merge(
        schedule, on="tanggal", how="left", validate="many_to_one", sort=False
    )
    result.attrs = attrs
    return result
