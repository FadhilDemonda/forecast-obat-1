"""Kalender Indonesia, tanpa memakai transaksi atau menganggap RS tutup saat libur."""
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from hijridate import Gregorian
from holidays.constants import GOVERNMENT, PUBLIC

CALENDAR_FEATURES = [
    "is_public_holiday", "is_collective_leave", "days_until_holiday",
    "days_since_holiday", "is_ramadan",
]
OPTIONAL_FEATURES = ["is_clinic_closed"]


def load_calendar_overrides(source=None):
    """CSV opsional: tanggal, flag 0/1; sel kosong mempertahankan default."""
    if source is None:
        return None
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    allowed = ["is_public_holiday", "is_collective_leave", "is_ramadan", "is_clinic_closed"]
    if "tanggal" not in frame or not set(frame.columns).issubset(["tanggal"] + allowed):
        raise ValueError("Override kalender: gunakan tanggal dan flag libur/Ramadan/tutup klinik saja.")
    frame["tanggal"] = pd.to_datetime(frame["tanggal"], errors="coerce")
    if frame.tanggal.isna().any() or frame.tanggal.duplicated().any():
        raise ValueError("Tanggal override kalender tidak valid atau duplikat.")
    for col in frame.columns.drop("tanggal"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if (frame[col].notna() & values.isna()).any() or not values.dropna().isin([0, 1]).all():
            raise ValueError(f"Flag kalender {col} wajib 0, 1, atau kosong.")
        frame[col] = values
    return frame


@lru_cache(maxsize=16)
def _default_calendar(first_year, last_year):
    # Tahun tetangga diperlukan untuk jarak libur pada pergantian tahun.
    years = range(first_year - 1, last_year + 2)
    dates = pd.date_range(f"{first_year - 1}-01-01", f"{last_year + 1}-12-31")
    public = holidays.country_holidays("ID", years=years, categories=PUBLIC)
    leave = holidays.country_holidays("ID", years=years, categories=GOVERNMENT)
    frame = pd.DataFrame({"tanggal": dates})
    frame["is_public_holiday"] = [int(d.date() in public) for d in dates]
    frame["is_collective_leave"] = [int(d.date() in leave) for d in dates]
    # Umm al-Qura adalah estimasi kalender, bukan keputusan sidang isbat Indonesia.
    try:
        frame["is_ramadan"] = [int(Gregorian(d.year, d.month, d.day).to_hijri().month == 9) for d in dates]
    except OverflowError as exc:
        raise ValueError("Tanggal di luar jangkauan kalender Hijriah; gunakan periode yang didukung hijridate.") from exc
    return frame


def build_calendar(dates, overrides=None):
    dates = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    if dates.empty:
        return pd.DataFrame(columns=["tanggal"] + CALENDAR_FEATURES)
    frame = _default_calendar(dates.min().year, dates.max().year).copy()
    if overrides is not None:
        frame = frame.set_index("tanggal")
        flags = overrides.set_index("tanggal")
        for col in flags:
            if col not in frame:
                frame[col] = np.nan
            known = flags[col].dropna()
            common = known.index.intersection(frame.index)
            frame.loc[common, col] = known.loc[common]
        frame = frame.reset_index()
    holiday_dates = frame.loc[frame.is_public_holiday.eq(1) | frame.is_collective_leave.eq(1), "tanggal"].to_numpy()
    target = dates.to_numpy()
    before = np.searchsorted(holiday_dates, target, side="right") - 1
    after = np.searchsorted(holiday_dates, target, side="left")
    result = frame.set_index("tanggal").loc[dates].reset_index(names="tanggal")
    if len(holiday_dates):
        previous = holiday_dates[np.clip(before, 0, len(holiday_dates) - 1)]
        following = holiday_dates[np.clip(after, 0, len(holiday_dates) - 1)]
        result["days_since_holiday"] = np.where(before >= 0, np.minimum((target - previous) / np.timedelta64(1, "D"), 28), 28)
        result["days_until_holiday"] = np.where(after < len(holiday_dates), np.minimum((following - target) / np.timedelta64(1, "D"), 28), 28)
    else:
        result["days_since_holiday"] = result["days_until_holiday"] = 28
    if "is_clinic_closed" in result and result.is_clinic_closed.isna().any():
        raise ValueError("Kalender tutup klinik harus mencakup seluruh tanggal training dan forecast; kosong bukan buka.")
    for col in result.columns.drop("tanggal"):
        result[col] = result[col].astype("int8")
    return result
