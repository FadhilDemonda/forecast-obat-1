"""Ablation kalender pada beberapa holdout 14 hari; tidak menyimpan ulang model."""
import argparse
from pathlib import Path

import pandas as pd

from .data_loader import load_daily
from .schedule import DEFAULT_SCHEDULE_CSV
from .train_xgb import TEST_DAYS, run_pipeline


def benchmark_calendar(daily, schedule=DEFAULT_SCHEDULE_CSV, folds=3, calendar_overrides=None):
    if folds < 1:
        raise ValueError("Jumlah fold minimal 1.")
    rows = []
    for fold in range(folds):
        end = daily.tanggal.max() - pd.Timedelta(days=fold * TEST_DAYS)
        subset = daily[daily.tanggal <= end].copy()
        if subset.tanggal.nunique() < 60:
            raise ValueError("Training terlalu pendek untuk jumlah fold yang diminta (minimal 60 hari per fold).")
        for enabled in (False, True):
            result = run_pipeline(subset, TEST_DAYS, schedule=schedule,
                                  include_calendar=enabled, calendar_overrides=calendar_overrides)
            metrics = result["metrics_recursive"].query("model == 'XGBoost'").copy()
            metrics["calendar_enabled"] = enabled
            metrics["split_date"] = result["split_date"]
            metrics["holdout_end"] = end
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/groubynama+spesialis+dokter.csv")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_CSV)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--calendar-overrides")
    parser.add_argument("--output", default="data/processed/calendar_benchmark.csv")
    args = parser.parse_args()
    result = benchmark_calendar(load_daily(args.csv), args.schedule, args.folds, args.calendar_overrides)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.pivot(index=["holdout_end", "kategori"], columns="calendar_enabled", values="WAPE (%)").round(2).to_string())
    print(f"Hasil: {args.output}; model dan forecast aktif tidak diubah.")
