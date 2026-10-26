"""Bandingkan panjang training; evaluasi saja, tanpa mengganti artefak aktif."""
import argparse
from pathlib import Path

import pandas as pd

from .data_loader import load_daily
from .features import BASE_FEATURES, build_features, make_id_map
from .forecast import recursive_forecast
from .schedule import DEFAULT_SCHEDULE_CSV, attach_schedule, load_schedule, schedule_columns
from .train_xgb import TEST_DAYS, eval_by_category, fit_with_early_stopping, make_model, pick_objective_for_category


def window_data(daily, split, days):
    """Simpan maksimal 28 hari pemanasan; baris pemanasan tidak ikut training."""
    start = split - pd.Timedelta(days=days - 1)
    if start <= daily.tanggal.min():
        raise ValueError("Histori tidak cukup untuk training penuh dan minimal satu hari pemanasan.")
    history = daily[daily.tanggal.between(start - pd.Timedelta(days=28), split)].copy()
    expected = pd.date_range(start, split)
    if len(expected.difference(history.tanggal.unique())):
        raise ValueError("Tanggal training tidak lengkap; gunakan data harian zero-filled.")
    warmup = history.loc[history.tanggal < start, "tanggal"].nunique()
    return history, start, warmup


def benchmark_windows(daily, schedule=DEFAULT_SCHEDULE_CSV, windows=(90, 120, 180), folds=2):
    windows = tuple(windows)
    if folds < 1 or not windows or any(w < 30 or int(w) != w for w in windows) or len(set(windows)) != len(windows):
        raise ValueError("Fold minimal 1; jendela unik berupa bilangan bulat minimal 30 hari.")
    ends = [daily.tanggal.max() - pd.Timedelta(days=i * TEST_DAYS) for i in range(folds)]
    # Validasi seluruh kombinasi sebelum training pertama.
    for end in ends:
        for days in windows:
            window_data(daily, end - pd.Timedelta(days=TEST_DAYS), days)
    if schedule is not None:
        daily = attach_schedule(daily, load_schedule(schedule))
    features = BASE_FEATURES + schedule_columns(daily)
    id_map = make_id_map(daily.id_obat)
    rows, predictions = [], []
    for end in ends:
        split = end - pd.Timedelta(days=TEST_DAYS)
        actual = daily[daily.tanggal.between(split + pd.Timedelta(days=1), end)]
        for days in windows:
            history, start, warmup = window_data(daily, split, days)
            feat = build_features(history, id_map, include_calendar=False)
            train = feat[feat.tanggal >= start]
            if train.lag_1.isna().any():
                raise ValueError("Lag awal training kosong; periksa kelengkapan grid obat-hari.")
            forecasts = []
            for category in sorted(daily.kategori_pareto.unique()):
                part = train[train.kategori_pareto == category]
                objective = pick_objective_for_category(category)
                trees = fit_with_early_stopping(part, objective, features)
                model = make_model(objective, n_estimators=trees)
                model.fit(part[features], part.qty_keluar, verbose=False)
                forecasts.append(recursive_forecast(
                    model, history[history.kategori_pareto == category], TEST_DAYS, id_map,
                    schedule=daily[["tanggal"] + schedule_columns(daily)].drop_duplicates("tanggal")
                    if schedule is not None else None))
            predicted = pd.concat(forecasts, ignore_index=True)
            compared = actual[["tanggal", "id_obat", "kategori_pareto", "qty_keluar"]].merge(
                predicted[["tanggal", "id_obat", "prediksi"]], on=["tanggal", "id_obat"],
                how="left", validate="one_to_one")
            if compared.prediksi.isna().any():
                raise ValueError("Prediksi tidak lengkap pada holdout.")
            compared["train_days"] = days
            compared["holdout_end"] = end
            predictions.append(compared)
            metrics = eval_by_category(compared, "prediksi", "XGBoost")
            metrics["train_days"] = days
            metrics["train_start"] = start
            metrics["split_date"] = split
            metrics["holdout_end"] = end
            metrics["warmup_days"] = warmup
            rows.append(metrics)
            overall = metrics.loc[metrics.kategori == "Overall", "WAPE (%)"].item()
            print(f"Holdout {end.date()} | training {days} hari | warmup {warmup} hari | WAPE {overall:.2f}%", flush=True)
    detail = pd.concat(rows, ignore_index=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    summaries = []
    for days in windows:
        summary = eval_by_category(all_predictions[all_predictions.train_days == days], "prediksi", "XGBoost")
        summary["train_days"] = days
        summary["folds"] = folds
        summaries.append(summary)
    return detail, pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/groubynama+spesialis+dokter.csv")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_CSV)
    parser.add_argument("--windows", nargs="+", type=int, default=[90, 120, 180])
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--output", default="data/processed/window_benchmark.csv")
    args = parser.parse_args()
    detail, summary = benchmark_windows(load_daily(args.csv), args.schedule, args.windows, args.folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print("\nWAPE gabungan seluruh holdout:")
    print(summary.pivot(index="kategori", columns="train_days", values="WAPE (%)").round(2).to_string())
    print(f"Hasil: {output}; model dan forecast aktif tidak diubah.")
