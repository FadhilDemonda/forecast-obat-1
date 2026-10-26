"""Rolling sales-only XGBoost/LSTM/SMA-7 benchmark; active artifacts untouched."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import _read_csv, build_daily, clean_raw
from .features import BASE_FEATURES, build_features, make_id_map
from .forecast import baseline_forecast, recursive_forecast
from .train_xgb import eval_by_category, fit_with_early_stopping, make_model, pick_objective_for_category

SALES_FEATURES = [c for c in BASE_FEATURES if not c.startswith("ctx_")]
LOOKBACK = 28


def prepare_fold(clean, end, horizon=14):
    """Freeze drug universe and Pareto lookup using training data only."""
    split = pd.Timestamp(end) - pd.Timedelta(days=horizon)
    training = clean[clean.tanggal <= split]
    if training.tanggal.nunique() < LOOKBACK + 30:
        raise ValueError("Training minimal 58 hari per fold.")
    daily = build_daily(training)
    if not set(daily.kategori_pareto).issubset({"A", "B", "C"}):
        raise ValueError("Kategori Pareto harus A, B, atau C.")
    meta = daily.drop_duplicates("id_obat")[["id_obat", "kategori_pareto"]]
    test = clean[(clean.tanggal > split) & (clean.tanggal <= end)]
    unknown = sorted(set(test.id_obat) - set(meta.id_obat))
    actual = test.groupby(["tanggal", "id_obat"])["qty_keluar"].sum()
    grid = pd.MultiIndex.from_product([
        pd.date_range(split + pd.Timedelta(days=1), end), meta.id_obat,
    ], names=["tanggal", "id_obat"])
    actual = actual.reindex(grid, fill_value=0).reset_index().merge(meta, on="id_obat", validate="many_to_one")
    return daily, actual, unknown


def summarize_predictions(predictions):
    pooled = pd.concat([
        eval_by_category(group, "prediksi", model)
        for model, group in predictions.groupby("model", sort=False)
    ], ignore_index=True)
    return pooled


def benchmark(clean, folds=3, horizon=14, epochs=20, seed=42, threads=4):
    import torch
    import xgboost
    from .lstm_forecaster import LSTMForecaster

    if folds < 1 or not 1 <= horizon <= 28 or epochs < 1 or threads < 1:
        raise ValueError("folds/epochs/threads harus positif; horizon 1-28 hari.")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    predictions, scores, fits, fold_info = [], [], [], []
    end_date = clean.tanggal.max()
    for fold in range(folds):
        end = end_date - pd.Timedelta(days=(folds - 1 - fold) * horizon)
        daily, actual, unknown = prepare_fold(clean, end, horizon)
        split = daily.tanggal.max()
        print(f"Fold {fold + 1}/{folds}: train sampai {split.date()}, test sampai {end.date()}", flush=True)
        id_map = make_id_map(daily.id_obat)
        feat = build_features(daily, id_map, include_calendar=False)
        # Same 28-day warmup for both models.
        feat = feat[feat.tanggal >= daily.tanggal.min() + pd.Timedelta(days=LOOKBACK)]
        forecasts = {"XGBoost": [], "LSTM": []}
        for category in sorted(daily.kategori_pareto.unique()):
            category_daily = daily[daily.kategori_pareto == category]
            category_features = feat[feat.kategori_pareto == category]
            objective = pick_objective_for_category(category)
            started = time.perf_counter()
            trees = fit_with_early_stopping(category_features, objective, SALES_FEATURES)
            model = make_model(objective, n_estimators=trees)
            model.set_params(n_jobs=threads)
            model.fit(category_features[SALES_FEATURES], category_features.qty_keluar, verbose=False)
            # Recursive lags need at most 28 days; avoid rebuilding entire year each step.
            recent = category_daily[category_daily.tanggal > split - pd.Timedelta(days=LOOKBACK)]
            forecasts["XGBoost"].append(recursive_forecast(model, recent, horizon, id_map))
            fits.append(dict(fold=fold + 1, model="XGBoost", kategori=category,
                             iterations=trees, seconds=time.perf_counter() - started))
            started = time.perf_counter()
            tuning = LSTMForecaster(category, seed=seed).fit(
                category_daily, epochs=epochs, validation_days=14)
            best_epoch = tuning.best_epoch
            lstm = LSTMForecaster(category, seed=seed).fit(category_daily, epochs=best_epoch)
            forecasts["LSTM"].append(lstm.forecast(category_daily, horizon))
            fits.append(dict(fold=fold + 1, model="LSTM", kategori=category,
                             iterations=best_epoch, seconds=time.perf_counter() - started))
            print(f"  {category}: XGBoost {trees} trees; LSTM {best_epoch} epochs", flush=True)
        forecasts["Baseline SMA-7"] = [baseline_forecast(daily, horizon)]
        for name, pieces in forecasts.items():
            forecast = pd.concat(pieces, ignore_index=True)
            comparison = actual.merge(forecast[["tanggal", "id_obat", "prediksi"]],
                                      on=["tanggal", "id_obat"], how="left", validate="one_to_one")
            if not np.isfinite(comparison.prediksi).all():
                raise ValueError(f"Prediksi hilang/tidak finite: {name}")
            comparison = comparison.assign(model=name, fold=fold + 1, split_date=split, holdout_end=end)
            predictions.append(comparison)
            scores.append(eval_by_category(comparison, "prediksi", name).assign(
                fold=fold + 1, split_date=split, holdout_end=end))
        fold_info.append(dict(fold=fold + 1, split_date=str(split.date()),
                              holdout_end=str(end.date()), drugs=daily.id_obat.nunique(),
                              unseen_test_drugs_excluded=unknown,
                              multi_pareto_drugs=daily.attrs.get("multi_pareto_drugs", 0)))
    predictions = pd.concat(predictions, ignore_index=True)
    metadata = dict(seed=seed, folds=fold_info, horizon=horizon, max_lstm_epochs=epochs,
                    threads=threads, lookback=LOOKBACK, python=platform.python_version(),
                    torch=torch.__version__, xgboost=xgboost.__version__,
                    numpy=np.__version__, pandas=pd.__version__, features=SALES_FEATURES,
                    calendar="day_of_week, is_weekend, day_of_month; no holiday/schedule",
                    validation="last 14 training days, one-step; refit full training",
                    lstm="per-category; hidden=32; embedding=8; batch=512; Adam lr=0.001; patience=4; per-drug mean scaling (training only)",
                    caveats=["Zero-fill assumes missing transactions are genuine zero sales.",
                             "Source Pareto labels may contain future information; availability not verified.",
                             "Only one year; nearby folds do not validate annual seasonality.",
                             "Different architectures/preprocessing; no exhaustive hyperparameter search."])
    return pd.concat(scores, ignore_index=True), summarize_predictions(predictions), predictions, pd.DataFrame(fits), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/penjualan-10-2025.csv")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="data/processed/benchmark_xgb_lstm")
    args = parser.parse_args()
    raw = _read_csv(args.csv)
    clean = clean_raw(raw)
    result = benchmark(clean, args.folds, args.horizon, args.epochs, args.seed, args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(["metrics_folds", "metrics_pooled", "predictions", "training"], result[:4]):
        frame.to_csv(output / f"{name}.csv", index=False)
    metadata = result[4]
    metadata.update(source=args.csv, source_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                    raw_rows=len(raw), start=str(clean.tanggal.min().date()),
                    end=str(clean.tanggal.max().date()),
                    raw_drugs=clean.loc[clean.id_asli.astype(str).str.strip().ne(""), "id_asli"].nunique(),
                    cleaned_drugs=clean.id_obat.nunique(),
                    empty_pareto_rows=clean.attrs.get("empty_pareto_rows", 0),
                    empty_pareto_drugs=clean.attrs.get("empty_pareto_drugs", 0))
    if len(raw) % 100000 == 0:
        metadata["caveats"].append("Row count is an exact multiple of 100,000; verify export was not truncated.")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result[1].round(2).to_string(index=False))
    print(f"Hasil: {output}; model dan forecast aktif tidak diubah.")


if __name__ == "__main__":
    main()
