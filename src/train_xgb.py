"""Training XGBoost + evaluasi per kategori Pareto vs baseline SMA-7 (Rule 3 & 5)."""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .data_loader import load_daily
from .features import FEATURES, build_features, make_id_map
from .forecast import baseline_forecast, recursive_forecast

TEST_DAYS = 14
MIN_TREES = 150  # validasi early-stopping kecil & berisik -> jangan biarkan model terlalu dangkal


# ------------------------------------------------------------------ metrics
def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y).sum()
    return float(np.abs(y - p).sum() / d * 100) if d > 0 else float("nan")


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "WAPE (%)": wape(y, p),
    }  # MAPE sengaja TIDAK dipakai (aktual bisa 0)


def eval_by_category(df: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    rows = [{"model": label, "kategori": "Overall", **metrics(df["qty_keluar"], df[pred_col])}]
    for cat, g in df.groupby("kategori_pareto"):
        rows.append({"model": label, "kategori": cat, **metrics(g["qty_keluar"], g[pred_col])})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- model
def pick_objective_for_category(category: str) -> str:
    """Rule 3.3: Kategori A -> squarederror; Kategori B / C -> tweedie."""
    return "reg:squarederror" if category == "A" else "reg:tweedie"


def make_model(objective: str, n_estimators=600, early_stopping_rounds=None) -> XGBRegressor:
    params = dict(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=4,
        min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
        objective=objective, random_state=42, n_jobs=-1,
    )
    if objective == "reg:tweedie":
        params["tweedie_variance_power"] = 1.5
    if early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    return XGBRegressor(**params)


def fit_with_early_stopping(train: pd.DataFrame, objective: str) -> int:
    """Cari jumlah pohon optimal memakai 14 hari terakhir dari TRAIN (test tidak disentuh)."""
    cut = train["tanggal"].max() - pd.Timedelta(days=TEST_DAYS)
    tr, va = train[train["tanggal"] <= cut], train[train["tanggal"] > cut]
    if len(tr) < 30 or len(va) == 0:
        return max(200, MIN_TREES)
    m = make_model(objective, early_stopping_rounds=30)
    m.fit(tr[FEATURES], tr["qty_keluar"], eval_set=[(va[FEATURES], va["qty_keluar"])], verbose=False)
    best = int(getattr(m, "best_iteration", m.n_estimators - 1)) + 1
    return max(best, MIN_TREES)


def run_pipeline(daily: pd.DataFrame, horizon: int = 14) -> dict:
    """Training model terpisah per kategori Pareto (Rule 3.3) + evaluasi + forecast."""
    horizon = int(np.clip(horizon, 1, 28))
    id_map = make_id_map(daily["id_obat"])

    max_date = daily["tanggal"].max()
    split = max_date - pd.Timedelta(days=TEST_DAYS)

    # --- Rule 5.1: time-based split
    feat = build_features(daily, id_map)
    train = feat[(feat["tanggal"] <= split) & feat["lag_1"].notna()]
    test = feat[feat["tanggal"] > split].copy()
    train_daily = daily[daily["tanggal"] <= split]
    full_train = feat[feat["lag_1"].notna()]

    categories = sorted(daily["kategori_pareto"].unique())
    eval_models, final_models = {}, {}
    objectives, n_trees_dict = {}, {}
    test_cat_list, rec_cat_list, fc_cat_list = [], [], []

    for cat in categories:
        tr_cat = train[train["kategori_pareto"] == cat]
        te_cat = test[test["kategori_pareto"] == cat].copy()
        tr_daily_cat = train_daily[train_daily["kategori_pareto"] == cat]
        daily_cat = daily[daily["kategori_pareto"] == cat]
        full_tr_cat = full_train[full_train["kategori_pareto"] == cat]

        obj = pick_objective_for_category(cat)
        n_trees = fit_with_early_stopping(tr_cat, obj)
        objectives[cat] = obj
        n_trees_dict[cat] = n_trees

        # Evaluasi model (train & evaluasi di holdout)
        eval_m = make_model(obj, n_estimators=n_trees)
        eval_m.fit(tr_cat[FEATURES], tr_cat["qty_keluar"], verbose=False)
        eval_models[cat] = eval_m

        pred = np.clip(eval_m.predict(te_cat[FEATURES]), 0, None)
        te_cat["pred_xgb"] = np.where(pred < 0.01, 0.0, pred)
        te_cat["pred_sma7"] = te_cat["rolling_mean_7"].fillna(0)
        test_cat_list.append(te_cat)

        rec_c = recursive_forecast(eval_m, tr_daily_cat, TEST_DAYS, id_map)
        rec_cat_list.append(rec_c)

        # Model final: latih ulang di SELURUH data kategori tersebut
        final_m = make_model(obj, n_estimators=n_trees)
        final_m.fit(full_tr_cat[FEATURES], full_tr_cat["qty_keluar"], verbose=False)
        final_models[cat] = final_m

        fc_c = recursive_forecast(final_m, daily_cat, horizon, id_map)
        fc_cat_list.append(fc_c)

    test_all = pd.concat(test_cat_list, ignore_index=True)
    rec_all = pd.concat(rec_cat_list, ignore_index=True)
    forecast_all = pd.concat(fc_cat_list, ignore_index=True)

    # --- Evaluasi 1: one-step-ahead (fitur dari data aktual)
    onestep = pd.concat(
        [eval_by_category(test_all, "pred_xgb", "XGBoost"),
         eval_by_category(test_all, "pred_sma7", "Baseline SMA-7")], ignore_index=True)

    # --- Evaluasi 2: recursive 14 hari (kondisi nyata saat forecasting)
    keys = ["tanggal", "id_obat"]
    actual = daily[daily["tanggal"] > split][keys + ["kategori_pareto", "qty_keluar"]]
    base = baseline_forecast(train_daily, TEST_DAYS)
    cmp_ = (actual.merge(rec_all[keys + ["prediksi"]].rename(columns={"prediksi": "pred_xgb"}), on=keys)
                  .merge(base[keys + ["prediksi"]].rename(columns={"prediksi": "pred_sma7"}), on=keys))
    recursive = pd.concat(
        [eval_by_category(cmp_, "pred_xgb", "XGBoost"),
         eval_by_category(cmp_, "pred_sma7", "Baseline SMA-7")], ignore_index=True)

    obj_summary = ", ".join(f"{k}: {v.replace('reg:', '')}" for k, v in objectives.items())
    ntree_summary = ", ".join(f"{k}: {v}" for k, v in n_trees_dict.items())

    return dict(
        models=final_models, eval_models=eval_models,
        model=final_models, eval_model=eval_models,
        id_map=id_map, objectives=objectives, n_trees_dict=n_trees_dict,
        objective=obj_summary, n_trees=ntree_summary,
        metrics_recursive=recursive, metrics_onestep=onestep,
        holdout=cmp_, test_features=test_all, train_features=train,
        forecast=forecast_all, split_date=split, max_date=max_date,
    )


def save_model(res: dict, path="models/xgb_forecaster.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        "models": res["models"],
        "id_map": res["id_map"],
        "objectives": res["objectives"],
        "n_trees": res["n_trees_dict"],
        "features": FEATURES,
    }, path)


def _print(res, daily=None):
    pd.set_option("display.float_format", "{:.2f}".format)
    print(f"Objective: {res['objective']} | pohon: {res['n_trees']} | split: {res['split_date'].date()}")
    if daily is not None:
        empty_c = daily.attrs.get("empty_pareto_drugs", 0)
        if empty_c > 0:
            print(f"[PERINGATAN] {empty_c} obat memiliki kategori Pareto kosong/NULL dan otomatis diisi 'C'.")
        multi_c = daily.attrs.get("multi_pareto_drugs", 0)
        if multi_c > 0:
            print(f"[INFO] {multi_c} obat memiliki >1 label kategori Pareto (dipilih label dengan total volume terbesar).")
    print("\n== Evaluasi recursive 14 hari (realistis) ==\n", res["metrics_recursive"].to_string(index=False))
    print("\n== Evaluasi one-step-ahead ==\n", res["metrics_onestep"].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/groubynama.csv")
    ap.add_argument("--horizon", type=int, default=14)
    a = ap.parse_args()
    daily = load_daily(a.csv)
    os.makedirs("data/processed", exist_ok=True)
    daily.to_csv("data/processed/daily_zero_filled.csv", index=False)
    res = run_pipeline(daily, a.horizon)
    _print(res, daily)
    save_model(res)
    res["forecast"].to_csv("data/processed/forecast.csv", index=False)
    print("\nModel -> models/xgb_forecaster.joblib | Forecast -> data/processed/forecast.csv")
