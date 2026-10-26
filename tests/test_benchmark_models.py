import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.benchmark_models import prepare_fold, summarize_predictions
from src.data_loader import clean_raw


def sample_clean(days=90):
    dates = pd.date_range("2025-10-01", periods=days)
    return clean_raw(pd.DataFrame({
        "tanggal": dates.strftime("%Y-%m-%d"), "id_obat": "1",
        "nama_obat": "OBAT", "qty_keluar": np.arange(days) + 1,
        "kategori_pareto": "A",
    }))


class BenchmarkTests(unittest.TestCase):
    def test_fold_freezes_pareto_before_holdout_and_zero_fills(self):
        clean = sample_clean()
        end = clean.tanggal.max()
        split = end - pd.Timedelta(days=14)
        clean.loc[clean.tanggal > split, "kategori_pareto"] = "C"
        clean.loc[clean.tanggal > split, "qty_keluar"] = 100000
        clean = clean[clean.tanggal != end]
        daily, actual, unknown = prepare_fold(clean, end)
        self.assertEqual(daily.tanggal.max(), split)
        self.assertEqual(set(daily.kategori_pareto), {"A"})
        self.assertEqual(set(actual.kategori_pareto), {"A"})
        self.assertEqual(actual.loc[actual.tanggal == end, "qty_keluar"].item(), 0)
        self.assertEqual(len(actual), 14)
        self.assertEqual(unknown, [])

    def test_future_only_drug_is_excluded_and_reported(self):
        clean = sample_clean()
        extra = clean.tail(1).assign(id_obat="NEW", nama_obat="NEW")
        daily, actual, unknown = prepare_fold(pd.concat([clean, extra]), clean.tanggal.max())
        self.assertNotIn("NEW", set(daily.id_obat))
        self.assertNotIn("NEW", set(actual.id_obat))
        self.assertEqual(unknown, ["NEW"])

    def test_short_history_rejected(self):
        clean = sample_clean(60)
        with self.assertRaisesRegex(ValueError, "58"):
            prepare_fold(clean, clean.tanggal.max())

    def test_pooled_wape_is_ratio_of_sums_not_fold_average(self):
        predictions = pd.DataFrame({
            "model": ["XGBoost", "XGBoost"], "kategori_pareto": ["A", "A"],
            "qty_keluar": [100., 1.], "prediksi": [90., 0.],
        })
        result = summarize_predictions(predictions).query("kategori == 'Overall'").iloc[0]
        self.assertAlmostEqual(result["WAPE (%)"], 11 / 101 * 100)

    def test_recursive_trimming_keeps_same_sales_features(self):
        from src.benchmark_models import SALES_FEATURES
        from src.data_loader import build_daily
        from src.features import make_id_map
        from src.forecast import recursive_forecast

        class RecordingModel:
            feature_names_in_ = SALES_FEATURES

            def __init__(self):
                self.inputs = []

            def predict(self, frame):
                self.inputs.append(frame.reset_index(drop=True))
                return frame["rolling_mean_7"].to_numpy()

        daily = build_daily(sample_clean())
        trimmed = daily[daily.tanggal > daily.tanggal.max() - pd.Timedelta(days=28)]
        full_model, short_model = RecordingModel(), RecordingModel()
        id_map = make_id_map(daily.id_obat)
        full = recursive_forecast(full_model, daily, 14, id_map)
        short = recursive_forecast(short_model, trimmed, 14, id_map)
        pd.testing.assert_frame_equal(full, short)
        for full_input, short_input in zip(full_model.inputs, short_model.inputs):
            pd.testing.assert_frame_equal(full_input, short_input)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional benchmark dependency torch")
class LSTMTests(unittest.TestCase):
    def test_sequences_do_not_include_target_or_future_quantities(self):
        from src.lstm_forecaster import sequence_arrays
        values = np.arange(65, dtype="float32").reshape(1, -1)
        dates = pd.date_range("2025-10-01", periods=65)
        x, calendar, ids, y = sequence_arrays(values, dates, np.ones(1, dtype="float32"))
        np.testing.assert_array_equal(x[0, :, 0], np.arange(28))
        self.assertEqual(y[0], 28)
        changed = values.copy()
        changed[:, 28:] += 1000
        changed_x, _, _, _ = sequence_arrays(changed, dates, np.ones(1, dtype="float32"))
        np.testing.assert_array_equal(x[0], changed_x[0])
        self.assertEqual(calendar.shape, (37, 3))
        self.assertEqual(ids.shape, (37,))

    def test_scale_fits_only_inner_training_and_forecast_is_nonnegative(self):
        import torch
        from src.lstm_forecaster import LSTMForecaster
        from src.data_loader import build_daily
        torch.set_num_threads(1)
        daily = build_daily(sample_clean(65))
        daily.loc[daily.tanggal > daily.tanggal.max() - pd.Timedelta(days=14), "qty_keluar"] = 99999
        model = LSTMForecaster("A").fit(daily, epochs=1, validation_days=14)
        self.assertAlmostEqual(float(model.scale[0]), np.arange(1, 52).mean())
        forecast = model.forecast(daily, 3)
        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast.tanggal.min(), daily.tanggal.max() + pd.Timedelta(days=1))
        self.assertTrue(np.isfinite(forecast.prediksi).all())
        self.assertTrue(forecast.prediksi.ge(0).all())

    def test_missing_grid_rejected(self):
        from src.lstm_forecaster import sales_matrix
        from src.data_loader import build_daily
        daily = build_daily(sample_clean(65)).drop(index=30)
        with self.assertRaisesRegex(ValueError, "grid"):
            sales_matrix(daily)

    def test_all_zero_tweedie_series_stays_finite(self):
        from src.lstm_forecaster import LSTMForecaster
        from src.data_loader import build_daily
        daily = build_daily(sample_clean(65)).assign(qty_keluar=0.0, kategori_pareto="C")
        model = LSTMForecaster("C").fit(daily, epochs=1)
        forecast = model.forecast(daily, 2)
        self.assertTrue(np.isfinite(forecast.prediksi).all())

    def test_end_to_end_uses_identical_holdout_keys_and_reports_all_models(self):
        from src.benchmark_models import benchmark
        clean = sample_clean(80)
        with patch("src.benchmark_models.fit_with_early_stopping", return_value=2):
            scores, pooled, predictions, training, metadata = benchmark(
                clean, folds=1, horizon=14, epochs=1, threads=1)
        self.assertEqual(set(pooled.model), {"XGBoost", "LSTM", "Baseline SMA-7"})
        self.assertEqual(len(scores), 6)  # Overall + A, three models
        self.assertEqual(len(training), 2)
        self.assertEqual(len(metadata["folds"]), 1)
        expected = None
        for _, group in predictions.groupby("model"):
            keys = set(zip(group.tanggal, group.id_obat))
            self.assertEqual(len(keys), 14)
            self.assertTrue((group.tanggal > group.split_date).all())
            self.assertTrue(np.isfinite(group.prediksi).all())
            if expected is None:
                expected = keys
            self.assertEqual(keys, expected)


if __name__ == "__main__":
    unittest.main()
