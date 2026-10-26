import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.benchmark_windows import benchmark_windows, window_data
from src.features import BASE_FEATURES


class WindowBenchmarkTests(unittest.TestCase):
    def daily(self, days=214):
        return pd.DataFrame({"tanggal": pd.date_range("2026-03-01", periods=days),
                             "id_obat": "A", "nama_obat": "A", "kategori_pareto": "A", "qty_keluar": 2.})

    def test_exact_training_bounds_and_partial_warmup(self):
        daily = self.daily()
        split = daily.tanggal.max() - pd.Timedelta(days=14)
        for days in (90, 120, 180):
            history, start, warmup = window_data(daily, split, days)
            self.assertEqual(history[history.tanggal >= start].tanggal.nunique(), days)
            self.assertEqual(warmup, min(28, 200 - days))
            self.assertEqual(history.tanggal.max(), split)

    def test_insufficient_folds_fail_before_training(self):
        with patch("src.benchmark_windows.make_model") as model:
            with self.assertRaises(ValueError):
                benchmark_windows(self.daily(), schedule=None, folds=3)
            model.assert_not_called()

    def test_same_holdouts_no_warmup_or_holdout_in_training(self):
        fits = []

        class Model:
            feature_names_in_ = np.array(BASE_FEATURES)

            def fit(self, frame, target, **kwargs):
                fits.append((len(frame), target.copy()))

            def predict(self, frame):
                return np.full(len(frame), 2.)

        with patch("src.benchmark_windows.make_model", side_effect=lambda *a, **k: Model()), \
                patch("src.benchmark_windows.fit_with_early_stopping", return_value=1):
            detail, summary = benchmark_windows(self.daily(), schedule=None)
        self.assertEqual([n for n, _ in fits], [90, 120, 180] * 2)
        self.assertEqual(detail.groupby("train_days").holdout_end.nunique().tolist(), [2, 2, 2])
        self.assertTrue(summary["WAPE (%)"].eq(0).all())
        self.assertEqual(detail.query("train_days == 180").warmup_days.unique().tolist(), [20, 6])


if __name__ == "__main__":
    unittest.main()
