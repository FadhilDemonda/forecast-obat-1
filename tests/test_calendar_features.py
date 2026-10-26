import io
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.calendar_features import CALENDAR_FEATURES, build_calendar, load_calendar_overrides
from src.features import BASE_FEATURES, FEATURES, build_features
from src.forecast import recursive_forecast
from src.benchmark_calendar import benchmark_calendar
from src.train_xgb import run_pipeline


class CalendarTests(unittest.TestCase):
    def test_pipeline_defaults_to_no_additional_calendar(self):
        self.assertIs(inspect.signature(run_pipeline).parameters["include_calendar"].default, False)

    def test_public_leave_and_distance_across_year(self):
        frame = build_calendar(pd.to_datetime(["2026-01-01", "2026-03-20", "2026-03-21", "2025-12-31"]))
        frame = frame.set_index("tanggal")
        self.assertEqual(frame.loc["2026-01-01", "is_public_holiday"], 1)
        self.assertEqual(frame.loc["2026-03-20", "is_collective_leave"], 1)
        self.assertEqual(frame.loc["2026-03-21", "days_since_holiday"], 0)
        self.assertEqual(frame.loc["2025-12-31", "days_until_holiday"], 1)
        self.assertTrue(frame.days_since_holiday.between(0, 28).all())

    def test_ramadan_estimate_can_be_overridden(self):
        date = pd.to_datetime(["2026-03-01"])
        self.assertEqual(build_calendar(date).is_ramadan.item(), 1)
        overrides = load_calendar_overrides(io.StringIO("tanggal,is_ramadan\n2026-03-01,0\n"))
        self.assertEqual(build_calendar(date, overrides).is_ramadan.item(), 0)

    def test_override_updates_holiday_distance(self):
        overrides = load_calendar_overrides(io.StringIO("tanggal,is_public_holiday\n2026-09-30,1\n"))
        frame = build_calendar(pd.to_datetime(["2026-09-29", "2026-09-30"]), overrides)
        self.assertEqual(frame.days_until_holiday.tolist(), [1, 0])

    def test_closure_not_inferred_and_requires_full_coverage(self):
        dates = pd.date_range("2026-01-01", periods=2)
        self.assertNotIn("is_clinic_closed", build_calendar(dates))
        overrides = load_calendar_overrides(io.StringIO("tanggal,is_clinic_closed\n2026-01-01,1\n"))
        with self.assertRaisesRegex(ValueError, "seluruh tanggal"):
            build_calendar(dates, overrides)
        self.assertEqual(build_calendar(dates[:1], overrides).is_clinic_closed.item(), 1)

    def test_invalid_overrides_rejected(self):
        for content in ["tanggal,is_ramadan\n2026-01-01,2", "tanggal,is_ramadan\n2026-01-01,yes",
                        "tanggal,is_ramadan\n2026-01-01,1\n2026-01-01,0"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_calendar_overrides(io.StringIO(content))

    def test_training_and_recursive_calendar_match(self):
        daily = pd.DataFrame({"tanggal": pd.to_datetime(["2025-12-31"]), "id_obat": ["A"],
                              "nama_obat": ["A"], "kategori_pareto": ["A"], "qty_keluar": [2.]})

        class Model:
            feature_names_in_ = np.array(FEATURES)

            def predict(self, frame):
                return frame.is_public_holiday.to_numpy()

        forecast = recursive_forecast(Model(), daily, 2, {"A": 0})
        self.assertEqual(forecast.prediksi.tolist(), [1, 0])
        combined = pd.concat([daily, daily.assign(tanggal=pd.Timestamp("2026-01-01"))], ignore_index=True)
        features = build_features(combined, {"A": 0})
        self.assertEqual(features.is_public_holiday.tolist(), [0, 1])
        legacy = build_features(combined, {"A": 0}, include_calendar=False)
        self.assertTrue(set(BASE_FEATURES).issubset(legacy))
        self.assertFalse(set(CALENDAR_FEATURES) & set(legacy))

    def test_recursive_uses_closure_override(self):
        dates = pd.date_range("2026-01-01", periods=3)
        overrides = load_calendar_overrides(pd.DataFrame({"tanggal": dates, "is_clinic_closed": [0, 1, 0]}))
        daily = pd.DataFrame({"tanggal": dates[:1], "id_obat": ["A"], "nama_obat": ["A"],
                              "kategori_pareto": ["A"], "qty_keluar": [2.]})

        class Model:
            feature_names_in_ = np.array(FEATURES + ["is_clinic_closed"])

            def predict(self, frame):
                return frame.is_clinic_closed.to_numpy()

        result = recursive_forecast(Model(), daily, 2, {"A": 0}, calendar_overrides=overrides)
        self.assertEqual(result.prediksi.tolist(), [1, 0])

    def test_benchmark_pairs_identical_dates_and_changes_only_calendar_option(self):
        daily = pd.DataFrame({"tanggal": pd.date_range("2026-01-01", periods=100)})

        def pipeline(subset, horizon, **kwargs):
            return {"split_date": subset.tanggal.max() - pd.Timedelta(days=14),
                    "metrics_recursive": pd.DataFrame([{"model": "XGBoost", "kategori": "Overall", "WAPE (%)": 30.}])}

        with patch("src.benchmark_calendar.run_pipeline", side_effect=pipeline) as mocked:
            result = benchmark_calendar(daily, folds=2)
        self.assertEqual(len(result), 4)
        for i in (0, 2):
            first, second = mocked.call_args_list[i:i + 2]
            pd.testing.assert_frame_equal(first.args[0], second.args[0])
            self.assertEqual(first.args[1], 14)
            self.assertFalse(first.kwargs["include_calendar"])
            self.assertTrue(second.kwargs["include_calendar"])


if __name__ == "__main__":
    unittest.main()
