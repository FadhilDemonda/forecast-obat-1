import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import context_correlations, wape_matrix


class EvaluationTests(unittest.TestCase):
    def test_evaluation_import_with_cached_legacy_features_module(self):
        # Streamlit dapat mempertahankan modul features versi lama di memori.
        # Visualisasi harus dapat diimpor tanpa bergantung pada konstanta model.
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys, types; "
                "sys.modules['src.features'] = types.ModuleType('src.features'); "
                "from src.evaluation import context_correlations, wape_matrix; "
                "import pandas as pd; "
                "corr, _ = context_correlations(pd.DataFrame({"
                "'qty_keluar': [1, 2, 3], 'ctx_n_dokter_lag_7': [1, 2, 3]})); "
                "assert corr.loc['Qty obat', 'Dokter per obat'] == 1.0"
            )],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wape_uses_reported_metrics_and_category_order(self):
        metrics = pd.DataFrame([
            {"model": "XGBoost", "kategori": "C", "WAPE (%)": 47.39},
            {"model": "XGBoost", "kategori": "Overall", "WAPE (%)": 33.88},
            {"model": "Baseline SMA-7", "kategori": "Overall", "WAPE (%)": 39.85},
        ])
        matrix = wape_matrix(metrics)
        self.assertEqual(matrix.columns.tolist(), ["Overall", "C"])
        self.assertEqual(matrix.loc["XGBoost", "C"], 47.39)
        self.assertTrue(pd.isna(matrix.loc["Baseline SMA-7", "C"]))

    def test_default_correlates_lag_not_same_day_context(self):
        frame = pd.DataFrame({
            "qty_keluar": [1, 2, 3, 4],
            "ctx_n_dokter": [4, 3, 2, 1],
            "ctx_n_dokter_lag_7": [1, 2, 3, 4],
        })
        corr, _ = context_correlations(frame)
        self.assertAlmostEqual(corr.loc["Qty obat", "Dokter per obat"], 1)
        same_day, _ = context_correlations(frame, suffix="")
        self.assertAlmostEqual(same_day.loc["Qty obat", "Dokter per obat"], -1)

    def test_constants_and_missing_values_are_not_reported_as_zero_correlation(self):
        frame = pd.DataFrame({
            "qty_keluar": [1, 2, 3, 4],
            "ctx_n_dokter_lag_7": [0, 0, 0, 0],
            "ctx_n_spesialis_lag_7": [None, None, None, None],
        })
        corr, constants = context_correlations(frame)
        self.assertTrue(corr.empty)
        self.assertIn("Dokter per obat", constants)
        self.assertIn("Spesialis per obat", constants)

    def test_constant_target_or_empty_frame_does_not_crash(self):
        for frame in (pd.DataFrame(), pd.DataFrame({
            "qty_keluar": [0, 0, 0], "ctx_n_dokter_lag_7": [1, 2, 3],
        })):
            corr, _ = context_correlations(frame)
            self.assertTrue(corr.empty)

    def test_pairwise_missing_values_stay_undefined_when_insufficient(self):
        frame = pd.DataFrame({
            "qty_keluar": [1, 2, 3, 4],
            "ctx_n_dokter_lag_7": [None, None, 3, 4],
        })
        corr, _ = context_correlations(frame)
        self.assertTrue(pd.isna(corr.loc["Qty obat", "Dokter per obat"]))

    def test_spearman_can_measure_monotonic_nonlinear_relation(self):
        frame = pd.DataFrame({
            "qty_keluar": [1, 4, 9, 16],
            "ctx_n_dokter_rolling_mean_7": [1, 2, 3, 4],
        })
        corr, _ = context_correlations(frame, "_rolling_mean_7", "spearman")
        self.assertAlmostEqual(corr.loc["Qty obat", "Dokter per obat"], 1)


if __name__ == "__main__":
    unittest.main()
