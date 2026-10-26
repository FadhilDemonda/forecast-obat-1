import io
import unittest

import numpy as np
import pandas as pd

from src.features import FEATURES
from src.forecast import recursive_forecast
from src.schedule import attach_schedule, load_schedule


class ScheduleTests(unittest.TestCase):
    def schedule(self):
        return load_schedule(io.StringIO(
            "tanggal,PENYAKIT DALAM,JANTUNG,total_skor\n"
            "2026-01-01,3,2,5\n2026-01-02,0,6,6\n2026-01-03,2,3,5\n"
        ))

    def test_join_preserves_rows_qty_and_uses_same_day(self):
        daily = pd.DataFrame({"tanggal": pd.to_datetime(["2026-01-01"] * 2),
                              "id_obat": ["A", "B"], "qty_keluar": [10, 20]})
        result = attach_schedule(daily, self.schedule())
        self.assertEqual(len(result), 2)
        self.assertEqual(result.qty_keluar.sum(), 30)
        self.assertEqual(result.skor_jadwal_penyakit_dalam.tolist(), [3, 3])

    def test_invalid_schedule_rejected(self):
        for rows in (
            "2026-01-01,3,4\n",  # Total salah
            "2026-01-01,-1,-1\n",
            "2026-01-01,3,3\n2026-01-01,3,3\n",
            "not-a-date,3,3\n",
            "2026-01-01,,3\n",
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_schedule(io.StringIO("tanggal,JANTUNG,total_skor\n" + rows))

    def test_missing_dates_are_not_zero_filled(self):
        with self.assertRaisesRegex(ValueError, "2026-01-04"):
            attach_schedule(pd.DataFrame({"tanggal": pd.to_datetime(["2026-01-04"])}), self.schedule())

    def test_recursive_forecast_uses_future_schedule_not_historical_average(self):
        class Model:
            feature_names_in_ = np.array(FEATURES + ["skor_jadwal_jantung"])

            def predict(self, frame):
                return frame["skor_jadwal_jantung"].to_numpy()

        daily = pd.DataFrame({"tanggal": pd.to_datetime(["2026-01-01"]),
                              "id_obat": ["A"], "nama_obat": ["A"],
                              "kategori_pareto": ["A"], "qty_keluar": [10.]})
        result = recursive_forecast(Model(), daily, 2, {"A": 0}, schedule=self.schedule())
        self.assertEqual(result.prediksi.tolist(), [6, 3])
        with self.assertRaisesRegex(ValueError, "wajib"):
            recursive_forecast(Model(), daily, 2, {"A": 0})
        with self.assertRaisesRegex(ValueError, "2026-01-04"):
            recursive_forecast(Model(), daily, 3, {"A": 0}, schedule=self.schedule())


if __name__ == "__main__":
    unittest.main()
