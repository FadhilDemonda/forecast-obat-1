import io
import unittest

import pandas as pd

from src.schedule import build_doctor_schedule, combine_forecast_schedule, load_schedule, require_schedule_dates


class DoctorScheduleTests(unittest.TestCase):
    def build(self, master=None, roster=None):
        return build_doctor_schedule(
            io.StringIO(master or "SPESIALIS,dokter_sensor,kategori_pareto\nANAK,A**,A\nANAK,B**,B\n"),
            io.StringIO(roster or "tanggal,SPESIALIS,dokter_sensor\n2026-10-01,ANAK,A**\n2026-10-01,ANAK,B**\n"))

    def test_scores_and_stable_ids(self):
        master, roster, scores = self.build()
        other, _, _ = self.build(master="SPESIALIS,dokter_sensor,kategori_pareto\n anak ,b**,B\nANAK,A**,A\n")
        self.assertEqual(master.set_index("dokter_sensor").id_dokter.to_dict(),
                         other.set_index("dokter_sensor").id_dokter.to_dict())
        self.assertEqual(scores.skor_jadwal_anak.tolist(), [5])
        self.assertEqual(roster.id_dokter.nunique(), 2)

    def test_reject_ambiguous_unknown_and_duplicate_doctors(self):
        with self.assertRaises(ValueError):
            self.build(master="SPESIALIS,dokter_sensor,kategori_pareto\nANAK,A**,A\nANAK,A**,B\n")
        with self.assertRaises(ValueError):
            self.build(roster="tanggal,SPESIALIS,dokter_sensor\n2026-10-01,ANAK,C**\n")
        with self.assertRaises(ValueError):
            self.build(roster="tanggal,SPESIALIS,dokter_sensor\n2026-10-01,ANAK,A**\n2026-10-01,ANAK,A**\n")

    def test_history_preserved_future_replaced_missing_date_not_filled(self):
        historical = load_schedule(io.StringIO(
            "tanggal,ANAK,JANTUNG,total_skor\n2026-09-30,2,3,5\n2026-10-01,9,9,18\n2026-10-02,9,9,18\n"))
        _, _, scores = self.build()
        result = combine_forecast_schedule(historical, scores, "2026-09-30")
        pd.testing.assert_frame_equal(result.iloc[:1], historical.iloc[:1])
        self.assertEqual(result.iloc[1].skor_jadwal_anak, 5)
        self.assertEqual(result.iloc[1].skor_jadwal_jantung, 0)
        with self.assertRaisesRegex(ValueError, "2026-10-02"):
            require_schedule_dates(result, pd.date_range("2026-09-30", "2026-10-02"))


if __name__ == "__main__":
    unittest.main()
