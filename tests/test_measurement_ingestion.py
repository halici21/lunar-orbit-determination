import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od import read_measurement_csv


class MeasurementIngestionTests(unittest.TestCase):
    def test_position_csv_with_station_names_maps_to_obs_data(self):
        csv_text = (
            "t_s,range_m,az_rad,el_rad,station_name,time_index\n"
            "0,1000,0.1,0.2,Canberra DSN,1\n"
            "60,1100,0.3,0.4,Madrid DSN,2\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "position.csv"
            path.write_text(csv_text, encoding="utf-8")
            result = read_measurement_csv(path, "position", station_names=("Canberra DSN", "Madrid DSN"))

        self.assertEqual(result.measurement_type, "position")
        np.testing.assert_allclose(
            result.obs_data,
            np.array(
                [
                    [0.0, 1000.0, 0.1, 0.2, 1.0, 1.0],
                    [60.0, 1100.0, 0.3, 0.4, 2.0, 2.0],
                ]
            ),
        )

    def test_range_rate_csv_with_arc_id_maps_to_obs_data(self):
        csv_text = (
            "t_s,range_m,range_rate_mps,az_rad,el_rad,station_id,time_index,arc_id\n"
            "0,1000,0.01,0.1,0.2,1,1,7\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rr.csv"
            path.write_text(csv_text, encoding="utf-8")
            result = read_measurement_csv(path, "range_rate", station_names=("Canberra DSN",))

        np.testing.assert_allclose(result.obs_data, [[0.0, 1000.0, 0.01, 0.1, 0.2, 1.0, 1.0, 7.0]])

    def test_ingestion_rejects_missing_columns_and_unknown_station(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_path = tmp_path / "missing.csv"
            missing_path.write_text("t_s,range_m,az_rad,station_id,time_index\n0,1,0,1,1\n", encoding="utf-8")
            unknown_path = tmp_path / "unknown.csv"
            unknown_path.write_text(
                "t_s,range_m,az_rad,el_rad,station_name,time_index\n0,1,0,0,Unknown,1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_measurement_csv(missing_path, "position")
            with self.assertRaises(ValueError):
                read_measurement_csv(unknown_path, "position", station_names=("Canberra DSN",))


if __name__ == "__main__":
    unittest.main()
