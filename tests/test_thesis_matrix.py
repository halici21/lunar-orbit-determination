import unittest

from lunar_od import (
    THESIS_ATOL,
    THESIS_COLD_START_SIGMA_POS_M,
    THESIS_COLD_START_SIGMA_VEL_MPS,
    THESIS_DURATION_H,
    THESIS_FACTORIAL_CASES,
    THESIS_MAX_ITER,
    THESIS_NETWORKS,
    THESIS_RTOL,
    THESIS_SAMPLE_STEP_S,
    thesis_case_rows,
    thesis_network_by_name,
    thesis_seed_for,
)


class ThesisMatrixTests(unittest.TestCase):
    def test_factorial_matrix_is_frozen_to_twenty_four_core_cases(self):
        keys = {
            (case.network, case.measurement_type, case.estimator_type, case.start_mode)
            for case in THESIS_FACTORIAL_CASES
        }

        self.assertEqual(len(THESIS_FACTORIAL_CASES), 24)
        self.assertEqual(len(keys), 24)
        self.assertEqual({case.network for case in THESIS_FACTORIAL_CASES}, {"single", "multi"})
        self.assertEqual({case.measurement_type for case in THESIS_FACTORIAL_CASES}, {"position", "range_rate"})
        self.assertEqual({case.estimator_type for case in THESIS_FACTORIAL_CASES}, {"bls_lm", "srif", "ukf"})
        self.assertEqual({case.start_mode for case in THESIS_FACTORIAL_CASES}, {"cold", "hot"})

    def test_station_networks_are_canonical(self):
        networks = {network.name: network.station_names for network in THESIS_NETWORKS}

        self.assertEqual(networks["single"], ("Canberra DSN",))
        self.assertEqual(
            networks["multi"],
            ("Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"),
        )
        self.assertEqual(thesis_network_by_name("single").station_names, ("Canberra DSN",))
        with self.assertRaises(KeyError):
            thesis_network_by_name("unknown")

    def test_case_labels_and_rows_are_stable(self):
        rows = thesis_case_rows()

        self.assertEqual(rows[0]["label"], "single position bls_lm cold")
        self.assertEqual(rows[-1]["label"], "multi range_rate ukf hot")
        self.assertEqual(
            [row["label"] for row in rows],
            [case.label for case in THESIS_FACTORIAL_CASES],
        )

    def test_numeric_knobs_are_frozen_for_thesis_report(self):
        self.assertEqual(THESIS_DURATION_H, 4.0)
        self.assertEqual(THESIS_SAMPLE_STEP_S, 240.0)
        self.assertEqual(THESIS_MAX_ITER, 10)
        self.assertEqual(THESIS_COLD_START_SIGMA_POS_M, 180.0)
        self.assertEqual(THESIS_COLD_START_SIGMA_VEL_MPS, 0.03)
        self.assertEqual(THESIS_RTOL, 1e-10)
        self.assertEqual(THESIS_ATOL, 1e-11)

    def test_cold_start_seeds_are_stable_per_network_and_measurement(self):
        self.assertEqual(thesis_seed_for("single", "position"), 242285)
        self.assertEqual(thesis_seed_for("single", "range_rate"), 242448)
        self.assertEqual(thesis_seed_for("multi", "position"), 242198)
        self.assertEqual(thesis_seed_for("multi", "range_rate"), 242361)


if __name__ == "__main__":
    unittest.main()
