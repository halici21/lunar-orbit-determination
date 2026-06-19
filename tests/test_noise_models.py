import unittest

import numpy as np

from lunar_od import MeasurementNoiseConfig, generate_measurement_noise


class NoiseModelTests(unittest.TestCase):
    def test_correlated_gaussian_matches_target_covariance(self):
        covariance = np.array([[4.0, 1.2], [1.2, 1.0]])
        samples = generate_measurement_noise(
            30_000,
            covariance,
            rng=np.random.default_rng(42),
            config=MeasurementNoiseConfig(model="correlated_gaussian"),
        )

        np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, rtol=0.04, atol=0.04)

    def test_ar1_noise_has_requested_temporal_correlation(self):
        samples = generate_measurement_noise(
            30_000,
            np.eye(1),
            rng=np.random.default_rng(43),
            config=MeasurementNoiseConfig(model="ar1", ar1_coefficient=0.8),
        )[:, 0]

        lag1 = np.corrcoef(samples[:-1], samples[1:])[0, 1]
        self.assertAlmostEqual(float(lag1), 0.8, delta=0.03)
        self.assertAlmostEqual(float(np.var(samples)), 1.0, delta=0.05)

    def test_student_t_noise_preserves_covariance_and_has_heavy_tails(self):
        samples = generate_measurement_noise(
            40_000,
            np.eye(1),
            rng=np.random.default_rng(44),
            config=MeasurementNoiseConfig(model="student_t", student_t_dof=4.0),
        )[:, 0]

        self.assertAlmostEqual(float(np.var(samples)), 1.0, delta=0.08)
        excess_kurtosis = np.mean((samples - np.mean(samples)) ** 4) / np.var(samples) ** 2 - 3.0
        self.assertGreater(float(excess_kurtosis), 2.0)

    def test_invalid_noise_config_is_rejected(self):
        with self.assertRaises(ValueError):
            MeasurementNoiseConfig(model="ar1", ar1_coefficient=1.0)
        with self.assertRaises(ValueError):
            MeasurementNoiseConfig(model="student_t", student_t_dof=2.0)


if __name__ == "__main__":
    unittest.main()
