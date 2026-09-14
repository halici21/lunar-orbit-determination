"""Tests for the opt-in RF link-budget and Doppler-noise model.

Three kinds of check, in increasing strength:

* **unit tests** - pure algebra with tight tolerances, because there is nothing
  to approximate in a logarithm
* **property tests** - the scaling laws that a link budget must obey, expressed
  as invariants rather than as memorised numbers
* **causal-chain tests** - the sequence that matters physically for lunar
  tracking: warm body in the beam raises Tsys, which lowers G/T, which lowers
  C/N0, which raises the Doppler noise

Nothing here requires SPICE or a propagated trajectory; the RF model is
deliberately independent of the astrodynamics.
"""

from __future__ import annotations

import math
import unittest

from lunar_od.rf import (
    HYDROGEN_MASER_SIGMA_60S_HIGH_MPS,
    HYDROGEN_MASER_SIGMA_60S_LOW_MPS,
    AntennaConfig,
    CarrierLoopConfig,
    LinkEnvironmentConfig,
    SpacecraftRadioConfig,
    antenna_gain_dbi,
    atmospheric_loss_db,
    build_two_way_link_budget,
    carrier_loop_snr,
    cn0_dbhz,
    cn0_dbhz_via_g_over_t,
    cn0_for_thermal_sigma_dbhz,
    compute_counted_doppler_thermal_noise,
    dbw_to_watts,
    eirp_dbw,
    free_space_loss_db,
    frequency_standard_sigma_mps,
    ground_terminal_gain_dbi,
    lock_status,
    reference_two_way_link,
    required_eirp_dbw,
    system_temperature_k,
    thermal_sigma_one_leg_mps,
    watts_to_dbw,
    wavelength_m,
    zenith_attenuation_db,
)
from lunar_od.rf.link_budget import BOLTZMANN_DBW_PER_K_PER_HZ

_X_DOWNLINK_HZ = (880.0 / 749.0) * 7.2e9
_LUNAR_RANGE_M = 3.844e8
_DOUBLING_DB = 20.0 * math.log10(2.0)  # 6.0206


class UnitConversionTests(unittest.TestCase):
    def test_watt_dbw_roundtrip(self):
        for w in (1.0, 10.0, 2.5e3, 1e-16):
            self.assertAlmostEqual(dbw_to_watts(watts_to_dbw(w)) / w, 1.0, places=12)

    def test_known_conversions(self):
        self.assertAlmostEqual(watts_to_dbw(1.0), 0.0, places=12)
        self.assertAlmostEqual(watts_to_dbw(10.0), 10.0, places=12)
        self.assertAlmostEqual(watts_to_dbw(20e3), 43.0103, places=4)

    def test_wavelength(self):
        self.assertAlmostEqual(wavelength_m(_X_DOWNLINK_HZ), 0.0354394, places=6)

    def test_non_positive_inputs_are_rejected(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                watts_to_dbw(bad)
            with self.assertRaises(ValueError):
                wavelength_m(bad)
            with self.assertRaises(ValueError):
                free_space_loss_db(bad, _X_DOWNLINK_HZ)

    def test_boltzmann_constant_in_db(self):
        """The -228.6 dBW/K/Hz that appears in every link budget."""
        self.assertAlmostEqual(BOLTZMANN_DBW_PER_K_PER_HZ, -228.5991, places=3)


class AntennaGainTests(unittest.TestCase):
    def test_thirty_four_metre_x_band_gain_is_physical(self):
        gain = antenna_gain_dbi(AntennaConfig(34.0, 0.65), _X_DOWNLINK_HZ)
        # A 34-m dish at X-band sits in the high sixties; the DSN publishes
        # 68.35 dBi for DSS-35 referenced to the feedhorn aperture.
        self.assertTrue(66.0 < gain < 70.0, gain)

    def test_doubling_diameter_adds_six_db(self):
        small = antenna_gain_dbi(AntennaConfig(17.0, 0.65), _X_DOWNLINK_HZ)
        large = antenna_gain_dbi(AntennaConfig(34.0, 0.65), _X_DOWNLINK_HZ)
        self.assertAlmostEqual(large - small, _DOUBLING_DB, places=9)

    def test_doubling_frequency_adds_six_db(self):
        low = antenna_gain_dbi(AntennaConfig(34.0, 0.65), 4.0e9)
        high = antenna_gain_dbi(AntennaConfig(34.0, 0.65), 8.0e9)
        self.assertAlmostEqual(high - low, _DOUBLING_DB, places=9)

    def test_halving_efficiency_costs_three_db(self):
        full = antenna_gain_dbi(AntennaConfig(34.0, 0.70), _X_DOWNLINK_HZ)
        half = antenna_gain_dbi(AntennaConfig(34.0, 0.35), _X_DOWNLINK_HZ)
        self.assertAlmostEqual(full - half, 10.0 * math.log10(2.0), places=9)

    def test_efficiency_must_be_physical(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                AntennaConfig(34.0, bad)


class FreeSpaceLossTests(unittest.TestCase):
    def test_lunar_x_band_value(self):
        self.assertAlmostEqual(
            free_space_loss_db(_LUNAR_RANGE_M, _X_DOWNLINK_HZ), 222.69, places=1
        )

    def test_doubling_range_adds_six_db(self):
        near = free_space_loss_db(_LUNAR_RANGE_M, _X_DOWNLINK_HZ)
        far = free_space_loss_db(2.0 * _LUNAR_RANGE_M, _X_DOWNLINK_HZ)
        self.assertAlmostEqual(far - near, _DOUBLING_DB, places=9)

    def test_doubling_frequency_adds_six_db(self):
        low = free_space_loss_db(_LUNAR_RANGE_M, 4.0e9)
        high = free_space_loss_db(_LUNAR_RANGE_M, 8.0e9)
        self.assertAlmostEqual(high - low, _DOUBLING_DB, places=9)

    def test_lunar_link_is_much_stronger_than_one_au(self):
        """Pure geometry, and the reason lunar Doppler can be quiet."""
        au = free_space_loss_db(1.495978707e11, _X_DOWNLINK_HZ)
        moon = free_space_loss_db(_LUNAR_RANGE_M, _X_DOWNLINK_HZ)
        self.assertAlmostEqual(au - moon, 51.8, places=1)


class Cn0Tests(unittest.TestCase):
    def test_two_formulations_agree_exactly(self):
        """Received-power route and G/T route must be the same number."""
        for tsys in (20.0, 45.0, 220.0):
            for eirp in (10.0, 40.0):
                direct = cn0_dbhz(eirp, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, tsys, 0.3)
                viagt = cn0_dbhz_via_g_over_t(
                    eirp, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, tsys, 0.3
                )
                self.assertAlmostEqual(direct, viagt, places=10)

    def test_eirp_increase_passes_straight_through(self):
        base = cn0_dbhz(30.0, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 25.0)
        more = cn0_dbhz(33.0, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 25.0)
        self.assertAlmostEqual(more - base, 3.0, places=9)

    def test_doubling_system_temperature_costs_three_db(self):
        cold = cn0_dbhz(30.0, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 25.0)
        warm = cn0_dbhz(30.0, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 50.0)
        self.assertAlmostEqual(cold - warm, 10.0 * math.log10(2.0), places=9)

    def test_required_eirp_inverts_cn0(self):
        target = 55.0
        eirp = required_eirp_dbw(target, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 25.0, 0.2)
        back = cn0_dbhz(eirp, _LUNAR_RANGE_M, _X_DOWNLINK_HZ, 68.0, 25.0, 0.2)
        self.assertAlmostEqual(back, target, places=9)

    def test_eirp_composition(self):
        self.assertAlmostEqual(eirp_dbw(20e3, 66.99, 0.0), 43.0103 + 66.99, places=3)


class CarrierLoopTests(unittest.TestCase):
    def test_loop_snr_is_dimensionless_not_db_hz(self):
        """The Phase-9 trap: rho_L and C/N0 are different quantities."""
        loop = CarrierLoopConfig(loop_bandwidth_hz=1.0)
        wide = CarrierLoopConfig(loop_bandwidth_hz=100.0)
        cn0 = 40.0
        self.assertAlmostEqual(carrier_loop_snr(cn0, loop), 10.0**4.0, places=3)
        # same C/N0, hundred-fold bandwidth: two decades of loop SNR difference
        ratio = carrier_loop_snr(cn0, loop) / carrier_loop_snr(cn0, wide)
        self.assertAlmostEqual(ratio, 100.0, places=6)

    def test_lock_classification(self):
        loop = CarrierLoopConfig(loop_bandwidth_hz=1.0)
        self.assertEqual(lock_status(10.0**2.0, loop), "LOCK_ROBUST")
        self.assertEqual(lock_status(10.0**0.8, loop), "LOCK_MARGINAL")
        self.assertEqual(lock_status(0.44, loop), "LOCK_NOT_SUPPORTED")
        self.assertEqual(lock_status(None, loop), "UNKNOWN_DUE_TO_MISSING_INPUT")

    def test_suppressed_carrier_is_refused_rather_than_guessed(self):
        loop = CarrierLoopConfig(loop_bandwidth_hz=1.0, carrier_mode="suppressed")
        with self.assertRaises(NotImplementedError):
            carrier_loop_snr(40.0, loop)

    def test_bandwidth_must_be_positive(self):
        with self.assertRaises(ValueError):
            CarrierLoopConfig(loop_bandwidth_hz=0.0)


class EnvironmentTests(unittest.TestCase):
    def test_zenith_attenuation_matches_published_table(self):
        self.assertAlmostEqual(zenith_attenuation_db("Canberra", 0.00), 0.039, places=6)
        self.assertAlmostEqual(zenith_attenuation_db("Canberra", 0.90), 0.058, places=6)
        self.assertAlmostEqual(zenith_attenuation_db("Goldstone", 0.50), 0.040, places=6)

    def test_weather_interpolation_is_monotone(self):
        values = [zenith_attenuation_db("Canberra", cd) for cd in (0.0, 0.25, 0.5, 0.9)]
        self.assertEqual(values, sorted(values))

    def test_unknown_site_is_rejected(self):
        with self.assertRaises(ValueError):
            zenith_attenuation_db("Ayazaga", 0.5)

    def test_atmospheric_loss_grows_towards_the_horizon(self):
        high = atmospheric_loss_db(0.046, 90.0)
        low = atmospheric_loss_db(0.046, 10.0)
        self.assertGreater(low, high)
        self.assertAlmostEqual(high, 0.046, places=9)

    def test_model_refuses_elevations_outside_its_stated_domain(self):
        for bad in (0.0, 5.9, 90.1):
            with self.assertRaises(ValueError):
                atmospheric_loss_db(0.046, bad)

    def test_reference_terminal_gain_is_near_published_value(self):
        cfg = reference_two_way_link("Canberra", weather_cd=0.50)
        gain = ground_terminal_gain_dbi(
            cfg.ground_terminal, 45.0, cfg.environment.zenith_attenuation_db
        )
        # G0 receive 68.35 dBi minus a small atmospheric term at 45 deg
        self.assertTrue(68.2 < gain < 68.4, gain)

    def test_system_temperature_is_cryogenic_and_elevation_dependent(self):
        cfg = reference_two_way_link("Canberra", weather_cd=0.50)
        zenith = system_temperature_k(cfg.ground_terminal, cfg.environment, 90.0)
        horizon = system_temperature_k(cfg.ground_terminal, cfg.environment, 10.0)
        self.assertTrue(10.0 < zenith < 40.0, zenith)
        self.assertGreater(horizon, zenith)


class LunarBackgroundChainTests(unittest.TestCase):
    """The causal chain that makes lunar tracking different from deep space."""

    def _budget(self, background_k):
        cfg = reference_two_way_link(
            "Canberra",
            weather_cd=0.50,
            spacecraft_eirp_dbw=20.0,
            spacecraft_g_over_t_db_per_k=0.0,
            transponder_loop_bandwidth_hz=10.0,
            lunar_background_k=background_k,
        )
        return compute_counted_doppler_thermal_noise(
            cfg, _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )

    def test_warm_beam_propagates_all_the_way_to_doppler_noise(self):
        cold = self._budget(0.0)
        warm = self._budget(200.0)
        self.assertGreater(warm.downlink.system_temperature_k, cold.downlink.system_temperature_k)
        self.assertLess(warm.downlink.g_over_t_db_per_k, cold.downlink.g_over_t_db_per_k)
        self.assertLess(warm.downlink.cn0_dbhz, cold.downlink.cn0_dbhz)
        self.assertGreater(
            warm.sigma_thermal_downlink_mps, cold.sigma_thermal_downlink_mps
        )

    def test_no_background_leaves_system_temperature_untouched(self):
        cfg = reference_two_way_link("Canberra", weather_cd=0.50)
        bare = system_temperature_k(cfg.ground_terminal, cfg.environment, 45.0)
        env = LinkEnvironmentConfig(
            weather_cumulative_distribution=0.50,
            zenith_attenuation_db=cfg.environment.zenith_attenuation_db,
            background_temperature_k=0.0,
        )
        self.assertAlmostEqual(
            system_temperature_k(cfg.ground_terminal, env, 45.0), bare, places=12
        )


class ThermalDopplerTests(unittest.TestCase):
    def test_longer_count_averages_noise_down_as_one_over_t(self):
        short = thermal_sigma_one_leg_mps(60.0, 1e4, _X_DOWNLINK_HZ)
        long = thermal_sigma_one_leg_mps(120.0, 1e4, _X_DOWNLINK_HZ)
        self.assertAlmostEqual(short / long, 2.0, places=9)

    def test_four_times_loop_snr_halves_the_noise(self):
        weak = thermal_sigma_one_leg_mps(60.0, 1e4, _X_DOWNLINK_HZ)
        strong = thermal_sigma_one_leg_mps(60.0, 4e4, _X_DOWNLINK_HZ)
        self.assertAlmostEqual(weak / strong, 2.0, places=9)

    def test_higher_carrier_resolves_phase_more_finely(self):
        low = thermal_sigma_one_leg_mps(60.0, 1e4, 4.0e9)
        high = thermal_sigma_one_leg_mps(60.0, 1e4, 8.0e9)
        self.assertAlmostEqual(low / high, 2.0, places=9)

    def test_inverse_cn0_round_trips(self):
        sigma = 1e-6
        cn0 = cn0_for_thermal_sigma_dbhz(sigma, 60.0, 1.0, _X_DOWNLINK_HZ)
        rho = 10.0 ** (cn0 / 10.0) / 1.0
        self.assertAlmostEqual(
            thermal_sigma_one_leg_mps(60.0, rho, _X_DOWNLINK_HZ), sigma, places=15
        )

    def test_frequency_standard_mapping_is_carrier_independent(self):
        """sigma_V from an Allan deviation does not depend on carrier frequency."""
        value = frequency_standard_sigma_mps(1e-14, independent_ends=True)
        self.assertAlmostEqual(value, 299792458.0 * math.sqrt(2.0) * 1e-14 / 2.0, places=18)
        correlated = frequency_standard_sigma_mps(1e-14, independent_ends=False)
        self.assertAlmostEqual(value / correlated, math.sqrt(2.0), places=12)

    def test_published_hmaser_interval_is_carried_forward(self):
        self.assertLess(
            HYDROGEN_MASER_SIGMA_60S_LOW_MPS, HYDROGEN_MASER_SIGMA_60S_HIGH_MPS
        )
        self.assertAlmostEqual(HYDROGEN_MASER_SIGMA_60S_LOW_MPS, 4.497e-7, places=10)


class TwoWayStructureTests(unittest.TestCase):
    def _full_config(self, **kwargs):
        defaults = dict(
            spacecraft_eirp_dbw=20.0,
            spacecraft_g_over_t_db_per_k=0.0,
            transponder_loop_bandwidth_hz=10.0,
        )
        defaults.update(kwargs)
        return reference_two_way_link("Canberra", weather_cd=0.50, **defaults)

    def test_both_legs_are_modelled_separately(self):
        up, down = build_two_way_link_budget(
            self._full_config(), _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )
        self.assertEqual(up.direction, "uplink")
        self.assertEqual(down.direction, "downlink")
        self.assertAlmostEqual(up.frequency_hz, 7.2e9, places=3)
        self.assertAlmostEqual(down.frequency_hz, _X_DOWNLINK_HZ, places=3)
        self.assertNotAlmostEqual(up.free_space_loss_db, down.free_space_loss_db)

    def test_combined_variance_is_never_below_either_component(self):
        result = compute_counted_doppler_thermal_noise(
            self._full_config(), _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )
        self.assertGreaterEqual(
            result.sigma_thermal_total_mps, result.sigma_thermal_uplink_mps
        )
        self.assertGreaterEqual(
            result.sigma_thermal_total_mps, result.sigma_thermal_downlink_mps
        )

    def test_missing_spacecraft_gives_parametric_not_a_silent_number(self):
        result = compute_counted_doppler_thermal_noise(
            reference_two_way_link("Canberra"), _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )
        self.assertEqual(result.status, "PARAMETRIC")
        self.assertIsNone(result.sigma_thermal_total_mps)
        self.assertTrue(result.missing_inputs)
        self.assertEqual(
            result.dominant_white_component, "UNRESOLVED_WITHOUT_LINK_INPUTS"
        )

    def test_downlink_only_refuses_to_report_a_two_way_total(self):
        """A one-sided total would understate two-way Doppler noise."""
        cfg = reference_two_way_link(
            "Canberra", spacecraft_eirp_dbw=20.0
        )  # no spacecraft G/T, so no uplink
        result = compute_counted_doppler_thermal_noise(
            cfg, _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )
        self.assertIsNotNone(result.sigma_thermal_downlink_mps)
        self.assertIsNone(result.sigma_thermal_total_mps)
        self.assertTrue(any("understate" in n for n in result.notes))

    def test_white_budget_excludes_layer_three_by_construction(self):
        result = compute_counted_doppler_thermal_noise(
            self._full_config(), _LUNAR_RANGE_M, _LUNAR_RANGE_M, 45.0
        )
        self.assertTrue(any("Layer 1 only" in n for n in result.notes))
        combined = math.hypot(
            result.sigma_thermal_total_mps, HYDROGEN_MASER_SIGMA_60S_LOW_MPS
        )
        self.assertAlmostEqual(
            result.sigma_short_term_white_low_mps, combined, places=15
        )

    def test_reference_configs_are_labelled_as_such(self):
        cfg = reference_two_way_link("Canberra")
        self.assertEqual(cfg.ground_terminal.scenario_class, "REFERENCE_SCENARIO")
        self.assertEqual(cfg.spacecraft.scenario_class, "PARAMETRIC")
        self.assertIn("810-005", cfg.ground_terminal.provenance)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_rf_package_is_not_imported_by_the_estimator(self):
        """The RF model is opt-in: importing the estimator must not pull it in."""
        import sys

        for module in list(sys.modules):
            if module.startswith("lunar_od.rf"):
                del sys.modules[module]
        import lunar_od.estimators  # noqa: F401

        self.assertFalse(
            [m for m in sys.modules if m.startswith("lunar_od.rf")],
            "importing the estimator must not import the RF package",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
