"""R0A configuration/UKF lunar-J2 parity and Earth-J2 fail-closed tests (T1-T9).

Covers SCI-001 (JSON scenario j2_moon must actually reach the truth and
estimator paths) and SCI-002 (standard, square-root, and fast SR-UKF
propagation must actually consume nonzero lunar J2), plus the R0A Earth-J2
fail-closed contract on official OD paths.

All tests are deterministic (no RNG) and fast-tier (no SPICE, synthetic
ephemerides only). Wiring tests use mock/spy interception, which the R0A task
explicitly allows for reversing silent parameter drops.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lunar_od import (
    PassGeometry,
    Station,
    UnscentedTransformConfig,
    compute_position_residuals_analytic,
    propagate_truth_with_ephemeris,
    run_batch_arc_sequence,
)
from lunar_od.constants import J2_MOON_UNNORMALIZED
from lunar_od.dynamics import make_fast_sigma_propagator, propagate_state
from lunar_od.filters import run_lunar_ukf
from lunar_od.scenario_config import (
    scenario_config_from_mapping,
    validate_official_earth_j2_support,
)
from lunar_od.scenarios import PreparedArc

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_scenario_config.py"
_spec = importlib.util.spec_from_file_location("r0a_runner_under_test", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

MU_MOON = 4.9028000661e12


class _Stop(Exception):
    """Sentinel used to short-circuit the runner at a spied call."""


def _zeros3(t):
    values = np.atleast_1d(np.asarray(t, dtype=float))
    return np.zeros((values.size, 3))


def _const3(vec):
    def _f(t):
        values = np.atleast_1d(np.asarray(t, dtype=float))
        return np.tile(np.asarray(vec, dtype=float), (values.size, 1))

    return _f


def _synthetic_ephemeris(t_end_s: float, step_s: float = 60.0):
    t = np.arange(0.0, t_end_s + step_s, step_s)
    return types.SimpleNamespace(
        t_ephem_s=t,
        earth_pos_m=np.tile(np.array([3.844e8, 0.0, 0.0]), (t.size, 1)),
        sun_pos_m=np.tile(np.array([1.496e11, 0.0, 0.0]), (t.size, 1)),
        earth_position=_const3([3.844e8, 0.0, 0.0]),
        earth_velocity=_zeros3,
        sun_position=_const3([1.496e11, 0.0, 0.0]),
    )


def _position_station() -> Station:
    return Station(
        name="R0A synthetic",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_m=0.0,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
        sigma_range_rate_mps=1e-4,
    )


def _position_case(duration_s: float = 600.0, step_s: float = 60.0):
    """Small zero-J2 truth arc with clean position observations (no RNG)."""
    r0 = 1.7374e6 + 100e3
    x0 = np.array([r0, 0.0, 0.0, 0.0, np.sqrt(MU_MOON / r0), 0.0])
    t_pass = np.arange(0.0, duration_s + step_s, step_s)
    # mu_earth = mu_sun = 0 keeps the dynamics two-body(+J2); the getters must
    # still stay away from the Moon center so third-body unit vectors are finite.
    get_earth = _const3([3.844e8, 0.0, 0.0])
    get_sun = _const3([1.496e11, 0.0, 0.0])
    truth = propagate_state(
        t_pass, x0, MU_MOON, 0.0, 0.0, get_earth, get_sun, rtol=1e-11, atol=1e-12
    )
    pass_geo = PassGeometry(
        t_s=t_pass,
        earth_pos_mci_m=np.zeros((t_pass.size, 3)),
        earth_vel_mci_mps=np.zeros((t_pass.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t_pass.size, axis=0),
        stations=(_position_station(),),
        measurement_type="position",
    )
    rows = []
    for time_idx, t_s in enumerate(t_pass, start=1):
        rows.append([t_s, 0.0, 0.0, 0.0, 1, time_idx])
    obs = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_position_residuals_analytic(truth, obs, pass_geo)
    obs[:, 1:4] = h_meas
    return x0, t_pass, truth, pass_geo, obs, get_earth, get_sun


def _prepared_position_arc():
    x0, t_pass, truth, pass_geo, obs, get_earth, get_sun = _position_case()
    arc = PreparedArc(
        arc_id=1,
        start_idx=0,
        end_idx=t_pass.size - 1,
        t_pass_s=t_pass,
        truth_state_history_mci=truth,
        obs_data=obs,
        pass_geo=pass_geo,
    )
    return arc, get_earth, get_sun


def _ukf_inputs(x0, truth):
    x_start = truth[0].copy()
    x_start[:3] += np.array([10.0, -5.0, 3.0])
    x_start[3:] += np.array([0.01, 0.005, -0.002])
    p0 = np.diag([100.0**2] * 3 + [0.1**2] * 3)
    return x_start, p0


def _run_ukf(covariance_form, j2_kwargs):
    x0, t_pass, truth, pass_geo, obs, get_earth, get_sun = _position_case()
    x_start, p0 = _ukf_inputs(x0, truth)
    return run_lunar_ukf(
        t_pass,
        obs,
        x_start,
        p0,
        pass_geo,
        MU_MOON,
        0.0,
        0.0,
        get_earth,
        get_sun,
        covariance_form=covariance_form,
        config=UnscentedTransformConfig(alpha=0.3),
        rtol=1e-10,
        atol=1e-11,
        **j2_kwargs,
    )


def _minimal_payload(**overrides):
    payload = {
        "name": "r0a_wiring",
        "measurement_type": "position",
        "estimator_type": "bls_lm",
        "start_mode": "cold",
        "network": "multi",
    }
    payload.update(overrides)
    return payload


class R0AForceConfigParityTests(unittest.TestCase):
    # T1a -- JSON truth symbol: lunar J2 measurably changes the trajectory ------
    def test_t1_truth_propagation_lunar_j2_mutation(self):
        ephemeris = _synthetic_ephemeris(7200.0, 600.0)
        r0 = 1.7374e6 + 100e3
        x0 = np.array([r0, 0.0, 0.0, 0.0, np.sqrt(MU_MOON / r0), 0.0])
        t = np.arange(0.0, 7200.0 + 600.0, 600.0)
        common = dict(rtol=1e-10, atol=1e-11)
        base = propagate_truth_with_ephemeris(
            t, x0, MU_MOON, 0.0, 0.0, ephemeris, **common
        )
        with_j2 = propagate_truth_with_ephemeris(
            t, x0, MU_MOON, 0.0, 0.0, ephemeris, j2_moon=J2_MOON_UNNORMALIZED, **common
        )
        final_diff_m = float(np.linalg.norm(with_j2[-1, :3] - base[-1, :3]))
        self.assertGreater(final_diff_m, 100.0)

    # T1b + T2 -- runner wiring: config.j2_moon reaches truth AND estimator ------
    def test_t1_t2_runner_forwards_config_j2_to_truth_and_estimator(self):
        config = scenario_config_from_mapping(
            _minimal_payload(j2_moon=J2_MOON_UNNORMALIZED)
        )
        arc, _, _ = _prepared_position_arc()

        fake_ephemeris = types.SimpleNamespace(
            earth_position=_zeros3, earth_velocity=_zeros3, sun_position=_zeros3
        )
        fake_truth = mock.MagicMock(
            side_effect=lambda t, *a, **k: np.zeros((np.size(np.asarray(t)), 6))
        )
        fake_visibility = mock.MagicMock(
            return_value=(np.array([0]), np.array([1]), None, None)
        )
        batch_spy = mock.MagicMock(side_effect=_Stop)
        spice_stub = types.SimpleNamespace(
            str2et=lambda _s: 0.0, kclear=lambda: None
        )

        fixture = {
            "initial_state": {
                "state_mci_j2000_m_mps": [1.8374e6, 0.0, 0.0, 0.0, 1633.0, 0.0],
                "mu_moon_m3_s2": MU_MOON,
                "r_moon_mean_m": 1.7374e6,
            },
            "constants": {
                "mu_earth_km3_s2": [398600.4418],
                "mu_sun_km3_s2": [1.32712440018e11],
            },
            "epoch_utc": "2026-01-01T00:00:00",
        }

        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "python_port" / "fixtures"
            fixture_dir.mkdir(parents=True)
            (fixture_dir / "spice_snapshots.json").write_text(
                json.dumps(fixture), encoding="utf-8"
            )
            os.chdir(tmp)
            try:
                with mock.patch.dict(sys.modules, {"spiceypy": spice_stub}), \
                        mock.patch.object(runner, "load_spice_kernels"), \
                        mock.patch.object(
                            runner, "sample_moon_centered_ephemeris",
                            return_value=fake_ephemeris,
                        ), \
                        mock.patch.object(
                            runner, "perturb_moon_centered_ephemeris",
                            return_value=fake_ephemeris,
                        ), \
                        mock.patch.object(
                            runner, "sample_j2000_to_itrf93_transforms",
                            side_effect=lambda _et, t: np.repeat(
                                np.eye(6)[None, :, :], np.size(np.asarray(t)), axis=0
                            ),
                        ), \
                        mock.patch.object(
                            runner, "propagate_truth_with_ephemeris", fake_truth
                        ), \
                        mock.patch(
                            "lunar_od.analyze_visibility_gap_with_transforms",
                            fake_visibility,
                        ), \
                        mock.patch.object(
                            runner, "build_measurement_arcs", return_value=(arc,)
                        ), \
                        mock.patch.object(
                            runner, "make_cold_start_bank",
                            return_value=(np.zeros(6),),
                        ), \
                        mock.patch("lunar_od.run_batch_arc_sequence", batch_spy):
                    with self.assertRaises(_Stop):
                        runner.run_configured_scenario(config)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(fake_truth.call_count, 1)
        self.assertEqual(
            fake_truth.call_args.kwargs["j2_moon"], J2_MOON_UNNORMALIZED
        )
        self.assertEqual(batch_spy.call_count, 1)
        self.assertEqual(
            batch_spy.call_args.kwargs["j2_moon"], J2_MOON_UNNORMALIZED
        )

    # T3 -- standard UKF mutation ------------------------------------------------
    def test_t3_standard_ukf_lunar_j2_mutation(self):
        base = _run_ukf("standard", {"j2_moon": 0.0})
        with_j2 = _run_ukf("standard", {"j2_moon": J2_MOON_UNNORMALIZED})
        state_diff_m = float(
            np.linalg.norm(with_j2.final_state[:3] - base.final_state[:3])
        )
        self.assertGreater(state_diff_m, 1e-3)
        self.assertFalse(
            np.array_equal(with_j2.final_covariance, base.final_covariance)
        )

    # T4 -- square-root UKF mutation ---------------------------------------------
    def test_t4_square_root_ukf_lunar_j2_mutation(self):
        base = _run_ukf("square_root", {"j2_moon": 0.0})
        with_j2 = _run_ukf("square_root", {"j2_moon": J2_MOON_UNNORMALIZED})
        state_diff_m = float(
            np.linalg.norm(with_j2.final_state[:3] - base.final_state[:3])
        )
        self.assertGreater(state_diff_m, 1e-3)
        self.assertFalse(
            np.array_equal(with_j2.final_covariance, base.final_covariance)
        )

    # T5 -- fast sigma RK4 mutation + nominal-dynamics reference ------------------
    def test_t5_fast_sigma_propagator_lunar_j2_mutation_and_reference(self):
        ephemeris = _synthetic_ephemeris(4000.0)
        fast_zero = make_fast_sigma_propagator(ephemeris, MU_MOON, 0.0, 0.0)
        fast_j2 = make_fast_sigma_propagator(
            ephemeris, MU_MOON, 0.0, 0.0, j2_moon=J2_MOON_UNNORMALIZED
        )
        if fast_zero is None or fast_j2 is None:
            self.skipTest("numba unavailable; fast sigma propagator not built")
        r0 = 1.7374e6 + 100e3
        x0 = np.array([r0, 0.0, 0.0, 0.0, np.sqrt(MU_MOON / r0), 0.0])
        out_zero = fast_zero(0.0, 60.0, x0)
        out_j2 = fast_j2(0.0, 60.0, x0)
        self.assertGreater(float(np.linalg.norm(out_j2[:3] - out_zero[:3])), 1e-3)

        reference = propagate_state(
            [0.0, 60.0],
            x0,
            MU_MOON,
            0.0,
            0.0,
            ephemeris.earth_position,
            ephemeris.sun_position,
            rtol=1e-12,
            atol=1e-13,
            j2_moon=J2_MOON_UNNORMALIZED,
        )[-1, :]
        # Pre-declared fast-path tolerance: documented numba RK4 accuracy is
        # ~mm over 60 s; 0.05 m / 1e-4 m/s stays 10x below the J2 signal.
        self.assertLess(float(np.linalg.norm(out_j2[:3] - reference[:3])), 0.05)
        self.assertLess(float(np.linalg.norm(out_j2[3:] - reference[3:])), 1e-4)

    # T6 -- desktop parity: config mapping + batch->UKF/fast-sigma forwarding -----
    def test_t6_desktop_parity_config_and_batch_forwarding(self):
        config = scenario_config_from_mapping(
            _minimal_payload(
                estimator_type="ukf",
                j2_moon=J2_MOON_UNNORMALIZED,
            )
        )
        self.assertEqual(config.j2_moon, J2_MOON_UNNORMALIZED)

        arc, get_earth, get_sun = _prepared_position_arc()
        captured: dict = {}
        real_ukf = run_lunar_ukf

        def _ukf_spy(*args, **kwargs):
            captured.update(kwargs)
            return real_ukf(*args, **kwargs)

        fast_spy = mock.MagicMock(return_value=None)
        with mock.patch("lunar_od.scenarios.run_lunar_ukf", side_effect=_ukf_spy), \
                mock.patch(
                    "lunar_od.scenarios.make_fast_sigma_propagator", fast_spy
                ):
            run_batch_arc_sequence(
                (arc,),
                "position",
                "cold",
                "ukf",
                MU_MOON,
                0.0,
                0.0,
                get_earth,
                get_sun,
                cold_start_bank=(np.zeros(6),),
                ephemeris=_synthetic_ephemeris(700.0),
                j2_moon=J2_MOON_UNNORMALIZED,
            )
        self.assertEqual(captured["j2_moon"], J2_MOON_UNNORMALIZED)
        self.assertEqual(fast_spy.call_count, 1)
        self.assertEqual(
            fast_spy.call_args.kwargs["j2_moon"], J2_MOON_UNNORMALIZED
        )

    # T7 -- Earth J2 fail-closed before any propagation ---------------------------
    def test_t7_earth_j2_rejected_before_any_propagation(self):
        with self.assertRaisesRegex(
            ValueError, "enable_earth_j2=True is not supported on the official OD path"
        ):
            scenario_config_from_mapping(_minimal_payload(enable_earth_j2=True))

        import dataclasses

        config = scenario_config_from_mapping(_minimal_payload())
        config_earth = dataclasses.replace(config, enable_earth_j2=True)

        spies = {
            name: mock.MagicMock()
            for name in (
                "truth",
                "filters_state",
                "filters_stm",
                "dynamics_state",
                "dynamics_stm",
                "ukf_entry",
                "bls_entry",
                "srif_entry",
                "fast_sigma",
            )
        }
        with mock.patch.object(runner, "propagate_truth_with_ephemeris", spies["truth"]), \
                mock.patch("lunar_od.filters.propagate_state", spies["filters_state"]), \
                mock.patch(
                    "lunar_od.filters.propagate_augmented_state", spies["filters_stm"]
                ), \
                mock.patch("lunar_od.dynamics.propagate_state", spies["dynamics_state"]), \
                mock.patch(
                    "lunar_od.dynamics.propagate_augmented_state",
                    spies["dynamics_stm"],
                ), \
                mock.patch("lunar_od.scenarios.run_lunar_ukf", spies["ukf_entry"]), \
                mock.patch(
                    "lunar_od.scenarios.estimate_position_bls_lm", spies["bls_entry"]
                ), \
                mock.patch(
                    "lunar_od.scenarios.estimate_position_srif", spies["srif_entry"]
                ), \
                mock.patch(
                    "lunar_od.scenarios.make_fast_sigma_propagator",
                    spies["fast_sigma"],
                ):
            with self.assertRaisesRegex(
                ValueError,
                "enable_earth_j2=True is not supported on the official OD path",
            ):
                runner.run_configured_scenario(config_earth)
        for name, spy in spies.items():
            self.assertEqual(spy.call_count, 0, f"unexpected {name} call")

        with self.assertRaisesRegex(ValueError, "official OD path"):
            validate_official_earth_j2_support(True, context="unit")
        validate_official_earth_j2_support(False, context="unit")

    # T8 -- zero-J2 default regression (exact) ------------------------------------
    def test_t8_zero_j2_defaults_bitwise_unchanged(self):
        legacy = _run_ukf("square_root", {})
        explicit = _run_ukf("square_root", {"j2_moon": 0.0})
        np.testing.assert_array_equal(explicit.final_state, legacy.final_state)
        np.testing.assert_array_equal(
            explicit.final_covariance, legacy.final_covariance
        )

        ephemeris = _synthetic_ephemeris(1200.0, 600.0)
        r0 = 1.7374e6 + 100e3
        x0 = np.array([r0, 0.0, 0.0, 0.0, np.sqrt(MU_MOON / r0), 0.0])
        t = np.arange(0.0, 1200.0 + 600.0, 600.0)
        legacy_truth = propagate_truth_with_ephemeris(
            t, x0, MU_MOON, 0.0, 0.0, ephemeris, rtol=1e-10, atol=1e-11
        )
        explicit_truth = propagate_truth_with_ephemeris(
            t, x0, MU_MOON, 0.0, 0.0, ephemeris, rtol=1e-10, atol=1e-11, j2_moon=0.0
        )
        np.testing.assert_array_equal(explicit_truth, legacy_truth)

    # T9 -- API compatibility: legacy call signatures still work ------------------
    def test_t9_api_compatibility_without_new_parameters(self):
        arc, get_earth, get_sun = _prepared_position_arc()
        result = run_batch_arc_sequence(
            (arc,),
            "position",
            "cold",
            "bls_lm",
            MU_MOON,
            0.0,
            0.0,
            get_earth,
            get_sun,
            cold_start_bank=(np.zeros(6),),
        )
        self.assertEqual(len(result.arc_results), 1)

        ukf_result = _run_ukf("standard", {})
        self.assertEqual(ukf_result.final_state.shape, (6,))

        ephemeris = _synthetic_ephemeris(700.0)
        fast = make_fast_sigma_propagator(ephemeris, MU_MOON, 0.0, 0.0)
        if fast is not None:
            out = fast(0.0, 60.0, np.array([1.8374e6, 0.0, 0.0, 0.0, 1633.0, 0.0]))
            self.assertEqual(out.shape, (6,))


if __name__ == "__main__":
    unittest.main()
