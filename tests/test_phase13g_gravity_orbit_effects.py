"""Phase 13G -- campaign-script smoke tests (NOT the campaign itself).

Verifies the SCRIPT-LOCAL helpers with SPICE-free, real-data-free inputs: the
``make_variant`` decomposition masks, ``rv2coe`` (round-trip against the
production ``coe2rv`` oracle + singularity behavior), RTN decomposition,
element-drift summary recovery, the C20-only-vs-classical-J2 acceleration
bridge on the committed synthetic fixture, a short synthetic-grid propagation
smoke, source hygiene (no bare quoted "MOON_PA"), and missing-real-data
handling.  The real campaign runs via ``examples/phase13g_gravity_orbit_effects.py``.
"""
import json
import math
import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.constants import (  # noqa: E402
    J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M,
)
from lunar_od.dynamics import _MCI_TO_MOON_BF  # noqa: E402
from lunar_od.force_models import body_j2_acceleration  # noqa: E402
from lunar_od.gravity_harmonics import spherical_harmonic_acceleration  # noqa: E402
from lunar_od.gravity_model_loader import load_lunar_gravity_model  # noqa: E402
from lunar_od.orbit import coe2rv  # noqa: E402

import phase13g_gravity_orbit_effects as p13g  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "gravity" / "synthetic_norm_sha.tab"

R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
GE = lambda t: R_ME    # noqa: E731
GS = lambda t: R_MS    # noqa: E731


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _fixture_model(nmax=4):
    return load_lunar_gravity_model(FIXTURE, nmax=nmax)


class MakeVariantTests(unittest.TestCase):
    # 1 -- decomposition masks are exactly right ------------------------------
    def test_c20_only(self):
        model = _fixture_model()
        v = p13g.make_variant(model, "c20_only")
        self.assertEqual((v.nmax, v.mmax), (2, 0))
        self.assertEqual(v.cbar[2, 0], model.cbar[2, 0])
        mask = np.ones_like(v.cbar, dtype=bool)
        mask[2, 0] = False
        self.assertTrue(np.all(v.cbar[mask] == 0.0))
        self.assertTrue(np.all(v.sbar == 0.0))

    def test_c20_c22(self):
        model = _fixture_model()
        v = p13g.make_variant(model, "c20_c22")
        self.assertEqual((v.nmax, v.mmax), (2, 2))
        self.assertEqual(v.cbar[2, 0], model.cbar[2, 0])
        self.assertEqual(v.cbar[2, 2], model.cbar[2, 2])
        self.assertEqual(v.sbar[2, 2], model.sbar[2, 2])
        self.assertEqual(v.cbar[2, 1], 0.0)      # (2,1) deliberately dropped
        self.assertEqual(v.sbar[2, 1], 0.0)

    def test_zonal_only(self):
        model = _fixture_model()
        v = p13g.make_variant(model, "zonal_only", nmax=3)
        self.assertEqual((v.nmax, v.mmax), (3, 0))
        self.assertTrue(np.all(v.cbar[:, 1:] == 0.0))
        self.assertTrue(np.all(v.sbar == 0.0))
        self.assertEqual(v.cbar[3, 0], model.cbar[3, 0])

    def test_tesseral_only(self):
        model = _fixture_model()
        v = p13g.make_variant(model, "tesseral_only", nmax=3)
        self.assertTrue(np.all(v.cbar[:, 0] == 0.0))   # C20/C30 removed
        self.assertEqual(v.cbar[2, 2], model.cbar[2, 2])
        self.assertEqual(v.cbar[3, 1], model.cbar[3, 1])
        self.assertTrue(np.any(v.cbar[:, 1:] != 0.0))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            p13g.make_variant(_fixture_model(), "sectoral_special")


class Rv2coeTests(unittest.TestCase):
    # 2 -- round-trip against the production coe2rv oracle --------------------
    def test_roundtrip(self):
        cases = [
            (R_MOON_M + 100e3, 0.01, 45.0, 30.0, 20.0, 10.0),
            (R_MOON_M + 500e3, 0.30, 63.4, 250.0, 120.0, 200.0),
            (R_MOON_M + 100e3, 0.05, 90.0, 10.0, 300.0, 350.0),
        ]
        for a, e, i_deg, raan_deg, argp_deg, nu_deg in cases:
            r, v = coe2rv(a, e, math.radians(i_deg), math.radians(raan_deg),
                          math.radians(argp_deg), math.radians(nu_deg),
                          MU_MOON_M3S2)
            coe = p13g.rv2coe(r, v, MU_MOON_M3S2)
            self.assertAlmostEqual(coe["a_m"] / a, 1.0, places=10)
            self.assertAlmostEqual(coe["e"], e, places=10)
            self.assertAlmostEqual(coe["i_rad"], math.radians(i_deg), places=10)
            self.assertAlmostEqual(coe["raan_rad"], math.radians(raan_deg), places=9)
            self.assertAlmostEqual(coe["argp_rad"], math.radians(argp_deg), places=9)
            self.assertAlmostEqual(coe["nu_rad"], math.radians(nu_deg), places=9)

    # 3 -- singularity behavior is defined, not garbage ------------------------
    def test_circular_orbit(self):
        r = np.array([R_MOON_M + 100e3, 0.0, 0.0])
        vmag = math.sqrt(MU_MOON_M3S2 / (R_MOON_M + 100e3))
        v = np.array([0.0, vmag * math.cos(math.radians(45.0)),
                      vmag * math.sin(math.radians(45.0))])
        coe = p13g.rv2coe(r, v, MU_MOON_M3S2)
        self.assertLess(coe["e"], 1e-9)
        self.assertTrue(math.isnan(coe["argp_rad"]))
        self.assertTrue(math.isnan(coe["nu_rad"]))
        self.assertFalse(math.isnan(coe["u_rad"]))     # argument of latitude OK

    def test_equatorial_orbit(self):
        r, v = coe2rv(R_MOON_M + 200e3, 0.05, 0.0, 0.0, math.radians(40.0),
                      math.radians(60.0), MU_MOON_M3S2)
        coe = p13g.rv2coe(r, v, MU_MOON_M3S2)
        self.assertLess(coe["i_rad"], 1e-9)
        self.assertTrue(math.isnan(coe["raan_rad"]))
        self.assertFalse(math.isnan(coe["true_longitude_rad"]))


class RtnTests(unittest.TestCase):
    # 4 -- orthonormal basis with the expected directions ----------------------
    def test_basis_orthonormal_and_signs(self):
        r = np.array([R_MOON_M + 100e3, 0.0, 0.0])
        v = np.array([0.0, 1650.0, 0.0])
        basis = p13g.rtn_basis(r, v)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0], atol=1e-14)  # R
        np.testing.assert_allclose(basis[1], [0.0, 1.0, 0.0], atol=1e-14)  # T
        np.testing.assert_allclose(basis[2], [0.0, 0.0, 1.0], atol=1e-14)  # N

    def test_along_track_projection(self):
        ref = np.array([[R_MOON_M, 0.0, 0.0, 0.0, 1650.0, 0.0]])
        traj = ref.copy()
        traj[0, 1] += 123.0                       # displaced along velocity
        summary = p13g.rtn_summary(traj, ref)
        self.assertAlmostEqual(summary["final_along_m"], 123.0, places=9)
        self.assertAlmostEqual(summary["final_radial_m"], 0.0, places=9)
        self.assertAlmostEqual(summary["final_cross_m"], 0.0, places=9)


class ElementDriftTests(unittest.TestCase):
    # 5 -- drift summary recovers a known slope + amplitude ---------------------
    def test_recovers_linear_drift_and_amplitude(self):
        # orbit-averaged differencing must recover the secular slope even with
        # a short-period amplitude 25x larger (a plain linear fit leaks here)
        t = np.arange(0.0, 86400.0 + 1.0, 120.0)
        slope = 2.0 / 86400.0                     # 2 units/day
        series = {"a_m": 1.0e6 + slope * t + 50.0 * np.sin(2 * np.pi * t / 7200.0)}
        row = p13g.element_drift_summary(t, series, period_s=7200.0)[0]
        self.assertAlmostEqual(row["drift_per_day"], 2.0, delta=0.05)
        self.assertAlmostEqual(row["short_period_amp"], 50.0, delta=2.0)

    def test_short_window_falls_back_to_endpoints(self):
        t = np.arange(0.0, 3600.0 + 1.0, 60.0)
        series = {"e": 0.01 + (1e-6 / 3600.0) * t}
        row = p13g.element_drift_summary(t, series, period_s=7200.0)[0]
        self.assertAlmostEqual(row["drift_per_day"], 1e-6 * 24.0, delta=1e-9)

    def test_nan_series_yields_nan_stats(self):
        t = np.arange(0.0, 600.0, 60.0)
        series = {"raan_rad": np.full(t.size, np.nan)}
        row = p13g.element_drift_summary(t, series)[0]
        self.assertTrue(math.isnan(row["drift_per_day"]))


class BridgeAndSmokeTests(unittest.TestCase):
    # 6 -- real-C20-only acceleration equals the classical J2 helper ------------
    def test_c20_bridge_acceleration(self):
        # fixture Cbar20 = -J2/sqrt(5) with R_ref = R_MOON_M by construction
        model = p13g.make_variant(_fixture_model(), "c20_only")
        r = np.array([R_MOON_M + 100e3, -40e3, 60e3])
        a_h = spherical_harmonic_acceleration(r, model,
                                              c_inertial_to_bf=_MCI_TO_MOON_BF)
        a_j2 = body_j2_acceleration(r, model.mu_m3_s2, model.r_ref_m,
                                    J2_MOON_UNNORMALIZED, _MCI_TO_MOON_BF)
        rel = np.linalg.norm(a_h - a_j2) / np.linalg.norm(a_j2)
        self.assertLess(rel, 1e-12)

    # 7 -- short propagation with a synthetic rotation grid is finite -----------
    def test_smoke_propagation_finite(self):
        model = p13g.make_variant(_fixture_model(), "c20_c22")
        cadence, t_end = 60.0, 600.0
        margin = max(2.0 * cadence, 120.0)
        t_grid = np.arange(-margin, t_end + margin + cadence / 2.0, cadence)
        omega = 2.6617e-6
        rots = np.array([_rotz(omega * t) for t in t_grid])
        res = p13g.run_case(GE, GS, p13g.initial_state(),
                            np.arange(0.0, t_end + 1.0, 60.0),
                            harmonic_model=model,
                            harmonic_rotation=(t_grid, rots))
        self.assertTrue(np.all(np.isfinite(res["traj"])))
        self.assertFalse(res["surface_crossing"])

    def test_frozen_and_constant_pairs(self):
        t_grid = np.arange(-120.0, 720.0, 60.0)
        rots = np.array([_rotz(2.6617e-6 * t) for t in t_grid])
        f_t, f_rots = p13g.frozen_pair((t_grid, rots))
        self.assertIs(f_t, t_grid)
        self.assertTrue(np.all(f_rots == rots[0]))
        c_t, c_rots = p13g.constant_pair(t_grid, _MCI_TO_MOON_BF)
        self.assertEqual(c_rots.shape, (t_grid.size, 3, 3))
        self.assertTrue(np.all(c_rots[5] == np.asarray(_MCI_TO_MOON_BF)))


class SensitivityCaseTests(unittest.TestCase):
    # 13G-c1 -- case factory sanity (SPICE-free, data-free) -------------------
    def test_altitude_cases(self):
        for spec, alt_km in ((p13g.SENSITIVITY_CASES[1], 200.0),
                             (p13g.SENSITIVITY_CASES[2], 500.0)):
            s0 = p13g.case_state(spec)
            coe = p13g.rv2coe(s0[:3], s0[3:], MU_MOON_M3S2)
            self.assertAlmostEqual(coe["a_m"], R_MOON_M + alt_km * 1e3, delta=1e-3)
            self.assertLess(coe["e"], 1e-10)
            # circular orbit: radius equals a everywhere
            self.assertAlmostEqual(float(np.linalg.norm(s0[:3])),
                                   R_MOON_M + alt_km * 1e3, delta=1e-3)

    def test_inclination_cases(self):
        for spec, incl_deg in zip(p13g.SENSITIVITY_CASES[3:7], (0.0, 30.0, 60.0, 90.0)):
            s0 = p13g.case_state(spec)
            coe = p13g.rv2coe(s0[:3], s0[3:], MU_MOON_M3S2)
            self.assertAlmostEqual(coe["i_rad"], math.radians(incl_deg), places=10)
            self.assertAlmostEqual(coe["a_m"], R_MOON_M + 100e3, delta=1e-3)

    def test_eccentric_case(self):
        spec = p13g.SENSITIVITY_CASES[7]
        s0 = p13g.case_state(spec)
        coe = p13g.rv2coe(s0[:3], s0[3:], MU_MOON_M3S2)
        peri_alt = coe["a_m"] * (1.0 - coe["e"]) - R_MOON_M
        apo_alt = coe["a_m"] * (1.0 + coe["e"]) - R_MOON_M
        self.assertAlmostEqual(peri_alt, 80e3, delta=1.0)
        self.assertAlmostEqual(apo_alt, 500e3, delta=1.0)
        self.assertAlmostEqual(coe["i_rad"], math.radians(45.0), places=10)

    def test_case_table_unique_and_complete(self):
        ids = [spec["case"] for spec in p13g.SENSITIVITY_CASES]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        axes = [spec["axis"] for spec in p13g.SENSITIVITY_CASES]
        self.assertEqual(axes.count("altitude"), 3)
        self.assertEqual(axes.count("inclination"), 4)
        self.assertEqual(axes.count("eccentric"), 1)

    def test_matrix_guard_no_explosion(self):
        plan = p13g.build_sensitivity_matrix()
        self.assertEqual(len(plan), 8)
        for case_id, entry in plan.items():
            self.assertLessEqual(len(entry["g660_runs"]), 9)
        gl_cases = [c for c, entry in plan.items() if entry["gl1800f"]]
        self.assertEqual(sorted(gl_cases),
                         sorted(["S1_alt100_i45", "S7_alt100_i90",
                                 "S8_ecc80x500_i45"]))

    def test_perilune_mask_selects_perilune(self):
        # synthetic elliptic states around the orbit: mask must pick |nu|<30
        spec = p13g.SENSITIVITY_CASES[7]
        s0 = p13g.case_state(spec)
        coe = p13g.rv2coe(s0[:3], s0[3:], MU_MOON_M3S2)
        states = []
        for nu_deg in range(0, 360, 20):        # off the +/-30 deg boundary
            r, v = coe2rv(coe["a_m"], coe["e"], coe["i_rad"], coe["raan_rad"],
                          coe["argp_rad"], math.radians(nu_deg), MU_MOON_M3S2)
            states.append(np.concatenate([r, v]))
        mask = p13g.perilune_mask_from(np.array(states), 30.0)
        # nu in {0, 20, 340} within +/-30 deg -> exactly 3 of 18 samples
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(mask[0] and mask[1] and mask[-1])

    def test_window_setup_uses_case_period(self):
        # pure orbital-period math (no ephemeris load): S3 500 km period
        s0 = p13g.case_state(p13g.SENSITIVITY_CASES[2])
        a0 = p13g.rv2coe(s0[:3], s0[3:], MU_MOON_M3S2)["a_m"]
        period = 2.0 * math.pi * math.sqrt(a0 ** 3 / MU_MOON_M3S2)
        self.assertGreater(period, 9000.0)        # ~9498 s, NOT the 7067 s
        self.assertLess(period, 10000.0)          # baseline period


class SevenDayMatrixTests(unittest.TestCase):
    # 13G-c2 -- plan/matrix guards (SPICE-free, data-free) ---------------------
    def test_matrix_cases_and_bounds(self):
        plan = p13g.build_sevenday_matrix()
        ids = [k for k in plan if not k.startswith("_")]
        self.assertEqual(ids, ["L1", "L2", "L3", "L4"])       # unique + ordered
        for required in ("L1", "L2", "L3"):
            self.assertIn(required, plan)
        self.assertIn("L4", plan)                             # explicit control
        self.assertLessEqual(plan["_total_runs"], 21)
        # GL1800F restricted: only L2/L3; 256 only on L3
        self.assertNotIn("g1800_128", plan["L1"]["runs"])
        self.assertNotIn("g1800_128", plan["L4"]["runs"])
        self.assertIn("g1800_128", plan["L2"]["runs"])
        self.assertNotIn("g1800_256", plan["L2"]["runs"])
        self.assertIn("g1800_256", plan["L3"]["runs"])

    def test_duration_and_sampling(self):
        self.assertEqual(p13g.SEVENDAY_T_END_S, 604800.0)
        for cid, step in p13g.SEVENDAY_OUT_STEP_S.items():
            self.assertEqual(604800.0 % step, 0.0,
                             f"{cid} sampling must cover the final epoch")
            self.assertEqual(86400.0 % step, 0.0,
                             f"{cid} daily boundaries must sit on the grid")
        # L3 eccentric uses finer cadence than the circular cases
        self.assertLess(p13g.SEVENDAY_OUT_STEP_S["L3"],
                        min(p13g.SEVENDAY_OUT_STEP_S[c]
                            for c in ("L1", "L2", "L4")))

    def test_initial_states_match_sensitivity_cases(self):
        refs = {s["id"]: s["case_ref"] for s in p13g.SEVENDAY_CASES}
        sens = {s["case"]: s for s in p13g.SENSITIVITY_CASES}
        for cid, ref in refs.items():
            self.assertIn(ref, sens, f"{cid} references unknown case {ref}")
            s0 = p13g.case_state(sens[ref])       # same factory, same state
            self.assertEqual(s0.shape, (6,))

    def test_preflight_gate_logic(self):
        good = {"j2_only": {"final_dpos_m": 0.001},
                "full128": {"final_dpos_m": 0.002}}
        comp = {"continuous_final_dpos_m": 2000.0,
                "artifact_final_dpos_m": 0.01, "artifact_max_dpos_m": 0.02}
        gate = p13g.evaluate_chunk_preflight(good, comp)
        self.assertTrue(gate["pass"])
        bad_model = {"j2_only": {"final_dpos_m": 5.0},
                     "full128": {"final_dpos_m": 0.001}}
        self.assertFalse(p13g.evaluate_chunk_preflight(bad_model, comp)["pass"])
        bad_comp = dict(comp, artifact_final_dpos_m=100.0)
        self.assertFalse(p13g.evaluate_chunk_preflight(good, bad_comp)["pass"])

    def test_preflight_fail_records_continuous_decision(self):
        # the chunked methodology was rejected; a failed gate no longer
        # aborts — it is preserved as evidence with the fixed decision
        failing = {"per_model": {"j2_only": {"final_dpos_m": 0.13},
                                 "full128": {"final_dpos_m": 0.45}},
                   "comparison": {"continuous_final_dpos_m": 36222.0,
                                  "artifact_final_dpos_m": 0.285,
                                  "artifact_max_dpos_m": 0.341}}
        failing["gate"] = p13g.evaluate_chunk_preflight(
            failing["per_model"], failing["comparison"])
        block = p13g.preflight_decision_block(failing)
        self.assertEqual(block["status"], "failed")
        self.assertEqual(block["decision"], "use_continuous_campaign_propagation")
        self.assertEqual(block["reason"], "multistep_integrator_restart_artifact")
        # measurements retained
        self.assertIn("per_model", block)
        self.assertIn("comparison", block)
        import inspect
        campaign_src = inspect.getsource(p13g.run_sevenday)
        self.assertNotIn("was NOT run", campaign_src)      # no abort path
        self.assertIn("preflight_decision_block", campaign_src)

    # §9A — campaign routing: scientific cases never use the chunked runner
    def test_campaign_routes_continuous_not_chunked(self):
        import inspect
        campaign_src = inspect.getsource(p13g.run_sevenday)
        self.assertIn("run_case_continuous_sevenday(", campaign_src)
        self.assertNotIn("run_case_chunked(", campaign_src)
        # the chunked runner and preflight helpers still EXIST (evidence
        # tooling, not campaign execution)
        self.assertTrue(callable(p13g.run_case_chunked))
        self.assertTrue(callable(p13g.run_chunk_preflight))


class SevenDayContinuousRunnerTests(unittest.TestCase):
    """§9B/C/D — one propagation call, grid contract, crossing truncation."""

    def _mock_run_case(self, radius_profile):
        calls = []

        def fake(get_earth, get_sun, s0, teval, **kw):
            calls.append(np.asarray(teval))
            n = np.asarray(teval).size
            traj = np.zeros((n, 6))
            traj[:, 0] = radius_profile(n)
            return {"traj": traj, "runtime_s": 1.0, "rhs_evals": 42,
                    "min_radius_m": float(traj[:, 0].min()),
                    "min_altitude_m": float(traj[:, 0].min() - R_MOON_M),
                    "surface_crossing": bool((traj[:, 0] <= R_MOON_M).any())}
        return fake, calls

    def test_single_call_and_grid_contract(self):
        from unittest import mock
        fake, calls = self._mock_run_case(
            lambda n: np.full(n, R_MOON_M + 90e3))
        with mock.patch.object(p13g, "run_case", fake):
            res = p13g.run_case_continuous_sevenday(
                None, None, np.zeros(6), p13g.SEVENDAY_T_END_S, 300.0)
        self.assertEqual(len(calls), 1)                    # ONE propagation
        teval = calls[0]
        self.assertEqual(teval[0], 0.0)
        self.assertEqual(teval[-1], 604800.0)
        self.assertAlmostEqual(float(np.diff(teval).max()), 300.0)
        self.assertEqual(teval.size, int(604800 / 300) + 1)
        self.assertEqual(res["status"], "complete")
        self.assertFalse(res["crossing_detected"])
        self.assertEqual(res["returned_samples"], res["requested_samples"])
        self.assertEqual(res["propagation_calls"], 1)
        self.assertEqual(res["valid_duration_s"], 604800.0)

    def test_crossing_truncates_single_uninterrupted_call(self):
        from unittest import mock

        def profile(n):
            r = np.full(n, R_MOON_M + 50e3)
            r[100:] = R_MOON_M - 500.0                  # crossing at index 100
            return r

        fake, calls = self._mock_run_case(profile)
        with mock.patch.object(p13g, "run_case", fake):
            res = p13g.run_case_continuous_sevenday(
                None, None, np.zeros(6), p13g.SEVENDAY_T_END_S, 300.0)
        self.assertEqual(len(calls), 1)                    # not split by crossing
        self.assertEqual(res["status"], "surface_crossing_detected")
        self.assertTrue(res["crossing_detected"])
        self.assertEqual(res["crossing_index"], 100)
        self.assertEqual(res["crossing_time_s"], 100 * 300.0)
        self.assertEqual(res["returned_samples"], 100)
        self.assertEqual(res["last_valid_time_s"], 99 * 300.0)
        self.assertIsNotNone(res["minimum_radius_margin_before_crossing_m"])
        self.assertLess(res["returned_samples"], res["requested_samples"])

    # §9E — comparisons use the minimum common valid horizon
    def test_common_horizon_comparison(self):
        t_full = np.arange(0.0, 604800.0 + 1.0, 300.0)
        n_short = 1000
        mk = lambda n, off: {                              # noqa: E731
            "t": t_full[:n],
            "traj": np.tile([R_MOON_M + 90e3 + off, 0, 0, 0, 1650.0, 0],
                            (n, 1)).astype(float),
            "validity": "valid" if n == t_full.size
                        else "invalid_after_first_crossing",
            "valid_duration_s": float(t_full[n - 1])}
        res_a = mk(t_full.size, 0.0)
        res_b = mk(n_short, 5.0)                           # truncated model
        row = p13g.sevenday_comparison_row(
            "x", "a", "b", res_a, res_b, case_id="L9",
            requested_duration_s=604800.0,
            day_boundaries=[k * 86400.0 for k in range(1, 8)])
        self.assertEqual(row["status"], "truncated")
        self.assertEqual(row["truncation_reason"], "surface_crossing")
        self.assertEqual(row["requested_duration_s"], 604800.0)
        self.assertEqual(row["run_a_valid_duration_s"], 604800.0)
        self.assertEqual(row["run_b_valid_duration_s"],
                         float(t_full[n_short - 1]))
        self.assertEqual(row["common_comparison_duration_s"],
                         float(t_full[n_short - 1]))
        # day boundaries beyond the common horizon are None
        self.assertIsNotNone(row["daily_endpoint_dpos_m"][0])   # day 1 < horizon
        self.assertIsNone(row["daily_endpoint_dpos_m"][6])      # day 7 > horizon

    # §9F — daily masks: a boundary sample counts in exactly one day
    def test_daily_masks_no_double_count(self):
        t = np.arange(0.0, 604800.0 + 1.0, 300.0)
        counts = np.zeros(t.size, dtype=int)
        for day in range(1, 8):
            counts += p13g.sevenday_daily_mask(t, day).astype(int)
        self.assertTrue(np.all(counts[1:] == 1))           # every t>0 in one day
        self.assertEqual(counts[0], 0)                     # t=0 in no day
        boundary = np.isclose(t, 86400.0)
        self.assertTrue(p13g.sevenday_daily_mask(t, 1)[boundary].all())
        self.assertFalse(p13g.sevenday_daily_mask(t, 2)[boundary].any())


class SevenDayHelpersTests(unittest.TestCase):
    def _circular_traj(self, e=0.0, argp_deg=20.0):
        a = R_MOON_M + 100e3
        period = 2.0 * math.pi * math.sqrt(a ** 3 / MU_MOON_M3S2)
        t = np.linspace(0.0, 2.0 * period, 41)
        states = []
        for k, _ in enumerate(t):
            nu = 2.0 * math.pi * k / 20.0
            r, v = coe2rv(a, e, math.radians(45.0), math.radians(30.0),
                          math.radians(argp_deg), nu, MU_MOON_M3S2)
            states.append(np.concatenate([r, v]))
        return t, np.array(states), period

    def test_circular_cases_report_no_argp(self):
        t, traj, period = self._circular_traj(e=0.0)
        rows = p13g.daily_element_rows(t, traj, period, run="j2_only",
                                       case_id="L1", circular=True)
        self.assertTrue(rows)
        for row in rows:
            self.assertNotIn("mean_argp_rad", row)
            self.assertNotIn("argp_drift_sign", row)
            self.assertIn("ecc_cos_comp", row)
            self.assertIn("ecc_sin_comp", row)

    def test_eccentric_case_reports_argp(self):
        t, traj, period = self._circular_traj(e=0.1)
        rows = p13g.daily_element_rows(t, traj, period, run="full128",
                                       case_id="L3", circular=False)
        self.assertIn("mean_argp_rad", rows[0])
        self.assertIn("argp_drift_sign", rows[0])

    def test_trailing_window_mask(self):
        t = np.arange(0.0, 604800.0 + 1.0, 300.0)
        lead = p13g.trailing_window_mask(t, 0.0, 7000.0, leading=True)
        self.assertTrue(np.all(t[lead] <= 7000.0 + 1e-6))
        trail = p13g.trailing_window_mask(t, 86400.0, 7000.0)
        self.assertTrue(np.all(t[trail] >= 86400.0 - 7000.0 - 1e-6))
        self.assertTrue(np.all(t[trail] <= 86400.0 + 1e-6))

    def test_crossing_truncation_excludes_post_crossing(self):
        t = np.arange(10.0) * 100.0
        traj = np.zeros((10, 6))
        radius = np.full(10, R_MOON_M + 50e3)
        radius[6:] = R_MOON_M - 1000.0                 # crossing at index 6
        traj[:, 0] = radius
        out = p13g.truncate_at_first_crossing(t, traj)
        self.assertEqual(out["validity"], "invalid_after_first_crossing")
        self.assertEqual(out["first_crossing_t_s"], 600.0)
        self.assertEqual(out["traj"].shape[0], 6)      # rows 6.. excluded
        clean = p13g.truncate_at_first_crossing(
            t, np.tile([R_MOON_M + 50e3, 0, 0, 0, 0, 0], (10, 1)))
        self.assertEqual(clean["validity"], "valid")

    def test_growth_profile_rules(self):
        near_zero = p13g.growth_profile([0.1, 1, 2, 3, 4, 5, 6])
        self.assertIsNone(near_zero["ratios"])
        self.assertEqual(near_zero["classification"], "indeterminate")
        linear = p13g.growth_profile([100, 200, 300, 400, 500, 600, 700])
        self.assertEqual(linear["classification"], "approximately linear")
        sub = p13g.growth_profile([100, 120, 130, 135, 140, 142, 145])
        self.assertTrue(sub["classification"].startswith("sublinear"))
        osc = p13g.growth_profile([100, 400, 150, 500, 120, 450, 130])
        self.assertIn("oscillatory", osc["classification"])
        self.assertEqual(linear["label"], p13g.GROWTH_LABEL)

    def test_gl256_gate_records_skip_reason(self):
        go = p13g.gl256_gate(50.0)
        self.assertTrue(go["run"])
        skip = p13g.gl256_gate(200.0)
        self.assertFalse(skip["run"])
        self.assertIn("exceeds", skip["reason"])
        self.assertIn("not a guarantee", skip["basis"])


class SevenDayStoreAndReportTests(unittest.TestCase):
    def test_init_stage_replaces_stale_partial(self):
        store = {"sevenday": {"status": "partial",
                              "cases": {"L1": {"stale": True}}},
                 "baseline": {"keep": True}}
        stage = p13g._init_sevenday_stage(store)
        self.assertEqual(stage["status"], "partial")
        self.assertEqual(stage["cases"], {})               # stale data gone
        self.assertEqual(store["baseline"], {"keep": True})  # others intact
        self.assertEqual(stage["planned_cases"], ["L1", "L2", "L3", "L4"])

    def test_atomic_store_write_preserves_old_on_failure(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "store.json"
            target.write_text('{"ok": 1}', encoding="utf-8")
            with mock.patch.object(p13g, "STORE_PATH", target), \
                    mock.patch.object(p13g, "OUT", Path(tmp)):
                p13g._atomic_store_dump({"ok": 2})
                self.assertEqual(json.loads(target.read_text())["ok"], 2)
                with mock.patch.object(p13g.json, "dumps",
                                       side_effect=RuntimeError("boom")):
                    with self.assertRaises(RuntimeError):
                        p13g._atomic_store_dump({"ok": 3})
                self.assertEqual(json.loads(target.read_text())["ok"], 2)

    def test_sevenday_report_language(self):
        import tempfile
        from unittest import mock
        stage = {
            "status": "complete", "completed_cases": ["L1"],
            "evidence_label": ("selected multi-day confirmation - Level-1 "
                               "internal model-vs-model; not estimator "
                               "performance"),
            "chunk_preflight": {
                "status": "failed",
                "decision": "use_continuous_campaign_propagation",
                "reason": "multistep_integrator_restart_artifact",
                "per_model": {"j2_only": {
                    "final_dpos_m": 1.286e-01, "max_dpos_m": 1.286e-01,
                    "final_dvel_mps": 1.15e-4, "final_da_m": 2.0e-3,
                    "final_de": 2.9e-10, "final_di_rad": 6.3e-12}},
                "comparison": {"continuous_final_dpos_m": 36222.4,
                               "chunked_final_dpos_m": 36222.1,
                               "artifact_final_dpos_m": 0.285,
                               "artifact_max_dpos_m": 0.341},
                "gate": {"pass": False, "rule": "test-rule",
                         "rel_limit_m": 362.2}},
            "gl256_gate": {"run": False, "reason": "projected too long"},
            "cases": {"L1": {
                "case_ref": "S1_alt100_i45", "out_step_s": 300.0,
                "period_s": 7068.0,
                "runs": [{"case_id": "L1", "run": "j2_only",
                          "status": "complete", "validity": "valid"}],
                "comparisons": [{
                    "case_id": "L1", "comparison": "j2only_vs_full128",
                    "status": "complete", "final_dpos_m": 90000.0,
                    "daily_endpoint_dpos_m": [1e4] * 7,
                    "growth": {"ratios": [1.0] * 7,
                               "classification": "approximately linear",
                               "label": p13g.GROWTH_LABEL}}],
                "elements": [], "daily": []}},
        }
        store = {"sevenday": stage, "generated_utc": "T"}
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "seven.md"
            with mock.patch.object(p13g, "SEVEN_MD_PATH", md_path):
                p13g._write_sevenday_md(store)
            md = md_path.read_text(encoding="utf-8")
        self.assertIn("selected multi-day confirmation", md)
        self.assertIn("Level-1", md)
        self.assertIn("not estimator performance", md)
        self.assertIn("trajectory separation != estimator error", md)
        self.assertIn(p13g.CROSSING_LIMITATION, md)
        self.assertIn("does not constitute a seven-day rerun of the "
                      "Phase 13G-b e=0.01 baseline case", md)
        self.assertIn("No surface crossing was observed over seven days", md)
        self.assertNotIn("is long-term stable", md)
        self.assertIn("no long-term orbital-stability claim is made", md)
        self.assertIn(p13g.GROWTH_LABEL, md)
        # §9G — preflight evidence retention + continuous-decision language
        self.assertIn("rejected by preflight", md)
        self.assertIn("uninterrupted continuous propagation", md)
        self.assertIn("failed", md)
        self.assertIn("use_continuous_campaign_propagation", md)
        self.assertIn("multistep_integrator_restart_artifact", md)


class HygieneTests(unittest.TestCase):
    # 8 -- no bare quoted "MOON_PA" anywhere in the campaign source -------------
    def test_no_bare_moon_pa_literal(self):
        source = (ROOT / "examples" / "phase13g_gravity_orbit_effects.py").read_text(
            encoding="utf-8"
        )
        bare = [m for m in re.finditer(r"[\"']MOON_PA[\"']", source)]
        self.assertEqual(bare, [], "bare quoted MOON_PA literal found")

    # 9 -- missing real data produces a clear error, not silence ----------------
    def test_missing_real_data_message(self):
        previous = os.environ.get("LUNAR_OD_GRAVITY_DIR")
        os.environ["LUNAR_OD_GRAVITY_DIR"] = str(FIXTURE.parent)  # exists, no grgm dir
        try:
            with self.assertRaisesRegex(FileNotFoundError, "real GRAIL file not found"):
                p13g.load_real_model("grgm660prim", 8)
        finally:
            if previous is None:
                del os.environ["LUNAR_OD_GRAVITY_DIR"]
            else:
                os.environ["LUNAR_OD_GRAVITY_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
