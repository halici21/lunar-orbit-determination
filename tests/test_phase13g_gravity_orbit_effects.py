"""Phase 13G -- campaign-script smoke tests (NOT the campaign itself).

Verifies the SCRIPT-LOCAL helpers with SPICE-free, real-data-free inputs: the
``make_variant`` decomposition masks, ``rv2coe`` (round-trip against the
production ``coe2rv`` oracle + singularity behavior), RTN decomposition,
element-drift summary recovery, the C20-only-vs-classical-J2 acceleration
bridge on the committed synthetic fixture, a short synthetic-grid propagation
smoke, source hygiene (no bare quoted "MOON_PA"), and missing-real-data
handling.  The real campaign runs via ``examples/phase13g_gravity_orbit_effects.py``.
"""
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
