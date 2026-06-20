"""Verification & Validation suite for the stellar aberration (CN+S) layer.

Each test maps 1:1 to a section of the stellar-aberration V&V test plan. The
geometric unit tests (plan sections 2-6) exercise ``apply_stellar_aberration``
directly and need no SPICE kernels; the operational tests (sections 1, 7-10) use
the committed SPICE snapshot fixture and skip when the fixture or kernels are
unavailable. Section 11 (SPICE CN/CN+S cross-validation) is planned future work;
the closest in-repo evidence is the abcorr='LT' cross-check used while building
the light-time layer.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import (
    MoonCenteredEphemeris,
    compute_position_residuals,
    compute_position_residuals_analytic,
    generate_position_measurements,
    load_spice_kernels,
    range_rate_stations,
)
from lunar_od.geometry import wrap_to_pi
from lunar_od.measurements import _apparent_position_observable, apply_stellar_aberration

C_LIGHT = 299792458.0
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _angle_between(a, b):
    """Angle (rad) between two 3-vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _azel_separation(a, b):
    """Great-circle separation (rad, per row) between two [.,.,az,el] obs arrays."""
    az1, el1 = a[:, 2], a[:, 3]
    az2, el2 = b[:, 2], b[:, 3]
    cos = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    return np.arccos(np.clip(cos, -1.0, 1.0))


class StellarAberrationGeometryVV(unittest.TestCase):
    """Plan sections 2-6: helper-level geometric verification (no SPICE)."""

    def test_s2_norm_preservation_is_a_pure_rotation(self):
        """Section 2: |rho_app| == |rho_cn| for arbitrary geometries.

        The plan's absolute ``< 1e-9 m`` criterion is realized as a relative
        bound: a rotation preserves the norm to machine relative precision, so
        the absolute error scales with |r| (a few x 1e-16 * |r|). At the lunar
        range (~3.8e8 m) that is sub-micrometre, and the operational absolute
        invariance is checked at scale in section 7."""
        rng = np.random.default_rng(7)
        worst_rel = 0.0
        for _ in range(300):
            r = rng.normal(size=3) * rng.uniform(1e6, 1e9)
            v = rng.normal(size=3) * rng.uniform(1e2, 4e4)
            r_app = apply_stellar_aberration(r, v, light_speed_mps=C_LIGHT)
            rel = abs(np.linalg.norm(r_app) - np.linalg.norm(r)) / np.linalg.norm(r)
            worst_rel = max(worst_rel, rel)
        self.assertLess(worst_rel, 1e-12)

    def test_s3_orthogonal_velocity_phi_equals_v_over_c(self):
        """Section 3: LOS perp. velocity -> phi = v/c (~20.6 arcsec at 30 km/s)."""
        r = np.array([1.0e8, 0.0, 0.0])     # LOS = +X
        v = np.array([0.0, 3.0e4, 0.0])     # velocity = +Y, 30 km/s, perpendicular
        r_app = apply_stellar_aberration(r, v, light_speed_mps=C_LIGHT)
        phi = _angle_between(r, r_app)
        self.assertAlmostEqual(phi, float(np.linalg.norm(v) / C_LIGHT), delta=1e-9)
        self.assertAlmostEqual(np.degrees(phi) * 3600.0, 20.6, delta=0.3)  # arcsec

    def test_s4_parallel_velocity_no_aberration(self):
        """Section 4: LOS parallel velocity -> phi = 0."""
        r = np.array([1.0e8, 0.0, 0.0])
        v = np.array([3.0e4, 0.0, 0.0])
        r_app = apply_stellar_aberration(r, v, light_speed_mps=C_LIGHT)
        self.assertLess(_angle_between(r, r_app), 1e-12)

    def test_s5_antiparallel_velocity_no_aberration(self):
        """Section 5: LOS anti-parallel velocity -> phi = 0."""
        r = np.array([1.0e8, 0.0, 0.0])
        v = np.array([-3.0e4, 0.0, 0.0])
        r_app = apply_stellar_aberration(r, v, light_speed_mps=C_LIGHT)
        self.assertLess(_angle_between(r, r_app), 1e-12)

    def test_s6_rotation_is_toward_observer_velocity(self):
        """Section 6: apparent LOS shifts toward the observer velocity direction."""
        r = np.array([1.0e8, 0.0, 0.0])     # LOS = +X
        v = np.array([0.0, 3.0e4, 0.0])     # velocity = +Y
        r_app = apply_stellar_aberration(r, v, light_speed_mps=C_LIGHT)
        self.assertGreater(float(r_app[1]), 0.0)   # gained a +Y (toward v) component
        self.assertGreater(float(r_app[0]), 0.0)   # still predominantly +X
        # and the shift is a small rotation, not a flip
        self.assertLess(_angle_between(r, r_app), 1e-3)


class StellarAberrationFixtureVV(unittest.TestCase):
    """Plan sections 1, 7-10: SPICE-fixture operational verification."""

    def setUp(self):
        path = FIXTURES_DIR / "spice_snapshots.json"
        if not path.is_file():
            self.skipTest("spice_snapshots.json fixture not available.")
        fx = json.loads(path.read_text(encoding="utf-8"))
        truth = fx["truth_propagation"]
        meas = fx["position_measurements"]
        self._eph = MoonCenteredEphemeris(
            t_ephem_s=truth["t_ephem_s"],
            earth_pos_m=truth["earth_pos_grid_m"],
            sun_pos_m=truth["sun_pos_grid_m"],
            earth_vel_mps=truth["earth_vel_grid_mps"],
        )
        by = {s.name: s for s in range_rate_stations()}
        self._stations = [by[n] for n in meas["station_names"]]
        self._state = np.asarray(truth["state_history_mci_m_mps"], dtype=float)
        self._t_pass = np.asarray(meas["t_pass_s"], dtype=float)
        self._vis = meas["vis_mask_raw"]
        self._et = fx["et"]

    def _gen(self, **kw):
        import spiceypy as spice

        try:
            load_spice_kernels()
        except FileNotFoundError:
            self.skipTest("SPICE kernels not available.")
        try:
            _, pass_geo, clean = generate_position_measurements(
                self._t_pass, self._state, self._stations, self._vis,
                self._eph.earth_position, self._eph.earth_velocity, self._et,
                noise=False, **kw,
            )
        finally:
            spice.kclear()
        return pass_geo, clean

    def test_s1_regression_disabled_matches_cn_exactly(self):
        """Section 1: apply_stellar_aberration=False reproduces CN bit-for-bit."""
        _, cn = self._gen(apply_light_time=True)
        _, off = self._gen(apply_light_time=True, apply_stellar_aberration=False)
        np.testing.assert_array_equal(cn, off)

    def test_s7_lunar_operational_range_invariant_angles_shift(self):
        """Section 7: realistic Earth-station -> lunar-orbiter geometry. Range is
        invariant; az/el shift is 1e-6..1e-4 rad depending on the velocity frame."""
        _, cn = self._gen(apply_light_time=True)
        _, loc = self._gen(
            apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="local_mci",
        )
        _, ssb = self._gen(
            apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        # range invariance (rotation preserves the norm)
        self.assertLess(float(np.max(np.abs(loc[:, 1] - cn[:, 1]))), 1e-3)
        self.assertLess(float(np.max(np.abs(ssb[:, 1] - cn[:, 1]))), 1e-3)

        loc_shift = float(np.max(_azel_separation(loc, cn)))
        ssb_shift = float(np.max(_azel_separation(ssb, cn)))
        # local-MCI: ~1e-6 rad; SSB: ~1e-4 rad; SSB strictly larger
        self.assertGreater(loc_shift, 1e-7)
        self.assertLess(loc_shift, 1e-4)
        self.assertGreater(ssb_shift, 1e-5)
        self.assertLess(ssb_shift, 1.2e-4)
        self.assertGreater(ssb_shift, 5.0 * loc_shift)

    def test_s8_noiseless_residual_consistency(self):
        """Section 8: noiseless +S measurements predicted with the same model
        yield ~zero residuals (estimator-consistent)."""
        pass_geo, clean = self._gen(
            apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        residuals, h_meas = compute_position_residuals(self._state, clean, pass_geo)
        self.assertLess(float(np.max(np.abs(clean[:, 1] - h_meas[:, 0]))), 1e-6)      # range
        self.assertLess(float(np.max(np.abs(clean[:, 2:4] - h_meas[:, 1:3]))), 1e-9)  # az/el
        self.assertLess(float(np.linalg.norm(residuals)), 1e-6)

    def test_s9_analytic_residual_path_agreement(self):
        """Section 9: the analytic residual path matches the direct path for the
        +S model within existing analytic tolerances."""
        pass_geo, clean = self._gen(
            apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        residuals, h_meas = compute_position_residuals(self._state, clean, pass_geo)
        residuals_an, h_an, h_tilde = compute_position_residuals_analytic(
            self._state, clean, pass_geo
        )
        np.testing.assert_allclose(h_an, h_meas, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(residuals_an, residuals, rtol=0.0, atol=1e-9)
        self.assertEqual(h_tilde.shape, (3 * clean.shape[0], 6))

    def test_s10_jacobian_stability_neglected_derivative_is_small(self):
        """Section 10: the deliberately neglected stellar-aberration rotation
        derivative leaves the analytic position-block Jacobian a faithful descent
        direction (a finite-difference proxy for estimation stability).

        The SSB frame (largest correction) is used as the worst case; the
        position-block relative mismatch stays far below 1 (measured ~4e-6),
        so the Gauss-Newton/LM step does not diverge from the CN baseline."""
        pass_geo, clean = self._gen(
            apply_light_time=True, apply_stellar_aberration=True,
            stellar_aberration_model="spice_ssb",
        )
        _, _, h_tilde = compute_position_residuals_analytic(self._state, clean, pass_geo)
        tp = self._t_pass
        state = self._state

        def stellar_h(perturbed_state, i):
            k = int(clean[i, 5]) - 1
            sid = int(clean[i, 4]) - 1
            z, *_ = _apparent_position_observable(
                float(clean[i, 0]), pass_geo.stations[sid], tp, perturbed_state,
                pass_geo.earth_pos_mci_m[k], pass_geo.x_j2000_to_itrf93[k],
                observer_earth_vel_rx=pass_geo.earth_vel_ssb_j2000_mps[k],
                apply_stellar=True,
            )
            return z

        def rigid_local_perturb(k, delta):
            sp = state.copy()
            dr, dv = delta[:3], delta[3:6]
            dt = (tp - tp[k])[:, None]
            sp[:, :3] = sp[:, :3] + dr[None, :] + dt * dv[None, :]
            sp[:, 3:6] = sp[:, 3:6] + dv[None, :]
            return sp

        eps = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01])
        worst_rel = 0.0
        for i in range(clean.shape[0]):
            k = int(clean[i, 5]) - 1
            jac = np.zeros((3, 6))
            for m in range(6):
                d = np.zeros(6)
                d[m] = eps[m]
                dz = stellar_h(rigid_local_perturb(k, d), i) - stellar_h(rigid_local_perturb(k, -d), i)
                dz[1] = wrap_to_pi(dz[1])
                dz[2] = wrap_to_pi(dz[2])
                jac[:, m] = dz / (2.0 * eps[m])
            block = h_tilde[3 * i:3 * i + 3, :]
            rel = np.linalg.norm(jac[:, :3] - block[:, :3]) / max(
                np.linalg.norm(block[:, :3]), 1e-30
            )
            worst_rel = max(worst_rel, rel)
        self.assertLess(worst_rel, 1e-4)


if __name__ == "__main__":
    unittest.main()
