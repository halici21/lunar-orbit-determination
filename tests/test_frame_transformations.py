"""SPICE-free frame-transformation contract tests (frame audit, 2026-07).

Locks the conventions documented in ``docs/frame_transformations.md``:
SEZ basis properties, cardinal azimuth/elevation directions, rotation
invariants, zenith singularity policies, degree/radian anchors, the
``J_SEZ = C_SEZ<-F C_F<-I J_I`` chain, and mutation detection (transpose,
reversed direction, double SEZ application, sign flip).

No SPICE, no kernels, no network: everything here runs from NumPy plus the
production ``lunar_od`` functions under test. Production frame math is only
*called*, never modified; deliberate "mutations" are computed inside the
tests to prove the assertions would catch them.
"""

import math
import unittest

import numpy as np

from lunar_od.config import Station
from lunar_od.geometry import (
    ecef2razel_sez,
    ecef2sez_dcm,
    geodetic_to_ecef_wgs84,
    wrap_to_pi,
)
from lunar_od.measurements import (
    ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM,
    MeasurementJacobianError,
    _position_measurement_jacobian_from_unit_los,
    _range_az_el_partials_sez,
)

# Latitude/longitude sample grid (degrees). Near-pole values deliberately stop
# short of the exact pole: at lat = +/-90 the S/E axes are longitude-dependent
# (no unique physical north), so the exact pole is tested for finiteness and
# orthogonality only, not for a specific azimuth.
_LAT_GRID_DEG = (-89.999999, -60.0, -30.0, 0.0, 30.0, 41.101, 60.0, 89.999999)
_LON_GRID_DEG = (0.0, 45.5, 90.0, -90.0, 180.0)

_ORTHO_TOL = 1e-14
_DET_TOL = 1e-14


def _station(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> Station:
    return Station(
        name="frame-test",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-3,
    )


def _azel_from_unit_sez(u_sez: np.ndarray) -> tuple[float, float]:
    south, east, zenith = float(u_sez[0]), float(u_sez[1]), float(u_sez[2])
    az = math.atan2(east, -south)
    if az < 0.0:
        az += 2.0 * math.pi
    el = math.atan2(zenith, math.hypot(south, east))
    return az, el


class SezDcmPropertyTests(unittest.TestCase):
    """C_SEZ<-ECEF is a proper right-handed rotation across the lat/lon grid."""

    def test_orthogonality_determinant_and_handedness_on_grid(self):
        max_ortho = 0.0
        max_det = 0.0
        for lat_deg in _LAT_GRID_DEG:
            for lon_deg in _LON_GRID_DEG:
                c = ecef2sez_dcm(math.radians(lat_deg), math.radians(lon_deg))
                ortho = float(np.max(np.abs(c @ c.T - np.eye(3))))
                det_err = abs(float(np.linalg.det(c)) - 1.0)
                max_ortho = max(max_ortho, ortho)
                max_det = max(max_det, det_err)
                self.assertLess(ortho, _ORTHO_TOL, msg=f"lat={lat_deg} lon={lon_deg}")
                self.assertLess(det_err, _DET_TOL, msg=f"lat={lat_deg} lon={lon_deg}")
                # Right-handed triad: S x E = +Z (never -Z).
                s_hat, e_hat, z_hat = c[0], c[1], c[2]
                np.testing.assert_allclose(
                    np.cross(s_hat, e_hat), z_hat, atol=1e-14,
                    err_msg=f"S x E != +Z at lat={lat_deg} lon={lon_deg}",
                )
        print(f"\n[SEZ dcm] max |CC^T - I| = {max_ortho:.3e}, max |det - 1| = {max_det:.3e}")
        self.assertLess(max_ortho, _ORTHO_TOL, msg=f"max orthogonality defect {max_ortho:.3e}")
        self.assertLess(max_det, _DET_TOL, msg=f"max determinant error {max_det:.3e}")

    def test_exact_pole_is_finite_orthogonal_and_longitude_dependent(self):
        # At the exact pole the matrix must stay a proper rotation, but the
        # S/E axes follow the (meaningless) longitude argument: document the
        # convention instead of inventing a unique physical north.
        for lon_deg in (0.0, 90.0):
            c = ecef2sez_dcm(math.radians(90.0), math.radians(lon_deg))
            self.assertTrue(np.all(np.isfinite(c)))
            np.testing.assert_allclose(c @ c.T, np.eye(3), atol=1e-15)
            np.testing.assert_allclose(c[2], [0.0, 0.0, 1.0], atol=1e-15)
        c0 = ecef2sez_dcm(math.radians(90.0), 0.0)
        c90 = ecef2sez_dcm(math.radians(90.0), math.radians(90.0))
        self.assertGreater(float(np.max(np.abs(c0 - c90))), 0.5)

    def test_scalar_input_contract(self):
        with self.assertRaises(ValueError):
            ecef2sez_dcm(np.array([0.1, 0.2]), 0.0)


class RotationInvarianceTests(unittest.TestCase):
    """Pure-rotation invariants: norm, dot product, transpose inverse."""

    def setUp(self):
        self.rng = np.random.default_rng(20260711)

    def test_norm_dot_and_roundtrip_invariance(self):
        max_norm_err = 0.0
        max_dot_err = 0.0
        max_rt_err = 0.0
        for _ in range(50):
            lat = math.radians(float(self.rng.uniform(-89.9, 89.9)))
            lon = math.radians(float(self.rng.uniform(-180.0, 180.0)))
            c = ecef2sez_dcm(lat, lon)
            a = self.rng.normal(size=3) * 1e7
            b = self.rng.normal(size=3) * 1e7
            ca, cb = c @ a, c @ b
            norm_err = abs(np.linalg.norm(ca) - np.linalg.norm(a)) / np.linalg.norm(a)
            dot_err = abs(float(ca @ cb - a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            rt_err = float(np.max(np.abs(c.T @ ca - a))) / np.linalg.norm(a)
            max_norm_err = max(max_norm_err, norm_err)
            max_dot_err = max(max_dot_err, dot_err)
            max_rt_err = max(max_rt_err, rt_err)
        print(f"\n[rotation invariants] norm {max_norm_err:.3e}, dot {max_dot_err:.3e}, "
              f"C^T round-trip {max_rt_err:.3e} (max rel errors)")
        self.assertLess(max_norm_err, 1e-15, msg=f"max relative norm error {max_norm_err:.3e}")
        self.assertLess(max_dot_err, 1e-14, msg=f"max relative dot error {max_dot_err:.3e}")
        self.assertLess(max_rt_err, 1e-15, msg=f"max round-trip error {max_rt_err:.3e}")


class CardinalDirectionTests(unittest.TestCase):
    """North/East/South/West/Zenith/Nadir through the production observable."""

    # (SEZ unit vector, expected azimuth deg or None, expected elevation deg)
    _CARDINALS = (
        ((-1.0, 0.0, 0.0), 0.0, 0.0),     # North = -S
        ((0.0, 1.0, 0.0), 90.0, 0.0),     # East
        ((1.0, 0.0, 0.0), 180.0, 0.0),    # South
        ((0.0, -1.0, 0.0), 270.0, 0.0),   # West (wrap into [0, 360))
        ((0.0, 0.0, 1.0), None, 90.0),    # Zenith (azimuth undefined)
        ((0.0, 0.0, -1.0), None, -90.0),  # Nadir (azimuth undefined)
    )

    def _check_station(self, lat_deg: float, lon_deg: float):
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        c = ecef2sez_dcm(lat, lon)
        rho = 2500.0
        for sez_dir, az_expected_deg, el_expected_deg in self._CARDINALS:
            r_rel_ecef = c.T @ (rho * np.asarray(sez_dir, dtype=float))
            az, el, rng = ecef2razel_sez(r_rel_ecef, lat, lon)
            self.assertAlmostEqual(rng, rho, places=9)
            self.assertAlmostEqual(math.degrees(el), el_expected_deg, places=9,
                                   msg=f"el at lat={lat_deg} lon={lon_deg} dir={sez_dir}")
            if az_expected_deg is not None:
                # Compare modulo 2*pi: north can round to 360-eps at some
                # stations, which is the same physical azimuth as 0.
                az_diff_deg = math.degrees(
                    float(wrap_to_pi(az - math.radians(az_expected_deg)))
                )
                self.assertAlmostEqual(az_diff_deg, 0.0, places=9,
                                       msg=f"az at lat={lat_deg} lon={lon_deg} dir={sez_dir}")
                self.assertGreaterEqual(az, 0.0)
                self.assertLess(az, 2.0 * math.pi + 1e-12)
            else:
                # Documented zenith/nadir fallback: azimuth defaults to 0.0.
                self.assertEqual(az, 0.0)

    def test_cardinals_at_itu_equator_and_near_pole(self):
        for lat_deg, lon_deg in (
            (41.101, 29.023),   # ITU Ayazaga
            (0.0, 0.0),
            (0.0, 90.0),
            (0.0, -90.0),
            (-35.40, 148.98),   # southern hemisphere (Canberra)
            (89.9, 0.0),        # near-pole: convention still exact
            (-89.9, -90.0),
        ):
            self._check_station(lat_deg, lon_deg)


class MutationDetectionTests(unittest.TestCase):
    """Deliberately wrong frame math must be clearly distinguishable."""

    def setUp(self):
        self.lat = math.radians(41.101)
        self.lon = math.radians(29.023)
        self.c = ecef2sez_dcm(self.lat, self.lon)
        # East-pointing relative vector: az exactly 90 deg in the correct chain.
        self.r_east_ecef = self.c.T @ np.array([0.0, 1000.0, 0.0])

    def _azel(self, rho_sez: np.ndarray) -> tuple[float, float]:
        return _azel_from_unit_sez(rho_sez / np.linalg.norm(rho_sez))

    def test_transpose_mutation_is_detected(self):
        az_ok, _ = self._azel(self.c @ self.r_east_ecef)
        az_bad, _ = self._azel(self.c.T @ self.r_east_ecef)  # C.T in place of C
        shift_deg = math.degrees(abs(float(wrap_to_pi(az_bad - az_ok))))
        print(f"\n[mutation] C.T in place of C: azimuth shift {shift_deg:.3f} deg")
        self.assertAlmostEqual(math.degrees(az_ok), 90.0, places=9)
        self.assertGreater(abs(wrap_to_pi(az_bad - az_ok)), math.radians(10.0))

    def test_reversed_direction_equals_transpose_mutation(self):
        # Row-vector application v @ C == C.T @ v: same failure signature.
        az_ok, _ = self._azel(self.c @ self.r_east_ecef)
        az_bad, _ = self._azel(self.r_east_ecef @ self.c)
        self.assertGreater(abs(wrap_to_pi(az_bad - az_ok)), math.radians(10.0))

    def test_double_sez_application_is_detected(self):
        rho_ok = self.c @ self.r_east_ecef
        rho_bad = self.c @ rho_ok  # SEZ rotation applied twice
        az_ok, el_ok = self._azel(rho_ok)
        az_bad, el_bad = self._azel(rho_bad)
        delta = math.hypot(wrap_to_pi(az_bad - az_ok), el_bad - el_ok)
        print(f"\n[mutation] SEZ applied twice: az/el shift {math.degrees(delta):.3f} deg")
        self.assertGreater(delta, math.radians(5.0))

    def test_sign_mutation_flips_elevation(self):
        zenith_ecef = self.c.T @ np.array([0.0, 0.0, 1000.0])
        _, el_ok = self._azel(self.c @ zenith_ecef)
        _, el_bad = self._azel(-self.c @ zenith_ecef)
        self.assertAlmostEqual(math.degrees(el_ok), 90.0, places=9)
        self.assertAlmostEqual(math.degrees(el_bad), -90.0, places=9)


class ZenithPolicyTests(unittest.TestCase):
    """Observable fallback and implicit-Jacobian raise are separate contracts."""

    def test_observable_zenith_fallback_returns_zero_azimuth(self):
        lat, lon = math.radians(30.0), math.radians(40.0)
        c = ecef2sez_dcm(lat, lon)
        r_rel_ecef = c.T @ np.array([0.0, 0.0, 5000.0])
        az, el, rng = ecef2razel_sez(r_rel_ecef, lat, lon)
        self.assertEqual(az, 0.0)
        self.assertAlmostEqual(math.degrees(el), 90.0, places=9)
        self.assertAlmostEqual(rng, 5000.0, places=9)

    def test_implicit_jacobian_raises_near_zenith(self):
        station = _station(30.0, 40.0)
        c_sez = ecef2sez_dcm(station.lat_rad, station.lon_rad)
        x_identity = np.eye(6)
        unit_los_zenith_mci = c_sez.T @ np.array([0.0, 0.0, 1.0])
        with self.assertRaises(MeasurementJacobianError):
            _position_measurement_jacobian_from_unit_los(
                np.zeros(6),
                unit_los_zenith_mci,
                np.zeros((3, 6)),
                station,
                x_identity,
            )
        # Just above the documented threshold the Jacobian must evaluate.
        tilt = 4.0 * ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM
        u_sez = np.array([tilt, 0.0, math.sqrt(1.0 - tilt**2)])
        block = _position_measurement_jacobian_from_unit_los(
            np.zeros(6),
            c_sez.T @ u_sez,
            np.zeros((3, 6)),
            station,
            x_identity,
        )
        self.assertEqual(block.shape, (3, 6))
        self.assertTrue(np.all(np.isfinite(block)))


class DegreeRadianAnchorTests(unittest.TestCase):
    """Unit-mismatch mutations move anchors by unmistakable amounts."""

    def test_geodetic_anchors_in_degrees(self):
        x, y, z = geodetic_to_ecef_wgs84(0.0, 90.0, 0.0)
        self.assertAlmostEqual(float(x), 0.0, places=6)
        self.assertAlmostEqual(float(y), 6378137.0, places=6)
        self.assertAlmostEqual(float(z), 0.0, places=9)
        x, y, z = geodetic_to_ecef_wgs84(-90.0, 0.0, 100.0)
        self.assertAlmostEqual(float(z), -(6356752.314245179 + 100.0), places=6)

    def test_radians_passed_as_degrees_is_detected(self):
        lat_deg = 41.101
        r_ok = np.array(geodetic_to_ecef_wgs84(lat_deg, 29.023, 0.0), dtype=float)
        r_bad = np.array(
            geodetic_to_ecef_wgs84(math.radians(lat_deg), 29.023, 0.0), dtype=float
        )
        self.assertGreater(float(np.linalg.norm(r_ok - r_bad)), 100_000.0)

    def test_degrees_passed_to_sez_dcm_is_detected(self):
        c_ok = ecef2sez_dcm(math.radians(41.101), math.radians(29.023))
        c_bad = ecef2sez_dcm(41.101, 29.023)  # degrees where radians expected
        self.assertGreater(float(np.max(np.abs(c_ok - c_bad))), 0.1)


class FrameJacobianChainTests(unittest.TestCase):
    """J_SEZ = C_SEZ<-F C_F<-I J_I against finite differences (SPICE-free).

    Uses a synthetic (but proper) rotation as C_F<-I so the chain logic is
    tested independently of kernels; the kernel-gated twin repeats this with
    the real receive-epoch sxform.
    """

    def setUp(self):
        self.station = _station(41.101, 29.023)
        # Stand-in inertial->fixed rotation: any proper rotation works.
        r_fi = ecef2sez_dcm(math.radians(-12.3), math.radians(57.9))
        self.x_fi = np.eye(6)
        self.x_fi[:3, :3] = r_fi
        self.x_fi[3:, 3:] = r_fi
        self.rng = np.random.default_rng(13)
        self.rho0_i = self.rng.normal(size=3)
        self.rho0_i *= 4.0e8 / np.linalg.norm(self.rho0_i)
        self.j_los = self.rng.normal(size=(3, 6)) * np.array([1.0] * 3 + [500.0] * 3)

    def _azel_of(self, rho_i: np.ndarray) -> np.ndarray:
        c_sez = ecef2sez_dcm(self.station.lat_rad, self.station.lon_rad)
        u_sez = c_sez @ (self.x_fi[:3, :3] @ (rho_i / np.linalg.norm(rho_i)))
        return np.array(_azel_from_unit_sez(u_sez))

    def test_production_jacobian_matches_finite_difference(self):
        rho = self.rho0_i
        range_m = float(np.linalg.norm(rho))
        u = rho / range_m
        proj = np.eye(3) - np.outer(u, u)
        j_unit = proj @ self.j_los / range_m
        d_range = u @ self.j_los
        block = _position_measurement_jacobian_from_unit_los(
            d_range, u, j_unit, self.station, self.x_fi
        )
        max_rel = 0.0
        for col in range(6):
            # Step sized so the LOS moves ~1e-5 * range: large enough to beat
            # double-precision cancellation, small enough for O(h^2) accuracy.
            col_gain = float(np.linalg.norm(self.j_los[:, col]))
            h = 1.0e-5 * range_m / max(col_gain, 1e-12)
            dx = np.zeros(6)
            dx[col] = h
            azel_p = self._azel_of(rho + self.j_los @ dx)
            azel_m = self._azel_of(rho - self.j_los @ dx)
            fd = wrap_to_pi(azel_p - azel_m) / (2.0 * h)
            for row, fd_val in ((1, fd[0]), (2, fd[1])):
                scale = max(abs(fd_val), 1e-12 / range_m)
                rel = abs(block[row, col] - fd_val) / scale
                max_rel = max(max_rel, rel)
        print(f"\n[frame jacobian FD, synthetic rotation] max rel mismatch {max_rel:.3e}")
        self.assertLess(max_rel, 5.0e-6, msg=f"max relative FD mismatch {max_rel:.3e}")

    def test_range_norm_invariance_through_chain(self):
        c_sez = ecef2sez_dcm(self.station.lat_rad, self.station.lon_rad)
        rho_f = self.x_fi[:3, :3] @ self.rho0_i
        rho_sez = c_sez @ rho_f
        n_i = np.linalg.norm(self.rho0_i)
        self.assertLess(abs(np.linalg.norm(rho_f) - n_i) / n_i, 1e-14)
        self.assertLess(abs(np.linalg.norm(rho_sez) - n_i) / n_i, 1e-14)

    def test_sez_partials_helper_consistency(self):
        # _range_az_el_partials_sez rows must match the analytic az/el partials
        # used by the implicit chain, at a generic geometry.
        rho_sez = np.array([-1200.0, 800.0, 1500.0])
        partials = _range_az_el_partials_sez(rho_sez)
        rng0 = np.linalg.norm(rho_sez)
        h = 1e-4
        for col in range(3):
            d = np.zeros(3)
            d[col] = h
            azel_p = np.array(_azel_from_unit_sez((rho_sez + d) / np.linalg.norm(rho_sez + d)))
            azel_m = np.array(_azel_from_unit_sez((rho_sez - d) / np.linalg.norm(rho_sez - d)))
            fd_az, fd_el = wrap_to_pi(azel_p - azel_m) / (2.0 * h)
            fd_rng = (np.linalg.norm(rho_sez + d) - np.linalg.norm(rho_sez - d)) / (2.0 * h)
            self.assertAlmostEqual(partials[0, col], fd_rng, delta=1e-7 * max(1.0, abs(fd_rng)))
            self.assertAlmostEqual(partials[1, col], fd_az, delta=1e-6 * max(1e-6, abs(fd_az)))
            self.assertAlmostEqual(partials[2, col], fd_el, delta=1e-6 * max(1e-6, abs(fd_el)))
        self.assertAlmostEqual(float(partials[0] @ rho_sez), rng0, places=6)


if __name__ == "__main__":
    unittest.main()
