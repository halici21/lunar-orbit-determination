"""Phase 12 -- GRAIL gravity model loader verification.

Tests ``lunar_od.gravity_model_loader`` against tiny committed synthetic
fixtures (tests/fixtures/gravity/): SHADR/.gfc parsing, unit conversion,
normalization canonicalization, load-time truncation, metadata validation
(hard ValueError, no silent fixes), checksums, directory resolution, and the
C20/J2 bridge that ties a loaded model to the trusted Moon-J2 helper.

Pure data layer: no frame handling, no dynamics integration, no downloads.
"""
import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lunar_od.constants import J2_MOON_UNNORMALIZED, MU_MOON_M3S2, R_MOON_M
from lunar_od.force_models import body_j2_acceleration
from lunar_od.gravity_harmonics import spherical_harmonic_acceleration
from lunar_od.gravity_model_loader import (
    default_gravity_candidates,
    describe_model,
    load_lunar_gravity_model,
    resolve_gravity_dir,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gravity"
SHADR_NORM = FIXTURES / "synthetic_norm_sha.tab"
SHADR_UNNORM = FIXTURES / "synthetic_unnorm_sha.tab"
GFC_NORM = FIXTURES / "synthetic_norm.gfc"


def _norm_factor_reference(n: int, m: int) -> float:
    """Independent N_nm via math.factorial (cross-checks the lgamma path)."""
    k = 2.0 if m > 0 else 1.0
    return math.sqrt(
        k * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
    )


def _pos(lat_deg, lon_deg, radius_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return radius_m * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )


def _shadr_text(rows, r_km="1.7374000000000000E+03",
                gm_km3="4.9028000661637960E+03", degree=3, order=3, flag=1):
    header = (f"  {r_km},  {gm_km3},  0.0000000000000000E+00,"
              f"    {degree},    {order},    {flag},  0.0,  0.0")
    return "\n".join([header] + list(rows)) + "\n"


class LoaderParsingTests(unittest.TestCase):
    # 1 -- SHADR fully-normalized fixture loads correctly ------------------
    def test_load_shadr_normalized(self):
        model = load_lunar_gravity_model(SHADR_NORM)
        self.assertEqual(model.nmax, 4)
        self.assertEqual(model.mmax, 4)
        # km -> m and km^3/s^2 -> m^3/s^2 unit conversion
        self.assertLess(abs(model.r_ref_m - R_MOON_M) / R_MOON_M, 1e-14)
        self.assertLess(abs(model.mu_m3_s2 - MU_MOON_M3S2) / MU_MOON_M3S2, 1e-15)
        # coefficients placed at [n, m]; absent slots stay zero
        self.assertLess(abs(model.cbar[2, 0] - (-J2_MOON_UNNORMALIZED / math.sqrt(5.0))),
                        1e-18)
        self.assertEqual(model.cbar[2, 2], 3.47e-5)
        self.assertEqual(model.sbar[2, 1], -3.0e-9)
        self.assertEqual(model.cbar[4, 1], 0.0)   # not in the file
        self.assertEqual(model.sbar[4, 0], 0.0)
        self.assertEqual(model.frame, "MOON_PA")

    # 2 -- ICGEM .gfc fixture loads correctly ------------------------------
    def test_load_gfc(self):
        model = load_lunar_gravity_model(GFC_NORM)
        self.assertEqual(model.nmax, 3)
        self.assertEqual(model.mu_m3_s2, 4.9028001261e12)     # SI already
        self.assertEqual(model.r_ref_m, 1.738e6)               # GRAIL R_ref != R_MOON_M
        self.assertEqual(model.cbar[3, 1], 2.6e-5)
        self.assertEqual(model.sbar[3, 1], 5.5e-6)
        self.assertEqual(model.cbar[2, 2], 3.47e-5)
        # time-variable gfct record skipped and counted
        self.assertEqual(model.metadata["skipped_time_variable"], 1)
        self.assertEqual(model.metadata["format"], "icgem-gfc")

    # 3 -- unnormalized SHADR is converted to canonical normalized ---------
    def test_unnormalized_conversion(self):
        model = load_lunar_gravity_model(SHADR_UNNORM)
        cases = [
            ((2, 0), -2.0346e-4, 0.0),
            ((2, 2), 2.2e-5, 5.0e-6),
            ((3, 0), -8.46e-6, 0.0),
            ((3, 3), 1.7e-6, -6.0e-7),
        ]
        for (n, m), c_unnorm, s_unnorm in cases:
            factor = _norm_factor_reference(n, m)
            self.assertLess(abs(model.cbar[n, m] - c_unnorm / factor),
                            abs(c_unnorm / factor) * 1e-12 + 1e-20,
                            msg=f"cbar({n},{m})")
            if s_unnorm != 0.0:
                self.assertLess(abs(model.sbar[n, m] - s_unnorm / factor),
                                abs(s_unnorm / factor) * 1e-12,
                                msg=f"sbar({n},{m})")
        self.assertEqual(model.metadata["native_normalization"], "unnormalized")

    # 4 -- fully-normalized values pass through untouched -------------------
    def test_normalized_passthrough(self):
        model = load_lunar_gravity_model(SHADR_NORM)
        self.assertEqual(model.cbar[3, 1], 2.6e-5)     # byte-exact, no conversion
        self.assertEqual(model.sbar[3, 3], -2.0e-6)
        self.assertEqual(model.metadata["native_normalization"], "fully_normalized")

    # 5 -- load-time truncation (nmax and mmax) -----------------------------
    def test_truncation(self):
        m2 = load_lunar_gravity_model(SHADR_NORM, nmax=2)
        self.assertEqual(m2.nmax, 2)
        self.assertEqual(m2.cbar.shape, (3, 3))
        self.assertEqual(m2.cbar[2, 2], 3.47e-5)
        zonal = load_lunar_gravity_model(SHADR_NORM, nmax=3, mmax=0)
        self.assertEqual(zonal.mmax, 0)
        self.assertEqual(zonal.cbar[2, 2], 0.0)        # m>0 rows skipped
        self.assertEqual(zonal.sbar[3, 1], 0.0)
        self.assertNotEqual(zonal.cbar[3, 0], 0.0)     # zonal kept
        with self.assertRaises(ValueError):            # beyond the file's degree
            load_lunar_gravity_model(SHADR_NORM, nmax=9)

    # 6 -- format auto-detect and explicit override --------------------------
    def test_format_autodetect(self):
        self.assertEqual(load_lunar_gravity_model(SHADR_NORM).metadata["format"], "shadr")
        self.assertEqual(load_lunar_gravity_model(GFC_NORM).metadata["format"], "icgem-gfc")
        with tempfile.TemporaryDirectory() as tmp:
            # unknown extension: content sniff (numeric first line -> SHADR)
            p = Path(tmp) / "model.dat"
            p.write_text(SHADR_NORM.read_text())
            self.assertEqual(load_lunar_gravity_model(p).metadata["format"], "shadr")
            with self.assertRaises(ValueError):
                load_lunar_gravity_model(SHADR_NORM, fmt="banana")

    # 7 -- checksum + provenance metadata ------------------------------------
    def test_checksum_metadata(self):
        model = load_lunar_gravity_model(SHADR_NORM, nmax=3)
        expected = hashlib.sha256(SHADR_NORM.read_bytes()).hexdigest()
        self.assertEqual(model.metadata["sha256"], expected)
        self.assertEqual(model.metadata["file_degree"], 4)
        self.assertEqual(model.metadata["requested_nmax"], 3)
        self.assertEqual(model.metadata["records_loaded"], 7)  # n<=3 rows of the fixture
        line = describe_model(model)
        self.assertIn("synthetic_norm_sha.tab", line)
        self.assertIn(expected[:12], line)
        self.assertIn("nmax=3", line)


class LoaderValidationTests(unittest.TestCase):
    def _load_text(self, text, name="bad.tab"):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / name
            p.write_text(text)
            return load_lunar_gravity_model(p)

    # 8 -- metadata validation rejects wrong models --------------------------
    def test_validation_rejects(self):
        good_row = "    2,    0, -9.0990078140421440E-05,  0.0, 0.0, 0.0"
        # positive Cbar20 (sign convention violated)
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text(["    2,    0,  9.0990078140421440E-05,  0.0, 0.0, 0.0"]))
        # Earth GM is not a lunar GM
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text([good_row], gm_km3="3.9860043543609590E+05"))
        # Earth radius is outside the lunar range
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text([good_row], r_km="6.3781363000000000E+03"))
        # no (2, 0) coefficient at all
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text(["    2,    2,  3.4700000000000000E-05,  1.0E-06, 0.0, 0.0"]))
        # sine coefficient on m=0
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text(["    2,    0, -9.0990078140421440E-05,  1.0E-09, 0.0, 0.0"]))
        # degree-1 coefficient not ~0 (.gfc)
        bad_gfc = (
            "radius 1.7380000000000000E+06\n"
            "earth_gravity_constant 4.9028001261000000E+12\n"
            "max_degree 2\n"
            "norm fully_normalized\n"
            "end_of_head =====\n"
            "gfc 1 1 1.0E-03 0.0 0.0 0.0\n"
            "gfc 2 0 -9.0884969839731000E-05 0.0 0.0 0.0\n"
        )
        with self.assertRaises(ValueError):
            self._load_text(bad_gfc, name="bad.gfc")

    # 9 -- corrupt files fail with clear errors, never load garbage ----------
    def test_corrupt_file_errors(self):
        # header too short
        with self.assertRaises(ValueError):
            self._load_text("1.0, 2.0, 3.0\n")
        # garbage coefficient record
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text(["abc, def, ghi, jkl"]))
        # duplicate (n, m)
        row = "    2,    0, -9.0990078140421440E-05,  0.0, 0.0, 0.0"
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text([row, row]))
        # coefficient degree beyond the declared maximum
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text([row, "    9,    0,  1.0E-06,  0.0, 0.0, 0.0"]))
        # unknown normalization flag
        with self.assertRaises(ValueError):
            self._load_text(_shadr_text([row], flag=7))
        # .gfc missing end_of_head
        with self.assertRaises(ValueError):
            self._load_text("radius 1.738E+06\nmax_degree 2\n", name="bad.gfc")
        # .gfc missing radius
        with self.assertRaises(ValueError):
            self._load_text(
                "earth_gravity_constant 4.9028001261E+12\nmax_degree 2\n"
                "end_of_head =====\ngfc 2 0 -9.0E-05 0.0 0.0 0.0\n",
                name="bad.gfc")
        # .gfc unknown record type
        with self.assertRaises(ValueError):
            self._load_text(
                "earth_gravity_constant 4.9028001261E+12\nradius 1.738E+06\n"
                "max_degree 2\nend_of_head =====\n"
                "xyz 2 0 -9.0E-05 0.0 0.0 0.0\n",
                name="bad.gfc")


class LoaderBridgeTests(unittest.TestCase):
    # 10 -- C20/J2 bridge: loaded model reproduces the trusted J2 helper -----
    def test_bridge_c20_matches_j2_helper(self):
        model = load_lunar_gravity_model(SHADR_NORM, nmax=2, mmax=0)   # C20-only
        positions = [
            _pos(0.0, 0.0, R_MOON_M + 100e3), _pos(0.0, 90.0, R_MOON_M + 100e3),
            _pos(35.0, 140.0, R_MOON_M + 100e3), _pos(-60.0, 250.0, R_MOON_M + 100e3),
            _pos(90.0, 0.0, R_MOON_M + 100e3), _pos(20.0, 300.0, R_MOON_M + 2000e3),
        ]
        worst = 0.0
        for r in positions:
            a_model = spherical_harmonic_acceleration(r, model, np.eye(3))
            a_ref = body_j2_acceleration(
                r, MU_MOON_M3S2, R_MOON_M, J2_MOON_UNNORMALIZED, np.eye(3))
            worst = max(worst, float(np.linalg.norm(a_model - a_ref)))
        self.assertLess(worst, 1e-14, msg=f"loader C20/J2 bridge worst |da| = {worst:.3e}")

    # 11 -- loaded model works end-to-end with the Phase 11B engine ----------
    def test_engine_smoke(self):
        model = load_lunar_gravity_model(GFC_NORM)
        a = spherical_harmonic_acceleration(_pos(30.0, 60.0, 1.738e6 + 100e3), model)
        self.assertTrue(bool(np.all(np.isfinite(a))))
        mag = float(np.linalg.norm(a))
        self.assertGreater(mag, 1e-6)     # C20-dominated LLO perturbation scale
        self.assertLess(mag, 1e-2)


class GravityDirResolutionTests(unittest.TestCase):
    # 12/13 -- directory resolution mirrors the kernel-dir pattern -----------
    def test_env_var_priority(self):
        saved = os.environ.get("LUNAR_OD_GRAVITY_DIR")
        try:
            os.environ["LUNAR_OD_GRAVITY_DIR"] = str(FIXTURES)
            candidates = default_gravity_candidates()
            self.assertEqual(candidates[0], FIXTURES)
            self.assertEqual(resolve_gravity_dir(), FIXTURES)
        finally:
            if saved is None:
                os.environ.pop("LUNAR_OD_GRAVITY_DIR", None)
            else:
                os.environ["LUNAR_OD_GRAVITY_DIR"] = saved

    def test_explicit_candidates_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_gravity_dir([tmp]), Path(tmp))
            missing = Path(tmp) / "definitely_absent"
            with self.assertRaises(FileNotFoundError):
                resolve_gravity_dir([missing])

    def test_default_candidate_order_without_env(self):
        saved = os.environ.pop("LUNAR_OD_GRAVITY_DIR", None)
        try:
            candidates = default_gravity_candidates()
            self.assertEqual(candidates[0], Path.home() / "Documents" / "mice" / "gravity")
            self.assertEqual(candidates[1].name, "gravity")
            self.assertEqual(candidates[1].parent.name, "data")
        finally:
            if saved is not None:
                os.environ["LUNAR_OD_GRAVITY_DIR"] = saved


if __name__ == "__main__":
    unittest.main()
