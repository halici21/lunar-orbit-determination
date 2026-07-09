"""Phase 12B -- OPTIONAL tests against the real GRGM660PRIM GRAIL file.

These tests run only when the real (never-committed, gitignored) GRAIL file
``data/gravity/grgm660prim/gggrx_0660pm_sha.tab`` is present locally; in any
other environment every test is skipped and the full suite stays green.

Rules: no network, no download, GRGM660PRIM only (GL1800F is a 198 MB parse
and deliberately stays out of the suite -- covered by the phase12b script),
nmax <= 8 so the whole module stays fast (~1 s).  No m > 0 propagation here:
a physically-legitimate single-point acceleration check uses a ZONAL-ONLY
(mmax=0) variant with a constant frame; the real tesseral campaign lives in
``examples/phase12b_real_grail_validation.py``.
"""
import dataclasses
import math
import unittest
from pathlib import Path

import numpy as np

from lunar_od.constants import (
    J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M,
)
from lunar_od.dynamics import propagate_augmented_state, propagate_state
from lunar_od.gravity_harmonics import spherical_harmonic_acceleration
from lunar_od.gravity_model_loader import (
    load_lunar_gravity_model, resolve_gravity_dir,
)

_SANITY_NMAX = 8


def _find_real_file(name: str) -> Path | None:
    """Locate a grgm660prim file without raising in dataless environments."""
    try:
        gravity_dir = resolve_gravity_dir()
    except FileNotFoundError:
        return None
    path = gravity_dir / "grgm660prim" / name
    return path if path.is_file() else None


_TAB_PATH = _find_real_file("gggrx_0660pm_sha.tab")
_LBL_PATH = _find_real_file("gggrx_0660pm_sha.lbl")

_SKIP_REASON = (
    "real GRGM660PRIM file not available locally "
    "(data/gravity/grgm660prim/gggrx_0660pm_sha.tab); optional test skipped"
)


def _getters():
    """Constant third-body getters (guards raise at setup, before integration)."""
    r_earth = np.array([384_400e3, 0.0, 0.0])
    r_sun = np.array([1.496e11, 0.0, 0.0])
    return (lambda t: r_earth), (lambda t: r_sun)


def _llo_state() -> np.ndarray:
    r = R_MOON_M + 100e3
    v = math.sqrt(MU_MOON_M3S2 / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


@unittest.skipUnless(_TAB_PATH is not None, _SKIP_REASON)
class RealGrail660Tests(unittest.TestCase):
    """Loader/metadata/guard checks with the real GRGM660PRIM coefficients."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_lunar_gravity_model(_TAB_PATH, nmax=_SANITY_NMAX)

    # 1 -- label file rides along with the coefficients --------------------
    def test_label_file_present(self):
        if _LBL_PATH is None:
            self.skipTest("gggrx_0660pm_sha.lbl not present next to the .tab")
        self.assertTrue(_LBL_PATH.is_file())

    # 2 -- loader sanity ----------------------------------------------------
    def test_loader_sanity(self):
        model = self.model
        self.assertEqual(model.nmax, _SANITY_NMAX)
        self.assertLessEqual(model.mmax, _SANITY_NMAX)
        self.assertTrue(np.isfinite(model.cbar).all())
        self.assertTrue(np.isfinite(model.sbar).all())
        self.assertTrue(np.isfinite(model.mu_m3_s2) and model.mu_m3_s2 > 0.0)
        # GRAIL header declares 1738.0 km exactly -- the MODEL's own radius,
        # never to be confused with the body constant R_MOON_M (1737.4 km).
        self.assertEqual(model.r_ref_m, 1_738_000.0)
        self.assertNotEqual(model.r_ref_m, R_MOON_M)

    # 3 -- C20 sign and the J2 bridge ---------------------------------------
    def test_c20_j2_sanity(self):
        cbar20 = float(self.model.cbar[2, 0])
        self.assertLess(cbar20, 0.0)
        j2_derived = -math.sqrt(5.0) * cbar20
        rel_diff = abs(j2_derived - J2_MOON_UNNORMALIZED) / J2_MOON_UNNORMALIZED
        # measured 0.118% vs the constants value; 1% keeps this robust without
        # demanding bit-agreement between different gravity solutions
        self.assertLess(rel_diff, 0.01)

    # 4 -- the real model is tesseral (m > 0 present) ------------------------
    def test_tesseral_coefficients_present(self):
        cbar, sbar = self.model.cbar, self.model.sbar
        self.assertNotEqual(float(cbar[2, 2]), 0.0)
        self.assertTrue(bool(np.any(cbar[:, 1:] != 0.0) or np.any(sbar[:, 1:] != 0.0)))

    # 5 -- double-count guard fires with the real model ----------------------
    def test_double_count_guard(self):
        get_earth, get_sun = _getters()
        with self.assertRaises(ValueError) as ctx:
            propagate_state(
                np.array([0.0, 60.0]), _llo_state(),
                MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, get_earth, get_sun,
                j2_moon=J2_MOON_UNNORMALIZED, harmonic_model=self.model,
            )
        self.assertIn("count J2 twice", str(ctx.exception))

    # 6 -- STM/augmented refusal fires with the real model --------------------
    def test_stm_guard(self):
        get_earth, get_sun = _getters()
        x_aug0 = np.concatenate([_llo_state(), np.eye(6).flatten(order="F")])
        with self.assertRaises(ValueError) as ctx:
            propagate_augmented_state(
                np.array([0.0, 60.0]), x_aug0,
                MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, get_earth, get_sun,
                harmonic_model=self.model,
            )
        self.assertIn("lunar harmonics gradient not implemented", str(ctx.exception))

    # 7 -- single-point acceleration, zonal-only + constant frame (legit) -----
    def test_zonal_only_acceleration_finite(self):
        model = self.model
        cbar = model.cbar.copy()
        sbar = np.zeros_like(model.sbar)
        cbar[:, 1:] = 0.0
        zonal = dataclasses.replace(model, cbar=cbar, sbar=sbar, mmax=0)
        r = np.array([R_MOON_M + 100e3, 0.0, 0.0])
        a = spherical_harmonic_acceleration(r, zonal, c_inertial_to_bf=np.eye(3))
        self.assertTrue(np.isfinite(a).all())
        a_norm = float(np.linalg.norm(a))
        a_central = MU_MOON_M3S2 / float(np.dot(r, r))
        # perturbation only: nonzero but far below the central term
        self.assertGreater(a_norm, 0.0)
        self.assertLess(a_norm, 1e-2 * a_central)


if __name__ == "__main__":
    unittest.main()
