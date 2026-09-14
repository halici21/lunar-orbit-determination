"""Numba Pines gravity-gradient twin: parity with the pure-Python reference.

The analytic Pines gradient drives the variational equations of the 42-state
augmented propagation. A Numba twin existed in ``accelerated.py`` but was wired
into nothing and had no parity test, so the qualified STM path ran on the pure
Python kernel (~830 s per 3-orbit fit arc at 100x100 -- the cost that made
Monte-Carlo OD infeasible).

These tests qualify the twin BEFORE it is allowed to drive an estimator:

  * elementwise agreement with the reference across degree, order and geometry;
  * the Laplace structure (symmetric, trace-free) the field must satisfy;
  * agreement of the assembled inertial gradient under a real DE440 rotation;
  * end-to-end 42-state STM parity through ``propagate_augmented_state``.

Tolerances are tight on purpose: the twin computes the same recursion in the
same order, so only floating-point reassociation should separate them. A
loosened tolerance here would defeat the entire point of the test.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from lunar_od.accelerated import pines_gradient_bf_fast
from lunar_od.gravity_harmonics import (
    _pines_gradient_bf,
    spherical_harmonic_gravity_gradient,
)
from lunar_od.gravity_model_loader import load_lunar_gravity_model

GL1800F_TAB = (
    "C:/Users/erayh/Documents/Python/Grad/python_port/data/gravity/"
    "gl1800f/jggrx_1800f_sha.tab"
)
#: Relative agreement required between the twin and the reference. Both run the
#: identical recursion, so this is a floating-point-reassociation budget, not a
#: physics budget.
RTOL = 1e-11

#: Body-fixed probe points: equatorial, polar, mid-latitude, near-surface and
#: high, so no single geometry can hide a term.
PROBES = (
    np.array([1.8374e6, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.8374e6]),
    np.array([1.0e6, 1.1e6, 0.9e6]),
    np.array([1.7500e6, 2.0e5, -3.0e5]),
    np.array([-2.5e6, 1.0e6, 2.0e6]),
)


def _models():
    cache = {}
    for n in (2, 8, 20, 50, 100):
        cache[n] = load_lunar_gravity_model(GL1800F_TAB, nmax=n, mmax=n)
    return cache


class PinesGradientNumbaParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = _models()
        # Warm the JIT once so timing-sensitive suites are not charged for it.
        m = cls.models[2]
        pines_gradient_bf_fast(PROBES[0], m.mu_m3_s2, m.r_ref_m,
                               m.cbar, m.sbar, m.nmax, m.mmax)

    def test_matches_reference_across_degree_and_geometry(self):
        for n, m in self.models.items():
            for r_bf in PROBES:
                ref = _pines_gradient_bf(r_bf, m.mu_m3_s2, m.r_ref_m,
                                         m.cbar, m.sbar, m.nmax, m.mmax)
                fast = pines_gradient_bf_fast(r_bf, m.mu_m3_s2, m.r_ref_m,
                                              m.cbar, m.sbar, m.nmax, m.mmax)
                self.assertEqual(fast.shape, (3, 3))
                scale = float(np.abs(ref).max())
                np.testing.assert_allclose(
                    fast, ref, rtol=RTOL, atol=RTOL * scale,
                    err_msg=f"nmax={n} r_bf={r_bf}",
                )

    def test_order_truncation_is_honoured_independently(self):
        """mmax < nmax must be respected identically by both kernels."""
        m = load_lunar_gravity_model(GL1800F_TAB, nmax=40, mmax=6)
        for r_bf in PROBES:
            ref = _pines_gradient_bf(r_bf, m.mu_m3_s2, m.r_ref_m,
                                     m.cbar, m.sbar, m.nmax, m.mmax)
            fast = pines_gradient_bf_fast(r_bf, m.mu_m3_s2, m.r_ref_m,
                                          m.cbar, m.sbar, m.nmax, m.mmax)
            np.testing.assert_allclose(fast, ref, rtol=RTOL,
                                       atol=RTOL * float(np.abs(ref).max()))

    def test_laplace_structure(self):
        """A gradient of a harmonic potential is symmetric and trace-free."""
        for n, m in self.models.items():
            for r_bf in PROBES:
                g = pines_gradient_bf_fast(r_bf, m.mu_m3_s2, m.r_ref_m,
                                           m.cbar, m.sbar, m.nmax, m.mmax)
                scale = float(np.abs(g).max())
                np.testing.assert_allclose(g, g.T, rtol=0.0, atol=1e-10 * scale)
                self.assertLess(abs(float(np.trace(g))), 1e-9 * scale,
                                msg=f"nmax={n} trace not zero")

    def test_min_radius_guard_matches_the_reference(self):
        """Both kernels must refuse the SAME near-centre input.

        The guard is ``|r| < 1.0`` m, so the probe has to sit strictly inside
        it -- a probe at exactly 1.0 m is accepted by both, which is correct.
        """
        from lunar_od.accelerated import _HARMONIC_MIN_RADIUS_M

        m = self.models[8]
        inside = np.array([0.5 * _HARMONIC_MIN_RADIUS_M, 0.0, 0.0])
        with self.assertRaises(ValueError):
            pines_gradient_bf_fast(inside, m.mu_m3_s2, m.r_ref_m,
                                   m.cbar, m.sbar, m.nmax, m.mmax)
        with self.assertRaises(ValueError):
            _pines_gradient_bf(inside, m.mu_m3_s2, m.r_ref_m,
                               m.cbar, m.sbar, m.nmax, m.mmax)


class InertialGradientParity(unittest.TestCase):
    """The assembled C^T G_bf C must agree under a real lunar rotation."""

    @classmethod
    def setUpClass(cls):
        import spiceypy as spice

        from lunar_od.spice_loader import load_spice_kernels
        load_spice_kernels(None, clear=False)
        cls.c_bf = np.asarray(
            spice.pxform("J2000", "MOON_PA_DE440", 857302356.99998), dtype=float
        )
        cls.model = dataclasses.replace(
            load_lunar_gravity_model(GL1800F_TAB, nmax=50, mmax=50),
            frame="MOON_PA_DE440",
        )

    def test_inertial_assembly_matches_reference(self):
        m = self.model
        for r_inertial in PROBES:
            ref = spherical_harmonic_gravity_gradient(r_inertial, m, self.c_bf)
            g_bf = pines_gradient_bf_fast(self.c_bf @ r_inertial, m.mu_m3_s2,
                                          m.r_ref_m, m.cbar, m.sbar,
                                          m.nmax, m.mmax)
            fast = self.c_bf.T @ g_bf @ self.c_bf
            np.testing.assert_allclose(
                fast, ref, rtol=RTOL, atol=RTOL * float(np.abs(ref).max()))


class AugmentedStmParity(unittest.TestCase):
    """End-to-end: the 42-state STM must not change when the twin is used."""

    def test_stm_parity_through_propagate_augmented_state(self):
        import lunar_od.dynamics as dyn
        from lunar_od.spice_loader import load_spice_kernels

        load_spice_kernels(None, clear=False)
        et0 = 857302356.99998
        model = dataclasses.replace(
            load_lunar_gravity_model(GL1800F_TAB, nmax=20, mmax=20),
            frame="MOON_PA_DE440",
        )
        x0 = np.array([1.8374e6, 0.0, 2.0e5, 0.0, 1.6e3, 0.0])
        aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
        teval = np.arange(0.0, 600.0 + 1.0, 120.0)
        mu = 4902.800066163796e9

        def _e(t):
            return np.array([384_400e3, 0.0, 0.0])

        def _s(t):
            return np.array([149.6e9, 0.0, 0.0])

        def _run():
            return dyn.propagate_augmented_state(
                teval, aug0, mu, 0.0, 0.0, _e, _s, method="ADAMS",
                harmonic_model=model, harmonic_epoch_et0=et0,
                harmonic_stm_opt_in=True,
            )

        original = dyn._HARMONIC_GRADIENT_FAST
        try:
            dyn._HARMONIC_GRADIENT_FAST = False
            slow = _run()
            dyn._HARMONIC_GRADIENT_FAST = True
            fast = _run()
        finally:
            dyn._HARMONIC_GRADIENT_FAST = original

        self.assertTrue(np.all(np.isfinite(fast)))
        # State must be identical: the gradient never touches the trajectory.
        np.testing.assert_allclose(fast[:, :6], slow[:, :6], rtol=0.0, atol=1e-9)
        stm_f = fast[-1, 6:].reshape(6, 6, order="F")
        stm_s = slow[-1, 6:].reshape(6, 6, order="F")
        np.testing.assert_allclose(
            stm_f, stm_s, rtol=1e-9, atol=1e-9 * float(np.abs(stm_s).max()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
