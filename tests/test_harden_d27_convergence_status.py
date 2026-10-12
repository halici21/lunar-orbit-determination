"""Phase 17-HARDEN-D27: BLS-LM convergence/status contract at damping exhaustion.

Damping exhaustion (lambda past the 1e12 guard) only says that no trial step
lowered the scored cost.  At a minimiser that is expected -- the remaining
Gauss-Newton gain is below the objective's integration noise -- and far from a
minimiser it is a genuine failure.  The contract under test:

- STATIONARY_SOLUTION_STATUS: exhaustion at a point whose undamped Gauss-Newton
  decrement is <= STATIONARITY_DECREMENT_TOL (0.1 posterior sigma) reports
  ``Converged`` / ``STATIONARY_AT_DAMPING_LIMIT``.
- GENUINE_FAILURE_STATUS: exhaustion anywhere else reports ``DampingLimit`` /
  ``LAMBDA_HARD_LIMIT`` exactly as before.
- Ordinary convergence keeps ``Converged`` / ``STEP_NORM``.
- The decision never changes the returned estimate, and is made from the
  linearisation at the caller's tolerance, not from lambda.

The two-way-range estimator's end-to-end cases live in
tests/test_estimator_lm_consistency.py (TerminationSemanticsTests and
DescentAcceptanceTests); this file covers the decrement itself, the position
BLS-LM path (no SPICE needed) and the diagnostics classification.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import lunar_od.estimators as estimators
from lunar_od.config import Station
from lunar_od.diagnostics import analyze_convergence
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.estimators import (
    STATIONARITY_DECREMENT_TOL,
    EstimatorStats,
    _damping_exhaustion_status,
    _gauss_newton_decrement,
    estimate_position_bls_lm,
)
from lunar_od.measurements import PassGeometry, compute_position_residuals_analytic

_MU = 4902.800066e9


class GaussNewtonDecrementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(27)
        self.h = rng.normal(size=(40, 5)) * np.array([1e3, 1.0, 1e-2, 5.0, 1e2])
        self.r = rng.normal(size=40)
        self.w = rng.uniform(0.5, 2.0, size=40)

    def _reference(self, h, r, w, prior=None, prior_rhs=None):
        n = h.T @ (w[:, None] * h)
        g = h.T @ (w * r)
        if prior is not None:
            n = n + prior
            g = g + prior_rhs
        return math.sqrt(float(g @ np.linalg.solve(n, g)))

    def test_equals_newton_decrement_data_only(self):
        got = _gauss_newton_decrement(self.h, self.r, self.w, np.eye(5), None, None)
        self.assertAlmostEqual(got, self._reference(self.h, self.r, self.w), delta=1e-9 * got)

    def test_equals_newton_decrement_with_prior(self):
        prior = np.diag([1e-6, 2.0, 1e3, 0.1, 1e-4])
        prior_rhs = prior @ np.array([0.3, -0.1, 1e-3, 0.2, 5.0])
        got = _gauss_newton_decrement(self.h, self.r, self.w, np.eye(5), prior, prior_rhs)
        ref = self._reference(self.h, self.r, self.w, prior, prior_rhs)
        self.assertAlmostEqual(got, ref, delta=1e-9 * ref)

    def test_invariant_to_bookkeeping_scale(self):
        base = _gauss_newton_decrement(self.h, self.r, self.w, np.eye(5), None, None)
        scaled = _gauss_newton_decrement(
            self.h, self.r, self.w, np.diag([1e6, 1e-3, 1e2, 1.0, 1e-2]), None, None)
        self.assertAlmostEqual(base, scaled, delta=1e-9 * base)

    def test_zero_gradient_gives_zero(self):
        # Residual orthogonal to the column space: stationary, decrement 0.
        q, _ = np.linalg.qr(np.sqrt(self.w)[:, None] * self.h, mode="complete")
        r_perp = q[:, 5] / np.sqrt(self.w)
        got = _gauss_newton_decrement(self.h, r_perp, self.w, np.eye(5), None, None)
        self.assertLess(got, 1e-10)

    def test_rank_deficient_direction_contributes_nothing(self):
        h = np.hstack([self.h, self.h[:, :1]])  # duplicated column
        got = _gauss_newton_decrement(h, self.r, self.w, np.eye(6), None, None)
        ref = self._reference(self.h, self.r, self.w)
        self.assertTrue(np.isfinite(got))
        self.assertAlmostEqual(got, ref, delta=1e-8 * ref)

    def test_non_finite_design_is_never_certified(self):
        h = self.h.copy()
        h[0, 0] = np.nan
        status, detail, dec = _damping_exhaustion_status(h, self.r, self.w, np.eye(5), None, None)
        self.assertEqual((status, detail), ("DampingLimit", "LAMBDA_HARD_LIMIT"))
        self.assertFalse(np.isfinite(dec))

    def test_threshold_partitions_the_status(self):
        # Scale the gradient so the decrement sits just either side of the tolerance.
        unit = _gauss_newton_decrement(self.h, self.r, self.w, np.eye(5), None, None)
        for factor, expected in ((0.99, "Converged"), (1.01, "DampingLimit")):
            r = self.r * factor * STATIONARITY_DECREMENT_TOL / unit
            status, detail, dec = _damping_exhaustion_status(
                self.h, r, self.w, np.eye(5), None, None)
            self.assertEqual(status, expected, dec)
            self.assertEqual(
                detail,
                "STATIONARY_AT_DAMPING_LIMIT" if expected == "Converged" else "LAMBDA_HARD_LIMIT",
            )


def _station(lat, lon, alt):
    return Station(name=f"S {lat} {lon}", lat_deg=lat, lon_deg=lon, alt_m=alt,
                   color_rgb=(0.0, 0.0, 0.0), sigma_range_m=1.0, sigma_angle_rad=1e-5)


def _position_problem():
    r0 = 1737.4e3 + 100e3
    x_true0 = np.array([r0, 30e3, -20e3, -15.0, math.sqrt(_MU / r0), 4.0])
    t = np.arange(0.0, 1201.0, 60.0)
    gep = lambda tt: np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(tt)), 1))
    gsp = lambda tt: np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(tt)), 1))
    truth = propagate_augmented_state(
        t, np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")]),
        _MU, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13)[:, :6]
    stations = (_station(0.0, 0.0, 0.0), _station(0.0, 90.0, 0.0), _station(45.0, -30.0, 500.0))
    geo = PassGeometry(
        t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0),
        stations=stations, measurement_type="position")
    rows = [[ti, 0.0, 0.0, 0.0, sid, k] for k, ti in enumerate(t, start=1)
            for sid in range(1, len(stations) + 1)]
    obs = np.asarray(rows, dtype=float)
    _, h_meas, _ = compute_position_residuals_analytic(truth, obs, geo)
    obs[:, 1:4] = h_meas
    return t, obs, x_true0, geo, gep, gsp


def _solve_position(x0, **kw):
    t, obs, _x, geo, gep, gsp = _position_problem()
    opts = dict(max_iter=60, rtol=1e-12, atol=1e-13)
    opts.update(kw)
    return estimate_position_bls_lm(t, obs, x0, geo, _MU, 0.0, 0.0, gep, gsp, **opts)


class PositionBlsStatusContractTests(unittest.TestCase):
    """The rule is general: the position BLS-LM path obeys the same contract."""

    def test_descent_to_the_noise_floor_is_certified_at_callers_tolerance(self):
        """The D27 regime: with the step-norm and cost-stability tests disabled
        the solve descends until no step beats the objective's integration
        noise and lambda exhausts -- at the minimiser, from a 150 m start."""
        x_true0 = _position_problem()[2]
        x_start = x_true0 + np.array([120.0, -90.0, 55.0, 0.05, -0.025, 0.018])
        calls = []
        original = estimators.propagate_augmented_state

        def traced(*args, **kwargs):
            calls.append((kwargs.get("rtol"), kwargs.get("atol")))
            return original(*args, **kwargs)

        estimators.propagate_augmented_state = traced
        try:
            x_hat, status, stats = _solve_position(
                x_start, max_iter=200, tol_step_norm=0.0, tol_cost_stability=0.0)
        finally:
            estimators.propagate_augmented_state = original

        np.testing.assert_allclose(x_hat, x_true0, rtol=0, atol=1e-6)
        self.assertEqual(status, "Converged")
        self.assertEqual(stats.termination_detail, "STATIONARY_AT_DAMPING_LIMIT")
        self.assertLessEqual(stats.gauss_newton_decrement, STATIONARITY_DECREMENT_TOL)
        # Damping exhausts while _adaptive_tol still loosens the pass (frac < 0.4
        # of 200 iterations -> rtol 1e-8), so the certificate must have
        # re-linearised at the caller's tolerance: the last propagation is 1e-12.
        self.assertLess(stats.iterations, 0.4 * 200)
        self.assertEqual(calls[-1], (1e-12, 1e-13))

    def test_genuine_failure_keeps_damping_limit(self):
        x_true0 = _position_problem()[2]
        original = estimators._lm_step

        def always_bad_step(h_initial, *args, **kwargs):
            step = np.zeros(h_initial.shape[1])
            step[0] = 5.0e3
            return step, 1.0e3, h_initial.shape[1], False

        estimators._lm_step = always_bad_step
        try:
            x_start = x_true0 + np.array([120.0, -90.0, 55.0, 0.05, -0.025, 0.018])
            x_hat, status, stats = _solve_position(x_start)
        finally:
            estimators._lm_step = original

        np.testing.assert_array_equal(x_hat, x_start)
        self.assertEqual(status, "DampingLimit")
        self.assertEqual(stats.termination_detail, "LAMBDA_HARD_LIMIT")
        self.assertGreater(stats.gauss_newton_decrement, 1e3 * STATIONARITY_DECREMENT_TOL)

    def test_ordinary_convergence_is_unchanged(self):
        x_true0 = _position_problem()[2]
        x_guess = x_true0 + np.array([120.0, -90.0, 55.0, 0.05, -0.025, 0.018])
        _x, status, stats = _solve_position(x_guess, tol_step_norm=1.0)
        self.assertEqual(status, "Converged")
        self.assertEqual(stats.termination_detail, "")  # step-norm path: no detail, as before
        self.assertTrue(math.isnan(stats.gauss_newton_decrement))


class DiagnosticsClassificationTests(unittest.TestCase):
    @staticmethod
    def _stats(detail):
        return EstimatorStats(iterations=10, final_cost=1.0, position_step_norm_m=0.0,
                              velocity_step_norm_mps=0.0, condition_number=1e3, rank=6,
                              termination_detail=detail)

    def test_three_outcomes_are_distinguished(self):
        stationary = analyze_convergence("Converged", stats=self._stats("STATIONARY_AT_DAMPING_LIMIT"),
                                         expected_rank=6)
        step = analyze_convergence("Converged", stats=self._stats("STEP_NORM"), expected_rank=6)
        failure = analyze_convergence("DampingLimit", stats=self._stats("LAMBDA_HARD_LIMIT"),
                                      expected_rank=6)

        self.assertTrue(stationary.converged)
        self.assertTrue(stationary.converged_by_stationarity)
        self.assertFalse(stationary.converged_by_step_norm)
        self.assertEqual(stationary.category, "converged_stationary")

        self.assertTrue(step.converged)
        self.assertTrue(step.converged_by_step_norm)
        self.assertFalse(step.converged_by_stationarity)
        self.assertEqual(step.category, "converged_step")

        self.assertFalse(failure.converged)
        self.assertTrue(failure.damping_limit_reached)
        self.assertEqual(failure.category, "damping_limit")


if __name__ == "__main__":
    unittest.main()
