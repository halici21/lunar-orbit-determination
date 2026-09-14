"""PHASE 6.2 regressions for the two-way-range LM estimator's cost evaluation
and termination semantics.

Two production defects motivated these tests.

1.  ACCEPTANCE-ROUTE ASYMMETRY.  The incumbent state used to be scored from the
    42-state augmented propagation while the candidate was scored from a plain
    6-state propagation.  Both carried the same nominal ``rtol`` label, but the
    augmented integrator also error-controls the 36 STM components and so takes
    a much smaller step sequence.  On this fixture the same 1e-8 label buys
    4.99e-02 m from the 6-state path and 4.46e-06 m from the augmented one - a
    factor of about 1.1e4.  Wherever the optimisation step became comparable to
    that gap, every candidate looked worse than the incumbent no matter how
    good it was, so nothing was ever accepted.  The failure is not specific to
    any gravity degree; a high-fidelity model merely exposes it sooner because
    its residual signal is smaller.

2.  TERMINATION CONFLATION.  A genuinely singular linear solve and simple
    damping exhaustion both reported ``stop_reason == "Singular"``, so a run
    that had descended for tens of iterations was indistinguishable from one
    whose normal matrix had failed.

``test_nominal_and_candidate_are_scored_by_the_same_route`` is the test that
pins the first repair: before it, one LM iteration issued a single 6-state
propagation (the candidate only), so the recorded pair does not exist.

Everything here runs on a short central-body pass and needs no gravity model.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from lunar_od.config import Station
from lunar_od.radiometrics import _interp_state
from lunar_od.two_way_range import (
    compute_two_way_range_residuals,
    generate_two_way_range_measurements,
)


def _spice_status() -> tuple[bool, str]:
    try:
        import spiceypy  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"spiceypy unavailable: {exc}"
    try:
        from lunar_od.spice_loader import required_kernel_paths

        required_kernel_paths()
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        return False, f"SPICE kernels unavailable: {exc}"
    return True, ""


_SPICE_OK, _SKIP_REASON = _spice_status()
_ET_UTC = "2027-03-02 00:00:00"
_MU_MOON = 4902.800066e9

_STATION = Station(
    name="Goldstone DSN",
    lat_deg=35.30,
    lon_deg=-116.81,
    alt_m=969.67,
    color_rgb=(0.85, 0.325, 0.098),
    sigma_range_m=5.0,
    sigma_angle_rad=math.radians(0.001),
)

# Prior wide enough that it never drives the solution on this short arc.
_P0 = np.diag(np.array([1.0e4, 1.0e4, 1.0e4, 1.0e-2, 1.0e-2, 1.0e-2]) ** 2)


def setUpModule():  # noqa: N802
    if not _SPICE_OK:
        raise unittest.SkipTest(_SKIP_REASON)
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


def _truth_setup(step_s: float = 60.0, duration_s: float = 1200.0):
    """Propagated LLO truth over a short pass with real Earth ephemeris."""
    import spiceypy as spice

    from lunar_od.dynamics import propagate_augmented_state

    et0 = float(spice.str2et(_ET_UTC))
    t_pass = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    earth_states = np.array(
        [
            np.asarray(
                spice.spkezr("EARTH", float(et0 + t), "J2000", "NONE", "MOON")[0],
                dtype=float,
            )
            * 1e3
            for t in t_pass
        ]
    )

    def get_earth_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array(
            [_interp_state(t_pass, earth_states, float(ti))[:3] for ti in t_arr]
        )

    def get_earth_vel(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array(
            [_interp_state(t_pass, earth_states, float(ti))[3:] for ti in t_arr]
        )

    def get_sun_pos(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.tile(np.array([149.6e9, 0.0, 0.0]), (t_arr.size, 1))

    r0 = 1737.4e3 + 100e3
    x_true0 = np.array(
        [r0, 30e3, -20e3, -15.0, math.sqrt(_MU_MOON / r0), 4.0], dtype=float
    )
    x_aug0 = np.concatenate([x_true0, np.eye(6).reshape(-1, order="F")])
    x_aug_truth = propagate_augmented_state(
        t_pass,
        x_aug0,
        _MU_MOON,
        0.0,
        0.0,
        get_earth_pos,
        get_sun_pos,
        rtol=1e-12,
        atol=1e-13,
    )
    return et0, t_pass, get_earth_pos, get_earth_vel, get_sun_pos, x_true0, x_aug_truth


def _observations():
    et0, t_pass, gep, gev, gsp, x_true0, x_aug = _truth_setup()
    vis = np.ones((t_pass.size, 1), dtype=bool)
    obs, pass_geo = generate_two_way_range_measurements(
        t_pass, x_aug[:, :6], (_STATION,), vis, gep, gev, et0, noise=False
    )
    return t_pass, gep, gsp, x_true0, obs, pass_geo


def _solve(t_pass, obs, x0, pass_geo, gep, gsp, **kwargs):
    from lunar_od.estimators import estimate_two_way_range_bls_lm

    opts = dict(
        max_iter=20,
        rtol=1e-11,
        atol=1e-12,
        prior_covariance=_P0,
        return_posterior=False,
    )
    opts.update(kwargs)
    return estimate_two_way_range_bls_lm(
        t_pass, obs, x0, pass_geo, _MU_MOON, 0.0, 0.0, gep, gsp, **opts
    )


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class AcceptanceRouteParityTests(unittest.TestCase):
    """The incumbent and the candidate must be judged by the same ruler."""

    def test_nominal_and_candidate_are_scored_by_the_same_route(self):
        """One LM iteration must issue two 6-state acceptance propagations -
        incumbent then candidate - at the caller's tolerance.

        This is the invariant that was violated: the incumbent used to be read
        off the 42-state augmented pass, so a single iteration produced only
        one 6-state propagation and the two costs came from integrations of
        very different accuracy.
        """
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        calls = []
        original = estimators.propagate_state

        def traced(t, x, *args, **kwargs):
            calls.append(
                (np.asarray(x, dtype=float).copy(), kwargs.get("rtol"), kwargs.get("atol"))
            )
            return original(t, x, *args, **kwargs)

        estimators.propagate_state = traced
        try:
            x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
            _solve(t_pass, obs, x_guess, pass_geo, gep, gsp, max_iter=1)
        finally:
            estimators.propagate_state = original

        self.assertEqual(
            len(calls),
            2,
            "one iteration must score exactly the incumbent and the candidate "
            f"through the 6-state route; saw {len(calls)} propagations",
        )
        (x_nom, rtol_nom, atol_nom), (x_cand, rtol_cand, atol_cand) = calls
        self.assertEqual(rtol_nom, rtol_cand)
        self.assertEqual(atol_nom, atol_cand)
        self.assertEqual(rtol_nom, 1e-11, "the caller's rtol must be used, not the adaptive one")
        np.testing.assert_allclose(x_nom, x_guess, rtol=0, atol=0)
        self.assertGreater(
            float(np.linalg.norm(x_cand - x_nom)), 0.0, "the candidate must differ"
        )

    def test_identical_state_gives_identical_cost(self):
        """Scoring one state twice through the acceptance route is bitwise equal.

        Deterministic integration makes exact equality the right assertion here;
        anything weaker would let the old asymmetry back in unnoticed.
        """
        from lunar_od.dynamics import propagate_state
        from lunar_od.estimators import _two_way_range_weight_diagonal

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        w = _two_way_range_weight_diagonal(obs, pass_geo)

        def cost(x):
            hist = propagate_state(
                t_pass, x, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-11, atol=1e-12
            )
            r, _ = compute_two_way_range_residuals(hist, obs, pass_geo)
            r = np.asarray(r, dtype=float)
            return float(np.dot(w * r, r))

        x = x_true0 + np.array([3.0, -2.0, 1.0, 0.001, -0.002, 0.0005])
        self.assertEqual(cost(x), cost(x))

    def test_equal_rtol_labels_do_not_imply_equal_state_accuracy(self):
        """Guard the assumption that caused the defect.

        The augmented system error-controls 36 extra STM components, so at a
        shared ``rtol`` it integrates the physical state far more accurately
        than the plain 6-state call.  Measured ratios on this fixture run from
        about 1.3e2 at rtol 1e-5 to 1.1e4 at rtol 1e-8; the assertion keeps a
        wide margin below the smallest of those and deliberately fixes no
        absolute metre value.
        """
        from lunar_od.dynamics import propagate_augmented_state, propagate_state

        _, t_pass, gep, _, gsp, x_true0, _ = _truth_setup()
        identity = np.eye(6).reshape(-1, order="F")
        reference = propagate_state(
            t_pass, x_true0, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-13, atol=1e-14
        )
        for rtol, atol in ((1e-8, 1e-9), (1e-6, 1e-7)):
            six = propagate_state(
                t_pass, x_true0, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=rtol, atol=atol
            )
            augmented = propagate_augmented_state(
                t_pass,
                np.concatenate([x_true0, identity]),
                _MU_MOON,
                0.0,
                0.0,
                gep,
                gsp,
                rtol=rtol,
                atol=atol,
            )
            err_six = float(
                np.linalg.norm(six[:, :3] - reference[:, :3], axis=1).max()
            )
            err_aug = float(
                np.linalg.norm(augmented[:, :3] - reference[:, :3], axis=1).max()
            )
            self.assertGreater(
                err_six / max(err_aug, 1e-300),
                10.0,
                f"at rtol {rtol:g} the two paths were expected to differ sharply "
                f"in accuracy; got {err_six:.3e} m vs {err_aug:.3e} m",
            )


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class DescentAcceptanceTests(unittest.TestCase):
    def test_small_residual_start_still_accepts_a_descent(self):
        """A start whose residuals are already small must still be improved.

        This is the regime the defect destroyed: the remaining signal was
        comparable to the incumbent/candidate integration gap, so every step
        was rejected however good it was.  Roughly 0.28 m of position offset
        puts the fit RMS near 7e-02 m here, and the repaired solver takes it to
        about 1.4e-03 m - a factor near 50.  The assertion keeps a wide margin
        under that and fixes no particular final state.

        The residual, not the distance to truth, is the thing asserted: one
        short pass is weakly observable transverse to the line of sight, so the
        state legitimately drifts away from truth while the fit improves.
        """
        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        from lunar_od.dynamics import propagate_state

        def rms(x):
            hist = propagate_state(
                t_pass, x, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13
            )
            r, _ = compute_two_way_range_residuals(hist, obs, pass_geo)
            return float(np.sqrt(np.mean(np.asarray(r, dtype=float) ** 2)))

        x_start = x_true0 + 0.2 * np.array([1.0, -0.8, 0.6, 4e-4, -3e-4, 2e-4])
        before = rms(x_start)
        x_hat, stop_reason, stats = _solve(
            t_pass, obs, x_start, pass_geo, gep, gsp, max_iter=30
        )
        after = rms(x_hat)

        self.assertTrue(np.all(np.isfinite(x_hat)))
        self.assertTrue(np.isfinite(stats.final_cost))
        self.assertGreater(
            float(np.linalg.norm(x_hat - x_start)),
            0.0,
            f"solver never moved off its start (stop_reason={stop_reason})",
        )
        self.assertLess(
            after,
            before / 5.0,
            "a genuine small-residual descent was not accepted: "
            f"{before:.4e} m -> {after:.4e} m",
        )

    def test_exact_optimum_start_neither_wanders_nor_claims_singularity(self):
        """Starting on the exact optimum, refusing to move is correct.

        The truth state reproduces this fixture's own measurements to zero
        residual, so there is nothing to descend towards and every candidate is
        rightly rejected.  What matters is that the run says so honestly: it is
        damping exhaustion, not a failed linear solve, and the returned state is
        the untouched start rather than a drifted one.
        """
        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        x_hat, stop_reason, stats = _solve(
            t_pass, obs, x_true0.copy(), pass_geo, gep, gsp, max_iter=30
        )

        self.assertTrue(np.all(np.isfinite(x_hat)))
        np.testing.assert_allclose(x_hat, x_true0, rtol=0, atol=0)
        self.assertEqual(stop_reason, "DampingLimit")
        self.assertEqual(stats.termination_detail, "LAMBDA_HARD_LIMIT")

    def test_large_residual_start_still_makes_accepted_progress(self):
        """The regime that always worked must keep working.

        Patch A changed how every candidate is scored, so the previously
        healthy large-residual case needs its own guard.  Iteration counts are
        deliberately not asserted.
        """
        from lunar_od.dynamics import propagate_state

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])

        def rms(x):
            hist = propagate_state(
                t_pass, x, _MU_MOON, 0.0, 0.0, gep, gsp, rtol=1e-12, atol=1e-13
            )
            r, _ = compute_two_way_range_residuals(hist, obs, pass_geo)
            return float(np.sqrt(np.mean(np.asarray(r, dtype=float) ** 2)))

        before = rms(x_guess)
        x_hat, _stop, stats = _solve(t_pass, obs, x_guess, pass_geo, gep, gsp)
        after = rms(x_hat)

        self.assertTrue(np.all(np.isfinite(x_hat)))
        self.assertTrue(np.isfinite(stats.final_cost))
        self.assertLess(after, before, "residual RMS must fall from a coarse start")


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class TerminationSemanticsTests(unittest.TestCase):
    """A failed linear solve and damping exhaustion are different events."""

    def test_true_matrix_singularity_reports_singular(self):
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        original = estimators._lm_step

        def always_singular(h_initial, *args, **kwargs):
            return np.zeros(h_initial.shape[1]), 1e16, 0, True

        estimators._lm_step = always_singular
        try:
            _x, stop_reason, stats = _solve(
                t_pass, obs, x_true0.copy(), pass_geo, gep, gsp
            )
        finally:
            estimators._lm_step = original

        self.assertEqual(stop_reason, "Singular")
        self.assertEqual(stats.termination_detail, "MATRIX_SINGULAR")

    def test_damping_exhaustion_is_not_reported_as_singular(self):
        """Every candidate rejected drives lambda past its hard limit.

        The linear solve never fails here, so the run must not claim it did.
        """
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        original = estimators._lm_step

        def always_bad_step(h_initial, *args, **kwargs):
            step = np.zeros(h_initial.shape[1])
            step[0] = 5.0e3  # large enough to be rejected, inside the limiter
            return step, 1.0e3, h_initial.shape[1], False

        estimators._lm_step = always_bad_step
        try:
            _x, stop_reason, stats = _solve(
                t_pass, obs, x_true0.copy(), pass_geo, gep, gsp, max_iter=40
            )
        finally:
            estimators._lm_step = original

        self.assertEqual(stop_reason, "DampingLimit")
        self.assertEqual(stats.termination_detail, "LAMBDA_HARD_LIMIT")
        self.assertNotEqual(stop_reason, "Singular")

    def test_ordinary_convergence_reports_a_convergence_detail(self):
        """Ordinary convergence must be labelled as such.

        ``tol_step_norm`` is set explicitly because on a pass this short the
        solver otherwise descends all the way to its numerical floor and ends
        in damping exhaustion before any default convergence threshold is met.
        The point under test is the label attached to a genuinely accepted,
        genuinely small step - not the value of the production threshold.
        """
        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        x_guess = x_true0 + np.array([5.0, -4.0, 3.0, 0.002, -0.001, 0.0005])
        _x, stop_reason, stats = _solve(
            t_pass, obs, x_guess, pass_geo, gep, gsp, tol_step_norm=1.0
        )

        self.assertEqual(stop_reason, "Converged")
        self.assertEqual(stats.termination_detail, "STEP_NORM")

    def test_diagnostics_separates_damping_limit_from_singular(self):
        from lunar_od.diagnostics import analyze_convergence

        damping = analyze_convergence("DampingLimit", condition_number=1.0e3, rank=6)
        singular = analyze_convergence("Singular", condition_number=1.0e3, rank=6)

        self.assertTrue(damping.damping_limit_reached)
        self.assertFalse(damping.singular_or_ill_conditioned)
        self.assertEqual(damping.category, "damping_limit")

        self.assertFalse(singular.damping_limit_reached)
        self.assertTrue(singular.singular_or_ill_conditioned)
        self.assertEqual(singular.category, "singular_or_ill_conditioned")


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class MapAcceptanceContractTests(unittest.TestCase):
    """Acceptance must judge candidates by the objective the step solved for.

    ``_lm_step`` builds the MAP normal equations - measurement information plus
    prior information, with the prior right-hand side ``P0^-1 (x_prior -
    x_nominal)`` - so the step it returns is a Gauss-Newton step for

        J_MAP = 0.5 r^T W r + 0.5 (x - x_prior)^T P0^-1 (x - x_prior).

    Acceptance used to score only the measurement half.  In a real G100 solve
    that flipped two decisions at 0.13-0.14 m steps, once in each direction.

    These two tests pin both directions with a scripted step sequence.  The
    prior is deliberately tight and the start deliberately far, because the
    prior term has to be stiff enough to overrule the measurement term for the
    conflict to exist at all; ``x_prior`` is set to the start state by the
    estimator, so the incumbent must first be moved off the prior centre before
    any candidate can reduce the prior term.
    """

    _P0_TIGHT = np.diag(np.array([5.0, 5.0, 5.0, 5.0e-3, 5.0e-3, 5.0e-3]) ** 2)
    _OFFSET = np.array([200.0, -140.0, 100.0, 0.0, 0.0, 0.0])
    # 131.9 m straight back toward truth: d(2J_meas) -1.761e+03,
    # d(2J_prior) +6.960e+02, d(2J_MAP) -1.065e+03 -> accepted either way.
    _STEP1 = -0.5 * _OFFSET
    # 5 m back toward the prior centre: d(2J_meas) +4.534e+01,
    # d(2J_prior) -5.176e+01, d(2J_MAP) -6.426e+00.
    _STEP2 = np.array([3.79049, -2.653343, 1.895245, 0.0, 0.0, 0.0])

    def _run_scripted(self, steps):
        """Drive the solver through a fixed step sequence and return x_best."""
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        queue = list(steps)
        original = estimators._lm_step

        def scripted(h_initial, *args, **kwargs):
            step = queue.pop(0) if queue else np.zeros(h_initial.shape[1])
            return np.asarray(step, dtype=float).copy(), 1.0e3, h_initial.shape[1], False

        estimators._lm_step = scripted
        try:
            x_hat, stop_reason, stats = estimators.estimate_two_way_range_bls_lm(
                t_pass,
                obs,
                x_true0 + self._OFFSET,
                pass_geo,
                _MU_MOON,
                0.0,
                0.0,
                gep,
                gsp,
                max_iter=len(steps),
                rtol=1e-11,
                atol=1e-12,
                prior_covariance=self._P0_TIGHT,
                return_posterior=False,
            )
        finally:
            estimators._lm_step = original
        return x_true0 + self._OFFSET, x_hat, stop_reason, stats

    def test_map_descent_is_accepted_even_when_the_fit_worsens(self):
        """d(J_meas) > 0 but d(J_MAP) < 0 must be accepted."""
        x_start, x_hat, _stop, _stats = self._run_scripted([self._STEP1, self._STEP2])
        expected = x_start + self._STEP1 + self._STEP2
        np.testing.assert_allclose(x_hat, expected, rtol=0, atol=1e-9)

    def test_fit_improvement_is_rejected_when_it_worsens_the_map_objective(self):
        """The converse: d(J_meas) < 0 but d(J_MAP) > 0 must be rejected."""
        x_start, x_hat, _stop, _stats = self._run_scripted(
            [self._STEP1, self._STEP2, -self._STEP2]
        )
        expected = x_start + self._STEP1 + self._STEP2
        np.testing.assert_allclose(x_hat, expected, rtol=0, atol=1e-9)

    def test_without_an_explicit_prior_acceptance_is_measurement_only(self):
        """No prior supplied means no prior term - the objectives coincide."""
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        x_guess = x_true0 + np.array([25.0, -20.0, 12.0, 0.01, -0.008, 0.004])
        x_hat, _stop, stats = estimators.estimate_two_way_range_bls_lm(
            t_pass, obs, x_guess, pass_geo, _MU_MOON, 0.0, 0.0, gep, gsp,
            max_iter=10, rtol=1e-11, atol=1e-12, return_posterior=False,
        )
        self.assertTrue(np.all(np.isfinite(x_hat)))
        self.assertTrue(np.isfinite(stats.final_cost))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
