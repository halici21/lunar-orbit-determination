"""Phase 17-HARDEN-D30: rescore-on-reject acceptance in two-way range BLS-LM.

At the caller's tolerance the scored cost carries integration noise, which used to reject damped steps
whose true gain was smaller than that noise. A rejected candidate is now rescored, with the incumbent, at
a tighter tolerance before lambda is raised. The tests below make the noise explicit: the loose
acceptance propagation is perturbed for every state except the start, the tight one is exact, and the
LM step is scripted. Same short central-body pass as tests/test_estimator_lm_consistency.py.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_estimator_lm_consistency import _MU_MOON, _SKIP_REASON, _SPICE_OK, _observations

_OFFSET = np.array([3.0, -2.0, 1.0, 0.0, 0.0, 0.0])
_LOOSE_RTOL = 1e-11
_NOISE_M = 5.0


def setUpModule():  # noqa: N802
    if not _SPICE_OK:
        raise unittest.SkipTest(_SKIP_REASON)
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class AcceptanceRescoreTests(unittest.TestCase):
    def _run(self, step, **kwargs):
        import lunar_od.estimators as estimators

        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        x_start = x_true0 + _OFFSET
        calls = []
        original_step, original_prop = estimators._lm_step, estimators.propagate_state

        def scripted(h_initial, *args, **kw):
            return np.asarray(step, dtype=float).copy(), 1.0e3, h_initial.shape[1], False

        def noisy(t, x, *args, **kw):
            hist = np.array(original_prop(t, x, *args, **kw), dtype=float, copy=True)
            calls.append(kw.get("rtol"))
            if kw.get("rtol") == _LOOSE_RTOL and not np.array_equal(np.asarray(x, float), x_start):
                hist[:, :3] += _NOISE_M  # "integration noise" seen only at the caller's tolerance
            return hist

        estimators._lm_step, estimators.propagate_state = scripted, noisy
        try:
            x_hat, stop, stats = estimators.estimate_two_way_range_bls_lm(
                t_pass, obs, x_start, pass_geo, _MU_MOON, 0.0, 0.0, gep, gsp,
                max_iter=1, rtol=_LOOSE_RTOL, atol=1e-12, return_posterior=False, **kwargs,
            )
        finally:
            estimators._lm_step, estimators.propagate_state = original_step, original_prop
        return x_start, x_hat, stats, calls

    def test_noise_rejected_descent_is_accepted_by_the_tight_rescore(self):
        step = -0.5 * _OFFSET
        x_start, x_hat, stats, calls = self._run(step)
        np.testing.assert_array_equal(x_hat, x_start + step)
        self.assertEqual(stats.acceptance_rescore_accepts, 1)
        np.testing.assert_allclose(calls, [1e-11, 1e-11, 1e-13, 1e-13], rtol=1e-12)  # then both tight

    def test_switch_off_keeps_the_historical_rejection(self):
        x_start, x_hat, stats, calls = self._run(-0.5 * _OFFSET, acceptance_rescore_tolerance_factor=None)
        np.testing.assert_array_equal(x_hat, x_start)
        self.assertEqual(stats.acceptance_rescore_accepts, 0)
        self.assertEqual(calls, [1e-11, 1e-11])

    def test_a_genuinely_worse_step_stays_rejected_without_a_rescore(self):
        """A step predicted to raise the objective is never rescored."""
        x_start, x_hat, stats, calls = self._run(2.0 * _OFFSET)
        np.testing.assert_array_equal(x_hat, x_start)
        self.assertEqual(stats.acceptance_rescore_accepts, 0)
        self.assertEqual(calls, [1e-11, 1e-11])

    def test_a_step_below_the_certification_gain_is_not_rescored(self):
        """Amendment A1: predicted gain < 0.01 chi2 (a 0.1 sigma move) keeps the historical rejection."""
        x_start, x_hat, stats, calls = self._run(-1e-7 * _OFFSET)
        np.testing.assert_array_equal(x_hat, x_start)
        self.assertEqual(stats.acceptance_rescore_accepts, 0)
        self.assertEqual(calls, [1e-11, 1e-11])

    def test_factor_must_be_in_the_open_unit_interval(self):
        t_pass, gep, gsp, x_true0, obs, pass_geo = _observations()
        from lunar_od.estimators import estimate_two_way_range_bls_lm

        for bad in (0.0, 1.0, -1e-2, float("nan")):
            with self.assertRaises(ValueError):
                estimate_two_way_range_bls_lm(
                    t_pass, obs, x_true0, pass_geo, _MU_MOON, 0.0, 0.0, gep, gsp, max_iter=1,
                    acceptance_rescore_tolerance_factor=bad,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
