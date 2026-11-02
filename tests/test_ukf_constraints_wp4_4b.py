"""WP4.4b (EST-13): frozen states as Schmidt-Kalman consider states; regularisation once.

Before WP4.4b a frozen state had its whole row and column of P reset to the predicted
values after each update, which discards the update of the cross-covariance P_xc that
a consider (Schmidt-Kalman) state keeps; and the regularisation pseudo-measurement
towards the initial mean was re-applied after EVERY update, so its prior counted N
times.  Oracles: the Joseph form with the consider rows of the gain zeroed (exact for a
linear-Gaussian update), and the information count of a state no measurement sees.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import lunar_od.filters as F
from lunar_od.filters import UKFState, UnscentedTransformConfig


def _linear_update(predicted, h, r, z):
    s = h @ predicted.p @ h.T + r
    k = predicted.p @ h.T @ np.linalg.inv(s)
    return UKFState(predicted.x + k @ (z - h @ predicted.x), predicted.p - k @ s @ k.T), k


def test_frozen_state_is_a_schmidt_consider_state():
    rng = np.random.default_rng(13)
    a = rng.standard_normal((4, 4))
    predicted = UKFState(rng.standard_normal(4), a @ a.T + np.eye(4))
    h = rng.standard_normal((2, 4))
    r = np.diag([0.3, 0.2])
    z = rng.standard_normal(2)
    updated, k = _linear_update(predicted, h, r, z)
    consider = (3,)
    constrained = F._apply_state_constraints(updated, predicted, predicted.x, consider, {},
                                             UnscentedTransformConfig().jitter)
    k_s = k.copy()
    k_s[list(consider), :] = 0.0
    i_kh = np.eye(4) - k_s @ h
    joseph = i_kh @ predicted.p @ i_kh.T + k_s @ r @ k_s.T
    np.testing.assert_allclose(constrained.p, joseph, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(constrained.x, predicted.x + k_s @ (z - h @ predicted.x), rtol=1e-12)


def test_regularisation_prior_is_applied_once():
    """In a real filter run the regularisation pseudo-measurement reaches the constraint
    step on the first update only; afterwards only frozen states are handled."""
    from tests.test_srif_acceptance_wp4_1 import MU, OFFSET, X_TRUE, _arc, _earth, _sun

    t, obs, geo = _arc("position")
    calls = []
    real = F._apply_state_constraints

    def spy(updated, predicted, prior_mean, frozen, regularization, jitter):
        calls.append(dict(regularization))
        return real(updated, predicted, prior_mean, frozen, regularization, jitter)

    with mock.patch.object(F, "_apply_state_constraints", spy):
        F.run_lunar_ukf(t, obs, X_TRUE + OFFSET, np.diag([1e4] * 3 + [1.0] * 3), geo, MU, 0.0, 0.0,
                        _earth, _sun, regularization_std_by_state={5: 2.0},
                        config=UnscentedTransformConfig(alpha=0.35), rtol=1e-12, atol=1e-13)
    assert obs.shape[0] > 1 and calls and calls[0] == {5: 2.0}
    assert not any(calls[1:])          # was: re-applied after every one of the updates
