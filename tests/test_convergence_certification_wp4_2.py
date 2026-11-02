"""WP4.2 (EST-04/05): convergence is declared only at the caller's tolerance and only
when the unit-free Gauss-Newton decrement certifies it.

Before WP4.2, BLS-LM declared ``Converged`` on a point last linearised at the
early-iteration rtol 1e-8 when the caller asked for 1e-12 (outputs/wp42_probe_before.txt
in the WP4.2 worktree), and every batch loop stopped on a relative cost change or a
mixed-unit step norm alone.
"""
from __future__ import annotations

import pytest

import lunar_od.estimators as E
from tests.test_srif_acceptance_wp4_1 import MU, OFFSET, X_TRUE, _arc, _earth, _sun

ESTIMATORS = {
    ("position", "bls"): E.estimate_position_bls_lm,
    ("position", "srif"): E.estimate_position_srif,
    ("range_rate", "bls"): E.estimate_range_rate_bls_lm,
    ("range_rate", "srif"): E.estimate_range_rate_srif,
}


def _run(key, **kwargs):
    t, obs, geo = _arc(key[0])
    return ESTIMATORS[key](t, obs, X_TRUE + OFFSET, geo, MU, 0.0, 0.0, _earth, _sun,
                           max_iter=40, rtol=1e-12, atol=1e-13, **kwargs)


@pytest.mark.parametrize("key", sorted(ESTIMATORS))
def test_convergence_is_declared_on_a_caller_tolerance_linearisation(key, monkeypatch):
    tolerances = []
    real = E.propagate_augmented_state

    def spy(*args, **kwargs):
        tolerances.append((kwargs.get("rtol"), kwargs.get("atol")))
        return real(*args, **kwargs)

    monkeypatch.setattr(E, "propagate_augmented_state", spy)
    _, reason, stats = _run(key)
    assert reason in {"Converged", "J-Stab"}, (reason, stats.termination_detail)
    assert tolerances[-1] == (1e-12, 1e-13)
    assert stats.gauss_newton_decrement <= E.STATIONARITY_DECREMENT_TOL


@pytest.mark.parametrize("key", sorted(ESTIMATORS))
def test_uncertified_point_is_never_labelled_converged(key, monkeypatch):
    """A decrement that never certifies (as far from a minimiser) must leave no Converged/J-Stab
    label, however small the cost change or step: the historical triggers only nominate."""
    monkeypatch.setattr(E, "_gauss_newton_decrement", lambda *args, **kwargs: 10.0)
    _, reason, stats = _run(key)
    assert reason not in {"Converged", "J-Stab"}, (reason, stats.termination_detail)
