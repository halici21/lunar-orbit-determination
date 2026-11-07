"""WP4.1 (EST-01/02/03): SRIF acceptance test, line search and stationarity.

Before WP4.1 the SRIF scored the incumbent on the 42-state augmented propagation
and the candidate on the 6-state one, and a rejected step reset to the incumbent
and recomputed the identical step.  Near the minimum every candidate lost to
integration noise, so a noise-free solve that had already recovered the truth to
micrometres ran to ``max_iter`` (16/16 probe cases, outputs/wp41_probe_before.txt
in the WP4.1 worktree).  Oracle: the noise-free truth.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_od import (
    PassGeometry,
    estimate_position_srif,
    estimate_range_rate_srif,
    propagate_augmented_state,
)
from lunar_od.estimators import SRIF_MAX_STEP_HALVINGS, _srif_backtrack
from tests.test_estimators import (
    _build_clean_position_observations,
    _build_clean_rr_observations,
    _synthetic_station,
)

MU = 4902.800066e9
R0 = 1737.4e3 + 100e3
X_TRUE = np.array([R0, 30e3, -20e3, -15.0, math.sqrt(MU / R0), 4.0])
OFFSET = np.array([5000.0, -4000.0, 3000.0, 2.0, -1.5, 1.0])
SOLVERS = {"position": estimate_position_srif, "range_rate": estimate_range_rate_srif}


def _earth(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _arc(kind: str):
    t = np.arange(0.0, 901.0, 60.0)
    truth = propagate_augmented_state(
        t, np.concatenate([X_TRUE, np.eye(6).reshape(-1, order="F")]),
        MU, 0.0, 0.0, _earth, _sun, rtol=1e-13, atol=1e-13)[:, :6]
    rr = kind == "range_rate"
    stations = tuple(_synthetic_station(*p, include_rr=rr)
                     for p in [(0, 0, 0), (0, 90, 0), (45, -30, 500), (-35, 150, 600)])
    geo = PassGeometry(
        t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None], t.size, axis=0),
        stations=stations, measurement_type=kind)
    build = _build_clean_rr_observations if rr else _build_clean_position_observations
    return t, build(t, truth, geo), geo


def _solve(kind, guess, **kwargs):
    t, obs, geo = _arc(kind)
    return SOLVERS[kind](t, obs, guess, geo, MU, 0.0, 0.0, _earth, _sun, max_iter=40, **kwargs)


@pytest.mark.parametrize("kind", sorted(SOLVERS))
@pytest.mark.parametrize("rtol, atol", [(1e-10, 1e-11), (1e-12, 1e-13)])
def test_noise_free_srif_converges_instead_of_running_to_max_iter(kind, rtol, atol):
    x_est, reason, stats = _solve(kind, X_TRUE + OFFSET, rtol=rtol, atol=atol)
    assert reason in {"Converged", "J-Stab"}, (reason, stats.termination_detail)
    assert stats.iterations <= 15
    assert np.linalg.norm(x_est[:3] - X_TRUE[:3]) < 1e-3
    assert np.linalg.norm(x_est[3:] - X_TRUE[3:]) < 1e-6


@pytest.mark.parametrize("kind", sorted(SOLVERS))
def test_srif_solution_is_a_fixed_point(kind):
    """Restarted at its own answer, the SRIF must certify it, not wander or run out."""
    x_est, _, _ = _solve(kind, X_TRUE + OFFSET, rtol=1e-12, atol=1e-13)
    x_again, reason, stats = _solve(kind, x_est, rtol=1e-12, atol=1e-13)
    assert reason in {"Converged", "J-Stab"}, (reason, stats.termination_detail)
    # F2 WP2.7 G12 (exploratory, amendment log): the bound was 3, which held only for the bit-exact
    # restart. At the optimum, noise-level steps can win the line search, mostly at the loosened
    # tolerance of the first iterations; restarts moved by 0-3 ulp take up to 8
    # (studies/phase17/results/wp2_7_g12_fixed_point_sweep.json). Before WP4.1: max_iter (40).
    assert stats.iterations <= 10
    assert np.linalg.norm(x_again[:3] - x_est[:3]) < 1e-4


def test_backtrack_halves_along_the_step_and_reports_exhaustion():
    step = np.array([4.0, -2.0])
    scored = []

    def score(trial):
        scored.append(trial.copy())
        return 0.0 if np.linalg.norm(trial) < 1.0 else 2.0

    accepted, cost, halvings = _srif_backtrack(score, step, current_cost=1.0)
    assert halvings == 3 and cost == 0.0
    np.testing.assert_array_equal(accepted, step / 8.0)
    for k, trial in enumerate(scored):
        np.testing.assert_array_equal(trial, step / 2.0**k)

    scored.clear()
    accepted, cost, halvings = _srif_backtrack(lambda s: scored.append(s) or 5.0, step, current_cost=1.0)
    assert accepted is None and math.isinf(cost)
    assert len(scored) == SRIF_MAX_STEP_HALVINGS + 1


# --- EST-02: the two-way MAP step is scored on the MAP objective ---------------------------

def _two_way_fixture():
    pytest.importorskip("spiceypy")
    from tests import test_harden_d29_srif_k_prior_contract as D29

    try:
        D29.setup_module(None)
    except FileNotFoundError as exc:
        pytest.skip(f"SPICE kernels unavailable: {exc}")
    return D29, D29._build_fixture()


def test_two_way_srif_with_tight_state_prior_is_map_stationary():
    """Tight explicit state prior centred 30 m / 3 cm/s off the truth: the returned point must be
    stationary for J = data + state prior + K prior (Gauss-Newton step <= 1e-2 posterior sigma).
    Before WP4.1 the step solved the MAP problem but was accepted only if the DATA cost fell."""
    D29, fx = _two_way_fixture()
    p0 = np.diag([10.0**2] * 3 + [0.01**2] * 3)
    x_start = fx["x_true0"] + D29.DX
    x, reason, st = D29._srif_k(fx, x_start, prior_covariance=p0)
    p0_inv = np.linalg.inv(p0)
    cov, _, grad = D29._declared(fx, np.asarray(x), st.k_srp_estimate, state_prior_inv=p0_inv)
    grad[:6] += p0_inv @ (x_start - np.asarray(x))
    assert reason in {"Converged", "J-Stab"}, (reason, st.termination_detail)
    assert np.max(np.abs(cov @ grad) / D29._sig(cov)) <= 1e-2
