"""Phase 17-R1O-R - permanent protection for the STM storage-order contract.

`lunar_od.dynamics` packs the 6x6 state-transition matrix COLUMN-MAJOR into
columns [6:42] of its augmented history (`phi_dot.reshape(-1, order="F")`).
Any consumer that unflattens that block with NumPy's default C order silently
gets Phi^T.  Phi is not symmetric, so this is a real numerical defect, and it
is invisible to every test that only checks Phi at t=0 (where Phi = I is
symmetric) or that only checks the K column (S_K is a plain 6-vector and needs
no reshape at all).

Phase 17-R1O published an entire cross-observable ranking built on Phi^T
because of exactly this.  These tests are behavioural, not source-text
matching: they assert the defining property of Phi against independently
re-propagated finite differences, so they fail if C-order unpacking is
reintroduced anywhere along the chain they cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_od.constants import AU_M, J2_MOON_UNNORMALIZED, MU_MOON_M3S2
from lunar_od.dynamics import propagate_state, propagate_state_with_k_sensitivity
from lunar_od.srp import SRPOptions

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from phase17_r1o_core import chain_to_augmented_columns  # noqa: E402

STATE0 = np.array([1.8377e6, 0.0, 2.0e5, 0.0, 1.5337e3, 4.0e2])
K_SRP = 0.01
RTOL, ATOL = 1e-12, 1e-13
DURATION_S = 900.0


def _sun(_t):
    u = np.array([-1.0, 0.05, 0.02])
    return u / np.linalg.norm(u) * AU_M


def _earth(_t):
    return np.array([3.844e8, 0.0, 0.0])


def _propagate48(t_grid, x0):
    return propagate_state_with_k_sensitivity(
        t_grid, x0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        srp=SRPOptions(k_srp_m2_per_kg=K_SRP), rtol=RTOL, atol=ATOL,
        j2_moon=J2_MOON_UNNORMALIZED)


def _propagate6(t_grid, x0):
    """Independent truth: the 6-state propagator carries no variational block."""
    return propagate_state(
        t_grid, x0, MU_MOON_M3S2, 0.0, 0.0, _earth, _sun,
        srp=SRPOptions(k_srp_m2_per_kg=K_SRP), rtol=RTOL, atol=ATOL,
        j2_moon=J2_MOON_UNNORMALIZED)


@pytest.fixture(scope="module")
def arc():
    t_grid = np.array([0.0, DURATION_S * 0.5, DURATION_S])
    return t_grid, _propagate48(t_grid, STATE0)


def _phi_f(nom48, i=-1):
    return np.asarray(nom48[i, 6:42], dtype=float).reshape((6, 6), order="F")


def _phi_c(nom48, i=-1):
    return np.asarray(nom48[i, 6:42], dtype=float).reshape(6, 6)


def test_propagated_phi_is_not_symmetric(arc):
    """Guards the guards: a symmetric Phi would make every test below vacuous."""
    _, nom48 = arc
    phi = _phi_f(nom48)
    assert not np.allclose(phi, phi.T, rtol=1e-6, atol=1e-9)


def test_column_major_unpacking_matches_finite_difference_truth(arc):
    """Phi is DEFINED by dx(t) = Phi dx0.  Measure dr(t)/dx0 independently."""
    t_grid, nom48 = arc
    phi_f, phi_c = _phi_f(nom48), _phi_c(nom48)

    for j, step in ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1e-3), (4, 1e-3), (5, 1e-3)):
        xp, xm = STATE0.copy(), STATE0.copy()
        xp[j] += step
        xm[j] -= step
        fd = (_propagate6(t_grid, xp)[-1, :3]
              - _propagate6(t_grid, xm)[-1, :3]) / (2.0 * step)

        scale = max(float(np.max(np.abs(fd))), 1e-300)
        err_f = float(np.max(np.abs(phi_f[:3, j] - fd))) / scale
        err_c = float(np.max(np.abs(phi_c[:3, j] - fd))) / scale

        assert err_f < 1e-6, (
            f"order='F' column {j} disagrees with re-propagated truth "
            f"(rel {err_f:.3e}); the dynamics STM packing contract changed")
        assert err_c > 1e3 * max(err_f, 1e-12), (
            f"C-order column {j} is NOT decisively wrong (F {err_f:.3e} vs C "
            f"{err_c:.3e}); this test can no longer detect the defect")


def test_chain_helper_uses_the_true_phi_not_its_transpose(arc):
    """The R1O analysis helper must reproduce dg/dr @ Phi[:3, :], not Phi^T."""
    _, nom48 = arc
    phi = _phi_f(nom48)
    dg_dr = np.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]])

    h_x0, _ = chain_to_augmented_columns(dg_dr, nom48[-1])

    assert np.allclose(h_x0, dg_dr @ phi[:3, :], rtol=1e-12, atol=1e-12)
    assert not np.allclose(h_x0, dg_dr @ phi.T[:3, :], rtol=1e-6, atol=1e-9)


def test_chain_helper_state_columns_match_finite_difference_truth(arc):
    """End-to-end: the helper's dg/dx0 against re-propagated trajectories.

    This is the assertion that would have caught the published R1O defect.
    """
    t_grid, nom48 = arc
    dg_dr = np.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]])
    h_x0, _ = chain_to_augmented_columns(dg_dr, nom48[-1])

    for j, step in ((0, 1.0), (3, 1e-3)):
        xp, xm = STATE0.copy(), STATE0.copy()
        xp[j] += step
        xm[j] -= step
        dr = (_propagate6(t_grid, xp)[-1, :3]
              - _propagate6(t_grid, xm)[-1, :3]) / (2.0 * step)
        expected = dg_dr @ dr
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        err = float(np.max(np.abs(h_x0[:, j] - expected))) / scale
        assert err < 1e-6, f"helper column {j} rel error {err:.3e}"


def test_k_column_is_independent_of_stm_unpacking_order(arc):
    """S_K is a 6-vector: no reshape, so no storage-order exposure.

    This is why the defect corrupted R1O's state basis while leaving every K
    column it published numerically correct.
    """
    _, nom48 = arc
    dg_dr = np.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]])
    _, h_k = chain_to_augmented_columns(dg_dr, nom48[-1])

    row = np.array(nom48[-1], dtype=float)
    row[6:42] = row[6:42].reshape((6, 6), order="F").T.reshape(-1, order="F")
    _, h_k_transposed_phi = chain_to_augmented_columns(dg_dr, row)

    assert np.array_equal(h_k, h_k_transposed_phi)
    assert np.array_equal(h_k, dg_dr @ np.asarray(nom48[-1][42:48])[:3])
