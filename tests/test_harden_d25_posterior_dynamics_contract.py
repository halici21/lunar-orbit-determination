"""Phase 17-HARDEN-D25 - the posterior covariance is linearised on the solve's own dynamics.

Before D25 the posterior passes of the two-way-range estimators dropped part of the
dynamics contract the solve iterated with:

  - estimate_two_way_range_bls_lm, K path:      lunar harmonic field;
  - estimate_two_way_range_bls_lm, six-state:   harmonic field AND fixed SRP;
  - estimate_two_way_range_srif,   six-state:   fixed SRP.

The estimate was right (the solve used the full model); only the reported uncertainty was
linearised on a different force model (KP2 CANON k0: sigma_K +9 %, sigma_z +3 %).

Each test compares the estimator's RETURNED posterior with an independent rebuild at the
returned solution using the solve's dynamics, and checks that the historical construction
(the same rebuild without the dropped terms) is measurably different on the same fixture,
so a regression to it cannot pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("spiceypy")

import lunar_od.estimators as E  # noqa: E402
from lunar_od.dynamics import propagate_augmented_state  # noqa: E402
from lunar_od.gravity_model_loader import load_lunar_gravity_model  # noqa: E402
from lunar_od.project_paths import gravity_data_root  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402
from lunar_od.two_way_range import generate_two_way_range_measurements  # noqa: E402

from tests import test_k_srp_bls_solve_for as base  # noqa: E402

TOL = dict(rtol=1e-12, atol=1e-13)
K_TRUE = 0.01
GL1800F_TAB = gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"


def setup_module(_module):
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


@pytest.fixture(scope="module")
def fx():
    if not GL1800F_TAB.exists():
        pytest.skip("GL1800F coefficient file not available")
    return _build_fixture()


def _build_fixture():
    f = base._fixture(duration_s=12000.0, step_s=120.0)
    model = dataclasses.replace(load_lunar_gravity_model(str(GL1800F_TAB), nmax=8, mmax=8), frame="MOON_PA_DE440")
    harm = dict(harmonic_model=model, harmonic_epoch_et0=f["et0"])
    srp = SRPOptions(k_srp_m2_per_kg=K_TRUE)
    aug0 = np.concatenate([f["x_true0"], np.eye(6).reshape(-1, order="F")])
    hist = propagate_augmented_state(f["t_pass"], aug0, base._MU_MOON, 0.0, 0.0, f["gep"], f["gsp"], srp=srp,
                                     **harm, harmonic_stm_opt_in=True, **TOL)
    vis = np.ones((f["t_pass"].size, 1), dtype=bool)
    obs, pg = generate_two_way_range_measurements(f["t_pass"], hist[:, :6], (base._STATION,), vis, f["gep"],
                                                  f["gev"], f["et0"], noise=False)
    return dict(f, obs=np.asarray(obs, float), pg=pg, harm=harm, srp=srp)


def _args(fx):
    return (fx["t_pass"], fx["obs"])


def _dyn(fx):
    return (base._MU_MOON, 0.0, 0.0, fx["gep"], fx["gsp"])


def _six_state_information(fx, x, *, srp, harm):
    """Independent six-state rebuild: H^T W H + P0^-1 on the given dynamics."""
    kw = {} if srp is None else dict(srp=srp)
    if harm:
        kw.update(harm, harmonic_stm_opt_in=True)
    aug = propagate_augmented_state(fx["t_pass"], np.concatenate([x, np.eye(6).reshape(-1, order="F")]),
                                    *_dyn(fx), **TOL, **kw)
    _, h = E.two_way_range_nominal_and_initial_jacobian(fx["obs"], fx["pg"], aug)
    w = E._two_way_range_weight_diagonal(fx["obs"], fx["pg"])
    return h.T @ (w[:, None] * h) + np.linalg.inv(PRIOR)


def _k_path_covariance(fx, x, k, *, harm):
    """Independent K-path rebuild: column-normalised QR of the whitened 7-column design.

    With no explicit prior the estimator's posterior carries its default weak state prior
    (15 km, 5 m/s; none on K) - part of its declared posterior, so the rebuild includes it.
    """
    kw = dict(harm, harmonic_stm_opt_in=True) if harm else {}
    aug = E.propagate_state_with_k_sensitivity(fx["t_pass"], x, *_dyn(fx),
                                               srp=SRPOptions(k_srp_m2_per_kg=k), **TOL, **kw)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(fx["obs"], fx["pg"], aug)
    hk = E._two_way_range_k_srp_column(fx["obs"], aug, hx)
    a = np.hstack([hx, hk[:, None]]) * np.sqrt(E._two_way_range_weight_diagonal(fx["obs"], fx["pg"]))[:, None]
    prior6, *_ = E._prior_information_and_scale(6, E._TWO_WAY_RANGE_NO_BIAS_CFG,
                                                E._two_way_range_bias_prior_and_scale, None, None)
    prior_rows = np.zeros((6, 7))
    prior_rows[:, :6] = np.diag(np.sqrt(np.diag(prior6)))
    a = np.vstack([a, prior_rows])
    c = np.linalg.norm(a, axis=0)
    r_inv = np.linalg.inv(np.linalg.qr(a / c, mode="r"))
    return (r_inv @ r_inv.T) / np.outer(c, c)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _sigma_rel(a, b):
    return float(np.max(np.abs(np.sqrt(np.diag(a)) / np.sqrt(np.diag(b)) - 1)))


PRIOR = np.diag([1e4] * 3 + [1e-2] * 3)
SRIF_SRP_EFFECT_FLOOR = 1e-7      # measured 1.5e-6 in sigma: SRP is weak on a 3.3 h arc, but far above the 1e-17 agreement


def test_posterior_dynamics_kwargs_mirror_the_solve():
    assert E._posterior_dynamics_kwargs(None, None) == {}
    assert E._posterior_dynamics_kwargs(None, {}) == {}
    s = SRPOptions(k_srp_m2_per_kg=0.02)
    assert E._posterior_dynamics_kwargs(s, None) == {"srp": s}
    h = {"harmonic_model": object(), "harmonic_epoch_et0": 1.0}
    assert E._posterior_dynamics_kwargs(None, h) == {**h, "harmonic_stm_opt_in": True}


def test_k_path_posterior_is_linearised_on_the_harmonic_field(fx):
    x_hat, _, st = E.estimate_two_way_range_bls_lm(
        *_args(fx), fx["x_true0"], fx["pg"], *_dyn(fx), max_iter=3, **TOL, srp=fx["srp"],
        solve_for_k_srp=True, k_srp_initial=K_TRUE, return_posterior=True, **fx["harm"])
    cov = np.asarray(st.posterior_covariance)
    good = _k_path_covariance(fx, np.asarray(x_hat), st.k_srp_estimate, harm=fx["harm"])
    historical = _k_path_covariance(fx, np.asarray(x_hat), st.k_srp_estimate, harm=None)
    assert _sigma_rel(cov, good) <= 1e-8
    assert _sigma_rel(historical, good) > 1e-3             # the D25 construction is visibly different here
    # covariance-only repair: the estimate is identical with and without the posterior pass
    x2, _, st2 = E.estimate_two_way_range_bls_lm(
        *_args(fx), fx["x_true0"], fx["pg"], *_dyn(fx), max_iter=3, **TOL, srp=fx["srp"],
        solve_for_k_srp=True, k_srp_initial=K_TRUE, return_posterior=False, **fx["harm"])
    assert np.array_equal(x_hat, x2) and st.k_srp_estimate == st2.k_srp_estimate


def test_six_state_bls_posterior_carries_the_field_and_the_fixed_srp(fx):
    x_hat, _, st = E.estimate_two_way_range_bls_lm(
        *_args(fx), fx["x_true0"], fx["pg"], *_dyn(fx), max_iter=3, **TOL, srp=fx["srp"],
        prior_covariance=PRIOR, return_posterior=True, **fx["harm"])
    info = np.asarray(st.posterior_information)
    x_hat = np.asarray(x_hat)
    good = _six_state_information(fx, x_hat, srp=fx["srp"], harm=fx["harm"])
    historical = _six_state_information(fx, x_hat, srp=None, harm=None)
    assert _rel(info, good) <= 1e-10
    assert _sigma_rel(np.linalg.inv(historical), np.linalg.inv(good)) > 1e-3


def test_six_state_srif_posterior_carries_the_fixed_srp(fx):
    # SRIF has no harmonic option: its data must come from the same SRP-only dynamics
    aug0 = np.concatenate([fx["x_true0"], np.eye(6).reshape(-1, order="F")])
    hist = propagate_augmented_state(fx["t_pass"], aug0, *_dyn(fx), srp=fx["srp"], **TOL)
    vis = np.ones((fx["t_pass"].size, 1), dtype=bool)
    obs, pg = generate_two_way_range_measurements(fx["t_pass"], hist[:, :6], (base._STATION,), vis, fx["gep"],
                                                  fx["gev"], fx["et0"], noise=False)
    f2 = dict(fx, obs=np.asarray(obs, float), pg=pg)
    x_hat, _, st = E.estimate_two_way_range_srif(
        f2["t_pass"], f2["obs"], f2["x_true0"], pg, *_dyn(f2), max_iter=3, **TOL, srp=f2["srp"],
        prior_covariance=PRIOR, return_posterior=True)
    info = np.asarray(st.posterior_information)
    good = _six_state_information(f2, np.asarray(x_hat), srp=f2["srp"], harm=None)
    historical = _six_state_information(f2, np.asarray(x_hat), srp=None, harm=None)
    assert _rel(info, good) <= 1e-10
    assert _sigma_rel(np.linalg.inv(historical), np.linalg.inv(good)) > SRIF_SRP_EFFECT_FLOOR
