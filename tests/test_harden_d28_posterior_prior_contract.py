"""Phase 17-HARDEN-D28 - with no explicit prior, the posterior is data-only.

Contract (docs/phase17_harden_d28_prior_covariance_contract.md):

  - no explicit prior: DATA-ONLY objective (the right-hand side never pulls toward x_prior;
    the BLS acceptance cost has no prior term) and a DATA-ONLY posterior (H^T W H);
  - the 15 km / 5 m/s diagonal supplied in that case is a numerical STEP REGULARIZER
    (MATLAB P_bar_inv heritage), not a prior; it never enters the posterior;
  - explicit prior: MAP objective and posterior carry it (unchanged);
  - BLS K path with only a K qualification prior: posterior carries only the K row;
  - SRIF K path: since Phase 17-HARDEN-D29 its objective carries only the required K prior
    (plus an explicit state prior if supplied), so its posterior carries exactly those.

Before D28 every posterior silently added the default diagonal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("spiceypy")

import lunar_od.estimators as E  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402

from tests import test_harden_d25_posterior_dynamics_contract as D25  # noqa: E402
from tests import test_scenarios as S  # noqa: E402

TOL = D25.TOL
MU = 4902.800066e9


def setup_module(_module):
    D25.setup_module(_module)


def _default_prior(nx=6):
    p, *_ = E._prior_information_and_scale(nx, E._TWO_WAY_RANGE_NO_BIAS_CFG,
                                           E._two_way_range_bias_prior_and_scale, None, None)
    return p


def _sigma_rel(a, b):
    return float(np.max(np.abs(np.sqrt(np.diag(a)) / np.sqrt(np.diag(b)) - 1)))


def _info_rel(a, b):
    """Information difference measured in the covariance it implies (scale-free)."""
    return _sigma_rel(np.linalg.inv(a), np.linalg.inv(b))


@pytest.fixture(scope="module")
def fx():
    if not D25.GL1800F_TAB.exists():
        pytest.skip("GL1800F coefficient file not available")
    return D25._build_fixture()


# ----------------------------------------------------------------------
# the rule itself
# ----------------------------------------------------------------------
def test_posterior_prior_terms_keep_only_an_explicit_prior():
    p, r = np.diag([1.0, 2.0]), np.diag([1.0, 3.0])
    assert E._posterior_prior_terms(p, r, True) == (p, r)
    zp, zr = E._posterior_prior_terms(p, r, False)
    assert np.array_equal(zp, np.zeros((2, 2))) and np.array_equal(zr, np.zeros((2, 2)))
    assert E._posterior_prior_terms(p, None, False)[1] is None


# ----------------------------------------------------------------------
# two-way range BLS / SRIF (the D25 fixture: 3.3 h, one station, degree-8 field, SRP)
# ----------------------------------------------------------------------
def _six_info(fx, x, prior_inv):
    aug = D25.propagate_augmented_state(fx["t_pass"], np.concatenate([x, np.eye(6).reshape(-1, order="F")]),
                                        *D25._dyn(fx), **TOL, srp=fx["srp"], **fx["harm"], harmonic_stm_opt_in=True)
    _, h = E.two_way_range_nominal_and_initial_jacobian(fx["obs"], fx["pg"], aug)
    w = E._two_way_range_weight_diagonal(fx["obs"], fx["pg"])
    return h.T @ (w[:, None] * h) + prior_inv


def test_two_way_bls_six_state_posterior_is_data_only_without_a_prior(fx):
    x, _, st = E.estimate_two_way_range_bls_lm(*D25._args(fx), fx["x_true0"], fx["pg"], *D25._dyn(fx), max_iter=3,
                                               **TOL, srp=fx["srp"], return_posterior=True, **fx["harm"])
    x = np.asarray(x)
    data_only = _six_info(fx, x, 0.0)
    historical = data_only + _default_prior()
    info = np.asarray(st.posterior_information)
    assert _info_rel(info, data_only) <= 1e-8
    assert _info_rel(historical, data_only) > 1e-5               # the hidden prior is visible on this arc
    # an explicit prior is carried exactly (unchanged contract)
    x2, _, st2 = E.estimate_two_way_range_bls_lm(*D25._args(fx), fx["x_true0"], fx["pg"], *D25._dyn(fx), max_iter=3,
                                                 **TOL, srp=fx["srp"], prior_covariance=D25.PRIOR,
                                                 return_posterior=True, **fx["harm"])
    explicit = _six_info(fx, np.asarray(x2), np.linalg.inv(D25.PRIOR))
    assert _info_rel(np.asarray(st2.posterior_information), explicit) <= 1e-8


def test_two_way_bls_k_path_posterior_is_data_only_or_carries_only_the_k_prior(fx):
    kw = dict(max_iter=3, **TOL, srp=fx["srp"], solve_for_k_srp=True, k_srp_initial=D25.K_TRUE,
              return_posterior=True, **fx["harm"])
    x, _, st = E.estimate_two_way_range_bls_lm(*D25._args(fx), fx["x_true0"], fx["pg"], *D25._dyn(fx), **kw)
    data_only = D25._k_path_covariance(fx, np.asarray(x), st.k_srp_estimate, harm=fx["harm"])
    assert _sigma_rel(np.asarray(st.posterior_covariance), data_only) <= 1e-8
    assert st.k_srp_prior_status == "DATA_ONLY"
    # historical construction: data + default state diagonal
    hist_info = np.linalg.inv(data_only)
    hist_info[:6, :6] += _default_prior()
    assert _sigma_rel(np.linalg.inv(hist_info), data_only) > 1e-6
    # K qualification prior only: exactly the K row enters
    xk, _, sk = E.estimate_two_way_range_bls_lm(*D25._args(fx), fx["x_true0"], fx["pg"], *D25._dyn(fx),
                                                k_srp_prior_sigma=0.005, **kw)
    with_k = np.linalg.inv(D25._k_path_covariance(fx, np.asarray(xk), sk.k_srp_estimate, harm=fx["harm"]))
    with_k[6, 6] += 1.0 / 0.005 ** 2
    assert _sigma_rel(np.asarray(sk.posterior_covariance), np.linalg.inv(with_k)) <= 1e-8


def test_two_way_srif_six_state_is_data_only_and_k_path_carries_only_the_k_prior(fx):
    srp_only = dict(fx, harm={})
    aug0 = np.concatenate([fx["x_true0"], np.eye(6).reshape(-1, order="F")])
    hist = D25.propagate_augmented_state(fx["t_pass"], aug0, *D25._dyn(fx), srp=fx["srp"], **TOL)
    obs, pg = D25.generate_two_way_range_measurements(fx["t_pass"], hist[:, :6], (D25.base._STATION,),
                                                      np.ones((fx["t_pass"].size, 1), bool), fx["gep"], fx["gev"],
                                                      fx["et0"], noise=False)
    f2 = dict(srp_only, obs=np.asarray(obs, float), pg=pg)
    x, _, st = E.estimate_two_way_range_srif(f2["t_pass"], f2["obs"], f2["x_true0"], pg, *D25._dyn(f2), max_iter=3,
                                             **TOL, srp=f2["srp"], return_posterior=True)
    info = np.asarray(st.posterior_information)
    aug = D25.propagate_augmented_state(f2["t_pass"], np.concatenate([np.asarray(x), np.eye(6).reshape(-1, order="F")]),
                                        *D25._dyn(f2), **TOL, srp=f2["srp"])
    _, h = E.two_way_range_nominal_and_initial_jacobian(f2["obs"], pg, aug)
    w = E._two_way_range_weight_diagonal(f2["obs"], pg)
    data_only = h.T @ (w[:, None] * h)
    assert _info_rel(info, data_only) <= 1e-8
    # SRIF K path (HARDEN-D29): data-only state + the required K prior, in objective and posterior
    xk, _, sk = E.estimate_two_way_range_srif(f2["t_pass"], f2["obs"], f2["x_true0"], pg, *D25._dyn(f2), max_iter=3,
                                              **TOL, srp=f2["srp"], solve_for_k_srp=True, k_srp_initial=0.01,
                                              k_srp_prior_sigma=0.005, return_posterior=True)
    aug = E.propagate_state_with_k_sensitivity(f2["t_pass"], np.asarray(xk), *D25._dyn(f2),
                                               srp=SRPOptions(k_srp_m2_per_kg=sk.k_srp_estimate), **TOL)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(f2["obs"], pg, aug)
    hfull = np.hstack([hx, E._two_way_range_k_srp_column(f2["obs"], aug, hx)[:, None]])
    full = hfull.T @ (w[:, None] * hfull)
    full[6, 6] += 1.0 / 0.005 ** 2
    assert _sigma_rel(np.asarray(sk.posterior_covariance), np.linalg.inv(full)) <= 1e-8


# ----------------------------------------------------------------------
# position SRIF / BLS (scenario fixture: four stations, range + angles)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("estimator", ["estimate_position_srif", "estimate_position_bls_lm"])
def test_position_posterior_is_data_only_without_a_prior(estimator):
    t_all, x_truth, gep, gsp = S._matrix_truth_history(MU)
    stations = (S._synthetic_station(0.0, 0.0, 0.0), S._synthetic_station(0.0, 90.0, 0.0),
                S._synthetic_station(45.0, -30.0, 500.0), S._synthetic_station(-35.0, 150.0, 600.0))
    arc = S._build_position_arc(1, 0, 8, t_all, x_truth, stations)
    x0 = arc.truth_state_history_mci[0] + np.array([60.0, -50.0, 35.0, 0.025, -0.020, 0.012])
    fn = getattr(E, estimator)
    x, _, st = fn(arc.t_pass_s, arc.obs_data, x0, arc.pass_geo, MU, 0.0, 0.0, gep, gsp, max_iter=10,
                  rtol=1e-12, atol=1e-13, return_posterior=True)
    w = E._position_weight_diagonal(arc.obs_data, arc.pass_geo)
    args = (arc.t_pass_s, arc.obs_data, np.asarray(x)[:6], arc.pass_geo, MU, 0.0, 0.0, gep, gsp,
            E._resolve_position_bias_config(6, 6, len(stations), None))
    data_only = E._position_posterior_information(*args, np.zeros((6, 6)), w, 1e-12, 1e-13)
    historical = E._position_posterior_information(*args, _default_prior(), w, 1e-12, 1e-13)
    info = np.asarray(st.posterior_information)
    assert np.array_equal(info, data_only)
    assert not np.array_equal(historical, data_only)
    assert np.allclose(historical - data_only, _default_prior(), rtol=0, atol=1e-6 * np.max(np.abs(data_only)))
    if st.posterior_sqrt_information is not None:                       # SRIF: R^T R is the same information
        r = np.asarray(st.posterior_sqrt_information)
        assert _info_rel(r.T @ r, data_only) <= 1e-8
    assert math.isfinite(float(np.max(np.sqrt(np.diag(np.asarray(st.posterior_covariance))))))
