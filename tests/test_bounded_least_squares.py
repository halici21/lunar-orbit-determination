"""Convex reference problems certify bounds, stationarity and posterior arithmetic."""
import numpy as np
import pytest

def api():
    from lunar_od.bounded_least_squares import solve_bounded,posterior_diagnostics
    return solve_bounded,posterior_diagnostics

def test_boundary_does_not_block_feasible_state_correction():
    solve,_=api()
    r=solve(lambda x:np.array([1-x[0],-1-x[1]]),lambda x:-np.eye(2),np.zeros(2),np.ones(2),lower=np.array([-np.inf,0.]))
    np.testing.assert_allclose(r.x,[1.,0.],atol=1e-7)
    assert r.diagnostics['stationary']
    assert r.diagnostics['kkt_satisfied']

@pytest.mark.parametrize('condition',[1.,1e6])
def test_svd_covariance_matches_independent_design_reference(condition):
    _,diagnostics=api()
    rng=np.random.default_rng(7);u,_=np.linalg.qr(rng.normal(size=(30,3)));v,_=np.linalg.qr(rng.normal(size=(3,3)))
    a=u@np.diag([1.,condition**-.5,1/condition])@v.T
    z=diagnostics(a,np.zeros(30),np.zeros(3),np.ones(3))
    expected=v@np.diag([1.,condition,condition**2])@v.T
    np.testing.assert_allclose(z['covariance'],expected,rtol=2e-9,atol=1e-3)
    assert z['rank']==3

def test_rank_deficiency_is_not_zero_variance():
    _,diagnostics=api();z=diagnostics(np.ones((4,2)),np.zeros(4),np.zeros(2),np.ones(2))
    assert z['rank']==1
    assert z['covariance'] is None
    assert len(z['null_directions'])==1

def test_small_old_improvement_cannot_certify_nonstationarity():
    _,diagnostics=api()
    z=diagnostics(np.eye(2),np.ones(2),np.zeros(2),np.ones(2))
    assert not z['stationary']

@pytest.mark.parametrize('scale',[[1.,1.],[1e3,1e-3]])
def test_scale_and_jacobian_sign(scale):
    solve,_=api();a=np.array([[1.,2.],[3.,-1.],[1.,1.]]);b=a@np.array([2.,3.])
    z=solve(lambda x:b-a@x,lambda x:-a,np.zeros(2),np.array(scale))
    np.testing.assert_allclose(z.x,[2.,3.],rtol=0,atol=1e-7)
def test_small_decrement_is_not_lost_beside_large_irreducible_residual():
    _, diagnostics = api()
    result = diagnostics(np.array([[1.], [0.]]), np.array([1., 1e9]),
                                   np.array([0.]), np.array([1.]))
    assert result['gn_decrement'] == pytest.approx(1.)
    assert not result['stationary']


def _curved_problem():
    """Nonlinear residuals with a non-zero minimum; counts Jacobian calls by point."""
    calls = []
    def residual(x):
        return np.array([10 * (x[1] - x[0]**2), 1 - x[0], 0.3 * np.sin(x[0] + x[1]), 0.5])
    def jacobian(x):
        calls.append(np.array(x, float))
        c = 0.3 * np.cos(x[0] + x[1])
        return np.array([[-20 * x[0], 10.], [-1., 0.], [c, c], [0., 0.]])
    return residual, jacobian, calls


def test_stop_when_stationary_ends_early_at_a_certified_point():
    solve, _ = api()
    residual, jacobian, _ = _curved_problem()
    full = solve(residual, jacobian, np.array([-1.2, 1.]), np.ones(2), max_nfev=200)
    residual, jacobian, calls = _curved_problem()
    early = solve(residual, jacobian, np.array([-1.2, 1.]), np.ones(2), max_nfev=200, stop_when_stationary=True)
    assert early.scipy_status == -2
    assert early.diagnostics['stationary'] and full.diagnostics['stationary']
    assert early.nfev < full.nfev
    # certified: at most 0.1^2 of chi-square left to gain, so it is that close to the full solve
    assert early.diagnostics['gn_decrement'] <= 0.1
    assert early.residual @ early.residual - full.residual @ full.residual <= 0.1**2 + 1e-12


def test_stop_when_stationary_certifies_each_accepted_point_once():
    solve, _ = api()
    residual, jacobian, calls = _curved_problem()
    fit = solve(residual, jacobian, np.array([-1.2, 1.]), np.ones(2), max_nfev=200, stop_when_stationary=True)
    points = {tuple(c) for c in calls}
    # solver Jacobians + one certification per accepted point + the final diagnostics: no point evaluated 4 times
    assert all(sum(np.array_equal(c, np.array(p)) for c in calls) <= 3 for p in points)
    assert len(calls) <= 2 * fit.njev + 1


def test_default_still_runs_to_max_nfev_without_cost_stopping():
    solve, _ = api()
    residual, jacobian, _ = _curved_problem()
    fit = solve(residual, jacobian, np.array([-1.2, 1.]), np.ones(2), max_nfev=40)
    assert fit.scipy_status != -2
