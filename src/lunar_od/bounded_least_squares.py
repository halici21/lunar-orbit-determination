"""Bounded, whitened least squares with independent final-point diagnostics.

Residual Jacobians differentiate the residual (y-h), hence are -H. Covariance
is local unconstrained formal covariance, NOT a calibrated or truncated-normal
uncertainty at an active bound. Rank-deficient covariance is explicitly absent.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, lsq_linear


@dataclass
class BoundedSolveResult:
    x: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    diagnostics: dict
    scipy_status: int
    message: str
    nfev: int
    njev: int


def posterior_diagnostics(jacobian, residual, x, scale, *, lower=None, upper=None,
                          stationarity_sigma=0.1):
    """SVD posterior, projected gradient and feasible linearised objective gap.

    The square root of the remaining feasible linearised chi-square decrease
    is the GN decrement; it is compared with the declared 0.1 formal-sigma
    numerical criterion, separately from rank/identifiability and fit accuracy.
    """
    j=np.asarray(jacobian,float);r=np.asarray(residual,float);x=np.asarray(x,float);scale=np.asarray(scale,float)
    if j.ndim!=2 or j.shape!=(r.size,x.size) or scale.shape!=x.shape:
        raise ValueError('Jacobian/residual/parameter/scale shapes are inconsistent.')
    if not (np.all(np.isfinite(j)) and np.all(np.isfinite(r)) and np.all(np.isfinite(x))
            and np.all(np.isfinite(scale)) and np.all(scale>0)):
        raise ValueError('Finite data and positive finite parameter scales are required.')
    lo=np.full(x.size,-np.inf) if lower is None else np.broadcast_to(lower,x.shape).astype(float)
    hi=np.full(x.size,np.inf) if upper is None else np.broadcast_to(upper,x.shape).astype(float)
    if np.any(lo>=hi) or np.any(x<lo) or np.any(x>hi):
        raise ValueError('Infeasible parameters or invalid bounds.')
    a=j*scale
    _,s,vh=np.linalg.svd(a,full_matrices=a.shape[0]<a.shape[1])
    tol=np.finfo(float).eps*max(a.shape)*(s[0] if s.size else 0.)
    rank=int(np.sum(s>tol));n=x.size
    covariance=None
    if rank==n:
        factor=scale[:,None]*vh.T/s[None,:]
        covariance=(factor@factor.T).tolist()
    gradient=a.T@r
    bound_tol=1e-8*np.maximum(np.abs(x),scale)
    at_lo=np.isfinite(lo)&(x-lo<=bound_tol)
    at_hi=np.isfinite(hi)&(hi-x<=bound_tol)
    projected=gradient.copy()
    projected[at_lo&(gradient>0)]=0.
    projected[at_hi&(gradient<0)]=0.
    norm=np.linalg.norm(a,axis=0)
    normalized=np.divide(projected,norm,out=np.zeros_like(projected),where=norm>0)
    if np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)):
        step=lsq_linear(a,-r,bounds=((lo-x)/scale,(hi-x)/scale),method='trf',
                        lsq_solver='exact',tol=1e-12,max_iter=200).x
    else:
        step=np.linalg.lstsq(a,-r,rcond=None)[0]
    projected_step=a@step
    # Avoid cancellation between two large, almost identical chi-squares.
    decrease=float(-2*(r@projected_step)-projected_step@projected_step)
    decrement=float(np.sqrt(max(0.,decrease)))
    projected_norm=float(np.max(np.abs(normalized),initial=0.))
    kkt=projected_norm<=stationarity_sigma
    return dict(rank=rank,n_parameters=n,identifiable=rank==n,
                condition_number=float(s[0]/s[-1]) if rank==n and s.size else None,
                singular_values=s.tolist(),rank_tolerance=float(tol),covariance=covariance,
                covariance_semantics='local_unconstrained_formal_not_calibrated',
                null_directions=(vh[rank:]*scale).tolist(),scaled_null_directions=vh[rank:].tolist(),
                gradient_scaled=gradient.tolist(),projected_gradient_scaled=projected.tolist(),
                normalized_projected_gradient_inf=projected_norm,active_lower=at_lo.tolist(),active_upper=at_hi.tolist(),
                kkt_satisfied=bool(kkt),gn_decrement=decrement,
                feasible_gn_step_scaled=step.tolist(),feasible_gn_step=(scale*step).tolist(),
                predicted_chi_square_decrease=decrease,stationarity_sigma=float(stationarity_sigma),
                stationary=bool(kkt and decrement<=stationarity_sigma),chi_square=float(r@r))


def solve_bounded(residual, jacobian, x0, scale, *, lower=None, upper=None,
                  max_nfev=200, stationarity_sigma=0.1):
    """Dense SVD trust-region reflection; no normal-equation inversion.

    Cost/step stopping is disabled: small improvements are not stationarity.
    Native solver termination and independent certification are both returned.
    """
    x0=np.asarray(x0,float);scale=np.asarray(scale,float)
    lo=np.full(x0.size,-np.inf) if lower is None else np.broadcast_to(lower,x0.shape).astype(float)
    hi=np.full(x0.size,np.inf) if upper is None else np.broadcast_to(upper,x0.shape).astype(float)
    if scale.shape!=x0.shape or np.any(scale<=0) or not np.all(np.isfinite(scale)):
        raise ValueError('Positive finite x_scale matching x0 is required.')
    fit=least_squares(residual,x0,jac=jacobian,bounds=(lo,hi),method='trf',tr_solver='exact',
                      x_scale=scale,loss='linear',ftol=None,xtol=None,gtol=1e-12,max_nfev=max_nfev)
    # Evaluate both at the actual returned point, independently of the stop code.
    r=np.asarray(residual(fit.x),float);j=np.asarray(jacobian(fit.x),float)
    diag=posterior_diagnostics(j,r,fit.x,scale,lower=lo,upper=hi,stationarity_sigma=stationarity_sigma)
    return BoundedSolveResult(fit.x.copy(),r,j,diag,int(fit.status),str(fit.message),int(fit.nfev),int(fit.njev or 0))
