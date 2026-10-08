"""PHASE 17-KP2 - nonlinear M3 K_SRP recovery: core (analysis layer).

The KP1-recommended M3 model K(t) = K_j on equal thirds of the arc is realised
EXACTLY by segmented production propagation: each segment is propagated with
the production ``propagate_state_with_k_sensitivity`` at its own constant K_j,
restarting from the previous segment's end state.  The STM is chained,
Phi(t, t0) = Phi_j(t, t_j) Phi(t_j, t0), and the trajectory sensitivity to each
segment coefficient is

    dx/dK_i (t) = 0                                  t before segment i
                = S_i(t)  (production dx/dK, zero at segment start)   t in segment i
                = Phi_j(t, t_j) dx/dK_i (t_j)        t in a later segment j

No production code is changed and no new derivative is integrated.

The estimator is a batch Gauss-Newton solve on the whitened design of the
PRODUCTION two-way range and counted-Doppler VALUES (D23-qualified path) with
their production Jacobians, solving for the initial state, the three segment
coefficients and (primary) one free constant bias per Doppler pass.  Solve and
covariance use QR / least squares on the column-normalised whitened design;
the normal matrix is never formed or inverted.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"
PRE_PATH = ARTIFACTS / "phase17_kp2_predeclaration.json"

K0 = 0.01                                     # m^2/kg, SYNTHETIC_CAMPAIGN_TRUTH common level
TRUTHS = {"T0_CONSTANT": (0.0, 0.0, 0.0),     # fractional segment deviations, zero mean
          "T1_STEP": (0.10, -0.10, 0.0),
          "T2_MIDDLE": (-0.05, 0.10, -0.05)}
SIGMA_RANGE_M = 5.0
SIGMA_DOPPLER_MPS = 1e-4
SIGMA_BIAS_TRUTH_MPS = 1e-4
PRIOR_SD = np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
#: orthonormal basis of the zero-mean (variation) subspace of the 3 fractional segment deviations
Q_VAR = np.column_stack([np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0), np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)])
CHI2_2_999 = 13.815510557964274               # chi^2_2 0.999 quantile
CHI2_9_999 = 27.877166449887585               # chi^2_9 0.999 quantile


class PredeclarationMismatch(RuntimeError):
    pass


def canonical_sha256(d: dict) -> str:
    import hashlib
    body = {k: v for k, v in d.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_predeclaration(pinned: str, path: Path = PRE_PATH) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    got = canonical_sha256(d)
    if not (got == d.get("canonical_sha256") == pinned):
        raise PredeclarationMismatch("KP2 predeclaration hash mismatch: %s / %s / %s"
                                     % (got, d.get("canonical_sha256"), pinned))
    return d


# ======================================================================
# segmented propagation
# ======================================================================
#: (rtol, atol) of the high-fidelity segment propagation (as phase17_r1od2_fidelity.propagate_fidelity)
HF_TOL = (1e-12, 1e-13)


def _prop_segment(scn, t_slice, x_start, k, dyn):
    if dyn == "HIGH_FIDELITY":
        import phase17_kp1_campaign as KPC
        return KPC._propagate(scn, t_slice, x_start, float(k), *HF_TOL)
    import phase17_geo_core as G
    return G.propagate48(t_slice, x_start, scn.et0, float(k), shadow_model=scn.shadow_model)


def boundaries(t_grid) -> list[int]:
    import phase17_kp1_core as KP
    return KP.boundaries(np.asarray(t_grid, float), 3)


def propagate_segmented(scn, x0, ks, dyn):
    """(X (N,6), PHI (N,6,6) chained from t0, S (3,N,6) dx/dK_j) for K = ks[j] on segment j."""
    t = scn.t_grid
    br = boundaries(t)
    n = t.size
    X, PHI, S = np.zeros((n, 6)), np.zeros((n, 6, 6)), np.zeros((3, n, 6))
    x_start, phi_start, s_start = np.asarray(x0, float), np.eye(6), np.zeros((3, 6))
    for j in range(3):
        a, b = br[j], br[j + 1]
        seg = _prop_segment(scn, t[a:b + 1], x_start, ks[j], dyn)
        ph = np.stack([r.reshape((6, 6), order="F") for r in seg[:, 6:42]])
        X[a:b + 1] = seg[:, :6]
        PHI[a:b + 1] = ph @ phi_start
        for i in range(3):
            if i < j:
                S[i, a:b + 1] = np.einsum("nij,j->ni", ph, s_start[i])
            elif i == j:
                S[i, a:b + 1] = seg[:, 42:48]
            else:
                S[i, a:b + 1] = 0.0
        x_start, phi_start, s_start = X[b].copy(), PHI[b].copy(), S[:, b].copy()
    return X, PHI, S


def nom48(X, PHI, s_col) -> np.ndarray:
    n = X.shape[0]
    return np.hstack([X, PHI.transpose(0, 2, 1).reshape(n, 36), s_col])


# ======================================================================
# fixed-row production measurement model
# ======================================================================
@dataclass
class Fixed:
    scn: object
    rb: object            # truth range block (vis, earth, xf, pass_geo, obs rows)
    rows: object          # truth CD1 Doppler rows (t_mid, station, session)
    stations: list
    nuis: np.ndarray      # (n_d, q) per-pass bias columns, 1.0 on the pass's rows


def build_fixed(scn, X, PHI, S, stations) -> tuple:
    import phase17_r1o_cd1_core as CD
    import phase17_r1od2_core as D2
    rb = D2.build_range_block(scn, nom48(X, PHI, S[0] + S[1] + S[2]))
    rows = CD.build_doppler_rows(scn, rb, 60.0, stations, with_values=True)
    nu = CD.nuisance_columns(0, rows, "D1_SESSION_CONSTANT_BIAS")
    return Fixed(scn, rb, rows, stations, nu), rb.obs[:, 1].copy(), np.asarray(rows.y, float).copy()


def evaluate(fx: Fixed, X, PHI, S) -> dict:
    """Production range and counted-Doppler values and Jacobians at a trajectory, on the fixed truth rows."""
    import phase17_kp1_core as KP
    import phase17_r1o_cd1_core as CD
    from lunar_od.estimators import _two_way_range_k_srp_column
    from lunar_od.measurement_contracts import generate_two_way_range_measurements_checked
    from lunar_od.radiometrics import (resolve_counted_doppler_station_state_provider,
                                       two_way_counted_doppler_initial_state_jacobian,
                                       two_way_counted_doppler_observable)
    from lunar_od.two_way_range import TwoWayRangeConfig, two_way_range_nominal_and_initial_jacobian

    rb, t = fx.rb, fx.scn.t_grid
    tight = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14, max_iter=200)
    obs, pg = generate_two_way_range_measurements_checked(t, X, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                          rb.earth.vel, fx.scn.et0, noise=False, rng=None,
                                                          config=tight)
    if obs.shape != rb.obs.shape or not np.array_equal(obs[:, [0, 2, 3]], rb.obs[:, [0, 2, 3]]):
        raise RuntimeError("range row set changed at the estimate")
    n42 = nom48(X, PHI, np.zeros((t.size, 6)))
    _, hx_r = two_way_range_nominal_and_initial_jacobian(obs, pg, n42[:, :42])
    hx_r = np.asarray(hx_r, float)
    hk_r = np.column_stack([np.asarray(_two_way_range_k_srp_column(obs, nom48(X, PHI, S[i]), hx_r), float)
                            for i in range(3)])
    pgt = rb.pass_geo
    cfg = CD.doppler_config(60.0)
    rows = fx.rows
    provs = {s: resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], t, pgt.earth_pos_mci_m,
                                                               pgt.earth_vel_mci_mps, et0_s=pgt.et0_s)
             for s in set(int(x) for x in rows.station)}
    slot = KP.slot_history(n42, [S[0], S[1], S[2]])
    yd, hx_d, hk_d = np.zeros(rows.n), np.zeros((rows.n, 6)), np.zeros((rows.n, 3))
    for i in range(rows.n):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])
        args = (tm, fx.stations[s], t)
        tail = (pgt.earth_pos_mci_m, pgt.earth_vel_mci_mps, pgt.x_j2000_to_itrf93, cfg)
        yd[i] = two_way_counted_doppler_observable(*args, X, *tail, et0_s=pgt.et0_s, station_state_provider=provs[s])
        hx_d[i] = two_way_counted_doppler_initial_state_jacobian(*args, n42[:, :42], *tail, et0_s=pgt.et0_s,
                                                                 station_state_provider=provs[s])
        hk_d[i] = np.asarray(two_way_counted_doppler_initial_state_jacobian(
            *args, slot, *tail, et0_s=pgt.et0_s, station_state_provider=provs[s]), float)[:3]
    return dict(yr=obs[:, 1].copy(), hx_r=hx_r, hk_r=hk_r, yd=yd, hx_d=hx_d, hk_d=hk_d)


# ======================================================================
# batch Gauss-Newton estimator
# ======================================================================
def _design(ev, fx, model, bias_mode, use_doppler):
    hk_r, hk_d = ev["hk_r"], ev["hk_d"]
    if model == "M0":
        hk_r, hk_d = hk_r.sum(axis=1, keepdims=True), hk_d.sum(axis=1, keepdims=True)
    blocks_r = [ev["hx_r"], hk_r]
    blocks_d = [ev["hx_d"], hk_d]
    q = fx.nuis.shape[1] if (use_doppler and bias_mode == "FREE") else 0
    if q:
        blocks_r.append(np.zeros((ev["hx_r"].shape[0], q)))
        blocks_d.append(fx.nuis)
    a = np.hstack(blocks_r)
    if use_doppler:
        a = np.vstack([a, np.hstack(blocks_d)])
    return a, q


def _gauss_newton(fx, z, w, x0_ap, prior_sd, theta, free, *, dyn, nk, q_all, model, bias_mode, use_doppler,
                  max_evals, conv, history, stage, start=None):
    """Gauss-Newton with step halving over the FREE parameters; others held fixed.

    A candidate whose K would be non-positive is rejected without evaluating the observable and the step
    halved (as the production BLS, Phase 17-R s18) -- never clipped."""
    free = np.asarray(free, bool)
    n_eval, rejected, converged, best, step, cand = 0, 0, False, None, None, theta.copy()
    while n_eval < max_evals:
        ks = cand[6:6 + nk] if nk == 3 else np.repeat(cand[6], 3)
        if np.any(ks <= 0):
            if best is None or rejected > 60:
                raise RuntimeError("non-positive K candidate with no step to reduce")
            rejected += 1
            step = 0.5 * step
            cand = best["theta"] + step
            continue
        if start is not None and best is None:            # reuse the previous stage's evaluation of this point
            ev, r, pr, cost = start["ev"], start["r"], start["pr"], start["cost"]
        else:
            X, PHI, S = propagate_segmented(fx.scn, cand[:6], ks, dyn)
            ev = evaluate(fx, X, PHI, S)
            n_eval += 1
            pred = ev["yr"]
            if use_doppler:
                pred = np.concatenate([pred, ev["yd"] + (fx.nuis @ cand[6 + nk:] if q_all else 0.0)])
            r = (z - pred) * w
            pr = (np.asarray(x0_ap) - cand[:6]) / prior_sd if prior_sd is not None else np.zeros(0)
            cost = float(r @ r + pr @ pr)
            history.append(dict(stage=stage, eval=n_eval, cost=cost))
        if best is not None and cost > best["cost"] * (1 + 1e-12):
            step = 0.5 * step
            cand = best["theta"] + step
            history[-1]["rejected"] = True
            continue
        a, _ = _design(ev, fx, model, bias_mode, use_doppler)
        aw = a * w[:, None]
        rr = r
        if prior_sd is not None:
            prior_rows = np.zeros((6, aw.shape[1]))
            prior_rows[:, :6] = np.diag(1 / prior_sd)
            aw, rr = np.vstack([aw, prior_rows]), np.concatenate([r, pr])
        aw = aw[:, free]
        cs = np.linalg.norm(aw, axis=0)
        cs[cs == 0] = 1.0
        an = aw / cs
        delta_f = np.linalg.lstsq(an, rr, rcond=None)[0] / cs
        rmat = np.linalg.qr(an, mode="r")
        rinv = np.linalg.solve(rmat, np.eye(rmat.shape[0]))
        cov_f = (rinv @ rinv.T) / np.outer(cs, cs)
        sd_f = np.sqrt(np.diag(cov_f))
        delta = np.zeros_like(cand)
        delta[free] = delta_f
        best = dict(theta=cand.copy(), cost=cost, cov=cov_f, sd=sd_f, r=r, ev=ev, pr=pr)
        step = delta
        if history:
            history[-1]["step_sigma_max"] = float(np.max(np.abs(delta_f) / sd_f))
            history[-1]["step_sigma_argmax"] = int(np.flatnonzero(free)[int(np.argmax(np.abs(delta_f) / sd_f))])
        if np.max(np.abs(delta_f) / sd_f) < conv:
            converged = True
            break
        cand = cand + delta
    return best, n_eval, rejected, converged


def solve(fx: Fixed, z_r, z_d, x0_ap, *, dyn, model="M3", bias_mode="FREE", use_doppler=True,
          prior_sd=PRIOR_SD, stage1_evals=4, max_evals=10, conv=1e-2) -> dict:
    """Staged batch Gauss-Newton (standard OD initialisation, no truth knowledge):

    stage 1: initial state (+ biases) with every K held at the nominal K0 (up to ``stage1_evals``, 1e-2 sigma);
    stage 2: all parameters free from the stage-1 point, to ``conv`` sigma (1e-2: the model evaluation
    carries integrator noise of ~1e-3 sigma, so a tighter step criterion cannot be met reproducibly).
    The reported covariance is the QR covariance of the full parameter set at the converged point."""
    nk = 3 if model == "M3" else 1
    q_all = fx.nuis.shape[1] if (use_doppler and bias_mode == "FREE") else 0
    w = np.concatenate([np.full(len(z_r), 1 / SIGMA_RANGE_M), np.full(len(z_d) if use_doppler else 0,
                                                                         1 / SIGMA_DOPPLER_MPS)])
    z = np.concatenate([z_r, z_d]) if use_doppler else np.asarray(z_r)
    theta = np.concatenate([np.asarray(x0_ap, float), np.full(nk, K0), np.zeros(q_all)])
    history = []
    kw = dict(dyn=dyn, nk=nk, q_all=q_all, model=model, bias_mode=bias_mode, use_doppler=use_doppler,
              history=history)
    free1 = np.ones(theta.size, bool)
    free1[6:6 + nk] = False
    b1, n1, rej1, c1 = _gauss_newton(fx, z, w, x0_ap, prior_sd, theta, free1, max_evals=stage1_evals, conv=1e-2,
                                     stage=1, **kw)
    b2, n2, rej2, c2 = _gauss_newton(fx, z, w, x0_ap, prior_sd, b1["theta"], np.ones(theta.size, bool),
                                     max_evals=max_evals, conv=conv, stage=2, start=b1, **kw)
    th, cov, sd, r = b2["theta"], b2["cov"], b2["sd"], b2["r"]
    nr = len(z_r)
    res_r, res_d = r[:nr], r[nr:]
    return dict(theta=th.tolist(), cov=cov.tolist(), sd=sd.tolist(), cost=b2["cost"], n_eval=n1 + n2,
                n_eval_stage1=n1, stage1_converged=c1, converged=c2, history=history,
                rejected_nonpositive_k=rej1 + rej2, model=model, bias_mode=bias_mode, use_doppler=use_doppler,
                n_params=int(th.size), n_range=nr, n_doppler=int(len(res_d)),
                range_wrms=float(np.sqrt(np.mean(res_r ** 2))),
                doppler_wrms=float(np.sqrt(np.mean(res_d ** 2))) if len(res_d) else None)


# ======================================================================
# recovery metrics
# ======================================================================
def recovery_metrics(sol: dict, x0_true, f_true) -> dict:
    th, cov = np.asarray(sol["theta"]), np.asarray(sol["cov"])
    out = dict(converged=sol["converged"], n_eval=sol["n_eval"])
    if sol["model"] != "M3":
        k, sk = th[6], math.sqrt(cov[6, 6])
        out.update(k_frac_err=float(k / K0 - 1 - np.mean(f_true)), k_frac_sd=float(sk / K0))
        return out
    f = th[6:9] / K0 - 1.0
    cf = cov[6:9, 6:9] / K0 ** 2
    v_hat, v_true = Q_VAR.T @ f, Q_VAR.T @ np.asarray(f_true)
    cv = Q_VAR.T @ cf @ Q_VAR
    cvi = np.linalg.inv(cv)
    e = v_hat - v_true
    one = np.ones(3) / 3.0
    m_hat, m_true = float(one @ f), float(np.mean(f_true))
    sm = float(math.sqrt(one @ cf @ one))
    xe = th[:6] - np.asarray(x0_true)
    full = np.concatenate([xe, f - np.asarray(f_true)])
    cfull = np.zeros((9, 9))
    cfull[:6, :6], cfull[:6, 6:], cfull[6:, :6], cfull[6:, 6:] = cov[:6, :6], cov[:6, 6:9] / K0, cov[6:9, :6] / K0, cf
    corr = cf / np.outer(np.sqrt(np.diag(cf)), np.sqrt(np.diag(cf)))
    out.update(f_hat=f.tolist(), f_sd=np.sqrt(np.diag(cf)).tolist(), v_hat=v_hat.tolist(), v_true=v_true.tolist(),
               v_sd=np.sqrt(np.diag(cv)).tolist(),
               chi2_var_err=float(e @ cvi @ e), chi2_var_detect=float(v_hat @ cvi @ v_hat),
               var_rel_err=float(np.linalg.norm(e) / np.linalg.norm(v_true)) if np.linalg.norm(v_true) > 0 else None,
               level_err=m_hat - m_true, level_sd=sm, level_z=(m_hat - m_true) / sm,
               nees9=float(full @ np.linalg.solve(cfull, full)),
               max_abs_segment_corr=float(np.max(np.abs(corr - np.eye(3)))))
    return out
