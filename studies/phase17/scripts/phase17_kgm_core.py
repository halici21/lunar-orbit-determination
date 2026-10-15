"""PHASE 17-KGM - final K closeout: first-order Gauss-Markov (stochastic) K_SRP, core (analysis layer).

Representation.  K(t) = K0 (1 + f_j) on N equal-duration segments (N = 15: one per orbit of the
15-orbit KP2 arc), realised EXACTLY by the KP2 construction generalised from 3 to N segments:
each segment is propagated by the production ``propagate_state_with_k_sensitivity`` at its own
constant K_j; the STM is chained and dx/dK_i is carried forward through later segments.

Stochastic model (the batch form of a first-order Gauss-Markov parameter; Tapley, Schutz & Born
2004 s4.9; Bierman 1977 ch. VII).  The segment deviations are

    f = m 1 + delta,   delta_{j+1} = phi delta_j + w_j,   phi = exp(-dt / tau),
    w_j ~ N(0, sigma^2 (1 - phi^2)),  delta_1 ~ N(0, sigma^2)        (stationary GM)

with the common level m given NO prior information (flat).  Integrating m out gives the prior
information on f

    Lambda = C^-1 - C^-1 1 1^T C^-1 / (1^T C^-1 1),   C_ij = sigma^2 phi^|i-j|,

of rank N - 1: the GM process constrains only the VARIATION of K; the common level stays
data-only, so common-level identifiability is untouched by the stochastic model.  The prior is
zero-mean in the variation, so it can only shrink a variation estimate toward none -- it cannot
manufacture one.

Estimator.  The KP2 staged batch Gauss-Newton (QR on the column-normalised whitened design;
production range + counted-Doppler VALUES and Jacobians; one free bias per Doppler pass), with
the GM rows sqrt(Lambda) appended to the K block.  Nothing in KP2 or production is changed.
"""
from __future__ import annotations

import math

import numpy as np

import phase17_kp1_core as KP
import phase17_kp2_core as K2

N_SEG = 15
M3_AVG = None   # set lazily: (3, N) segment -> third averaging matrix


# ======================================================================
# N-segment propagation and evaluation (KP2 generalised; KP2 untouched)
# ======================================================================
def boundaries(t_grid, n=N_SEG):
    return KP.boundaries(np.asarray(t_grid, float), n)


def propagate_segmented(scn, x0, ks, dyn):
    """(X (T,6), PHI (T,6,6) chained from t0, S (N,T,6) dx/dK_j) for K = ks[j] on segment j."""
    ks = np.asarray(ks, float)
    n = ks.size
    t = scn.t_grid
    br = boundaries(t, n)
    nt = t.size
    X, PHI, S = np.zeros((nt, 6)), np.zeros((nt, 6, 6)), np.zeros((n, nt, 6))
    x_start, phi_start, s_start = np.asarray(x0, float), np.eye(6), np.zeros((n, 6))
    for j in range(n):
        a, b = br[j], br[j + 1]
        seg = K2._prop_segment(scn, t[a:b + 1], x_start, ks[j], dyn)
        ph = np.stack([r.reshape((6, 6), order="F") for r in seg[:, 6:42]])
        X[a:b + 1] = seg[:, :6]
        PHI[a:b + 1] = ph @ phi_start
        for i in range(n):
            if i < j:
                S[i, a:b + 1] = np.einsum("nij,j->ni", ph, s_start[i])
            elif i == j:
                S[i, a:b + 1] = seg[:, 42:48]
            else:
                S[i, a:b + 1] = 0.0
        x_start, phi_start, s_start = X[b].copy(), PHI[b].copy(), S[:, b].copy()
    return X, PHI, S


def evaluate(fx, X, PHI, S):
    """K2.evaluate for any number of K columns (Doppler K columns in slot batches of <= 6)."""
    import phase17_r1o_cd1_core as CD
    from lunar_od.estimators import _two_way_range_k_srp_column
    from lunar_od.measurement_contracts import generate_two_way_range_measurements_checked
    from lunar_od.radiometrics import (resolve_counted_doppler_station_state_provider,
                                       two_way_counted_doppler_initial_state_jacobian,
                                       two_way_counted_doppler_observable)
    from lunar_od.two_way_range import TwoWayRangeConfig, two_way_range_nominal_and_initial_jacobian

    n = S.shape[0]
    rb, t = fx.rb, fx.scn.t_grid
    tight = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14, max_iter=200)
    obs, pg = generate_two_way_range_measurements_checked(t, X, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                          rb.earth.vel, fx.scn.et0, noise=False, rng=None,
                                                          config=tight)
    if obs.shape != rb.obs.shape or not np.array_equal(obs[:, [0, 2, 3]], rb.obs[:, [0, 2, 3]]):
        raise RuntimeError("range row set changed at the estimate")
    n42 = K2.nom48(X, PHI, np.zeros((t.size, 6)))
    _, hx_r = two_way_range_nominal_and_initial_jacobian(obs, pg, n42[:, :42])
    hx_r = np.asarray(hx_r, float)
    hk_r = np.column_stack([np.asarray(_two_way_range_k_srp_column(obs, K2.nom48(X, PHI, S[i]), hx_r), float)
                            for i in range(n)])
    pgt = rb.pass_geo
    cfg = CD.doppler_config(60.0)
    rows = fx.rows
    provs = {s: resolve_counted_doppler_station_state_provider(cfg, fx.stations[s], t, pgt.earth_pos_mci_m,
                                                               pgt.earth_vel_mci_mps, et0_s=pgt.et0_s)
             for s in set(int(x) for x in rows.station)}
    groups = [list(range(g, min(g + 6, n))) for g in range(0, n, 6)]
    slots = [KP.slot_history(n42, [S[i] for i in grp]) for grp in groups]
    yd, hx_d, hk_d = np.zeros(rows.n), np.zeros((rows.n, 6)), np.zeros((rows.n, n))
    for i in range(rows.n):
        s, tm = int(rows.station[i]), float(rows.t_mid[i])
        args = (tm, fx.stations[s], t)
        tail = (pgt.earth_pos_mci_m, pgt.earth_vel_mci_mps, pgt.x_j2000_to_itrf93, cfg)
        yd[i] = two_way_counted_doppler_observable(*args, X, *tail, et0_s=pgt.et0_s, station_state_provider=provs[s])
        hx_d[i] = two_way_counted_doppler_initial_state_jacobian(*args, n42[:, :42], *tail, et0_s=pgt.et0_s,
                                                                 station_state_provider=provs[s])
        for grp, slot in zip(groups, slots):
            hk_d[i, grp] = np.asarray(two_way_counted_doppler_initial_state_jacobian(
                *args, slot, *tail, et0_s=pgt.et0_s, station_state_provider=provs[s]), float)[:len(grp)]
    return dict(yr=obs[:, 1].copy(), hx_r=hx_r, hk_r=hk_r, yd=yd, hx_d=hx_d, hk_d=hk_d)


# ======================================================================
# Gauss-Markov prior on the variation
# ======================================================================
def gm_covariance(n, sigma, phi):
    idx = np.arange(n)
    return sigma ** 2 * phi ** np.abs(idx[:, None] - idx[None, :])


def gm_variation_information(n, sigma, tau_segments):
    """Lambda (n x n, rank n-1): stationary GM on the deviations, flat prior on the common level."""
    phi = math.exp(-1.0 / tau_segments)
    ci = np.linalg.inv(gm_covariance(n, sigma, phi))
    one = np.ones(n)
    lam = ci - np.outer(ci @ one, ci @ one) / float(one @ ci @ one)
    return 0.5 * (lam + lam.T)


def sqrt_information(lam):
    ev, vec = np.linalg.eigh(lam)
    keep = ev > 1e-12 * ev.max()
    return np.sqrt(ev[keep])[:, None] * vec[:, keep].T            # L with L^T L = Lambda


def sample_gm_truth(n, sigma, tau_segments, seed):
    """Zero-mean GM realisation (sample mean removed so the common level is K0, as every KP2 truth)."""
    rng = np.random.default_rng(seed)
    phi = math.exp(-1.0 / tau_segments)
    f = np.zeros(n)
    f[0] = rng.normal(0.0, sigma)
    for j in range(1, n):
        f[j] = phi * f[j - 1] + rng.normal(0.0, sigma * math.sqrt(1 - phi ** 2))
    return f - f.mean()


def thirds_matrix(n=N_SEG):
    """(3, n) averaging of the n segment deviations onto the M3 thirds (n divisible by 3)."""
    a = np.zeros((3, n))
    for j in range(3):
        a[j, j * n // 3:(j + 1) * n // 3] = 3.0 / n
    return a


# ======================================================================
# staged batch Gauss-Newton with the GM rows (KP2 solver generalised)
# ======================================================================
def _gauss_newton(fx, z, w, x0_ap, prior_sd, theta, free, *, dyn, nk, q_all, use_doppler, gm_l,
                  max_evals, conv, history, stage, start=None):
    free = np.asarray(free, bool)
    n_eval, rejected, converged, best, step, cand = 0, 0, False, None, None, theta.copy()
    while n_eval < max_evals:
        ks = cand[6:6 + nk]
        if np.any(ks <= 0):
            if best is None or rejected > 60:
                raise RuntimeError("non-positive K candidate with no step to reduce")
            rejected += 1
            step = 0.5 * step
            cand = best["theta"] + step
            continue
        if start is not None and best is None:
            ev = start["ev"]
            r, pr, gr, cost = start["r"], start["pr"], start["gr"], start["cost"]
        else:
            X, PHI, S = propagate_segmented(fx.scn, cand[:6], ks, dyn)
            ev = evaluate(fx, X, PHI, S)
            n_eval += 1
            pred = ev["yr"]
            if use_doppler:
                pred = np.concatenate([pred, ev["yd"] + (fx.nuis @ cand[6 + nk:] if q_all else 0.0)])
            r = (z - pred) * w
            pr = (np.asarray(x0_ap) - cand[:6]) / prior_sd
            gr = -(gm_l @ (ks / K2.K0 - 1.0)) if gm_l is not None else np.zeros(0)
            cost = float(r @ r + pr @ pr + gr @ gr)
            history.append(dict(stage=stage, eval=n_eval, cost=cost))
        if best is not None and cost > best["cost"] * (1 + 1e-12):
            step = 0.5 * step
            cand = best["theta"] + step
            history[-1]["rejected"] = True
            continue
        hk_r, hk_d = ev["hk_r"], ev["hk_d"]
        blocks_r, blocks_d = [ev["hx_r"], hk_r], [ev["hx_d"], hk_d]
        if q_all:
            blocks_r.append(np.zeros((ev["hx_r"].shape[0], q_all)))
            blocks_d.append(fx.nuis)
        a = np.hstack(blocks_r)
        if use_doppler:
            a = np.vstack([a, np.hstack(blocks_d)])
        aw = a * w[:, None]
        prior_rows = np.zeros((6, aw.shape[1]))
        prior_rows[:, :6] = np.diag(1 / prior_sd)
        aw, rr = np.vstack([aw, prior_rows]), np.concatenate([r, pr])
        if gm_l is not None:
            g_rows = np.zeros((gm_l.shape[0], aw.shape[1]))
            g_rows[:, 6:6 + nk] = gm_l / K2.K0
            aw, rr = np.vstack([aw, g_rows]), np.concatenate([rr, gr])
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
        best = dict(theta=cand.copy(), cost=cost, cov=cov_f, sd=sd_f, r=r, ev=ev, pr=pr, gr=gr)
        step = delta
        history[-1]["step_sigma_max"] = float(np.max(np.abs(delta_f) / sd_f))
        if np.max(np.abs(delta_f) / sd_f) < conv:
            converged = True
            break
        cand = cand + delta
    return best, n_eval, rejected, converged


def solve(fx, z_r, z_d, x0_ap, *, dyn, n_seg=N_SEG, gm_sigma, gm_tau_segments, prior_sd=K2.PRIOR_SD,
          stage1_evals=4, max_evals=10, conv=1e-2) -> dict:
    """KP2's staged solve (stage 1: state + biases at K0; stage 2: all free) with the GM variation prior."""
    nk = n_seg
    q_all = fx.nuis.shape[1]
    w = np.concatenate([np.full(len(z_r), 1 / K2.SIGMA_RANGE_M), np.full(len(z_d), 1 / K2.SIGMA_DOPPLER_MPS)])
    z = np.concatenate([z_r, z_d])
    gm_l = sqrt_information(gm_variation_information(nk, gm_sigma, gm_tau_segments))
    theta = np.concatenate([np.asarray(x0_ap, float), np.full(nk, K2.K0), np.zeros(q_all)])
    history = []
    kw = dict(dyn=dyn, nk=nk, q_all=q_all, use_doppler=True, gm_l=gm_l, history=history)
    free1 = np.ones(theta.size, bool)
    free1[6:6 + nk] = False
    b1, n1, rej1, c1 = _gauss_newton(fx, z, w, x0_ap, prior_sd, theta, free1, max_evals=stage1_evals, conv=1e-2,
                                     stage=1, **kw)
    b2, n2, rej2, c2 = _gauss_newton(fx, z, w, x0_ap, prior_sd, b1["theta"], np.ones(theta.size, bool),
                                     max_evals=max_evals, conv=conv, stage=2, start=b1, **kw)
    nr = len(z_r)
    r = b2["r"]
    return dict(theta=b2["theta"].tolist(), cov=b2["cov"].tolist(), sd=b2["sd"].tolist(), cost=b2["cost"],
                n_eval=n1 + n2, converged=c2, history=history, n_seg=nk, gm_sigma=gm_sigma,
                gm_tau_segments=gm_tau_segments, rejected_nonpositive_k=rej1 + rej2,
                range_wrms=float(np.sqrt(np.mean(r[:nr] ** 2))), doppler_wrms=float(np.sqrt(np.mean(r[nr:] ** 2))))


# ======================================================================
# metrics
# ======================================================================
def variation_basis(n):
    """Orthonormal basis (n, n-1) of the zero-mean subspace."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, :n - 1]]))
    return q[:, 1:]


def metrics(sol, x0_true, f_true, gm_sigma, gm_tau_segments) -> dict:
    """Predeclared figures: M3-projected recovery (KP2 rule), full-resolution detection, reconstruction,
    consistency, prior share (degrees of freedom for signal) and the common level (reported, not claimed)."""
    th, cov = np.asarray(sol["theta"]), np.asarray(sol["cov"])
    n = sol["n_seg"]
    f_true = np.asarray(f_true, float)
    f = th[6:6 + n] / K2.K0 - 1.0
    cf = cov[6:6 + n, 6:6 + n] / K2.K0 ** 2
    a3 = thirds_matrix(n)
    q3 = K2.Q_VAR
    v_hat, v_true = q3.T @ (a3 @ f), q3.T @ (a3 @ f_true)
    cv = q3.T @ a3 @ cf @ a3.T @ q3
    e = v_hat - v_true
    qn = variation_basis(n)
    u_hat, u_true = qn.T @ f, qn.T @ f_true
    cu = qn.T @ cf @ qn
    eu = u_hat - u_true
    lam = gm_variation_information(n, gm_sigma, gm_tau_segments)
    # degrees of freedom for signal in the variation subspace: (n-1) - tr(Lambda_u C_post,u)
    lam_u = qn.T @ lam @ qn
    dfs = float((n - 1) - np.trace(lam_u @ cu))
    var_true_rms = float(np.sqrt(np.mean((f_true - f_true.mean()) ** 2)))
    var_err_rms = float(np.sqrt(np.mean(((f - f.mean()) - (f_true - f_true.mean())) ** 2)))
    one = np.ones(n) / n
    m_hat, m_true, sm = float(one @ f), float(f_true.mean()), float(math.sqrt(one @ cf @ one))
    from scipy.stats import chi2
    return dict(converged=sol["converged"], n_eval=sol["n_eval"],
                f_hat=f.tolist(), f_sd=np.sqrt(np.diag(cf)).tolist(),
                m3_v_hat=v_hat.tolist(), m3_v_true=v_true.tolist(), m3_v_sd=np.sqrt(np.diag(cv)).tolist(),
                m3_chi2_err=float(e @ np.linalg.solve(cv, e)), m3_chi2_detect=float(v_hat @ np.linalg.solve(cv, v_hat)),
                full_chi2_err=float(eu @ np.linalg.solve(cu, eu)),
                full_chi2_detect=float(u_hat @ np.linalg.solve(cu, u_hat)),
                full_chi2_999=float(chi2.ppf(0.999, n - 1)),
                var_true_rms=var_true_rms, var_err_rms=var_err_rms,
                reconstruction_skill=(1.0 - var_err_rms / var_true_rms) if var_true_rms > 0 else None,
                dfs_variation=dfs, dfs_fraction=dfs / (n - 1),
                level_err=m_hat - m_true, level_sd=sm, level_z=(m_hat - m_true) / sm,
                x0_err_m=(th[:6] - np.asarray(x0_true)).tolist())
