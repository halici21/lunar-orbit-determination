"""Phase 17-HARDEN-D25: qualification of the posterior-covariance dynamics contract.

Gates are frozen in studies/phase17/configs/phase17_d25_predeclaration.json (written before any number here).

Modes (each writes one section of studies/phase17/results/phase17_d25_qualification.json):

  --kp2    KP2 CANON k0 high-fidelity K path at the BLS1 production solution (taken from
           studies/phase17/results/phase17_bls1_qualification.json, i.e. reached from the declared a priori):
           repaired vs pre-D25 production posterior, O1 independent assembly, O2 central finite
           differences of the solve's own nonlinear model, and the frozen KP2 sigma criterion
           against a fresh KP2 solver run on the same data.
  --e2e    the frozen KP2 job_bls rerun end to end with the repaired estimator (confirmatory).
  --stage-f  blast radius: the gravity campaign's 40 stored six-state solutions (read-only from
           od_covariance_campaign): reproduction of the stored P_hat by the pre-D25 posterior,
           repaired posteriors, stage G statistics re-derived with stage G's formulas, and O1.

The estimator is never re-solved here except in --e2e; covariances are evaluated at stored
solutions with the production helpers, exactly as the estimators call them after the solve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import lunar_od.estimators as E  # noqa: E402  (this worktree's lunar_od first)
from lunar_od.project_paths import campaign_root  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase17_d25_qualification.json"
BLS1 = P.ARTIFACTS / "phase17_bls1_qualification.json"
STAGE_F = campaign_root() / "05_reports" / "stage_f_n20_raw.json"
FD_TOL = dict(rtol=1e-13, atol=1e-14)
FD_STEPS = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-4])


# ----------------------------------------------------------------------
# shared algebra (independent of the production covariance helpers)
# ----------------------------------------------------------------------
def cov_from_design(h, w, prior_inv=None):
    """Column-normalised QR inverse of the whitened (and prior-augmented) design."""
    a = h * np.sqrt(w)[:, None]
    if prior_inv is not None and np.any(prior_inv):
        ev, v = np.linalg.eigh(0.5 * (prior_inv + prior_inv.T))
        a = np.vstack([a, np.sqrt(np.clip(ev, 0, None))[:, None] * v.T])
    c = np.linalg.norm(a, axis=0)
    r_inv = np.linalg.inv(np.linalg.qr(a / c, mode="r"))
    cn = r_inv @ r_inv.T
    return cn / np.outer(c, c)


def compare(cov, ref):
    s, sr = np.sqrt(np.diag(cov)), np.sqrt(np.diag(ref))
    corr = lambda m: m / np.outer(np.sqrt(np.diag(m)), np.sqrt(np.diag(m)))  # noqa: E731
    return dict(sigma_rel=(s / sr - 1).tolist(), max_sigma_rel=float(np.max(np.abs(s / sr - 1))),
                max_corr_abs=float(np.max(np.abs(corr(cov) - corr(ref)))))


# ----------------------------------------------------------------------
# KP2 high-fidelity K path
# ----------------------------------------------------------------------
_CTX = {}


def _kp2_ctx():
    if _CTX:
        return _CTX
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from lunar_od.srp import SRPOptions
    from phase17_r1od2_fidelity import NMAX
    P.K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), P.HF)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rng = np.random.default_rng([98, 0, 0])                    # the frozen job's draws
    dx = rng.normal(0.0, K2.PRIOR_SD)
    z_r = zr + rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    obs = np.array(fx.rb.obs, float, copy=True)
    obs[:, 1] = z_r
    hf = json.loads(BLS1.read_text(encoding="utf-8"))["high_fidelity"]["production"]
    _CTX.update(
        C=C, fx=fx, zr=zr, zd=zd, z_r=z_r, dx=dx, x0=x0, scn=scn, obs=obs, pg=fx.rb.pass_geo, t=scn.t_grid,
        x_hat=x0 + np.asarray(hf["x0_err_m"], float), k_hat=float(hf["k"]),
        args=(MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, lambda t: G.earth_state_mci(scn.et0 + float(t))[:3],
              lambda t: G.sun_mci(scn.et0 + float(t))),
        harm=dict(harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0),
        srp=SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model),
        tol=dict(rtol=1e-12, atol=1e-13), j2=0.0)
    return _CTX


def _kp2_residual(theta):
    c = _kp2_ctx()
    s = replace(c["srp"], k_srp_m2_per_kg=float(theta[6]))
    hist = E.propagate_state(c["t"], np.asarray(theta[:6]), *c["args"], srp=s, j2_moon=c["j2"], **FD_TOL,
                             **c["harm"])
    r, _ = E.compute_two_way_range_residuals(hist, c["obs"], c["pg"])
    return np.asarray(r, float)


def _kp2_fd_column(j, step=None):
    c = _kp2_ctx()
    th = np.concatenate([c["x_hat"], [c["k_hat"]]])
    h = FD_STEPS[j] if step is None else step
    e = np.zeros(7)
    e[j] = h
    return j, -(_kp2_residual(th + e) - _kp2_residual(th - e)) / (2 * h)   # residual = obs - model


def _kp2_production():
    """Production posterior (repaired and pre-D25) exactly as estimate_two_way_range_bls_lm forms it."""
    c = _kp2_ctx()
    w = E._two_way_range_weight_diagonal(c["obs"], c["pg"])
    # the estimator's own posterior contract with no explicit prior: its default weak state prior
    # (15 km, 5 m/s; none on K) is part of the reported posterior, so every rebuild includes it
    prior6, _, scale6, _, _ = E._prior_information_and_scale(6, E._TWO_WAY_RANGE_NO_BIAS_CFG,
                                                             E._two_way_range_bias_prior_and_scale, None, None)
    scale = np.diag(list(np.diag(scale6)) + [1e-2])
    prior_inv = np.zeros((7, 7))
    prior_inv[:6, :6] = prior6
    out = {}
    for name, hk in (("repaired", c["harm"]), ("pre_d25", None)):
        d = E._two_way_range_posterior_design_with_k(
            c["t"], c["obs"], c["x_hat"], c["k_hat"], c["srp"], c["pg"], *c["args"],
            c["tol"]["rtol"], c["tol"]["atol"], c["j2"], harmonic_kwargs=hk)
        out[name], _ = E._square_root_covariance_from_design(d, w, prior_inv, scale)
    # O1: independent assembly - own propagation call with the solve's kwargs, own QR
    aug = E.propagate_state_with_k_sensitivity(
        c["t"], c["x_hat"], *c["args"], srp=replace(c["srp"], k_srp_m2_per_kg=c["k_hat"]), j2_moon=c["j2"],
        **c["tol"], **c["harm"], harmonic_stm_opt_in=True)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(c["obs"], c["pg"], aug)
    hk_col = E._two_way_range_k_srp_column(c["obs"], aug, hx)
    out["o1"] = cov_from_design(np.hstack([hx, hk_col[:, None]]), w, prior_inv)
    out["w"], out["prior_inv"] = w, prior_inv
    return out


def _kp2_solver():
    c = _kp2_ctx()
    mine = K2.solve(c["fx"], c["z_r"], c["zd"], c["x0"] + c["dx"], dyn=P.HF, model="M0", bias_mode="NONE",
                    use_doppler=False, prior_sd=None)
    return dict(theta=mine["theta"], sd=mine["sd"], converged=mine["converged"])


def _run_task(task):
    t0 = time.time()
    if task == "production":
        out = _kp2_production()
    elif task == "solver":
        out = _kp2_solver()
    elif task.startswith("K@"):
        out = _kp2_fd_column(6, float(task[2:]))
    elif task.startswith("V"):                                  # "V<j>@<step>" velocity column j (3..5)
        j, h = task[1:].split("@")
        out = _kp2_fd_column(int(j), float(h))
    else:
        out = _kp2_fd_column(int(task))
    return task, out, time.time() - t0


POSTHOC_K_STEPS = (5e-3, 2.5e-3)    # simplified K-column step study (error ~1/h); central steps must keep K > 0 (K_hat 0.0054)


def run_kp2_posthoc_fd(workers: int) -> dict:
    """POST-HOC (not a predeclared gate): O2 rebuilt with the K step moved out of the FD noise floor.

    The predeclared K step 1e-4 sits in the integrator-noise floor: in simplified CANON the K-column
    FD error falls ~1/h (3.8e-2 at 1e-4, 3.3e-3 at 1e-3, 3.9e-4 at 3e-3, 8.8e-5 at 9e-3), with and
    without shadow. The analytic column is not at fault. State steps are unchanged.
    """
    tasks = ["production"] + [str(j) for j in range(6)] + ["K@%g" % h for h in POSTHOC_K_STEPS]
    with ProcessPoolExecutor(workers) as ex:
        res = {t: (o, s) for t, o, s in ex.map(_run_task, tasks)}
    prod = res["production"][0]
    state_cols = [res[str(j)][0][1] for j in range(6)]
    out = dict(label="POST-HOC O2 with K step out of the FD noise floor; predeclared G4 result unchanged",
               k_steps=list(POSTHOC_K_STEPS), per_k_step={})
    for h in POSTHOC_K_STEPS:
        h_fd = np.column_stack(state_cols + [res["K@%g" % h][0][1]])
        o2 = cov_from_design(h_fd, prod["w"], prod["prior_inv"])
        out["per_k_step"]["%g" % h] = dict(vs_repaired=compare(prod["repaired"], o2),
                                           vs_pre_d25=compare(prod["pre_d25"], o2))
    k1, k2 = (res["K@%g" % h][0][1] for h in POSTHOC_K_STEPS)
    out["k_column_step_consistency_rel"] = float(np.linalg.norm(k1 - k2) / np.linalg.norm(k1))
    return out


def run_kp2() -> dict:
    tasks = ["solver", "production"] + [str(j) for j in range(7)]
    with ProcessPoolExecutor(len(tasks)) as ex:
        res = {t: (o, s) for t, o, s in ex.map(_run_task, tasks)}
    prod = res["production"][0]
    h_fd = np.column_stack([res[str(j)][0][1] for j in range(7)])
    o2 = cov_from_design(h_fd, prod["w"], prod["prior_inv"])
    rep, pre = prod["repaired"], prod["pre_d25"]
    o2_rep, o2_pre = compare(rep, o2), compare(pre, o2)
    o1 = compare(rep, prod["o1"])
    solver = res["solver"][0]
    ratio = np.asarray(solver["sd"]) / np.sqrt(np.diag(rep))
    ratio_pre = np.asarray(solver["sd"]) / np.sqrt(np.diag(pre))
    gates = dict(
        G3_O1=o1["max_sigma_rel"] <= 1e-6 and o1["max_corr_abs"] <= 1e-6,
        G4_O2=o2_rep["max_sigma_rel"] <= 1e-3 and o2_rep["max_corr_abs"] <= 1e-3,
        G4_positive_control=abs(o2_pre["sigma_rel"][6]) > 1e-2,
        G5_frozen_sigma=bool(np.max(np.abs(ratio - 1)) <= 0.01))
    return dict(
        point="BLS1 production solution from the declared a priori (studies/phase17/results/phase17_bls1_qualification.json)",
        sigma_repaired=np.sqrt(np.diag(rep)).tolist(), sigma_pre_d25=np.sqrt(np.diag(pre)).tolist(),
        o1_vs_repaired=o1, o2_vs_repaired=o2_rep, o2_vs_pre_d25=o2_pre,
        fd=dict(tol=FD_TOL, steps=FD_STEPS.tolist()),
        kp2_solver=dict(converged=solver["converged"], k=solver["theta"][6],
                        sigma_ratio_repaired=ratio.tolist(), sigma_ratio_pre_d25=ratio_pre.tolist()),
        gates=gates, elapsed_s={t: s for t, (o, s) in res.items()})


def run_e2e() -> dict:
    P.K._init()
    frozen = P.job_bls(dict(x0=P.x0s()["CANON"]))
    return dict(frozen_job_bls=frozen,
                estimate_within_0p02_sigma=max(map(abs, frozen["diff_over_sigma"])) <= 0.02,
                sigma_within_1pct=max(abs(x - 1) for x in frozen["sigma_ratio"]) <= 0.01)


# ----------------------------------------------------------------------
# stage F / G blast radius (six-state, gravity campaign)
# ----------------------------------------------------------------------
_SF = {}


def _sf_row(i):
    if not _SF:
        import od_gravity_covariance_campaign as C  # noqa: N811
        _SF["C"] = C
        _SF["rows"] = json.loads(STAGE_F.read_text(encoding="utf-8"))
    C, row = _SF["C"], _SF["rows"][i]
    t0 = time.time()
    obs, geo = C.make_obs(C.T_FIT, C.X_TRUTH_FIT, C.VIS_FIT, row["seed"])
    kw = C.CASES[row["case"]]
    x = np.asarray(row["x_est"], float)
    w = E._two_way_range_weight_diagonal(obs, geo)
    prior_inv = np.linalg.inv(C.P0)
    base = (C.T_FIT, obs, x, geo, C.MU, 0.0, 0.0, C.get_earth_pos, C.get_sun_pos, prior_inv, w,
            C.EST_RTOL, C.EST_ATOL, kw["j2_moon"])
    harm = dict(harmonic_model=kw["harmonic_model"], harmonic_epoch_et0=kw["harmonic_epoch_et0"])
    info_pre = E._two_way_range_posterior_information(*base)
    info_rep = E._two_way_range_posterior_information(*base, harmonic_kwargs=harm)
    p_pre, p_rep = np.linalg.inv(info_pre), np.linalg.inv(info_rep)      # the campaign's own convention
    p_pre, p_rep = 0.5 * (p_pre + p_pre.T), 0.5 * (p_rep + p_rep.T)
    stored = np.asarray(row["P_hat"], float)
    out = dict(seed=row["seed"], case=row["case"], e0=row["e0"], stop=row["stop"],
               reproduction_max_rel=float(np.max(np.abs(p_pre - stored)) / np.max(np.abs(stored))),
               P_pre=p_pre.tolist(), P_rep=p_rep.tolist())
    if i in (0, 1, 2, 3):                                                  # O1 on a subset (both cases)
        aug = E.propagate_augmented_state(C.T_FIT, np.concatenate([x, np.eye(6).reshape(-1, order="F")]),
                                          C.MU, 0.0, 0.0, C.get_earth_pos, C.get_sun_pos, rtol=C.EST_RTOL,
                                          atol=C.EST_ATOL, j2_moon=kw["j2_moon"], **harm, harmonic_stm_opt_in=True)
        _, h = E.two_way_range_nominal_and_initial_jacobian(obs, geo, aug)
        out["o1_vs_repaired"] = compare(p_rep, cov_from_design(h, w, prior_inv))
    out["elapsed_s"] = time.time() - t0
    return out


def stage_g(rows, key):
    """Stage G's own statistics (06_scripts/stage_g_statistics.py), evaluated on covariance `key`."""
    from scipy.stats import chi2
    out = {}
    for case in ("E50_G50", "E100_G100"):
        rs = sorted([r for r in rows if r["case"] == case], key=lambda r: r["seed"])
        n = len(rs)
        e = np.array([r["e0"] for r in rs])
        ps = np.array([r[key] for r in rs])
        p_mean = ps.mean(axis=0)
        p_mean = 0.5 * (p_mean + p_mean.T)
        nees = np.array([v @ np.linalg.solve(0.5 * (p + p.T), v) for v, p in zip(e, ps)])
        anees = float(nees.mean())
        lo, hi = float(chi2.ppf(0.025, n * 6) / n), float(chi2.ppf(0.975, n * 6) / n)
        z = e / np.sqrt(np.array([np.diag(p) for p in ps]))
        c_emp = np.cov(e, rowvar=False)
        out[case] = dict(n=n, anees=anees, anees_band=[lo, hi], anees_ok=bool(lo <= anees <= hi),
                         overconfidence_ratio=anees / 6, sigma_scale_that_would_fix=float(np.sqrt(anees / 6)),
                         coverage_1sigma=float((np.abs(z) <= 1).mean()),
                         mahalanobis_95=float((nees <= chi2.ppf(0.95, 6)).mean()),
                         formal_sigma=np.sqrt(np.diag(p_mean)).tolist(),
                         formal_3d_position_sigma_m=float(np.sqrt(np.trace(p_mean[:3, :3]))),
                         empirical_3d_position_sigma_m=float(np.sqrt(np.trace(c_emp[:3, :3]))))
    return out


def run_stage_f(workers: int, limit: int | None = None) -> dict:
    all_rows = json.loads(STAGE_F.read_text(encoding="utf-8"))
    idx = [i for i, r in enumerate(all_rows) if r["status"] == "OK"][:limit]   # stage G's own row filter
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_sf_row, idx))
    pre, rep = stage_g(rows, "P_pre"), stage_g(rows, "P_rep")
    sig = {c: (np.array(rep[c]["formal_sigma"]) / np.array(pre[c]["formal_sigma"]) - 1).tolist() for c in pre}
    return dict(
        n_rows=len(rows),
        G7_reproduction_max_rel=max(r["reproduction_max_rel"] for r in rows),
        G7_reproduction_ok=max(r["reproduction_max_rel"] for r in rows) <= 1e-6,
        o1=[dict(seed=r["seed"], case=r["case"], **r["o1_vs_repaired"]) for r in rows if "o1_vs_repaired" in r],
        stage_g_pre_d25=pre, stage_g_repaired=rep, mean_formal_sigma_change=sig,
        per_row=[dict(seed=r["seed"], case=r["case"], reproduction_max_rel=r["reproduction_max_rel"],
                      sigma_pre=np.sqrt(np.diag(r["P_pre"])).tolist(),
                      sigma_rep=np.sqrt(np.diag(r["P_rep"])).tolist(), elapsed_s=r["elapsed_s"]) for r in rows])


def run_invariance() -> dict:
    """G1/G2: pre-D25 estimator source (742576f, private namespace) vs the current module on cheap
    fixtures: estimates bitwise everywhere; posteriors bitwise where D25 changes nothing (defaults)."""
    import subprocess

    from lunar_od.srp import SRPOptions
    sys.path.insert(0, str(HERE.parents[2]))
    from tests import test_harden_d25_posterior_dynamics_contract as T
    T.setup_module(None)
    src = subprocess.run(["git", "show", "742576f:lunar_od/estimators.py"], capture_output=True, text=True,
                         cwd=HERE.parents[2], check=True).stdout
    old = dict(E.__dict__)
    exec(compile(src, E.__file__, "exec"), old)                 # production module untouched
    fx = T._build_fixture()
    dyn = T._dyn(fx)
    base = dict(max_iter=3, **T.TOL, return_posterior=True)
    srp = fx["srp"]
    cases = {
        "bls_six_default": ("estimate_two_way_range_bls_lm", dict()),
        "srif_six_default": ("estimate_two_way_range_srif", dict()),
        "bls_k_no_field": ("estimate_two_way_range_bls_lm", dict(srp=srp, solve_for_k_srp=True, k_srp_initial=0.01)),
        "srif_k": ("estimate_two_way_range_srif", dict(srp=srp, solve_for_k_srp=True, k_srp_initial=0.01,
                                                       k_srp_prior_sigma=0.01)),
        "bls_k_field": ("estimate_two_way_range_bls_lm", dict(srp=srp, solve_for_k_srp=True, k_srp_initial=0.01,
                                                              **fx["harm"])),
        "bls_six_field_srp": ("estimate_two_way_range_bls_lm", dict(srp=srp, prior_covariance=T.PRIOR, **fx["harm"])),
        "bls_six_srp": ("estimate_two_way_range_bls_lm", dict(srp=srp, prior_covariance=T.PRIOR)),
        "srif_six_srp": ("estimate_two_way_range_srif", dict(srp=srp, prior_covariance=T.PRIOR)),
    }
    changed_by_d25 = {"bls_k_field", "bls_six_field_srp", "bls_six_srp", "srif_six_srp"}
    out = {}
    for name, (fn, kw) in cases.items():
        res = []
        for ns in (old, E.__dict__):
            x, stop, st = ns[fn](fx["t_pass"], fx["obs"], fx["x_true0"], fx["pg"], *dyn, **base, **kw)
            res.append((np.asarray(x), stop, st))
        (xo, so, sto), (xn, sn, stn) = res
        est_same = (np.array_equal(xo, xn) and so == sn and sto.iterations == stn.iterations
                    and sto.final_cost == stn.final_cost and sto.k_srp_estimate == stn.k_srp_estimate)
        po, pn = np.asarray(sto.posterior_covariance), np.asarray(stn.posterior_covariance)
        out[name] = dict(estimate_bitwise_identical=bool(est_same), posterior_bitwise_identical=bool(np.array_equal(po, pn)),
                         posterior_expected_to_change=name in changed_by_d25,
                         max_sigma_rel_change=float(np.max(np.abs(np.sqrt(np.diag(pn)) / np.sqrt(np.diag(po)) - 1))))
    g1 = all(v["posterior_bitwise_identical"] for k, v in out.items() if k not in changed_by_d25)
    g2 = all(v["estimate_bitwise_identical"] for v in out.values())
    return dict(cases=out, G1_default_invariance=g1, G2_estimate_invariance=g2)


STAGE_I = STAGE_F.parent.parent / "09_checkpoints" / "stage_i_cubature.jsonl"


def _si_row(i):
    """Stage I point i: pre-D25 and repaired six-state posterior at its stored solution. The design depends
    on the observation-row geometry (times, stations), not on the noisy values; stage I's rows share the
    noise-free fit geometry, which the reproduction check of the stored P_hat verifies."""
    if not _SF:
        import od_gravity_covariance_campaign as C  # noqa: N811
        _SF["C"] = C
    C = _SF["C"]
    rows = [json.loads(line) for line in STAGE_I.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = [r for r in rows if r["status"] == "OK"][i]
    t0 = time.time()
    kw = C.CASES[row["case"]]
    x = np.asarray(row["x_est"], float)
    w = E._two_way_range_weight_diagonal(C.OBS_FIT, C.GEO_FIT)
    base = (C.T_FIT, C.OBS_FIT, x, C.GEO_FIT, C.MU, 0.0, 0.0, C.get_earth_pos, C.get_sun_pos,
            np.linalg.inv(C.P0), w, C.EST_RTOL, C.EST_ATOL, kw["j2_moon"])
    harm = dict(harmonic_model=kw["harmonic_model"], harmonic_epoch_et0=kw["harmonic_epoch_et0"])
    p_pre = np.linalg.inv(E._two_way_range_posterior_information(*base))
    p_rep = np.linalg.inv(E._two_way_range_posterior_information(*base, harmonic_kwargs=harm))
    p_pre, p_rep = 0.5 * (p_pre + p_pre.T), 0.5 * (p_rep + p_rep.T)
    stored = np.asarray(row["P_hat"], float)
    return dict(seed=row["seed"], e0=row["e0"], P_pre=p_pre.tolist(), P_rep=p_rep.tolist(),
                reproduction_max_rel=float(np.max(np.abs(p_pre - stored)) / np.max(np.abs(stored))),
                elapsed_s=time.time() - t0)


def _stage_i_stats(rows, key, weight):
    import scipy.linalg as sla
    e = np.array([r["e0"] for r in rows])
    c_raw = weight * (e.T @ e)
    c_raw = 0.5 * (c_raw + c_raw.T)
    p = np.array([r[key] for r in rows]).mean(axis=0)
    p = 0.5 * (p + p.T)
    nees_equiv = float(np.trace(np.linalg.solve(p, c_raw)))
    return dict(nees_equiv=nees_equiv, overconfidence=nees_equiv / 6.0,
                formal_sigma=np.sqrt(np.diag(p)).tolist(),
                sigma_ratio=(np.sqrt(np.diag(c_raw)) / np.sqrt(np.diag(p))).tolist(),
                formal_3d_position_sigma_m=float(np.sqrt(np.trace(p[:3, :3]))),
                generalized_eigenvalues=sla.eigh(c_raw, p, eigvals_only=True).tolist())


def run_stage_i(workers: int) -> dict:
    rows_all = [json.loads(line) for line in STAGE_I.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = sum(r["status"] == "OK" for r in rows_all)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_si_row, range(n)))
    w = 1.0 / n                                                   # stage I's equal cubature weight (1/2L)
    return dict(n_points=n, G7_reproduction_max_rel=max(r["reproduction_max_rel"] for r in rows),
                G7_reproduction_ok=max(r["reproduction_max_rel"] for r in rows) <= 1e-6,
                pre_d25=_stage_i_stats(rows, "P_pre", w), repaired=_stage_i_stats(rows, "P_rep", w),
                per_point=[dict(seed=r["seed"], reproduction_max_rel=r["reproduction_max_rel"],
                                elapsed_s=r["elapsed_s"]) for r in rows])


POSTHOC_V_STEPS = (1e-2, 3e-4)      # velocity-step study around the predeclared 1e-3 m/s


def run_kp2_posthoc_vel(workers: int) -> dict:
    """POST-HOC (not a gate): velocity-column FD step study. With K at 5e-3 (out of its noise floor) the
    velocity sigmas still differ by 0.25-0.33 % at the predeclared 1e-3 m/s step. Columns are rerun at
    1e-2 and 3e-4 m/s; FD columns and the production design are saved for column-level comparison."""
    tasks = (["production", "0", "1", "2", "K@0.005"]
             + ["V%d@%g" % (j, h) for h in POSTHOC_V_STEPS for j in (3, 4, 5)])
    with ProcessPoolExecutor(workers) as ex:
        res = {t: (o, s) for t, o, s in ex.map(_run_task, tasks)}
    prod = res["production"][0]
    c = _kp2_ctx()
    design = E._two_way_range_posterior_design_with_k(
        c["t"], c["obs"], c["x_hat"], c["k_hat"], c["srp"], c["pg"], *c["args"], c["tol"]["rtol"],
        c["tol"]["atol"], c["j2"], harmonic_kwargs=c["harm"])
    out = dict(label="POST-HOC velocity FD step study; predeclared G4 result unchanged", per_v_step={})
    cols = {}
    for h in POSTHOC_V_STEPS:
        fd = np.column_stack([res[str(j)][0][1] for j in range(3)] + [res["V%d@%g" % (j, h)][0][1] for j in (3, 4, 5)]
                             + [res["K@0.005"][0][1]])
        cols["%g" % h] = fd
        o2 = cov_from_design(fd, prod["w"], prod["prior_inv"])
        out["per_v_step"]["%g" % h] = dict(
            vs_repaired=compare(prod["repaired"], o2), vs_pre_d25=compare(prod["pre_d25"], o2),
            column_rel_vs_analytic=(np.linalg.norm(fd - design, axis=0) / np.linalg.norm(design, axis=0)).tolist())
    a, b = (cols["%g" % h] for h in POSTHOC_V_STEPS)
    out["velocity_column_step_consistency_rel"] = (np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=0)
                                                   / np.linalg.norm(a[:, 3:6], axis=0)).tolist()
    np.savez_compressed(P.ARTIFACTS / "phase17_d25_posthoc_fd_columns.npz", design=design,
                        **{"fd_v%g" % h: cols["%g" % h] for h in POSTHOC_V_STEPS})
    return out


def _save(key, value):
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(P.K._clean(data), indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp2", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--stage-f", action="store_true")
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--stage-i", action="store_true")
    ap.add_argument("--kp2-posthoc-fd", action="store_true")
    ap.add_argument("--kp2-posthoc-vel", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.invariance:
        out = run_invariance()
        _save("invariance", out)
        print(json.dumps(out, indent=1, default=str))
    if a.kp2_posthoc_vel:
        out = run_kp2_posthoc_vel(a.workers)
        _save("kp2_posthoc_velocity_step", out)
        print(json.dumps(out, indent=1, default=str))
    if a.kp2_posthoc_fd:
        out = run_kp2_posthoc_fd(a.workers)
        _save("kp2_posthoc_fd_step", out)
        print(json.dumps(out, indent=1, default=str))
    if a.stage_i:
        out = run_stage_i(a.workers)                    # separate file: runs beside chain A, no write race
        (P.ARTIFACTS / "phase17_d25_stage_i.json").write_text(json.dumps(P.K._clean(out), indent=1),
                                                             encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("n_points", "G7_reproduction_max_rel", "pre_d25", "repaired")},
                         indent=1, default=str))
    if a.stage_f:
        out = run_stage_f(a.workers, a.limit)
        _save("stage_f_blast_radius" if a.limit is None else "stage_f_smoke", out)
        print(json.dumps({k: out[k] for k in ("n_rows", "G7_reproduction_max_rel", "o1", "mean_formal_sigma_change")},
                         indent=1, default=str))
    if a.kp2:
        out = run_kp2()
        _save("kp2_k_path", out)
        print(json.dumps({k: out[k] for k in ("o1_vs_repaired", "o2_vs_repaired", "o2_vs_pre_d25", "kp2_solver",
                                              "gates")}, indent=1, default=str))
    if a.e2e:
        out = run_e2e()                                   # separate file: hours long, no write race
        (P.ARTIFACTS / "phase17_d25_e2e.json").write_text(json.dumps(P.K._clean(out), indent=1),
                                                         encoding="utf-8")
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
