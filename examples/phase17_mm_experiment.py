"""Phase 17-MM: gravity-field model-mismatch masquerade falsification (K-closeout report section 3.4).

Design, thresholds and outcome rule are frozen in artifacts/phase17_mm_predeclaration.json.

  --run [--workers N]   4 baseline re-evaluations (truth = model, stored KP2 T0 solutions) + 8 mismatch solves;
                        resume-safe: one JSON per job in artifacts/phase17_mm_jobs/
  --analyse             apply the predeclared rule -> artifacts/phase17_mm_summary.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

ART = P.ARTIFACTS
JOBS = ART / "phase17_mm_jobs"
PRE = ART / "phase17_mm_predeclaration.json"
CELLS = (("CANON", 0, 0), ("O1_LOW_BETA_ECLIPSE_RICH", 12, 3), ("O2_FULL_SUN_HIGH_BETA", 0, 4),
         ("O5_FAVOURABLE_GEO_MAX_FPERP", 0, 8))
ALPHAS = (1.0, 0.3)
HF = "HIGH_FIDELITY"
GAP_S = 600.0


# ----------------------------------------------------------------------
# truth field
# ----------------------------------------------------------------------
def mixed_field(alpha: float, gl, gr):
    """GL + alpha (GR - GL) in GL's (mu, R) normalisation, degrees n >= 2 (the engine's perturbing field)."""
    n = gl.cbar.shape[0]
    deg = np.arange(n)[:, None]
    s = (gr.mu_m3_s2 / gl.mu_m3_s2) * (gr.r_ref_m / gl.r_ref_m) ** deg
    c_gr, s_gr = gr.cbar[:n, :n] * s, gr.sbar[:n, :n] * s
    cbar, sbar = gl.cbar.copy(), gl.sbar.copy()
    cbar[2:] = gl.cbar[2:] + alpha * (c_gr[2:] - gl.cbar[2:])
    sbar[2:] = gl.sbar[2:] + alpha * (s_gr[2:] - gl.sbar[2:])
    meta = dict(getattr(gl, "metadata", {}) or {})
    meta["phase17_mm"] = f"GL1800F + {alpha} (GRGM1200L - GL1800F), rescaled to GL mu/R"
    return dataclasses.replace(gl, cbar=cbar, sbar=sbar, metadata=meta)


def _fields():
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    gl = C._M[50]
    gr = load_lunar_gravity_model(C.GL1800F.replace("gl1800f/jggrx_1800f_sha.tab", "grgm1200l/gggrx_1200l_sha.tab"),
                                  nmax=50, mmax=50)
    return gl, gr


def _truth_segment(model):
    def prop(scn, t_slice, x_start, k, dyn):
        import phase17_geo_core as G
        from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
        from lunar_od.dynamics import propagate_state_with_k_sensitivity
        from lunar_od.srp import SRPOptions
        return propagate_state_with_k_sensitivity(
            t_slice, x_start, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
            lambda t: G.earth_state_mci(scn.et0 + float(t))[:3], lambda t: G.sun_mci(scn.et0 + float(t)),
            srp=SRPOptions(k_srp_m2_per_kg=float(k), shadow_model=scn.shadow_model), rtol=K2.HF_TOL[0],
            atol=K2.HF_TOL[1], j2_moon=0.0, harmonic_model=model, harmonic_epoch_et0=scn.et0,
            harmonic_stm_opt_in=True)
    return prop


# ----------------------------------------------------------------------
# one cell: truth rows, KP2 draws
# ----------------------------------------------------------------------
def _cell(orbit, k, ci, truth_model):
    P.K._init()
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = P.K.scenario(orbit, x0.tolist(), k)
    ks = np.full(3, K2.K0)
    if truth_model is None:
        X, PHI, S = K2.propagate_segmented(scn, x0, ks, HF)
    else:
        real = K2._prop_segment
        K2._prop_segment = _truth_segment(truth_model)
        try:
            X, PHI, S = K2.propagate_segmented(scn, x0, ks, HF)
        finally:
            K2._prop_segment = real
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rng = np.random.default_rng([ci, 0, 0])                      # KP2 job_main order, ti = 0 (T0_CONSTANT)
    dx = rng.normal(0.0, K2.PRIOR_SD)
    nr = rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    b_true = rng.normal(0.0, K2.SIGMA_BIAS_TRUTH_MPS, fx.nuis.shape[1])
    nd = rng.normal(0.0, K2.SIGMA_DOPPLER_MPS, zd.size)
    return dict(x0=x0, fx=fx, z_r=zr + nr, z_d=zd + fx.nuis @ b_true + nd, x0_ap=x0 + dx, b_true=b_true)


def _residuals_at(c, theta):
    """Whitened data residuals (range, Doppler) of the estimator model at theta, KP2's own algebra."""
    th = np.asarray(theta, float)
    X, PHI, S = K2.propagate_segmented(c["fx"].scn, th[:6], th[6:9], HF)
    ev = K2.evaluate(c["fx"], X, PHI, S)
    rr = (c["z_r"] - ev["yr"]) / K2.SIGMA_RANGE_M
    rd = (c["z_d"] - (ev["yd"] + c["fx"].nuis @ th[9:])) / K2.SIGMA_DOPPLER_MPS
    return rr, rd


def _passes(c):
    fx = c["fx"]
    d_pass = np.argmax(np.asarray(fx.nuis), axis=1)
    obs = np.asarray(fx.rb.obs, float)
    r_pass, cur, last = np.zeros(obs.shape[0], int), {}, {}
    nxt = 0
    for i in range(obs.shape[0]):
        st, t = int(obs[i, 2]), float(obs[i, 0])
        if st not in cur or t - last[st] > GAP_S:
            cur[st] = nxt
            nxt += 1
        r_pass[i], last[st] = cur[st], t
    return r_pass, d_pass


def _runs_z(res, groups):
    zs = []
    for g in np.unique(groups):
        s = np.sign(res[groups == g])
        s = s[s != 0]
        if s.size < 10:
            continue
        n1, n2 = int(np.sum(s > 0)), int(np.sum(s < 0))
        if n1 == 0 or n2 == 0:
            zs.append(-math.sqrt(s.size))                      # one sign throughout: maximal clustering
            continue
        runs = 1 + int(np.sum(s[1:] != s[:-1]))
        n = n1 + n2
        mu = 2 * n1 * n2 / n + 1
        var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
        zs.append((runs - mu) / math.sqrt(var))
    return dict(n_passes=len(zs), stouffer_z=float(np.sum(zs) / math.sqrt(len(zs))) if zs else float("nan"),
                n_passes_abs_z_gt_3=int(np.sum(np.abs(zs) > 3)))


def _diagnostics(c, rr, rd):
    rp, dp = _passes(c)
    return dict(chi2_data=float(rr @ rr + rd @ rd), n_range=int(rr.size), n_doppler=int(rd.size),
                range_wrms=float(np.sqrt(np.mean(rr ** 2))), doppler_wrms=float(np.sqrt(np.mean(rd ** 2))),
                runs_range=_runs_z(rr, rp), runs_doppler=_runs_z(rd, dp))


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------
def job(spec):
    t0 = time.time()
    orbit, k, ci, alpha = spec["orbit"], spec["k"], spec["ci"], spec["alpha"]
    if alpha is None:                                            # baseline: truth = model, stored KP2 solution
        c = _cell(orbit, k, ci, None)
        raw = json.loads((ART / "phase17_kp2_raw_results.json").read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("results", raw)
        kp2 = next(r for r in rows if r.get("kind") == "MAIN" and r["ci"] == ci and r["truth"] == "T0_CONSTANT")
        sol = kp2["M3"]["solve"]
        rr, rd = _residuals_at(c, sol["theta"])
        pr = (c["x0_ap"] - np.asarray(sol["theta"][:6])) / K2.PRIOR_SD
        cost = float(rr @ rr + rd @ rd + pr @ pr)
        out = dict(kind="BASELINE", **_diagnostics(c, rr, rd), stored_cost=sol["cost"], recomputed_cost=cost,
                   harness_cost_rel=abs(cost / sol["cost"] - 1), metrics=kp2["M3"]["metrics"])
    else:
        gl, gr = _fields()
        c = _cell(orbit, k, ci, mixed_field(alpha, gl, gr))
        cap = []
        real = K2._gauss_newton

        def spy(*a, **kw):
            res = real(*a, **kw)
            cap.append(res[0])
            return res
        K2._gauss_newton = spy
        try:
            sol = K2.solve(c["fx"], c["z_r"], c["z_d"], c["x0_ap"], dyn=HF, model="M3", bias_mode="FREE")
        finally:
            K2._gauss_newton = real
        r = np.asarray(cap[-1]["r"], float)
        rr, rd = r[:len(c["z_r"])], r[len(c["z_r"]):]
        th, sd = np.asarray(sol["theta"]), np.asarray(sol["sd"])
        metrics = K2.recovery_metrics(sol, c["x0"], np.zeros(3))
        out = dict(kind="MISMATCH", **_diagnostics(c, rr, rd), converged=sol["converged"], n_eval=sol["n_eval"],
                   theta=th.tolist(), sd=sd.tolist(), metrics=metrics,
                   x0_err_over_sigma=((th[:6] - c["x0"]) / sd[:6]).tolist(),
                   bias_err_rms=float(np.sqrt(np.mean((th[9:] - c["b_true"]) ** 2))))
    out.update(orbit=orbit, k=k, ci=ci, alpha=alpha, elapsed_s=time.time() - t0)
    return P.K._clean(out)


def _name(s):
    return f"{s['orbit']}_k{s['k']}_" + ("baseline" if s["alpha"] is None else f"alpha{s['alpha']}")


def _run_one(spec):
    path = JOBS / (_name(spec) + ".json")
    if path.exists():
        return _name(spec), "cached"
    try:
        out = job(spec)
    except Exception as exc:                                     # recorded, never hidden
        out = dict(error=repr(exc)[:800], **spec)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return _name(spec), "error" if "error" in out else "ok"


def run(workers: int):
    JOBS.mkdir(exist_ok=True)
    specs = [dict(orbit=o, k=k, ci=ci, alpha=None) for o, k, ci in CELLS]
    specs += [dict(orbit=o, k=k, ci=ci, alpha=a) for a in ALPHAS for o, k, ci in CELLS]
    with ProcessPoolExecutor(workers) as ex:
        for name, status in ex.map(_run_one, specs):
            print(name, status, flush=True)


def analyse() -> dict:
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    base = {}
    rows = []
    for o, k, ci in CELLS:
        base[(o, k)] = json.loads((JOBS / f"{o}_k{k}_baseline.json").read_text(encoding="utf-8"))
    for a in ALPHAS:
        for o, k, ci in CELLS:
            m = json.loads((JOBS / f"{o}_k{k}_alpha{a}.json").read_text(encoding="utf-8"))
            b = base[(o, k)]
            n = m["n_range"] + m["n_doppler"]
            dchi2 = m["chi2_data"] - b["chi2_data"]
            thr = 2 * math.sqrt(2 * n)
            det = m["metrics"]["chi2_var_detect"]
            rows.append(dict(orbit=o, k=k, alpha=a, converged=m["converged"], chi2_detect=det,
                             detect=det >= 13.82, delta_chi2=dchi2, delta_chi2_threshold=thr,
                             residual_undetectable=dchi2 <= thr,
                             masquerade=bool(m["converged"] and det >= 13.82 and dchi2 <= thr),
                             runs_range_z=m["runs_range"]["stouffer_z"], runs_doppler_z=m["runs_doppler"]["stouffer_z"],
                             runs_blind=bool(abs(m["runs_range"]["stouffer_z"]) <= 3
                                             and abs(m["runs_doppler"]["stouffer_z"]) <= 3),
                             range_wrms=m["range_wrms"], doppler_wrms=m["doppler_wrms"],
                             range_wrms_base=b["range_wrms"], doppler_wrms_base=b["doppler_wrms"],
                             level_err=m["metrics"]["level_err"], level_z=m["metrics"].get("level_z"),
                             x0_err_over_sigma_max=float(np.max(np.abs(m["x0_err_over_sigma"]))),
                             var_rel_err=m["metrics"].get("var_rel_err")))
    decisive_unconverged = [r for r in rows if not r["converged"]]
    if any(r["masquerade"] for r in rows):
        outcome = "H_MM_SURVIVES_MASQUERADE"
    elif decisive_unconverged:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "H_MM_FALSIFIED"
    harness = max(b["harness_cost_rel"] for b in base.values())
    return dict(predeclaration=pre["phase"], outcome=outcome, rows=rows,
                harness_max_cost_rel=harness, harness_ok=harness <= 1e-6,
                baselines={f"{o}_k{k}": dict(chi2_data=b["chi2_data"], range_wrms=b["range_wrms"],
                                              doppler_wrms=b["doppler_wrms"], runs_range=b["runs_range"],
                                              runs_doppler=b["runs_doppler"]) for (o, k), b in base.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.workers)
    if a.analyse:
        out = analyse()
        (ART / "phase17_mm_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "baselines"}, indent=1))


if __name__ == "__main__":
    main()
