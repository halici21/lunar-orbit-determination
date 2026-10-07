"""PHASE 17-KP2 - nonlinear M3 K_SRP recovery confirmation: campaign, verification, analysis.

Every rule is in artifacts/phase17_kp2_predeclaration.json (hash-pinned, committed before any
high-fidelity recovery result).  Counted-Doppler VALUES are used on the D23-qualified path.

    python examples/phase17_kp2_campaign.py            # all jobs + analysis
    python examples/phase17_kp2_campaign.py --analyse-only
"""
from __future__ import annotations

import argparse
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
import phase17_kp1_core as KP  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase17_r1o_cd1_campaign as K  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402

D2 = K.D2
ARTIFACTS = K2.ARTIFACTS
PREDECLARATION_SHA256 = "111856fb3adc01a5774dbde7984cd267d1182d7a748621d4b8aac332b3ff05f3"
HF = "HIGH_FIDELITY"
CELLS = tuple((o, k) for o in K.ORBITS for k in (0, 12))
TRUTH_NAMES = tuple(K2.TRUTHS)
M0_CELLS = (("CANON", 0), ("O5_FAVOURABLE_GEO_MAX_FPERP", 0))
RAW_PATH = ARTIFACTS / "phase17_kp2_raw_results.json"


def x0s() -> dict:
    g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    return {x["case_id"]: x["x0"] for x in g["geometries"] if x["case_id"] in K.ORBITS}


def _strip(sol: dict) -> dict:
    keep = {k: v for k, v in sol.items() if k not in ("cov", "history")}
    keep["cov_x0_k"] = (np.asarray(sol["cov"])[:6 + (3 if sol["model"] == "M3" else 1),
                                               :6 + (3 if sol["model"] == "M3" else 1)]).tolist()
    keep["history"] = sol["history"]
    return keep


# ======================================================================
# jobs
# ======================================================================
def job_main(spec: dict) -> dict:
    t0 = time.time()
    ci, orbit, k, tname = spec["ci"], spec["orbit"], spec["k"], spec["truth"]
    ti = TRUTH_NAMES.index(tname)
    x0 = np.asarray(spec["x0"], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    f_true = np.asarray(K2.TRUTHS[tname])
    X, PHI, S = K2.propagate_segmented(scn, x0, K2.K0 * (1 + f_true), HF)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, K.stations())
    rng = np.random.default_rng([ci, ti, 0])
    dx = rng.normal(0.0, K2.PRIOR_SD)
    nr = rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    b_true = rng.normal(0.0, K2.SIGMA_BIAS_TRUTH_MPS, fx.nuis.shape[1])
    nd = rng.normal(0.0, K2.SIGMA_DOPPLER_MPS, zd.size)
    z_r, z_d = zr + nr, zd + fx.nuis @ b_true + nd
    x0_ap = x0 + dx
    out = dict(kind="MAIN", ci=ci, orbit=orbit, k=k, truth=tname, n_range=int(zr.size), n_doppler=int(zd.size),
               n_passes=int(fx.nuis.shape[1]), dx_prior=dx.tolist())
    sol = K2.solve(fx, z_r, z_d, x0_ap, dyn=HF, model="M3", bias_mode="FREE")
    out["M3"] = dict(solve=_strip(sol), metrics=K2.recovery_metrics(sol, x0, f_true),
                     bias_err_rms=float(np.sqrt(np.mean((np.asarray(sol["theta"][9:]) - b_true) ** 2))))
    if tname == "T0_CONSTANT":
        out["kp1_anchor"] = kp1_anchor(fx, X, PHI, S, orbit, k)
        if (orbit, k) in M0_CELLS:
            s0 = K2.solve(fx, z_r, z_d, x0_ap, dyn=HF, model="M0", bias_mode="FREE")
            out["M0"] = dict(solve=_strip(s0), metrics=K2.recovery_metrics(s0, x0, f_true))
    if tname == "T1_STEP" and k == 0:
        sd0 = K2.solve(fx, z_r, zd + nd, x0_ap, dyn=HF, model="M3", bias_mode="NONE")
        out["D0"] = dict(solve=_strip(sd0), metrics=K2.recovery_metrics(sd0, x0, f_true))
    out["elapsed_s"] = time.time() - t0
    return K._clean(out)


def kp1_anchor(fx, X, PHI, S, orbit, k) -> dict:
    """KP1 linear geometry recomputed from the segmented design at the constant truth."""
    ev = K2.evaluate(fx, X, PHI, S)
    z = lambda n: np.zeros((n, 1))  # noqa: E731
    rbank = np.hstack([ev["hk_r"].sum(axis=1, keepdims=True), z(ev["hk_r"].shape[0]), ev["hk_r"][:, :2],
                       z(ev["hk_r"].shape[0])])
    dbank = np.hstack([ev["hk_d"].sum(axis=1, keepdims=True), z(ev["hk_d"].shape[0]), ev["hk_d"][:, :2],
                       z(ev["hk_d"].shape[0])])
    w_r = np.full(ev["hx_r"].shape[0], 1.0 / K2.SIGMA_RANGE_M ** 2)
    w_d = np.full(ev["hx_d"].shape[0], 1.0 / K2.SIGMA_DOPPLER_MPS ** 2)
    nu = CD.nuisance_columns(ev["hx_r"].shape[0], fx.rows, "D1_SESSION_CONSTANT_BIAS")
    c, p = KP.assemble([(ev["hx_r"], rbank, w_r), (ev["hx_d"], dbank, w_d)], nuisance=nu)
    m = KP.block_metrics(c, p, "M3_THREE_SEGMENT")
    kp1 = {(r["orbit"], r["k"], r["dynamics"]): r for r in json.loads(
        (ARTIFACTS / "phase17_kp1_raw_results.json").read_text(encoding="utf-8"))}[(orbit, k, HF)]
    ref = kp1["cases"]["RPD_D1FREE"]["M3_THREE_SEGMENT"]
    rel_sigma = max(abs(a - b) / b for a, b in zip(m["sigma_frac"], ref["sigma_frac"]))
    return dict(sigma_frac=m["sigma_frac"], kp1_sigma_frac=ref["sigma_frac"], s_var_min=m["s_var_min"],
                kp1_s_var_min=ref["s_var_min"], rel_sigma_max=rel_sigma,
                rel_s_var=abs(m["s_var_min"] - ref["s_var_min"]) / ref["s_var_min"])


def job_fd(spec: dict) -> dict:
    scn = K.scenario(spec["orbit"], spec["x0"], spec["k"], n_orbits=2.0)
    x0 = np.asarray(spec["x0"], float)
    ks = K2.K0 * (1 + np.asarray(K2.TRUTHS["T1_STEP"]))
    _, _, S = K2.propagate_segmented(scn, x0, ks, HF)
    errs = []
    for j in range(3):
        best = None
        for eps in (1e-3, 3e-3):
            kp, km = ks.copy(), ks.copy()
            kp[j] += eps
            km[j] -= eps
            fd = (K2.propagate_segmented(scn, x0, kp, HF)[0] - K2.propagate_segmented(scn, x0, km, HF)[0]) / (2 * eps)
            e = max(float(np.max(np.abs(fd[:, c] - S[j][:, c])) / (np.max(np.abs(S[j][:, c])) or 1.0))
                    for c in range(6))
            best = e if best is None else min(best, e)
        errs.append(best)
    return dict(kind="FD", orbit=spec["orbit"], k=spec["k"], best_rel_err_per_segment=errs, worst=max(errs))


def job_noise_free(spec: dict) -> dict:
    x0 = np.asarray(spec["x0"], float)
    scn = K.scenario("CANON", x0.tolist(), 0)
    f_true = np.asarray(K2.TRUTHS["T1_STEP"])
    X, PHI, S = K2.propagate_segmented(scn, x0, K2.K0 * (1 + f_true), HF)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, K.stations())
    dx = np.random.default_rng([99, 1, 0]).normal(0.0, K2.PRIOR_SD)
    sol = K2.solve(fx, zr, zd, x0 + dx, dyn=HF, model="M3", bias_mode="FREE")
    th, sd = np.asarray(sol["theta"]), np.asarray(sol["sd"])
    f_err = (th[6:9] / K2.K0 - 1 - f_true) / (sd[6:9] / K2.K0)
    x_err = (th[:6] - x0) / sd[:6]
    return K._clean(dict(kind="NOISE_FREE", converged=sol["converged"], n_eval=sol["n_eval"],
                         f_err_over_sigma=f_err.tolist(), x0_err_over_sigma=x_err.tolist(),
                         bias_max_abs=float(np.max(np.abs(th[9:])))))


def job_bls(spec: dict) -> dict:
    """KP2 solver (M0, range only, no prior) vs production estimate_two_way_range_bls_lm."""
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from lunar_od.estimators import estimate_two_way_range_bls_lm
    from lunar_od.srp import SRPOptions
    from phase17_r1od2_fidelity import NMAX
    x0 = np.asarray(spec["x0"], float)
    scn = K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), HF)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, K.stations())
    rng = np.random.default_rng([98, 0, 0])
    dx = rng.normal(0.0, K2.PRIOR_SD)
    z_r = zr + rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    mine = K2.solve(fx, z_r, zd, x0 + dx, dyn=HF, model="M0", bias_mode="NONE", use_doppler=False, prior_sd=None)
    obs = np.array(fx.rb.obs, float, copy=True)
    obs[:, 1] = z_r
    x_best, status, stats = estimate_two_way_range_bls_lm(
        scn.t_grid, obs, x0 + dx, fx.rb.pass_geo, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        lambda t: G.earth_state_mci(scn.et0 + float(t))[:3], lambda t: G.sun_mci(scn.et0 + float(t)),
        rtol=1e-12, atol=1e-13, j2_moon=0.0, harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0,
        srp=SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model), solve_for_k_srp=True,
        k_srp_initial=K2.K0, return_posterior=True)
    prod = np.concatenate([np.asarray(x_best, float), [float(stats.k_srp_estimate)]])
    pcov = np.asarray(stats.posterior_covariance, float)
    th, sd = np.asarray(mine["theta"]), np.asarray(mine["sd"])
    return K._clean(dict(kind="BLS", status=str(status), mine_converged=mine["converged"],
                         diff_over_sigma=((th - prod) / sd).tolist(),
                         sigma_ratio=(sd / np.sqrt(np.diag(pcov))).tolist(),
                         k_mine=float(th[6]), k_prod=float(prod[6])))


def run_job(spec: dict) -> dict:
    t0 = time.time()
    fn = {"MAIN": job_main, "FD": job_fd, "NOISE_FREE": job_noise_free, "BLS": job_bls}[spec["kind"]]
    try:
        out = fn(spec)
    except Exception as exc:                         # recorded, never hidden
        out = dict(kind=spec["kind"], error=repr(exc)[:800], spec={k: v for k, v in spec.items() if k != "x0"})
    out["job_elapsed_s"] = time.time() - t0
    return out


# ======================================================================
# analysis (predeclared rules)
# ======================================================================
def analyse(pre: dict, results: list) -> dict:
    main = [r for r in results if r.get("kind") == "MAIN" and "M3" in r]
    errors = [r for r in results if "error" in r]
    kp1 = {(r["orbit"], r["k"]): r["cases"]["RPD_D1FREE"]["M3_THREE_SEGMENT"]["s_var_min"]
           for r in json.loads((ARTIFACTS / "phase17_kp1_raw_results.json").read_text(encoding="utf-8"))
           if r["dynamics"] == HF}
    lin_sep = {c: kp1[c] >= 0.35 for c in CELLS}
    chi = K2.CHI2_2_999

    def rec(m):
        return bool(m["converged"] and m["chi2_var_err"] <= chi and m["chi2_var_detect"] >= chi)

    var = [r for r in main if r["truth"] != "T0_CONSTANT"]
    sep_cells = [c for c in CELLS if lin_sep[c]]
    pool = [r for r in var if lin_sep[(r["orbit"], r["k"])]] if len(sep_cells) >= 3 else var
    weak = [r for r in var if not lin_sep[(r["orbit"], r["k"])]]
    r_sep = float(np.mean([rec(r["M3"]["metrics"]) for r in pool])) if pool else float("nan")
    t0s = [r for r in main if r["truth"] == "T0_CONSTANT"]
    fp = float(np.mean([r["M3"]["metrics"]["chi2_var_detect"] >= chi for r in t0s])) if t0s else float("nan")
    conv = [r for r in main if r["M3"]["metrics"]["converged"]]
    cons = float(np.mean([r["M3"]["metrics"]["chi2_var_err"] <= chi for r in conv])) if conv else float("nan")
    nees = float(np.mean([r["M3"]["metrics"]["nees9"] <= K2.CHI2_9_999 for r in conv])) if conv else float("nan")
    if r_sep >= 0.80 and fp <= 0.20 and cons >= 0.80:
        primary = "M3_VARIATION_NONLINEARLY_RECOVERABLE"
    elif r_sep >= 0.30:
        primary = "M3_VARIATION_PARTIALLY_RECOVERABLE"
    else:
        primary = "M3_LINEAR_SEPARATION_DOES_NOT_TRANSLATE_TO_NONLINEAR_RECOVERY"
    consistency = ("CONSISTENT" if cons >= 0.9 and nees >= 0.9 else
                   "MARGINAL" if cons >= 0.7 and nees >= 0.7 else "INCONSISTENT")
    d0 = [r for r in main if "D0" in r]
    ratio = [float(np.median(np.asarray(r["M3"]["metrics"]["v_sd"]) / np.asarray(r["D0"]["metrics"]["v_sd"])))
             for r in d0]
    rec_d1 = sum(rec(r["M3"]["metrics"]) for r in d0)
    rec_d0 = sum(rec(r["D0"]["metrics"]) for r in d0)
    bias = dict(n=len(d0), median_sigma_v_ratio_D1_over_D0=float(np.median(ratio)) if ratio else None,
                recovered_D1=rec_d1, recovered_D0=rec_d0,
                status=("MATERIAL" if ratio and (np.median(ratio) >= 1.5 or rec_d1 < rec_d0) else "MINOR"))
    level = dict(level_z=[r["M3"]["metrics"]["level_z"] for r in main],
                 level_sd_median=float(np.median([r["M3"]["metrics"]["level_sd"] for r in main])),
                 status="NOT_CLAIMED")
    anchors = [r["kp1_anchor"] for r in t0s if "kp1_anchor" in r]
    fd = [r for r in results if r.get("kind") == "FD" and "worst" in r]
    nf = next((r for r in results if r.get("kind") == "NOISE_FREE" and "converged" in r), None)
    bls = next((r for r in results if r.get("kind") == "BLS" and "diff_over_sigma" in r), None)
    cd1 = {(r["orbit"], int(r["k"]), r["dynamics"]): r for r in __import__("csv").DictReader(
        (ARTIFACTS / "phase17_r1o_cd1_primary_results.csv").open())}
    m0 = [dict(orbit=r["orbit"], k=r["k"], k_frac_sd=r["M0"]["metrics"]["k_frac_sd"],
               cd1_sigma=float(cd1[(r["orbit"], r["k"], HF)]["sigma_k_frac_D1FREE"]),
               k_frac_err=r["M0"]["metrics"]["k_frac_err"]) for r in main if "M0" in r]
    gates = dict(
        FD="PASS" if fd and max(r["worst"] for r in fd) <= 1e-4 else "FAIL",
        KP1_ANCHOR="PASS" if anchors and max(max(a["rel_sigma_max"], a["rel_s_var"]) for a in anchors) <= 1e-5
        else "FAIL",
        NOISE_FREE="PASS" if nf and nf["converged"] and max(map(abs, nf["f_err_over_sigma"])) <= 0.02
        and max(map(abs, nf["x0_err_over_sigma"])) <= 0.02 else "FAIL",
        PRODUCTION_BLS="PASS" if bls and max(map(abs, bls["diff_over_sigma"])) <= 0.02
        and max(abs(x - 1) for x in bls["sigma_ratio"]) <= 0.01 else "FAIL",
        M0_BASELINE="PASS" if m0 and all(abs(x["k_frac_sd"] / x["cd1_sigma"] - 1) <= 0.05 for x in m0) else "FAIL",
        NO_JOB_ERRORS="PASS" if not errors else "FAIL")
    trials = [dict(orbit=r["orbit"], k=r["k"], truth=r["truth"], lin_sep=lin_sep[(r["orbit"], r["k"])],
                   kp1_s_var_min=kp1[(r["orbit"], r["k"])], recovered=rec(r["M3"]["metrics"]),
                   **{key: r["M3"]["metrics"][key] for key in ("converged", "n_eval", "chi2_var_err",
                                                               "chi2_var_detect", "var_rel_err", "level_z",
                                                               "nees9", "max_abs_segment_corr")},
                   v_hat=r["M3"]["metrics"]["v_hat"], v_true=r["M3"]["metrics"]["v_true"],
                   v_sd=r["M3"]["metrics"]["v_sd"], f_hat=r["M3"]["metrics"]["f_hat"],
                   f_sd=r["M3"]["metrics"]["f_sd"], range_wrms=r["M3"]["solve"]["range_wrms"],
                   doppler_wrms=r["M3"]["solve"]["doppler_wrms"], bias_err_rms=r["M3"]["bias_err_rms"])
              for r in main]
    out = dict(phase="17-KP2", predeclaration_sha256=PREDECLARATION_SHA256, truth_level="SYNTHETIC_CAMPAIGN_TRUTH",
               PRIMARY_CLASS=primary, R_sep=r_sep, n_pool=len(pool), lin_sep_cells=[list(c) for c in sep_cells],
               R_weak=float(np.mean([rec(r["M3"]["metrics"]) for r in weak])) if weak else None, n_weak=len(weak),
               FP=fp, CONS=cons, NEES_frac=nees, NONLINEAR_ESTIMATOR_CONSISTENCY_STATUS=consistency,
               DOPPLER_BIAS_IMPACT=bias, COMMON_LEVEL=level, m0_baseline=m0, anchors=anchors, fd=fd,
               noise_free=nf, production_bls=bls, gates=gates, job_errors=errors, trials=trials)
    (ARTIFACTS / "phase17_kp2_summary.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--workers", type=int, default=K.N_WORKERS)
    args = ap.parse_args()
    pre = K2.load_predeclaration(PREDECLARATION_SHA256)
    if args.analyse_only:
        results = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        xs = x0s()
        specs = [dict(kind="MAIN", ci=ci, orbit=o, k=k, truth=t, x0=xs[o])
                 for ci, (o, k) in enumerate(CELLS) for t in TRUTH_NAMES]
        specs.sort(key=lambda s: 0 if (s["truth"] == "T1_STEP" and s["k"] == 0)
                   or (s["truth"] == "T0_CONSTANT" and (s["orbit"], s["k"]) in M0_CELLS) else 1)
        specs += [dict(kind="BLS", x0=xs["CANON"]), dict(kind="NOISE_FREE", x0=xs["CANON"])]
        specs += [dict(kind="FD", orbit=o, k=k, x0=xs[o]) for o, k in (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12))]
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=K._init) as ex:
            for i, r in enumerate(ex.map(run_job, specs)):
                results.append(r)
                print("%2d/%d %s %s %s %s %.0f s%s" % (i + 1, len(specs), r.get("kind"), r.get("orbit", ""),
                                                      r.get("k", ""), r.get("truth", ""), time.time() - t0,
                                                      " ERROR" if "error" in r else ""), flush=True)
                RAW_PATH.write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
    out = analyse(pre, results)
    print(out["PRIMARY_CLASS"], "R_sep %.3f FP %.3f CONS %.3f" % (out["R_sep"], out["FP"], out["CONS"]), out["gates"])


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
