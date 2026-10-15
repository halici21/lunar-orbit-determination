"""PHASE 17-KGM - final K closeout: bounded stochastic (Gauss-Markov) K_SRP experiment.

    python studies/phase17/scripts/phase17_kgm_campaign.py --predeclare     # write + hash the predeclaration (commit it first)
    python studies/phase17/scripts/phase17_kgm_campaign.py --anchor         # GM-core vs KP2-core construction anchor (simplified)
    python studies/phase17/scripts/phase17_kgm_campaign.py --run            # the declared high-fidelity jobs
    python studies/phase17/scripts/phase17_kgm_campaign.py --analyse        # classification from the raw results

Every rule is in studies/phase17/configs/phase17_kgm_predeclaration.json.  M3 results on T0/T1 are READ from the
frozen KP2 raw results (identical data: same cell, truth, seeds); nothing in KP2 is re-run or changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kgm_core as G  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

K = P.K
ART = K2.ARTIFACTS
PRE = ART.parent / "configs" / "phase17_kgm_predeclaration.json"
RAW = ART / "phase17_kgm_raw_results.json"
SUMMARY = ART / "phase17_kgm_summary.json"
HF = P.HF

PREDECLARATION = dict(
    phase="PHASE17_KGM_FINAL_K_CLOSEOUT",
    question=("Does a first-order Gauss-Markov K representation recover the time-variation information KP2 "
              "demonstrated with M3, without manufacturing variation or resting on the prior?"),
    truth_level="SYNTHETIC_CAMPAIGN_TRUTH",
    model=dict(
        name="GM15", n_segments=15, segment="one per orbit of the 15-orbit KP2 arc (equal durations)",
        parameterisation="K_j = K0 (1 + f_j); f = m 1 + delta; delta stationary first-order GM; m flat (no prior)",
        prior_information="Lambda = C^-1 - C^-1 11^T C^-1 / (1^T C^-1 1), C_ij = sigma^2 phi^|i-j|, rank 14",
        primary=dict(sigma=0.30, tau_segments=5.0),
        hyperparameter_rules=dict(
            tau=("tau = T_arc / 3 = 5 orbits: the resolution at which KP2/M3 asserted variation information; "
                 "the like-for-like comparison of the two representations"),
            sigma=("sigma = 0.30: >= 3.7x every KP2 posterior M3 sigma_v (max 0.081, O2 k0), so the prior is "
                   "diffuse wherever KP2 found information; chosen from KP2 POSTERIORS, not from the truths"),
        ),
        sensitivity=dict(rule="declared now, run on CANON k0 T0 and T1 only; no selection among them",
                         tau_segments=[1.0, 15.0]),
    ),
    estimator=("KP2 staged batch Gauss-Newton (stage 1: x0 + biases at K0, 4 evals; stage 2: all free, 10 evals, "
               "every step < 1e-2 sigma), production range + counted-Doppler values/Jacobians, one free bias per "
               "Doppler pass, a priori N(0, diag(100 m, 0.1 m/s)^2), GM rows appended to the K block"),
    cells=[["CANON", 0], ["O2_FULL_SUN_HIGH_BETA", 0], ["O1_LOW_BETA_ECLIPSE_RICH", 12]],
    cell_rationale=("one cell per KP2 behaviour: CANON k0 strong recovery; O2 k0 information-limited miss; "
                    "O1 k12 marginal (T2 missed at chi2 12.4, T1 recovered)"),
    truths=dict(
        T0_CONSTANT="f = 0 (15 segments)",
        T1_STEP="KP2 T1 (+0.10, -0.10, 0) on thirds, i.e. 5 segments each",
        T3_GM=dict(rule="one GM realisation, sample mean removed (level K0), sigma 0.10, tau 2 orbits "
                        "(deliberately NOT the model tau)", seed=[17, 3, 0]),
    ),
    seeds="per (cell, truth): rng([KP2 cell index, truth index, 0]) with truth index T0=0, T1=1, T3_GM=3 "
          "(T0/T1: same truth K(t), same draws as KP2; the truth is propagated with 15 instead of 3 segment "
          "restarts, which differ only at integrator-noise level)",
    m3_comparison=("M3 (KP2 solver, unchanged) on T0/T1 read from studies/phase17/results/phase17_kp2_raw_results.json; "
                   "M3 on T3_GM solved here on the same data"),
    metrics=dict(
        recovered_m3_projection=("converged and chi2_err <= 13.82 and chi2_detect >= 13.82 on the 2-dof M3 "
                                 "variation of the GM estimate averaged onto thirds (the KP2 rule)"),
        false_variation=("T0: chi2_detect >= 13.82 (M3 projection) or full 14-dof chi2_detect >= chi2_14(0.999) "
                         "= 36.12"),
        full_resolution=("full 14-dof chi2_err / chi2_detect vs chi2_14(0.999); reconstruction skill "
                         "1 - rms(var err)/rms(var truth) on T3_GM"),
        prior_share=("DFS = 14 - tr(Lambda_u C_post,u); PRIOR_DOMINATED if DFS/14 < 0.5 in a cell where GM "
                     "reports recovery; sigma ratio GM/M3 on the M3 projection (< 0.8 flags prior information)"),
        common_level="z of the mean deviation, REPORTED ONLY; the level is not claimed",
    ),
    classification=dict(
        SUPPORTS_M3_VARIATION_RESULT=("GM recovers T1 in every cell where KP2/M3 did and not where M3 did not "
                                      "(agreement 3/3 cells), 0 false variations on T0, not prior-dominated"),
        PARTIALLY_SUPPORTS_M3_VARIATION_RESULT=("agreement in 2/3 cells, <= 1 false variation, or a "
                                                "prior-influence flag"),
        DOES_NOT_ADD_RELIABLE_INFORMATION=("false variation in >= 2 cells, or every GM recovery prior-dominated, or "
                                           "agreement <= 1/3"),
        INCONCLUSIVE="fewer than 2 of 3 primary T1 solves converge",
        T3_GM_role=("reported: full-resolution reconstruction and whether GM or M3 better represents a variation "
                    "that M3 cannot represent exactly; not part of the class"),
    ),
    budget="9 primary GM15 solves + 3 M3 solves on T3_GM + 2 sensitivity solves; no Monte Carlo, no grid",
)


def write_predeclaration():
    d = dict(PREDECLARATION)
    d["canonical_sha256"] = K2.canonical_sha256(d)
    PRE.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(d["canonical_sha256"])


def _pre():
    d = json.loads(PRE.read_text(encoding="utf-8"))
    if K2.canonical_sha256(d) != d["canonical_sha256"]:
        raise RuntimeError("KGM predeclaration hash mismatch")
    return d


def truth_f(name: str) -> np.ndarray:
    pre = _pre()
    if name == "T0_CONSTANT":
        return np.zeros(G.N_SEG)
    if name == "T1_STEP":
        return np.repeat(np.asarray(K2.TRUTHS["T1_STEP"]), G.N_SEG // 3)
    t3 = pre["truths"]["T3_GM"]
    return G.sample_gm_truth(G.N_SEG, 0.10, 2.0, t3["seed"])


TRUTH_INDEX = {"T0_CONSTANT": 0, "T1_STEP": 1, "T3_GM": 3}


def _data(orbit, k, tname):
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    f_true = truth_f(tname)
    X, PHI, S = G.propagate_segmented(scn, x0, K2.K0 * (1 + f_true), HF)
    fx, zr, zd = K2.build_fixed(scn, X, PHI, S, K.stations())
    ci = P.CELLS.index((orbit, k))
    rng = np.random.default_rng([ci, TRUTH_INDEX[tname], 0])        # KP2 draw order
    dx = rng.normal(0.0, K2.PRIOR_SD)
    nr = rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    b_true = rng.normal(0.0, K2.SIGMA_BIAS_TRUTH_MPS, fx.nuis.shape[1])
    nd = rng.normal(0.0, K2.SIGMA_DOPPLER_MPS, zd.size)
    return x0, f_true, fx, zr + nr, zd + fx.nuis @ b_true + nd, x0 + dx, dx


def job(spec: dict) -> dict:
    t0 = time.time()
    K._init()
    orbit, k, tname, model = spec["orbit"], spec["k"], spec["truth"], spec["model"]
    x0, f_true, fx, z_r, z_d, x0_ap, dx = _data(orbit, k, tname)
    out = dict(orbit=orbit, k=k, truth=tname, model=model, tau_segments=spec.get("tau"), dx_prior=dx.tolist())
    if model == "GM15":
        sigma = _pre()["model"]["primary"]["sigma"]
        sol = G.solve(fx, z_r, z_d, x0_ap, dyn=HF, gm_sigma=sigma, gm_tau_segments=spec["tau"])
        out["metrics"] = G.metrics(sol, x0, f_true, sigma, spec["tau"])
    else:                                                               # M3, KP2 solver unchanged
        sol = K2.solve(fx, z_r, z_d, x0_ap, dyn=HF, model="M3", bias_mode="FREE")
        f3 = G.thirds_matrix(G.N_SEG) @ f_true
        out["metrics"] = K2.recovery_metrics(sol, x0, f3)
        out["m3_truth_projection"] = f3.tolist()
    out["solve"] = {k_: v for k_, v in sol.items() if k_ not in ("cov",)}
    out["elapsed_s"] = time.time() - t0
    return K._clean(out)


def specs():
    pre = _pre()
    tau = pre["model"]["primary"]["tau_segments"]
    out = []
    for orbit, k in pre["cells"]:
        for tname in ("T0_CONSTANT", "T1_STEP", "T3_GM"):
            out.append(dict(orbit=orbit, k=k, truth=tname, model="GM15", tau=tau))
        out.append(dict(orbit=orbit, k=k, truth="T3_GM", model="M3"))
    for tau_s in pre["model"]["sensitivity"]["tau_segments"]:
        for tname in ("T0_CONSTANT", "T1_STEP"):
            out.append(dict(orbit="CANON", k=0, truth=tname, model="GM15", tau=tau_s))
    return out


def anchor() -> dict:
    """Construction anchor in SIMPLIFIED dynamics: the N-segment core at N = 3 must reproduce the KP2 core."""
    K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = K.scenario("CANON", x0.tolist(), 0, n_orbits=3.0)
    ks = K2.K0 * (1 + np.asarray(K2.TRUTHS["T2_MIDDLE"]))
    a = K2.propagate_segmented(scn, x0, ks, "SIMPLIFIED")
    b = G.propagate_segmented(scn, x0, ks, "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(scn, *a, K.stations())
    ea, eb = K2.evaluate(fx, *a), G.evaluate(fx, *b)
    prop = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    ev = max(float(np.max(np.abs(np.asarray(ea[key]) - np.asarray(eb[key])))) for key in ea)
    # 15-segment model at a constant K: sum of K columns equals the scalar-K column of a 1-segment run
    t0 = time.time()
    c15 = G.propagate_segmented(scn, x0, np.full(15, K2.K0), "SIMPLIFIED")
    c1 = G.propagate_segmented(scn, x0, np.full(1, K2.K0), "SIMPLIFIED")
    e15 = G.evaluate(fx, *c15)
    e1 = G.evaluate(fx, *c1)
    col_r = float(np.max(np.abs(e15["hk_r"].sum(axis=1) - e1["hk_r"][:, 0])) / np.max(np.abs(e1["hk_r"])))
    col_d = float(np.max(np.abs(e15["hk_d"].sum(axis=1) - e1["hk_d"][:, 0])) / np.max(np.abs(e1["hk_d"])))
    return dict(n3_propagation_max_abs_diff=prop, n3_evaluation_max_abs_diff=ev,
                n15_sum_of_k_columns_vs_scalar_rel=dict(range=col_r, doppler=col_d),
                n15_eval_elapsed_s=time.time() - t0)


def kp2_m3(orbit, k, tname) -> dict:
    for r in json.loads(P.RAW_PATH.read_text(encoding="utf-8")):
        if r.get("kind") == "MAIN" and r["orbit"] == orbit and r["k"] == k and r["truth"] == tname:
            return r["M3"]["metrics"]
    raise KeyError((orbit, k, tname))


def _recovered(m, key_err, key_det, thr=K2.CHI2_2_999):
    return bool(m["converged"] and m[key_err] <= thr and m[key_det] >= thr)


def analyse() -> dict:
    pre = _pre()
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    tau = pre["model"]["primary"]["tau_segments"]
    prim = {(r["orbit"], r["k"], r["truth"]): r for r in rows if r["model"] == "GM15" and r["tau_segments"] == tau}
    m3t3 = {(r["orbit"], r["k"]): r for r in rows if r["model"] == "M3"}
    cells, agree, false_var, prior_flags, conv_t1 = [], 0, 0, 0, 0
    for orbit, k in pre["cells"]:
        g0, g1, g3 = (prim[(orbit, k, t)]["metrics"] for t in ("T0_CONSTANT", "T1_STEP", "T3_GM"))
        m0, m1 = kp2_m3(orbit, k, "T0_CONSTANT"), kp2_m3(orbit, k, "T1_STEP")
        gm_rec = _recovered(g1, "m3_chi2_err", "m3_chi2_detect")
        m3_rec = _recovered(m1, "chi2_var_err", "chi2_var_detect")
        fv = bool(g0["m3_chi2_detect"] >= K2.CHI2_2_999 or g0["full_chi2_detect"] >= g0["full_chi2_999"])
        sd_ratio = float(np.mean(np.asarray(g1["m3_v_sd"]) / np.asarray(m1["v_sd"])))
        flag = bool((gm_rec and g1["dfs_fraction"] < 0.5) or sd_ratio < 0.8)
        agree += int(gm_rec == m3_rec)
        false_var += int(fv)
        prior_flags += int(flag)
        conv_t1 += int(g1["converged"])
        m3g = m3t3[(orbit, k)]["metrics"]
        cells.append(dict(
            cell=[orbit, k],
            T1=dict(gm_recovered=gm_rec, m3_recovered_kp2=m3_rec, gm_chi2_err=g1["m3_chi2_err"],
                    gm_chi2_detect=g1["m3_chi2_detect"], m3_chi2_err=m1["chi2_var_err"],
                    m3_chi2_detect=m1["chi2_var_detect"], gm_over_m3_sigma_v=sd_ratio,
                    dfs_fraction=g1["dfs_fraction"], converged=g1["converged"]),
            T0=dict(gm_false_variation=fv, gm_m3_chi2_detect=g0["m3_chi2_detect"],
                    gm_full_chi2_detect=g0["full_chi2_detect"], m3_chi2_detect_kp2=m0["chi2_var_detect"],
                    dfs_fraction=g0["dfs_fraction"], level_z=g0["level_z"]),
            T3_GM=dict(gm_reconstruction_skill=g3["reconstruction_skill"], gm_full_chi2_err=g3["full_chi2_err"],
                       gm_full_chi2_detect=g3["full_chi2_detect"], gm_m3_chi2_err=g3["m3_chi2_err"],
                       gm_m3_chi2_detect=g3["m3_chi2_detect"], m3_chi2_err=m3g["chi2_var_err"],
                       m3_chi2_detect=m3g["chi2_var_detect"], dfs_fraction=g3["dfs_fraction"],
                       gm_converged=g3["converged"], m3_converged=m3g["converged"]),
            prior_flag=flag))
    n = len(pre["cells"])
    if conv_t1 < 2:
        cls = "INCONCLUSIVE"
    elif false_var >= 2 or agree <= 1:
        cls = "DOES_NOT_ADD_RELIABLE_INFORMATION"
    elif agree == n and false_var == 0 and prior_flags == 0:
        cls = "SUPPORTS_M3_VARIATION_RESULT"
    else:
        cls = "PARTIALLY_SUPPORTS_M3_VARIATION_RESULT"
    sens = [dict(tau_segments=r["tau_segments"], truth=r["truth"], m3_chi2_err=r["metrics"]["m3_chi2_err"],
                 m3_chi2_detect=r["metrics"]["m3_chi2_detect"], full_chi2_detect=r["metrics"]["full_chi2_detect"],
                 dfs_fraction=r["metrics"]["dfs_fraction"], converged=r["metrics"]["converged"])
            for r in rows if r["model"] == "GM15" and r["tau_segments"] != tau]
    out = dict(STOCHASTIC_K_RESULT=cls, agreement_cells=f"{agree}/{n}", false_variations=f"{false_var}/{n}",
               prior_flags=f"{prior_flags}/{n}", t1_converged=f"{conv_t1}/{n}", cells=cells, sensitivity=sens,
               predeclaration_sha256=pre["canonical_sha256"])
    SUMMARY.write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predeclare", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.predeclare:
        write_predeclaration()
    if a.anchor:
        out = anchor()
        (ART / "phase17_kgm_anchor.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
    if a.run:
        _pre()
        rows = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else []
        key = lambda d: (d["orbit"], d["k"], d["truth"], d["model"], d.get("tau") or d.get("tau_segments"))  # noqa
        done = {key(r) for r in rows if "error" not in r}
        rows = [r for r in rows if "error" not in r]
        todo = [s for s in specs() if key(s) not in done]
        with ProcessPoolExecutor(max(1, min(a.workers, len(todo)))) as ex:
            futs = {ex.submit(job, s): s for s in todo}
            for fut in as_completed(futs):
                try:
                    rows.append(fut.result())
                except Exception as exc:                                  # recorded, never silently dropped
                    rows.append(dict(futs[fut], tau_segments=futs[fut].get("tau"), error=repr(exc)))
                RAW.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                print("done", key(rows[-1]) if "error" not in rows[-1] else rows[-1], flush=True)
    if a.analyse:
        out = analyse()
        print(json.dumps({k: v for k, v in out.items() if k not in ("cells", "sensitivity")}, indent=1))


if __name__ == "__main__":
    main()
