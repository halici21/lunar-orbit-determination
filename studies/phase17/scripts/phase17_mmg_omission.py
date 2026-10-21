"""Phase 17-MM-G: GL1800F-internal omission error (degrees 51..D) as the mismatch truth.

Frozen in artifacts/phase17_mmg_predeclaration.json (written before any MM-G number).

  --linear [--workers N]  stage 1: omission columns D = 100, 200 on the 10 FE-P designs (+ gate L2 on CANON k0);
                          one npz per cell in <FEP_DESIGNS>/../mmg_columns/
  --score-linear          gate L1, N = 0 / FE data-only / FE x1 prior metrics, linear masquerade flags
                          -> artifacts/phase17_mmg_linear.json
  --nonlinear [--workers] stage 2: the predeclared run set with truth GL1800F 100x100 -> artifacts/phase17_mmg_jobs/
  --analyse               predeclared outcome rule -> artifacts/phase17_mmg_summary.json
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

import phase17_fe_linear as FE  # noqa: E402
import phase17_fe_prior as F  # noqa: E402
import phase17_kgm_core as G  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase17_mm_experiment as MM  # noqa: E402

K = P.K
ART = K2.ARTIFACTS
COLS = F.DESIGNS.parent / "mmg_columns"
JOBS = ART / "phase17_mmg_jobs"
CHI2 = K2.CHI2_2_999
DEGREES = (100, 200)
CELLS = [(o, k) for o in K.ORBITS for k in (0, 12)]          # KP2 ci = index
MM_CELLS = [(o, k, ci) for o, k, ci in MM.CELLS]


def _gl(nmax):
    import od_gravity_covariance_campaign as C  # noqa: N811
    if nmax in C._M:
        return C._M[nmax]
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    return dataclasses.replace(load_lunar_gravity_model(C.GL1800F, nmax=nmax, mmax=nmax), frame=C.FRAME)


# ----------------------------------------------------------------------
# fine-grid convolution (added after gate L2 failed: degree 51..100 terms vary along track faster than the 90 s grid)
# ----------------------------------------------------------------------
def convolve_fine(scn, t, X, PHI, models_minus, sub):
    """S(t_i) = Phi(t_i) int_0^t_i Phi^-1 [0; a_m - a_ref] dtau with the integrand on a grid sub x finer than t.

    Position and Phi are cubic-spline interpolated between the grid nodes (both are smooth on the orbital time
    scale); the acceleration difference is evaluated exactly at every sub-step. models_minus = [(model, ref), ...].
    """
    import spiceypy as spice
    from scipy.interpolate import CubicSpline
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    tf = np.concatenate([np.linspace(t[i], t[i + 1], sub + 1)[:-1] for i in range(t.size - 1)] + [t[-1:]])
    pos = CubicSpline(t, X[:, :3], axis=0)(tf)
    phi = CubicSpline(t, PHI.reshape(t.size, -1), axis=0)(tf).reshape(tf.size, 6, 6)
    rot = [np.asarray(spice.pxform("J2000", C.FRAME, scn.et0 + float(ti))) for ti in tf]
    node = np.searchsorted(tf, t)
    cols = []
    for model, ref in models_minus:
        acc = np.stack([sha(pos[i], model, rot[i]) - sha(pos[i], ref, rot[i]) for i in range(tf.size)])
        integ = np.stack([np.linalg.solve(phi[i], np.concatenate([np.zeros(3), acc[i]])) for i in range(tf.size)])
        psi = np.zeros_like(integ)
        psi[1:] = np.cumsum(0.5 * np.diff(tf)[:, None] * (integ[1:] + integ[:-1]), axis=0)
        cols.append(np.einsum("nij,nj->ni", PHI, psi[node]))
    return cols


# ----------------------------------------------------------------------
# stage 1
# ----------------------------------------------------------------------
def cell_columns(spec):
    t0 = time.time()
    K._init()
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    orbit, k = spec["orbit"], spec["k"]
    path = COLS / f"{orbit}_k{k}.npz"
    if path.exists():
        return orbit, k, "cached"
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), FE.HF)
    t = scn.t_grid
    rot = FE._rotations(scn, t)
    gl, gr = MM._fields()
    mix1 = MM.mixed_field(1.0, gl, gr)
    a50 = np.stack([sha(X[i, :3], gl, rot[i]) for i in range(t.size)])
    acc = [np.stack([sha(X[i, :3], mix1, rot[i]) for i in range(t.size)]) - a50]
    for d in DEGREES:
        g = _gl(d)
        acc.append(np.stack([sha(X[i, :3], g, rot[i]) for i in range(t.size)]) - a50)
    s_cols = [FE.MM_convolve(t, PHI, a) for a in acc]
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, K.stations())
    ev = G.evaluate(fx, X, PHI, np.concatenate([S, np.stack(s_cols)], axis=0))
    wr, wd = 1 / K2.SIGMA_RANGE_M, 1 / K2.SIGMA_DOPPLER_MPS
    hk = np.vstack([ev["hk_r"] * wr, ev["hk_d"] * wd])
    out = dict(h_mm=hk[:, 3], h_om100=hk[:, 4], h_om200=hk[:, 5],
               accel_rms=np.array([float(np.sqrt(np.mean(np.sum(a ** 2, axis=1)))) for a in acc]))
    if orbit == "CANON" and k == 0:                              # gate L2: convolution vs propagated difference
        real = K2._prop_segment
        K2._prop_segment = MM._truth_segment(_gl(100))
        try:
            Xt, _, _ = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), FE.HF)
        finally:
            K2._prop_segment = real
        diff = Xt[:, :3] - X[:, :3]
        out["L2_rel"] = np.array(float(np.max(np.linalg.norm(s_cols[1][:, :3] - diff, axis=1))
                                       / np.max(np.linalg.norm(diff, axis=1))))
        out["L2_max_pos_diff_m"] = np.array(float(np.max(np.linalg.norm(diff, axis=1))))
    out["elapsed"] = np.array(time.time() - t0)
    np.savez_compressed(path, **out)
    return orbit, k, "ok"


COLS_FINE = F.DESIGNS.parent / "mmg_columns_fine"
SUB = 6


def cell_fine(spec):
    """Fine-grid omission columns: GL100 - GL50 (primary) and GL200 - GL100 (post-hoc secondary), sub = SUB."""
    t0 = time.time()
    K._init()
    orbit, k = spec["orbit"], spec["k"]
    path = COLS_FINE / f"{orbit}_k{k}.npz"
    if path.exists():
        return orbit, k, "cached"
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), FE.HF)
    t = scn.t_grid
    s_cols = convolve_fine(scn, t, X, PHI, [(_gl(100), _gl(50)), (_gl(200), _gl(100))], SUB)
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, K.stations())
    ev = G.evaluate(fx, X, PHI, np.concatenate([S, np.stack(s_cols)], axis=0))
    wr, wd = 1 / K2.SIGMA_RANGE_M, 1 / K2.SIGMA_DOPPLER_MPS
    hk = np.vstack([ev["hk_r"] * wr, ev["hk_d"] * wd])
    np.savez_compressed(path, h_om100=hk[:, 3], h_om200_100=hk[:, 4], elapsed=np.array(time.time() - t0))
    return orbit, k, "ok"


def score_linear_fine() -> dict:
    coeffs = F.FE.coefficient_list(6)
    sig = F.sigma_vector(coeffs, F.parse_gl1800f(6))
    rows = {}
    for o, k in CELLS:
        name = f"{o}_k{k}"
        d = F._load(F.DESIGNS / f"{name}.npz")
        c = dict(np.load(COLS_FINE / f"{name}.npz"))
        n_data = d["hx"].shape[0]
        window = 2 * math.sqrt(2 * n_data)
        r = dict(n_data=n_data, window=window)
        for lab, col in (("D100", c["h_om100"]), ("D200", c["h_om100"] + c["h_om200_100"]),
                         ("RES_101_200_posthoc", c["h_om200_100"])):
            dd = dict(d, h_mm=col)
            per = {}
            for cfg, n_deg, s in (("N0", 0, None), ("N4_data", 4, None), ("N6_data", 6, None),
                                  ("N4_prior_x1", 4, sig), ("N6_prior_x1", 6, sig)):
                m = F.score(dd, n_deg, s)["alpha1.0"]
                per[cfg] = dict(fv_chi2=m["fv_chi2"], resid_chi2_data=m["resid_chi2_data"],
                                level_bias_over_sigma=m["level_bias_over_sigma"])
            per["linear_masquerade"] = bool(per["N0"]["fv_chi2"] >= CHI2 and per["N0"]["resid_chi2_data"] <= window)
            r[lab] = per
        rows[name] = r
    labs = ("D100", "D200", "RES_101_200_posthoc")
    sup = {lab: {cfg: all(rows[n][lab][cfg]["fv_chi2"] < CHI2 for n in rows)
                 for cfg in ("N0", "N4_data", "N6_data", "N4_prior_x1", "N6_prior_x1")} for lab in labs}
    l2 = json.loads((ART / "phase17_mmg_gate_L2_fine.json").read_text(encoding="utf-8"))
    return dict(sub=SUB, gate_L2_fine=l2, gate_L2_fine_pass=l2[str(SUB)]["rel"] <= 2e-2, suppresses=sup,
                linear_masquerade={lab: [n for n in rows if rows[n][lab]["linear_masquerade"]] for lab in labs},
                rows=rows)


def score_linear() -> dict:
    coeffs = F.FE.coefficient_list(6)
    sig = F.sigma_vector(coeffs, F.parse_gl1800f(6))
    rows, l1, l2 = {}, 0.0, None
    for o, k in CELLS:
        name = f"{o}_k{k}"
        d = F._load(F.DESIGNS / f"{name}.npz")
        c = dict(np.load(COLS / f"{name}.npz"))
        l1 = max(l1, float(np.max(np.abs(c["h_mm"] - d["h_mm"])) / np.max(np.abs(d["h_mm"]))))
        if "L2_rel" in c:
            l2 = dict(rel=float(c["L2_rel"]), max_pos_diff_m=float(c["L2_max_pos_diff_m"]))
        n_data = d["hx"].shape[0]
        window = 2 * math.sqrt(2 * n_data)
        r = dict(n_data=n_data, window=window, accel_rms_mm_om100_om200=c["accel_rms"].tolist())
        for deg in DEGREES:
            dd = dict(d, h_mm=c[f"h_om{deg}"])
            per = {}
            for label, n_deg, s in (("N0", 0, None), ("N4_data", 4, None), ("N6_data", 6, None),
                                    ("N4_prior_x1", 4, sig), ("N6_prior_x1", 6, sig)):
                m = F.score(dd, n_deg, s)["alpha1.0"]
                per[label] = dict(fv_chi2=m["fv_chi2"], resid_chi2_data=m["resid_chi2_data"],
                                  level_bias_over_sigma=m["level_bias_over_sigma"],
                                  state_bias_pos_m=m["state_bias_pos_m"])
            n0 = per["N0"]
            per["linear_masquerade"] = bool(n0["fv_chi2"] >= CHI2 and n0["resid_chi2_data"] <= window)
            r[f"D{deg}"] = per
        rows[name] = r
    sup = {f"D{deg}": {lab: all(rows[n][f"D{deg}"][lab]["fv_chi2"] < CHI2 for n in rows)
                       for lab in ("N0", "N4_data", "N6_data", "N4_prior_x1", "N6_prior_x1")} for deg in DEGREES}
    flagged = [n for n in rows if rows[n]["D100"]["linear_masquerade"]]
    mm_names = [f"{o}_k{k}" for o, k, _ in MM_CELLS]
    run_set = mm_names + [n for n in flagged if n not in mm_names]
    return dict(gates=dict(L1_max_rel=l1, L1_pass=l1 <= 1e-10, L2=l2, L2_pass=bool(l2 and l2["rel"] <= 2e-2)),
                suppresses=sup, linear_masquerade_D100=flagged,
                linear_masquerade_D200=[n for n in rows if rows[n]["D200"]["linear_masquerade"]],
                nonlinear_run_set=run_set, rows=rows)


# ----------------------------------------------------------------------
# stage 2
# ----------------------------------------------------------------------
def nl_job(spec):
    t0 = time.time()
    orbit, k, ci = spec["orbit"], spec["k"], spec["ci"]
    if spec["kind"] == "baseline":
        out = MM.job(dict(orbit=orbit, k=k, ci=ci, alpha=None))
        out["elapsed_s"] = time.time() - t0
        return out
    c = MM._cell(orbit, k, ci, _gl(100))
    cap = []
    real = K2._gauss_newton

    def spy(*a, **kw):
        res = real(*a, **kw)
        cap.append(res[0])
        return res
    K2._gauss_newton = spy
    try:
        sol = K2.solve(c["fx"], c["z_r"], c["z_d"], c["x0_ap"], dyn=MM.HF, model="M3", bias_mode="FREE")
    finally:
        K2._gauss_newton = real
    r = np.asarray(cap[-1]["r"], float)
    rr, rd = r[:len(c["z_r"])], r[len(c["z_r"]):]
    th, sd = np.asarray(sol["theta"]), np.asarray(sol["sd"])
    out = dict(kind="MISMATCH_GL1800F_100", **MM._diagnostics(c, rr, rd), converged=sol["converged"],
               n_eval=sol["n_eval"], theta=th.tolist(), sd=sd.tolist(),
               metrics=K2.recovery_metrics(sol, c["x0"], np.zeros(3)),
               x0_err_over_sigma=((th[:6] - c["x0"]) / sd[:6]).tolist(),
               bias_err_rms=float(np.sqrt(np.mean((th[9:] - c["b_true"]) ** 2))))
    out.update(orbit=orbit, k=k, ci=ci, elapsed_s=time.time() - t0)
    return P.K._clean(out)


def _run_nl(spec):
    path = JOBS / f"{spec['orbit']}_k{spec['k']}_{spec['kind']}.json"
    if path.exists():
        return path.stem, "cached"
    try:
        out = nl_job(spec)
    except Exception as exc:                                     # recorded, never hidden
        out = dict(error=repr(exc)[:800], **spec)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return path.stem, "error" if "error" in out else "ok"


def run_nonlinear(workers):
    JOBS.mkdir(exist_ok=True)
    lin = json.loads((ART / "phase17_mmg_linear.json").read_text(encoding="utf-8"))
    mm_names = {f"{o}_k{k}" for o, k, _ in MM_CELLS}
    specs = []
    for name in lin["nonlinear_run_set"]:
        o, k = next((o, k) for o, k in CELLS if f"{o}_k{k}" == name)
        ci = CELLS.index((o, k))
        specs.append(dict(orbit=o, k=k, ci=ci, kind="mismatch"))
        if name not in mm_names:
            specs.append(dict(orbit=o, k=k, ci=ci, kind="baseline"))
    with ProcessPoolExecutor(workers) as ex:
        for name, status in ex.map(_run_nl, specs):
            print(name, status, flush=True)


def analyse() -> dict:
    lin = json.loads((ART / "phase17_mmg_linear.json").read_text(encoding="utf-8"))
    rows = []
    for name in lin["nonlinear_run_set"]:
        m = json.loads((JOBS / f"{name}_mismatch.json").read_text(encoding="utf-8"))
        bp = MM.JOBS / f"{name}_baseline.json"
        b = json.loads((bp if bp.exists() else JOBS / f"{name}_baseline.json").read_text(encoding="utf-8"))
        if "error" in m or "error" in b:
            rows.append(dict(cell=name, converged=False, error=m.get("error", b.get("error")), masquerade=False))
            continue
        n = m["n_range"] + m["n_doppler"]
        dchi2 = m["chi2_data"] - b["chi2_data"]
        thr = 2 * math.sqrt(2 * n)
        det = m["metrics"]["chi2_var_detect"]
        rows.append(dict(cell=name, converged=m["converged"], chi2_detect=det, detect=det >= CHI2, delta_chi2=dchi2,
                         delta_chi2_threshold=thr, residual_undetectable=dchi2 <= thr,
                         masquerade=bool(m["converged"] and det >= CHI2 and dchi2 <= thr),
                         runs_range_z=m["runs_range"]["stouffer_z"], runs_doppler_z=m["runs_doppler"]["stouffer_z"],
                         doppler_wrms=m["doppler_wrms"], doppler_wrms_base=b["doppler_wrms"],
                         level_err=m["metrics"]["level_err"], level_z=m["metrics"].get("level_z"),
                         x0_err_over_sigma_max=float(np.max(np.abs(m["x0_err_over_sigma"]))),
                         linear_D100=lin["rows"][name]["D100"]["N0"]))
    gates_ok = lin["gates"]["L1_pass"] and lin["gates"]["L2_pass"]
    conv = [r for r in rows if r["converged"]]
    if not gates_ok or any(not r["converged"] for r in rows):
        outcome = "MMG_INCONCLUSIVE"
        if gates_ok and any(r["masquerade"] for r in conv):
            outcome = "MMG_OMISSION_MASQUERADES"             # a failed run cannot undo a converged masquerade
    elif any(r["masquerade"] for r in conv):
        outcome = "MMG_OMISSION_MASQUERADES"
    elif any(r["detect"] for r in conv):
        outcome = "MMG_OMISSION_FALSE_VARIATION_DETECTABLE"
    else:
        outcome = "MMG_OMISSION_NEGLIGIBLE"
    return dict(outcome=outcome, gates=lin["gates"], rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--score-linear", action="store_true")
    ap.add_argument("--linear-fine", action="store_true")
    ap.add_argument("--score-linear-fine", action="store_true")
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.linear:
        COLS.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for o, k, st in ex.map(cell_columns, [dict(orbit=o, k=k) for o, k in CELLS]):
                print(o, k, st, flush=True)
    if a.linear_fine:
        COLS_FINE.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for o, k, st in ex.map(cell_fine, [dict(orbit=o, k=k) for o, k in CELLS]):
                print(o, k, st, flush=True)
    if a.score_linear_fine:
        out = score_linear_fine()
        (ART / "phase17_mmg_linear_fine.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    if a.score_linear:
        out = score_linear()
        (ART / "phase17_mmg_linear.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    if a.nonlinear:
        run_nonlinear(a.workers)
    if a.analyse:
        out = analyse()
        (ART / "phase17_mmg_summary.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
