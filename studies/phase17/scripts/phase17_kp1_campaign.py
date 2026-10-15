"""PHASE 17-KP1 - alternative K_SRP parameterizations: campaign, verification and analysis.

Every rule comes from artifacts/phase17_kp1_predeclaration.json (hash-pinned,
committed before any separation metric).  Analysis only: production code is
not modified; counted-Doppler VALUES are never used.

    python examples/phase17_kp1_campaign.py                 # cells + FD + analysis
    python examples/phase17_kp1_campaign.py --analyse-only
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_kp1_core as KP  # noqa: E402
import phase17_r1o_cd1_campaign as K  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402

D2 = K.D2
ARTIFACTS = KP.ARTIFACTS
PREDECLARATION_SHA256 = "4f7f2632706f3c07e2444043d827c79ba2d5f9aa79ac36d4c6a6ab8ec906e495"
CASES = ("RANGE", "RPD_D0", "RPD_D1FREE", "RPD_D1P1E-4")
PRIMARY_CASE = "RPD_D1FREE"
REP_K = (0, 6, 12, 18)
FD_CASES = (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12))
FD_BASES = ("SEG2_A", "SEG3_B", "DRIFT")
FD_EPS = (1e-3, 3e-3)
RAW_PATH = ARTIFACTS / "phase17_kp1_raw_results.json"


def x0_set(pre: dict) -> dict:
    import hashlib
    g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    x0s = {x["case_id"]: x["x0"] for x in g["geometries"] if x["case_id"] in K.ORBITS}
    for o, x0 in x0s.items():
        assert hashlib.sha256(json.dumps(x0).encode()).hexdigest() == pre["population"]["orbits"]["set"][o]["x0_sha256"]
    assert sorted(x0s) == sorted(K.ORBITS)
    return x0s


# ======================================================================
# one cell
# ======================================================================
def case_inputs(rb, rows, rbank, dbank, case):
    w60 = np.full(rows.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
    if case == "RANGE":
        return KP.assemble([(rb.h_x0, rbank, rb.w)])
    blocks = [(rb.h_x0, rbank, rb.w), (rows.h_x0, dbank, w60)]
    if case == "RPD_D0":
        return KP.assemble(blocks)
    nu = CD.nuisance_columns(rb.n_obs, rows, "D1_SESSION_CONSTANT_BIAS")
    return KP.assemble(blocks, nuisance=nu, prior_sigma=None if case == "RPD_D1FREE" else 1e-4)


def run_cell(spec: dict) -> dict:
    t0 = time.time()
    orbit, k, dyn = spec["orbit"], spec["k"], spec["dynamics"]
    scn = K.scenario(orbit, spec["x0"], k)
    if dyn == "HIGH_FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    st = K.stations()
    rows = CD.build_doppler_rows(scn, rb, 60.0, st, with_values=False)
    bank = KP.sensitivity_bank(rb.nom48, rb.t_grid)
    rbank = KP.range_parameter_columns(rb, bank)
    dbank = KP.doppler_parameter_columns(rb, rows, st, 60.0, bank)
    out = dict(orbit=orbit, k=k, dynamics=dyn, n_range=int(rb.n_obs), n_doppler=int(rows.n),
               n_sessions=int(rows.n_sessions), feasible=bool(rows.n >= 7),
               boundaries=bank["boundaries"], t_end=float(rb.t_grid[-1]),
               anchor_bank_k_range_bitwise=bool(np.array_equal(rbank[:, 0], rb.h_k)),
               anchor_bank_k_doppler_bitwise=bool(np.array_equal(dbank[:, 0], rows.h_k)),
               cases={}, rebuild={})
    for case in CASES:
        c, p = case_inputs(rb, rows, rbank, dbank, case)
        out["cases"][case] = {}
        for model in KP.MODELS:
            m = KP.block_metrics(c, p, model)
            out["cases"][case][model] = m
            if dyn == "HIGH_FIDELITY":
                ind = KP.independent_sines(c, p @ KP.model_matrix(model))
                out["rebuild"]["%s|%s" % (case, model)] = float(np.max(np.abs(np.sort(ind) - np.sort(m["sines"]))))
        # diagnostic only: individual-column f_perp of every bank column
        qc = KP._orth(c)
        out["cases"][case]["column_f_perp"] = {
            n: float(np.linalg.norm(p[:, j] - qc @ (qc.T @ p[:, j])) / np.linalg.norm(p[:, j]))
            for j, n in enumerate(KP.BANK)}
    if dyn == "HIGH_FIDELITY" and orbit == "CANON" and k in REP_K:
        c, p = case_inputs(rb, rows, rbank, dbank, PRIMARY_CASE)
        ex = {}
        for model in KP.MODELS:
            exact = KP.exact_sines(c, p @ KP.model_matrix(model))
            wr = sorted(out["cases"][PRIMARY_CASE][model]["sines"], reverse=True)
            ex[model] = dict(exact=[str(x) for x in exact],
                             max_abs_diff=max(abs(float(e) - w) for e, w in zip(exact, wr)))
        out["exact_oracle"] = ex
    out["elapsed_s"] = time.time() - t0
    return K._clean(out)


# ======================================================================
# FD oracle: segmented repropagation with K(t) = K_truth + eps b(t)
# ======================================================================
def _propagate(scn, t_slice, x0, k_srp, rtol, atol):
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    from phase17_r1od2_fidelity import NMAX
    return propagate_state_with_k_sensitivity(
        t_slice, x0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        lambda t: G.earth_state_mci(scn.et0 + float(t))[:3],
        lambda t: G.sun_mci(scn.et0 + float(t)),
        srp=SRPOptions(k_srp_m2_per_kg=k_srp, shadow_model=scn.shadow_model),
        rtol=rtol, atol=atol, j2_moon=0.0,
        harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0, harmonic_stm_opt_in=True)


def segmented_states(scn, values: np.ndarray, breaks: list[int], rtol=1e-13, atol=1e-14) -> np.ndarray:
    """(N, 6) HF states with K piecewise constant: values[s] on nodes breaks[s]..breaks[s+1]."""
    t = scn.t_grid
    x = np.zeros((t.size, 6))
    x[0] = scn.x0_array
    for s in range(len(breaks) - 1):
        a, b = breaks[s], breaks[s + 1]
        seg = _propagate(scn, t[a:b + 1], x[a], float(values[s]), rtol, atol)[:, :6]
        x[a:b + 1] = seg
    return x


def basis_pieces(name: str, t_grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    """(breaks, b-values per piece) of a bank basis on this grid."""
    n = t_grid.size
    if name == "SEG2_A":
        b = KP.boundaries(t_grid, 2)
        return [0, b[1], n - 1], np.array([1.0, 0.0])
    if name == "SEG3_B":
        b = KP.boundaries(t_grid, 3)
        return [0, b[1], b[2], n - 1], np.array([0.0, 1.0, 0.0])
    if name == "DRIFT":
        return list(range(n)), KP.drift_interval_values(t_grid)
    raise ValueError(name)


def fd_job(spec: dict) -> dict:
    scn = K.scenario(spec["orbit"], spec["x0"], spec["k"], n_orbits=2.0)
    kt = D2.M.K_TRUTH
    if spec["basis"] == "CONTROL":
        seg = segmented_states(scn, np.full(scn.t_grid.size - 1, kt), list(range(scn.t_grid.size)))
        ref = _propagate(scn, scn.t_grid, scn.x0_array, kt, 1e-13, 1e-14)[:, :6]
        return dict(case=spec["case"], basis="CONTROL",
                    max_pos_m=float(np.max(np.linalg.norm(seg[:, :3] - ref[:, :3], axis=1))),
                    max_vel_mps=float(np.max(np.linalg.norm(seg[:, 3:] - ref[:, 3:], axis=1))))
    breaks, vals = basis_pieces(spec["basis"], scn.t_grid)
    xs = segmented_states(scn, kt + spec["eps"] * vals, breaks)
    return dict(case=spec["case"], basis=spec["basis"], eps=spec["eps"], x=xs.tolist())


def fd_nominal(spec: dict) -> dict:
    scn = K.scenario(spec["orbit"], spec["x0"], spec["k"], n_orbits=2.0)
    from phase17_r1od2_fidelity import propagate_fidelity
    nom = propagate_fidelity(scn)
    bank = KP.sensitivity_bank(nom, scn.t_grid)
    return dict(case=spec["case"], **{b: bank[b].tolist() for b in FD_BASES})


def run_fd(ex, x0s) -> dict:
    noms = list(ex.map(fd_nominal, [dict(case=i, orbit=o, k=k, x0=x0s[o]) for i, (o, k) in enumerate(FD_CASES)]))
    jobs = [dict(case=i, orbit=o, k=k, x0=x0s[o], basis="CONTROL") for i, (o, k) in enumerate(FD_CASES)]
    for i, (o, k) in enumerate(FD_CASES):
        for b in FD_BASES:
            for e in FD_EPS:
                for sgn in (1.0, -1.0):
                    jobs.append(dict(case=i, orbit=o, k=k, x0=x0s[o], basis=b, eps=sgn * e))
    res = list(ex.map(fd_job, jobs))
    out = dict(cases=[], controls=[r for r in res if r["basis"] == "CONTROL"])
    for i, (o, k) in enumerate(FD_CASES):
        for b in FD_BASES:
            an = np.asarray(noms[i][b])
            errs = {}
            for e in FD_EPS:
                xp = np.asarray(next(r["x"] for r in res if r["case"] == i and r["basis"] == b and r["eps"] == e))
                xm = np.asarray(next(r["x"] for r in res if r["case"] == i and r["basis"] == b and r["eps"] == -e))
                fd = (xp - xm) / (2 * e)
                errs[str(e)] = [float(np.max(np.abs(fd[:, j] - an[:, j])) / (np.max(np.abs(an[:, j])) or 1.0))
                                for j in range(6)]
            best = [min(errs[str(e)][j] for e in FD_EPS) for j in range(6)]
            out["cases"].append(dict(orbit=o, k=k, basis=b, rel_err_by_eps=errs, best_per_component=best,
                                     worst_component=max(best)))
    worst = max(c["worst_component"] for c in out["cases"])
    out["worst"] = worst
    out["control_max_pos_m"] = max(c["max_pos_m"] for c in out["controls"])
    out["KP1_FD_GATE"] = "PASS" if worst <= 1e-4 else "FAIL"
    return out


# ======================================================================
# analysis (predeclared rules)
# ======================================================================
def _frac(v, thr):
    v = np.asarray(v, float)
    return float(np.mean(v >= thr))


def analyse(pre, cells, fd, elapsed) -> dict:
    hf = [c for c in cells if c["dynamics"] == "HIGH_FIDELITY" and c["feasible"]]
    simp = [c for c in cells if c["dynamics"] == "SIMPLIFIED" and c["feasible"]]
    committed = {(r["orbit"], int(r["k"]), r["dynamics"]): r for r in csv.DictReader(
        (ARTIFACTS / "phase17_r1o_cd1_primary_results.csv").open())}
    anchors = []
    for c in cells:
        r = committed[(c["orbit"], c["k"], c["dynamics"])]
        anchors += [abs(c["cases"]["RANGE"]["M0_SCALAR"]["s_min"] - float(r["range_f_perp"])),
                    abs(c["cases"]["RPD_D0"]["M0_SCALAR"]["s_min"] - float(r["combined_f_perp_D0"])),
                    abs(c["cases"]["RPD_D1FREE"]["M0_SCALAR"]["s_min"] - float(r["combined_f_perp_D1FREE"]))]
    bitwise = all(c["anchor_bank_k_range_bitwise"] and c["anchor_bank_k_doppler_bitwise"] for c in cells)
    rebuild = max(v for c in hf for v in c["rebuild"].values())
    exact = [m["max_abs_diff"] for c in hf if "exact_oracle" in c for m in c["exact_oracle"].values()]

    def summary(pop, case, model):
        ms = [c["cases"][case][model] for c in pop]
        d = dict(n=len(ms), s_min_median=float(np.median([m["s_min"] for m in ms])),
                 s_max_median=float(np.median([m["s_max"] for m in ms])),
                 condition_median=float(np.median([m["condition"] for m in ms])),
                 sigma_frac_median=[float(np.median([m["sigma_frac"][j] for m in ms])) for j in range(ms[0]["q"])])
        if ms[0]["q"] > 1:
            sv = [m["s_var_min"] for m in ms]
            mc = [m["max_abs_corr"] for m in ms]
            d.update(s_var_min_median=float(np.median(sv)), s_var_min_p90=float(np.percentile(sv, 90)),
                     s_var_min_max=float(np.max(sv)),
                     VAR=_frac(sv, 0.35), VAR_STRONG=_frac(sv, 0.50),
                     INT=float(np.mean(np.asarray(mc) <= 0.90)), max_abs_corr_median=float(np.median(mc)),
                     weakest_combination_median_abs=[float(np.median([abs(m["weakest_combination"][j]) for m in ms]))
                                                     for j in range(ms[0]["q"])])
            d["CLASS"] = ("DISTINGUISHABLE" if d["VAR"] >= 0.50 and d["INT"] >= 0.50 else
                          "PARTIAL" if d["VAR"] >= 0.10 else "NO_NEW_INFORMATION")
        return d

    table = {pop_name: {case: {m: summary(pop, case, m) for m in KP.MODELS} for case in CASES}
             for pop_name, pop in (("HIGH_FIDELITY", hf), ("SIMPLIFIED", simp))}
    prim = table["HIGH_FIDELITY"][PRIMARY_CASE]
    multi = [m for m in KP.MODELS if m != "M0_SCALAR"]
    dist = [m for m in multi if prim[m]["CLASS"] == "DISTINGUISHABLE"]
    part = [m for m in multi if prim[m]["CLASS"] == "PARTIAL"]
    if dist:
        primary = "ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS"
        rec = sorted(dist, key=lambda m: (len(prim[m]["sigma_frac_median"]),       # dimension, then M1 < M2 < M3
                                          ["M1_TWO_SEGMENT", "M2_LINEAR_DRIFT", "M3_THREE_SEGMENT"].index(m)))[0]
    elif part:
        primary, rec = "ALTERNATIVE_K_PARAMETERIZATION_REVEALS_ONLY_PARTIAL_SEPARATION", "M0_SCALAR"
    else:
        primary, rec = "ADDITIONAL_K_PARAMETERS_INCREASE_DEGENERACY_WITHOUT_NEW_INFORMATION", "M0_SCALAR"
    split = prim["M3_THREE_SEGMENT"]["s_var_min_median"] < prim["M1_TWO_SEGMENT"]["s_var_min_median"]
    nuis = {m: dict(s_var_min_median_D0=table["HIGH_FIDELITY"]["RPD_D0"][m]["s_var_min_median"],
                    s_var_min_median_D1FREE=prim[m]["s_var_min_median"],
                    VAR_D0=table["HIGH_FIDELITY"]["RPD_D0"][m]["VAR"], VAR_D1FREE=prim[m]["VAR"])
            for m in multi}
    gates = dict(anchors_max_abs_diff=float(max(anchors)), anchors_gate=1e-10, bank_k_bitwise=bitwise,
                 rebuild_max_abs_diff=float(rebuild), rebuild_gate=1e-8,
                 exact_oracle_max_abs_diff=float(max(exact)) if exact else None, exact_gate=1e-12,
                 fd_worst=fd["worst"], fd_gate=1e-4, fd_control_max_pos_m=fd["control_max_pos_m"])
    gates["ALL_PASS"] = bool(gates["anchors_max_abs_diff"] <= 1e-10 and bitwise and rebuild <= 1e-8
                             and exact and max(exact) <= 1e-12 and fd["KP1_FD_GATE"] == "PASS")
    summary_out = dict(
        phase="17-KP1", predeclaration_sha256=PREDECLARATION_SHA256, truth_level="SYNTHETIC_CAMPAIGN_TRUTH",
        PRIMARY_CLASS=primary, RECOMMENDED_K_PARAMETERIZATION=rec,
        per_model_class={m: prim[m]["CLASS"] for m in multi}, complexity_SPLITS_WEAK_DIRECTION=split,
        nuisance_impact=nuis, gates=gates, table=table, elapsed_s=elapsed,
        production_code_changed=False, doppler_values_used=False)
    (ARTIFACTS / "phase17_kp1_summary.json").write_text(json.dumps(K._clean(summary_out), indent=1),
                                                         encoding="utf-8")
    with (ARTIFACTS / "phase17_kp1_primary_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "k", "dynamics", "case", "model", "q", "s_min", "s_max", "s_var_min", "condition",
                    "max_abs_corr", "sigma_frac", "weakest_combination", "n_range", "n_doppler", "n_sessions"])
        for c in cells:
            for case in CASES:
                for m in KP.MODELS:
                    x = c["cases"][case][m]
                    w.writerow([c["orbit"], c["k"], c["dynamics"], case, m, x["q"], x["s_min"], x["s_max"],
                                x.get("s_var_min", ""), x["condition"], x.get("max_abs_corr", ""),
                                ";".join("%.6g" % v for v in x.get("sigma_frac", [])),
                                ";".join("%.6g" % v for v in x["weakest_combination"]),
                                c["n_range"], c["n_doppler"], c["n_sessions"]])
    return summary_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--workers", type=int, default=K.N_WORKERS)
    args = ap.parse_args()
    pre = KP.load_predeclaration(PREDECLARATION_SHA256)
    fd_path = ARTIFACTS / "phase17_kp1_fd_oracle.json"
    if args.analyse_only:
        cells = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        fd = json.loads(fd_path.read_text(encoding="utf-8"))
        s = analyse(pre, cells, fd, None)
    else:
        x0s = x0_set(pre)
        specs = [dict(orbit=o, x0=x0s[o], k=k, dynamics=dyn) for dyn in ("HIGH_FIDELITY", "SIMPLIFIED")
                 for o in K.ORBITS for k in K.EPOCHS]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=K._init) as ex:
            fd = run_fd(ex, x0s)
            fd_path.write_text(json.dumps(K._clean({k: v for k, v in fd.items()}), indent=1), encoding="utf-8")
            print("FD %s worst %.3g control %.3g m (%.0f s)" % (fd["KP1_FD_GATE"], fd["worst"],
                                                             fd["control_max_pos_m"], time.time() - t0), flush=True)
            cells = []
            for i, c in enumerate(ex.map(run_cell, specs)):
                cells.append(c)
                print("%3d/%d %s k%02d %s %.0f s" % (i + 1, len(specs), c["orbit"], c["k"], c["dynamics"],
                                                     time.time() - t0), flush=True)
        RAW_PATH.write_text(json.dumps(cells, separators=(",", ":")), encoding="utf-8")
        s = analyse(pre, cells, fd, time.time() - t0)
    print(s["PRIMARY_CLASS"], s["RECOMMENDED_K_PARAMETERIZATION"], s["per_model_class"], s["gates"]["ALL_PASS"])


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
