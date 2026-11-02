"""Phase F3c: short-arc batch fits (262; 261-262) with F2p's recipe, predicting 263.

Predeclaration: studies/phase17/configs/phase_f3c_short_arc_batch_predeclaration.json.

  --variant A|B [--workers N]  one fit (hours; resumable checkpoint); writes results/phase_f3c_short_arc_<V>.json
  --merge                      gates, shares, decision -> results/phase_f3c_short_arc_batch.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f3_lro_dmc_smoother as D  # noqa: E402  (sets F2p's historical numerics)
from phase_f3_lro_dmc_smoother import M  # noqa: E402

PRE = "studies/phase17/configs/phase_f3c_short_arc_batch_predeclaration.json"
RESULTS = HERE.parent / "results"
OUT = RESULTS / "phase_f3c_short_arc_batch.json"
VARIANTS = {"A": ("trk/lro_no_01/2009262",), "B": ("trk/lro_no_01/2009261", "trk/lro_no_01/2009262")}
#: et0 of F2p's / F3's 260-263 problem: directory 2009260 starts at 2009-09-16 23:57 UTC (checked against LOLA below)
F3_ET0_UTC = "2009-09-16T00:00:00"
CHI2_SIGMA_STAR = 3642.6118945392136       # F3 (c15d5c8)
SIGMA_DOP_MMPS = 2.0


def part_path(v: str) -> Path:
    return RESULTS / f"phase_f3c_short_arc_{v}.json"


def f2p_trajectory():
    """F2p's state history on F3's cached 20 s grid, as absolute ET -> spline, plus the G_START check."""
    import spiceypy as spice
    z = D.load()
    et0 = float(spice.str2et(F3_ET0_UTC))
    t = float(z["t_cmp_fit"][0])
    lola = np.asarray(spice.spkezr(M.F.LRO, et0 + t, "J2000", "NONE", "MOON")[0]) * 1e3
    assert np.array_equal(lola, z["lola_cmp"][0]), "F3_ET0_UTC is not the cached problem's et0"
    return et0 + z["grid"], z["hist"][:, :6]


def start_state(et_epoch: float) -> tuple[np.ndarray, dict]:
    et, x = f2p_trajectory()
    spl = CubicSpline(et, x, axis=0)
    k = int(np.searchsorted(et, et_epoch))
    nodes = [k - 1, k]
    node_err = float(max(np.linalg.norm(spl(et[j])[:3] - x[j, :3]) for j in nodes))
    # informative: leave the nearest node out and interpolate it from the rest
    j = nodes[int(np.argmin(np.abs(et[nodes] - et_epoch)))]
    keep = np.ones(et.size, bool)
    keep[j] = False
    loo = float(np.linalg.norm(CubicSpline(et[keep], x[keep], axis=0)(et[j])[:3] - x[j, :3]))
    return spl(et_epoch), dict(PASS=node_err <= 1e-3, node_error_m=node_err, leave_one_out_error_m=loo)


def run_variant(v: str, workers: int) -> None:
    t_start = time.time()
    days = VARIANTS[v]
    d, _ = D.f2p()
    M.set_tolerance(*M.F2K_TOL)
    M.set_doppler_edits(d["doppler_edits"])
    probe = M.Problem(300, doppler=True, days=days, n_emp=len(M.EMP_NAMES))
    x0, g_start = start_state(probe.setup.et0 + probe.t_epoch)
    print(f"variant {v}: epoch {probe.t_epoch:.3f} s after {probe.setup.et0:.3f}; G_START {g_start}", flush=True)
    del probe
    run = M.solve(300, 1.0, M.F2H_PRIORS, f"f3c{v}", True, days, x0, M.F2J_PREDICT_DAYS, workers,
                  len(M.EMP_NAMES), **M._f2j_solver_options(f"phase_f3c_{v}_checkpoint.json"))
    run.pop("est_positions")
    out = dict(predeclaration=PRE, variant=v, fit_days=days, predict_days=M.F2J_PREDICT_DAYS, G_START=g_start,
               priors={str(k): val for k, val in M.F2H_PRIORS.items()}, rtol_atol=M.F2K_TOL,
               doppler_edits=sorted(M.DOPPLER_EDITS), nmax=300, run=run, wall_s=time.time() - t_start)
    part_path(v).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(variant=v, status=run["status"], fit=[run["postfit_rms_m"], run["doppler_postfit_rms_mmps"]],
                          p263=[run["prediction"]["range_rms_m"], run["prediction"]["doppler_rms_mmps"]]), indent=1))


def chi2_of(pred: dict) -> float:
    return (pred["n_range"] * pred["range_rms_m"] ** 2
            + pred["n_doppler"] * (pred["doppler_rms_mmps"] / SIGMA_DOP_MMPS) ** 2)


def merge() -> None:
    _, f2p_run = D.f2p()
    f3 = json.loads((RESULTS / "phase_f3_lro_dmc_smoother.json").read_text(encoding="utf-8"))
    star = f3["sweep"]["1e-09"]
    chi2_f2p = chi2_of(f2p_run["prediction"])
    variants = {}
    for v in VARIANTS:
        if not part_path(v).exists():
            continue
        part = json.loads(part_path(v).read_text(encoding="utf-8"))
        run = part["run"]
        pred = run["prediction"]
        gates = dict(G1=run["status"] in ("Converged", "STATIONARY"),
                     G3=run["postfit_rms_m"] <= 2.0 and run["doppler_postfit_rms_mmps"] <= 5.0,
                     G_START=part["G_START"]["PASS"])
        chi2 = chi2_of(pred)
        variants[v] = dict(
            fit_days=part["fit_days"], gates=gates, qualified=all(gates.values()), status=run["status"],
            iterations=run["iterations"], wall_h=part["wall_s"] / 3600.0,
            fit_range_rms_m=run["postfit_rms_m"], fit_doppler_rms_mmps=run["doppler_postfit_rms_mmps"],
            fit_ric_vs_lola_m=run["ric_rms_m"],
            p263=dict(range_rms_m=pred["range_rms_m"], range_mean_m=pred["range_mean_m"],
                      doppler_rms_mmps=pred["doppler_rms_mmps"], doppler_mean_mmps=pred["doppler_mean_mmps"],
                      n_range=pred["n_range"], n_doppler=pred["n_doppler"], ric_vs_lola_m=pred["ric_rms_m"]),
            chi2_263=chi2, share=(chi2_f2p - chi2) / (chi2_f2p - CHI2_SIGMA_STAR),
            k_srp=run["k_srp"], k_srp_sigma=run["k_srp_sigma"], empirical_mps2=run["empirical_mps2"],
            empirical_sigma_mps2=run["empirical_sigma_mps2"], biases_m=run["biases_m"],
            doppler_bias_mmps=run["doppler_bias_mmps"])
    qualified = {v: r for v, r in variants.items() if r["qualified"]}
    if not qualified:
        decision, best = "F3C_NOT_QUALIFIED", None
        h = dict(H_SHORT=None, H_PARTIAL=None)
    else:
        best = max(qualified, key=lambda v: qualified[v]["share"])
        s = qualified[best]["share"]
        h = dict(H_SHORT=s >= 0.8, H_PARTIAL=0.3 <= s < 0.8)
        decision = ("F3C_SHORT_ARC_REPRODUCES" if h["H_SHORT"] else
                    "F3C_SHORT_ARC_PARTIAL" if h["H_PARTIAL"] else "F3C_NO_SHORT_ARC_GAIN")
    out = dict(predeclaration=PRE, variants_present=sorted(variants), complete=len(variants) == len(VARIANTS),
               decision=decision, best_variant=best, hypotheses_as_computed=h,
               reference=dict(F2p=dict(chi2_263=chi2_f2p, k_srp=f2p_run["k_srp"],
                                       empirical_mps2=f2p_run["empirical_mps2"],
                                       p263_range_rms_m=f2p_run["prediction"]["range_rms_m"],
                                       p263_doppler_rms_mmps=f2p_run["prediction"]["doppler_rms_mmps"],
                                       ric_263_m=f2p_run["prediction"]["ric_rms_m"]),
                              sigma_star=dict(chi2_263=CHI2_SIGMA_STAR, k_srp=star["k_srp"],
                                              ric_263_m=star["ric_263_m"],
                                              p263_range_rms_m=star["residuals"]["263"]["range_rms_m"],
                                              p263_doppler_rms_mmps=star["residuals"]["263"]["doppler_rms_mmps"])),
               variants=variants)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("complete", "decision", "best_variant", "hypotheses_as_computed")},
                     indent=1))
    for v, r in variants.items():
        print(v, r["gates"], f"share {r['share']:.3f}", r["p263"])


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
        run_variant(sys.argv[sys.argv.index("--variant") + 1], w)
