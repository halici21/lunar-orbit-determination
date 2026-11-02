"""Phase F3d: F3's sweep and F3b's split again, with accurate local transitions (no F3 file edited).

Predeclaration: studies/phase17/configs/phase_f3d_local_transition_requalification_predeclaration.json.
Needs F3's cached reference and C6's cached gradients (lro_data/f3_cache/c6_gradients.npy; run
phase_c6_lunar_radiation_survey.py first).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f3_lro_dmc_smoother as D  # noqa: E402  (sets F2p's historical numerics)
from phase_f3_lro_dmc_smoother import C  # noqa: E402
import phase_f3b_prediction_split as B  # noqa: E402
import phase_c6_forced_columns as FC  # noqa: E402

PRE = "studies/phase17/configs/phase_f3d_local_transition_requalification_predeclaration.json"
RESULTS = HERE.parent / "results"
OUT = RESULTS / "phase_f3d_local_transition_requalification.json"
F3_JSON = RESULTS / "phase_f3_lro_dmc_smoother.json"
F3B_JSON = RESULTS / "phase_f3b_prediction_split.json"
_ORIGINAL_TRANSITION = C.transition


def install(ld: FC.LocalDynamics) -> None:
    """Replace phase_f3_core.transition: F3's composition with Phi(t_b, t_a) from ``ld`` instead of the FD Phi."""
    def transition(ref, t_a, t_b, sigma, tau, substep=C.SUBSTEP_S):
        nw = 0 if sigma is None else 3
        n = 6 + nw + ref.nc
        m = max(1, int(np.ceil((t_b - t_a) / substep - 1e-9)))
        ts = np.linspace(t_a, t_b, m + 1)
        _, s = ref.at(ts)
        f_tot, q_tot = np.eye(n), np.zeros((n, n))
        for i in range(m):
            p10 = ld.phi(ts[i], ts[i + 1])
            ds = s[i + 1] - p10 @ s[i]
            f, q = C.substep_matrices(p10, ds, ts[i + 1] - ts[i], sigma, tau, n)
            f_tot = f @ f_tot
            q_tot = f @ q_tot @ f.T + (q if q is not None else 0.0)
        return f_tot, 0.5 * (q_tot + q_tot.T)
    C.transition = transition


def g_phi(z, ld) -> dict:
    ref = D.reference_of(z)
    grid = z["grid"]
    fit = grid <= float(z["t_cmp_fit"][-1])
    acc = np.empty_like(ref.phi)
    acc[0] = np.eye(6)
    for k in range(grid.size - 1):
        acc[k + 1] = ld.phi(grid[k], grid[k + 1]) @ acc[k]
    worst = {}
    for j in range(6):
        col_fd, col_acc = ref.phi[fit, :, j], acc[fit, :, j]
        for part, sl in (("position", slice(0, 3)), ("velocity", slice(3, 6))):
            size = float(np.max(np.linalg.norm(col_fd[:, sl], axis=1)))
            diff = float(np.max(np.linalg.norm(col_acc[:, sl] - col_fd[:, sl], axis=1)))
            worst[f"col{j}_{part}"] = diff / size
    full = {}
    for j in range(6):
        size = float(np.max(np.linalg.norm(ref.phi[:, :3, j], axis=1)))
        full[f"col{j}"] = float(np.max(np.linalg.norm(acc[:, :3, j] - ref.phi[:, :3, j], axis=1))) / size
    rel = max(worst.values())
    return dict(PASS=rel <= 1e-3, max_relative_fit_arc=rel, per_column_fit_arc=worst,
                position_relative_incl_263=full, initial_identity=bool(np.allclose(ref.phi[0], np.eye(6))))


def main() -> None:
    from lunar_od.project_paths import project_root
    t_start = time.time()
    z = D.load()
    g_nodes = np.load(project_root() / "lro_data" / "f3_cache" / "c6_gradients.npy")
    ld = FC.LocalDynamics(z["grid"], g_nodes)
    gp = g_phi(z, ld)
    print("G_PHI", json.dumps(gp), flush=True)

    # step 2: F3's sweep with the accurate transitions (F3's own code, output redirected)
    f3_before = json.loads(F3_JSON.read_text(encoding="utf-8"))
    install(ld)
    D.OUT = RESULTS / "phase_f3d_sweep.json"
    D.sweep()
    sweep = json.loads(D.OUT.read_text(encoding="utf-8"))

    # step 3: F3b's split at the new sigma* (F3b's own code, its G_F3 against the F3d sweep)
    f3b_before = json.loads(F3B_JSON.read_text(encoding="utf-8"))
    star = float(sweep["sigma_star"])
    B.SIGMA_STAR = star
    B.F3_CHI2 = {0.0: sweep["sweep"]["0"]["chi2_263"], star: sweep["sweep"][f"{star:g}"]["chi2_263"]}
    B.OUT = RESULTS / "phase_f3d_split.json"
    B.split()
    split = json.loads(B.OUT.read_text(encoding="utf-8"))
    C.transition = _ORIGINAL_TRANSITION

    g1 = sweep["gates"]["G1"]
    chi2_star = sweep["sweep"][f"{star:g}"]["chi2_263"]
    if not g1:
        decision = "F3D_NOT_QUALIFIED"
    elif (sweep["decision"] == "F3_NO_DMC_GAIN" and star == 1e-9
          and abs(chi2_star / f3_before["sweep"]["1e-09"]["chi2_263"] - 1.0) <= 0.2):
        decision = "F3D_F3_STANDS"
    else:
        decision = "F3D_F3_REVISED"
    compare = {}
    for s in sweep["sweep"]:
        new, old = sweep["sweep"][s], f3_before["sweep"][s]
        compare[s] = dict(chi2_263=[old["chi2_263"], new["chi2_263"]], ric_fit_m=[old["ric_fit_m"], new["ric_fit_m"]],
                          ric_263_m=[old["ric_263_m"], new["ric_263_m"]], k_srp=[old["k_srp"], new["k_srp"]],
                          mean_nis=[old["mean_nis"], new["mean_nis"]],
                          range_263_m=[old["residuals"]["263"]["range_rms_m"], new["residuals"]["263"]["range_rms_m"]],
                          doppler_263_mmps=[old["residuals"]["263"]["doppler_rms_mmps"],
                                            new["residuals"]["263"]["doppler_rms_mmps"]])
    shares = {k: [f3b_before["scores"][k]["share"], v["share"]] for k, v in split["scores"].items()}
    out = dict(predeclaration=PRE, gates=dict(G_LOC="tests/test_phase_c6_lunar_radiation.py", G_PHI=gp["PASS"],
                                              G1=g1),
               G_PHI=gp, decision=decision,
               f3_recomputed=dict(decision=sweep["decision"], sigma_star=star,
                                  hypotheses=sweep["hypotheses_as_computed"], gates=sweep["gates"]),
               f3_before=dict(decision=f3_before["decision"], sigma_star=f3_before["sigma_star"]),
               sweep_old_vs_new=compare,
               f3b_recomputed=dict(decision=split["decision"], gates=split["gates"],
                                   hypotheses=split["hypotheses_as_computed"]),
               f3b_before=dict(decision=f3b_before["decision"]), f3b_shares_old_vs_new=shares,
               memoised_transitions=len(ld.cache), wall_s=time.time() - t_start)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gates", "decision", "f3_recomputed", "f3b_recomputed")}, indent=1))


if __name__ == "__main__":
    main()
