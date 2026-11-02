"""Phase C7-I: F2p's recipe with the lunar solid tide (k2, k3; Earth and Sun) in the propagation.

Predeclaration: studies/phase17/configs/phase_c7i_tide_refit_predeclaration.json.

  --gates          G_OFF and G_LIN (about 20 min, one N = 300 propagation over 260-263)
  --fit [--workers N]  the refit (hours; resumable checkpoint), then the result JSON
The substitution below runs at import, so ProcessPoolExecutor workers (which re-import this module) get it too.
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
from phase_f3_lro_dmc_smoother import M  # noqa: E402
import phase_c7_dynamics_candidates as T  # noqa: E402

PRE = "studies/phase17/configs/phase_c7i_tide_refit_predeclaration.json"
RESULTS = HERE.parent / "results"
OUT = RESULTS / "phase_c7i_tide_refit.json"
TIDE = {"scale": 1.0}
_ORIGINAL = M.Problem._propagate_empirical


def tide_acceleration(pb, t, r):
    """C7's elastic tide on one position at propagation time t, from the Problem's own ephemeris and GM."""
    r1 = np.asarray(r, float)[None, :]
    e = np.asarray(pb.eph.earth_position(t), float).reshape(1, 3)
    s = np.asarray(pb.eph.sun_position(t), float).reshape(1, 3)
    a = np.zeros((1, 3))
    for d, mu in ((e, pb.mu_e), (s, pb.mu_s)):
        for n, k in ((2, T.K2_GL1800F), (3, T.K3_GL1800F)):
            a += T.tide_acceleration(r1, d, mu, n, k)
    return a[0]


def _propagate_empirical_tide(self, t, x0, k, emp):
    """F2's _propagate_empirical (phase_f2_lro_od.py) with the tide added to the same RHS."""
    from lunar_od import dynamics as dyn
    from lunar_od.srp import SRPOptions
    srp = SRPOptions(k_srp_m2_per_kg=k)
    rhs, get_sun = dyn._build_state_rhs(t, self.mu_m, self.mu_e, self.mu_s, self.eph.earth_position,
                                        self.eph.sun_position, 0.0, 0.0, "indirect", self.model, None,
                                        self.setup.et0, srp)
    e = np.asarray(emp, float)
    scale = TIDE["scale"]

    def rhs_e(tt, y):
        out = np.array(rhs(tt, y), dtype=float, copy=True)
        out[3:6] += M.empirical_acceleration(y, e)
        if scale != 0.0:
            out[3:6] += scale * tide_acceleration(self, tt, y[:3])
        return out

    cuts = dyn._shadow_contact_times(rhs_e, t, x0, get_sun, srp, self.rtol, self.atol)
    return np.asarray(dyn._integrate_segmented(t, x0, rhs_e, self.rtol, self.atol, "DOP853", cuts,
                                               "C7-I tide propagation",
                                               contact_check=dyn._shadow_contact_check(get_sun, srp)))


M.Problem._propagate_empirical = _propagate_empirical_tide


def gates() -> dict:
    import spiceypy as spice
    import phase_c7_dynamics_candidates_survey as S7
    t_start = time.time()
    z = D.load()
    d, run = D.f2p()
    M.set_tolerance(*M.F2K_TOL)
    M.set_doppler_edits(d["doppler_edits"])
    pb = M.Problem(300, doppler=True, days=D.DAYS, n_emp=len(M.EMP_NAMES))
    assert (pb.grid_s, pb.rtol, pb.atol) == (20.0, 1e-12, 1e-13), "not F2p's numerics"
    p = z["p"]
    grid = z["grid"]
    # G_OFF: scale 0 reproduces the original function bit for bit over the first 2 h
    t2 = grid[grid <= grid[0] + 7200.0]
    TIDE["scale"] = 0.0
    a = pb._propagate_empirical(t2, p[:6], p[6], p[11:15])
    b = _ORIGINAL(pb, t2, p[:6], p[6], p[11:15])
    g_off = dict(PASS=bool(np.array_equal(a, b)), max_abs_difference=float(np.max(np.abs(a - b))))
    # G_LIN: nonlinear tide effect at F2p's parameters vs C7's linear column (recomputed the C7 way)
    TIDE["scale"] = 1.0
    traj = pb._propagate_empirical(grid, p[:6], p[6], p[11:15])
    nonlin = traj[:, :3] - z["hist"][:, :3]
    lin = linear_tide_column(z)
    diff = float(np.max(np.linalg.norm(nonlin - lin, axis=1)))
    size = float(np.max(np.linalg.norm(lin, axis=1)))
    limit = max(0.1, 0.02 * size)
    g_lin = dict(PASS=diff <= limit, max_position_difference_m=diff, linear_column_max_m=size, limit_m=limit,
                 nonlinear_max_m=float(np.max(np.linalg.norm(nonlin, axis=1))))
    out = dict(G_OFF=g_off, G_LIN=g_lin, wall_s=time.time() - t_start)
    (RESULTS / "phase_c7i_tide_refit_gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


def linear_tide_column(z) -> np.ndarray:
    """C7's linear tide column (position, m) on F3's grid, recomputed exactly as the C7 survey does."""
    import spiceypy as spice
    from scipy.interpolate import CubicHermiteSpline
    import phase_c6_forced_columns as FC
    import phase_c6_lunar_radiation_survey as S6
    from lunar_od.project_paths import project_root
    import phase_c7_dynamics_candidates_survey as S7
    grid, states = z["grid"], z["hist"][:, :6]
    et0 = float(spice.str2et(S6.F3_ET0_UTC))
    sub, r_sub = FC.sub_nodes(grid, states)
    gm = {b: float(spice.bodvrd(c, "GM", 1)[1][0]) * 1e9 for b, c in (("EARTH", "399"), ("SUN", "10"))}
    tide = np.zeros_like(r_sub)
    for b in gm:
        tab = S7.body_positions(b, et0 + grid)
        pos = np.array([np.interp(sub, grid, tab[:, i]) for i in range(3)]).T
        for n, k in ((2, T.K2_GL1800F), (3, T.K3_GL1800F)):
            tide += T.tide_acceleration(r_sub, pos, gm[b], n, k)
    g_nodes = np.load(project_root() / "lro_data" / "f3_cache" / "c6_gradients.npy")
    return FC.forced_columns(grid, states, tide[:, :, None], None, g_nodes=g_nodes)[:, :3, 0]


def chi2_of(pred: dict) -> float:
    return pred["n_range"] * pred["range_rms_m"] ** 2 + pred["n_doppler"] * (pred["doppler_rms_mmps"] / 2.0) ** 2


def fit(workers: int) -> None:
    t_start = time.time()
    d, f2p_run = D.f2p()
    M.set_tolerance(*M.F2K_TOL)
    M.set_doppler_edits(d["doppler_edits"])
    x0 = np.array(json.loads((RESULTS / "phase_f2h_lro_alongtrack_empirical.json").read_text(
        encoding="utf-8"))["runs"]["S1"]["x0"], float)
    run = M.solve(300, 1.0, M.F2H_PRIORS, "c7i", True, M.F2J_FIT_DAYS, x0, M.F2J_PREDICT_DAYS, workers,
                  len(M.EMP_NAMES), **M._f2j_solver_options("phase_c7i_checkpoint.json"))
    run.pop("est_positions")
    g = json.loads((RESULTS / "phase_c7i_tide_refit_gates.json").read_text(encoding="utf-8"))
    gates_ = dict(G_OFF=g["G_OFF"]["PASS"], G_LIN=g["G_LIN"]["PASS"],
                  G1=run["status"] in ("Converged", "STATIONARY"),
                  G3=run["postfit_rms_m"] <= 2.0 and run["doppler_postfit_rms_mmps"] <= 5.0)
    c_new, c_f2p = chi2_of(run["prediction"]), chi2_of(f2p_run["prediction"])
    h = dict(H_PRED=c_new <= 0.5 * c_f2p, H_K=run["k_srp"] <= 0.05)
    if not all(gates_.values()):
        decision = "C7I_NOT_QUALIFIED"
    else:
        decision = ("C7I_TIDE_CONFIRMED" if h["H_PRED"] else "C7I_TIDE_PARTIAL" if h["H_K"]
                    else "C7I_TIDE_NOT_CONFIRMED")
    out = dict(predeclaration=PRE, gates=gates_, gate_details=g, hypotheses_as_computed=h, decision=decision,
               chi2_263=c_new, chi2_263_f2p=c_f2p, love_numbers=dict(k2=T.K2_GL1800F, k3=T.K3_GL1800F),
               f2p=dict(k_srp=f2p_run["k_srp"], empirical_mps2=f2p_run["empirical_mps2"],
                        fit=[f2p_run["postfit_rms_m"], f2p_run["doppler_postfit_rms_mmps"]],
                        p263=[f2p_run["prediction"]["range_rms_m"], f2p_run["prediction"]["doppler_rms_mmps"]],
                        ric_fit_m=f2p_run["ric_rms_m"], ric_263_m=f2p_run["prediction"]["ric_rms_m"]),
               priors={str(k): v for k, v in M.F2H_PRIORS.items()}, rtol_atol=M.F2K_TOL,
               doppler_edits=sorted(M.DOPPLER_EDITS), run=run, wall_s=time.time() - t_start)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gates", "hypotheses_as_computed", "decision", "chi2_263")}, indent=1))


if __name__ == "__main__":
    if "--gates" in sys.argv:
        gates()
    if "--fit" in sys.argv:
        fit(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8)
