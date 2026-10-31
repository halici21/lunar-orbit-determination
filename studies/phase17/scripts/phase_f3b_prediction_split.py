"""Phase F3b: does the sigma* DMC smoother's 263 prediction gain come from the end-of-fit state or the constants?

Predeclaration: studies/phase17/configs/phase_f3b_prediction_split_predeclaration.json.  Uses F3's cached
reference (phase_f3_lro_dmc_smoother.py --reference) and F3's smoother.

  --lin [--workers N]  G_LIN: one nonlinear F2 propagation with F2p's parameters plus sigma*'s dynamic constant
                       deviation (about an hour at N = 300); cached next to F3's reference
  --split              the split, gates, hypotheses, decision (seconds); reads the G_LIN cache when present
With neither flag both run in order.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f3_lro_dmc_smoother as D  # noqa: E402  (sets F2p's historical numerics)
from phase_f3_lro_dmc_smoother import C, M  # noqa: E402

PRE = "studies/phase17/configs/phase_f3b_prediction_split_predeclaration.json"
OUT = HERE.parent / "results" / "phase_f3b_prediction_split.json"
SIGMA_STAR = 1e-9
F3_CHI2 = {0.0: 69899.78968941572, SIGMA_STAR: 3642.6118945392136}   # F3's JSON (c15d5c8)
#: constant indices (0..8 = K, range biases 103/105/126, Doppler bias, C_T, S_T, C_N, S_N)
MEAS_CONST = [1, 2, 3, 4]
DYN_CONST = [0, 5, 6, 7, 8]


def lin_path() -> Path:
    return D.cache_path().with_name("f3b_lin.npz")


def star_constants(f) -> np.ndarray:
    out = D.smooth_sigma(f, SIGMA_STAR)
    return out["x"][0][9:]


def build_lin(workers: int) -> None:
    """Nonlinear F2 trajectory at F2p's epoch state with sigma*'s dynamic constants."""
    t_start = time.time()
    z = D.load()
    dc = star_constants(D.prepare(z))
    d, _ = D.f2p()
    M.set_tolerance(*M.F2K_TOL)
    M.set_doppler_edits(d["doppler_edits"])
    pb = M.Problem(300, doppler=True, days=D.DAYS, n_emp=len(M.EMP_NAMES))
    assert (pb.grid_s, pb.rtol, pb.atol, pb.emp_fd_step) == (20.0, 1e-12, 1e-13, 1e-9), "not F2p's numerics"
    assert abs(pb.t_epoch - float(z["t_epoch"])) == 0.0
    p = z["p"].copy()
    p[6 + np.array(DYN_CONST)] += dc[DYN_CONST]
    traj = pb.propagate_p(p, t_eval=np.concatenate([[pb.t_epoch], z["t_cmp_263"]]))[1:]
    np.savez_compressed(lin_path(), p=p, dc=dc, traj=traj, elapsed_s=time.time() - t_start)
    print(f"G_LIN propagation done in {time.time() - t_start:.0f} s", flush=True)


def g_lin(z, f) -> dict:
    if not lin_path().exists():
        return dict(PASS=None, note="not run (f3b_lin.npz absent)")
    with np.load(lin_path()) as q:
        traj, dc = q["traj"], q["dc"]
    dc_dyn = np.zeros(9)
    dc_dyn[DYN_CONST] = dc[DYN_CONST]
    _, s = f["ref"].at(z["t_cmp_263"])
    lin = np.einsum("nij,j->ni", s, dc_dyn)[:, :3]
    nonlin = traj[:, :3] - z["ref_cmp"][f["nf"]:, :3]
    diff = float(np.max(np.linalg.norm(nonlin - lin, axis=1)))
    size = float(np.max(np.linalg.norm(lin, axis=1)))
    limit = max(0.1, 0.02 * size)
    return dict(PASS=diff <= limit, max_position_difference_m=diff, max_linear_offset_m=size, limit_m=limit)


def split() -> None:
    z = D.load()
    _, run = D.f2p()
    f = D.prepare(z)
    ref, r, hx, hc, kind, is263, nf = (f[k] for k in ("ref", "r", "hx", "hc", "kind", "is263", "nf"))
    row_epoch, cmp_epoch = f["row_epoch"], f["cmp_epoch"]

    def post_of(dx, dc, rows_sel):
        return r[rows_sel] - np.einsum("ni,ni->n", hx[rows_sel], dx) - np.einsum("ni,ni->n", hc[rows_sel], dc)

    # G_F3: the two smoother runs reproduce F3's 263 chi2
    runs = {s: D.smooth_sigma(f, s) for s in (0.0, SIGMA_STAR)}
    chi2_f3 = {}
    for s, out in runs.items():
        nw = 0 if s == 0.0 else 3
        xs = out["x"][1:]
        e = row_epoch[is263]
        chi2_f3[s] = float(np.sum(post_of(xs[e, :6], xs[e, 6 + nw:], is263) ** 2))
    gf3 = {f"{s:g}": dict(chi2=chi2_f3[s], f3=F3_CHI2[s], rel=abs(chi2_f3[s] / F3_CHI2[s] - 1.0))
           for s in chi2_f3}
    gf3_pass = all(v["rel"] <= 1e-6 for v in gf3.values())

    # the end-of-fit state and the sequential propagation of its parts
    out = runs[SIGMA_STAR]
    times = out["times"]
    first263 = min(f["epochs"][row_epoch[is263]].min(), f["epochs"][cmp_epoch[nf:]].min())
    k_e = int(np.searchsorted(times, first263) - 1)
    assert all(row is None for row in f["rows"][k_e:]), "a fit row after t_e"
    x_e = out["x"][k_e]
    cols = np.zeros((x_e.size, 3))
    cols[:9, 0] = x_e[:9]                                     # STATE: dx, w
    cols[9 + np.array(DYN_CONST), 1] = x_e[9 + np.array(DYN_CONST)]
    cols[9 + np.array(MEAS_CONST), 2] = x_e[9 + np.array(MEAS_CONST)]
    pred_t = times[k_e + 1:]
    parts = C.propagate_from(ref, times[k_e], cols, pred_t, SIGMA_STAR, D.TAU_S)    # (m, 18, 3)
    g_add = float(np.max(np.abs(parts.sum(axis=2)[:, :3] - out["x"][k_e + 1:, :3])))

    combos = {"F2p": [], "STATE": [0], "CONST": [1, 2], "DYN": [1], "MEAS": [2], "STATE+DYN": [0, 1],
              "FULL": [0, 1, 2]}
    sel263 = np.where(is263)[0]
    pe = row_epoch[sel263] + 1 - (k_e + 1)                    # index into pred_t
    ce = cmp_epoch[nf:] + 1 - (k_e + 1)
    assert pe.min() >= 0 and ce.min() >= 0
    scores = {}
    for name, idx in combos.items():
        x = parts[:, :, idx].sum(axis=2) if idx else np.zeros(parts.shape[:2])
        post = post_of(x[pe, :6], x[pe, 9:], is263)
        with warnings.catch_warnings():                      # the (empty) fit-day half of _resid_stats
            warnings.simplefilter("ignore", RuntimeWarning)
            st = D._resid_stats(post, kind[is263], np.ones(post.size, bool))["263"]
        est = z["ref_cmp"][nf:, :3] + x[ce, :3]
        scores[name] = dict(chi2_263=float(np.sum(post ** 2)), **st, ric_263_m=D._ric_rms(est, z["lola_cmp"][nf:]))
    c0, cf = scores["F2p"]["chi2_263"], scores["FULL"]["chi2_263"]
    for v in scores.values():
        v["share"] = (c0 - v["chi2_263"]) / (c0 - cf)

    gl = g_lin(z, f)
    gates = dict(G_F3=gf3_pass, G_ADD=g_add <= 1e-3, G_LIN=gl["PASS"])
    h_state = scores["STATE"]["share"] >= 0.8
    h_const = scores["CONST"]["share"] >= 0.8
    if not all(v is True for v in gates.values()):
        decision = "F3B_NOT_QUALIFIED" if any(v is False for v in gates.values()) else "PENDING_G_LIN"
    else:
        decision = {(True, False): "F3B_STATE_CARRIES", (False, True): "F3B_CONSTANTS_CARRY"}.get(
            (h_state, h_const), "F3B_MIXED")
    # the end-of-fit state in the RIC frame of the nearest fit comparison epoch (approximate frame, reporting only)
    j = int(np.argmin(np.abs(z["t_cmp_fit"] - times[k_e])))
    ref6 = z["ref_cmp"][j][None, :]
    ric = lambda v: M._ric(np.hstack([ref6[:, :3] + v[None, :], np.zeros((1, 3))]), ref6)[0].tolist()  # noqa: E731
    dc = x_e[9:]
    result = dict(
        predeclaration=PRE, gates=gates, G_F3=gf3, G_ADD_max_position_difference_m=g_add, G_LIN=gl,
        hypotheses_as_computed=dict(H_STATE=h_state, H_CONST=h_const), decision=decision,
        t_e_s=float(times[k_e]), first_263_epoch_s=float(first263), n_263_rows=int(sel263.size), scores=scores,
        end_of_fit=dict(dx_ric_m=ric(x_e[:3]), dv_ric_mps=ric(x_e[3:6]), w_ric_mps2=ric(x_e[6:9]),
                        ric_frame_epoch_offset_s=float(z["t_cmp_fit"][j] - times[k_e])),
        sigma_star_constant_deviation=dict(zip(["K", "rb103_m", "rb105_m", "rb126_m", "doppler_bias_mps", "C_T",
                                                "S_T", "C_N", "S_N"], dc.tolist())),
        f2p_reference=dict(ric_263_m=run["prediction"]["ric_rms_m"]))
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, v in scores.items():
        print(f"{name:10s} chi2 {v['chi2_263']:10.1f} share {v['share']:6.3f} range {v['range_rms_m']:.2f} m "
              f"Doppler {v['doppler_rms_mmps']:.2f} mm/s (mean {v['doppler_mean_mmps']:+.2f}) RIC "
              f"{np.round(v['ric_263_m'], 2)}")
    print(json.dumps({k: result[k] for k in ("gates", "G_ADD_max_position_difference_m", "G_LIN",
                                             "hypotheses_as_computed", "decision", "end_of_fit")}, indent=1))


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if "--split" not in sys.argv:
        build_lin(workers)
    if "--lin" not in sys.argv:
        split()
