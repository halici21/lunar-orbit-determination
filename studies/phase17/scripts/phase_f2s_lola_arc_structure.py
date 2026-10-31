"""Phase F2s: is F2r's cross-track misfit LOLA's arc stitching or our dynamics?

Predeclaration: studies/phase17/configs/phase_f2s_lola_arc_structure_predeclaration.json.  Analysis only: the F2
study driver's Problem and RHS, F2r's helpers, no production code changes.

Modes (parts under results/phase_f2s_parts/; --summary merges them):
  --detect               step A: acceleration residual of LOLA against our model, method check, taper windows
  --fit-all [--shard i/n] [--workers N]   step B: window fits (windows follow from step A's part)
  --summary              decision as predeclared
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2r_postfit_growth as R  # noqa: E402

PARTS = HERE.parent / "results" / "phase_f2s_parts"
OUT = HERE.parent / "results" / "phase_f2s_lola_arc_structure.json"
DT_A = 30.0
H_STENCIL = 5.0
METHOD_CHECK_MAX = 1e-9
ENV_FACTOR = 3.0
MIN_TAPER_S = 3 * 3600.0
MERGE_GAP_S = 2 * 3600.0
MIN_CORE_S = 6 * 3600.0
N_RMS_E2 = 8.44
H_REF_CORE_MAX = 0.25 * N_RMS_E2

# F2s-1 amendment (configs/phase_f2s1_stencil_step_amendment_predeclaration.json): h = 1 s + SPK noise check
H1 = "--h1" in sys.argv
if H1:
    H_STENCIL = 1.0
    OUT = HERE.parent / "results" / "phase_f2s1_lola_arc_structure.json"
H_NOISE = 2.0
SPK_NOISE_MAX = 0.1
A_PART = "A_h1" if H1 else "A"
W_PREFIX = "W1_" if H1 else ""


def write_part(name, obj):
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")


def read_part(name):
    return json.loads((PARTS / f"{name}.json").read_text(encoding="utf-8"))


def e2_solution():
    e2 = json.loads((R.PARTS / "E2.json").read_text(encoding="utf-8"))
    emp = np.array([e2["empirical_mps2"][n] for n in M.EMP_NAMES])
    return np.array(e2["x0"]), float(e2["k_srp"]), emp


def rtn_frame(ref):
    r, v = ref[:, :3], ref[:, 3:]
    Rh = r / np.linalg.norm(r, axis=1)[:, None]
    Nh = np.cross(r, v)
    Nh /= np.linalg.norm(Nh, axis=1)[:, None]
    return Rh, np.cross(Nh, Rh), Nh


def to_rtn(vec, ref):
    return np.column_stack([np.sum(vec * u, axis=1) for u in rtn_frame(ref)])


# ----------------------------------------------------------------------
# step A
# ----------------------------------------------------------------------
def stencil(v_m2, v_m1, v_p1, v_p2, h):
    return (v_m2 - 8.0 * v_m1 + 8.0 * v_p1 - v_p2) / (12.0 * h)


def model_accel(pb, t, states, k, emp):
    from lunar_od import dynamics as dyn
    from lunar_od.srp import SRPOptions
    rhs, _ = dyn._build_state_rhs(t, pb.mu_m, pb.mu_e, pb.mu_s, pb.eph.earth_position, pb.eph.sun_position,
                                  0.0, 0.0, "indirect", pb.model, None, pb.setup.et0, SRPOptions(k_srp_m2_per_kg=k))
    return np.array([np.asarray(rhs(tt, y), float)[3:6] + M.empirical_acceleration(y, emp)
                     for tt, y in zip(t, states)])


def running_mean(x, w):
    """Centred running mean over w samples, window shrunk at the edges."""
    n = len(x)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) + w // 2 + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None] if x.ndim > 1 else (c[hi] - c[lo]) / (hi - lo)


def runs(mask, t):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append([float(t[i]), float(t[j])])
            i = j + 1
        else:
            i += 1
    return out


def main_detect():
    t0 = time.time()
    pb, spans = R.problem()
    et0 = pb.setup.et0
    x0, k, emp = e2_solution()
    t = np.arange(spans["F"][0], spans["P"][1], DT_A)

    # LOLA
    ref = R.lola(et0, t)
    a_mod = model_accel(pb, t, ref, k, emp)

    def lola_res(h):
        return to_rtn(stencil(*[R.lola(et0, t + o)[:, 3:] for o in np.array([-2, -1, 1, 2]) * h], h) - a_mod, ref)
    a_res = lola_res(H_STENCIL)

    # method check on our own E2 trajectory (5 s grid through every stencil point)
    t5 = np.arange(t[0] - 2 * H_STENCIL, t[-1] + 2 * H_STENCIL + 1e-6, H_STENCIL)
    own = pb.propagate(x0, k, t_eval=np.concatenate([[pb.t_epoch], t5]), sensitivity=False, emp=emp)[1:]
    idx = np.searchsorted(t5, t)
    a_own = stencil(*[own[idx + d, 3:] for d in (-2, -1, 1, 2)], H_STENCIL)
    chk = to_rtn(a_own - model_accel(pb, t, own[idx], k, emp), own[idx])
    chk_rms = np.sqrt(np.mean(chk ** 2, axis=0))

    # envelope and taper windows
    r = np.linalg.norm(ref[:, :3], axis=1)
    v2 = np.sum(ref[:, 3:] ** 2, axis=1)
    sma = float(np.mean(1.0 / (2.0 / r - v2 / pb.mu_m)))
    period = 2 * np.pi * np.sqrt(sma ** 3 / pb.mu_m)
    w = int(round(period / DT_A))
    mean_rev = running_mean(a_res, w)
    env = np.sqrt(running_mean(np.sum((a_res - mean_rev) ** 2, axis=1), w))
    thr = ENV_FACTOR * float(np.median(env))
    raw = runs(env > thr, t)
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    tapers = [x for x in merged if x[1] - x[0] >= MIN_TAPER_S]
    lens = [(b - a) / 3600.0 for a, b in tapers]
    gaps = [(tapers[i + 1][0] - tapers[i][0]) / 86400.0 for i in range(len(tapers) - 1)]
    found = bool(tapers) and all(4.0 <= x <= 16.0 for x in lens) and all(1.3 <= g <= 2.7 for g in gaps)
    import spiceypy as spice
    utc = lambda s: spice.et2utc(et0 + s, "ISOC", 0)
    out = dict(spans=dict(F=spans["F"], P=spans["P"]), period_s=period, window_samples=w,
               method_check=dict(rms_rtn_mps2=chk_rms.tolist(), PASS=bool(np.all(chk_rms <= METHOD_CHECK_MAX))),
               a_res_rms_rtn_mps2=np.sqrt(np.mean(a_res ** 2, axis=0)).tolist(),
               envelope_median_mps2=float(np.median(env)), threshold_mps2=thr,
               raw_runs=[[a, b, utc(a), utc(b)] for a, b in raw],
               tapers=[[a, b, utc(a), utc(b)] for a, b in tapers], taper_hours=lens, taper_spacing_days=gaps,
               ARC_STRUCTURE="FOUND" if found else "NOT_FOUND",
               series_5min=dict(t=t[::10].tolist(), env=env[::10].tolist(),
                                mean_rev_rtn=mean_rev[::10].tolist()),
               elapsed_s=time.time() - t0)
    if H1:
        d = float(np.sqrt(np.mean(np.sum((a_res - lola_res(H_NOISE)) ** 2, axis=1))))
        out["h_stencil_s"] = H_STENCIL
        out["spk_noise_check"] = dict(D_mps2=d, limit_mps2=SPK_NOISE_MAX * out["envelope_median_mps2"],
                                      PASS=bool(d <= SPK_NOISE_MAX * out["envelope_median_mps2"]))
    write_part(A_PART, out)
    print(json.dumps({x: out.get(x) for x in ("period_s", "method_check", "spk_noise_check", "a_res_rms_rtn_mps2",
                                              "envelope_median_mps2",
                                          "threshold_mps2", "raw_runs", "tapers", "taper_hours",
                                          "taper_spacing_days", "ARC_STRUCTURE")}, indent=1))


# ----------------------------------------------------------------------
# step B
# ----------------------------------------------------------------------
def windows(a, spans) -> list:
    F0, P1 = spans["F"][0], spans["P"][1]
    tap = [(x[0], x[1]) for x in a["tapers"] if x[1] > F0 and x[0] < P1]
    if a["ARC_STRUCTURE"] == "FOUND":
        edges = [F0] + [v for ab in tap for v in ab] + [P1]
        cores = [(max(edges[i], F0), min(edges[i + 1], P1)) for i in range(0, len(edges), 2)]
        cores = [c for c in cores if c[1] - c[0] >= MIN_CORE_S]
        L = float(np.median([b - a_ for a_, b in cores])) if cores else 86400.0
        L = min(L, P1 - F0)
        out = [dict(name=f"{W_PREFIX}core{i}", kind="core", ws=c[0], we=c[1]) for i, c in enumerate(cores)]
        for i, (a_, b) in enumerate(tap):
            mid = 0.5 * (a_ + b)
            ws = min(max(mid - L / 2, F0), P1 - L)
            out.append(dict(name=f"{W_PREFIX}straddle{i}", kind="straddle", ws=ws, we=ws + L))
        return out
    out, ws, i = [], F0, 0
    while ws + 86400.0 <= P1 + 1e-6:
        out.append(dict(name=f"{W_PREFIX}slide{i}", kind="slide", ws=ws, we=ws + 86400.0))
        ws += 43200.0
        i += 1
    return out


def set_window(pb, ws, we):
    pb.t_epoch = float(ws)
    pb.grid = np.arange(ws, we + M.GRID_S, M.GRID_S)


def _worker_init(ws, we, t_end, tol):
    M._worker_init(300, True, M.F2J_FIT_DAYS, len(M.EMP_NAMES), tol, sorted(M.DOPPLER_EDITS))
    R.apply_variant(M._WORKER["pb"], "base", t_end)
    set_window(M._WORKER["pb"], ws, we)


def fit_window(win, workers):
    t0 = time.time()
    pb, spans = R.problem()
    et0 = pb.setup.et0
    x0e, ke, empe = e2_solution()
    # E2's own residual restricted to this window (E2 epoch, full-span solution)
    ts = np.arange(win["ws"], win["we"] + 1e-6, R.DT_S)
    e2h = pb.propagate(x0e, ke, t_eval=np.concatenate([[pb.t_epoch], ts]), sensitivity=False, emp=empe)[1:]
    ref_all = R.lola(et0, ts)
    e2_rms = np.sqrt(np.mean(to_rtn(e2h[:, :3] - ref_all[:, :3], ref_all) ** 2, axis=0))

    set_window(pb, win["ws"], win["we"])
    if workers > 1:
        pb.pool = ProcessPoolExecutor(workers, initializer=_worker_init,
                                      initargs=(win["ws"], win["we"], spans["P"][1], M.F2K_TOL))
    sel = np.arange(0, pb.grid.size, int(R.DT_S / M.GRID_S))
    ref = R.lola(et0, pb.grid[sel])
    cn = empe[2:]                                    # C_N, S_N fixed at E2's values
    q = np.concatenate([ref[0], [ke], empe[:2]])     # x0 (LOLA at ws), K, C_T, S_T
    pri = [(6, *M.F2H_PRIORS[6]), (7, *M.F2H_PRIORS[11]), (8, *M.F2H_PRIORS[12])]
    scale = np.array([1e3] * 3 + [1.0] * 3 + [1e-3] + [M.EMP_SCALE] * 2)
    sig = R.SIGMA_POS_M

    def emp_of(qq):
        return np.concatenate([qq[7:9], cn])

    def res_jac(qq, partials=True):
        if partials:
            h = pb.fd_history(qq[:6], qq[6], emp_of(qq))
        else:
            h = pb.propagate(qq[:6], qq[6], sensitivity=False, emp=emp_of(qq))
        r = (h[sel, :3] - ref[:, :3]).ravel() / sig
        rp = np.array([(qq[j] - m) / s for j, m, s in pri]) * sig
        if not partials:
            return np.concatenate([r, rp]), None, h
        n = sel.size
        J = np.zeros((n, 3, 9))
        for j in range(6):
            J[:, :, j] = h[sel, 6 + 6 * j:6 + 6 * j + 3]
        J[:, :, 6] = h[sel, 42:45]
        for kk in range(2):
            J[:, :, 7 + kk] = h[sel, 48 + 6 * kk:48 + 6 * kk + 3]
        Jp = np.zeros((len(pri), 9))
        for row, (j, _, s) in enumerate(pri):
            Jp[row, j] = sig / s
        return np.concatenate([r, rp]), np.vstack([J.reshape(-1, 9) / sig, Jp]), h

    lam, log, status = 1e-3, [], "MaxIter"
    r, J, h = res_jac(q)
    cost = float(r @ r)
    for it in range(1, R.MAX_FIT_ITER + 1):
        Js = J * scale
        A, g = Js.T @ Js, Js.T @ r
        dec = float(g @ np.linalg.solve(A, g))
        log.append(dict(iteration=it, cost=cost, rms_pos_m=float(np.sqrt(np.mean(r[:sel.size * 3] ** 2))),
                        gn_decrement=dec, K=float(q[6]), elapsed_s=time.time() - t0))
        print(win["name"], json.dumps(log[-1]), flush=True)
        if dec < R.GN_REL * cost:
            status = "STATIONARY"
            break
        while True:
            qt = q - np.linalg.solve(A + lam * np.diag(np.diag(A)), g) * scale
            qt[6] = max(qt[6], 0.0)
            rt, _, _ = res_jac(qt, partials=False)
            if float(rt @ rt) < cost:
                q, lam = qt, max(lam / 10.0, 1e-9)
                break
            lam *= 10.0
            if lam > 1e10:
                status = "DampingLimit"
                break
        if status == "DampingLimit":
            break
        r, J, h = res_jac(q)
        cost = float(r @ r)
    pb.close_pool()
    write_part(win["name"] + "_q", dict(q=q.tolist(), status=status, iterations=log))
    # ts starts at the window epoch ws = pb.t_epoch: no extra epoch sample (solve_ivp needs strictly increasing t)
    hf = pb.propagate(q[:6], q[6], t_eval=ts, sensitivity=False, emp=emp_of(q))
    fit_rms = np.sqrt(np.mean(to_rtn(hf[:, :3] - ref_all[:, :3], ref_all) ** 2, axis=0))
    out = dict(**win, hours=(win["we"] - win["ws"]) / 3600.0, status=status, iterations=log,
               k_srp=float(q[6]), C_T=float(q[7]), S_T=float(q[8]),
               fit_rms_RTN_m=fit_rms.tolist(), e2_restricted_rms_RTN_m=e2_rms.tolist(), elapsed_s=time.time() - t0)
    write_part(win["name"], out)
    print(win["name"], "DONE", status, "fit RTN", fit_rms.round(3).tolist(), "E2 RTN", e2_rms.round(3).tolist(),
          flush=True)


def main_fit_all(workers):
    a = read_part(A_PART)
    wins = windows(a, a["spans"])
    i, n = (map(int, sys.argv[sys.argv.index("--shard") + 1].split("/")) if "--shard" in sys.argv else (0, 1))
    for k, w in enumerate(wins):
        if k % n == i and not (PARTS / f"{w['name']}.json").exists():
            fit_window(w, workers)


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
def main_summary():
    a = read_part(A_PART)
    checks = a["method_check"]["PASS"] and a.get("spk_noise_check", {}).get("PASS", True)
    # a failed check is inconclusive whatever the fits give, so step B is not needed for the label
    names = [w["name"] for w in windows(a, a["spans"])]
    wins = [read_part(x) for x in names if (PARTS / f"{x}.json").exists() or checks]
    ok = [w for w in wins if w["status"] == "STATIONARY"]
    N = lambda w: w["fit_rms_RTN_m"][2]
    res = dict(method_check=a["method_check"]["PASS"], ARC_STRUCTURE=a["ARC_STRUCTURE"],
               spk_noise_check=a.get("spk_noise_check"), n_windows=len(wins), n_converged=len(ok))
    if not checks or len(ok) * 2 < len(wins):
        label = "F2S1_INCONCLUSIVE" if H1 else "F2S_INCONCLUSIVE"
    elif a["ARC_STRUCTURE"] == "FOUND":
        core = [w for w in ok if w["kind"] == "core"]
        strad = [w for w in ok if w["kind"] == "straddle"]
        mc = float(np.median([N(w) for w in core]))
        ms = float(np.median([N(w) for w in strad])) if strad else float("nan")
        me2 = float(np.median([w["e2_restricted_rms_RTN_m"][2] for w in core]))
        res.update(median_core_N=mc, median_straddle_N=ms, median_e2_core_N=me2, straddle_over_core=ms / mc)
        if mc <= H_REF_CORE_MAX and ms >= 2 * mc:
            label = "H_REF_SUPPORTED"
        elif mc >= 0.5 * me2 and ms / mc < 1.5:
            label = "H_DYN_SUPPORTED"
        else:
            label = "MIXED"
    else:
        ns = [N(w) for w in ok]
        res.update(min_N=min(ns), max_N=max(ns), max_over_min=max(ns) / min(ns))
        if max(ns) / min(ns) >= 3 and min(ns) <= H_REF_CORE_MAX:
            label = "H_REF_SUPPORTED"
        elif max(ns) / min(ns) < 1.5 and min(ns) > H_REF_CORE_MAX:
            label = "H_DYN_SUPPORTED"
        else:
            label = "MIXED"
    out = dict(phase="F2s-1" if H1 else "F2s",
               predeclaration=("configs/phase_f2s1_stencil_step_amendment_predeclaration.json + " if H1 else "")
               + "configs/phase_f2s_lola_arc_structure_predeclaration.json",
               classification=label, decision_inputs=res,
               step_A={x: y for x, y in a.items() if x != "series_5min"},
               windows=[{x: y for x, y in w.items() if x != "iterations"} | dict(n_iter=len(w["iterations"]))
                        for w in wins])
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(classification=label, **res), indent=1))
    for w in out["windows"]:
        print(w["name"], w["status"], round(w["hours"], 1), "fit", np.round(w["fit_rms_RTN_m"], 2).tolist(),
              "E2", np.round(w["e2_restricted_rms_RTN_m"], 2).tolist(), "K", round(w["k_srp"], 4))


if __name__ == "__main__":
    if "--detect" in sys.argv:
        main_detect()
    elif "--fit-all" in sys.argv:
        main_fit_all(M._workers_arg() or 8)
    elif "--summary" in sys.argv:
        main_summary()
