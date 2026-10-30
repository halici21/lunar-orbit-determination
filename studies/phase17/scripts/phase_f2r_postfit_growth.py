"""Phase F2r: post-fit orbit-error growth against LOLA, dynamics root-cause isolation.

Predeclaration: studies/phase17/configs/phase_f2r_postfit_growth_predeclaration.json.  Analysis only: every
propagation goes through phase_f2_lro_od.Problem (the F2 study driver), every substitution is applied to a
Problem after it is built (main process and workers alike), and no production code changes.

Modes (each writes one part file under results/phase_f2r_parts/; --summary merges them):
  --e0            E0 baseline (F2p, rtol 1e-12) and E1 integrator check (rtol 1e-13)
  --e2 VARIANT    fit our dynamics to LOLA positions over the fit interval, then predict; VARIANT is
                  base (E2), de421 (E3) or grgm1200l (E4)
  --e2b           E2b: F2p's state with E2's parameters and the converse
  --summary       classification as predeclared
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2k_tight_tolerance as K  # noqa: E402
import phase_f2l_lola_scoring as L  # noqa: E402

F = M.F
PARTS = HERE.parent / "results" / "phase_f2r_parts"
OUT = HERE.parent / "results" / "phase_f2r_postfit_growth.json"
DT_S = 60.0
SIGMA_POS_M = 1.0
E1_TOL = (1e-13, 1e-14)
MAX_FIT_ITER = 10
GN_REL = 1e-3


# ----------------------------------------------------------------------
# set-up: Problem, substitutions, intervals
# ----------------------------------------------------------------------
def apply_variant(pb, variant: str, t_end: float) -> None:
    """E3 / E4 substitutions on a built Problem; always (re)samples Earth/Sun up to t_end."""
    import spiceypy as spice
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    if variant == "de421":
        from lunar_od.spice_loader import resolve_kernel_dir
        spice.furnsh(str(Path(resolve_kernel_dir()) / "de421.bsp"))    # loaded last: takes precedence
        pb.model = dataclasses.replace(pb.model, frame="MOON_PA_DE421")
    elif variant == "grgm1200l":
        from lunar_od.gravity_model_loader import load_lunar_gravity_model
        from lunar_od.project_paths import gravity_data_root
        model = load_lunar_gravity_model(gravity_data_root() / "grgm1200l" / "gggrx_1200l_sha.tab",
                                         nmax=pb.nmax, mmax=pb.nmax)
        pb.model = dataclasses.replace(model, frame=pb.model.frame)
        pb.mu_m = float(pb.model.mu_m3_s2)
    elif variant != "base":
        raise ValueError(variant)
    pb.eph = sample_moon_centered_ephemeris(pb.setup.et0, np.arange(pb.t_epoch - 600.0, t_end + 600.0, 60.0))


def _worker_init(variant, t_end, tol):
    M._worker_init(300, True, M.F2J_FIT_DAYS, len(M.EMP_NAMES), tol, sorted(M.DOPPLER_EDITS))
    apply_variant(M._WORKER["pb"], variant, t_end)


def problem(variant="base", tol=M.F2K_TOL, workers=0):
    import warnings
    warnings.simplefilter("ignore")
    M.set_tolerance(*tol)
    M.set_doppler_edits(M.f2o_edits())
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    spans = intervals(pb)
    apply_variant(pb, variant, spans["P"][1])
    if workers > 1:
        pb.pool = ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(variant, spans["P"][1], tol))
    return pb, spans


def intervals(pb) -> dict:
    """F = fit-day measurement span, P = day-263 measurement span (TDB s from et0), Doppler mid-times per day."""
    from lunar_od import station_time as ST
    import phase_f1c_lro_doppler as D
    s = pb.setup
    out = dict(F=(float(min([s.t3.min()] + [d["tm"] - d["tc"] for d in pb.dop])),
                  float(max([s.t3.max()] + [d["tm"] + d["tc"] for d in pb.dop]))), mids={})
    t_lo, t_hi = [], []
    for day in L.DAYS:
        rows = F.read_day(pb.data, day)[0]
        t3 = [ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0] for q, r in rows]
        drows = D.read_doppler(pb.data, day)
        drows.sort(key=lambda x: (x[0], x[1].utc))
        td = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                       for q, r in drows])
        mids = []
        for i, j in D.intervals(drows, td):
            q, ra = drows[i]
            rb = drows[j][1]
            a = (rb.doppler_count_raw - ra.doppler_count_raw) / (rb.utc - ra.utc).total_seconds()
            obs = -F.C * (a - D.BIAS_HZ) / (2.0 * ra.transmit_frequency_hz * D.TURNAROUND * D.M_MULT)
            if abs(obs) > M.MAX_DOPPLER_MPS or (q, ra.utc.isoformat()) in M.DOPPLER_EDITS:
                continue
            mids.append((int(q), float(0.5 * (td[i] + td[j]))))
        out["mids"][day.rsplit("/", 1)[1]] = mids
        if day in M.F2J_PREDICT_DAYS:
            t_lo += [min(t3), td.min()]
            t_hi += [max(t3), td.max()]
    out["P"] = (float(min(t_lo)), float(max(t_hi)))
    return out


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def lola(et0, t):
    import spiceypy as spice
    return np.array([spice.spkezr(F.LRO, et0 + x, "J2000", "NONE", "MOON")[0] for x in t]) * 1e3


def rtn(est, ref):
    """Delta = est - ref on ref's R, T, N: (n, 6) = position R T N, velocity R T N."""
    r, v = ref[:, :3], ref[:, 3:]
    R = r / np.linalg.norm(r, axis=1)[:, None]
    N = np.cross(r, v)
    N /= np.linalg.norm(N, axis=1)[:, None]
    T = np.cross(N, R)
    d = est - ref
    return np.column_stack([np.sum(d[:, :3] * u, axis=1) for u in (R, T, N)]
                           + [np.sum(d[:, 3:] * u, axis=1) for u in (R, T, N)])


def los(et0, t, sc):
    import spiceypy as spice
    e = np.array([spice.spkpos("EARTH", et0 + x, "J2000", "NONE", "MOON")[0] for x in t]) * 1e3
    u = sc[:, :3] - e
    return u / np.linalg.norm(u, axis=1)[:, None]


def metrics(t, est, ref, spans, et0) -> dict:
    """Baseline-relative growth over P against the mean over F (predeclared definitions)."""
    d = rtn(est, ref)
    dop = np.sum((est[:, 3:] - ref[:, 3:]) * los(et0, t, ref), axis=1) * 1e3      # mm/s
    f = (t >= spans["F"][0]) & (t <= spans["F"][1])
    p = (t >= spans["P"][0]) & (t <= spans["P"][1])
    names = ("R", "T", "N", "vR", "vT", "vN")
    b = d[f].mean(axis=0)
    g = d - b
    out = dict(baseline={n: float(x) for n, x in zip(names, b)},
               fit_rms={n: float(x) for n, x in zip(names, np.sqrt(np.mean(d[f] ** 2, axis=0)))},
               G={n: float(x) for n, x in zip(names, np.sqrt(np.mean(g[p] ** 2, axis=0)))},
               growth_mean_P={n: float(x) for n, x in zip(names, g[p].mean(axis=0))},
               raw_rms_P={n: float(x) for n, x in zip(names, np.sqrt(np.mean(d[p] ** 2, axis=0)))})
    q = t >= spans["F"][1]
    x = (t[q] - spans["F"][1]) / 3600.0
    out["shape_T"] = dict(linear_m_per_h=np.polyfit(x, g[q, 1], 1).tolist(),
                          quadratic_m_per_h2=np.polyfit(x, g[q, 1], 2).tolist())
    db = dop[f].mean()
    out["doppler_proj"] = dict(baseline_mmps=float(db), G_mmps=float(np.sqrt(np.mean((dop[p] - db) ** 2))),
                               growth_mean_P_mmps=float((dop[p] - db).mean()))
    out["series_T_P"] = g[p, 1].tolist()            # for the shape correlation between experiments
    return out


def grid_for(pb, spans, extra=()):
    t = np.arange(spans["F"][0], spans["P"][1], DT_S)
    t = np.unique(np.concatenate([t, np.asarray(extra, float)]))
    return np.concatenate([[pb.t_epoch], t])


def propagate_and_score(pb, spans, p, extra=()):
    t = grid_for(pb, spans, extra)
    hist = pb.propagate_p(p, t_eval=t)[1:]
    t = t[1:]
    return t, hist


def write_part(name, obj):
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")


def f2p_vector():
    return K.point_vector(json.loads(M.OUT_F2P.read_text(encoding="utf-8"))["runs"]["S1"])


# ----------------------------------------------------------------------
# E0 / E1
# ----------------------------------------------------------------------
def _e0_case(tol):
    t0 = time.time()
    pb, spans = problem(tol=tol)
    mids = [m for d in spans["mids"].values() for _, m in d]
    t, est = propagate_and_score(pb, spans, f2p_vector(), extra=mids)
    ref = lola(pb.setup.et0, t)
    out = dict(tol=tol, metrics=metrics(t, est, ref, spans, pb.setup.et0), spans=dict(F=spans["F"], P=spans["P"]))
    if tol == M.F2K_TOL:
        # Doppler projection check: pass means of -(Delta_v . u) at the interval mid-times vs F2q
        sel = np.isin(t, mids)
        d = -np.sum((est[sel, 3:] - ref[sel, 3:]) * los(pb.setup.et0, t[sel], ref[sel]), axis=1) * 1e3
        at = dict(zip(t[sel], d))
        f2q = json.loads((HERE.parent / "results" / "phase_f2q_doppler_offset_diagnostic.json").read_text(
            encoding="utf-8"))["passes"]
        mine = []
        for day, ms in spans["mids"].items():
            pads = [q for q, _ in ms]
            tm = np.array([m for _, m in ms])
            for q, g in L._groups(pads, tm):
                mine.append(dict(day=day, pad=int(q), proj_mmps=float(np.mean([at[x] for x in tm[g]]))))
        ref_q = [dict(day=x["day"], pad=x["pad"], f2q_mmps=x["F2p"]["mean"] - x["LOLA"]["mean"]) for x in f2q]
        assert [(a["day"], a["pad"]) for a in mine] == [(b["day"], b["pad"]) for b in ref_q]
        diff = np.array([a["proj_mmps"] - b["f2q_mmps"] for a, b in zip(mine, ref_q)])
        out["doppler_projection_check"] = dict(
            passes=[{**a, "f2q_mmps": b["f2q_mmps"]} for a, b in zip(mine, ref_q)],
            rms_diff_mmps=float(np.sqrt(np.mean(diff ** 2))), PASS=bool(np.sqrt(np.mean(diff ** 2)) <= 0.5))
    out["elapsed_s"] = time.time() - t0
    return out


def main_e0():
    with ProcessPoolExecutor(2) as ex:
        e0, e1 = ex.map(_e0_case, [M.F2K_TOL, E1_TOL])
    G0, G1 = e0["metrics"]["G"]["T"], e1["metrics"]["G"]["T"]
    m0, m1 = e0["metrics"]["growth_mean_P"]["T"], e1["metrics"]["growth_mean_P"]["T"]
    e1["contribution"] = dict(dG_T=G1 - G0, dmean_T=m1 - m0,
                              negligible=bool(abs(G1 - G0) < max(0.05 * G0, 0.0) and abs(G1 - G0) < 0.1
                                              and abs(m1 - m0) < 0.1))
    write_part("E0", e0)
    write_part("E1", e1)
    for k, v in (("E0", e0), ("E1", e1)):
        print(k, json.dumps({x: v["metrics"][x] for x in ("baseline", "G", "growth_mean_P", "shape_T", "doppler_proj")}))
    print("E1 contribution", e1["contribution"])
    print("Doppler projection check", e0["doppler_projection_check"]["rms_diff_mmps"],
          e0["doppler_projection_check"]["PASS"])


# ----------------------------------------------------------------------
# E2 (fit to LOLA) and its E3 / E4 variants
# ----------------------------------------------------------------------
def main_e2(variant: str, workers: int):
    t0 = time.time()
    pb, spans = problem(variant, workers=workers)
    et0 = pb.setup.et0
    sel = np.flatnonzero((pb.grid >= spans["F"][0]) & (pb.grid <= spans["F"][1]))[::int(DT_S / M.GRID_S)]
    ref = lola(et0, pb.grid[sel])[:, :3]
    p = f2p_vector()
    q = np.concatenate([p[:7], p[11:15]])            # x0, K, C_T, S_T, C_N, S_N
    # F2p's priors on K (p index 6) and the empirical terms (p 11..14 -> q 7..10); bias priors do not apply
    pri = [(6 if j == 6 else j - 4, m, s) for j, (m, s) in sorted(M.F2H_PRIORS.items()) if j == 6 or j >= 11]
    scale = np.array([1e3] * 3 + [1.0] * 3 + [1e-3] + [M.EMP_SCALE] * 4)

    def res_jac(qq, partials=True):
        if partials:
            h = pb.fd_history(qq[:6], qq[6], qq[7:])
        else:
            h = pb.propagate(qq[:6], qq[6], sensitivity=False, emp=qq[7:])
        r = (h[sel, :3] - ref).ravel() / SIGMA_POS_M
        rp = np.array([(qq[j] - m) / s for j, m, s in pri]) * SIGMA_POS_M
        if not partials:
            return np.concatenate([r, rp]), None
        n = sel.size
        J = np.zeros((n, 3, 11))
        for j in range(6):
            J[:, :, j] = h[sel, 6 + 6 * j:6 + 6 * j + 3]
        J[:, :, 6] = h[sel, 42:45]
        for k in range(4):
            J[:, :, 7 + k] = h[sel, 48 + 6 * k:48 + 6 * k + 3]
        Jp = np.zeros((len(pri), 11))
        for row, (j, _, s) in enumerate(pri):
            Jp[row, j] = SIGMA_POS_M / s
        return np.concatenate([r, rp]), np.vstack([J.reshape(-1, 11) / SIGMA_POS_M, Jp])

    lam, hist_log, status = 1e-3, [], "MaxIter"
    r, J = res_jac(q)
    cost = float(r @ r)
    for it in range(1, MAX_FIT_ITER + 1):
        Js = J * scale
        A, g = Js.T @ Js, Js.T @ r
        dec = float(g @ np.linalg.solve(A, g))
        hist_log.append(dict(iteration=it, cost=cost, rms_pos_m=float(np.sqrt(np.mean(r[:sel.size * 3] ** 2))),
                             gn_decrement=dec, K=float(q[6]), elapsed_s=time.time() - t0))
        print(json.dumps(hist_log[-1]), flush=True)
        if dec < GN_REL * cost:
            status = "STATIONARY"
            break
        while True:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A)), g) * scale
            qt = q + step
            qt[6] = max(qt[6], 0.0)
            rt, _ = res_jac(qt, partials=False)
            ct = float(rt @ rt)
            if ct < cost:
                q, lam = qt, max(lam / 10.0, 1e-9)
                break
            lam *= 10.0
            if lam > 1e10:
                status = "DampingLimit"
                break
        if status == "DampingLimit":
            break
        r, J = res_jac(q)
        cost = float(r @ r)
    pb.close_pool()
    pv = np.concatenate([q[:7], p[7:11], q[7:]])
    t, est = propagate_and_score(pb, spans, pv)
    out = dict(variant=variant, status=status, iterations=hist_log, x0=q[:6].tolist(), k_srp=float(q[6]),
               empirical_mps2=dict(zip(M.EMP_NAMES, q[7:].tolist())), n_samples=int(sel.size),
               metrics=metrics(t, est, lola(et0, t), spans, et0), elapsed_s=time.time() - t0)
    name = dict(base="E2", de421="E3", grgm1200l="E4")[variant]
    write_part(name, out)
    print(name, json.dumps({x: out["metrics"][x] for x in ("baseline", "fit_rms", "G", "growth_mean_P", "shape_T",
                                                            "doppler_proj")}))


def main_e2b():
    pb, spans = problem()
    et0 = pb.setup.et0
    p = f2p_vector()
    e2 = json.loads((PARTS / "E2.json").read_text(encoding="utf-8"))
    par2 = np.array([e2["k_srp"]] + [e2["empirical_mps2"][n] for n in M.EMP_NAMES])
    out = {}
    for name, x0, par in (("F2p_state_E2_params", p[:6], par2),
                          ("E2_state_F2p_params", np.array(e2["x0"]), np.concatenate([[p[6]], p[11:15]]))):
        pv = np.concatenate([x0, par[:1], p[7:11], par[1:]])
        t, est = propagate_and_score(pb, spans, pv)
        out[name] = metrics(t, est, lola(et0, t), spans, et0)
        print(name, out[name]["G"]["T"], out[name]["growth_mean_P"]["T"])
    write_part("E2b", out)


if __name__ == "__main__":
    if "--e0" in sys.argv:
        main_e0()
    elif "--e2" in sys.argv:
        main_e2(sys.argv[sys.argv.index("--e2") + 1], M._workers_arg() or 8)
    elif "--e2b" in sys.argv:
        main_e2b()
