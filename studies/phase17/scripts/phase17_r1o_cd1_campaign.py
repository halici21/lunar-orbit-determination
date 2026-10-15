"""PHASE 17-R1O-CD1 - production counted-Doppler K_SRP complementarity: campaign.

Every rule comes from artifacts/phase17_r1o_cd1_predeclaration.json (hash-pinned,
committed before any CD1 K metric).  Matrix: 5 orbits x 24 D3 epochs x
{HIGH_FIDELITY (primary), SIMPLIFIED (control)}; production range baseline
recomputed on every propagation; production two-way counted Doppler.

    python examples/phase17_r1o_cd1_campaign.py            # campaign + FD + analysis
    python examples/phase17_r1o_cd1_campaign.py --analyse-only
"""
from __future__ import annotations

import argparse
import csv
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
import phase17_r1o_cd1_core as CD  # noqa: E402
import phase17_r1od3_core as D3  # noqa: E402

D2 = D3.D2
ARTIFACTS = CD.ARTIFACTS
PREDECLARATION_SHA256 = "bbaa5cee8687926699465de6fe75185fde87b10df840be652ba390ff7af3571a"
ORBITS = ("CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA", "O3_HIGH_ALTITUDE_500KM",
          "O5_FAVOURABLE_GEO_MAX_FPERP")
EPOCHS = tuple(range(24))
REP_CELLS = (("CANON", 0), ("CANON", 6), ("CANON", 12), ("CANON", 18))
FD_CASES = (("CANON", 0), ("O2_FULL_SUN_HIGH_BETA", 12))
FD_STEPS = {0: (10.0, 100.0), 1: (10.0, 100.0), 2: (10.0, 100.0), 3: (0.01, 0.1), 4: (0.01, 0.1), 5: (0.01, 0.1),
            6: (1e-3, 3e-3)}
FD_NAMES = ("x0", "y0", "z0", "vx0", "vy0", "vz0", "K_SRP")
N_WORKERS = 8          # 15 exhausted memory while each worker parsed GL1800F (MemoryError)
PRIORS = {"FREE": None, "1e-05": 1e-5, "0.0001": 1e-4, "0.001": 1e-3}
KEEP = ("n_obs", "n_nuisance", "f_perp", "theta_k_deg", "i_kk", "conditional_k_information", "k_column_norm",
        "sigma_k", "sigma_k_frac", "design_condition", "weakest_mode_k_component", "max_abs_k_state_correlation",
        "max_abs_k_bias_corr", "max_abs_k_drift_corr", "bias_sigma_mps_median")


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    D2.install_legacy_earth_guard()


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, np.ndarray):
        return _clean(d.tolist())
    if isinstance(d, np.generic):
        return d.item()
    if isinstance(d, float) and not math.isfinite(d):
        return None if math.isnan(d) else ("inf" if d > 0 else "-inf")
    return d


def _pick(m: dict, full: bool = False) -> dict:
    out = {k: m[k] for k in KEEP if k in m}
    if full:
        out.update(design_singular_values=m["design_singular_values"],
                   weakest_right_singular_vector=m["weakest_right_singular_vector"])
    return out


def stations():
    by = D2.stations_by_name()
    return [by[n] for n in D2.DSN3]


def scenario(orbit: str, x0, k: int, n_orbits: float = 15.0, k_srp: float | None = None):
    grid = D3.load_grid()
    _, t_orbit = D2.M.campaign_epoch()
    scn = D2.Scenario("CD1_%s_k%02d" % (orbit, k), "cd1", D3.epoch_et(k, grid), tuple(x0), n_orbits * t_orbit)
    return scn if k_srp is None else dataclasses.replace(scn, k_srp=k_srp)


def propagate_hf(scn, rtol: float = 1e-12, atol: float = 1e-13) -> np.ndarray:
    """phase17_r1od2_fidelity.propagate_fidelity with the tolerances exposed (FD oracle).
    At rtol 1e-12 / atol 1e-13 it must reproduce propagate_fidelity bitwise (checked)."""
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    from phase17_r1od2_fidelity import NMAX
    return propagate_state_with_k_sensitivity(
        scn.t_grid, scn.x0_array, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2,
        lambda t: G.earth_state_mci(scn.et0 + float(t))[:3],
        lambda t: G.sun_mci(scn.et0 + float(t)),
        srp=SRPOptions(k_srp_m2_per_kg=scn.k_srp, shadow_model=scn.shadow_model),
        rtol=rtol, atol=atol, j2_moon=0.0,
        harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0, harmonic_stm_opt_in=True)


# ======================================================================
# one cell
# ======================================================================
def run_cell(spec: dict) -> dict:
    t0 = time.time()
    orbit, k, dyn = spec["orbit"], spec["k"], spec["dynamics"]
    scn = scenario(orbit, spec["x0"], k)
    if dyn == "HIGH_FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    audit = D2.earth_epoch_audit(scn, rb)
    st = stations()
    m_r = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    f0 = m_r["f_perp"]
    tcs = (60.0, 10.0, 540.0) if dyn == "HIGH_FIDELITY" else (60.0,)
    rows = {tc: CD.build_doppler_rows(scn, rb, tc, st, with_values=False) for tc in tcs}
    r60 = rows[60.0]
    out = dict(orbit=orbit, k=k, dynamics=dyn, et0=scn.et0, range=_pick(m_r, True), range_n=int(rb.n_obs),
               doppler_n={str(tc): r.n for tc, r in rows.items()},
               doppler_candidates={str(tc): [r.n_candidates, r.n_dropped_visibility, r.n_dropped_domain]
                                   for tc, r in rows.items()},
               n_sessions=r60.n_sessions, feasible=bool(r60.n >= 7), cfg={})
    geo = D2.geometry_descriptors(scn, rb)
    out["geometry"] = dict(sun_moon_earth_deg=geo["sun_moon_earth_deg"], beta_deg=geo["beta_deg"],
                           moon_dec_deg=geo["moon_geocentric_declination_deg"],
                           sep_deg_median=float(np.median(r60.sep_deg)) if r60.n else None)
    # ---------------- contracts (no K information)
    vis_ok = True
    for i in range(r60.n):
        lo, hi = CD.count_interval_nodes(int(r60.node[i]), 60.0, rb.t_grid)
        vis_ok &= bool(np.all(rb.vis[lo:hi + 1, r60.station[i]]))
    nu1 = CD.nuisance_columns(rb.n_obs, r60, "D1_SESSION_CONSTANT_BIAS")
    own_ok = bool(np.all(nu1[:rb.n_obs] == 0) and np.all(nu1[rb.n_obs:].sum(axis=1) == 1)
                  and all(np.array_equal(np.where(nu1[rb.n_obs:, j] == 1)[0], np.where(r60.session == j)[0])
                          for j in range(r60.n_sessions)))
    w60 = np.full(r60.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
    out["contract"] = dict(earth_audit_passed=bool(audit["passed"]), doppler_count_intervals_visible=vis_ok,
                           nuisance_ownership=own_ok, doppler_whitening_exact=bool(np.all(w60 == 1e8)),
                           range_whitening_exact=bool(np.all(rb.w == 1.0 / 25.0)))
    if not out["feasible"]:
        out["elapsed_s"] = time.time() - t0
        return _clean(out)

    def put(name, m, full=False):
        d = _pick(m, full)
        d["delta_f_perp"] = m["f_perp"] - f0
        out["cfg"][name] = d

    # ---------------- primary blocks
    put("DOP|D0", CD.metrics(D2, rb, r60, w60, "D0_IDEAL", None, with_range=False), True)
    put("DOP|D1|FREE", CD.metrics(D2, rb, r60, w60, "D1_SESSION_CONSTANT_BIAS", None, with_range=False))
    put("RPD|D0", CD.metrics(D2, rb, r60, w60, "D0_IDEAL", None), True)
    for pn, pv in PRIORS.items():
        put("RPD|D1|%s" % pn, CD.metrics(D2, rb, r60, w60, "D1_SESSION_CONSTANT_BIAS", pv), pn == "FREE")
    if dyn == "HIGH_FIDELITY":
        for pn, pv in PRIORS.items():
            put("RPD|D2|%s" % pn, CD.metrics(D2, rb, r60, w60, "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT", pv))
        # ---------------- noise bracket, operational lower bound, count interval
        for s in CD.SIGMA_D60_BRACKET_MPS:
            w = np.full(r60.n, 1.0 / s ** 2)
            put("NOISE|%g|D0" % s, CD.metrics(D2, rb, r60, w, "D0_IDEAL", None))
            put("NOISE|%g|D1FREE" % s, CD.metrics(D2, rb, r60, w, "D1_SESSION_CONSTANT_BIAS", None))
        w_op = 1.0 / CD.operational_lower_bound_sigma(r60.sep_deg) ** 2
        put("OPLB|D0", CD.metrics(D2, rb, r60, w_op, "D0_IDEAL", None))
        put("OPLB|D1FREE", CD.metrics(D2, rb, r60, w_op, "D1_SESSION_CONSTANT_BIAS", None))
        out["oplb_sigma_mps"] = [float(np.min(1 / np.sqrt(w_op))), float(np.median(1 / np.sqrt(w_op))),
                                 float(np.max(1 / np.sqrt(w_op)))]
        for tc in (10.0, 540.0):
            r = rows[tc]
            if r.n < 7:
                continue
            for sn, ex in CD.TC_NOISE_SCALINGS.items():
                w = np.full(r.n, 1.0 / CD.sigma_tc(CD.SIGMA_D60_PRIMARY_MPS, tc, ex) ** 2)
                put("TC|%g|%s|D0" % (tc, sn), CD.metrics(D2, rb, r, w, "D0_IDEAL", None))
                put("TC|%g|%s|D1FREE" % (tc, sn), CD.metrics(D2, rb, r, w, "D1_SESSION_CONSTANT_BIAS", None))
        # ---------------- secondary schedule
        rs = r60.subset(CD.regular_session_mask(r60))
        out["regular_session_n"] = rs.n
        if rs.n >= 7:
            ws = np.full(rs.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
            put("REG|RPD|D0", CD.metrics(D2, rb, rs, ws, "D0_IDEAL", None))
            put("REG|RPD|D1FREE", CD.metrics(D2, rb, rs, ws, "D1_SESSION_CONSTANT_BIAS", None))
            put("REG|DOP|D0", CD.metrics(D2, rb, rs, ws, "D0_IDEAL", None, with_range=False))
        # ---------------- independent rebuild
        rebuild = {}
        for name, model, prior, wr in (("DOP|D0", "D0_IDEAL", None, False), ("RPD|D0", "D0_IDEAL", None, True),
                                       ("RPD|D1|FREE", "D1_SESSION_CONSTANT_BIAS", None, True),
                                       ("RPD|D2|FREE", "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT", None, True),
                                       ("RPD|D1|0.0001", "D1_SESSION_CONSTANT_BIAS", 1e-4, True)):
            f_ind = CD.independent_f_perp(D2, rb, r60, w60, model, prior, with_range=wr)
            rebuild[name] = abs(f_ind - out["cfg"][name]["f_perp"])
        rebuild["RANGE"] = abs(CD.independent_f_perp(D2, rb, dataclasses.replace(
            r60, h_x0=np.zeros((0, 6)), h_k=np.zeros(0), session=np.zeros(0, int), t_mid=np.zeros(0)),
            np.zeros(0), "D0_IDEAL", None) - f0)
        out["rebuild_f_perp"] = rebuild
        out["rebuild_rows"] = rebuild_rows(rb, r60, st)
        if (orbit, k) in REP_CELLS:
            out["interpretation"] = interpret(scn, rb, r60, w60)
    out["elapsed_s"] = time.time() - t0
    return _clean(out)


def rebuild_rows(rb, r60, st) -> dict:
    """Doppler rows via the R4 four-event module (M3 3x3 implicit sensitivity) at zero
    delay, and range K rows via the event-epoch STM-slot route."""
    from lunar_od.radiometrics import RangeRatePhysicsConfig
    from lunar_od.two_way_counted_doppler import four_event_counted_doppler_initial_state_jacobian
    from lunar_od.two_way_range import two_way_range_nominal_and_initial_jacobian
    pg = rb.pass_geo
    cfg4 = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0,
                                  counted_doppler_model="four_event_delay")
    kh = CD.k_history(rb.nom48)
    idx = np.unique(np.linspace(0, r60.n - 1, 12).round().astype(int))
    ds, dk = [], []
    for i in idx:
        s = st[r60.station[i]]
        args = (float(r60.t_mid[i]), s, rb.t_grid)
        tail = (pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93, cfg4)
        h4 = four_event_counted_doppler_initial_state_jacobian(*args, rb.nom48[:, :42], *tail, et0_s=pg.et0_s)
        k4 = four_event_counted_doppler_initial_state_jacobian(*args, kh, *tail, et0_s=pg.et0_s)[0]
        ds.append(float(np.linalg.norm(h4 - r60.h_x0[i]) / np.linalg.norm(r60.h_x0[i])))
        dk.append(abs(float(k4) - float(r60.h_k[i])))
    sel = np.unique(np.linspace(0, rb.n_obs - 1, 12).round().astype(int))
    _, hk = two_way_range_nominal_and_initial_jacobian(rb.obs[sel], pg, kh)
    kmax_d = float(np.max(np.abs(r60.h_k))) or 1.0
    kmax_r = float(np.max(np.abs(rb.h_k))) or 1.0
    return dict(doppler_state_rel_max=max(ds), doppler_k_abs_over_colmax=max(dk) / kmax_d,
                range_k_abs_over_colmax=float(np.max(np.abs(np.asarray(hk)[:, 0] - rb.h_k[sel]))) / kmax_r)


def interpret(scn, rb, r60, w60) -> dict:
    """Representative cells only: time localization and physical signature (interpretation)."""
    import phase17_geo_core as G
    pg = rb.pass_geo
    t_end = float(rb.t_grid[-1])
    t_r = np.asarray(rb.obs[:, 0], float)
    cum = []
    for frac in np.arange(1, 11) / 10.0:
        tau = frac * t_end
        mr, md = t_r <= tau, r60.t_mid <= tau
        rr = dict(tau_frac=float(frac), n_range=int(mr.sum()), n_doppler=int(md.sum()))
        if mr.sum() >= 7:
            m = D2.info_metrics([(rb.h_x0[mr], rb.h_k[mr], rb.w[mr])])
            rr.update(range_f=m["f_perp"], range_iperp=m["conditional_k_information"])
        if md.sum() >= 7:
            sub = r60.subset(md)
            m = D2.info_metrics([(sub.h_x0, sub.h_k, w60[md])])
            rr.update(dop_f=m["f_perp"], dop_iperp=m["conditional_k_information"])
            if mr.sum():
                m = D2.info_metrics([(rb.h_x0[mr], rb.h_k[mr], rb.w[mr]), (sub.h_x0, sub.h_k, w60[md])])
                rr.update(comb_f=m["f_perp"], comb_iperp=m["conditional_k_information"])
        cum.append(rr)
    final = cum[-1].get("comb_iperp")
    prev, shares = 0.0, []
    for c in cum:
        v = c.get("comb_iperp", 0.0) or 0.0
        shares.append((v - prev) / final if final else float("nan"))
        prev = v
    loc = "LOCALIZED" if final and max(shares) > 0.5 else "BROAD"
    # physical signature
    earth = pg.earth_pos_mci_m
    def los(node):
        g = rb.nom48[node, :3] - earth[node]
        return g / np.linalg.norm(g)
    u_d = np.array([los(n) for n in r60.node])
    dv = np.einsum("ij,ij->i", u_d, rb.nom48[r60.node, 45:48])
    nodes_r = np.asarray(rb.obs[:, 3], int) - 1
    u_r = np.array([los(n) for n in nodes_r])
    dr = np.einsum("ij,ij->i", u_r, rb.nom48[nodes_r, 42:45])
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")  # noqa: E731
    # Doppler-row orthogonal energy in the combined D0 projection
    sw = np.sqrt(np.concatenate([rb.w, w60]))
    a = sw[:, None] * np.vstack([rb.h_x0, r60.h_x0])
    b = sw * np.concatenate([rb.h_k, r60.h_k])
    q, _ = np.linalg.qr(a)
    bp = b - q @ (q.T @ b)
    e_d = bp[rb.n_obs:] ** 2
    hist = G.srp_history(rb.nom48, rb.t_grid, scn.et0)
    lit = hist["nu"] >= 0.5
    trans = rb.t_grid[1:][lit[1:] != lit[:-1]]
    near_ecl = np.array([bool(trans.size) and np.min(np.abs(trans - t)) <= 3600.0 for t in r60.t_mid])
    near_edge = np.array([min(abs(t - r60.session_window[s][0]), abs(t - r60.session_window[s][1])) <= 600.0
                          for t, s in zip(r60.t_mid, r60.session)])
    share = lambda m: float(e_d[m].sum() / e_d.sum()) if e_d.sum() > 0 else float("nan")  # noqa: E731
    return dict(
        cumulative=cum, slice_share_of_final_comb_iperp=shares, TIME_LOCALIZATION=loc,
        corr_doppler_hk_vs_los_dvdk=corr(r60.h_k, dv), corr_range_hk_vs_los_drdk=corr(rb.h_k, dr),
        eclipse_transitions=int(trans.size), illuminated_fraction=float(lit.mean()),
        near_eclipse_row_fraction=float(near_ecl.mean()), near_eclipse_energy_share=share(near_ecl),
        near_pass_edge_row_fraction=float(near_edge.mean()), near_pass_edge_energy_share=share(near_edge),
        doppler_orth_energy_fraction_of_total=float(e_d.sum() / (bp ** 2).sum()) if (bp ** 2).sum() > 0 else None,
        doppler_rows_t=r60.t_mid.tolist(), doppler_rows_hk=r60.h_k.tolist(), doppler_rows_orth=bp[rb.n_obs:].tolist(),
        doppler_rows_station=r60.station.tolist(), nu=hist["nu"].tolist(), t_grid=rb.t_grid.tolist())


# ======================================================================
# FD spot checks
# ======================================================================
def fd_nominal(spec: dict) -> dict:
    orbit, k = spec["orbit"], spec["k"]
    scn = scenario(orbit, spec["x0"], k, n_orbits=2.0)
    from phase17_r1od2_fidelity import propagate_fidelity
    nom = propagate_fidelity(scn)
    same = bool(np.array_equal(nom, propagate_hf(scn)))
    rb = D2.build_range_block(scn, nom)
    st = stations()
    r = CD.build_doppler_rows(scn, rb, 60.0, st, with_values=False)
    idx = np.unique(np.linspace(0, r.n - 1, 6).round().astype(int))
    an = np.hstack([r.h_x0[idx], r.h_k[idx, None]])
    pg = rb.pass_geo
    return dict(orbit=orbit, k=k, x0=list(spec["x0"]), bitwise_same_config=same, analytic=an.tolist(),
                rows=[(float(r.t_mid[i]), int(r.station[i])) for i in idx],
                pg=dict(t_grid=rb.t_grid, earth_pos=pg.earth_pos_mci_m, earth_vel=pg.earth_vel_mci_mps,
                        xf=pg.x_j2000_to_itrf93, et0=pg.et0_s))


def fd_eval(spec: dict) -> dict:
    """Production observable on a repropagated HF trajectory (oracle tolerance)."""
    from lunar_od.radiometrics import two_way_counted_doppler_observable
    nomd = spec["nominal"]
    x0 = np.asarray(nomd["x0"], float)
    j, h = spec["j"], spec["h"]
    k_srp = D2.M.K_TRUTH
    if j < 6:
        x0 = x0.copy()
        x0[j] += h
    else:
        k_srp = k_srp + h
    scn = scenario(nomd["orbit"], x0.tolist(), nomd["k"], n_orbits=2.0, k_srp=k_srp)
    x6 = propagate_hf(scn, rtol=1e-13, atol=1e-14)[:, :6]
    pg = nomd["pg"]
    st = stations()
    cfg = CD.doppler_config(60.0)
    y = [two_way_counted_doppler_observable(t, st[s], pg["t_grid"], x6, pg["earth_pos"], pg["earth_vel"], pg["xf"],
                                            cfg, et0_s=pg["et0"]) for t, s in nomd["rows"]]
    return dict(case=spec["case"], j=j, h=h, y=y)


def run_fd(ex, x0s) -> dict:
    noms = list(ex.map(fd_nominal, [dict(orbit=o, k=k, x0=x0s[o]) for o, k in FD_CASES]))
    jobs = []
    for ci, nomd in enumerate(noms):
        for j in range(7):
            for h in FD_STEPS[j]:
                for sgn in (1.0, -1.0):
                    jobs.append(dict(case=ci, nominal=nomd, j=j, h=sgn * h))
    res = list(ex.map(fd_eval, jobs))
    out = {}
    for ci, nomd in enumerate(noms):
        an = np.asarray(nomd["analytic"])
        cols = {}
        for j in range(7):
            errs = []
            for h in FD_STEPS[j]:
                yp = next(r["y"] for r in res if r["case"] == ci and r["j"] == j and r["h"] == h)
                ym = next(r["y"] for r in res if r["case"] == ci and r["j"] == j and r["h"] == -h)
                fd = (np.asarray(yp) - np.asarray(ym)) / (2 * h)
                errs.append(float(np.linalg.norm(fd - an[:, j]) / np.linalg.norm(an[:, j])))
            cols[FD_NAMES[j]] = dict(steps=list(FD_STEPS[j]), rel_err=errs, best=min(errs))
        out["%s_k%02d" % (nomd["orbit"], nomd["k"])] = dict(
            rows=nomd["rows"], oracle="HF repropagation rtol 1e-13 atol 1e-14, central differences",
            analytic="production Jacobian, HF rtol 1e-12 (propagate_fidelity)",
            fd_wrapper_bitwise_equals_propagate_fidelity=nomd["bitwise_same_config"], columns=cols)
    state_ok = all(c["columns"][n]["best"] <= 1e-5 for c in out.values() for n in FD_NAMES[:6])
    k_ok = all(c["columns"]["K_SRP"]["best"] <= 1e-3 for c in out.values())
    cfg_ok = all(c["fd_wrapper_bitwise_equals_propagate_fidelity"] for c in out.values())
    return dict(cases=out, COUNTED_DOPPLER_STATE_FD_GATE="PASS" if state_ok and cfg_ok else "FAIL",
                COUNTED_DOPPLER_K_FD_GATE="PASS" if k_ok and cfg_ok else "FAIL",
                gates=dict(state="every state column best rel err <= 1e-5", k="K best rel err <= 1e-3"))


# ======================================================================
# main
# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    args = ap.parse_args()
    pre = CD.load_predeclaration(PREDECLARATION_SHA256)
    import hashlib
    g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
    x0s = {}
    for x in g["geometries"]:
        if x["case_id"] in ORBITS:
            assert hashlib.sha256(json.dumps(x["x0"]).encode()).hexdigest() == \
                pre["orbits"]["set"][x["case_id"]]["x0_sha256"], x["case_id"]
            x0s[x["case_id"]] = x["x0"]
    assert sorted(x0s) == sorted(ORBITS)
    raw_path = ARTIFACTS / "phase17_r1o_cd1_raw_results.json"
    fd_path = ARTIFACTS / "phase17_r1o_cd1_fd_oracle.json"
    if args.analyse_only:
        results = json.loads(raw_path.read_text(encoding="utf-8"))
        fd = json.loads(fd_path.read_text(encoding="utf-8"))
        analyse(pre, results, fd, None)
        return
    import phase17_r1m_core as M
    assert np.array_equal(np.asarray(x0s["CANON"]), M.campaign_initial_state())
    t_start = time.time()
    specs = [dict(orbit=o, x0=x0s[o], k=k, dynamics=dyn) for dyn in ("HIGH_FIDELITY", "SIMPLIFIED")
             for o in ORBITS for k in EPOCHS]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        results = list(ex.map(run_cell, specs))
        print("cells done in %.0f s" % (time.time() - t_start), flush=True)
        raw_path.write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
        fd = run_fd(ex, x0s)
    fd_path.write_text(json.dumps(_clean(fd), indent=1), encoding="utf-8")
    print("FD done in %.0f s: %s %s" % (time.time() - t_start, fd["COUNTED_DOPPLER_STATE_FD_GATE"],
                                         fd["COUNTED_DOPPLER_K_FD_GATE"]), flush=True)
    analyse(pre, results, fd, time.time() - t_start)


def _f(v):
    return float(v) if v is not None else float("nan")


def population(values) -> dict:
    """Predeclared population rule POP: frac(delta > 0) >= 0.75 AND frac(delta > 0.02) >= 0.50."""
    v = np.asarray(values, float)
    return dict(n=int(v.size), frac_pos=float(np.mean(v > 0)), frac_modest_or_strong=float(np.mean(v > 0.02)),
                frac_strong=float(np.mean(v > 0.05)), frac_loss=float(np.mean(v < -0.02)),
                median=float(np.median(v)), max=float(np.max(v)), min=float(np.min(v)),
                POP=bool(np.mean(v > 0) >= 0.75 and np.mean(v > 0.02) >= 0.50))


def decision(p0: dict, p1: dict, abs_ok: bool) -> str:
    """Predeclared decision tree, first match in the order A, B, C, D, E."""
    if p0["POP"] and p1["POP"] and abs_ok:
        return "COUNTED_DOPPLER_ROBUSTLY_ADDS_K_SRP_DIRECTION"
    if p0["POP"] and p1["POP"]:
        return "COUNTED_DOPPLER_ROBUSTLY_IMPROVES_DIRECTION_BUT_K_REMAINS_WEAK"
    if not p0["POP"] and p0["frac_modest_or_strong"] >= 0.10:
        return "COUNTED_DOPPLER_GAIN_IS_GEOMETRY_DEPENDENT"
    if p0["POP"] and not p1["POP"]:
        return "COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE"
    return "COUNTED_DOPPLER_ADDS_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY"


def nuisance_retention_class(ret: list) -> str:
    if not ret:
        return "NOT_APPLICABLE_NO_IDEAL_GAIN"
    mr = float(np.median(ret))
    return "RETAINED" if mr >= 0.75 else "PARTIALLY_RETAINED" if mr >= 0.25 else "LOST"


def fidelity_class(pop_simplified: bool, pop_hf: bool, rms_d_delta: float) -> str:
    if pop_simplified != pop_hf:
        return "FIDELITY_CHANGES_POPULATION_VERDICT"
    if rms_d_delta > 0.02:
        return "FIDELITY_MATERIAL_VERDICT_STABLE"
    return "FIDELITY_NEGLIGIBLE"


def analyse(pre, results, fd, elapsed) -> None:
    by = {(r["orbit"], r["k"], r["dynamics"]): r for r in results}
    hf = [by[(o, k, "HIGH_FIDELITY")] for o in ORBITS for k in EPOCHS]
    feas = [r for r in hf if r["feasible"]]

    def d(r, name):
        return _f(r["cfg"][name]["delta_f_perp"]) if r["feasible"] and name in r["cfg"] else float("nan")

    def f(r, name):
        return _f(r["cfg"][name]["f_perp"]) if r["feasible"] and name in r["cfg"] else float("nan")

    def pop(name, rs=None):
        rs = feas if rs is None else [r for r in rs if r["feasible"]]
        return population([d(r, name) for r in rs])

    p0, p1 = pop("RPD|D0"), pop("RPD|D1|FREE")
    abs_frac = float(np.mean([f(r, "RPD|D1|FREE") >= 0.35 for r in feas]))
    ABS = abs_frac >= 0.50
    primary = decision(p0, p1, ABS)

    # nuisance retention
    ret = [d(r, "RPD|D1|FREE") / d(r, "RPD|D0") for r in feas if d(r, "RPD|D0") > 0.02]
    nuis_class = nuisance_retention_class(ret)

    # fidelity decomposition (D0)
    simp = [by[(o, k, "SIMPLIFIED")] for o in ORBITS for k in EPOCHS]
    ps0 = pop("RPD|D0", simp)
    dec = []
    for o in ORBITS:
        for k in EPOCHS:
            a, b = by[(o, k, "HIGH_FIDELITY")], by[(o, k, "SIMPLIFIED")]
            if not (a["feasible"] and b["feasible"]):
                continue
            dec.append(dict(orbit=o, k=k, range_hf=_f(a["range"]["f_perp"]), range_simp=_f(b["range"]["f_perp"]),
                            dop_hf=f(a, "DOP|D0"), dop_simp=f(b, "DOP|D0"), comb_hf=f(a, "RPD|D0"),
                            comb_simp=f(b, "RPD|D0"), delta_hf=d(a, "RPD|D0"), delta_simp=d(b, "RPD|D0"),
                            delta_d1_hf=d(a, "RPD|D1|FREE"), delta_d1_simp=d(b, "RPD|D1|FREE")))
    for x in dec:
        x.update(d_range=x["range_hf"] - x["range_simp"], d_dop=x["dop_hf"] - x["dop_simp"],
                 d_comb=x["comb_hf"] - x["comb_simp"], d_delta=x["delta_hf"] - x["delta_simp"])
    rms = lambda key: float(np.sqrt(np.mean([x[key] ** 2 for x in dec])))  # noqa: E731
    fid_class = fidelity_class(ps0["POP"], p0["POP"], rms("d_delta"))
    corr_dd = float(np.corrcoef([x["d_delta"] for x in dec], [-x["d_range"] for x in dec])[0, 1])

    # verification
    contracts = all(all(r["contract"].values()) for r in results)
    rb_f = [v for r in hf if r["feasible"] for v in r["rebuild_f_perp"].values()]
    rows_s = [r["rebuild_rows"]["doppler_state_rel_max"] for r in feas]
    rows_k = [r["rebuild_rows"]["doppler_k_abs_over_colmax"] for r in feas]
    rows_r = [r["rebuild_rows"]["range_k_abs_over_colmax"] for r in feas]
    rebuild_gate = (max(rb_f) <= 1e-10 and max(rows_s) <= 1e-9 and max(rows_k) <= 1e-9 and max(rows_r) <= 1e-7
                    and contracts)
    d3 = {int(r["k"]): float(r["range_f_perp"]) for r in csv.DictReader(
        (ARTIFACTS / "phase17_r1od3_primary_results.csv").open()) if r["strategy"] == "GC_ONLY"}
    anch3 = [abs(_f(by[("CANON", k, "SIMPLIFIED")]["range"]["f_perp"]) - d3[k]) for k in EPOCHS]
    d4 = json.loads((ARTIFACTS / "phase17_r1od4_raw_results.json").read_text(encoding="utf-8"))
    anch4 = []
    for x in d4:
        if x["phase"] % 15 == 0 and 4 <= x["phase"] / 15 <= 11:
            r = by[(x["orbit"], int(x["phase"] / 15), x["dynamics"])]
            anch4.append(abs(_f(r["range"]["f_perp"]) - x["diag"]["range_f_perp"]))
    anchors_ok = max(anch3) <= 1e-12 and max(anch4) <= 1e-12
    (ARTIFACTS / "phase17_r1o_cd1_independent_rebuild.json").write_text(json.dumps(_clean(dict(
        f_perp=dict(n=len(rb_f), max_abs_diff=max(rb_f), gate=1e-10,
                    configs=["DOP|D0", "RPD|D0", "RPD|D1|FREE", "RPD|D2|FREE", "RPD|D1|0.0001", "RANGE"]),
        doppler_rows_r4_four_event=dict(n_cells=len(rows_s), state_rel_max=max(rows_s), k_abs_over_colmax=max(rows_k),
                                        gate=1e-9),
        range_k_rows_event_epoch_route=dict(n_cells=len(rows_r), max=max(rows_r), gate=1e-7),
        contracts_all=contracts,
        range_baseline_anchors=dict(d3_canon_simplified_24=dict(max_abs_diff=max(anch3)),
                                    d4_all_orbits_k4_11=dict(n=len(anch4), max_abs_diff=max(anch4)),
                                    gate=1e-12, reproduced=anchors_ok),
        INDEPENDENT_REBUILD_GATE="PASS" if rebuild_gate else "FAIL")), indent=1), encoding="utf-8")

    # ---------------- CSVs
    cls = D2.relative_gain_class
    with (ARTIFACTS / "phase17_r1o_cd1_primary_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "k", "synodic_phase_deg", "dynamics", "feasible", "n_range", "n_doppler", "n_sessions",
                    "sep_deg_median", "range_f_perp", "range_sigma_k_frac", "doppler_only_f_perp_D0",
                    "doppler_only_f_perp_D1FREE", "combined_f_perp_D0", "delta_D0", "relative_class_D0",
                    "absolute_class_D0", "combined_f_perp_D1FREE", "delta_D1FREE", "relative_class_D1FREE",
                    "absolute_class_D1FREE", "theta_k_deg_D0", "iperp_D0", "i_kk_D0", "k_column_norm_D0",
                    "sigma_k_D0", "sigma_k_frac_D0", "sigma_k_frac_D1FREE", "design_condition_D0",
                    "weakest_mode_k_component_D0", "range_weakest_mode_k_component", "k_bias_corr_D1FREE"])
        for r in results:
            row = [r["orbit"], r["k"], 15 * r["k"], r["dynamics"], r["feasible"], r["range_n"], r["doppler_n"]["60.0"],
                   r["n_sessions"], r["geometry"]["sep_deg_median"], r["range"]["f_perp"], r["range"]["sigma_k_frac"]]
            if r["feasible"]:
                c0, c1 = r["cfg"]["RPD|D0"], r["cfg"]["RPD|D1|FREE"]
                row += [r["cfg"]["DOP|D0"]["f_perp"], r["cfg"]["DOP|D1|FREE"]["f_perp"], c0["f_perp"],
                        c0["delta_f_perp"], cls(c0["delta_f_perp"]), D2.absolute_class(c0["f_perp"]), c1["f_perp"],
                        c1["delta_f_perp"], cls(c1["delta_f_perp"]), D2.absolute_class(c1["f_perp"]),
                        c0["theta_k_deg"], c0["conditional_k_information"], c0["i_kk"], c0["k_column_norm"],
                        c0["sigma_k"], c0["sigma_k_frac"], c1["sigma_k_frac"], c0["design_condition"],
                        c0["weakest_mode_k_component"], r["range"]["weakest_mode_k_component"],
                        c1.get("max_abs_k_bias_corr")]
            w.writerow(row)
    with (ARTIFACTS / "phase17_r1o_cd1_noise_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "k", "config", "family", "tc_s", "sigma60_or_scaling", "nuisance", "n_doppler", "f_perp",
                    "delta_f_perp", "relative_class", "sigma_k_frac"])
        for r in feas:
            for name, c in r["cfg"].items():
                p = name.split("|")
                if p[0] == "NOISE":
                    fam, tc, sc, nu = "BRACKET", 60, p[1], p[2]
                elif p[0] == "OPLB":
                    fam, tc, sc, nu = "OPERATIONAL_LOWER_BOUND", 60, "202E_scint+FTS", p[1]
                elif p[0] == "TC":
                    fam, tc, sc, nu = "COUNT_INTERVAL", p[1], p[2], p[3]
                elif p[0] == "REG":
                    fam, tc, sc, nu = "REGULAR_SESSION_" + p[1], 60, "0.0001", p[2]
                else:
                    continue
                w.writerow([r["orbit"], r["k"], name, fam, tc, sc, nu, c["n_obs"] - (r["range_n"] if "DOP" not in name
                            else 0), c["f_perp"], c["delta_f_perp"], cls(c["delta_f_perp"]), c["sigma_k_frac"]])
    with (ARTIFACTS / "phase17_r1o_cd1_nuisance_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orbit", "k", "model", "prior_mps", "n_nuisance", "f_perp", "delta_f_perp", "relative_class",
                    "f_perp_loss_vs_D0", "sigma_k_inflation_vs_D0", "max_abs_k_bias_corr", "max_abs_k_drift_corr",
                    "bias_sigma_mps_median"])
        for r in feas:
            c0 = r["cfg"]["RPD|D0"]
            w.writerow([r["orbit"], r["k"], "D0_IDEAL", "", 0, c0["f_perp"], c0["delta_f_perp"],
                        cls(c0["delta_f_perp"]), 0.0, 1.0, "", "", ""])
            for mdl in ("D1", "D2"):
                for pn in PRIORS:
                    c = r["cfg"]["RPD|%s|%s" % (mdl, pn)]
                    w.writerow([r["orbit"], r["k"], {"D1": "D1_SESSION_CONSTANT_BIAS",
                                                     "D2": "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT"}[mdl], pn,
                                c["n_nuisance"], c["f_perp"], c["delta_f_perp"], cls(c["delta_f_perp"]),
                                c0["f_perp"] - c["f_perp"], _f(c["sigma_k"]) / _f(c0["sigma_k"]),
                                c.get("max_abs_k_bias_corr"), c.get("max_abs_k_drift_corr"),
                                c.get("bias_sigma_mps_median")])
    with (ARTIFACTS / "phase17_r1o_cd1_fidelity_decomposition.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dec[0].keys()))
        w.writeheader()
        w.writerows(dec)

    # ---------------- sensitivity summaries
    def pop_cfg(name):
        return pop(name) if all(name in r["cfg"] for r in feas) else None
    noise = {name: pop_cfg(name) for name in sorted({n for r in feas for n in r["cfg"]
                                                      if n.startswith(("NOISE", "OPLB", "TC", "REG"))})}
    nuis = {("%s|%s" % (m, p)): pop("RPD|%s|%s" % (m, p)) for m in ("D1", "D2") for p in PRIORS}
    by_orbit = {o: pop("RPD|D0", [by[(o, k, "HIGH_FIDELITY")] for k in EPOCHS]) for o in ORBITS}
    by_orbit_d1 = {o: pop("RPD|D1|FREE", [by[(o, k, "HIGH_FIDELITY")] for k in EPOCHS]) for o in ORBITS}
    by_epoch = {k: [d(by[(o, k, "HIGH_FIDELITY")], "RPD|D0") for o in ORBITS] for k in EPOCHS}
    abs0 = [D2.absolute_class(f(r, "RPD|D0")) for r in feas]
    abs1 = [D2.absolute_class(f(r, "RPD|D1|FREE")) for r in feas]
    sk_ratio = [_f(r["cfg"]["RPD|D0"]["sigma_k"]) / _f(r["range"]["sigma_k"]) for r in feas]
    interp = {"%s_k%02d" % (o, k): {kk: vv for kk, vv in by[(o, k, "HIGH_FIDELITY")]["interpretation"].items()
                                    if not kk.startswith(("doppler_rows", "nu", "t_grid"))} for o, k in REP_CELLS}
    (ARTIFACTS / "phase17_r1o_cd1_time_localization.json").write_text(json.dumps(_clean(dict(
        cells=interp, rule="LOCALIZED if one 10% slice adds > 50% of the final combined I(K|x), else BROAD")),
        indent=1), encoding="utf-8")
    med = lambda xs: float(np.median([x for x in xs if np.isfinite(x)]))  # noqa: E731
    summary = dict(
        phase="17-R1O-CD1", predeclaration_sha256=PREDECLARATION_SHA256, elapsed_s=elapsed,
        primary_cells=len(hf), feasible_cells=len(feas),
        population_D0=p0, population_D1FREE=p1, ABS_fraction_D1FREE_ge_0p35=abs_frac, ABS=ABS,
        PRIMARY_CLASS=primary, ROBUST_COMPLEMENTARITY_FOUND=primary.startswith("COUNTED_DOPPLER_ROBUSTLY"),
        medians=dict(range_f=med([_f(r["range"]["f_perp"]) for r in feas]),
                     doppler_only_f_D0=med([f(r, "DOP|D0") for r in feas]),
                     doppler_only_f_D1FREE=med([f(r, "DOP|D1|FREE") for r in feas]),
                     combined_f_D0=med([f(r, "RPD|D0") for r in feas]),
                     combined_f_D1FREE=med([f(r, "RPD|D1|FREE") for r in feas]),
                     sigma_k_ratio_combined_over_range_D0=med(sk_ratio)),
        absolute_counts_D0={c: abs0.count(c) for c in ("WEAK_K_SEPARATION", "MODERATE_K_SEPARATION",
                                                      "STRONG_K_SEPARATION")},
        absolute_counts_D1FREE={c: abs1.count(c) for c in ("WEAK_K_SEPARATION", "MODERATE_K_SEPARATION",
                                                          "STRONG_K_SEPARATION")},
        NUISANCE_RETENTION_CLASS=nuis_class,
        nuisance_retention_median=float(np.median(ret)) if ret else None, nuisance_retention_n=len(ret),
        nuisance_populations=nuis,
        FIDELITY_EFFECT_CLASS=fid_class, simplified_population_D0=ps0,
        fidelity_decomposition=dict(rms_d_range=rms("d_range"), rms_d_doppler_only=rms("d_dop"),
                                    rms_d_combined=rms("d_comb"), rms_d_delta=rms("d_delta"),
                                    corr_d_delta_vs_minus_d_range=corr_dd),
        noise_and_schedule_populations=noise, by_orbit_D0=by_orbit, by_orbit_D1FREE=by_orbit_d1,
        by_epoch_D0_median={str(k): float(np.nanmedian(v)) for k, v in by_epoch.items()},
        time_localization={c: v["TIME_LOCALIZATION"] for c, v in interp.items()},
        signature={c: dict(corr_doppler_hk_vs_los_dvdk=v["corr_doppler_hk_vs_los_dvdk"],
                           corr_range_hk_vs_los_drdk=v["corr_range_hk_vs_los_drdk"],
                           near_eclipse=[v["near_eclipse_row_fraction"], v["near_eclipse_energy_share"]],
                           near_pass_edge=[v["near_pass_edge_row_fraction"], v["near_pass_edge_energy_share"]])
                   for c, v in interp.items()},
        COUNTED_DOPPLER_STATE_FD_GATE=fd["COUNTED_DOPPLER_STATE_FD_GATE"],
        COUNTED_DOPPLER_K_FD_GATE=fd["COUNTED_DOPPLER_K_FD_GATE"],
        INDEPENDENT_REBUILD_GATE="PASS" if rebuild_gate else "FAIL", contracts_all=contracts,
        range_anchors_reproduced=anchors_ok,
        NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED="MAY_BE_JUSTIFIED_SEPARATELY_AUTHORISED"
        if primary.startswith("COUNTED_DOPPLER_ROBUSTLY") else "NO",
        ALTERNATIVE_K_PARAMETERIZATION_JUSTIFIED="NO" if primary.startswith("COUNTED_DOPPLER_ROBUSTLY") else "YES",
        doppler_rows=dict(median_primary=float(np.median([r["doppler_n"]["60.0"] for r in feas])),
                          sessions_median=float(np.median([r["n_sessions"] for r in feas]))))
    (ARTIFACTS / "phase17_r1o_cd1_summary.json").write_text(json.dumps(_clean(summary), indent=1), encoding="utf-8")
    figures(by, dec, feas, interp_raw={c: by[(o, k, "HIGH_FIDELITY")]["interpretation"] for c, (o, k)
                                       in zip(interp, REP_CELLS)})
    print(json.dumps(_clean({k: summary[k] for k in (
        "feasible_cells", "population_D0", "population_D1FREE", "ABS_fraction_D1FREE_ge_0p35", "PRIMARY_CLASS",
        "medians", "absolute_counts_D0", "absolute_counts_D1FREE", "NUISANCE_RETENTION_CLASS",
        "FIDELITY_EFFECT_CLASS", "fidelity_decomposition", "COUNTED_DOPPLER_STATE_FD_GATE",
        "COUNTED_DOPPLER_K_FD_GATE", "INDEPENDENT_REBUILD_GATE", "range_anchors_reproduced")}), indent=1))


def figures(by, dec, feas, interp_raw) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = [15 * k for k in EPOCHS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.86, bottom=0.16)
    for ax, name, title in ((axes[0], "RPD|D0", "D0 ideal"), (axes[1], "RPD|D1|FREE", "D1 per-pass bias, free")):
        m = np.array([[by[(o, k, "HIGH_FIDELITY")]["cfg"][name]["delta_f_perp"]
                       if by[(o, k, "HIGH_FIDELITY")]["feasible"] else np.nan for k in EPOCHS] for o in ORBITS])
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                       extent=[-7.5, 352.5, len(ORBITS) - 0.5, -0.5])
        ax.set_yticks(range(len(ORBITS)))
        ax.set_yticklabels(ORBITS, fontsize=7)
        ax.set_xlabel("synodic phase [deg]")
        ax.set_title("HF range+Doppler delta f_perp, %s" % title, fontsize=9)
    fig.colorbar(im, ax=axes, label="delta f_perp")
    fig.savefig(ARTIFACTS / "phase17_r1o_cd1_delta_heatmap.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rf = [r["range"]["f_perp"] for r in feas]
    axes[0].scatter(rf, [r["cfg"]["DOP|D0"]["f_perp"] for r in feas], s=10, label="Doppler only")
    axes[0].scatter(rf, [r["cfg"]["RPD|D0"]["f_perp"] for r in feas], s=10, label="range + Doppler")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    for y0 in (0.35, 0.5):
        axes[0].axhline(y0, color="grey", lw=0.6, ls=":")
    axes[0].set_xlabel("range-only f_perp")
    axes[0].set_ylabel("f_perp")
    axes[0].set_title("HF, D0, all 120 cells", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].scatter([x["d_range"] for x in dec], [x["d_delta"] for x in dec], s=10, label="d(delta) vs d(range f)")
    axes[1].scatter([x["d_range"] for x in dec], [x["d_comb"] for x in dec], s=10, label="d(combined f) vs d(range f)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_xlabel("HF - SIMPLIFIED range f_perp")
    axes[1].set_title("fidelity decomposition (D0)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].scatter([r["cfg"]["RPD|D0"]["f_perp"] for r in feas], [r["cfg"]["RPD|D0"]["sigma_k_frac"] for r in feas],
                    s=10, label="range + Doppler")
    axes[2].scatter(rf, [r["range"]["sigma_k_frac"] for r in feas], s=10, label="range only")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("f_perp")
    axes[2].set_ylabel("sigma_K / K")
    axes[2].set_title("magnitude vs direction (HF, D0)", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1o_cd1_information_geometry.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = ["RPD|D0"] + ["RPD|D1|%s" % p for p in PRIORS] + ["RPD|D2|%s" % p for p in PRIORS]
    data = [[r["cfg"][n]["delta_f_perp"] for r in feas] for n in names]
    ax.boxplot(data, tick_labels=[n.replace("RPD|", "") for n in names], showfliers=False)
    ax.axhline(0.02, color="grey", ls=":", lw=0.8)
    ax.axhline(0.05, color="k", ls="--", lw=0.8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("delta f_perp")
    ax.set_title("nuisance impact, HF range + Doppler (120 cells)", fontsize=9)
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=20)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1o_cd1_nuisance_impact.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for c, v in interp_raw.items():
        cum = v["cumulative"]
        axes[0].plot([x["tau_frac"] for x in cum], [x.get("comb_iperp", np.nan) for x in cum], "o-", ms=3, label=c)
        axes[1].plot([x["tau_frac"] for x in cum], [x.get("comb_f", np.nan) for x in cum], "o-", ms=3, label=c)
        axes[1].plot([x["tau_frac"] for x in cum], [x.get("range_f", np.nan) for x in cum], ":", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("cumulative I(K|x), range+Doppler")
    axes[1].set_ylabel("cumulative f_perp (dotted: range only)")
    axes[1].set_xlabel("fraction of arc")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "phase17_r1o_cd1_time_localization.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
