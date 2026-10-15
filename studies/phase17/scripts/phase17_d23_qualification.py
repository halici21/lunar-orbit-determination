"""PHASE 17-HARDEN-D23 - counted-Doppler Earth ephemeris: value-level qualification.

Compares the PRODUCTION counted-Doppler observable VALUES with an independent
direct-SPICE oracle over the full CD1/KP1 population (5 orbits x 24 epochs x
{HIGH_FIDELITY, SIMPLIFIED}, every ALL_VISIBLE Tc = 60 s row), and re-qualifies
the CD1 and KP1 information results on the repaired production path.

Oracle (independent of the production light-time code and Earth handling):
    the frozen R2 model S single-bounce solver (and, on every 10th row, the M3
    four-event model F solver, both in lunar_od.two_way_counted_doppler_reference)
    fed by a station-state provider evaluated DIRECTLY from SPICE at every event
    epoch the solver requests:
        station(t) = spkezr(EARTH, et0 + t, J2000, NONE, MOON) + sxform(J2000, ITRF93)^-1 [r_ecef; 0]
Historical mutant:
    the production solver with the pre-repair LINEAR Earth provider, rebuilt
    explicitly here (the code R3-P22 froze), so the D23 offset is reproduced.
Solver floor:
    production solver vs model S solver with the SAME linear provider: two
    independent implementations on identical inputs.

Acceptance criteria are frozen in artifacts/phase17_d23_acceptance.json
(hash-pinned) before the repaired run.

    python examples/phase17_d23_qualification.py
    python examples/phase17_d23_qualification.py --analyse-only   # needs the uncommitted 18.8 MB per-row raw;
                                                                  # artifacts/phase17_d23_cells.json is the compact record
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_kp1_core as KP  # noqa: E402
import phase17_r1o_cd1_campaign as K  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402

D2 = K.D2
ARTIFACTS = CD.ARTIFACTS
ACCEPTANCE_PATH = ARTIFACTS / "phase17_d23_acceptance.json"
ACCEPTANCE_SHA256 = "2d1cb37028c8486cd0b6acaf0f5c30c2e2fe01c248e36a133a0e1f055f04b7af"
RAW_PATH = ARTIFACTS / "phase17_d23_raw_results.json"
F_STRIDE = 10
DIAG_ROWS = 25


def load_acceptance() -> dict:
    d = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    got = CD.canonical_sha256(d)
    if not (got == d.get("canonical_sha256") == ACCEPTANCE_SHA256):
        raise CD.PredeclarationMismatch("D23 acceptance hash mismatch: %s / %s / %s"
                                        % (got, d.get("canonical_sha256"), ACCEPTANCE_SHA256))
    return d


# ======================================================================
# providers
# ======================================================================
def direct_spice_provider(station, et0: float):
    """Reference-module provider whose every state is evaluated directly from SPICE."""
    import spiceypy as spice
    from lunar_od.two_way_counted_doppler_reference import ExactEventStationStateProvider
    ecef = np.concatenate([np.asarray(station.r_ecef_m, float).reshape(3), np.zeros(3)])
    counter = [0]

    def state_fn(t: float) -> np.ndarray:
        et = float(et0) + float(t)
        earth = np.asarray(spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")[0], float) * 1e3
        xform = np.asarray(spice.sxform("J2000", "ITRF93", et), float)
        counter[0] += 1
        return earth + np.linalg.solve(xform, ecef)

    return ExactEventStationStateProvider(state_fn=state_fn, station_state_method="direct_spice_oracle",
                                          earth_ephemeris_method="direct_spice_spkezr",
                                          _sxform_counter=counter)


def et_exact_spice_provider(station, et0: float, correct_site: bool = True):
    """POST-HOC DIAGNOSTIC oracle (not the frozen acceptance oracle).

    fl(et0 + t) is quantised to ulp(8.6e8 s) = 1.19e-7 s, i.e. ~5e-5 m of Earth
    motion: a floor of the frozen oracle, not of production (production's
    Hermite works in exact scenario seconds and its grid nodes et0 + 90 k are
    exact).  Here the rounding remainder d = (et0 + t) - fl(et0 + t) is
    recovered by an error-free two-sum and applied to first order with the SPICE
    velocity (second-order term ~ 3e-18 m).  ``correct_site`` applies the same
    correction to the station site, whose sxform production also evaluates at
    fl(et0 + t)."""
    import spiceypy as spice
    from lunar_od.two_way_counted_doppler_reference import ExactEventStationStateProvider
    ecef = np.concatenate([np.asarray(station.r_ecef_m, float).reshape(3), np.zeros(3)])
    e0 = float(et0)

    def state_fn(t: float) -> np.ndarray:
        t = float(t)
        s = e0 + t
        bb = s - e0
        d = (e0 - (s - bb)) + (t - bb)
        earth = np.asarray(spice.spkezr("EARTH", s, "J2000", "NONE", "MOON")[0], float) * 1e3
        earth[:3] += earth[3:] * d
        site = np.linalg.solve(np.asarray(spice.sxform("J2000", "ITRF93", s), float), ecef)
        if correct_site:
            site[:3] += site[3:] * d
        return earth + site

    return ExactEventStationStateProvider(state_fn=state_fn, station_state_method="et_exact_spice_diagnostic",
                                          earth_ephemeris_method="direct_spice_spkezr_two_sum")


def et_exact_diagnostic(spec: dict) -> dict:
    """Production vs the ET-exact oracle (Earth only, and Earth + site) on one cell."""
    from lunar_od.two_way_counted_doppler_reference import (CountedDopplerReferenceConfig,
                                                            exact_station_single_bounce_counted_doppler_reference)
    scn = K.scenario(spec["orbit"], spec["x0"], spec["k"])
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    pg = rb.pass_geo
    st = K.stations()
    rows = CD.build_doppler_rows(scn, rb, 60.0, st, with_values=True)
    rcfg = CountedDopplerReferenceConfig.from_legacy(CD.doppler_config(60.0))   # R2 convention: production tolerances
    x6 = rb.nom48[:, :6]
    out = {}
    for label, cs in (("earth_only", False), ("earth_and_site", True)):
        prov = {s: et_exact_spice_provider(st[s], pg.et0_s, cs) for s in range(len(st))}
        d = np.array([rows.y[i] - exact_station_single_bounce_counted_doppler_reference(
            float(rows.t_mid[i]), prov[int(rows.station[i])], rb.t_grid, x6, rcfg).observable
            for i in range(rows.n)])
        out[label] = dict(max=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d ** 2))))
    return K._clean(dict(orbit=spec["orbit"], k=spec["k"], n_rows=int(rows.n), **out))


def historical_linear_provider(station, et0: float, t_grid, earth_pos, earth_vel):
    """The pre-repair production provider (linear Earth, exact sxform), rebuilt verbatim."""
    import spiceypy as spice
    from lunar_od.radiometrics import CountedDopplerStationStateProvider, _interp_vector
    ecef = np.concatenate([np.asarray(station.r_ecef_m, float).reshape(3), np.zeros(3)])

    def state_fn(t: float) -> np.ndarray:
        earth = np.concatenate([_interp_vector(t_grid, earth_pos, t), _interp_vector(t_grid, earth_vel, t)])
        return earth + np.linalg.solve(np.asarray(spice.sxform("J2000", "ITRF93", float(et0) + float(t)), float),
                                       ecef)

    return CountedDopplerStationStateProvider(state_fn=state_fn, station_name=str(station.name),
                                              earth_ephemeris_method="historical_linear_grid_interpolation")


# ======================================================================
# one cell
# ======================================================================
def run_cell(spec: dict) -> dict:
    from lunar_od.radiometrics import (RangeRatePhysicsConfig, _interp_state, _interp_vector,
                                       resolve_counted_doppler_station_state_provider,
                                       two_way_counted_doppler_observable)
    from lunar_od.two_way_counted_doppler_reference import (
        CountedDopplerReferenceConfig, exact_station_four_event_counted_doppler_reference,
        exact_station_single_bounce_counted_doppler_reference, make_exact_event_station_state_provider)
    from phase17_r1m_core import _spice_earth_state_mci

    t0 = time.time()
    orbit, k, dyn = spec["orbit"], spec["k"], spec["dynamics"]
    scn = K.scenario(orbit, spec["x0"], k)
    if dyn == "HIGH_FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    pg = rb.pass_geo
    st = K.stations()
    rows = CD.build_doppler_rows(scn, rb, 60.0, st, with_values=True)       # repaired PRODUCTION values
    x6 = rb.nom48[:, :6]
    cfg = CD.doppler_config(60.0)
    rcfg = CountedDopplerReferenceConfig.from_legacy(CD.doppler_config(60.0))   # R2 convention: production tolerances
    prod_meta = {}
    oracle = {s: direct_spice_provider(st[s], pg.et0_s) for s in range(len(st))}
    oracle_et = {s: et_exact_spice_provider(st[s], pg.et0_s, False) for s in range(len(st))}   # SECONDARY, post-hoc
    linear = {s: historical_linear_provider(st[s], pg.et0_s, rb.t_grid, pg.earth_pos_mci_m, pg.earth_vel_mci_mps)
              for s in range(len(st))}
    ref_linear = {s: make_exact_event_station_state_provider(st[s], pg.et0_s, rb.t_grid, pg.earth_pos_mci_m,
                                                             pg.earth_vel_mci_mps) for s in range(len(st))}
    prod_prov = {s: resolve_counted_doppler_station_state_provider(cfg, st[s], rb.t_grid, pg.earth_pos_mci_m,
                                                                   pg.earth_vel_mci_mps, et0_s=pg.et0_s)
                 for s in range(len(st))}
    prod_meta["earth_ephemeris_method"] = prod_prov[0].earth_ephemeris_method
    y_s, y_mut, y_floor_p, y_floor_s, f_idx, y_f, diag = [], [], [], [], [], [], []
    y_s_et = []
    hist = np.hstack([pg.earth_pos_mci_m, pg.earth_vel_mci_mps])
    diag_idx = set(np.unique(np.linspace(0, max(rows.n - 1, 0), DIAG_ROWS).round().astype(int)).tolist())
    from lunar_od.two_way_range import TwoWayEventConvergenceError
    kept, failures = [], []
    for i in range(rows.n):
        t, s = float(rows.t_mid[i]), int(rows.station[i])
        try:
            rs = exact_station_single_bounce_counted_doppler_reference(t, oracle[s], rb.t_grid, x6, rcfg)
            ys_et = exact_station_single_bounce_counted_doppler_reference(t, oracle_et[s], rb.t_grid, x6,
                                                                          rcfg).observable
            yf_s = exact_station_single_bounce_counted_doppler_reference(t, ref_linear[s], rb.t_grid, x6,
                                                                         rcfg).observable
            yf_f = (exact_station_four_event_counted_doppler_reference(t, oracle[s], rb.t_grid, x6, rcfg).observable
                    if i % F_STRIDE == 0 else None)
        except TwoWayEventConvergenceError as exc:      # oracle-side convergence floor: counted, never hidden
            failures.append(dict(i=i, t_mid=t, station=s, error=str(exc)[:240]))
            continue
        kept.append(i)
        y_s.append(rs.observable)
        y_mut.append(two_way_counted_doppler_observable(t, st[s], rb.t_grid, x6, pg.earth_pos_mci_m,
                                                        pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93, cfg,
                                                        et0_s=pg.et0_s, station_state_provider=linear[s]))
        y_floor_s.append(yf_s)
        y_s_et.append(ys_et)
        y_floor_p.append(y_mut[-1])
        if yf_f is not None:
            f_idx.append(len(kept) - 1)
            y_f.append(yf_f)
        if i in diag_idx:
            ev = {"t3s": rs.start_solution.t3_s, "t1s": rs.start_solution.t1_s,
                  "t3e": rs.end_solution.t3_s, "t1e": rs.end_solution.t1_s}
            row = {}
            for name, te in ev.items():
                spice_e = _spice_earth_state_mci(pg.et0_s + te)
                lin = np.concatenate([_interp_vector(rb.t_grid, pg.earth_pos_mci_m, te),
                                      _interp_vector(rb.t_grid, pg.earth_vel_mci_mps, te)])
                her = _interp_state(rb.t_grid, hist, te)
                j = int(np.searchsorted(rb.t_grid, te, side="right")) - 1
                row[name] = dict(frac_in_interval=float((te - rb.t_grid[j]) / (rb.t_grid[j + 1] - rb.t_grid[j])),
                                 lin_err=(lin[:3] - spice_e[:3]).tolist(), herm_err=(her[:3] - spice_e[:3]).tolist(),
                                 lin_vel_err=float(np.linalg.norm(lin[3:] - spice_e[3:])),
                                 herm_vel_err=float(np.linalg.norm(her[3:] - spice_e[3:])))
            sc = rb.nom48[int(rows.node[i]), :3]
            los = sc - oracle[s].state(t)[:3]
            los /= np.linalg.norm(los)
            asym_lin = np.dot(np.subtract(row["t1e"]["lin_err"], row["t1s"]["lin_err"]), los)
            sym_lin = np.dot(np.subtract(row["t3e"]["lin_err"], row["t3s"]["lin_err"]), los)
            diag.append(dict(i=i, events=row, transmit_los_diff_m=float(asym_lin), receive_los_diff_m=float(sym_lin),
                             implied_offset_mps=float((asym_lin + sym_lin) / (2 * 60.0)),
                             mutant_minus_oracle_mps=float(y_mut[-1] - y_s[-1])))
    y_p = np.asarray(rows.y, float)[kept]
    y_s, y_mut = np.asarray(y_s), np.asarray(y_mut)
    out = dict(orbit=orbit, k=k, dynamics=dyn, n_rows=int(len(kept)), n_rows_total=int(rows.n),
               oracle_failures=failures, prod=prod_meta,
               value=dict(prod_minus_oracleS_max=float(np.max(np.abs(y_p - y_s))) if kept else 0.0,
                          prod_minus_oracleS_rms=float(np.sqrt(np.mean((y_p - y_s) ** 2))) if kept else 0.0,
                          prod_minus_oracleF_max=float(np.max(np.abs(y_p[f_idx] - np.asarray(y_f)))) if f_idx else 0.0,
                          oracleS_minus_oracleF_max=float(np.max(np.abs(y_s[f_idx] - np.asarray(y_f)))) if f_idx else 0.0,
                          mutant_minus_oracleS_max=float(np.max(np.abs(y_mut - y_s))) if kept else 0.0,
                          mutant_minus_oracleS_rms=float(np.sqrt(np.mean((y_mut - y_s) ** 2))) if kept else 0.0,
                          mutant_minus_oracleS_mean=float(np.mean(y_mut - y_s)) if kept else 0.0,
                          solver_floor_max=float(np.max(np.abs(np.asarray(y_floor_p) - np.asarray(y_floor_s))))
                          if kept else 0.0,
                          prod_minus_oracleS_abs=np.abs(y_p - y_s).tolist(),
                          prod_minus_oracleET_abs=np.abs(y_p - np.asarray(y_s_et)).tolist()),
               diagnostics=diag)
    # ---------------- CD1 requalification (repaired production partials)
    if rows.n >= 7:
        w60 = np.full(rows.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
        f0 = float(D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])["f_perp"])
        cd1 = {"range": f0}
        for name, model, wr in (("DOP|D0", "D0_IDEAL", False), ("DOP|D1|FREE", "D1_SESSION_CONSTANT_BIAS", False),
                                ("RPD|D0", "D0_IDEAL", True), ("RPD|D1|FREE", "D1_SESSION_CONSTANT_BIAS", True)):
            cd1[name] = float(CD.metrics(D2, rb, rows, w60, model, None, with_range=wr)["f_perp"])
        out["cd1"] = cd1
        # ---------------- KP1 requalification
        import phase17_kp1_campaign as KPC
        bank = KP.sensitivity_bank(rb.nom48, rb.t_grid)
        rbank, dbank = KP.range_parameter_columns(rb, bank), KP.doppler_parameter_columns(rb, rows, st, 60.0, bank)
        kp = {}
        for case in KPC.CASES:
            c, p = KPC.case_inputs(rb, rows, rbank, dbank, case)
            kp[case] = {}
            for m in KP.MODELS:
                mm = KP.block_metrics(c, p, m)
                kp[case][m] = dict(s_min=mm["s_min"], s_var_min=mm.get("s_var_min"), max_abs_corr=mm.get("max_abs_corr"))
        out["kp1"] = kp
    out["elapsed_s"] = time.time() - t0
    return K._clean(out)


# ======================================================================
# analysis
# ======================================================================
def analyse(acc: dict, cells: list, fd: dict | None) -> dict:
    a = acc["criteria"]
    v = lambda key: [c["value"][key] for c in cells if c["n_rows"]]  # noqa: E731
    allabs = np.concatenate([np.asarray(c["value"]["prod_minus_oracleS_abs"]) for c in cells if c["n_rows"]])
    worst = max((c for c in cells if c["n_rows"]), key=lambda c: c["value"]["prod_minus_oracleS_max"])
    value = dict(rows=int(allabs.size), cells=len(cells),
                 prod_minus_oracleS_max=float(allabs.max()), prod_minus_oracleS_rms=float(np.sqrt(np.mean(allabs ** 2))),
                 prod_minus_oracleS_p50=float(np.percentile(allabs, 50)),
                 prod_minus_oracleS_p99=float(np.percentile(allabs, 99)),
                 worst_cell=dict(orbit=worst["orbit"], k=worst["k"], dynamics=worst["dynamics"]),
                 prod_minus_oracleF_max=float(max(v("prod_minus_oracleF_max"))),
                 oracleS_minus_oracleF_max=float(max(v("oracleS_minus_oracleF_max"))),
                 mutant_minus_oracleS_max=float(max(v("mutant_minus_oracleS_max"))),
                 mutant_minus_oracleS_rms_median=float(np.median(v("mutant_minus_oracleS_rms"))),
                 solver_floor_max=float(max(v("solver_floor_max"))),
                 production_earth_method=sorted({c["prod"]["earth_ephemeris_method"] for c in cells}),
                 rows_total=int(sum(c["n_rows_total"] for c in cells)),
                 oracle_convergence_failures=[dict(orbit=c["orbit"], k=c["k"], dynamics=c["dynamics"], **f)
                                              for c in cells for f in c["oracle_failures"]])
    value["reduction_factor"] = value["mutant_minus_oracleS_max"] / max(value["prod_minus_oracleS_max"], 1e-300)
    et_abs = np.concatenate([np.asarray(c["value"]["prod_minus_oracleET_abs"]) for c in cells if c["n_rows"]])
    value["SECONDARY_POSTHOC_et_exact_oracle"] = dict(
        label="post-hoc diagnostic, not a frozen gate: model S with ET-exact Earth (two-sum), site as production",
        max=float(et_abs.max()), rms=float(np.sqrt(np.mean(et_abs ** 2))), p99=float(np.percentile(et_abs, 99)))
    diag = [d for c in cells for d in c["diagnostics"]]
    epochs = dict(
        n=len(diag),
        transmit_los_diff_m_abs_max=float(max(abs(d["transmit_los_diff_m"]) for d in diag)),
        receive_los_diff_m_abs_max=float(max(abs(d["receive_los_diff_m"]) for d in diag)),
        linear_pos_err_max_m=float(max(np.linalg.norm(e["lin_err"]) for d in diag for e in d["events"].values())),
        hermite_pos_err_max_m=float(max(np.linalg.norm(e["herm_err"]) for d in diag for e in d["events"].values())),
        hermite_vel_err_max_mps=float(max(e["herm_vel_err"] for d in diag for e in d["events"].values())),
        implied_vs_mutant_corr=float(np.corrcoef([d["implied_offset_mps"] for d in diag],
                                                 [d["mutant_minus_oracle_mps"] for d in diag])[0, 1]),
        transmit_frac_start_median=float(np.median([d["events"]["t1s"]["frac_in_interval"] for d in diag])),
        transmit_frac_end_median=float(np.median([d["events"]["t1e"]["frac_in_interval"] for d in diag])))
    # CD1
    committed = {(r["orbit"], int(r["k"]), r["dynamics"]): r for r in csv.DictReader(
        (ARTIFACTS / "phase17_r1o_cd1_primary_results.csv").open())}
    cols = {"range": "range_f_perp", "DOP|D0": "doppler_only_f_perp_D0", "DOP|D1|FREE": "doppler_only_f_perp_D1FREE",
            "RPD|D0": "combined_f_perp_D0", "RPD|D1|FREE": "combined_f_perp_D1FREE"}
    dfs, flips, cd1_cells = [], [], []
    for c in cells:
        if "cd1" not in c:
            continue
        r = committed[(c["orbit"], c["k"], c["dynamics"])]
        for key, col in cols.items():
            dfs.append(abs(c["cd1"][key] - float(r[col])))
        for key, dcol in (("RPD|D0", "delta_D0"), ("RPD|D1|FREE", "delta_D1FREE")):
            new_d = c["cd1"][key] - c["cd1"]["range"]
            if D2.relative_gain_class(new_d) != D2.relative_gain_class(float(r[dcol])) \
                    or D2.absolute_class(c["cd1"][key]) != D2.absolute_class(float(r[cols[key]])) \
                    or (new_d > 0) != (float(r[dcol]) > 0):
                flips.append(dict(orbit=c["orbit"], k=c["k"], dynamics=c["dynamics"], config=key))
        cd1_cells.append(c)
    hf = [c for c in cd1_cells if c["dynamics"] == "HIGH_FIDELITY"]
    simp = [c for c in cd1_cells if c["dynamics"] == "SIMPLIFIED"]
    p0 = K.population([c["cd1"]["RPD|D0"] - c["cd1"]["range"] for c in hf])
    p1 = K.population([c["cd1"]["RPD|D1|FREE"] - c["cd1"]["range"] for c in hf])
    abs_ok = float(np.mean([c["cd1"]["RPD|D1|FREE"] >= 0.35 for c in hf])) >= 0.50
    ps0 = K.population([c["cd1"]["RPD|D0"] - c["cd1"]["range"] for c in simp])
    cd1 = dict(f_perp_abs_diff_max=float(max(dfs)), class_or_sign_changes=flips,
               PRIMARY_CLASS=K.decision(p0, p1, abs_ok), POP_D0=p0["POP"], POP_D1FREE=p1["POP"], ABS=abs_ok,
               POP_D0_SIMPLIFIED=ps0["POP"])
    cd1_sum = json.loads((ARTIFACTS / "phase17_r1o_cd1_summary.json").read_text(encoding="utf-8"))
    cd1["decision_identical"] = bool(cd1["PRIMARY_CLASS"] == cd1_sum["PRIMARY_CLASS"] and not flips)
    # KP1
    kp_raw = {(c["orbit"], c["k"], c["dynamics"]): c for c in json.loads(
        (ARTIFACTS / "phase17_kp1_raw_results.json").read_text(encoding="utf-8"))}
    kp_sum = json.loads((ARTIFACTS / "phase17_kp1_summary.json").read_text(encoding="utf-8"))
    sd = []
    for c in cd1_cells:
        old = kp_raw[(c["orbit"], c["k"], c["dynamics"])]["cases"]
        for case, models in c["kp1"].items():
            for m, x in models.items():
                sd.append(abs(x["s_min"] - old[case][m]["s_min"]))
                if x["s_var_min"] is not None:
                    sd.append(abs(x["s_var_min"] - old[case][m]["s_var_min"]))
    classes, var_flip = {}, 0
    for m in ("M1_TWO_SEGMENT", "M2_LINEAR_DRIFT", "M3_THREE_SEGMENT"):
        sv = [c["kp1"]["RPD_D1FREE"][m]["s_var_min"] for c in hf]
        mc = [c["kp1"]["RPD_D1FREE"][m]["max_abs_corr"] for c in hf]
        var, intf = float(np.mean(np.asarray(sv) >= 0.35)), float(np.mean(np.asarray(mc) <= 0.90))
        classes[m] = "DISTINGUISHABLE" if var >= 0.5 and intf >= 0.5 else "PARTIAL" if var >= 0.10 else \
            "NO_NEW_INFORMATION"
        for c in hf:
            old = kp_raw[(c["orbit"], c["k"], "HIGH_FIDELITY")]["cases"]["RPD_D1FREE"][m]["s_var_min"]
            var_flip += int((old >= 0.35) != (c["kp1"]["RPD_D1FREE"][m]["s_var_min"] >= 0.35))
    kp1 = dict(metric_abs_diff_max=float(max(sd)), per_model_class=classes,
               classes_identical=bool(classes == kp_sum["per_model_class"]), per_cell_var_flips=var_flip)
    # gates (criteria frozen in the acceptance file)
    gates = dict(
        DIRECT_SPICE_VALUE_ORACLE="PASS" if (not value["oracle_convergence_failures"]   # stricter: every row
                                             and value["prod_minus_oracleS_max"] <= a["value_max_abs_mps"]
                                             and value["prod_minus_oracleF_max"] <= a["value_max_abs_mps"]
                                             and value["prod_minus_oracleS_rms"] <= a["value_rms_mps"]) else "FAIL",
        REDUCTION="PASS" if value["reduction_factor"] >= a["min_reduction_factor"] else "FAIL",
        HISTORICAL_D23_MUTANT_REPRODUCED="PASS" if a["mutant_reproduction_mps"][0] <= value["mutant_minus_oracleS_max"]
        <= a["mutant_reproduction_mps"][1] else "FAIL",
        SOLVER_FLOOR="PASS" if value["solver_floor_max"] <= a["solver_floor_max_mps"] else "FAIL",
        EARTH_EPOCH_CONTRACT="PASS" if (epochs["hermite_pos_err_max_m"] <= a["earth_epoch_pos_err_max_m"]
                                        and set(value["production_earth_method"]) == {a["production_earth_method"]})
        else "FAIL",
        CD1_INFORMATION="PASS" if (cd1["decision_identical"] and cd1["f_perp_abs_diff_max"] <= a["cd1_f_perp_max"])
        else "FAIL",
        KP1_INFORMATION="PASS" if (kp1["classes_identical"] and kp1["metric_abs_diff_max"] <= a["kp1_metric_max"])
        else "FAIL")
    if fd is not None:
        gates["CD1_FD_ORACLE"] = "PASS" if (fd["COUNTED_DOPPLER_STATE_FD_GATE"] == "PASS"
                                            and fd["COUNTED_DOPPLER_K_FD_GATE"] == "PASS") else "FAIL"
    gates["PHASE17_HARDEN_D23_REPAIR_GATE"] = "PASS" if all(x == "PASS" for x in gates.values()) else "FAIL"
    out = dict(phase="17-HARDEN-D23", acceptance_sha256=ACCEPTANCE_SHA256, value=value, epochs=epochs,
               cd1=cd1, kp1=kp1, fd=fd, gates=gates)
    (ARTIFACTS / "phase17_d23_qualification.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--workers", type=int, default=K.N_WORKERS)
    ap.add_argument("--reuse-fd", action="store_true",
                    help="reuse artifacts/phase17_d23_cd1_fd_rerun.json (computed on the same repaired production)")
    ap.add_argument("--epoch-scan", action="store_true",
                    help="post-hoc: production Hermite Earth vs ET-exact and raw SPICE at 7 points in EVERY grid "
                         "interval of all 24 epochs (the Earth grid depends on the epoch only, not the orbit)")
    ap.add_argument("--et-exact-diagnostic", action="store_true",
                    help="post-hoc diagnostic: production vs ET-exact SPICE oracle on SIMPLIFIED cells "
                         "(5 orbits x k = 0, 6, 12, 18); never part of the frozen gate")
    args = ap.parse_args()
    acc = load_acceptance()
    if args.epoch_scan:
        K._init()
        from lunar_od.radiometrics import _interp_state
        from phase17_r1m_core import EARTH_FIXTURE_CANONICAL, _spice_earth_state_mci, measurement_earth_ephemeris
        g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
        x0 = [x["x0"] for x in g["geometries"] if x["case_id"] == "CANON"][0]
        fracs = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
        rows = []
        for k in K.EPOCHS:
            scn = K.scenario("CANON", x0, k)
            et0 = float(scn.et0)
            tg = scn.t_grid
            earth = measurement_earth_ephemeris(EARTH_FIXTURE_CANONICAL, et0, float(tg[-1]))
            hist = np.hstack([earth.pos(tg), earth.vel(tg)])
            e_exact, e_raw, d_max = 0.0, 0.0, 0.0
            for i in range(tg.size - 1):
                for f in fracs:
                    t = float(tg[i] + f * (tg[i + 1] - tg[i]))
                    h = _interp_state(tg, hist, t)[:3]
                    s = et0 + t
                    bb = s - et0
                    d = (et0 - (s - bb)) + (t - bb)
                    sp = _spice_earth_state_mci(s)
                    e_raw = max(e_raw, float(np.linalg.norm(h - sp[:3])))
                    e_exact = max(e_exact, float(np.linalg.norm(h - (sp[:3] + sp[3:] * d))))
                    d_max = max(d_max, abs(d))
            rows.append(dict(k=k, et0=et0, n_points=int((tg.size - 1) * len(fracs)), hermite_vs_et_exact_max_m=e_exact,
                             hermite_vs_raw_spice_max_m=e_raw, et_rounding_max_s=d_max))
            print(k, "%.3g %.3g" % (e_exact, e_raw), flush=True)
        out = dict(label="POST-HOC epoch-contract scan (frozen criterion: earth_epoch_pos_err_max_m <= 1e-6)",
                   epochs=rows, hermite_vs_et_exact_max_m=max(r["hermite_vs_et_exact_max_m"] for r in rows),
                   hermite_vs_raw_spice_max_m=max(r["hermite_vs_raw_spice_max_m"] for r in rows),
                   et_rounding_max_s=max(r["et_rounding_max_s"] for r in rows),
                   points=int(sum(r["n_points"] for r in rows)))
        (ARTIFACTS / "phase17_d23_epoch_scan.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "epochs"}, indent=1))
        return
    if args.et_exact_diagnostic:
        g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
        x0s = {x["case_id"]: x["x0"] for x in g["geometries"] if x["case_id"] in K.ORBITS}
        specs = [dict(orbit=o, x0=x0s[o], k=k) for o in K.ORBITS for k in (0, 6, 12, 18)]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=K._init) as ex:
            res = list(ex.map(et_exact_diagnostic, specs))
        summ = {lab: dict(max=max(r[lab]["max"] for r in res), rms_max=max(r[lab]["rms"] for r in res))
                for lab in ("earth_only", "earth_and_site")}
        (ARTIFACTS / "phase17_d23_et_exact_diagnostic.json").write_text(json.dumps(K._clean(dict(
            label="POST-HOC DIAGNOSTIC (not the frozen acceptance oracle)", cells=res, summary=summ)), indent=1),
            encoding="utf-8")
        print(json.dumps(summ, indent=1))
        return
    fd_path = ARTIFACTS / "phase17_d23_cd1_fd_rerun.json"
    if args.analyse_only:
        cells = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        fd = json.loads(fd_path.read_text(encoding="utf-8")) if fd_path.exists() else None
    else:
        import hashlib
        pre = CD.load_predeclaration(K.PREDECLARATION_SHA256)
        g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
        x0s = {x["case_id"]: x["x0"] for x in g["geometries"] if x["case_id"] in K.ORBITS}
        for o, x0 in x0s.items():
            assert hashlib.sha256(json.dumps(x0).encode()).hexdigest() == pre["orbits"]["set"][o]["x0_sha256"]
        specs = [dict(orbit=o, x0=x0s[o], k=k, dynamics=dyn) for dyn in ("HIGH_FIDELITY", "SIMPLIFIED")
                 for o in K.ORBITS for k in K.EPOCHS]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=K._init) as ex:
            if args.reuse_fd and fd_path.exists():
                fd = json.loads(fd_path.read_text(encoding="utf-8"))
            else:
                fd = K.run_fd(ex, x0s)
                fd_path.write_text(json.dumps(K._clean(fd), indent=1), encoding="utf-8")
            print("CD1 FD rerun: state %s K %s (%.0f s)" % (fd["COUNTED_DOPPLER_STATE_FD_GATE"],
                                                         fd["COUNTED_DOPPLER_K_FD_GATE"], time.time() - t0), flush=True)
            cells = []
            for i, c in enumerate(ex.map(run_cell, specs)):
                cells.append(c)
                print("%3d/%d %s k%02d %s %.0f s" % (i + 1, len(specs), c["orbit"], c["k"], c["dynamics"],
                                                     time.time() - t0), flush=True)
        RAW_PATH.write_text(json.dumps(cells, separators=(",", ":")), encoding="utf-8")
    out = analyse(acc, cells, fd)
    print(json.dumps(out["gates"], indent=1))
    print("value max %.3g rms %.3g | mutant max %.3g | floor %.3g | reduction %.3g"
          % (out["value"]["prod_minus_oracleS_max"], out["value"]["prod_minus_oracleS_rms"],
             out["value"]["mutant_minus_oracleS_max"], out["value"]["solver_floor_max"],
             out["value"]["reduction_factor"]))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
