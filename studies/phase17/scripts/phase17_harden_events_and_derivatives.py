"""PHASE 17-HARDEN s24-s28, s43, s44, s53 - measurement event ledgers, derivative
qualification (counted Doppler, optical, ΔDOR spot check) and cross-measurement
geometry consistency, on the canonical epoch-consistent geometry.

ANALYSIS SPACE ONLY.  Every observable and Jacobian is the production one.

Oracles are independent: perturbed trajectories come from the 6-state
`lunar_od.dynamics.propagate_state` (no STM, no dx/dK), and the production
observable is re-evaluated on them with the observation definition FROZEN
(receive tag, station, count interval; for optical: epoch, landmark AND camera
attitude -- attitude is an independently measured input, not a function of the
estimated state).  K columns: re-propagation with K +/- dK.

Outputs: artifacts/phase17_harden_{range,doppler,ddor,optical}_event_ledger.json,
artifacts/phase17_harden_derivatives.json, artifacts/phase17_harden_cross_measurement.json

``--f1`` (Phase 17-F1 requalification): identical checks plus a two-way range
block, written ONLY to artifacts/phase17_f1_measurement_derivatives.json under a
run tag; the HARDEN artifacts (the pre-F1 evidence) are left untouched.
  --rtol X            analytic (48-state) AND oracle (6-state) tolerance
                      (default: science analytic 1e-12, oracle 1e-13)
  --old-variational   ANALYSIS-ONLY negative control: the SRP position gradient
                      is zeroed in-process, reproducing the pre-F1 variational
                      equations in every path (no production switch exists)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"
RTOL_ORACLE = 1e-13
F1_MODE = "--f1" in sys.argv
RTOL_ANALYTIC = None                    # None = the science default of propagate_nom48
if "--rtol" in sys.argv:
    RTOL_ORACLE = RTOL_ANALYTIC = float(sys.argv[sys.argv.index("--rtol") + 1])
OLD_VARIATIONAL = "--old-variational" in sys.argv
RUN_TAG = "%s_%s" % ("OLD" if OLD_VARIATIONAL else "NEW",
                     "science_config" if RTOL_ANALYTIC is None else "rtol%g" % RTOL_ANALYTIC)


def _write(name: str, text: str) -> None:
    """HARDEN evidence files are never rewritten by an F1 requalification run."""
    if not F1_MODE:
        (ARTIFACTS / name).write_text(text)


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()
    if OLD_VARIATIONAL:
        import lunar_od.srp as srp_mod
        srp_mod.srp_acceleration_position_gradient = lambda *_a, **_k: np.zeros((3, 3))
    if RTOL_ANALYTIC is not None:
        import phase17_geo_core as G
        D2.propagate_nom48 = lambda scn: G.propagate48(scn.t_grid, scn.x0_array, scn.et0, scn.k_srp,
                                                       rtol=RTOL_ANALYTIC, atol=0.1 * RTOL_ANALYTIC,
                                                       shadow_model=scn.shadow_model)


def _prop6(scn, t_grid, x0, k):
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions
    return propagate_state(t_grid, x0, G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]),
                           lambda tt: G.sun_mci(scn.et0 + float(tt)),
                           srp=SRPOptions(k_srp_m2_per_kg=k, shadow_model=scn.shadow_model),
                           rtol=RTOL_ORACLE, atol=RTOL_ORACLE * 0.1, j2_moon=J2_MOON_UNNORMALIZED)


def _fd_columns(scn, t_grid, observe, steps_pos=(10.0, 100.0), steps_vel=(0.01, 0.1),
                steps_k=(1e-3, 3e-3)):
    """observe(state_history_6col) -> (M,) observables on a FROZEN observation set."""
    out = {}
    for j in range(7):
        steps = steps_pos if j < 3 else steps_vel if j < 6 else steps_k
        fds = []
        for h in steps:
            vals = []
            for sgn in (+1, -1):
                x0 = scn.x0_array.copy()
                k = scn.k_srp
                if j < 6:
                    x0[j] += sgn * h
                else:
                    k += sgn * h
                vals.append(observe(_prop6(scn, t_grid, x0, k)))
            fds.append((vals[0] - vals[1]) / (2 * h))
        out[j] = (steps, fds)
    return out


def _rel(fd, an):
    return float(np.linalg.norm(fd - an) / max(np.linalg.norm(an), 1e-300))


def main() -> None:
    _init()
    import spiceypy as spice

    import phase17_geo_core as G
    import phase17_r1m_core as M
    import phase17_r1od2_core as D2
    from lunar_od.radiometrics import (RangeRatePhysicsConfig, resolve_counted_doppler_station_state_provider,
                                       solve_two_way_light_time, two_way_counted_doppler_initial_state_jacobian,
                                       two_way_counted_doppler_observable)
    from lunar_od.two_way_range import (_station_state_providers_for_pass, solve_two_way_range_events,
                                        two_way_range_config)

    names = ("x0", "y0", "z0", "vx0", "vy0", "vz0", "K_SRP")
    et0, t_orbit = M.campaign_epoch()
    scn = D2.Scenario("CANON_4ORB", "harden", et0, tuple(M.campaign_initial_state().tolist()), 4.0 * t_orbit)
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    pg = rb.pass_geo
    by = D2.stations_by_name()
    derivs, cross = {}, {}

    # ------------------------------------------------------------------ range ledger
    providers = _station_state_providers_for_pass(pg)
    cfg_r = two_way_range_config(pg.two_way_range)
    pick = np.unique(np.linspace(0, rb.n_obs - 1, 8).round().astype(int))
    range_ledger = []
    for n in pick:
        t3, sid = float(rb.obs[n, 0]), int(rb.obs[n, 2]) - 1
        sol = solve_two_way_range_events(t3, providers[sid], rb.t_grid, nom[:, :6], cfg_r)
        ev = {}
        for tag, t in (("t1_uplink_transmit", sol.t1_s), ("t2u_uplink_spacecraft", sol.t2u_s),
                       ("t2d_downlink_spacecraft", sol.t2d_s), ("t3_receive", sol.t3_s)):
            st = providers[sid].state(t) if tag in ("t1_uplink_transmit", "t3_receive") else None
            earth = rb.earth.pos([t])[0]
            ev[tag] = dict(t_local_s=t, et_s=pg.et0_s + t,
                           station_state_epoch_et=(pg.et0_s + t) if st is not None else None,
                           earth_ephemeris_epoch_et=pg.et0_s + t,
                           earth_orientation_epoch_et=(pg.et0_s + t) if st is not None else None,
                           spacecraft_state_epoch_et=(pg.et0_s + t) if st is None else None,
                           earth_pos_matches_spice_m=float(np.max(np.abs(
                               earth - M._spice_earth_state_mci(pg.et0_s + t)[:3]))))
        range_ledger.append(dict(obs_index=int(n), station=pg.stations[sid].name, frame="J2000 axes",
                                 origin="Moon centre", units="s, m", events=ev,
                                 round_trip_s=sol.t3_s - sol.t1_s,
                                 converged=bool(sol.converged)))
    _write("phase17_harden_range_event_ledger.json", json.dumps(dict(
        contract="t3 fixed (grid receive tag); t2d, t2u, t1 solved; station at t1/t3 via exact sxform "
                 "at pass_geo.et0_s + t and Hermite Earth; spacecraft at t2u/t2d by Hermite state "
                 "interpolation; all local times share origin pass_geo.et0_s",
        et0_s=pg.et0_s, rows=range_ledger), indent=1))

    # ------------------------------------------------------------------ counted Doppler
    cfg_d = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0)
    dsel = [n for n in pick if 60.0 < rb.obs[n, 0] < rb.t_grid[-1] - 60.0][:6]
    dop_rows = [(float(rb.obs[n, 0]), pg.stations[int(rb.obs[n, 2]) - 1]) for n in dsel]

    def dop(x6):
        return np.array([two_way_counted_doppler_observable(t, st, rb.t_grid, x6, pg.earth_pos_mci_m,
                                                            pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
                                                            cfg_d, et0_s=pg.et0_s) for t, st in dop_rows])

    an_d = np.array([two_way_counted_doppler_initial_state_jacobian(
        t, st, rb.t_grid, nom[:, :42], pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
        cfg_d, et0_s=pg.et0_s) for t, st in dop_rows])
    fake = nom[:, :42].copy()                       # K column: S_K in the first STM column slot
    fake[:, 6:42] = 0.0
    fake[:, 6:12] = nom[:, 42:48]
    an_dk = np.array([two_way_counted_doppler_initial_state_jacobian(
        t, st, rb.t_grid, fake, pg.earth_pos_mci_m, pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93,
        cfg_d, et0_s=pg.et0_s)[0] for t, st in dop_rows])
    an_d = np.hstack([an_d, an_dk[:, None]])
    fdd = _fd_columns(scn, rb.t_grid, dop)
    derivs["counted_doppler"] = {names[j]: dict(steps=list(s), rel_err=[_rel(f, an_d[:, j]) for f in fds])
                                 for j, (s, fds) in fdd.items()}
    dop_ledger = []
    for t, st in dop_rows:
        prov = resolve_counted_doppler_station_state_provider(cfg_d, st, rb.t_grid, pg.earth_pos_mci_m,
                                                              pg.earth_vel_mci_mps, et0_s=pg.et0_s)
        eps = {}
        for lab, tr in (("count_start", t - 30.0), ("count_end", t + 30.0)):
            sol = solve_two_way_light_time(tr, st, rb.t_grid, nom[:, :6], pg.earth_pos_mci_m,
                                           pg.earth_vel_mci_mps, pg.x_j2000_to_itrf93, cfg_d,
                                           station_state_provider=prov)
            eps[lab] = dict(receive_local_s=sol.receive_time_s, transponder_local_s=sol.transponder_time_s,
                            transmit_local_s=sol.transmit_time_s,
                            receive_et=pg.et0_s + sol.receive_time_s,
                            transmit_et=pg.et0_s + sol.transmit_time_s,
                            round_trip_s=sol.round_trip_light_time_s, converged=bool(sol.converged),
                            station_state_receive=prov.state(sol.receive_time_s).tolist(),
                            station_state_transmit=prov.state(sol.transmit_time_s).tolist())
        dop_ledger.append(dict(receive_mid_local_s=t, station=st.name, count_interval_s=60.0,
                               uplink_frequency_hz=cfg_d.uplink_frequency_hz,
                               turnaround_ratio=cfg_d.turnaround_ratio,
                               earth_interpolation=prov.earth_ephemeris_method,
                               station_state_method=prov.station_state_method, endpoints=eps,
                               distinct_endpoint_events=eps["count_start"]["transmit_local_s"]
                               != eps["count_end"]["transmit_local_s"]))
    _write("phase17_harden_doppler_event_ledger.json", json.dumps(dict(
        contract="two endpoints receive at t_mid -/+ Tc/2; each solves its own transponder and transmit "
                 "events; one provider shared by both endpoints (exact event-epoch sxform, LINEAR Earth "
                 "interpolation by the documented R3-P22 design)", et0_s=pg.et0_s, rows=dop_ledger), indent=1))

    # ------------------------------------------------------------------ cross-measurement geometry
    # the same station at the same epochs through the range, Doppler and ΔDOR paths
    st = by["Goldstone DSN"]
    prov_r = providers[D2.DSN3.index("Goldstone DSN")]
    prov_d = resolve_counted_doppler_station_state_provider(cfg_d, st, rb.t_grid, pg.earth_pos_mci_m,
                                                            pg.earth_vel_mci_mps, et0_s=pg.et0_s)
    rows_gc = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
    from lunar_od.radiometrics import _interp_state
    node_t = rb.t_grid[100:110]
    mid_t = node_t + 45.0
    def diffs(ts):
        r = np.array([prov_r.state(t) for t in ts])
        d = np.array([prov_d.state(t) for t in ts])
        ddr = np.array([_interp_state(rb.t_grid, rows_gc.hist_a, t) for t in ts])
        return dict(range_vs_doppler_pos_m=float(np.max(np.linalg.norm(r[:, :3] - d[:, :3], axis=1))),
                    range_vs_doppler_vel_mps=float(np.max(np.linalg.norm(r[:, 3:] - d[:, 3:], axis=1))),
                    range_vs_ddor_pos_m=float(np.max(np.linalg.norm(r[:, :3] - ddr[:, :3], axis=1))),
                    range_vs_ddor_vel_mps=float(np.max(np.linalg.norm(r[:, 3:] - ddr[:, 3:], axis=1))))
    exact_earth_mid = np.array([M._spice_earth_state_mci(pg.et0_s + t) for t in mid_t])
    lin_earth_mid = np.array([np.interp(t, rb.t_grid, pg.earth_pos_mci_m[:, a]) for t in mid_t
                              for a in range(3)]).reshape(-1, 3)
    herm_earth_mid = np.array([_interp_state(rb.t_grid, np.hstack([pg.earth_pos_mci_m, pg.earth_vel_mci_mps]), t)[:3]
                               for t in mid_t])
    cross = dict(
        at_grid_nodes=diffs(node_t), at_mid_nodes=diffs(mid_t),
        earth_interp_error_mid_node_m=dict(
            linear_doppler=float(np.max(np.linalg.norm(lin_earth_mid - exact_earth_mid[:, :3], axis=1))),
            hermite_range_ddor=float(np.max(np.linalg.norm(herm_earth_mid - exact_earth_mid[:, :3], axis=1)))),
        orientation_transform_same_epoch=bool(np.array_equal(
            np.asarray(spice.sxform("J2000", "ITRF93", pg.et0_s + float(node_t[0]))), pg.x_j2000_to_itrf93[100])),
        spacecraft_nominal_same=bool(np.array_equal(rows_gc.hist_a.shape, (rb.t_grid.size, 6))),
    )
    # counted-Doppler error implied by LINEAR vs exact Earth over one count interval
    def rr_err(tm):
        e0 = np.interp(tm - 30.0, rb.t_grid, pg.earth_pos_mci_m[:, 0]) - M._spice_earth_state_mci(pg.et0_s + tm - 30.0)[0]
        e1 = np.interp(tm + 30.0, rb.t_grid, pg.earth_pos_mci_m[:, 0]) - M._spice_earth_state_mci(pg.et0_s + tm + 30.0)[0]
        return abs(e1 - e0) / 60.0
    cross["linear_earth_count_rate_error_x_component_max_mps"] = float(max(rr_err(t) for t in rb.t_grid[100:160]))
    _write("phase17_harden_cross_measurement.json", json.dumps(cross, indent=1))

    # ------------------------------------------------------------------ ΔDOR ledger + light FD
    from lunar_od.delta_dor import solve_common_transmit_event
    ddor_ledger = []
    for n in np.unique(np.linspace(0, rows_gc.n - 1, 5).round().astype(int)):
        t_obs = float(rows_gc.t_used[n])
        a = _interp_state(rb.t_grid, rows_gc.hist_a, t_obs)[:3]
        c = solve_common_transmit_event(t_obs, a, lambda t: _interp_state(rb.t_grid, rows_gc.hist_b, t)[:3],
                                        lambda t: _interp_state(rb.t_grid, nom, t)[:3])
        ddor_ledger.append(dict(
            t_obs_local_s=t_obs, station_a_receive_et=pg.et0_s + t_obs,
            common_transmit_local_s=c.transmit_time_s, common_transmit_et=pg.et0_s + c.transmit_time_s,
            station_b_receive_local_s=c.station_b_receive_time_s,
            station_b_receive_et=pg.et0_s + c.station_b_receive_time_s,
            station_a_state_epoch="t_obs (Hermite of exact-sxform station history on the grid)",
            station_b_state_epoch="t_B",
            earth_orientation_epochs_et=[pg.et0_s + t_obs, pg.et0_s + c.station_b_receive_time_s],
            baseline_construction_epoch="grid epoch t_obs (weights / diagnostics only)",
            quasar_frame_contract="fixed J2000 unit vector; D_Q independent of spacecraft",
            both_stations_visible=bool(rb.vis[rows_gc.idx[n], 0] and rb.vis[rows_gc.idx[n], 2]),
            d_s_s=c.spacecraft_differential_delay_s))
    _write("phase17_harden_ddor_event_ledger.json", json.dumps(dict(
        pair="GC", row_rule=rows_gc.row_rule, rows=ddor_ledger), indent=1))
    sel = np.unique(np.linspace(0, rows_gc.n - 1, 12).round().astype(int))

    def ddor_obs(x6):
        out = []
        for n in sel:
            t_obs = float(rows_gc.t_used[n])
            a = _interp_state(rb.t_grid, rows_gc.hist_a, t_obs)[:3]
            c = solve_common_transmit_event(t_obs, a, lambda t: _interp_state(rb.t_grid, rows_gc.hist_b, t)[:3],
                                            lambda t: _interp_state(rb.t_grid, x6, t)[:3])
            out.append(c.spacecraft_differential_delay_s)
        return np.array(out)

    an_ddor = np.hstack([rows_gc.h_x0[sel], rows_gc.h_k[sel][:, None]])
    fdo = _fd_columns(scn, rb.t_grid, ddor_obs)
    derivs["ddor"] = {names[j]: dict(steps=list(s), rel_err=[_rel(f, an_ddor[:, j]) for f in fds])
                      for j, (s, fds) in fdo.items()}

    # ------------------------------------------------------------------ optical
    from lunar_od.constants import R_MOON_M
    from lunar_od.lunar_landmark_optical import (DEFAULT_CAMERA, landmark_inertial_position_m,
                                                 nadir_pointing_camera_frame, pinhole_project)
    from phase17_r1o_opt_core import build_production_optical_arc
    t30 = np.arange(0.0, scn.duration_s + 15.0, 30.0)
    scn30 = D2.Scenario("CANON_4ORB_30S", "harden", et0, scn.x0, scn.duration_s, cadence_s=30.0)
    nom30 = D2.propagate_nom48(scn30)
    arc = build_production_optical_arc(nom30, t30, et0, sigma_centroid_px=0.25)
    pairs = []
    for r in range(0, arc.n_rows, 2):
        i = int(np.searchsorted(t30, arc.t_used_s[r]))
        li = int(arc.landmark_used_idx[r])
        lat, lon = np.radians(arc.landmark_latlon_deg[li])
        pairs.append((i, landmark_inertial_position_m(lat, lon, R_MOON_M, et0 + t30[i]),
                      nadir_pointing_camera_frame(nom30[i, :3])))

    def opt(x6):
        v = []
        for i, r_lm, c_ci in pairs:
            p = pinhole_project(x6[i, :3], r_lm, c_ci, DEFAULT_CAMERA)
            v += [p.u_px, p.v_px]
        return np.array(v)

    an_o = np.hstack([arc.h_x0, arc.h_k[:, None]])
    fdp = _fd_columns(scn30, t30, opt)
    derivs["optical"] = {names[j]: dict(steps=list(s), rel_err=[_rel(f, an_o[:, j]) for f in fds])
                         for j, (s, fds) in fdp.items()}
    derivs["optical_rows"] = int(arc.n_rows)
    opt_ledger = [dict(grid_index=i, shutter_local_s=float(t30[i]), shutter_et=et0 + float(t30[i]),
                       spacecraft_state_epoch_et=et0 + float(t30[i]),
                       landmark_transform_epoch_et=et0 + float(t30[i]),
                       moon_orientation_frame="MOON_PA_DE440 (moon_pa_de440_rotation_at_et)",
                       camera_attitude="nadir frame from the NOMINAL position at the shutter epoch; frozen "
                                       "as an independently measured input",
                       light_time_applied=False,
                       light_time_s=float(np.linalg.norm(r_lm - nom30[i, :3]) / 299792458.0))
                  for i, r_lm, c_ci in pairs]
    _write("phase17_harden_optical_event_ledger.json", json.dumps(dict(
        contract="instantaneous: spacecraft, landmark and camera frame all at the shutter epoch; no light "
                 "time (documented R1O-OPT limitation)", rows=opt_ledger), indent=1))

    # ------------------------------------------------------------------ two-way range
    from lunar_od.two_way_range import two_way_range_nominal_and_initial_jacobian

    def rng(x6):
        hist = nom[:, :42].copy()
        hist[:, :6] = x6
        z, _ = two_way_range_nominal_and_initial_jacobian(rb.obs, rb.pass_geo, hist)
        return np.asarray(z, float).reshape(-1)

    an_r = np.hstack([rb.h_x0, rb.h_k[:, None]])
    fdr = _fd_columns(scn, rb.t_grid, rng)
    derivs["range"] = {names[j]: dict(steps=list(s), rel_err=[_rel(f, an_r[:, j]) for f in fds])
                       for j, (s, fds) in fdr.items()}

    derivs["n_rows"] = dict(counted_doppler=len(dop_rows), ddor=int(sel.size), optical=int(arc.n_rows),
                            range=int(rb.n_obs))
    if F1_MODE:
        f1_path = ARTIFACTS / "phase17_f1_measurement_derivatives.json"
        store = json.loads(f1_path.read_text()) if f1_path.exists() else {}
        store = store if "runs" in store else {"runs": {}}
        derivs["oracle_rtol"], derivs["analytic_rtol"] = RTOL_ORACLE, RTOL_ANALYTIC or 1e-12
        store["runs"][RUN_TAG] = derivs
        f1_path.write_text(json.dumps(store, indent=1))
    else:
        (ARTIFACTS / "phase17_harden_derivatives.json").write_text(json.dumps(derivs, indent=1))
    for m in ("range", "counted_doppler", "ddor", "optical"):
        print(m, {k: ["%.1e" % e for e in v["rel_err"]] for k, v in derivs[m].items()})
    print(json.dumps(cross, indent=1))


if __name__ == "__main__":
    main()
