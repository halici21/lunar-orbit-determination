"""PHASE 17-R1O-D2 s28-s30 - end-to-end finite-difference qualification of the
full production ΔDOR design (all rows, all seven columns).

ANALYSIS SPACE ONLY.

Oracle, per predeclared case (grid `fd_cases`):
  * the nominal PRODUCTION_VISIBILITY row set is FROZEN (receive tags t_obs and
    station histories fixed; visibility is never re-evaluated, so a perturbation
    cannot change the row set, s29)
  * each perturbed trajectory is an INDEPENDENT 6-state re-propagation
    (`lunar_od.dynamics.propagate_state`: no STM, no dx/dK) with the same force
    model, tolerances and 90 s grid as the nominal
  * the production `solve_common_transmit_event` re-solves every row on it
  * central differences over a predeclared step sweep; K by re-propagating with
    K +/- dK
A row whose event solve fails on any perturbed trajectory is excluded and listed.

Also checks s13 formal quasar invariance: the FD derivative of the full ΔDOR
observable D_S - D_Q is the same for three quasar directions.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parents[2] / "studies" / "phase17" / "results"
GRID_SHA256_LF = "2a99c50e95c49bc4c1837787c328bcc2537fa653a2e933bf5a9ea39221c07c11"


def _grid() -> dict:
    text = (ARTIFACTS.parent / "configs" / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8")
    assert hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest() == GRID_SHA256_LF
    return json.loads(text)


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()


@functools.lru_cache(maxsize=3)
def context(fd_key: str):
    import phase17_r1od2_core as D2
    grid = _grid()
    spec = grid["fd_cases"][fd_key]
    g = next(x for x in grid["geometries"] if x["case_id"] == spec["geometry"])
    scn = D2.Scenario(g["case_id"], g["family"], float(g["et0"]), tuple(g["x0"]),
                      float(g["duration_s"]), float(g["k_srp"]), g["shadow_model"])
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    rows = D2.build_ddor_rows(scn, rb, spec["pair"], D2.ROW_RULE_PRODUCTION_VISIBILITY)
    return scn, rb, rows


def ds_on_trajectory(scn, rows, x_hist):
    from lunar_od.delta_dor import DeltaDorEventError, solve_common_transmit_event
    from lunar_od.radiometrics import _interp_state

    t_grid = scn.t_grid
    out = np.full(rows.n, np.nan)
    for n, t_obs in enumerate(rows.t_used):
        a = _interp_state(t_grid, rows.hist_a, t_obs)[:3]
        try:
            c = solve_common_transmit_event(float(t_obs), a,
                                            lambda t: _interp_state(t_grid, rows.hist_b, t)[:3],
                                            lambda t: _interp_state(t_grid, x_hist, t)[:3])
            out[n] = c.spacecraft_differential_delay_s
        except DeltaDorEventError:
            pass
    return out


def perturbed(job):
    fd_key, col, h = job[:3]
    rtol, atol = (job[3], job[4]) if len(job) > 3 else (1e-12, 1e-13)
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions

    scn, _, rows = context(fd_key)
    x0 = scn.x0_array.copy()
    k = scn.k_srp
    if col < 6:
        x0[col] += h
    else:
        k += h
    t_grid = scn.t_grid
    x_hist = propagate_state(
        t_grid - t_grid[0], x0, G.mu_moon(), 0.0, 0.0,
        lambda _t: np.array([384_400e3, 0.0, 0.0]),
        lambda t: G.sun_mci(scn.et0 + t_grid[0] + float(t)),
        srp=SRPOptions(k_srp_m2_per_kg=float(k), shadow_model=scn.shadow_model),
        rtol=rtol, atol=atol, j2_moon=J2_MOON_UNNORMALIZED)
    return (fd_key, col, h, ds_on_trajectory(scn, rows, x_hist))


def k_refinement() -> None:
    """Floor-aware follow-up for the K column ONLY (s30): the predeclared oracle
    (rtol 1e-12) has a halving floor of a few 1e-2 in K, so it cannot resolve
    the predeclared 1e-3 gate.  Same predeclared K steps, tighter-tolerance
    independent oracle; the gate itself is unchanged."""
    import phase17_r1od2_core as D2

    grid = _grid()
    fdg = grid["fd_cases"]
    keys = [k for k in fdg if k.startswith("FD")]
    out = {}
    for rtol, atol in ((1e-13, 1e-14), (1e-14, 1e-15)):
        jobs = [(k, 6, h, rtol, atol) for k in keys for h in fdg["k_steps"]]
        jobs += [(k, 6, -h, rtol, atol) for k in keys for h in fdg["k_steps"]]
        with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
            res = list(ex.map(perturbed, jobs, chunksize=1))
        ds = {(k, h): v for k, c, h, v in res}
        _init()
        for key in keys:
            _, rb, rows = context(key)
            an = rows.h_k
            fds = [(ds[(key, h)] - ds[(key, -h)]) / (2 * h) for h in fdg["k_steps"]]
            errs = [float(np.linalg.norm(f - an) / np.linalg.norm(an)) for f in fds]
            halv = [float(np.linalg.norm(fds[i] - fds[i + 1]) / np.linalg.norm(an))
                    for i in range(len(fds) - 1)]
            out.setdefault(key, {})["rtol_%g" % rtol] = dict(rel_err=errs, halving=halv,
                                                             best_rel_err=min(errs),
                                                             oracle_floor=min(halv))
            print("%s rtol %g  K rel err %s  halving %s" % (key, rtol, np.array2string(np.array(errs), precision=2),
                                                            np.array2string(np.array(halv), precision=2)))
    (ARTIFACTS / "phase17_r1od2_fd_k_refinement.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    import phase17_r1od2_core as D2

    grid = _grid()
    fdg = grid["fd_cases"]
    keys = [k for k in fdg if k.startswith("FD")]
    pos, vel, ks = fdg["position_steps_m"], fdg["velocity_steps_mps"], fdg["k_steps"]
    jobs = []
    for key in keys:
        for col in range(7):
            steps = pos if col < 3 else vel if col < 6 else ks
            for h in steps:
                jobs += [(key, col, h), (key, col, -h)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        res = list(ex.map(perturbed, jobs, chunksize=1))
    print("perturbed propagations: %d in %.0f s" % (len(jobs), time.time() - t0))
    ds = {(k, c, h): v for k, c, h, v in res}

    _init()
    out = {}
    names = ("x0", "y0", "z0", "vx0", "vy0", "vz0", "K_SRP")
    for key in keys:
        scn, rb, rows = context(key)
        an = np.hstack([rows.h_x0, rows.h_k[:, None]])
        bad = np.zeros(rows.n, bool)
        for (k, c, h), v in ds.items():
            if k == key:
                bad |= ~np.isfinite(v)
        good = ~bad
        cols = {}
        fd_best = np.zeros_like(an)
        for col in range(7):
            steps = pos if col < 3 else vel if col < 6 else ks
            fds, errs = [], []
            for h in steps:
                fd = (ds[(key, col, h)] - ds[(key, col, -h)]) / (2.0 * h)
                fds.append(fd)
                errs.append(float(np.linalg.norm(fd[good] - an[good, col]) / np.linalg.norm(an[good, col])))
            halving = [float(np.linalg.norm(fds[i][good] - fds[i + 1][good]) / np.linalg.norm(an[good, col]))
                       for i in range(len(steps) - 1)]
            ib = int(np.argmin(errs))
            fd_best[:, col] = fds[ib]
            cols[names[col]] = dict(steps=steps, rel_err=errs, halving_diff=halving,
                                    best_step=steps[ib], best_rel_err=errs[ib],
                                    oracle_floor=float(min(halving)),
                                    analytic_norm=float(np.linalg.norm(an[good, col])))
        w5 = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
        rblock = (rb.h_x0, rb.h_k, rb.w)
        m_an = D2.info_metrics([rblock, (an[good, :6], an[good, 6], w5[good])])
        m_fd = D2.info_metrics([rblock, (fd_best[good, :6], fd_best[good, 6], w5[good])])
        m_r = D2.info_metrics([rblock])
        state_max = max(cols[n]["best_rel_err"] for n in names[:6])
        k_err = cols["K_SRP"]["best_rel_err"]
        out[key] = dict(
            geometry=scn.case_id, pair=rows.pair, n_rows=rows.n, n_excluded=int(bad.sum()),
            excluded_t_obs=rows.t_used[bad].tolist(), columns=cols,
            max_state_best_rel_err=state_max, k_best_rel_err=k_err,
            f_perp_combined_analytic=m_an["f_perp"], f_perp_combined_fd=m_fd["f_perp"],
            f_perp_materiality=abs(m_fd["f_perp"] - m_an["f_perp"]),
            delta_f_perp_analytic=m_an["f_perp"] - m_r["f_perp"],
            sigma_k_frac_analytic=m_an["sigma_k_frac"], sigma_k_frac_fd=m_fd["sigma_k_frac"],
            state_gate=state_max <= 1e-4, k_gate=k_err <= 1e-3,
            material=abs(m_fd["f_perp"] - m_an["f_perp"]) > 1e-3)
        print("\n%s  %s %s  rows=%d excluded=%d" % (key, scn.case_id, rows.pair, rows.n, bad.sum()))
        for n in names:
            c = cols[n]
            print("   %-6s best step %-8g rel err %.3e  floor %.3e  sweep %s"
                  % (n, c["best_step"], c["best_rel_err"], c["oracle_floor"],
                     np.array2string(np.array(c["rel_err"]), precision=2)))
        print("   f_perp combined analytic %.6f  FD %.6f  |diff| %.2e   delta_f %.4f"
              % (m_an["f_perp"], m_fd["f_perp"], out[key]["f_perp_materiality"],
                 out[key]["delta_f_perp_analytic"]))

    # s13: formal quasar invariance of the full ΔDOR observable's derivative
    from lunar_od.delta_dor import QuasarDirection, quasar_differential_delay_s
    from lunar_od.radiometrics import _interp_state
    scn, rb, rows = context("FD1")
    h = pos[2]
    dp, dm = ds[("FD1", 0, h)], ds[("FD1", 0, -h)]
    fd_ds = (dp - dm) / (2 * h)
    inv = []
    for ra, dec in ((2.1, -0.3), (0.3, 0.9), (4.5, 0.1)):
        q = QuasarDirection(ra, dec)
        dq = np.array([quasar_differential_delay_s(_interp_state(scn.t_grid, rows.hist_a, t)[:3],
                                                   _interp_state(scn.t_grid, rows.hist_b, t)[:3], q)
                       for t in rows.t_used])
        fd_ddor = ((dp - dq) - (dm - dq)) / (2 * h)
        inv.append(float(np.nanmax(np.abs(fd_ddor - fd_ds)) / np.nanmax(np.abs(fd_ds))))
    out["quasar_invariance"] = dict(max_rel_diff_per_quasar=inv, step_m=h,
                                    invariant=bool(max(inv) < 1e-9))
    print("\nquasar invariance of dDDOR/dx0: max rel diff per quasar %s" % inv)
    state_ok = all(out[k]["state_gate"] for k in keys)
    k_ok = all(out[k]["k_gate"] for k in keys)
    material = any(out[k]["material"] for k in keys)
    out["gates"] = dict(
        DDOR_STATE_JACOBIAN_FD_GATE="PASS" if state_ok else ("FAIL_NOT_MATERIAL" if not material else "FAIL"),
        DDOR_K_JACOBIAN_FD_GATE="PASS" if k_ok else ("FAIL_NOT_MATERIAL" if not material else "FAIL"),
        MAX_DDOR_STATE_FD_RELERR=max(out[k]["max_state_best_rel_err"] for k in keys),
        MAX_DDOR_K_FD_RELERR=max(out[k]["k_best_rel_err"] for k in keys),
        MAX_F_PERP_MATERIALITY=max(out[k]["f_perp_materiality"] for k in keys))
    print(out["gates"])
    (ARTIFACTS / "phase17_r1od2_fd.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if "--k-refine" in sys.argv:
        k_refinement()
    else:
        main()
