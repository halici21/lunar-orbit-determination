"""PHASE 17-HARDEN s39-s41 - root cause of the two-way range Jacobian discrepancy.

ANALYSIS SPACE ONLY.

GEO measured a tolerance-independent analytic-vs-FD difference of ~1e-6..3e-5
relative in the production two-way range state Jacobian.  This script splits
the chain  x0 -> trajectory -> event solution -> range  into separately
testable links, on ONE frozen observation set (receive tags and station
providers fixed; perturbed trajectories only re-solve the events):

  LIN   linearized-trajectory oracle: states_i + Phi(t_i) dx0 (position AND
        velocity rows) on the grid, then the production event solver.  Isolates
        the event derivative + Hermite interpolation from the STM itself.
  PROP  full independent re-propagation (6-state `propagate_state`, same force
        model/tolerances) + the production event solver.
  STM   direct STM check: (x(t; x0+h e_j) - x(t; x0-h e_j)) / 2h vs Phi(t) e_j
        on the grid, i.e. the variational equations themselves.

Each is run for three force variants of the SAME orbit:
  CONICAL_PENUMBRA SRP (the science model), NO_SHADOW SRP, and K = 0 (no SRP).
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parents[2] / "studies" / "phase17" / "results"
VARIANTS = (("SRP_CONICAL_PENUMBRA", 0.01, "CONICAL_PENUMBRA"),
            ("SRP_NO_SHADOW", 0.01, "NO_SHADOW"),
            ("NO_SRP_K0", 0.0, "CONICAL_PENUMBRA"))
STEPS = {"pos": (1.0, 10.0), "vel": (1e-3, 1e-2)}


def _init() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    warnings.simplefilter("ignore")
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()


def _scenario(k, shadow, orbits):
    import phase17_r1od2_core as D2
    import phase17_r1m_core as M
    et0, t_orbit = M.campaign_epoch()
    return D2.Scenario("CANON_%g_%s" % (k, shadow), "harden", et0,
                       tuple(M.campaign_initial_state().tolist()), orbits * t_orbit, k, shadow)


def _prop6(scn, x0, rtol, atol):
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions
    t = scn.t_grid
    return propagate_state(t, x0, G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]),
                           lambda tt: G.sun_mci(scn.et0 + float(tt)),
                           srp=SRPOptions(k_srp_m2_per_kg=scn.k_srp, shadow_model=scn.shadow_model),
                           rtol=rtol, atol=atol, j2_moon=J2_MOON_UNNORMALIZED)


def run_variant(args) -> dict:
    name, k, shadow, orbits, rtol = args
    import phase17_geo_core as G
    import phase17_r1od2_core as D2
    from lunar_od.two_way_range import two_way_range_nominal_and_initial_jacobian

    atol = rtol * 0.1
    scn = _scenario(k, shadow, orbits)
    nom = G.propagate48(scn.t_grid, scn.x0_array, scn.et0, k, rtol=rtol, atol=atol, shadow_model=shadow)
    rb = D2.build_range_block(scn, nom)
    phi = np.array([row.reshape((6, 6), order="F") for row in nom[:, 6:42]])

    def z(hist48):
        zz, _ = two_way_range_nominal_and_initial_jacobian(rb.obs, rb.pass_geo, hist48[:, :42])
        return np.asarray(zz, float).reshape(-1)

    out = dict(variant=name, k=k, shadow=shadow, orbits=orbits, rtol=rtol, n_obs=rb.n_obs, columns={})
    for j in range(6):
        steps = STEPS["pos"] if j < 3 else STEPS["vel"]
        col = dict(analytic_norm=float(np.max(np.abs(rb.h_x0[:, j]))))
        for h in steps:
            e = np.zeros(6)
            e[j] = h
            # LIN: linearized trajectory, full state rows perturbed by Phi e
            lin_p, lin_m = nom.copy(), nom.copy()
            lin_p[:, :6] += phi @ e
            lin_m[:, :6] -= phi @ e
            fd_lin = (z(lin_p) - z(lin_m)) / (2 * h)
            # PROP: independent re-propagation
            xp = _prop6(scn, scn.x0_array + e, rtol, atol)
            xm = _prop6(scn, scn.x0_array - e, rtol, atol)
            hp, hm = nom.copy(), nom.copy()
            hp[:, :6], hm[:, :6] = xp, xm
            fd_prop = (z(hp) - z(hm)) / (2 * h)
            # STM: variational equations vs re-propagation, at the final grid point
            stm_fd = (xp - xm) / (2 * h)
            stm_an = phi @ (e / h)
            stm_err = float(np.max(np.abs(stm_fd[:, :3] - stm_an[:, :3])) / np.max(np.abs(stm_an[:, :3])))
            an = rb.h_x0[:, j]
            col["h=%g" % h] = dict(
                lin_rel=float(np.max(np.abs(fd_lin - an)) / np.max(np.abs(an))),
                prop_rel=float(np.max(np.abs(fd_prop - an)) / np.max(np.abs(an))),
                prop_vs_lin_rel=float(np.max(np.abs(fd_prop - fd_lin)) / np.max(np.abs(an))),
                stm_pos_rel=stm_err)
        out["columns"]["x0 y0 z0 vx0 vy0 vz0".split()[j]] = col
    return out


def main() -> None:
    jobs = [(n, k, s, 4.0, rtol) for n, k, s in VARIANTS for rtol in (1e-12, 1e-13)]
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init) as ex:
        res = list(ex.map(run_variant, jobs))
    for r in res:
        print("\n%-22s rtol %g  n_obs %d" % (r["variant"], r["rtol"], r["n_obs"]))
        for c, v in r["columns"].items():
            s = "  ".join("%s: LIN %.1e PROP %.1e STM %.1e" % (h, x["lin_rel"], x["prop_rel"], x["stm_pos_rel"])
                          for h, x in v.items() if h.startswith("h="))
            print("   %-4s %s" % (c, s))
    (ARTIFACTS / "phase17_harden_range_jacobian_decomposition.json").write_text(json.dumps(res, indent=1))


def predict_omitted_srp_gradient(orbits: float = 4.0, dt: float = 1.0, rtol: float = 1e-13,
                                 h_pos: float = 10.0, h_vel: float = 0.01,
                                 shadow: str = "CONICAL_PENUMBRA") -> dict:
    """First-order prediction of the STM error caused by omitting d(a_SRP)/dr
    from the variational matrix (the documented Phase 13/16 approximation):

        dPhi(t) = Phi(t,0) * integral_0^t Phi(tau,0)^-1 B(tau) Phi(tau,0) dtau,
        B = [[0, 0], [G_SRP, 0]],  G_SRP = K * d g_SRP / dr  (production kernel, FD 0.5 m)

    compared with the OBSERVED STM error Phi_FD - Phi from independent
    re-propagation (h = 10 m / 0.01 m/s) on the 90 s grid.
    """
    import phase17_geo_core as G
    from lunar_od.srp import SRPOptions, srp_acceleration_kernel

    _init()
    scn = _scenario(0.01, shadow, orbits)
    t_f = np.arange(0.0, scn.t_grid[-1] + 0.5 * dt, dt)
    nom = G.propagate48(t_f, scn.x0_array, scn.et0, 0.01, rtol=rtol, atol=rtol * 0.1,
                        shadow_model=shadow)
    phi = np.array([row.reshape((6, 6), order="F") for row in nom[:, 6:42]])
    opts = SRPOptions(k_srp_m2_per_kg=1.0, shadow_model=shadow)
    integrand = np.zeros((t_f.size, 6, 6))
    g_norm = np.zeros(t_f.size)
    for i, t in enumerate(t_f):
        r = nom[i, :3]
        sun = G.sun_mci(scn.et0 + float(t))
        gmat = np.zeros((3, 3))
        for a in range(3):
            d = np.zeros(3)
            d[a] = 0.5
            gmat[:, a] = 0.01 * (srp_acceleration_kernel(r + d, sun, opts)
                                 - srp_acceleration_kernel(r - d, sun, opts)) / 1.0
        g_norm[i] = np.linalg.norm(gmat)
        b = np.zeros((6, 6))
        b[3:, :3] = gmat
        integrand[i] = np.linalg.solve(phi[i], b @ phi[i])
    cum = np.concatenate([np.zeros((1, 6, 6)),
                          np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dt, axis=0)])
    dphi_pred = np.einsum("nij,njk->nik", phi, cum)
    # observed on the 90 s grid
    idx = np.searchsorted(t_f, scn.t_grid)
    out = dict(orbits=orbits, dt=dt, rtol=rtol, h_pos=h_pos, h_vel=h_vel, shadow=shadow,
               g_srp_norm_max=float(g_norm.max()),
               g_srp_norm_median=float(np.median(g_norm)), columns={})
    for j in range(6):
        h = h_pos if j < 3 else h_vel
        e = np.zeros(6)
        e[j] = h
        xp = _prop6(scn, scn.x0_array + e, rtol, rtol * 0.1)
        xm = _prop6(scn, scn.x0_array - e, rtol, rtol * 0.1)
        obs_err = (xp - xm)[:, :3] / (2 * h) - phi[idx][:, :3, j]
        pred = dphi_pred[idx][:, :3, j]
        norm = np.max(np.abs(phi[idx][:, :3, j]))
        resid = obs_err - pred
        out["columns"]["x0 y0 z0 vx0 vy0 vz0".split()[j]] = dict(
            observed_rel=float(np.max(np.abs(obs_err)) / norm),
            predicted_rel=float(np.max(np.abs(pred)) / norm),
            residual_after_prediction_rel=float(np.max(np.abs(resid)) / norm),
            correlation=float(np.corrcoef(obs_err.ravel(), pred.ravel())[0, 1]))
    return out


def _predict_job(cfg):
    rtol, hp, hv, shadow = cfg
    return predict_omitted_srp_gradient(rtol=rtol, h_pos=hp, h_vel=hv, shadow=shadow)


if __name__ == "__main__":
    if "--predict" in sys.argv:
        cfgs = [(1e-13, 10.0, 0.01, "CONICAL_PENUMBRA"), (1e-13, 100.0, 0.1, "CONICAL_PENUMBRA"),
                (1e-14, 10.0, 0.01, "CONICAL_PENUMBRA"), (1e-14, 100.0, 0.1, "CONICAL_PENUMBRA"),
                (1e-14, 100.0, 0.1, "NO_SHADOW")]
        with ProcessPoolExecutor(max_workers=len(cfgs), initializer=_init) as ex:
            res = list(ex.map(_predict_job, cfgs))
        for r in res:
            print("rtol %g h %g/%g %-16s gmax %.2e" % (r["rtol"], r["h_pos"], r["h_vel"], r["shadow"],
                                                      r["g_srp_norm_max"]))
            for c, v in r["columns"].items():
                print("   %-4s observed %.2e predicted %.2e residual %.2e corr %+.3f"
                      % (c, v["observed_rel"], v["predicted_rel"], v["residual_after_prediction_rel"],
                         v["correlation"]))
        (ARTIFACTS / "phase17_harden_range_jacobian_srp_gradient_prediction.json").write_text(
            json.dumps(res, indent=1))
    else:
        main()
