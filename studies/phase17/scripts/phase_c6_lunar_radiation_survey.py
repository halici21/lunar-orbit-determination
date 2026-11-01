"""Phase C6: lunar albedo + thermal-IR radiation pressure, linear effect-size survey on the LRO arc 2009-260..263.

Predeclaration: studies/phase17/configs/phase_c6_lunar_radiation_survey_predeclaration.json.  Uses F3's cached
reference (phase_f3_lro_dmc_smoother.py --reference: F2p's parameters, Phi and the K / empirical columns on the
20 s grid) and F3's local measurement rows; no new fits.  Runs in minutes.
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
import phase_c6_lunar_radiation as L  # noqa: E402
import phase_c6_forced_columns as FC  # noqa: E402

PRE = "studies/phase17/configs/phase_c6_lunar_radiation_survey_predeclaration.json"
OUT = HERE.parent / "results" / "phase_c6_lunar_radiation_survey.json"
#: A3: the first run used quadrature_column (inverts F3's FD Phi) and failed G_QUAD; this run integrates the
#: forced variational equations directly (phase_c6_forced_columns.py)
METHOD = "forced_variational_rk4_exact_forcing_A4"
F3_ET0_UTC = "2009-09-16T00:00:00"          # F2p's et0 (checked against LOLA in phase_f3c_short_arc_batch.py)
K_LRP = 0.010                               # m^2/kg, F2H_PRIORS' physical K_SRP mean
CHI2_F3_SIGMA0 = 69899.78968941572          # F3 (c15d5c8)
CHI2_SIGMA_STAR = 3642.6118945392136
SCALES = np.array([1.0] * 3 + [1e-3] * 3 + D.CONST_SCALES)
NAMES = ["x", "y", "z", "vx", "vy", "vz", "K", "rb103", "rb105", "rb126", "dop_bias", "C_T", "S_T", "C_N", "S_N"]


def sun_positions(et: np.ndarray) -> np.ndarray:
    import spiceypy as spice
    return np.array([spice.spkpos("SUN", float(e), "J2000", "NONE", "MOON")[0] for e in et]) * 1e3


def quadrature_column(grid: np.ndarray, phi: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """S(t) = Phi(t) int_{grid[0]}^{t} Phi(tau)^-1 [0; a(tau)] dtau, trapezoid on the grid; (k, 6)."""
    f = np.einsum("kij,kj->ki", np.linalg.inv(phi)[:, :, 3:], acc)
    integral = np.zeros_like(f)
    integral[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid)[:, None], axis=0)
    return np.einsum("kij,kj->ki", phi, integral)


def ric_split(acc: np.ndarray, states: np.ndarray) -> np.ndarray:
    r_hat = states[:, :3] / np.linalg.norm(states[:, :3], axis=1)[:, None]
    c_hat = np.cross(states[:, :3], states[:, 3:])
    c_hat /= np.linalg.norm(c_hat, axis=1)[:, None]
    a_hat = np.cross(c_hat, r_hat)
    return np.stack([np.einsum("ki,ki->k", acc, e) for e in (r_hat, a_hat, c_hat)], axis=1)


def solve(h, r, pm, pc):
    """Prior-weighted linear least squares, column-scaled (the raw normal matrix has condition ~1e13)."""
    sc = SCALES if h.shape[1] == 15 else np.append(SCALES, 1.0)
    hs = h * sc
    info = np.linalg.inv(pc) * np.outer(sc, sc)
    a = hs.T @ hs + info
    b = hs.T @ r + info @ (pm / sc)
    cov = np.linalg.inv(a)
    return sc * (cov @ b), cov * np.outer(sc, sc)


def score(name, x, cov, h, r, f, p):
    post = r - h @ x
    is263, kind = f["is263"], f["kind"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st = D._resid_stats(post, kind, is263)
    chi2 = float(np.sum(post[is263] ** 2))
    sig = np.sqrt(np.diag(cov))
    out = dict(chi2_263=chi2, chi2_fit=float(np.sum(post[~is263] ** 2)), residuals=st,
               share=(CHI2_F3_SIGMA0 - chi2) / (CHI2_F3_SIGMA0 - CHI2_SIGMA_STAR),
               k_srp=float(p[6] + x[6]), k_srp_sigma=float(sig[6]),
               C_T=float(p[11] + x[11]), C_T_sigma=float(sig[11]), S_T=float(p[12] + x[12]),
               S_T_sigma=float(sig[12]), epoch_shift_m=float(np.linalg.norm(x[:3])))
    if x.size == 16:
        out.update(lrp_scale=float(x[15]), lrp_scale_sigma=float(sig[15]),
                   corr_scale_k=float(cov[15, 6] / np.sqrt(cov[15, 15] * cov[6, 6])))
    print(f"{name:4s} chi2_263 {chi2:10.1f} share {out['share']:6.3f} K {out['k_srp']:.4f} "
          f"C_T {out['C_T']:.2e} S_T {out['S_T']:.2e} 263 range {st['263']['range_rms_m']:.2f} m Doppler "
          f"{st['263']['doppler_rms_mmps']:.2f} mm/s; fit {st['fit']['range_rms_m']:.3f} m "
          f"{st['fit']['doppler_rms_mmps']:.3f} mm/s" + (f" scale {x[15]:.3f}+-{sig[15]:.3f}" if x.size == 16 else ""))
    return out


def main() -> None:
    import spiceypy as spice
    from lunar_od.project_paths import project_root
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    from lunar_od.project_paths import gravity_data_root
    from lunar_od.srp import SRPOptions, srp_acceleration_kernel
    t_start = time.time()
    M.F.load_kernels(project_root() / "lro_data", "DE440")
    z = D.load()
    f = D.prepare(z)
    ref = f["ref"]
    grid, states, p = z["grid"], z["hist"][:, :6], z["p"]
    et0 = float(spice.str2et(F3_ET0_UTC))
    lola0 = np.asarray(spice.spkezr(M.F.LRO, et0 + float(z["t_cmp_fit"][0]), "J2000", "NONE", "MOON")[0]) * 1e3
    assert np.array_equal(lola0, z["lola_cmp"][0]), "F3_ET0_UTC is not the cached problem's et0"
    assert abs(grid[0] - float(z["t_epoch"])) == 0.0
    sun = sun_positions(et0 + grid)
    t_fit_end = float(z["t_cmp_fit"][-1])
    fit = grid <= t_fit_end

    # G_QUAD: the column method reproduces the cached K and C_T columns
    srp = SRPOptions(k_srp_m2_per_kg=1.0)
    a_srp = np.array([srp_acceleration_kernel(s[:3], q, srp) for s, q in zip(states, sun)])
    a_ct = np.array([M.empirical_acceleration(s, [1.0, 0.0, 0.0, 0.0]) for s in states])
    models = {"nominal": L.LunarRadiation(), "albedo_0.19": L.LunarRadiation(albedo=0.19)}
    acc = {name: L.acceleration_kernel(states[:, :3], sun, model) for name, model in models.items()}
    earth = np.array([spice.spkpos("EARTH", float(e), "J2000", "NONE", "MOON")[0] for e in et0 + grid]) * 1e3
    gravity = load_lunar_gravity_model(gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab", nmax=300, mmax=300)
    interp = lambda tab: (lambda t: np.array([np.interp(t, grid, tab[:, i]) for i in range(3)]))  # noqa: E731
    gradient = FC.lunar_gradient(et0, gravity, float(spice.bodvrd("399", "GM", 1)[1][0]) * 1e9,
                                 float(spice.bodvrd("10", "GM", 1)[1][0]) * 1e9, interp(earth), interp(sun))
    # A4: every forcing evaluated at the RK4 sub-step nodes (run 2 interpolated the shadow transitions linearly)
    sub, r_sub = FC.sub_nodes(grid, states)
    sun_sub = np.array([np.interp(sub, grid, sun[:, i]) for i in range(3)]).T
    # empirical acceleration needs the velocity: Hermite derivative at the nodes
    from scipy.interpolate import CubicHermiteSpline
    v_sub = CubicHermiteSpline(grid, states[:, :3], states[:, 3:6], axis=0).derivative()(sub)
    t_col = time.time()
    f_sub = np.stack([np.array([srp_acceleration_kernel(r, q, srp) for r, q in zip(r_sub, sun_sub)]),
                      np.array([M.empirical_acceleration(np.concatenate([r, v]), [1.0, 0.0, 0.0, 0.0])
                                for r, v in zip(r_sub, v_sub)]),
                      L.acceleration_kernel(r_sub, sun_sub, models["nominal"]),
                      L.acceleration_kernel(r_sub, sun_sub, models["albedo_0.19"])], axis=2)
    print(f"forcings at {sub.size} sub-step nodes in {time.time() - t_col:.0f} s", flush=True)
    g_cache = project_root() / "lro_data" / "f3_cache" / "c6_gradients.npy"
    g_nodes = np.load(g_cache) if g_cache.exists() else np.array([gradient(t, r) for t, r in zip(sub, r_sub)])
    if not g_cache.exists():
        np.save(g_cache, g_nodes)
    xcols = FC.forced_columns(grid, states, f_sub, gradient, g_nodes=g_nodes)
    print(f"forced columns in {time.time() - t_col:.0f} s", flush=True)
    gq = {}
    for name, j, col in (("K", 0, ref.s[:, :, 0]), ("C_T", 1, ref.s[:, :, 5])):
        s_q = xcols[:, :, j]
        diff = float(np.max(np.linalg.norm(s_q[fit, :3] - col[fit, :3], axis=1)))
        size = float(np.max(np.linalg.norm(col[fit, :3], axis=1)))
        diff_all = float(np.max(np.linalg.norm(s_q[:, :3] - col[:, :3], axis=1)))
        gq[name] = dict(max_position_difference=diff, column_max_position=size, relative=diff / size,
                        relative_incl_263=diff_all / float(np.max(np.linalg.norm(col[:, :3], axis=1))),
                        PASS=diff <= 1e-3 * size)
    print("G_QUAD", json.dumps(gq), flush=True)

    cols = {"nominal": xcols[:, :, 2], "albedo_0.19": xcols[:, :, 3]}
    lit = np.linalg.norm(a_srp, axis=1) > 0.0
    a_nom = K_LRP * acc["nominal"]
    ric = ric_split(a_nom, states)
    accel = dict(
        lrp_rms_mps2=float(np.sqrt(np.mean(np.sum(a_nom ** 2, axis=1)))),
        lrp_max_mps2=float(np.max(np.linalg.norm(a_nom, axis=1))),
        lrp_ric_rms_mps2=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(),
        lrp_ric_mean_mps2=np.mean(ric, axis=0).tolist(),
        srp_rms_sunlit_mps2=float(np.sqrt(np.mean(np.sum((K_LRP * a_srp[lit]) ** 2, axis=1)))),
        sunlit_fraction=float(np.mean(lit)),
        ratio_lrp_to_srp_rms_sunlit=float(np.sqrt(np.mean(np.sum(a_nom[lit] ** 2, axis=1)))
                                          / np.sqrt(np.mean(np.sum((K_LRP * a_srp[lit]) ** 2, axis=1)))))
    size = float(np.max(np.linalg.norm(K_LRP * cols["nominal"][:, :3], axis=1)))
    g_size = dict(max_position_m=size, PASS=size <= 200.0)
    print("accel", json.dumps(accel), "G_SIZE", json.dumps(g_size))

    # measurement columns
    hx, h, r, is263 = f["hx"], f["h"], f["r"], f["is263"]
    t_row = z["t_row"]
    sel = ~is263
    h_lrp = {}
    for name, col in cols.items():
        _, s_r = C.Reference(grid, ref.phi, col[:, :, None]).at(t_row)
        h_lrp[name] = np.einsum("ni,ni->n", hx, s_r[:, :, 0])
    pm, pc = f["pm"], f["pc"]
    exps = {}
    x0, c0 = solve(h[sel], r[sel], pm, pc)
    exps["E0"] = score("E0", x0, c0, h, r, f, p)
    g_e0 = dict(chi2=exps["E0"]["chi2_263"], f3=CHI2_F3_SIGMA0,
                relative=abs(exps["E0"]["chi2_263"] / CHI2_F3_SIGMA0 - 1.0))
    g_e0["PASS"] = g_e0["relative"] <= 1e-6
    for tag, name in (("E1", "nominal"), ("E1b", "albedo_0.19")):
        r1 = r - K_LRP * h_lrp[name]
        x1, c1 = solve(h[sel], r1[sel], pm, pc)
        exps[tag] = score(tag, x1, c1, h, r1, f, p)
    h2 = np.hstack([h, K_LRP * h_lrp["nominal"][:, None]])
    pm2, pc2 = np.append(pm, 0.0), np.zeros((16, 16))
    pc2[:15, :15] = pc
    pc2[15, 15] = 1e6 ** 2                                   # "no prior"
    x2, c2 = solve(h2[sel], r[sel], pm2, pc2)
    exps["E2"] = score("E2", x2, c2, h2, r, f, p)

    gates = dict(G_UNIT="see tests/test_phase_c6_lunar_radiation.py (run before this script)",
                 G_QUAD=all(v["PASS"] for v in gq.values()), G_E0=g_e0["PASS"], G_SIZE=g_size["PASS"])
    dk = exps["E0"]["k_srp"] - exps["E1"]["k_srp"]
    e2 = exps["E2"]
    h_k = dk >= 0.03
    h_pred = exps["E1"]["share"] >= 0.3
    h_scale = (abs(e2["lrp_scale"]) > 3 * e2["lrp_scale_sigma"]
               and abs(e2["lrp_scale"] - 1.0) <= 3 * e2["lrp_scale_sigma"])
    if not all(v is True for k, v in gates.items() if k != "G_UNIT"):
        decision = "C6_NOT_QUALIFIED"
    else:
        decision = "C6_LRP_MATERIAL" if (h_k or h_pred) else "C6_LRP_IMMATERIAL"
    result = dict(predeclaration=PRE, method=METHOD, gates=gates, G_QUAD=gq, G_E0=g_e0, G_SIZE=g_size,
                  hypotheses_as_computed=dict(H_K=h_k, H_PRED=h_pred, H_SCALE=h_scale), decision=decision,
                  k_drop_E1=dk, k_lrp_nominal=K_LRP, accelerations=accel, experiments=exps,
                  lrp_column_max_position_m={k: float(np.max(np.linalg.norm(K_LRP * v[:, :3], axis=1)))
                                             for k, v in cols.items()},
                  wall_s=time.time() - t_start)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gates", "hypotheses_as_computed", "decision", "k_drop_E1")},
                     indent=1))


if __name__ == "__main__":
    main()
