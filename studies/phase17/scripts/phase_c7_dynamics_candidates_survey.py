"""Phase C7: lunar tide / Moon 1PN / Venus-Jupiter, linear effect-size survey on the LRO arc 2009-260..263.

Predeclaration: studies/phase17/configs/phase_c7_dynamics_candidates_survey_predeclaration.json. Same column method
as C6 run 3 (forced variational columns, exact sub-step forcing, C6's cached N = 300 gradients).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_c6_lunar_radiation_survey as S6  # noqa: E402  (F3 / F2 setup, solve, score)
from phase_c6_lunar_radiation_survey import C, D, M  # noqa: E402
import phase_c6_forced_columns as FC  # noqa: E402
import phase_c7_dynamics_candidates as T  # noqa: E402

PRE = "studies/phase17/configs/phase_c7_dynamics_candidates_survey_predeclaration.json"
OUT = HERE.parent / "results" / "phase_c7_dynamics_candidates_survey.json"
CANDIDATES = ("TIDE", "PN", "PLANETS")


def body_positions(name: str, et: np.ndarray) -> np.ndarray:
    import spiceypy as spice
    return np.array([spice.spkpos(name, float(e), "J2000", "NONE", "MOON")[0] for e in et]) * 1e3


def main() -> None:
    import spiceypy as spice
    from lunar_od.project_paths import project_root
    t_start = time.time()
    M.F.load_kernels(project_root() / "lro_data", "DE440")
    z = D.load()
    f = D.prepare(z)
    ref = f["ref"]
    grid, states, p = z["grid"], z["hist"][:, :6], z["p"]
    et0 = float(spice.str2et(S6.F3_ET0_UTC))
    lola0 = np.asarray(spice.spkezr(M.F.LRO, et0 + float(z["t_cmp_fit"][0]), "J2000", "NONE", "MOON")[0]) * 1e3
    assert np.array_equal(lola0, z["lola_cmp"][0]), "F3_ET0_UTC is not the cached problem's et0"
    fit = grid <= float(z["t_cmp_fit"][-1])
    gm = {b: float(spice.bodvrd(c, "GM", 1)[1][0]) * 1e9 for b, c in
          (("EARTH", "399"), ("SUN", "10"), ("VENUS", "299"), ("JUPITER BARYCENTER", "5"))}
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    from lunar_od.project_paths import gravity_data_root
    mu_moon = float(load_lunar_gravity_model(gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab", nmax=2,
                                             mmax=2).mu_m3_s2)
    sub, r_sub = FC.sub_nodes(grid, states)
    v_sub = CubicHermiteSpline(grid, states[:, :3], states[:, 3:6], axis=0).derivative()(sub)
    pos = {}
    for b in gm:
        tab = body_positions(b, et0 + grid)
        pos[b] = np.array([np.interp(sub, grid, tab[:, i]) for i in range(3)]).T

    # forcings at the sub-step nodes: C_T (G_QUAD), tide, 1PN, planets
    tide = sum(T.tide_acceleration(r_sub, pos[b], gm[b], n, k)
               for b in ("EARTH", "SUN") for n, k in ((2, T.K2_GL1800F), (3, T.K3_GL1800F)))
    tide_parts = {f"{b}_k{n}": T.tide_acceleration(r_sub, pos[b], gm[b], n, k)
                  for b in ("EARTH", "SUN") for n, k in ((2, T.K2_GL1800F), (3, T.K3_GL1800F))}
    pn = T.pn_acceleration(r_sub, v_sub, mu_moon)
    planets = (T.third_body_acceleration(r_sub, pos["VENUS"], gm["VENUS"])
               + T.third_body_acceleration(r_sub, pos["JUPITER BARYCENTER"], gm["JUPITER BARYCENTER"]))
    a_ct = np.array([M.empirical_acceleration(np.concatenate([r, v]), [1.0, 0.0, 0.0, 0.0])
                     for r, v in zip(r_sub, v_sub)])
    forcing = np.stack([a_ct, tide, pn, planets], axis=2)
    g_nodes = np.load(project_root() / "lro_data" / "f3_cache" / "c6_gradients.npy")
    t_col = time.time()
    xcols = FC.forced_columns(grid, states, forcing, None, g_nodes=g_nodes)
    print(f"forced columns in {time.time() - t_col:.0f} s", flush=True)

    col = ref.s[:, :, 5]
    diff = float(np.max(np.linalg.norm(xcols[fit, :3, 0] - col[fit, :3], axis=1)))
    size = float(np.max(np.linalg.norm(col[fit, :3], axis=1)))
    gq = dict(C_T=dict(max_position_difference=diff, column_max_position=size, relative=diff / size,
                       PASS=diff <= 1e-3 * size))
    cols = {"TIDE": xcols[:, :, 1], "PN": xcols[:, :, 2], "PLANETS": xcols[:, :, 3]}
    acc = {"TIDE": tide, "PN": pn, "PLANETS": planets}
    states_sub = np.hstack([r_sub, v_sub])
    accel = {}
    for k, a in acc.items():
        ric = S6.ric_split(a, states_sub)
        accel[k] = dict(rms_mps2=float(np.sqrt(np.mean(np.sum(a ** 2, axis=1)))),
                        max_mps2=float(np.max(np.linalg.norm(a, axis=1))),
                        ric_rms_mps2=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(),
                        ric_mean_mps2=np.mean(ric, axis=0).tolist())
    accel["TIDE_parts_rms_mps2"] = {k: float(np.sqrt(np.mean(np.sum(v ** 2, axis=1)))) for k, v in tide_parts.items()}
    g_size = {k: dict(max_position_m=float(np.max(np.linalg.norm(v[:, :3], axis=1)))) for k, v in cols.items()}
    for v in g_size.values():
        v["PASS"] = v["max_position_m"] <= 200.0
    print("G_QUAD", json.dumps(gq), "\nG_SIZE", json.dumps(g_size), "\naccel", json.dumps(accel), flush=True)

    hx, h, r, is263 = f["hx"], f["h"], f["r"], f["is263"]
    t_row = z["t_row"]
    sel = ~is263
    h_c = {}
    for k, c in cols.items():
        _, s_r = C.Reference(grid, ref.phi, c[:, :, None]).at(t_row)
        h_c[k] = np.einsum("ni,ni->n", hx, s_r[:, :, 0])
    pm, pc = f["pm"], f["pc"]
    exps = {}
    x0, c0 = S6.solve(h[sel], r[sel], pm, pc)
    exps["E0"] = S6.score("E0", x0, c0, h, r, f, p)
    g_e0 = dict(chi2=exps["E0"]["chi2_263"], relative=abs(exps["E0"]["chi2_263"] / S6.CHI2_F3_SIGMA0 - 1.0))
    g_e0["PASS"] = g_e0["relative"] <= 1e-6
    for name, keys in (("E_TIDE", ["TIDE"]), ("E_PN", ["PN"]), ("E_PLANETS", ["PLANETS"]), ("E_ALL", list(cols))):
        r1 = r - sum(h_c[k] for k in keys)
        x1, c1 = S6.solve(h[sel], r1[sel], pm, pc)
        exps[name] = S6.score(name, x1, c1, h, r1, f, p)

    common = gq["C_T"]["PASS"] and g_e0["PASS"]
    hyp = dict(H_TIDE=exps["E_TIDE"]["share"] >= 0.3 or exps["E0"]["k_srp"] - exps["E_TIDE"]["k_srp"] >= 0.03,
               H_PN=abs(exps["E_PN"]["share"]) >= 0.05, H_PLANETS=abs(exps["E_PLANETS"]["share"]) >= 0.05)
    decisions = {}
    for k, hk in (("TIDE", "H_TIDE"), ("PN", "H_PN"), ("PLANETS", "H_PLANETS")):
        if not common:
            decisions[k] = "C7_NOT_QUALIFIED"
        elif not g_size[k]["PASS"]:
            decisions[k] = f"C7_{k}_NONLINEAR_NEEDED"
        else:
            decisions[k] = f"C7_{k}_MATERIAL" if hyp[hk] else f"C7_{k}_IMMATERIAL"
    result = dict(predeclaration=PRE, gates=dict(G_UNIT="tests/test_phase_c7_dynamics_candidates.py",
                                                 G_QUAD=gq["C_T"]["PASS"], G_E0=g_e0["PASS"],
                                                 G_SIZE={k: v["PASS"] for k, v in g_size.items()}),
                  G_QUAD=gq, G_E0=g_e0, G_SIZE=g_size, hypotheses_as_computed=hyp, decisions=decisions,
                  love_numbers=dict(k2=T.K2_GL1800F, k3=T.K3_GL1800F, source="jggrx_1800f_sha.lbl lines 31-34"),
                  gm_m3_s2=dict(gm, MOON=mu_moon), accelerations=accel, experiments=exps,
                  wall_s=time.time() - t_start)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gates", "hypotheses_as_computed", "decisions")}, indent=1))


if __name__ == "__main__":
    main()
