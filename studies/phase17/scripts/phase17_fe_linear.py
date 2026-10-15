"""Phase 17-FE-L: linear study of a low-degree gravity-field-error parameterisation alongside M3.

Frozen in artifacts/phase17_fe_predeclaration.json (written before any number here).

  --run [--workers N]    the 10 KP2 cells, HIGH_FIDELITY: FE columns (degrees 2..6), the MM mismatch signature,
                         metrics for N in {0, 2, 4, 6}; one JSON per cell in artifacts/phase17_fe_cells/
  --v1                   gate V1: convolution vs propagated finite difference for C22 and C31 (CANON k0, HF)
  --analyse              gates + predeclared decision -> artifacts/phase17_fe_summary.json
"""
from __future__ import annotations

import argparse
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
warnings.simplefilter("ignore")

import phase17_kgm_core as G  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase17_mm_experiment as MM  # noqa: E402

K = P.K
ART = K2.ARTIFACTS
CELLS_DIR = ART / "phase17_fe_cells"
N_FE = (2, 4, 6)
ALPHAS = (0.3, 1.0)
HF = "HIGH_FIDELITY"
CHI2 = K2.CHI2_2_999


def coefficient_list(nmax: int) -> list[tuple[str, int, int]]:
    out = []
    for n in range(2, nmax + 1):
        out += [("C", n, m) for m in range(0, n + 1)]
        out += [("S", n, m) for m in range(1, n + 1)]
    return out


def unit_model(gl, kind: str, n: int, m: int, nmax: int):
    """GL1800F-normalised model whose only nonzero coefficient is (kind, n, m) = 1; evaluated to degree nmax."""
    cbar = np.zeros((nmax + 1, nmax + 1))
    sbar = np.zeros((nmax + 1, nmax + 1))
    (cbar if kind == "C" else sbar)[n, m] = 1.0
    return dataclasses.replace(gl, cbar=cbar, sbar=sbar, nmax=nmax, mmax=nmax)


def _rotations(scn, t):
    import spiceypy as spice
    import od_gravity_covariance_campaign as C  # noqa: N811
    return [np.asarray(spice.pxform("J2000", C.FRAME, scn.et0 + float(ti))) for ti in t]


def _metrics(an, cs, h_mm, n_fe_cols):
    """Linear alias / information metrics on a column-normalised whitened design (prior rows included)."""
    r = np.linalg.qr(an, mode="r")
    rinv = np.linalg.solve(r, np.eye(r.shape[0]))
    cov = (rinv @ rinv.T) / np.outer(cs, cs)
    cf = cov[6:9, 6:9] / K2.K0 ** 2
    cv = K2.Q_VAR.T @ cf @ K2.Q_VAR
    sd_v = np.sqrt(np.diag(cv))
    one = np.ones(3) / 3
    out = dict(sigma_v=sd_v.tolist(), n_fe=n_fe_cols, cond_equilibrated=float(np.linalg.cond(an)))
    for name, f in (("T1_STEP", K2.TRUTHS["T1_STEP"]), ("T2_MIDDLE", K2.TRUTHS["T2_MIDDLE"])):
        v = K2.Q_VAR.T @ np.asarray(f)
        out[f"noncentrality_{name}"] = float(v @ np.linalg.solve(cv, v))
    for a in ALPHAS:
        sol = np.linalg.lstsq(an, a * h_mm, rcond=None)[0]
        th = sol / cs
        df = th[6:9] / K2.K0
        v = K2.Q_VAR.T @ df
        q, _ = np.linalg.qr(an)
        res = a * h_mm - q @ (q.T @ (a * h_mm))
        out[f"alpha{a}"] = dict(fv_chi2=float(v @ np.linalg.solve(cv, v)), resid_chi2=float(res @ res),
                                variation_bias=v.tolist(),
                                level_bias_over_sigma=float(one @ df / math.sqrt(one @ cf @ one)),
                                state_bias_pos_m=float(np.linalg.norm(th[:3])))
    return out


def cell(spec):
    t0 = time.time()
    K._init()
    orbit, k = spec["orbit"], spec["k"]
    path = CELLS_DIR / f"{orbit}_k{k}.json"
    if path.exists():
        return orbit, k, "cached"
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), HF)
    t = scn.t_grid
    rot = _rotations(scn, t)
    gl, gr = MM._fields()
    mix1 = MM.mixed_field(1.0, gl, gr)
    a_mm = np.stack([sha(X[i, :3], mix1, rot[i]) - sha(X[i, :3], gl, rot[i]) for i in range(t.size)])
    coeffs = coefficient_list(max(N_FE))
    units = [unit_model(gl, *c, max(N_FE)) for c in coeffs]
    a_fe = np.stack([[sha(X[i, :3], u, rot[i]) for i in range(t.size)] for u in units])   # (P, N, 3)
    s_mm = MM_convolve(t, PHI, a_mm)
    s_fe = np.stack([MM_convolve(t, PHI, a) for a in a_fe])
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, K.stations())
    ev = G.evaluate(fx, X, PHI, np.concatenate([S, s_mm[None], s_fe], axis=0))
    wr, wd = 1 / K2.SIGMA_RANGE_M, 1 / K2.SIGMA_DOPPLER_MPS
    q = fx.nuis.shape[1]
    hx = np.vstack([ev["hx_r"] * wr, ev["hx_d"] * wd])
    hk = np.vstack([ev["hk_r"] * wr, ev["hk_d"] * wd])            # columns: 3 K segments, mismatch, FE ...
    nu = np.vstack([np.zeros((ev["hx_r"].shape[0], q)), fx.nuis * wd])
    h_mm = hk[:, 3]
    h_fe = hk[:, 4:]
    prior = np.zeros((6, 6 + 3 + q))
    prior[:, :6] = np.diag(1 / K2.PRIOR_SD)
    pad = lambda v: np.concatenate([v, np.zeros(6)])  # noqa: E731
    h_mm_p = pad(h_mm)
    res = dict(orbit=orbit, k=k, n_rows=int(hx.shape[0]), n_passes=int(q), coefficients=[list(c) for c in coeffs],
               mean_accel_mm_over_srp=None, per_N={})
    for n_deg in (0,) + N_FE:
        sel = [j for j, c in enumerate(coeffs) if c[1] <= n_deg]
        blocks = [hx, hk[:, :3], nu] + ([h_fe[:, sel]] if sel else [])
        a = np.hstack(blocks)
        pr = np.zeros((6, a.shape[1]))
        pr[:, :6] = np.diag(1 / K2.PRIOR_SD)
        a = np.vstack([a, pr])
        cs = np.linalg.norm(a, axis=0)
        cs[cs == 0] = 1.0
        res["per_N"][str(n_deg)] = _metrics(a / cs, cs, h_mm_p, len(sel))
    res["elapsed_s"] = time.time() - t0
    path.write_text(json.dumps(K._clean(res), indent=1), encoding="utf-8")
    return orbit, k, "ok"


def MM_convolve(t, PHI, acc):
    """S(t) = Phi(t) int Phi^-1 [0; a] dtau, trapezoid on the grid (the MM-probe construction)."""
    integrand = np.stack([np.linalg.solve(PHI[i], np.concatenate([np.zeros(3), acc[i]])) for i in range(t.size)])
    psi = np.zeros_like(integrand)
    psi[1:] = np.cumsum(0.5 * np.diff(t)[:, None] * (integrand[1:] + integrand[:-1]), axis=0)
    return np.einsum("nij,nj->ni", PHI, psi)


def gate_v1() -> dict:
    """Convolution column vs propagated finite difference (delta = 1e-7) for C22 and C31, CANON k0, HF."""
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = K.scenario("CANON", x0.tolist(), 0)
    ks = np.full(3, K2.K0)
    X, PHI, _ = K2.propagate_segmented(scn, x0, ks, HF)
    t = scn.t_grid
    rot = _rotations(scn, t)
    gl, _ = MM._fields()
    out = {}
    for kind, n, m in (("C", 2, 2), ("C", 3, 1)):
        u = unit_model(gl, kind, n, m, 6)
        s = MM_convolve(t, PHI, np.stack([sha(X[i, :3], u, rot[i]) for i in range(t.size)]))
        delta = 1e-7
        cb = gl.cbar.copy()
        cb[n, m] += delta
        pert = dataclasses.replace(gl, cbar=cb)
        real = K2._prop_segment
        K2._prop_segment = MM._truth_segment(pert)
        try:
            Xp, _, _ = K2.propagate_segmented(scn, x0, ks, HF)
        finally:
            K2._prop_segment = real
        fd = (Xp[:, :3] - X[:, :3]) / delta
        out[f"{kind}{n}{m}"] = float(np.max(np.linalg.norm(s[:, :3] - fd, axis=1)) / np.max(np.linalg.norm(fd, axis=1)))
    out["gate_pass"] = max(v for v in out.values()) <= 2e-2
    return out


def gate_v3() -> dict:
    """Linearity: sum dC * da/dC over degrees 2..6 == a(GRGM-mix, truncated 6) - a(GL, truncated 6)."""
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    K._init()
    gl, gr = MM._fields()
    mix = MM.mixed_field(1.0, gl, gr)
    r, c = np.array([1.2e6, -1.0e6, 9.0e5]), np.eye(3)
    trunc = lambda mdl: dataclasses.replace(mdl, nmax=6, mmax=6)  # noqa: E731
    ref = sha(r, trunc(mix), c) - sha(r, trunc(gl), c)
    acc = np.zeros(3)
    for kind, n, m in coefficient_list(6):
        d = (mix.cbar if kind == "C" else mix.sbar)[n, m] - (gl.cbar if kind == "C" else gl.sbar)[n, m]
        acc += d * sha(r, unit_model(gl, kind, n, m, 6), c)
    rel = float(np.linalg.norm(acc - ref) / np.linalg.norm(ref))
    return dict(rel=rel, gate_pass=rel <= 1e-9)


def analyse() -> dict:
    cells = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CELLS_DIR.glob("*.json"))]
    decision = {}
    for n_deg in N_FE:
        sup = all(c["per_N"][str(n_deg)][f"alpha{a}"]["fv_chi2"] < CHI2 for c in cells for a in ALPHAS)
        pres = all(c["per_N"][str(n_deg)][f"noncentrality_{t}"] >= CHI2
                   for c in cells for t in ("T1_STEP", "T2_MIDDLE") if c["per_N"]["0"][f"noncentrality_{t}"] >= CHI2)
        decision[str(n_deg)] = dict(SUPPRESSES=sup, PRESERVES=pres)
    both = [n for n in N_FE if decision[str(n)]["SUPPRESSES"] and decision[str(n)]["PRESERVES"]]
    if both:
        outcome = "FE_RESOLVES_MASQUERADE"
    elif any(decision[str(n)]["SUPPRESSES"] for n in N_FE):
        outcome = "FE_SUPPRESSES_BUT_COSTS_VARIATION"
    else:
        outcome = "FE_DOES_NOT_SUPPRESS"
    table = []
    for c in cells:
        for n_deg in (0,) + N_FE:
            m = c["per_N"][str(n_deg)]
            table.append(dict(orbit=c["orbit"], k=c["k"], N=n_deg, n_fe=m["n_fe"],
                              fv03=m["alpha0.3"]["fv_chi2"], fv1=m["alpha1.0"]["fv_chi2"],
                              resid03=m["alpha0.3"]["resid_chi2"], resid1=m["alpha1.0"]["resid_chi2"],
                              nc_T1=m["noncentrality_T1_STEP"], nc_T2=m["noncentrality_T2_MIDDLE"],
                              sigma_v=m["sigma_v"], level03=m["alpha0.3"]["level_bias_over_sigma"],
                              cond=m["cond_equilibrated"]))
    gates = {}
    for name in ("v1", "v3"):
        p = ART / f"phase17_fe_gate_{name}.json"
        if p.exists():
            gates[name] = json.loads(p.read_text(encoding="utf-8"))
    return dict(outcome=outcome, smallest_resolving_N=(min(both) if both else None), decision=decision,
                gates=gates, table=table, n_cells=len(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--v1", action="store_true")
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.v3:
        out = gate_v3()
        (ART / "phase17_fe_gate_v3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(out)
    if a.v1:
        out = gate_v1()
        (ART / "phase17_fe_gate_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(out)
    if a.run:
        CELLS_DIR.mkdir(exist_ok=True)
        specs = [dict(orbit=o, k=k) for o in K.ORBITS for k in (0, 12)]
        with ProcessPoolExecutor(a.workers) as ex:
            for o, k, st in ex.map(cell, specs):
                print(o, k, st, flush=True)
    if a.analyse:
        out = analyse()
        (ART / "phase17_fe_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
