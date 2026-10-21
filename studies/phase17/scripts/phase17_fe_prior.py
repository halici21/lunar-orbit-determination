"""Phase 17-FE-P: external gravity prior (GL1800F per-coefficient uncertainties) on the FE-L field corrections.

Frozen in artifacts/phase17_fep_predeclaration.json (written before any FE-P output).

  --designs [--workers N]  per cell: the FE-L design blocks (HIGH_FIDELITY) + the MM mismatch column + 60 seeded
                           prior-consistent random-error columns; saved to <scratch>/phase17_fep_designs/*.npz
  --score                  gates P1-P5, all predeclared configurations, classification -> artifacts/phase17_fep_summary.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_fe_linear as FE  # noqa: E402
import phase17_kgm_core as G  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402
import phase17_mm_experiment as MM  # noqa: E402

K = P.K
ART = K2.ARTIFACTS
DESIGNS = Path(os.environ.get("FEP_DESIGNS", str(ART.parent / ".fep_designs")))
GL_TAB = Path("C:/Users/erayh/Documents/Python/lunar-orbit-determination/python_port/data/gravity/gl1800f/"
              "jggrx_1800f_sha.tab")
CHI2 = K2.CHI2_2_999
N_SET = (4, 6)
SCALES = (1.0, 10.0, 100.0)          # x1 primary; x10, x100 sensitivity
N_DRAWS = 20
DRAW_NMAX = 50
ALPHAS = FE.ALPHAS


# ----------------------------------------------------------------------
# prior: GL1800F uncertainty columns, parsed independently of the loader
# ----------------------------------------------------------------------
def parse_gl1800f(nmax: int) -> dict:
    """{(n, m): (C, S, sigma_C, sigma_S)} for n <= nmax, straight from the SHADR .tab lines."""
    out = {}
    with open(GL_TAB, encoding="ascii", errors="strict") as f:
        f.readline()                                           # header record
        for line in f:
            p = line.split(",")
            n, m = int(p[0]), int(p[1])
            if n > nmax:
                break
            out[(n, m)] = tuple(float(x) for x in p[2:6])
    return out


def sigma_vector(coeffs, table) -> np.ndarray:
    return np.array([table[(n, m)][2] if kind == "C" else table[(n, m)][3] for kind, n, m in coeffs])


def stress_vector(coeffs, gl, gr) -> np.ndarray:
    """|GRGM1200L (rescaled to GL mu/R) - GL1800F| per coefficient - the STRESS scale (not an uncertainty)."""
    mix = MM.mixed_field(1.0, gl, gr)
    return np.array([abs((mix.cbar if k == "C" else mix.sbar)[n, m] - (gl.cbar if k == "C" else gl.sbar)[n, m])
                     for k, n, m in coeffs])


# ----------------------------------------------------------------------
# designs
# ----------------------------------------------------------------------
def cell_design(spec):
    t0 = time.time()
    K._init()
    from lunar_od.gravity_harmonics import spherical_harmonic_acceleration as sha
    orbit, k, ci = spec["orbit"], spec["k"], spec["ci"]
    out_path = DESIGNS / f"{orbit}_k{k}.npz"
    if out_path.exists():
        return orbit, k, "cached"
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = K.scenario(orbit, x0.tolist(), k)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), FE.HF)
    t = scn.t_grid
    rot = FE._rotations(scn, t)
    gl, gr = MM._fields()
    mix1 = MM.mixed_field(1.0, gl, gr)
    a_mm = np.stack([sha(X[i, :3], mix1, rot[i]) - sha(X[i, :3], gl, rot[i]) for i in range(t.size)])
    coeffs = FE.coefficient_list(max(N_SET))
    units = [FE.unit_model(gl, *c, max(N_SET)) for c in coeffs]
    a_fe = np.stack([[sha(X[i, :3], u, rot[i]) for i in range(t.size)] for u in units])
    table = parse_gl1800f(DRAW_NMAX)
    draws = []
    for si, sc in enumerate(SCALES):
        for d in range(N_DRAWS):
            rng = np.random.default_rng([ci, si, d])
            cb = np.zeros((DRAW_NMAX + 1, DRAW_NMAX + 1))
            sb = np.zeros_like(cb)
            for n in range(2, DRAW_NMAX + 1):
                for m in range(0, n + 1):
                    cb[n, m] = rng.normal(0.0, sc * table[(n, m)][2])
                    if m > 0:
                        sb[n, m] = rng.normal(0.0, sc * table[(n, m)][3])
            mdl = dataclasses.replace(gl, cbar=cb, sbar=sb, nmax=DRAW_NMAX, mmax=DRAW_NMAX)
            draws.append(np.stack([sha(X[i, :3], mdl, rot[i]) for i in range(t.size)]))
    s_cols = [FE.MM_convolve(t, PHI, a) for a in [a_mm] + list(a_fe) + draws]
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, K.stations())
    ev = G.evaluate(fx, X, PHI, np.concatenate([S, np.stack(s_cols)], axis=0))
    wr, wd = 1 / K2.SIGMA_RANGE_M, 1 / K2.SIGMA_DOPPLER_MPS
    q = fx.nuis.shape[1]
    hx = np.vstack([ev["hx_r"] * wr, ev["hx_d"] * wd])
    hk = np.vstack([ev["hk_r"] * wr, ev["hk_d"] * wd])
    nu = np.vstack([np.zeros((ev["hx_r"].shape[0], q)), fx.nuis * wd])
    nfe = len(coeffs)
    np.savez_compressed(out_path, hx=hx, hk3=hk[:, :3], nu=nu, h_mm=hk[:, 3], h_fe=hk[:, 4:4 + nfe],
                        h_draws=hk[:, 4 + nfe:], coeffs=np.array(coeffs, dtype=object),
                        elapsed=time.time() - t0)
    return orbit, k, "ok"


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def _assemble(d, n_deg, fe_sigma):
    """Whitened design rows: data | state prior | FE prior (1/sigma on each FE column). Returns (A, n_data, fe_idx, sel)."""
    coeffs = [tuple(c) for c in d["coeffs"]]
    sel = [j for j, c in enumerate(coeffs) if c[1] <= n_deg]
    blocks = [d["hx"], d["hk3"], d["nu"]] + ([d["h_fe"][:, sel]] if sel else [])
    a = np.hstack(blocks)
    n_data = a.shape[0]
    rows = [a]
    pr = np.zeros((6, a.shape[1]))
    pr[:, :6] = np.diag(1 / K2.PRIOR_SD)
    rows.append(pr)
    base = 6 + 3 + d["nu"].shape[1]
    fe_idx = list(range(base, base + len(sel)))
    if sel and fe_sigma is not None:
        fp = np.zeros((len(sel), a.shape[1]))
        fp[np.arange(len(sel)), fe_idx] = 1.0 / np.asarray(fe_sigma)[sel]
        rows.append(fp)
    return np.vstack(rows), n_data, fe_idx, sel


def score(d, n_deg, fe_sigma, extra_cols=None):
    a, n_data, fe_idx, sel = _assemble(d, n_deg, fe_sigma)
    cs = np.linalg.norm(a, axis=0)
    cs[cs == 0] = 1.0
    an = a / cs
    r = np.linalg.qr(an, mode="r")
    rinv = np.linalg.solve(r, np.eye(r.shape[0]))
    cov = (rinv @ rinv.T) / np.outer(cs, cs)
    cf = cov[6:9, 6:9] / K2.K0 ** 2
    cv = K2.Q_VAR.T @ cf @ K2.Q_VAR
    one = np.ones(3) / 3
    pad = lambda h: np.concatenate([h, np.zeros(a.shape[0] - n_data)])  # noqa: E731

    def alias(h):
        th = np.linalg.lstsq(an, pad(h), rcond=None)[0] / cs
        df = th[6:9] / K2.K0
        v = K2.Q_VAR.T @ df
        res = pad(h) - a @ th
        return dict(fv_chi2=float(v @ np.linalg.solve(cv, v)), resid_chi2_data=float(res[:n_data] @ res[:n_data]),
                    prior_penalty=float(res[n_data:] @ res[n_data:]),
                    level_bias_over_sigma=float(one @ df / math.sqrt(one @ cf @ one)),
                    state_bias_pos_m=float(np.linalg.norm(th[:3])))
    out = dict(N=n_deg, sigma_v=np.sqrt(np.diag(cv)).tolist())
    for name in ("T1_STEP", "T2_MIDDLE"):
        v = K2.Q_VAR.T @ np.asarray(K2.TRUTHS[name])
        out[f"nc_{name}"] = float(v @ np.linalg.solve(cv, v))
    for al in ALPHAS:
        out[f"alpha{al}"] = alias(al * d["h_mm"])
    if fe_idx and fe_sigma is not None:
        post = np.sqrt(np.diag(cov)[fe_idx])
        prior = np.asarray(fe_sigma)[sel]
        out["field_post_over_prior_median"] = float(np.median(post / prior))
        out["field_post_over_prior_min"] = float(np.min(post / prior))
        sd = np.sqrt(np.diag(cov))
        corr = cov[np.ix_(range(6, 9), fe_idx)] / np.outer(sd[6:9], sd[fe_idx])
        out["max_abs_corr_K_field"] = float(np.max(np.abs(corr)))
    if extra_cols is not None:
        fv = np.array([alias(extra_cols[:, j])["fv_chi2"] for j in range(extra_cols.shape[1])])
        out["draws_fv"] = fv.tolist()
    return out


def _load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def gates(designs, sig) -> dict:
    out = {}
    name, d = next(iter(designs.items()))
    # P1: regularised normal equations
    a, n_data, fe_idx, sel = _assemble(d, 6, sig)
    a0, _, _, _ = _assemble(d, 6, None)
    info = a0.T @ a0
    info[np.ix_(fe_idx, fe_idx)] += np.diag(1.0 / np.asarray(sig)[sel] ** 2)
    ata = a.T @ a
    out["P1"] = float(np.max(np.abs(ata - info)) / np.max(np.abs(info)))
    # P4: sigma mapping
    coeffs = [tuple(c) for c in d["coeffs"]]
    lines = {}
    with open(GL_TAB) as f:
        f.readline()
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if int(p[0]) > 6:
                break
            lines[(int(p[0]), int(p[1]))] = p
    indep = np.array([float(lines[(n, m)][4 if k == "C" else 5]) for k, n, m in coeffs])
    spot = dict(C20=sig[coeffs.index(("C", 2, 0))], C21=sig[coeffs.index(("C", 2, 1))],
                S21=sig[coeffs.index(("S", 2, 1))])
    out["P4"] = dict(exact=bool(np.array_equal(indep, sig)), spot=spot,
                     spot_ok=bool(spot["C20"] == 4.593682635805e-12 and spot["C21"] == 1.303864780869e-12
                                  and spot["S21"] == 1.390771628190e-12))
    # P2 / P3 / P5 on every cell
    def close(x, y):
        return abs(x - y) <= max(1e-6 * abs(y), 1e-6)
    p2 = p3 = p5 = True
    worst = dict(P2=0.0, P3=0.0, P5=0.0)
    for name, d in designs.items():
        fel = json.loads((FE.CELLS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        n0 = score(d, 0, None)
        for key in ("nc_T1_STEP", "nc_T2_MIDDLE"):
            ref = fel["per_N"]["0"]["noncentrality_" + key[3:]]
            p5 &= abs(n0[key] - ref) <= 1e-8 * max(abs(ref), 1.0)
        for al in ALPHAS:
            ref = fel["per_N"]["0"][f"alpha{al}"]["fv_chi2"]
            p5 &= abs(n0[f"alpha{al}"]["fv_chi2"] - ref) <= 1e-8 * max(abs(ref), 1.0)
        for n_deg in N_SET:
            free = score(d, n_deg, None)
            weak = score(d, n_deg, sig * 1e8)
            strong = score(d, n_deg, sig * 1e-8)
            for key in ("nc_T1_STEP", "nc_T2_MIDDLE"):
                p2 &= close(weak[key], free[key])
                p3 &= close(strong[key], n0[key])
                worst["P2"] = max(worst["P2"], abs(weak[key] - free[key]) / max(abs(free[key]), 1.0))
                worst["P3"] = max(worst["P3"], abs(strong[key] - n0[key]) / max(abs(n0[key]), 1.0))
                ref = fel["per_N"][str(n_deg)]["noncentrality_" + key[3:]]
                p5 &= abs(free[key] - ref) <= 1e-8 * max(abs(ref), 1.0)
                worst["P5"] = max(worst["P5"], abs(free[key] - ref) / max(abs(ref), 1.0))
            for al in ALPHAS:
                k = f"alpha{al}"
                p2 &= close(weak[k]["fv_chi2"], free[k]["fv_chi2"])
                p3 &= close(strong[k]["fv_chi2"], n0[k]["fv_chi2"])
                ref = fel["per_N"][str(n_deg)][k]["fv_chi2"]
                p5 &= abs(free[k]["fv_chi2"] - ref) <= 1e-8 * max(abs(ref), 1.0)
                worst["P5"] = max(worst["P5"], abs(free[k]["fv_chi2"] - ref) / max(abs(ref), 1.0))
                worst["P2"] = max(worst["P2"], abs(weak[k]["fv_chi2"] - free[k]["fv_chi2"]) / max(abs(free[k]["fv_chi2"]), 1.0))
                worst["P3"] = max(worst["P3"], abs(strong[k]["fv_chi2"] - n0[k]["fv_chi2"]) / max(abs(n0[k]["fv_chi2"]), 1.0))
    out.update(P2=bool(p2), P3=bool(p3), P5=bool(p5), worst=worst)
    # supplementary (added after P2 failed as computed; does not replace P2): weak-limit convergence in the prior
    # scale - a correct prior gives deviation ~ scale^-2 once the prior is weaker than every data-only direction
    conv = {}
    for mult in (1e8, 1e9, 1e10, 1e11):
        dev = 0.0
        for name, d in designs.items():
            for n_deg in N_SET:
                free, weak = score(d, n_deg, None), score(d, n_deg, sig * mult)
                vals = [(weak[k], free[k]) for k in ("nc_T1_STEP", "nc_T2_MIDDLE")]
                vals += [(weak[f"alpha{al}"]["fv_chi2"], free[f"alpha{al}"]["fv_chi2"]) for al in ALPHAS]
                dev = max(dev, max(abs(w - f) / max(abs(f), 1.0) for w, f in vals))
        conv[f"{mult:.0e}"] = dev
    out["P2_supplementary_convergence_max_rel"] = conv
    out["pass"] = bool(out["P1"] <= 1e-10 and out["P4"]["exact"] and out["P4"]["spot_ok"] and p2 and p3 and p5)
    return out


def classify(rows_by_cell, eligible) -> dict:
    res = {}
    for n_deg in N_SET:
        cells_ok, sup_all, pres_all = 0, True, True
        for name, r in rows_by_cell.items():
            m = r[str(n_deg)]
            sup = all(m[f"alpha{a}"]["fv_chi2"] < CHI2 for a in ALPHAS)
            pres = all(m[f"nc_{t}"] >= CHI2 for t in eligible[name])
            sup_all &= sup
            pres_all &= pres
            cells_ok += int(sup and pres)
        res[str(n_deg)] = dict(SUPPRESSES=sup_all, PRESERVES=pres_all, cells_resolved=cells_ok)
    if any(v["SUPPRESSES"] and v["PRESERVES"] for v in res.values()):
        outcome = "FE_PRIOR_RESOLVES_DEGENERACY"
    elif any(v["SUPPRESSES"] for v in res.values()):
        outcome = "FE_PRIOR_SUPPRESSES_BUT_COSTS_VARIATION"
    elif any(v["cells_resolved"] >= 5 for v in res.values()):
        outcome = "FE_PRIOR_PARTIALLY_RESOLVES_DEGENERACY"
    else:
        outcome = "FE_PRIOR_TOO_WEAK_TO_SUPPRESS"
    return dict(outcome=outcome, per_N=res)


def run_score() -> dict:
    designs = {p.stem: _load(p) for p in sorted(DESIGNS.glob("*.npz"))}
    first = next(iter(designs.values()))
    coeffs = [tuple(c) for c in first["coeffs"]]
    table = parse_gl1800f(6)
    sig = sigma_vector(coeffs, table)
    gl, gr = MM._fields()
    stress = stress_vector(coeffs, gl, gr)
    g = gates(designs, sig)
    eligible = {}
    configs = {f"x{int(s)}": sig * s for s in SCALES}
    configs["stress_|dC|"] = stress
    rows = {c: {} for c in configs}
    n0 = {}
    draws = {}
    for name, d in designs.items():
        base = score(d, 0, None, extra_cols=d["h_draws"])
        n0[name] = base
        eligible[name] = [t for t in ("T1_STEP", "T2_MIDDLE") if base[f"nc_{t}"] >= CHI2]
        for cname, s in configs.items():
            rows[cname][name] = {str(n): score(d, n, s, extra_cols=d["h_draws"] if cname == "x1" else None)
                                 for n in N_SET}
        draws[name] = dict(N0=base["draws_fv"], **{f"x1_N{n}": rows["x1"][name][str(n)]["draws_fv"] for n in N_SET})
    science = {c: classify(rows[c], eligible) for c in rows}          # the rule without the gate veto
    cls = {c: (science[c] if g["pass"] else dict(outcome="FE_PRIOR_INCONCLUSIVE", gate_veto=True,
                                                    science_rule=science[c])) for c in rows}
    # prior-consistent masquerade risk (secondary)
    risk = {}
    for si, s in enumerate(SCALES):
        sl = slice(si * N_DRAWS, (si + 1) * N_DRAWS)
        for key in ("N0", "x1_N4", "x1_N6"):
            vals = np.concatenate([np.asarray(draws[n][key])[sl] for n in draws])
            risk[f"draw_scale_x{int(s)}_{key}"] = dict(median=float(np.median(vals)), max=float(np.max(vals)),
                                                       frac_ge_13p82=float(np.mean(vals >= CHI2)))
    return dict(gates=g, primary=cls["x1"], sensitivity={c: cls[c] for c in ("x10", "x100")},
                stress=cls["stress_|dC|"], eligible=eligible, prior_consistent_masquerade_risk=risk,
                rows=rows, n0=n0, sigma_deg2_6=dict(zip([f"{k}{n}{m}" for k, n, m in coeffs], sig.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.designs:
        DESIGNS.mkdir(parents=True, exist_ok=True)
        cells = [(o, k) for o in K.ORBITS for k in (0, 12)]
        specs = [dict(orbit=o, k=k, ci=i) for i, (o, k) in enumerate(cells)]
        with ProcessPoolExecutor(a.workers) as ex:
            for o, k, st in ex.map(cell_design, specs):
                print(o, k, st, flush=True)
    if a.score:
        out = run_score()
        (ART / "phase17_fep_summary.json").write_text(json.dumps(K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("gates", "primary", "sensitivity", "stress",
                                              "prior_consistent_masquerade_risk")}, indent=1, default=str))


if __name__ == "__main__":
    main()
