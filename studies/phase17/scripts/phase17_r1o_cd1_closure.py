"""PHASE 17-R1O-CD1 closure - HARDEN-D23 materiality and the rebuild numerical floor.

Analysis only.  The CD1 preregistration, decision tree, thresholds, noise and
nuisance models are loaded unchanged (hash-pinned); production code is not
modified.

1. HARDEN-D23 materiality.  Production counted Doppler interpolates the Earth
   LINEARLY between the 90 s grid nodes (R3-P22).  Every primary cell (5 orbits x
   24 epochs x {HIGH_FIDELITY, SIMPLIFIED}) is rebuilt twice on the same
   propagation: once exactly as CD1 did, once with a reference station-state
   provider whose Earth term is the SAME SPICE source the grid was sampled from
   (``spkezr EARTH / J2000 / NONE / MOON``) evaluated at the exact event epoch:

       reference_state(t) = production_state(t) - linear_earth(t) + spice_earth(t)

   so the exact event-epoch sxform station term is untouched and only the D23
   approximation is replaced.  It is injected through the production
   ``station_state_provider`` seam; no production code changes.
2. Numerical floor.  The one strict rebuild exceedance (O2 k=6 RPD|D2|FREE,
   1.46e-10 > 1e-10) and the next two largest comparisons are re-derived against
   an EXACT rational oracle: f_perp^2 = det(G) / (det(G_AA) * G_bb) for the exact
   Gram matrix G of the whitened [state | nuisance | K] design, by integer
   Bareiss elimination (float64 inputs are dyadic, so there is zero arithmetic error).

    python examples/phase17_r1o_cd1_closure.py              # both, full population
    python examples/phase17_r1o_cd1_closure.py --analyse-only
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_r1o_cd1_campaign as K  # noqa: E402
import phase17_r1o_cd1_core as CD  # noqa: E402

D2 = K.D2
ARTIFACTS = CD.ARTIFACTS
TC = 60.0
#: the strict exceedance and the next two largest rebuild comparisons (all RPD|D2|FREE)
ORACLE_CELLS = (("O2_FULL_SUN_HIGH_BETA", 6), ("O5_FAVOURABLE_GEO_MAX_FPERP", 14),
                ("O5_FAVOURABLE_GEO_MAX_FPERP", 21))
#: decision-driving configurations (predeclared decision tree, retention and fidelity classes)
CONFIGS = (("DOP|D0", "D0_IDEAL", False), ("DOP|D1|FREE", "D1_SESSION_CONSTANT_BIAS", False),
           ("RPD|D0", "D0_IDEAL", True), ("RPD|D1|FREE", "D1_SESSION_CONSTANT_BIAS", True))
RAW_PATH = ARTIFACTS / "phase17_r1o_cd1_closure_raw.json"


# ======================================================================
# Reference Earth provider (analysis seam, production untouched)
# ======================================================================
@contextlib.contextmanager
def reference_earth(record: dict):
    """Within the block, counted-Doppler rows resolve a provider whose Earth term is
    SPICE at the exact event epoch instead of the linear grid interpolation."""
    import lunar_od.radiometrics as RM
    from phase17_r1m_core import _spice_earth_state_mci

    original = RM.resolve_counted_doppler_station_state_provider

    def resolve(config, station, t_grid_s, earth_pos_mci_m, earth_vel_mci_mps, *, et0_s=None, **kw):
        prod = original(config, station, t_grid_s, earth_pos_mci_m, earth_vel_mci_mps, et0_s=et0_s, **kw)
        t_grid = np.asarray(t_grid_s, float)
        ep, ev = np.asarray(earth_pos_mci_m, float), np.asarray(earth_vel_mci_mps, float)

        def state_fn(t: float) -> np.ndarray:
            lin = np.concatenate([RM._interp_vector(t_grid, ep, t), RM._interp_vector(t_grid, ev, t)])
            ref = _spice_earth_state_mci(float(et0_s) + float(t))
            dp, dv = float(np.linalg.norm(ref[:3] - lin[:3])), float(np.linalg.norm(ref[3:] - lin[3:]))
            record["earth_pos_m"] = max(record.get("earth_pos_m", 0.0), dp)
            record["earth_vel_mps"] = max(record.get("earth_vel_mps", 0.0), dv)
            record["n_events"] = record.get("n_events", 0) + 1
            if float(t) in record["nodes"]:
                record["node_pos_m"] = max(record.get("node_pos_m", 0.0), dp)
            return np.asarray(prod.state_fn(t), float) - lin + ref

        return RM.CountedDopplerStationStateProvider(state_fn=state_fn,
                                                     station_name=prod.station_name)

    RM.resolve_counted_doppler_station_state_provider = resolve
    try:
        yield record
    finally:
        RM.resolve_counted_doppler_station_state_provider = original


# ======================================================================
# Exact rational f_perp oracle
# ======================================================================
def _bareiss_leading_minors(mat: list[list[int]]) -> list[int]:
    """Fraction-free elimination; returns det of every leading principal submatrix."""
    a = [row[:] for row in mat]
    n = len(a)
    prev, minors = 1, []
    for k in range(n):
        if a[k][k] == 0:
            raise ZeroDivisionError("zero leading minor at %d (exactly singular)" % k)
        minors.append(a[k][k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
        prev = a[k][k]
    return minors


def exact_f_perp(h_other: np.ndarray, h_k: np.ndarray, w: np.ndarray) -> Decimal:
    """f_perp of h_k against span(h_other) under weights w, exact up to the final sqrt.

    G = [A | b]^T W [A | b] exactly; the Schur complement of G_AA is det(G)/det(G_AA)
    and f^2 = schur / G_bb.  Projection is invariant to column scaling, so the
    oracle needs neither the D2 design scale nor the nuisance unit.
    """
    h = np.hstack([np.asarray(h_other, float), np.asarray(h_k, float)[:, None]])
    n = h.shape[1]
    hf = [[Fraction(float(v)) for v in row] for row in h]
    wf = [Fraction(float(v)) for v in np.asarray(w, float)]
    g = [[Fraction(0)] * n for _ in range(n)]
    for row, wi in zip(hf, wf):
        nz = [(i, v * wi) for i, v in enumerate(row) if v]
        for i, ri in nz:
            for j, vj in enumerate(row):
                if j >= i and vj:
                    g[i][j] += ri * vj
    for i in range(n):
        for j in range(i):
            g[i][j] = g[j][i]
    den = 1
    for row in g:
        for v in row:
            den = den * v.denominator // math.gcd(den, v.denominator)
    gi = [[int(v * den) for v in row] for row in g]
    minors = _bareiss_leading_minors(gi)
    schur = Fraction(minors[-1], minors[-2])
    ratio = schur / Fraction(gi[-1][-1])
    getcontext().prec = 40
    return (Decimal(ratio.numerator) / Decimal(ratio.denominator)).sqrt()


def floor_check(rb, r60, w60) -> dict:
    model = "D2_SESSION_BIAS_PLUS_LINEAR_DRIFT"
    nu = CD.nuisance_columns(rb.n_obs, r60, model)
    kappa = CD.DOPPLER_NUIS_UNIT_MPS / D2.NUISANCE_SCALE_S
    h_x0 = np.vstack([rb.h_x0, r60.h_x0])
    h_k = np.concatenate([rb.h_k, r60.h_k])
    w = np.concatenate([rb.w, w60])
    t0 = time.time()
    exact = exact_f_perp(np.hstack([h_x0, nu * kappa]), h_k, w)
    wrapper = CD.metrics(D2, rb, r60, w60, model, None)
    indep = CD.independent_f_perp(D2, rb, r60, w60, model, None)
    # wrapper-equivalent alternative float routes on the identical design
    a_other = np.sqrt(w)[:, None] * np.hstack([h_x0, nu * kappa])
    b = np.sqrt(w) * h_k
    u, s, _ = np.linalg.svd(a_other, full_matrices=False)
    u = u[:, s > s[0] * a_other.shape[0] * np.finfo(float).eps]
    f_svd = float(np.linalg.norm(b - u @ (u.T @ b)) / np.linalg.norm(b))
    ex = float(exact)
    return dict(n_rows=int(h_k.size), n_cols_other=int(h_x0.shape[1] + nu.shape[1]),
                design_condition=float(wrapper["design_condition"]),
                f_exact=str(exact), f_wrapper=float(wrapper["f_perp"]), f_independent_lstsq=indep,
                f_svd_projector=f_svd,
                err_wrapper=abs(float(wrapper["f_perp"]) - ex), err_independent=abs(indep - ex),
                err_svd=abs(f_svd - ex), wrapper_minus_independent=abs(float(wrapper["f_perp"]) - indep),
                oracle_s=time.time() - t0)


# ======================================================================
# one cell
# ======================================================================
def _metrics(rb, rows, w, f0) -> dict:
    out = {}
    for name, model, with_range in CONFIGS:
        m = CD.metrics(D2, rb, rows, w, model, None, with_range=with_range)
        out[name] = dict(f_perp=float(m["f_perp"]), delta_f_perp=float(m["f_perp"]) - f0)
    return out


def run_cell(spec: dict) -> dict:
    t0 = time.time()
    orbit, k, dyn = spec["orbit"], spec["k"], spec["dynamics"]
    scn = K.scenario(orbit, spec["x0"], k)
    if dyn == "HIGH_FIDELITY":
        from phase17_r1od2_fidelity import propagate_fidelity
        nom = propagate_fidelity(scn)
    else:
        nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    st = K.stations()
    f0 = float(D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])["f_perp"])
    prod = CD.build_doppler_rows(scn, rb, TC, st, with_values=True)
    rec = dict(nodes=set(float(t) for t in rb.t_grid))
    with reference_earth(rec):
        ref = CD.build_doppler_rows(scn, rb, TC, st, with_values=True)
    rec.pop("nodes")
    same_rows = bool(prod.n == ref.n and np.array_equal(prod.t_mid, ref.t_mid)
                     and np.array_equal(prod.station, ref.station) and np.array_equal(prod.session, ref.session))
    out = dict(orbit=orbit, k=k, dynamics=dyn, range_f_perp=f0, n_doppler=int(prod.n),
               feasible=bool(prod.n >= 7), same_rows=same_rows, earth=rec)
    if prod.n:
        hx_rel = np.linalg.norm(ref.h_x0 - prod.h_x0, axis=1) / np.linalg.norm(prod.h_x0, axis=1)
        out.update(
            y_abs_max_mps=float(np.max(np.abs(ref.y - prod.y))),
            y_abs_rms_mps=float(np.sqrt(np.mean((ref.y - prod.y) ** 2))),
            hx_rel_max=float(np.max(hx_rel)),
            hk_abs_over_colmax=float(np.max(np.abs(ref.h_k - prod.h_k)) / (np.max(np.abs(prod.h_k)) or 1.0)))
    if out["feasible"]:
        w60 = np.full(prod.n, 1.0 / CD.SIGMA_D60_PRIMARY_MPS ** 2)
        out["prod"], out["ref"] = _metrics(rb, prod, w60, f0), _metrics(rb, ref, w60, f0)
        if dyn == "HIGH_FIDELITY" and (orbit, k) in ORACLE_CELLS:
            out["floor"] = floor_check(rb, prod, w60)
    out["elapsed_s"] = time.time() - t0
    return K._clean(out)


# ======================================================================
# analysis
# ======================================================================
def _verdict(cells: list, side: str) -> dict:
    """The predeclared CD1 decision fields, recomputed with campaign.py's own rules."""
    by = {(c["orbit"], c["k"], c["dynamics"]): c for c in cells}
    hf = [by[(o, k, "HIGH_FIDELITY")] for o in K.ORBITS for k in K.EPOCHS]
    simp = [by[(o, k, "SIMPLIFIED")] for o in K.ORBITS for k in K.EPOCHS]
    d = lambda c, n: c[side][n]["delta_f_perp"]  # noqa: E731
    feas = [c for c in hf if c["feasible"]]
    p0, p1 = K.population([d(c, "RPD|D0") for c in feas]), K.population([d(c, "RPD|D1|FREE") for c in feas])
    abs_ok = float(np.mean([c[side]["RPD|D1|FREE"]["f_perp"] >= 0.35 for c in feas])) >= 0.50
    ret = [d(c, "RPD|D1|FREE") / d(c, "RPD|D0") for c in feas if d(c, "RPD|D0") > 0.02]
    ps0 = K.population([d(c, "RPD|D0") for c in simp if c["feasible"]])
    pairs = [(by[(o, k, "HIGH_FIDELITY")], by[(o, k, "SIMPLIFIED")]) for o in K.ORBITS for k in K.EPOCHS]
    pairs = [(a, b) for a, b in pairs if a["feasible"] and b["feasible"]]
    rms = float(np.sqrt(np.mean([(d(a, "RPD|D0") - d(b, "RPD|D0")) ** 2 for a, b in pairs])))
    return dict(PRIMARY_CLASS=K.decision(p0, p1, abs_ok), POP_D0=p0, POP_D1FREE=p1, ABS=abs_ok,
                NUISANCE_RETENTION_CLASS=K.nuisance_retention_class(ret),
                nuisance_retention_median=float(np.median(ret)) if ret else None,
                POP_D0_SIMPLIFIED=ps0["POP"], FIDELITY_EFFECT_CLASS=K.fidelity_class(ps0["POP"], p0["POP"], rms))


def analyse(cells: list) -> dict:
    import csv
    committed = {(r["orbit"], int(r["k"]), r["dynamics"]): r for r in csv.DictReader(
        (ARTIFACTS / "phase17_r1o_cd1_primary_results.csv").open())}
    feas = [c for c in cells if c["feasible"]]
    # anchor: the production rebuild here must reproduce the committed CD1 numbers
    anchor = []
    for c in feas:
        r = committed[(c["orbit"], c["k"], c["dynamics"])]
        for col, name in (("combined_f_perp_D0", "RPD|D0"), ("combined_f_perp_D1FREE", "RPD|D1|FREE"),
                          ("doppler_only_f_perp_D0", "DOP|D0"), ("doppler_only_f_perp_D1FREE", "DOP|D1|FREE")):
            anchor.append(abs(float(r[col]) - c["prod"][name]["f_perp"]))
        anchor.append(abs(float(r["range_f_perp"]) - c["range_f_perp"]))
    # per-cell perturbation and class changes
    df, flips, margins = [], [], []
    for c in feas:
        for name, _, _ in CONFIGS:
            fp, fr = c["prod"][name]["f_perp"], c["ref"][name]["f_perp"]
            df.append(abs(fr - fp))
            if name.startswith("RPD"):
                dp, dr = c["prod"][name]["delta_f_perp"], c["ref"][name]["delta_f_perp"]
                margins.append(min(abs(dp - t) for t in (0.0, 0.02, 0.05, -0.02)))
                margins.append(min(abs(fp - t) for t in (0.35, 0.50)))
                if (D2.relative_gain_class(dp) != D2.relative_gain_class(dr)
                        or D2.absolute_class(fp) != D2.absolute_class(fr) or (dp > 0) != (dr > 0)):
                    flips.append(dict(orbit=c["orbit"], k=c["k"], dynamics=c["dynamics"], config=name,
                                      delta_prod=dp, delta_ref=dr, f_prod=fp, f_ref=fr))
    v_prod, v_ref = _verdict(cells, "prod"), _verdict(cells, "ref")
    keys = ("PRIMARY_CLASS", "ABS", "NUISANCE_RETENTION_CLASS", "POP_D0_SIMPLIFIED", "FIDELITY_EFFECT_CLASS")
    same = all(v_prod[k] == v_ref[k] for k in keys) and all(
        v_prod[p]["POP"] == v_ref[p]["POP"] for p in ("POP_D0", "POP_D1FREE"))
    gate = "PASS" if (same and not flips and max(anchor) <= 1e-12
                      and all(c["same_rows"] for c in cells)) else "FAIL"
    ex = lambda key: float(np.max([c[key] for c in cells if key in c]))  # noqa: E731
    d23 = dict(
        method="reference provider: production station state - linear Earth + SPICE Earth at the exact event epoch",
        cells=len(cells), feasible_cells=len(feas), events=int(sum(c["earth"].get("n_events", 0) for c in cells)),
        same_rows_all=all(c["same_rows"] for c in cells),
        earth_pos_diff_max_m=float(np.max([c["earth"].get("earth_pos_m", 0.0) for c in cells])),
        earth_vel_diff_max_mps=float(np.max([c["earth"].get("earth_vel_mps", 0.0) for c in cells])),
        earth_pos_diff_at_grid_nodes_max_m=float(np.max([c["earth"].get("node_pos_m", 0.0) for c in cells])),
        observable_diff_max_mps=ex("y_abs_max_mps"), observable_diff_rms_max_mps=ex("y_abs_rms_mps"),
        state_partial_rel_diff_max=ex("hx_rel_max"), k_partial_diff_over_colmax_max=ex("hk_abs_over_colmax"),
        f_perp_abs_diff_max=float(np.max(df)), f_perp_abs_diff_median=float(np.median(df)),
        closest_decision_margin=float(np.min(margins)),
        margin_over_perturbation=float(np.min(margins) / max(np.max(df), 1e-300)),
        class_or_sign_changes=flips,
        production_anchor_max_abs_diff=float(max(anchor)), production_anchor_gate=1e-12,
        verdict_production=v_prod, verdict_reference=v_ref, decision_fields_identical=same,
        HARDEN_D23_CD1_MATERIALITY_GATE=gate)
    floors = [dict(orbit=c["orbit"], k=c["k"], **c["floor"]) for c in cells if "floor" in c]
    return dict(d23=d23, floor=floors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--workers", type=int, default=K.N_WORKERS)
    args = ap.parse_args()
    pre = CD.load_predeclaration(K.PREDECLARATION_SHA256)
    if args.analyse_only:
        cells = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        import hashlib
        g = json.loads((ARTIFACTS / "phase17_r1od2_predeclared_grid.json").read_text(encoding="utf-8"))
        x0s = {x["case_id"]: x["x0"] for x in g["geometries"] if x["case_id"] in K.ORBITS}
        for o, x0 in x0s.items():
            assert hashlib.sha256(json.dumps(x0).encode()).hexdigest() == pre["orbits"]["set"][o]["x0_sha256"], o
        assert sorted(x0s) == sorted(K.ORBITS)
        specs = [dict(orbit=o, x0=x0s[o], k=k, dynamics=dyn) for dyn in ("HIGH_FIDELITY", "SIMPLIFIED")
                 for o in K.ORBITS for k in K.EPOCHS]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=K._init) as ex:
            cells = []
            for i, c in enumerate(ex.map(run_cell, specs)):
                cells.append(c)
                print("%3d/%d %s k%02d %s %.0f s" % (i + 1, len(specs), c["orbit"], c["k"], c["dynamics"],
                                                     time.time() - t0), flush=True)
        RAW_PATH.write_text(json.dumps(cells, separators=(",", ":")), encoding="utf-8")
    res = analyse(cells)
    res = dict(phase="17-R1O-CD1 closure", predeclaration_sha256=K.PREDECLARATION_SHA256,
               truth_level="SYNTHETIC_CAMPAIGN_TRUTH", production_code_changed=False, **res)
    (ARTIFACTS / "phase17_r1o_cd1_closure.json").write_text(json.dumps(K._clean(res), indent=1), encoding="utf-8")
    d = res["d23"]
    print("D23 gate %s: earth %.3g m / %.3g m/s, y %.3g m/s, df max %.3g, margin %.3g, flips %d"
          % (d["HARDEN_D23_CD1_MATERIALITY_GATE"], d["earth_pos_diff_max_m"], d["earth_vel_diff_max_mps"],
             d["observable_diff_max_mps"], d["f_perp_abs_diff_max"], d["closest_decision_margin"],
             len(d["class_or_sign_changes"])))
    for f in res["floor"]:
        print("floor %s k%d: exact %s wrapper err %.2e indep err %.2e svd err %.2e"
              % (f["orbit"], f["k"], f["f_exact"][:16], f["err_wrapper"], f["err_independent"], f["err_svd"]))


if __name__ == "__main__":
    main()
