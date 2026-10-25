"""Phase D1: integrator and propagator accuracy, shadow events, and the estimator closure floor (analysis only).

Predeclared in studies/phase17/configs/phase_d1_integrator_predeclaration.json (committed before any number).
Writes studies/phase17/results/phase_d1_integrator.json.

    python studies/phase17/scripts/phase_d1_integrator.py [--workers 9]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase_d1_integrator.json"
ORBITS = ("CANON", "O1_LOW_BETA_ECLIPSE_RICH", "O2_FULL_SUN_HIGH_BETA")
DAY = 86400.0


def _config(orbit: str, shadow: str | None = None):
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions
    P.K._init()
    x0 = np.asarray(P.x0s()[orbit], float)
    scn = P.K.scenario(orbit, x0.tolist(), 0)
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]),
            lambda tt: G.sun_mci(scn.et0 + float(tt)))
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=shadow or scn.shadow_model)
    return dict(x0=x0, scn=scn, t=np.asarray(scn.t_grid, float), args=args, srp=srp,
                j2=J2_MOON_UNNORMALIZED)


def _illum(c, t, r):
    from lunar_od.srp import _illumination_for
    return float(_illumination_for(np.asarray(r, float), np.asarray(c["args"][4](t), float), c["srp"]))


def _events(c, dense_t, dense_x):
    """Penumbra/umbra boundary epochs by bisection on the Hermite-interpolated preliminary trajectory."""
    from lunar_od.radiometrics import _interp_state
    nu = np.array([_illum(c, t, x[:3]) for t, x in zip(dense_t, dense_x)])
    cls = np.where(nu >= 1.0, 2, np.where(nu <= 0.0, 0, 1))
    ev = []
    for i in np.flatnonzero(cls[1:] != cls[:-1]):
        a, b = float(dense_t[i]), float(dense_t[i + 1])
        ca = cls[i]
        for _ in range(60):
            m = 0.5 * (a + b)
            nm = _illum(c, m, _interp_state(dense_t, dense_x, m)[:3])
            cm = 2 if nm >= 1.0 else (0 if nm <= 0.0 else 1)
            if cm == ca:
                a = m
            else:
                b = m
            if b - a < 1e-4:
                break
        ev.append(dict(t=0.5 * (a + b), bracket_s=b - a, from_=int(ca), to=int(cls[i + 1])))
    return ev


def _segmented(c, t_out, events, rtol, atol):
    from lunar_od.dynamics import propagate_state
    cuts = [e["t"] for e in events if t_out[0] < e["t"] < t_out[-1]]
    bounds = [float(t_out[0])] + cuts + [float(t_out[-1])]
    x = np.asarray(c["x0"], float)
    out = np.zeros((t_out.size, 6))
    out[0] = x
    for a, b in zip(bounds[:-1], bounds[1:]):
        inside = t_out[(t_out > a) & (t_out <= b)]
        te = np.unique(np.concatenate([[a], inside, [b]]))
        seg = np.asarray(propagate_state(te, x, *c["args"], srp=c["srp"], j2_moon=c["j2"], rtol=rtol, atol=atol,
                                         method="DOP853"), float)
        for k, tk in enumerate(te):
            j = np.searchsorted(t_out, tk)
            if j < t_out.size and t_out[j] == tk:
                out[j] = seg[k]
        x = seg[-1]
    return out


def run_orbit(orbit: str) -> dict:
    from lunar_od.dynamics import propagate_augmented_state, propagate_state, propagate_state_with_k_sensitivity
    t0 = time.time()
    c = _config(orbit)
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12)
    dense_t = np.arange(t[0], t[-1] + 1e-9, 10.0)
    prelim = np.asarray(propagate_state(dense_t, c["x0"], *c["args"], method="DOP853", **(kw | dict(rtol=1e-12, atol=1e-13))))
    events = _events(c, dense_t, prelim)
    ref = _segmented(c, t, events, 1e-13, 1e-14)
    ref12 = _segmented(c, t, events, 1e-12, 1e-13)
    aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
    var = {
        "A": np.asarray(propagate_state(t, c["x0"], *c["args"], **kw))[:, :6],
        "B": np.asarray(propagate_augmented_state(t, aug0, *c["args"], **kw))[:, :6],
        "C": np.asarray(propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw))[:, :6],
        "D": np.asarray(propagate_state(t, c["x0"], *c["args"], method="DOP853", **kw))[:, :6],
    }
    i_day = int(np.argmin(np.abs(t - (t[0] + DAY))))

    def err(x, r):
        e = np.linalg.norm(x[:, :3] - r[:, :3], axis=1)
        return dict(max_m=float(e.max()), at_1day_m=float(e[i_day]))
    q1 = {k: err(v, ref) for k, v in var.items()}
    # E: no shadow, ADAMS vs a (no-event) DOP853 1e-13 reference
    cn = _config(orbit, "NO_SHADOW")
    kwn = dict(srp=cn["srp"], j2_moon=cn["j2"])
    a_ns = np.asarray(propagate_state(t, cn["x0"], *cn["args"], rtol=1e-11, atol=1e-12, **kwn))
    r_ns = np.asarray(propagate_state(t, cn["x0"], *cn["args"], rtol=1e-13, atol=1e-14, method="DOP853", **kwn))
    q1["E_no_shadow_ADAMS"] = err(a_ns, r_ns)
    # R3: A reproduces the closure-gate truth construction
    r3 = bool(np.array_equal(var["A"], np.asarray(propagate_state(t, c["x0"], *c["args"], **kw))[:, :6]))
    # R2: event brackets
    r2 = all(e["bracket_s"] <= 1e-3 for e in events)
    return dict(orbit=orbit, n_events=len(events), events=events, elapsed_s=time.time() - t0,
                R1_reference_convergence_m=float(np.max(np.linalg.norm(ref[:, :3] - ref12[:, :3], axis=1))),
                R2_events=r2, R3_reproduction=r3, Q1=q1,
                Q2=dict(A_minus_B_max_m=float(np.max(np.linalg.norm(var["A"][:, :3] - var["B"][:, :3], axis=1))),
                        A_minus_C_max_m=float(np.max(np.linalg.norm(var["A"][:, :3] - var["C"][:, :3], axis=1)))),
                reference=ref.tolist() if orbit == "CANON" else None)


def run_closure(spec) -> dict:
    truth_kind, estimator, reference = spec
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import propagate_augmented_state, propagate_state
    c = _config("CANON")
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12)
    if truth_kind == "A":
        truth = np.asarray(propagate_state(t, c["x0"], *c["args"], **kw))
    elif truth_kind == "REFERENCE":
        truth = np.asarray(reference, float)
    else:
        aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
        truth = np.asarray(propagate_augmented_state(t, aug0, *c["args"], **kw))[:, :6]
    X, PHI, S = K2.propagate_segmented(c["scn"], c["x0"], np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(c["scn"], X, PHI, S, P.K.stations())
    rb = fx.rb
    obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                        rb.earth.vel, c["scn"].et0, model=R.DSN_C2_LIGHT_TIME_MODEL,
                                                        noise=False)
    rng = np.random.default_rng([98, 0, 0])
    start = c["x0"] + rng.normal(0.0, K2.PRIOR_SD)
    fn = {"BLS_LM": E.estimate_two_way_range_bls_lm, "SRIF": E.estimate_two_way_range_srif}[estimator]
    ts = time.time()
    xh, status, st = fn(t, obs, start, pg, *c["args"], srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12)
    xh = np.asarray(xh, float)
    return dict(truth=truth_kind, estimator=estimator, status=status, detail=st.termination_detail,
                dr_m=float(np.linalg.norm(xh[:3] - c["x0"][:3])), dv_mps=float(np.linalg.norm(xh[3:] - c["x0"][3:])),
                iterations=st.iterations, elapsed_s=time.time() - ts)


def q4_hermite() -> dict:
    from lunar_od.radiometrics import _interp_state
    c = _config("CANON")
    from lunar_od.dynamics import propagate_state
    t = c["t"]
    x = np.asarray(propagate_state(t, c["x0"], *c["args"], srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12))
    out = {}
    for label, i in (("first", 0), ("interior", t.size // 2), ("last", t.size - 2)):
        worst = 0.0
        for f in np.linspace(0.05, 0.95, 19):
            tm = t[i] + f * (t[i + 1] - t[i])
            h = 1e-3
            fd = (_interp_state(t, x, tm + h)[:3] - _interp_state(t, x, tm - h)[:3]) / (2 * h)
            worst = max(worst, float(np.linalg.norm(_interp_state(t, x, tm)[3:] - fd)))
        out[label] = worst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        orbits = list(ex.map(run_orbit, ORBITS))
        ref = next(o["reference"] for o in orbits if o["orbit"] == "CANON")
        specs = [(k, e, ref) for k in ("A", "REFERENCE", "B") for e in ("BLS_LM", "SRIF")]
        closure = list(ex.map(run_closure, specs))
        q4 = ex.submit(q4_hermite).result()
    for o in orbits:
        o.pop("reference", None)
    by = {(r["truth"], r["estimator"]): r for r in closure}
    bls_b, bls_a = by[("B", "BLS_LM")], by[("A", "BLS_LM")]
    verdict = ("HYPOTHESIS_CONFIRMED" if bls_b["status"] == "Converged" and bls_b["dr_m"] <= 1e-3
               and bls_a["status"] != "Converged" else "NOT_CONFIRMED")
    out = dict(predeclaration="studies/phase17/configs/phase_d1_integrator_predeclaration.json",
               orbits=orbits, closure=closure, Q4_hermite_velocity_minus_position_derivative_mps=q4,
               closure_floor_verdict=verdict,
               gates=dict(R1=all(o["R1_reference_convergence_m"] <= 1e-3 for o in orbits),
                          R2=all(o["R2_events"] for o in orbits), R3=all(o["R3_reproduction"] for o in orbits)),
               elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(gates=out["gates"], verdict=verdict,
                                     Q1={o["orbit"]: o["Q1"] for o in orbits},
                                     Q2={o["orbit"]: o["Q2"] for o in orbits},
                                     R1={o["orbit"]: o["R1_reference_convergence_m"] for o in orbits},
                                     events={o["orbit"]: o["n_events"] for o in orbits},
                                     closure=[{k: r[k] for k in ("truth", "estimator", "status", "dr_m", "iterations")}
                                              for r in closure], Q4=q4)), indent=1))


if __name__ == "__main__":
    main()
