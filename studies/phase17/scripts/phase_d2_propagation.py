"""Phase D2 qualification: shadow-event propagation and the consistent estimator profile.

Gates G1-G8 of studies/phase17/configs/phase_d2_propagation_predeclaration.json.
The independent reference is D1's construction (phase_d1_integrator._events /
_segmented): contacts by bisection of the illumination factor, integration with
shadow_events=False restarted there. It shares no event code with production.
"""
from __future__ import annotations

import argparse
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
warnings.simplefilter("ignore")

import phase_d1_integrator as D1  # noqa: E402
import phase_d2_baseline as B  # noqa: E402

P = D1.P
K2 = D1.K2
OUT = P.ARTIFACTS / "phase_d2_propagation.json"
REPO = HERE.parents[2]
TOL = dict(rtol=1e-11, atol=1e-12)


def _profile_kw():
    from lunar_od.dynamics import PropagationProfile
    return PropagationProfile().propagator_kwargs()


def _rmax(a, b):
    return float(np.max(np.linalg.norm(np.asarray(a)[:, :3] - np.asarray(b)[:, :3], axis=1)))


def g1_default_invariance() -> dict:
    from lunar_od.dynamics import propagate_augmented_state, propagate_state
    base = dict(np.load(B.OUT))
    now = B.default_outputs()
    eq = {k: bool(np.array_equal(base[k], now[k])) for k in base}
    # shadow_events=True must be inert without a penumbra to segment at
    c = D1._config("CANON")
    cn = D1._config("CANON", "NO_SHADOW")
    t = c["t"]
    aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
    eq["state_no_shadow_events_on"] = bool(np.array_equal(
        base["state_no_shadow"],
        propagate_state(t, c["x0"], *cn["args"], srp=cn["srp"], j2_moon=c["j2"], shadow_events=True)))
    eq["augmented_no_shadow_events_on"] = bool(np.array_equal(
        base["augmented_no_shadow"],
        propagate_augmented_state(t, aug0, *cn["args"], srp=cn["srp"], j2_moon=c["j2"], shadow_events=True)))
    eq["state_no_srp_events_on"] = bool(np.array_equal(
        base["state_no_srp"], propagate_state(t, c["x0"], *c["args"], j2_moon=c["j2"], shadow_events=True, **TOL)))
    return dict(equal=eq, PASS=all(eq.values()))


def run_orbit(orbit: str) -> dict:
    from lunar_od.dynamics import (propagate_augmented_state, propagate_state, propagate_state_with_k_sensitivity,
                                   shadow_contact_times)
    t0 = time.time()
    c = D1._config(orbit)
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"], **TOL)
    dense_t = np.arange(t[0], t[-1] + 1e-9, 10.0)
    prelim = np.asarray(propagate_state(dense_t, c["x0"], *c["args"], method="DOP853",
                                        **(kw | dict(rtol=1e-12, atol=1e-13))))
    ref_events = D1._events(c, dense_t, prelim)
    ref = D1._segmented(c, t, ref_events, 1e-13, 1e-14)
    pk = _profile_kw()
    aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
    timing = {}

    def timed(name, f):
        ts = time.time()
        out = np.asarray(f())
        timing[name] = time.time() - ts
        return out
    st = timed("state_profile", lambda: propagate_state(t, c["x0"], *c["args"], **kw, **pk))
    ag = timed("augmented_profile", lambda: propagate_augmented_state(t, aug0, *c["args"], **kw, **pk))
    ks = timed("ksens_profile", lambda: propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw, **pk))
    timed("state_default", lambda: propagate_state(t, c["x0"], *c["args"], **kw))
    timed("augmented_default", lambda: propagate_augmented_state(t, aug0, *c["args"], **kw))
    timed("ksens_default", lambda: propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw))
    contacts = shadow_contact_times(t, c["x0"], *c["args"], srp=c["srp"], j2_moon=c["j2"], **TOL)
    ref_t = np.array([e["t"] for e in ref_events])
    if len(contacts) == ref_t.size and ref_t.size:
        dt_max = float(np.max(np.abs(np.sort(np.asarray(contacts)) - np.sort(ref_t))))
    elif len(contacts) == ref_t.size == 0:
        dt_max = 0.0
    else:
        dt_max = float("inf")
    g3 = {"state": _rmax(st, ref), "augmented": _rmax(ag, ref), "ksens": _rmax(ks, ref)}
    g5 = {"state_minus_augmented": _rmax(st, ag), "state_minus_ksens": _rmax(st, ks)}
    return dict(orbit=orbit, n_contacts=len(contacts), n_reference_events=int(ref_t.size),
                G3_max_error_m=g3, G4_contact_dt_max_s=dt_max, G5_mismatch_m=g5, timing_s=timing,
                elapsed_s=time.time() - t0, reference=ref.tolist() if orbit == "CANON" else None)


def g6_variational() -> dict:
    from dataclasses import replace

    from lunar_od.dynamics import propagate_augmented_state, propagate_state, propagate_state_with_k_sensitivity
    c = D1._config("CANON")
    t = c["t"]
    pk = _profile_kw()
    kw = dict(srp=c["srp"], j2_moon=c["j2"], **TOL, **pk)
    aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
    phi = np.asarray(propagate_augmented_state(t, aug0, *c["args"], **kw))[-1, 6:42].reshape((6, 6), order="F")
    s_k = np.asarray(propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw))[-1, 42:48]
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
    col_err = []
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = steps[j]
        xp = np.asarray(propagate_state(t, c["x0"] + dx, *c["args"], **kw))[-1]
        xm = np.asarray(propagate_state(t, c["x0"] - dx, *c["args"], **kw))[-1]
        fd = (xp - xm) / (2 * steps[j])
        col_err.append(float(np.linalg.norm(phi[:, j] - fd) / np.linalg.norm(fd)))
    hk = 1e-3 * K2.K0
    kp = dict(kw, srp=replace(c["srp"], k_srp_m2_per_kg=K2.K0 + hk))
    km = dict(kw, srp=replace(c["srp"], k_srp_m2_per_kg=K2.K0 - hk))
    fdk = (np.asarray(propagate_state(t, c["x0"], *c["args"], **kp))[-1]
           - np.asarray(propagate_state(t, c["x0"], *c["args"], **km))[-1]) / (2 * hk)
    k_err = float(np.linalg.norm(s_k - fdk) / np.linalg.norm(fdk))
    worst = max(col_err + [k_err])
    return dict(stm_column_relative_error=col_err, dxdk_relative_error=k_err, max=worst, PASS=worst <= 1e-5)


def run_closure(spec) -> dict:
    estimator, reference = spec
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import PropagationProfile
    c = D1._config("CANON")
    t = c["t"]
    truth = np.asarray(reference, float)
    X, PHI, S = K2.propagate_segmented(c["scn"], c["x0"], np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(c["scn"], X, PHI, S, P.K.stations())
    rb = fx.rb
    obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                        rb.earth.vel, c["scn"].et0, model=R.DSN_C2_LIGHT_TIME_MODEL,
                                                        noise=False)
    start = c["x0"] + np.random.default_rng([98, 0, 0]).normal(0.0, K2.PRIOR_SD)
    fn, max_iter = {"BLS_LM": (E.estimate_two_way_range_bls_lm, 80), "SRIF": (E.estimate_two_way_range_srif, 40)}[estimator]
    ts = time.time()
    xh, status, st = fn(t, obs, start, pg, *c["args"], srp=c["srp"], j2_moon=c["j2"], **TOL, max_iter=max_iter,
                        propagation_profile=PropagationProfile())
    xh = np.asarray(xh, float)
    dr = float(np.linalg.norm(xh[:3] - c["x0"][:3]))
    return dict(estimator=estimator, status=status, detail=st.termination_detail, dr_m=dr,
                dv_mps=float(np.linalg.norm(xh[3:] - c["x0"][3:])), iterations=st.iterations,
                final_cost=st.final_cost, elapsed_s=time.time() - ts,
                PASS=status in ("Converged", "J-Stab") and dr <= 1e-3)


def g8_refreeze() -> dict:
    files = ["src/lunar_od/dynamics.py", "src/lunar_od/srp.py", "src/lunar_od/estimators.py"]
    sha = {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in files}
    manifest = (REPO / "docs/development/path-migration.json").read_text(encoding="utf-8")
    recorded = {f: (h in manifest) for f, h in sha.items()}
    return dict(sha256=sha, recorded_in_path_migration=recorded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-closure", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        f_g1 = ex.submit(g1_default_invariance)
        f_g6 = ex.submit(g6_variational)
        orbits = list(ex.map(run_orbit, D1.ORBITS))
        ref = next(o["reference"] for o in orbits if o["orbit"] == "CANON")
        closure = [] if a.skip_closure else list(ex.map(run_closure, [("BLS_LM", ref), ("SRIF", ref)]))
        g1, g6 = f_g1.result(), f_g6.result()
    for o in orbits:
        o.pop("reference", None)
    g8 = g8_refreeze()
    gates = dict(
        G1=g1["PASS"],
        G3=all(v <= 1e-3 for o in orbits for v in o["G3_max_error_m"].values()),
        G4=all(o["G4_contact_dt_max_s"] <= 1e-3 and o["n_contacts"] == o["n_reference_events"] for o in orbits),
        G5=all(v <= 1e-3 for o in orbits for v in o["G5_mismatch_m"].values()),
        G6=g6["PASS"],
        G7=bool(closure) and all(r["PASS"] for r in closure),
        G8=all(g8["recorded_in_path_migration"].values()),
    )
    out = dict(predeclaration="studies/phase17/configs/phase_d2_propagation_predeclaration.json",
               G1=g1, orbits=orbits, G6=g6, closure=closure, G8=g8, gates=gates, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(gates=gates, G1=g1, G6=g6, closure=closure,
                                     orbits=[{k: o[k] for k in ("orbit", "n_contacts", "n_reference_events",
                                                                "G3_max_error_m", "G4_contact_dt_max_s",
                                                                "G5_mismatch_m", "timing_s")} for o in orbits])),
                     indent=1))


if __name__ == "__main__":
    main()
