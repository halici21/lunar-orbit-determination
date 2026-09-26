"""PHASE 17-F1 - physical-force / state-truth parity across the variational correction.

``--capture PATH`` (run on the ENTERING dynamics) and ``--compare PATH`` (run
after F1) propagate the same canonical initial state with the same force model,
K, Sun/Earth providers, shadow model and integrator settings through every
SRP-active entry point, plus the SRP acceleration at sampled states:

  STATE_ONLY    propagate_state (6-state, SRP)            -> must be bitwise equal
  ACCELERATION  srp_acceleration_with_lunar_shadow / kernel at 2000 states -> bitwise
  AUG42_FAST    propagate_augmented_state (numba RHS + SRP wrapper), state part
  AUG42_REF     same with the reference (ode_fun_v3) RHS, state part
  AUG48         propagate_state_with_k_sensitivity, state part

The augmented integrations share one adaptive integrator with the STM, so their
STATE part may move at the integrator-tolerance level when the variational RHS
changes (AUGMENTED_INTEGRATOR_NUMERICAL_PARITY).  That is measured, not assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run() -> dict:
    import lunar_od.dynamics as dyn
    import phase17_geo_core as G
    import phase17_r1m_core as M
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.srp import SRPOptions, srp_acceleration_kernel, srp_acceleration_with_lunar_shadow

    load_spice_kernels(None, clear=True)
    et0, t_orbit = M.campaign_epoch()
    x0 = M.campaign_initial_state()
    t = np.arange(0.0, 4.0 * t_orbit + 1.0, 90.0)
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    earth = lambda _t: np.array([384_400e3, 0.0, 0.0])  # noqa: E731  (GEO canonical force model)
    sun = lambda tt: G.sun_mci(et0 + float(tt))          # noqa: E731
    kw = dict(rtol=1e-12, atol=1e-13, j2_moon=J2_MOON_UNNORMALIZED, srp=srp)
    out = {"t": t}
    out["STATE_ONLY"] = dyn.propagate_state(t, x0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    out["AUG42_FAST"] = dyn.propagate_augmented_state(t, aug0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
    fast = dyn._FAST_DYNAMICS
    dyn._FAST_DYNAMICS = False
    try:
        out["AUG42_REF"] = dyn.propagate_augmented_state(t, aug0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
    finally:
        dyn._FAST_DYNAMICS = fast
    out["AUG48"] = dyn.propagate_state_with_k_sensitivity(t, x0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
    rng = np.random.default_rng(20260926)
    r = out["STATE_ONLY"][:, :3]
    pts = r[rng.integers(0, len(r), 2000)] + rng.normal(scale=2e4, size=(2000, 3))
    s = np.array([sun(tt) for tt in rng.uniform(0, t[-1], 2000)])
    out["ACCELERATION"] = np.array([srp_acceleration_with_lunar_shadow(p, q, srp) for p, q in zip(pts, s)])
    out["KERNEL"] = np.array([srp_acceleration_kernel(p, q, srp) for p, q in zip(pts, s)])
    return out


def convergence() -> dict:
    """Position error (max over the grid, m) of every entry point against a tight
    (rtol 3e-15) state-only reference, versus rtol: separates integrator
    tolerance behaviour from any physical change."""
    import lunar_od.dynamics as dyn
    import phase17_geo_core as G
    import phase17_r1m_core as M
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions

    et0, t_orbit = M.campaign_epoch()
    x0 = M.campaign_initial_state()
    t = np.arange(0.0, 4.0 * t_orbit + 1.0, 90.0)
    earth = lambda _t: np.array([384_400e3, 0.0, 0.0])  # noqa: E731
    sun = lambda tt: G.sun_mci(et0 + float(tt))          # noqa: E731
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    ref = dyn.propagate_state(t, x0, G.mu_moon(), 0.0, 0.0, earth, sun, srp=srp, rtol=3e-15, atol=3e-16,
                              j2_moon=J2_MOON_UNNORMALIZED)
    err = lambda a: float(np.max(np.linalg.norm(a[:, :3] - ref[:, :3], axis=1)))  # noqa: E731
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    out = {}
    for rtol in (1e-11, 1e-12, 1e-13, 1e-14):
        kw = dict(rtol=rtol, atol=rtol / 10, j2_moon=J2_MOON_UNNORMALIZED, srp=srp)
        a42 = dyn.propagate_augmented_state(t, aug0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
        a48 = dyn.propagate_state_with_k_sensitivity(t, x0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)
        out["%g" % rtol] = dict(
            state_only_m=err(dyn.propagate_state(t, x0, G.mu_moon(), 0.0, 0.0, earth, sun, **kw)),
            aug42_adams_m=err(a42), aug48_adams_m=err(a48),
            aug42_dop853_m=err(dyn.propagate_augmented_state(t, aug0, G.mu_moon(), 0.0, 0.0, earth, sun,
                                                             method="DOP853", **kw)),
            phi42_vs_phi48_rel=float(np.max(np.abs(a42[:, 6:42] - a48[:, 6:42])) / np.max(np.abs(a48[:, 6:42]))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture")
    g.add_argument("--compare")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run()
    if a.capture:
        np.savez_compressed(a.capture, **res)
        print("captured", a.capture)
        return
    ref = np.load(a.compare)
    rep = {}
    for key in ("STATE_ONLY", "ACCELERATION", "KERNEL", "AUG42_FAST", "AUG42_REF", "AUG48"):
        x, y = ref[key], res[key]
        cols = slice(0, 6) if key.startswith("AUG") else slice(None)
        dx = np.abs(x[:, cols] - y[:, cols])
        rep[key] = dict(bitwise_equal=bool(np.array_equal(x[:, cols], y[:, cols])),
                        max_abs_pos_m=float(dx[:, :3].max()) if key != "KERNEL" and key != "ACCELERATION" else None,
                        max_abs=float(dx.max()),
                        max_rel=float((dx / np.maximum(np.abs(x[:, cols]), 1e-300)).max()))
    rep["augmented_integrator_convergence"] = convergence()
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
