"""Phase D2 diagnostics for the gates that failed as computed (G4, G6, G7 SRIF); not gates."""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase_d1_integrator as D1  # noqa: E402

K2 = D1.K2
OUT = D1.P.ARTIFACTS / "phase_d2_diagnostics.json"


def g4(orbit="O1_LOW_BETA_ECLIPSE_RICH") -> dict:
    """Contact epochs: production at two pre-pass tolerances vs the bisection reference at two grids."""
    from lunar_od.dynamics import propagate_state, shadow_contact_times
    from lunar_od.srp import apparent_radii_and_separation
    c = D1._config(orbit)
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"])
    prod = {r: np.array(shadow_contact_times(t, c["x0"], *c["args"], **kw, rtol=r, atol=r * 0.1))
            for r in (1e-11, 1e-13)}
    ref = {}
    for h in (10.0, 2.0):
        dense_t = np.arange(t[0], t[-1] + 1e-9, h)
        prelim = np.asarray(propagate_state(dense_t, c["x0"], *c["args"], method="DOP853", **kw,
                                            rtol=1e-12, atol=1e-13))
        ref[h] = np.array(sorted(e["t"] for e in D1._events(c, dense_t, prelim)))
    i = int(np.argmax(np.abs(prod[1e-11] - ref[10.0])))
    # contact function at the two candidate epochs on a tight trajectory
    tt = np.array([t[0], prod[1e-11][i], ref[10.0][i]])
    x = np.asarray(propagate_state(tt, c["x0"], *c["args"], method="DOP853", shadow_events=True, **kw,
                                   rtol=1e-13, atol=1e-14))
    g = []
    for k in (1, 2):
        a_s, a_o, gam, _, _ = apparent_radii_and_separation(x[k, :3], c["args"][4](tt[k]), np.zeros(3),
                                                            c["srp"].occulting_body_radius_m)
        g.append(dict(outer=gam - (a_s + a_o), inner=gam - abs(a_s - a_o)))
    return dict(orbit=orbit, worst_index=i,
                prod_1e11_minus_ref10_max_s=float(np.max(np.abs(prod[1e-11] - ref[10.0]))),
                prod_1e13_minus_ref10_max_s=float(np.max(np.abs(prod[1e-13] - ref[10.0]))),
                prod_1e11_minus_ref2_max_s=float(np.max(np.abs(prod[1e-11] - ref[2.0]))),
                prod_1e11_minus_prod_1e13_max_s=float(np.max(np.abs(prod[1e-11] - prod[1e-13]))),
                ref10_minus_ref2_max_s=float(np.max(np.abs(ref[10.0] - ref[2.0]))),
                contact_function_rad_at_prod_and_ref=g)


def g6(rtol: float) -> dict:
    """dx/dK at the final epoch vs central differences, both at the given tolerance."""
    from lunar_od.dynamics import PropagationProfile, propagate_state, propagate_state_with_k_sensitivity
    c = D1._config("CANON")
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"], rtol=rtol, atol=rtol * 0.1, **PropagationProfile().propagator_kwargs())
    s_k = np.asarray(propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw))[-1, 42:48]
    out = {}
    for rel in (1e-3, 1e-2):
        hk = rel * K2.K0
        fd = (np.asarray(propagate_state(t, c["x0"], *c["args"], **(kw | dict(srp=replace(c["srp"], k_srp_m2_per_kg=K2.K0 + hk)))))[-1]
              - np.asarray(propagate_state(t, c["x0"], *c["args"], **(kw | dict(srp=replace(c["srp"], k_srp_m2_per_kg=K2.K0 - hk)))))[-1]) / (2 * hk)
        out[f"step_{rel:g}K0"] = float(np.linalg.norm(s_k - fd) / np.linalg.norm(fd))
    return dict(rtol=rtol, dxdk_relative_error=out, dxdk_position_norm_m_per_K=float(np.linalg.norm(s_k[:3])))


def main():
    with ProcessPoolExecutor(4) as ex:
        f4 = ex.submit(g4)
        f6 = [ex.submit(g6, r) for r in (1e-11, 1e-13)]
        out = dict(G4=f4.result(), G6=[f.result() for f in f6])
    OUT.write_text(json.dumps(D1.P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(D1.P.K._clean(out), indent=1))


if __name__ == "__main__":
    main()
