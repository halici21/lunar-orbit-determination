"""Phase D2 G1 baseline: default-path outputs recorded from the unmodified code.

Run once before any D2 edit; phase_d2_propagation.py later recomputes the same
calls on the modified code and requires bit identity.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase_d1_integrator as D1  # noqa: E402

OUT = D1.P.ARTIFACTS / "phase_d2_default_baseline.npz"


def default_outputs() -> dict:
    """The G1 calls, all with default propagation arguments."""
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import propagate_augmented_state, propagate_state, propagate_state_with_k_sensitivity
    c = D1._config("CANON")
    cn = D1._config("CANON", "NO_SHADOW")
    t = c["t"]
    kw = dict(srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12)
    aug0 = np.concatenate([c["x0"], np.eye(6).reshape(-1, order="F")])
    out = {
        "state": propagate_state(t, c["x0"], *c["args"], **kw),
        "augmented": propagate_augmented_state(t, aug0, *c["args"], **kw),
        "ksens": propagate_state_with_k_sensitivity(t, c["x0"], *c["args"], **kw),
        "state_dop853": propagate_state(t, c["x0"], *c["args"], method="DOP853", **kw),
        "state_no_srp": propagate_state(t, c["x0"], *c["args"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12),
        "augmented_no_srp": propagate_augmented_state(t, aug0, *c["args"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12),
        "state_no_shadow": propagate_state(t, c["x0"], *cn["args"], srp=cn["srp"], j2_moon=c["j2"]),
        "augmented_no_shadow": propagate_augmented_state(t, aug0, *cn["args"], srp=cn["srp"], j2_moon=c["j2"]),
    }
    truth = out["state"][:, :6]
    import phase17_kp2_core as K2
    X, PHI, S = K2.propagate_segmented(c["scn"], c["x0"], np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(c["scn"], X, PHI, S, D1.P.K.stations())
    rb = fx.rb
    obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                        rb.earth.vel, c["scn"].et0, model=R.DSN_C2_LIGHT_TIME_MODEL,
                                                        noise=False)
    start = c["x0"] + np.random.default_rng([98, 0, 0]).normal(0.0, K2.PRIOR_SD)
    for name, fn in (("bls", E.estimate_two_way_range_bls_lm), ("srif", E.estimate_two_way_range_srif)):
        xh, status, st = fn(t, obs, start, pg, *c["args"], srp=c["srp"], j2_moon=c["j2"], rtol=1e-11, atol=1e-12,
                            max_iter=3)
        out[f"{name}_x"] = np.asarray(xh, float)
        out[f"{name}_status"] = np.array(status)
        out[f"{name}_cost"] = np.array(st.final_cost)
    return {k: np.asarray(v) for k, v in out.items()}


if __name__ == "__main__":
    o = default_outputs()
    np.savez(OUT, **o)
    print({k: (v.shape, str(v) if v.ndim == 0 else "") for k, v in o.items()})
