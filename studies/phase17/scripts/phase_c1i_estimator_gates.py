"""Phase C1-I: estimator gates G6 (closure on DSN-model truth) and G7 (nonlinear check of the C1 bias mapping).

Predeclared in studies/phase17/configs/phase_c1i_light_time_implementation_predeclaration.json.
Writes studies/phase17/results/phase_c1i_estimator_gates.json.

CANON k0, KP2 SIMPLIFIED dynamics (Moon point mass + J2 + cannonball SRP at K0), range only. The truth is
propagated with the estimator's own propagator and tolerances, so the gates see the light-time model and
nothing else. Observations come from relativistic_light_time.generate_two_way_range_measurements_ssb
(DSN model: SSB frame, Sun + Earth Shapiro, TT interval), noise-free.

    python studies/phase17/scripts/phase_c1i_estimator_gates.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase_c1i_estimator_gates.json"
CONTROL = "--moon-control" in sys.argv
if CONTROL:
    OUT = OUT.with_name("phase_c1i_estimator_gates_moon_control.json")


def main():
    import phase17_geo_core as G
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state, propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    from lunar_od.two_way_range import two_way_range_nominal_and_initial_jacobian

    t0 = time.time()
    P.K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())        # visibility, stations, Earth providers
    rb = fx.rb
    t = scn.t_grid
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]), lambda tt: G.sun_mci(scn.et0 + float(tt)))
    kw = dict(j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-11, atol=1e-12)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    truth = np.asarray(propagate_state(t, x0, *args, srp=srp, **kw), float)
    if CONTROL:     # post-hoc control: the historical Moon-centred model end to end (production generator)
        from lunar_od.two_way_range import generate_two_way_range_measurements
        obs, pg = generate_two_way_range_measurements(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                      rb.earth.vel, scn.et0, noise=False)
    else:
        obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                            rb.earth.vel, scn.et0, model=R.DSN_LIGHT_TIME_MODEL,
                                                            noise=False)
    rng = np.random.default_rng([98, 0, 0])
    start = x0 + rng.normal(0.0, K2.PRIOR_SD)

    # ---------------- G6: SSB estimator on SSB truth ----------------
    g6 = {}
    for name, fn in (("BLS_LM", E.estimate_two_way_range_bls_lm), ("SRIF", E.estimate_two_way_range_srif)):
        ts = time.time()
        xh, status, st = fn(t, obs, start, pg, *args, srp=srp, **kw)
        g6[name] = dict(status=status, detail=st.termination_detail,
                        dr_m=float(np.linalg.norm(np.asarray(xh)[:3] - x0[:3])),
                        dv_mps=float(np.linalg.norm(np.asarray(xh)[3:] - x0[3:])), iterations=st.iterations,
                        elapsed_s=time.time() - ts)
    g6_pass = all(v["dr_m"] <= 1e-3 and v["dv_mps"] <= 1e-6 and v["status"] == "Converged" for v in g6.values())

    if CONTROL:
        out = dict(G6_control=dict(runs=g6, n_obs=int(obs.shape[0]), profile=pg.measurement_model_profile),
                   elapsed_s=time.time() - t0)
        OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
        print(json.dumps(P.K._clean(out), indent=1))
        return

    # ---------------- G7: MOON estimator on SSB truth vs the linear bias mapping ----------------
    pg_moon = replace(pg, measurement_model_profile="two_way_light_time")
    p0 = np.diag(K2.PRIOR_SD ** 2)
    xh, status, st = E.estimate_two_way_range_bls_lm(t, obs, x0, pg_moon, *args, srp=srp, solve_for_k_srp=True,
                                                     k_srp_initial=K2.K0, prior_covariance=p0,
                                                     return_posterior=True, **kw)
    d_nl = np.concatenate([np.asarray(xh) - x0, [st.k_srp_estimate - K2.K0]])
    sig = np.sqrt(np.diag(np.asarray(st.posterior_covariance)))
    # linear prediction at the truth, same design, weights and prior
    aug = np.asarray(propagate_state_with_k_sensitivity(t, x0, *args, srp=srp, **kw), float)
    h_moon, hx = two_way_range_nominal_and_initial_jacobian(obs, pg_moon, aug[:, :42])
    hk = E._two_way_range_k_srp_column(obs, aug, hx)
    h = np.hstack([hx, hk[:, None]])
    w = E._two_way_range_weight_diagonal(obs, pg_moon)
    info = h.T @ (w[:, None] * h)
    info[:6, :6] += np.linalg.inv(p0)
    d_lin = np.linalg.solve(info, h.T @ (w * (obs[:, 1] - h_moon)))
    sig_lin = np.sqrt(np.diag(np.linalg.inv(info)))
    z_nl, z_lin = d_nl / sig, d_lin / sig_lin
    j = int(np.argmax(np.abs(z_lin)))
    g7_rel = abs(z_nl[j] - z_lin[j]) / abs(z_lin[j])
    out = dict(
        G6=dict(passed=bool(g6_pass), runs=g6, n_obs=int(obs.shape[0]), profile=pg.measurement_model_profile),
        G7=dict(passed=bool(g7_rel <= 0.10), status=status, detail=st.termination_detail,
                z_nonlinear=z_nl.tolist(), z_linear=z_lin.tolist(), largest_component=j,
                relative_difference_largest=float(g7_rel), max_abs_z_nonlinear=float(np.max(np.abs(z_nl))),
                range_bias_m=dict(min=float(np.min(obs[:, 1] - h_moon)), max=float(np.max(obs[:, 1] - h_moon)))),
        elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(out), indent=1))


if __name__ == "__main__":
    main()
