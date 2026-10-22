"""Phase C2-I: estimator gates G6 (closure no worse than the C1-I model) and G7 (nonlinear check of the bias
mapping of the station-time change).

Predeclared in studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json (G6 as amended
by A1). Writes studies/phase17/results/phase_c2i_estimator_gates.json.

The C1-I G6/G7 construction (phase_c1i_estimator_gates.py): CANON k0, KP2 SIMPLIFIED dynamics, range only,
truth propagated with the estimator's own propagator, noise-free observations from
relativistic_light_time.generate_two_way_range_measurements_ssb.

  G6  BLS-LM and SRIF, from the C1-I start (x0 + N(0, PRIOR_SD), seed [98, 0, 0]), on (a) C1-I-model truth
      with the C1-I model and (b) DSN_C2 truth with the DSN_C2 model; per estimator, |x_hat_b - x_hat_a| <= 1 mm
      in position and the same status (A1).
  G7  the C1-I-model estimator (six-state + K, prior, from the truth) on DSN_C2 truth vs the linear bias mapping
      of the same change at the truth: largest sigma-normalised component within 10 %.

    python studies/phase17/scripts/phase_c2i_estimator_gates.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase_c2i_estimator_gates.json"


def _setup(model_name: str):
    import phase17_geo_core as G
    from lunar_od import relativistic_light_time as R
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions

    P.K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb = fx.rb
    t = scn.t_grid
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]),
            lambda tt: G.sun_mci(scn.et0 + float(tt)))
    kw = dict(j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-11, atol=1e-12)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    truth = np.asarray(propagate_state(t, x0, *args, srp=srp, **kw), float)
    model = {"C1I": R.DSN_LIGHT_TIME_MODEL, "C2": R.DSN_C2_LIGHT_TIME_MODEL}[model_name]
    obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                        rb.earth.vel, scn.et0, model=model, noise=False)
    return dict(x0=x0, t=t, args=args, kw=kw, srp=srp, obs=obs, pg=pg)


def g6_run(spec) -> dict:
    model_name, estimator = spec
    import lunar_od.estimators as E
    s = _setup(model_name)
    rng = np.random.default_rng([98, 0, 0])
    start = s["x0"] + rng.normal(0.0, K2.PRIOR_SD)
    fn = {"BLS_LM": E.estimate_two_way_range_bls_lm, "SRIF": E.estimate_two_way_range_srif}[estimator]
    ts = time.time()
    xh, status, st = fn(s["t"], s["obs"], start, s["pg"], *s["args"], srp=s["srp"], **s["kw"])
    xh = np.asarray(xh, float)
    return dict(model=model_name, estimator=estimator, status=status, detail=st.termination_detail,
                x_hat=xh.tolist(), dr_m=float(np.linalg.norm(xh[:3] - s["x0"][:3])),
                dv_mps=float(np.linalg.norm(xh[3:] - s["x0"][3:])), iterations=st.iterations,
                elapsed_s=time.time() - ts, n_obs=int(s["obs"].shape[0]),
                station_time=s["pg"].measurement_metadata["light_time_model"]["station_time"])


def g7_run(_=None) -> dict:
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import propagate_state_with_k_sensitivity

    s = _setup("C2")
    obs, pg, x0 = s["obs"], s["pg"], s["x0"]
    meta = dict(pg.measurement_metadata)
    meta["light_time_model"] = R._model_metadata(R.DSN_LIGHT_TIME_MODEL)
    pg_c1 = replace(pg, measurement_metadata=meta)
    p0 = np.diag(K2.PRIOR_SD ** 2)
    ts = time.time()
    xh, status, st = E.estimate_two_way_range_bls_lm(s["t"], obs, x0, pg_c1, *s["args"], srp=s["srp"],
                                                     solve_for_k_srp=True, k_srp_initial=K2.K0, prior_covariance=p0,
                                                     return_posterior=True, **s["kw"])
    d_nl = np.concatenate([np.asarray(xh) - x0, [st.k_srp_estimate - K2.K0]])
    sig = np.sqrt(np.diag(np.asarray(st.posterior_covariance)))
    aug = np.asarray(propagate_state_with_k_sensitivity(s["t"], x0, *s["args"], srp=s["srp"], **s["kw"]), float)
    h_c1, hx = R.two_way_range_nominal_and_initial_jacobian_ssb(obs, pg_c1, aug[:, :42])
    hk = E._two_way_range_k_srp_column(obs, aug, hx)
    h = np.hstack([hx, np.asarray(hk)[:, None]])
    w = E._two_way_range_weight_diagonal(obs, pg_c1)
    info = h.T @ (w[:, None] * h)
    info[:6, :6] += np.linalg.inv(p0)
    d_lin = np.linalg.solve(info, h.T @ (w * (obs[:, 1] - h_c1)))
    sig_lin = np.sqrt(np.diag(np.linalg.inv(info)))
    z_nl, z_lin = d_nl / sig, d_lin / sig_lin
    j = int(np.argmax(np.abs(z_lin)))
    rel = abs(z_nl[j] - z_lin[j]) / abs(z_lin[j])
    return dict(passed=bool(rel <= 0.10), status=status, detail=st.termination_detail, z_nonlinear=z_nl.tolist(),
                z_linear=z_lin.tolist(), largest_component=j, relative_difference_largest=float(rel),
                range_change_m=dict(min=float(np.min(obs[:, 1] - h_c1)), max=float(np.max(obs[:, 1] - h_c1))),
                elapsed_s=time.time() - ts)


def main():
    t0 = time.time()
    specs = [(m, e) for m in ("C1I", "C2") for e in ("BLS_LM", "SRIF")]
    with ProcessPoolExecutor(5) as ex:
        f7 = ex.submit(g7_run)
        runs = list(ex.map(g6_run, specs))
        g7 = f7.result()
    g6 = {}
    for est in ("BLS_LM", "SRIF"):
        a = next(r for r in runs if r["model"] == "C1I" and r["estimator"] == est)
        b = next(r for r in runs if r["model"] == "C2" and r["estimator"] == est)
        dpos = float(np.linalg.norm(np.asarray(b["x_hat"])[:3] - np.asarray(a["x_hat"])[:3]))
        g6[est] = dict(c1i=a, c2=b, x_hat_position_difference_m=dpos,
                       dr_difference_m=b["dr_m"] - a["dr_m"], same_status=a["status"] == b["status"],
                       passed=bool(dpos <= 1e-3 and a["status"] == b["status"]))
    out = dict(predeclaration="studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json",
               G6=dict(passed=all(v["passed"] for v in g6.values()), runs=g6), G7=g7, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(dict(G6={k: {kk: v[kk] for kk in ("x_hat_position_difference_m", "same_status",
                                                                     "passed")} for k, v in g6.items()},
                                     G7={k: g7[k] for k in ("passed", "relative_difference_largest", "status")})),
                     indent=1))


if __name__ == "__main__":
    main()
