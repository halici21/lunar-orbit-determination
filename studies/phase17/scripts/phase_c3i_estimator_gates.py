"""Phase C3-I gate G5: estimator closure with the station displacement (BLS-LM and SRIF from the truth start).

Predeclared in studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json (G5). CANON k0, KP2
SIMPLIFIED dynamics, range only, six-state, noise-free observations from
relativistic_light_time.generate_two_way_range_measurements_ssb under DSN_C2_LIGHT_TIME_MODEL, generated and
estimated (a) without and (b) with station_displacement {solid_tide, ITRF2020 coordinates}. Same status and
|x_hat_b - x_hat_a| <= 1 mm in position. Writes studies/phase17/results/phase_c3i_estimator_gates.json.

    python studies/phase17/scripts/phase_c3i_estimator_gates.py
"""
from __future__ import annotations

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

OUT = P.ARTIFACTS / "phase_c3i_estimator_gates.json"
DISPLACEMENT = dict(solid_tide=True, coordinate_frame="ITRF2020", include_scale=True)


def run(spec) -> dict:
    with_disp, estimator = spec
    import phase17_geo_core as G
    import lunar_od.estimators as E
    from lunar_od import relativistic_light_time as R
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions

    P.K._init()
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, _, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rb, t = fx.rb, scn.t_grid
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]),
            lambda tt: G.sun_mci(scn.et0 + float(tt)))
    kw = dict(j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-11, atol=1e-12)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    truth = np.asarray(propagate_state(t, x0, *args, srp=srp, **kw), float)
    obs, pg = R.generate_two_way_range_measurements_ssb(t, truth, tuple(fx.stations), rb.vis, rb.earth.pos,
                                                        rb.earth.vel, scn.et0, model=R.DSN_C2_LIGHT_TIME_MODEL,
                                                        noise=False,
                                                        station_displacement=DISPLACEMENT if with_disp else None)
    fn = {"BLS_LM": E.estimate_two_way_range_bls_lm, "SRIF": E.estimate_two_way_range_srif}[estimator]
    ts = time.time()
    xh, status, st = fn(t, obs, x0.copy(), pg, *args, srp=srp, **kw)
    xh = np.asarray(xh, float)
    return dict(displacement=with_disp, estimator=estimator, status=status, detail=st.termination_detail,
                x_hat=xh.tolist(), dr_m=float(np.linalg.norm(xh[:3] - x0[:3])), iterations=st.iterations,
                elapsed_s=time.time() - ts, n_obs=int(obs.shape[0]),
                metadata_has_displacement="station_displacement" in (pg.measurement_metadata or {}))


def main():
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        runs = list(ex.map(run, [(d, e) for d in (False, True) for e in ("BLS_LM", "SRIF")]))
    g5 = {}
    for est in ("BLS_LM", "SRIF"):
        a = next(r for r in runs if r["estimator"] == est and not r["displacement"])
        b = next(r for r in runs if r["estimator"] == est and r["displacement"])
        d = float(np.linalg.norm(np.asarray(b["x_hat"])[:3] - np.asarray(a["x_hat"])[:3]))
        g5[est] = dict(without=a, with_=b, x_hat_position_difference_m=d, same_status=a["status"] == b["status"],
                       passed=bool(d <= 1e-3 and a["status"] == b["status"] and b["metadata_has_displacement"]))
    out = dict(predeclaration="studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json",
               G5=dict(passed=all(v["passed"] for v in g5.values()), runs=g5), elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean({k: {kk: v[kk] for kk in ("x_hat_position_difference_m", "same_status", "passed")}
                                 for k, v in g5.items()}), indent=1))


if __name__ == "__main__":
    main()
