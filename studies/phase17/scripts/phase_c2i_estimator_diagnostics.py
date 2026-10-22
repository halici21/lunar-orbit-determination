"""Phase C2-I: POST-HOC diagnostics for G6/G7 (informational; they do not change any gate verdict).

Written after phase_c2i_estimator_gates.py reported G6 FAIL as computed and a G7 range change of +-1.8 m
(the C2 survey bounds the station-time change by ~5 cm). Writes
studies/phase17/results/phase_c2i_estimator_diagnostics.json.

  D1  split of G7's data-minus-model: (a) C2-truth minus C1-I-truth observations, both from the generator on
      the same propagate_state truth (the station-time change alone); (b) the C1-I model on the
      propagate_state_with_k_sensitivity trajectory minus the C1-I observations on the propagate_state truth
      (the propagator mismatch the G7 construction inherits from C1-I G7).
  D2  closure from the TRUTH start for both models (BLS-LM and SRIF, six-state): how far each estimator walks
      away from a consistent solution - the estimator/integrator floor that G6's comparison sits on.

    python studies/phase17/scripts/phase_c2i_estimator_diagnostics.py
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
import phase_c2i_estimator_gates as EG  # noqa: E402

OUT = P.ARTIFACTS / "phase_c2i_estimator_diagnostics.json"


def d1(_=None) -> dict:
    import lunar_od.estimators as E  # noqa: F401
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    c2, c1 = EG._setup("C2"), EG._setup("C1I")
    a = c2["obs"][:, 1] - c1["obs"][:, 1]
    aug = np.asarray(propagate_state_with_k_sensitivity(c1["t"], c1["x0"], *c1["args"], srp=c1["srp"], **c1["kw"]),
                     float)
    h_aug, _ = R.two_way_range_nominal_and_initial_jacobian_ssb(c1["obs"], c1["pg"], aug[:, :42])
    b = h_aug - c1["obs"][:, 1]
    traj = np.max(np.linalg.norm(aug[:, :3] - np.asarray(
        __import__("lunar_od.dynamics", fromlist=["propagate_state"]).propagate_state(
            c1["t"], c1["x0"], *c1["args"], srp=c1["srp"], **c1["kw"]), float)[:, :3], axis=1))
    return dict(station_time_change_m=dict(min=float(a.min()), max=float(a.max())),
                propagator_mismatch_range_m=dict(min=float(b.min()), max=float(b.max())),
                propagator_mismatch_position_max_m=float(traj))


def d2(spec) -> dict:
    model_name, estimator = spec
    import lunar_od.estimators as E
    s = EG._setup(model_name)
    fn = {"BLS_LM": E.estimate_two_way_range_bls_lm, "SRIF": E.estimate_two_way_range_srif}[estimator]
    ts = time.time()
    xh, status, st = fn(s["t"], s["obs"], s["x0"].copy(), s["pg"], *s["args"], srp=s["srp"], **s["kw"])
    xh = np.asarray(xh, float)
    return dict(model=model_name, estimator=estimator, status=status, detail=st.termination_detail,
                dr_m=float(np.linalg.norm(xh[:3] - s["x0"][:3])), dv_mps=float(np.linalg.norm(xh[3:] - s["x0"][3:])),
                iterations=st.iterations, elapsed_s=time.time() - ts)


def main():
    t0 = time.time()
    with ProcessPoolExecutor(5) as ex:
        f1 = ex.submit(d1)
        runs = list(ex.map(d2, [(m, e) for m in ("C1I", "C2") for e in ("BLS_LM", "SRIF")]))
        r1 = f1.result()
    out = dict(post_hoc=True, D1=r1, D2=runs, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps(P.K._clean(out), indent=1))


if __name__ == "__main__":
    main()
