"""Phase F2k, part R: the F2h and F2j end points rescored at integrator rtol/atol 1e-10/1e-11 and 1e-12/1e-13.

Predeclaration: studies/phase17/configs/phase_f2k_tight_tolerance_predeclaration.json.  No refit: each stored
parameter vector is evaluated (chi^2 with the F2H priors, post-fit RMS, RIC vs LOLA, prediction day) at both
tolerances, one process per (point, tolerance).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records
import phase_f2j_g1_diagnostic as G  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2k_rescore.json"
TOLS = {"loose": (M.RTOL, M.ATOL), "tight": M.F2K_TOL}
POINTS = {"F2h": (M.OUT_F2H, M.F2E_FIT_DAYS, M.F2E_PREDICT_DAYS),
          "F2j": (M.OUT_F2J, M.F2J_FIT_DAYS, M.F2J_PREDICT_DAYS)}


def point_vector(run: dict) -> np.ndarray:
    return np.array(run["x0"] + [run["k_srp"]] + [run["biases_m"][str(k)] for k in M.PADS]
                    + [run["doppler_bias_mmps"] * 1e-3] + [run["empirical_mps2"][n] for n in M.EMP_NAMES], float)


def case(key):
    import warnings
    import spiceypy as spice
    warnings.simplefilter("ignore")
    name, tol = key
    t0 = time.time()
    M.set_tolerance(*TOLS[tol])
    path, fit_days, predict_days = POINTS[name]
    run = json.loads(path.read_text(encoding="utf-8"))["runs"]["S1"]
    p = point_vector(run)
    pb = M.Problem(300, doppler=True, days=fit_days, n_emp=len(M.EMP_NAMES))
    nr = len(pb.rows)
    res, _, _ = pb.residual(p, partials=False)
    ra, _ = G.aug(res, None, p)
    t_cmp = np.arange(float(pb.setup.t3.min()), float(pb.setup.t3.max()), 60.0)
    est = pb.propagate_p(p, t_eval=np.concatenate([[pb.t_epoch], t_cmp]))[1:]
    ref = np.array([spice.spkezr(M.F.LRO, pb.setup.et0 + t, "J2000", "NONE", "MOON")[0] for t in t_cmp]) * 1e3
    ric = M._ric(est, ref)
    pred = M.predict(pb, p, predict_days)
    return key, dict(chi2=float(ra @ ra), postfit_rms_m=float(np.sqrt(np.mean(res[:nr] ** 2))),
                     doppler_postfit_rms_mmps=float(np.sqrt(np.mean(res[nr:] ** 2)) * M.SIGMA_DOP / M.SIGMA_M * 1e3),
                     ric_rms_m=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(), prediction=pred,
                     stored=dict(postfit_rms_m=run["postfit_rms_m"],
                                 doppler_postfit_rms_mmps=run["doppler_postfit_rms_mmps"],
                                 ric_rms_m=run["ric_rms_m"], prediction_range_rms_m=run["prediction"]["range_rms_m"],
                                 prediction_doppler_rms_mmps=run["prediction"]["doppler_rms_mmps"]),
                     elapsed_s=time.time() - t0)


def main():
    keys = [(n, t) for n in POINTS for t in TOLS]
    with ProcessPoolExecutor(len(keys)) as ex:
        res = dict(ex.map(case, keys))
    out = dict(predeclaration="studies/phase17/configs/phase_f2k_tight_tolerance_predeclaration.json (R)",
               tolerances=TOLS, points={})
    for n in POINTS:
        lo, ti = res[(n, "loose")], res[(n, "tight")]
        s = lo["stored"]
        repro = max(abs(lo["postfit_rms_m"] / s["postfit_rms_m"] - 1),
                    abs(lo["doppler_postfit_rms_mmps"] / s["doppler_postfit_rms_mmps"] - 1),
                    abs(lo["prediction"]["range_rms_m"] / s["prediction_range_rms_m"] - 1),
                    abs(lo["prediction"]["doppler_rms_mmps"] / s["prediction_doppler_rms_mmps"] - 1))
        out["points"][n] = dict(loose=lo, tight=ti, loose_reproduces_stored=repro <= 1e-6,
                                loose_vs_stored_max_relative=repro,
                                tight_minus_loose=dict(
                                    chi2=ti["chi2"] - lo["chi2"],
                                    postfit_rms_m=ti["postfit_rms_m"] - lo["postfit_rms_m"],
                                    doppler_postfit_rms_mmps=ti["doppler_postfit_rms_mmps"]
                                    - lo["doppler_postfit_rms_mmps"],
                                    prediction_range_rms_m=ti["prediction"]["range_rms_m"]
                                    - lo["prediction"]["range_rms_m"],
                                    prediction_doppler_rms_mmps=ti["prediction"]["doppler_rms_mmps"]
                                    - lo["prediction"]["doppler_rms_mmps"]))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({n: {k: v for k, v in d.items() if k in ("loose_reproduces_stored", "loose_vs_stored_max_relative",
                                                              "tight_minus_loose")} for n, d in out["points"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
