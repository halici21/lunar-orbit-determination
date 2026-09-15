"""PHASE 17-R1COV - COV-G, production requalification (s43-s48).

Runs the REAL production estimators after the repair and checks:

  s43  point estimates are unchanged versus the pre-repair code
  s44  G0/G1 formal sigma_K now agrees with the exact oracle
  s45  the prior diagnostic responds correctly after repair
  s48  the best existing linked-span case, with a trustworthy uncertainty

The point-estimate check imports the PRE-REPAIR estimators module from git as a
separate package, so the comparison is against the code as it actually was, not
against a remembered number.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from phase17_r1cov_core import (  # noqa: E402
    K_INDEX, exact_covariance, relative_error, scale_matrix,
)

ARTIFACTS = REPO / "artifacts"
SCRATCH = Path(r"C:/Users/erayh/AppData/Local/Temp/claude/"
               r"c--Users-erayh-Documents-Python-Grad/"
               r"31d33312-f3ed-4a57-8c4f-99d981791301/scratchpad")
BASELINE_REF = "HEAD"   # the pre-R1COV commit; nothing is committed yet


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def load_pre_repair_estimators():
    """Load the pre-repair estimators.py as a sibling module of the REAL package.

    Copying the whole lunar_od tree does not work: the copy defines its own
    TwoWayRangeConfig, PassGeometry and so on, so objects built by the real
    package fail the copy's isinstance checks and the estimator refuses them.
    The classes must be shared, and only estimators.py may differ.

    Binding __package__ to "lunar_od" makes the old module's relative imports
    ("from .dynamics import ...") resolve to the real, already-imported
    submodules, so the two estimator versions run against byte-identical
    physics, measurement and dynamics code -- which is precisely the
    comparison s43 requires.
    """
    import importlib.util

    import lunar_od  # noqa: F401  (ensures the real package is initialised)

    src = subprocess.run(
        ["git", "-C", str(REPO), "show", "%s:lunar_od/estimators.py" % BASELINE_REF],
        capture_output=True, text=True, check=True).stdout
    name = "lunar_od._prerepair_estimators"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "lunar_od"
    mod.__file__ = str(REPO / "lunar_od" / "estimators.py") + " @" + BASELINE_REF
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from phase17_r1m_core import build_range_arc, campaign_epoch

    load_spice_kernels(None, clear=True)
    _, t_orbit = campaign_epoch()

    # ==================================================================
    hdr("s43  POINT-ESTIMATE INVARIANCE  (pre-repair code vs repaired code)")
    old_est = load_pre_repair_estimators()
    import lunar_od.estimators as new_est
    print("  pre-repair module : %s" % old_est.__file__)
    print("  repaired module   : %s" % new_est.__file__)
    print("  pre-repair has square-root helper: %s"
          % hasattr(old_est, "_square_root_covariance_from_design"))
    print("  repaired  has square-root helper: %s"
          % hasattr(new_est, "_square_root_covariance_from_design"))

    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.srp import SRPOptions
    import spiceypy as spice
    et0, _ = campaign_epoch()

    def sun_at(t_s):
        return spice.spkezr("SUN", et0 + float(t_s), "J2000", "NONE",
                            "MOON")[0][:3] * 1000.0

    def earth_at(_t):
        return np.array([384_400e3, 0.0, 0.0])

    from lunar_od.constants import J2_MOON_UNNORMALIZED
    arc = build_range_arc(0.0, 2.0 * t_orbit, label="G1")
    x_guess = arc.x_true[0] + np.array([25.0, -15.0, 10.0, 0.0, 0.0, 0.0])
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    t_local = arc.t_grid - arc.t_grid[0]

    common = dict(max_iter=12, rtol=1e-12, atol=1e-13, srp=srp,
                  solve_for_k_srp=True, k_srp_initial=0.012,
                  j2_moon=J2_MOON_UNNORMALIZED)
    inv_rows = []
    for name, mod in (("pre_repair", old_est), ("repaired", new_est)):
        x_hat, _, stats = mod.estimate_two_way_range_bls_lm(
            t_local, arc.obs, x_guess, arc.pass_geo, C.MU, 0.0, 0.0,
            lambda t: earth_at(t), lambda t: sun_at(t),
            return_posterior=True, **common)
        inv_rows.append(dict(
            variant=name, k_estimate=float(stats.k_srp_estimate),
            final_cost=float(stats.final_cost),
            iterations=int(stats.iterations),
            state=x_hat.copy(),
            sigma_k=float(np.sqrt(stats.posterior_covariance[K_INDEX, K_INDEX]))))
    a, b = inv_rows
    k_same = a["k_estimate"] == b["k_estimate"]
    cost_same = a["final_cost"] == b["final_cost"]
    iter_same = a["iterations"] == b["iterations"]
    state_same = bool(np.array_equal(a["state"], b["state"]))
    print("\n  K estimate   pre %.17e  post %.17e   bitwise equal: %s"
          % (a["k_estimate"], b["k_estimate"], k_same))
    print("  final cost   pre %.17e  post %.17e   bitwise equal: %s"
          % (a["final_cost"], b["final_cost"], cost_same))
    print("  iterations   pre %d  post %d   equal: %s"
          % (a["iterations"], b["iterations"], iter_same))
    print("  state vector bitwise equal: %s  (max |diff| = %.3e)"
          % (state_same, float(np.max(np.abs(a["state"] - b["state"])))))
    print("\n  sigma_K      pre %.10e  post %.10e   ratio %.2fx"
          % (a["sigma_k"], b["sigma_k"], b["sigma_k"] / a["sigma_k"]))
    print("  -> the ESTIMATE is untouched; only the UNCERTAINTY changed.")
    point_ok = k_same and cost_same and iter_same and state_same
    print("\n  POINT_ESTIMATE_INVARIANCE_GATE = %s" % ("PASS" if point_ok else "FAIL"))

    # ==================================================================
    hdr("s44/s45  G0/G1/G3 REQUALIFICATION AND PRIOR DIAGNOSTIC")
    req_rows = []
    requal_ok = True
    prior_ok = True
    scale = scale_matrix()
    for label, orbits in (("G0", 1.3), ("G1", 2.0), ("G3", 5.0)):
        a_arc = build_range_arc(0.0, orbits * t_orbit, label=label)
        h = np.hstack([a_arc.h_x0, a_arc.h_k[:, None]])
        for pname, psig in (("data_only", None), ("broad_k_sigma_1.0", 1.0),
                            ("moderate_k_sigma_0.01", 0.01)):
            pinv = None
            if psig is not None:
                pinv = np.zeros((7, 7))
                pinv[K_INDEX, K_INDEX] = 1.0 / psig ** 2
            cov_new, sqrt_new = new_est._square_root_covariance_from_design(
                h, a_arc.w, pinv if pinv is not None else np.zeros((7, 7)), scale)
            sig_new = float(np.sqrt(cov_new[K_INDEX, K_INDEX]))
            info_old = h.T @ (a_arc.w[:, None] * h) + (
                pinv if pinv is not None else np.zeros((7, 7)))
            is_ = scale.T @ info_old @ scale
            cov_old = scale @ old_est._safe_covariance_from_information(is_) @ scale.T
            sig_old = float(np.sqrt(cov_old[K_INDEX, K_INDEX]))
            _, sig_ex = exact_covariance(h, a_arc.w, pinv)
            rel = relative_error(sig_new, sig_ex)
            requal_ok = requal_ok and rel < 1e-6
            req_rows.append(dict(
                case=label, prior=pname, old_sigma_k=sig_old,
                qualified_sigma_k=sig_new, oracle_sigma_k=sig_ex,
                qualified_relative_error=rel,
                old_overconfidence_factor=sig_ex / sig_old,
                sigma_k_over_k_truth=sig_new / 0.01))
        rows = {r["prior"]: r for r in req_rows if r["case"] == label}
        print("\n  %s" % label)
        for pname in ("data_only", "broad_k_sigma_1.0", "moderate_k_sigma_0.01"):
            r = rows[pname]
            print("    %-22s old %.6e -> qualified %.6e   (oracle %.6e, rel %.1e)"
                  % (pname, r["old_sigma_k"], r["qualified_sigma_k"],
                     r["oracle_sigma_k"], r["qualified_relative_error"]))
        d_new = relative_error(rows["broad_k_sigma_1.0"]["qualified_sigma_k"],
                               rows["moderate_k_sigma_0.01"]["qualified_sigma_k"])
        d_old = relative_error(rows["broad_k_sigma_1.0"]["old_sigma_k"],
                               rows["moderate_k_sigma_0.01"]["old_sigma_k"])
        print("    broad-vs-moderate separation:  old %.2e   qualified %.2e" % (d_old, d_new))
        prior_ok = prior_ok and d_new > 1e-3
    print("\n  PRIOR_DIAGNOSTIC_POST_REPAIR_GATE = %s" % ("PASS" if prior_ok else "FAIL"))

    # ==================================================================
    hdr("s48  BEST EXISTING RADIOMETRIC CASE, WITH A TRUSTWORTHY UNCERTAINTY")
    print("  R1M's best configuration was M3 Model B: five 0.5-orbit windows")
    print("  linked across a 5.50-orbit span. Recomputed here with the")
    print("  qualified covariance rather than an information-domain proxy.\n")
    window = 0.5 * t_orbit
    starts = [0.50, 1.50, 2.50, 3.50, 5.50]
    stations = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
    long_arc = build_range_arc(0.0, (starts[-1] + 0.5) * t_orbit,
                               label="M3_long", station_filter=stations)
    t_obs = np.asarray(long_arc.obs[:, 0], float)
    keep = np.zeros(t_obs.shape, dtype=bool)
    for s0 in starts:
        keep |= ((t_obs >= s0 * t_orbit - 1e-6)
                 & (t_obs <= s0 * t_orbit + window + 1e-6))
    h_m3 = np.hstack([long_arc.h_x0[keep], long_arc.h_k[keep][:, None]])
    w_m3 = long_arc.w[keep]
    cov_m3, _ = new_est._square_root_covariance_from_design(
        h_m3, w_m3, np.zeros((7, 7)), scale)
    sig_m3 = float(np.sqrt(cov_m3[K_INDEX, K_INDEX]))
    _, sig_m3_ex = exact_covariance(h_m3, w_m3, None)
    info_m3 = h_m3.T @ (w_m3[:, None] * h_m3)
    is_m3 = scale.T @ info_m3 @ scale
    sig_m3_old = float(np.sqrt(
        (scale @ old_est._safe_covariance_from_information(is_m3) @ scale.T)[K_INDEX, K_INDEX]))
    frac = sig_m3 / 0.01
    print("  observations used            : %d" % int(keep.sum()))
    print("  OLD reported sigma_K         : %.6e" % sig_m3_old)
    print("  QUALIFIED sigma_K            : %.6e  (oracle %.6e, rel %.1e)"
          % (sig_m3, sig_m3_ex, relative_error(sig_m3, sig_m3_ex)))
    print("  K_truth                      : 1.000000e-02")
    print("  fractional uncertainty       : %.1f%%  (%.2f sigma separation from zero)"
          % (100.0 * frac, 1.0 / frac))
    print("  old path understated it by   : %.2fx" % (sig_m3 / sig_m3_old))

    with (ARTIFACTS / "r1cov_requalification.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(req_rows[0].keys()))
        w_.writeheader(); w_.writerows(req_rows)
    summary = dict(
        point_estimate_invariance_gate="PASS" if point_ok else "FAIL",
        prior_diagnostic_post_repair_gate="PASS" if prior_ok else "FAIL",
        requalification_matches_oracle="PASS" if requal_ok else "FAIL",
        g0_qualified_sigma_k=[r for r in req_rows
                              if r["case"] == "G0" and r["prior"] == "data_only"][0]["qualified_sigma_k"],
        g1_qualified_sigma_k=[r for r in req_rows
                              if r["case"] == "G1" and r["prior"] == "data_only"][0]["qualified_sigma_k"],
        best_existing_sigma_k=sig_m3,
        best_existing_fractional=frac,
        best_existing_old_sigma_k=sig_m3_old,
        best_existing_observations=int(keep.sum()),
        pre_repair_k_estimate=a["k_estimate"], repaired_k_estimate=b["k_estimate"],
        pre_repair_sigma_k=a["sigma_k"], repaired_sigma_k=b["sigma_k"],
    )
    (ARTIFACTS / "r1cov_requalification.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print("\n  wrote r1cov_requalification.csv / .json")
    print("\n  SUMMARY")
    print("    POINT_ESTIMATE_INVARIANCE_GATE    = %s" % summary["point_estimate_invariance_gate"])
    print("    PRIOR_DIAGNOSTIC_POST_REPAIR_GATE = %s" % summary["prior_diagnostic_post_repair_gate"])
    print("    REQUALIFICATION_MATCHES_ORACLE    = %s" % summary["requalification_matches_oracle"])


if __name__ == "__main__":
    main()
