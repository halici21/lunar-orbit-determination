"""Phase 17-HARDEN-BLS1: qualification of the repaired production BLS K_SRP path.

Modes (each writes one section of artifacts/phase17_bls1_qualification.json):

  --hf        the FROZEN KP2 production-BLS verification job (``job_bls``, unchanged:
              CANON k0 HF, T0, range only, scalar K, no prior, the declared a priori)
              run against the repaired estimator, plus two labelled diagnostics at the
              production solution: an independent Gauss-Newton stationarity check and
              the posterior sigma recomputed WITH the solve's harmonic field (isolates
              HARDEN-D25, which is NOT repaired here).
  --ablation  the same cross-check problem in SIMPLIFIED dynamics, with each BLS1 change
              removed in turn (source-level variants exec'd into a private namespace;
              the production module is not modified).

The production estimator is never started from the KP2 solution here.
"""
from __future__ import annotations

import argparse
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

OUT = P.ARTIFACTS / "phase17_bls1_qualification.json"
ABLATIONS = ("final", "no_square_root_step", "no_bound_active_step", "no_caller_tolerance_jacobian")


def _hf_args(scn):
    import od_gravity_covariance_campaign as C  # noqa: N811
    import phase17_geo_core as G
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2
    from phase17_r1od2_fidelity import NMAX
    args = (MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, lambda t: G.earth_state_mci(scn.et0 + float(t))[:3],
            lambda t: G.sun_mci(scn.et0 + float(t)))
    harm = dict(harmonic_model=C._M[NMAX], harmonic_epoch_et0=scn.et0)
    return args, harm


def _gauss_newton_at(t, obs, pg, x, k, srp, args, kw, harm):
    """Undamped GN step + formal sigmas at (x, k) by column-normalised lstsq/QR (independent of the LM code)."""
    from dataclasses import replace

    import lunar_od.estimators as E
    s = replace(srp, k_srp_m2_per_kg=k)
    aug = E.propagate_state_with_k_sensitivity(t, x, *args, srp=s, **kw, **harm, harmonic_stm_opt_in=True)
    _, hx = E.two_way_range_nominal_and_initial_jacobian(obs, pg, aug)
    hk = E._two_way_range_k_srp_column(obs, aug, hx)
    hist = E.propagate_state(t, x, *args, srp=s, **kw, **harm)
    r, _ = E.compute_two_way_range_residuals(hist, obs, pg)
    w = E._two_way_range_weight_diagonal(obs, pg)
    a = np.hstack([hx, hk[:, None]]) * np.sqrt(w)[:, None]
    c = np.linalg.norm(a, axis=0)
    step = np.linalg.lstsq(a / c, np.sqrt(w) * np.asarray(r), rcond=None)[0] / c
    r_inv = np.linalg.inv(np.linalg.qr(a / c, mode="r"))
    sd = np.sqrt(np.sum(r_inv ** 2, axis=1)) / c
    return step, sd, float(np.sum(w * np.asarray(r) ** 2))


def run_hf() -> dict:
    import lunar_od.estimators as E
    P.K._init()
    cap = {}
    real_solve, real_bls = K2.solve, E.estimate_two_way_range_bls_lm

    def solve_cap(*a, **k):
        cap["mine"] = real_solve(*a, **k)
        return cap["mine"]

    def bls_cap(*a, **k):
        t0 = time.time()
        out = real_bls(*a, **k)
        cap.update(bls_args=a, bls_kw=k, bls_out=out, bls_s=time.time() - t0)
        return out

    K2.solve, E.estimate_two_way_range_bls_lm = solve_cap, bls_cap
    try:
        frozen = P.job_bls(dict(x0=P.x0s()["CANON"]))           # the frozen job, unchanged
    finally:
        K2.solve, E.estimate_two_way_range_bls_lm = real_solve, real_bls
    gate = dict(estimate=max(map(abs, frozen["diff_over_sigma"])) <= 0.02,
                sigma=max(abs(x - 1) for x in frozen["sigma_ratio"]) <= 0.01)

    t, obs, x_ap, pg = cap["bls_args"][:4]
    kw_all = cap["bls_kw"]
    x_best, status, stats = cap["bls_out"]
    scn = P.K.scenario("CANON", P.x0s()["CANON"], 0)
    args, harm = _hf_args(scn)
    tol = dict(rtol=kw_all["rtol"], atol=kw_all["atol"], j2_moon=kw_all["j2_moon"])
    step, sd_ind, chi2 = _gauss_newton_at(t, obs, pg, np.asarray(x_best), stats.k_srp_estimate, kw_all["srp"],
                                          args, tol, harm)
    mine_sd = np.asarray(cap["mine"]["sd"])
    return dict(
        frozen_job_bls=frozen,
        frozen_gate_criteria=dict(estimate_within_0p02_sigma=gate["estimate"], sigma_within_1pct=gate["sigma"],
                                  PRODUCTION_BLS="PASS" if all(gate.values()) else "FAIL"),
        production=dict(status=status, termination_detail=stats.termination_detail, iterations=stats.iterations,
                        k=stats.k_srp_estimate, negative_k_trials=stats.k_srp_negative_step_rejections,
                        bound_active_steps=stats.k_srp_bound_active_steps, elapsed_s=cap["bls_s"],
                        x0_err_m=(np.asarray(x_best) - np.asarray(P.x0s()["CANON"])).tolist(),
                        apriori_err_m=(np.asarray(x_ap) - np.asarray(P.x0s()["CANON"])).tolist()),
        diagnostic_stationarity=dict(
            label="DIAGNOSTIC: undamped Gauss-Newton remainder at the production solution, harmonic field "
                  "included, column-normalised lstsq (independent of the LM code)",
            chi2=chi2, gn_step_over_sigma=(step / sd_ind).tolist()),
        diagnostic_d25=dict(
            label="DIAGNOSTIC (HARDEN-D25 not repaired): sigma at the production solution recomputed WITH the "
                  "solve's harmonic field; KP2 sigma / this sigma",
            sigma_ratio_with_harmonics=(mine_sd / sd_ind).tolist(),
            sigma_ratio_reported=frozen["sigma_ratio"]),
    )


def _variant_namespace(name: str) -> dict:
    import lunar_od.estimators as E
    src = Path(E.__file__).read_text(encoding="utf-8")

    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, (name, old[:70])
        src = src.replace(old, new)

    if name == "no_square_root_step":
        rep("_step_fn = _lm_step_square_root if solve_for_k_srp else _lm_step", "_step_fn = _lm_step")
    elif name == "no_bound_active_step":                      # the pre-BLS1 discard-and-raise-lambda path
        rep("                prior_rhs_x = prior_rhs_scaled[:nx]\n",
            "                x_nominal = x_best.copy()\n                k_current = k_best\n"
            "                lambda_damping *= 10.0\n                if lambda_damping > 1e12:\n"
            "                    stop_reason = 'DampingLimit'\n                    break\n                continue\n"
            "                prior_rhs_x = prior_rhs_scaled[:nx]\n")
    elif name == "no_caller_tolerance_jacobian":
        rep("            _atol_r, _atol_a = rtol, atol\n            x_aug_hist = propagate_state_with_k_sensitivity(",
            "            x_aug_hist = propagate_state_with_k_sensitivity(")
    ns = dict(E.__dict__)
    exec(compile(src, E.__file__, "exec"), ns)               # private copy; production module untouched
    return ns


def run_ablation(name: str) -> dict:
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions
    P.K._init()
    ns = _variant_namespace(name)
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, zr, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rng = np.random.default_rng([98, 0, 0])                    # the frozen job's draws
    dx = rng.normal(0.0, K2.PRIOR_SD)
    obs = np.array(fx.rb.obs, float, copy=True)
    obs[:, 1] = zr + rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]), lambda t: G.sun_mci(scn.et0 + float(t)))
    kw = dict(j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-11, atol=1e-12)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    t0 = time.time()
    xb, st, s = ns["estimate_two_way_range_bls_lm"](scn.t_grid, obs, x0 + dx, fx.rb.pass_geo, *args, srp=srp,
                                                     solve_for_k_srp=True, k_srp_initial=K2.K0, **kw)
    el = time.time() - t0
    step, sd, chi2 = _gauss_newton_at(scn.t_grid, obs, fx.rb.pass_geo, np.asarray(xb), s.k_srp_estimate, srp,
                                      args, kw, {})
    return dict(variant=name, status=st, k=s.k_srp_estimate, iterations=s.iterations,
                negative_k_trials=s.k_srp_negative_step_rejections, chi2=chi2,
                gn_step_over_sigma_max=float(np.max(np.abs(step / sd))),
                pos_err_m=float(np.linalg.norm(np.asarray(xb)[:3] - x0[:3])), elapsed_s=el)


def _save(key, value):
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(P.K._clean(data), indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    if a.ablation:
        with ProcessPoolExecutor(len(ABLATIONS)) as ex:
            rows = list(ex.map(run_ablation, ABLATIONS))
        _save("simplified_ablation", dict(
            label="DIAGNOSTIC: KP2 cross-check problem (same a priori and noise draws) in SIMPLIFIED dynamics, "
                  "one BLS1 change removed per row", rows=rows))
        print(json.dumps(rows, indent=1))
    if a.hf:
        out = run_hf()
        _save("high_fidelity", out)
        print(json.dumps({k: out[k] for k in ("frozen_gate_criteria", "production")}, indent=1, default=str))


if __name__ == "__main__":
    main()
