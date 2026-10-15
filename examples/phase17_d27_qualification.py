"""Phase 17-HARDEN-D27: qualification of the BLS-LM convergence/status contract.

Writes artifacts/phase17_d27_qualification.json (never the BLS1 artifact).

  --simplified  the KP2 cross-check problem in SIMPLIFIED dynamics (the BLS1 ablation's data: same
                a priori, same noise draws) solved by the CURRENT production estimator:
                  apriori      declared a priori start, default tolerances: estimate must equal BLS1's
                               "final" row bitwise; status/detail recorded
                  truth_start  started at the truth state and K0 -- the BLS1 D27 fixture (was DampingLimit)
                  floor        a priori start with the step-norm / cost-stability tests disabled, so the
                               solve can only end at damping exhaustion (the D27 regime by construction)
                  bad_jacobian a priori start with the K_SRP Jacobian column sign-flipped (private module
                               copy): steps stop being descent directions far from the minimiser -- the
                               genuine-failure case; must stay DampingLimit
                Each row carries the production decrement and the BLS1 independent GN remainder in sigma.
  --hf          the production decrement at the STORED high-fidelity production solution of the frozen KP2
                job (artifacts/phase17_bls1_qualification.json, high_fidelity.production), with the job's
                exact data rebuilt (rng [98, 0, 0]); and at the declared a priori for contrast.
"""
from __future__ import annotations

import argparse
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

import phase17_bls1_qualification as B  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase17_d27_qualification.json"
BLS1 = P.ARTIFACTS / "phase17_bls1_qualification.json"
CASES = ("apriori", "truth_start", "floor", "bad_jacobian")


def _decrement_at(t, obs, pg, x, k, srp, args, kw, harm):
    """The production certificate (_gauss_newton_decrement) on the K path's own linearisation."""
    import lunar_od.estimators as E
    aug = E.propagate_state_with_k_sensitivity(t, x, *args, srp=replace(srp, k_srp_m2_per_kg=k), **kw,
                                               **({} if not harm else {**harm, "harmonic_stm_opt_in": True}))
    h_nom, hx = E.two_way_range_nominal_and_initial_jacobian(obs, pg, aug)
    hk = E._two_way_range_k_srp_column(obs, aug, hx)
    r = E._two_way_range_residual_from_h(obs, h_nom)
    w = E._two_way_range_weight_diagonal(obs, pg)
    return E._gauss_newton_decrement(np.hstack([hx, hk[:, None]]), r, w, np.eye(7), None, None)


def _namespace(case):
    import lunar_od.estimators as E
    if case != "bad_jacobian":
        return E.__dict__
    ns = dict(E.__dict__)
    src = Path(E.__file__).read_text(encoding="utf-8")
    old = "            h_initial = np.hstack([h_x0, h_k[:, None]])\n"
    first = src.index(old)                                  # the BLS-LM site precedes the SRIF one
    assert src.index("def estimate_two_way_range_bls_lm(") < first < src.index("def estimate_two_way_range_srif(")
    src = src[:first] + "            h_initial = np.hstack([h_x0, -h_k[:, None]])\n" + src[first + len(old):]
    exec(compile(src, E.__file__, "exec"), ns)              # private copy; production module untouched
    return ns


def run_simplified(case: str) -> dict:
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions
    P.K._init()
    ns = _namespace(case)
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), "SIMPLIFIED")
    fx, zr, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rng = np.random.default_rng([98, 0, 0])
    dx = rng.normal(0.0, K2.PRIOR_SD)
    obs = np.array(fx.rb.obs, float, copy=True)
    obs[:, 1] = zr + rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    args = (G.mu_moon(), 0.0, 0.0, lambda _t: np.array([384_400e3, 0.0, 0.0]), lambda t: G.sun_mci(scn.et0 + float(t)))
    kw = dict(j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-11, atol=1e-12)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    start = x0 if case == "truth_start" else x0 + dx
    extra = dict(tol_step_norm=0.0, tol_cost_stability=0.0) if case == "floor" else {}
    t0 = time.time()
    xb, st, s = ns["estimate_two_way_range_bls_lm"](scn.t_grid, obs, start, fx.rb.pass_geo, *args, srp=srp,
                                                     solve_for_k_srp=True, k_srp_initial=K2.K0, **kw, **extra)
    el = time.time() - t0
    step, sd, chi2 = B._gauss_newton_at(scn.t_grid, obs, fx.rb.pass_geo, np.asarray(xb), s.k_srp_estimate, srp,
                                        args, kw, {})
    return dict(case=case, status=st, termination_detail=s.termination_detail,
                production_decrement=s.gauss_newton_decrement,
                independent_decrement=_decrement_at(scn.t_grid, obs, fx.rb.pass_geo, np.asarray(xb),
                                                    s.k_srp_estimate, srp, args, kw, {}),
                gn_step_over_sigma_max=float(np.max(np.abs(step / sd))), chi2=chi2,
                k=s.k_srp_estimate, x=np.asarray(xb).tolist(), iterations=s.iterations,
                pos_err_m=float(np.linalg.norm(np.asarray(xb)[:3] - x0[:3])), elapsed_s=el)


def run_hf() -> dict:
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.srp import SRPOptions
    from phase17_r1od2_fidelity import NMAX
    P.K._init()
    stored = json.loads(BLS1.read_text(encoding="utf-8"))["high_fidelity"]["production"]
    x0 = np.asarray(P.x0s()["CANON"], float)
    scn = P.K.scenario("CANON", x0.tolist(), 0)
    X, PHI, S = K2.propagate_segmented(scn, x0, np.full(3, K2.K0), P.HF)   # exactly job_bls's data
    fx, zr, _ = K2.build_fixed(scn, X, PHI, S, P.K.stations())
    rng = np.random.default_rng([98, 0, 0])
    dx = rng.normal(0.0, K2.PRIOR_SD)
    obs = np.array(fx.rb.obs, float, copy=True)
    obs[:, 1] = zr + rng.normal(0.0, K2.SIGMA_RANGE_M, zr.size)
    args, harm = B._hf_args(scn)
    kw = dict(rtol=1e-12, atol=1e-13, j2_moon=0.0)
    srp = SRPOptions(k_srp_m2_per_kg=K2.K0, shadow_model=scn.shadow_model)
    assert harm["harmonic_model"] is C._M[NMAX]
    x_prod = x0 + np.asarray(stored["x0_err_m"], float)
    out = dict(stored_status=stored["status"], stored_detail=stored["termination_detail"],
               a_priori_matches_stored=bool(np.allclose(dx, np.asarray(stored["apriori_err_m"]), rtol=0, atol=1e-9)))
    t0 = time.time()
    out["decrement_at_production_solution"] = _decrement_at(scn.t_grid, obs, fx.rb.pass_geo, x_prod,
                                                             float(stored["k"]), srp, args, kw, harm)
    out["decrement_at_a_priori"] = _decrement_at(scn.t_grid, obs, fx.rb.pass_geo, x0 + dx, K2.K0, srp, args,
                                                 kw, harm)
    out["elapsed_s"] = time.time() - t0
    import lunar_od.estimators as E
    out["status_under_d27_rule"] = ("Converged/STATIONARY_AT_DAMPING_LIMIT"
                                    if out["decrement_at_production_solution"] <= E.STATIONARITY_DECREMENT_TOL
                                    else "DampingLimit/LAMBDA_HARD_LIMIT")
    return out


def _save(key, value):
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(P.K._clean(data), indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simplified", action="store_true")
    ap.add_argument("--hf", action="store_true")
    a = ap.parse_args()
    if a.simplified:
        with ProcessPoolExecutor(len(CASES)) as ex:
            rows = list(ex.map(run_simplified, CASES))
        bls1 = json.loads(BLS1.read_text(encoding="utf-8"))["simplified_ablation"]["rows"][0]
        _save("simplified", dict(
            label="KP2 cross-check problem, SIMPLIFIED dynamics, current production estimator",
            bls1_final_row_k=bls1["k"], apriori_k_bitwise_equal_to_bls1=rows[0]["k"] == bls1["k"], rows=rows))
        print(json.dumps([{k: r[k] for k in r if k != "x"} for r in rows], indent=1))
    if a.hf:
        out = run_hf()
        _save("high_fidelity", out)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
