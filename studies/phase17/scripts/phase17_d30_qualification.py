"""Phase 17-HARDEN-D30: qualification of the rescore-on-reject acceptance rule.

Predeclared in studies/phase17/configs/phase17_d30_predeclaration.json (committed before any number).
Writes studies/phase17/results/phase17_d30_qualification.json.

Re-runs the four D27 --simplified cases (KP2 cross-check problem, SIMPLIFIED dynamics, rng [98, 0, 0])
twice: with the default (repaired) rule, and with acceptance_rescore_tolerance_factor=None (historical
rule), and scores gates Q1-Q5 and Q7 against the stored D27 rows.
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

import phase17_bls1_qualification as B  # noqa: E402
import phase17_d27_qualification as D27  # noqa: E402
import phase17_kp2_campaign as P  # noqa: E402
import phase17_kp2_core as K2  # noqa: E402

OUT = P.ARTIFACTS / "phase17_d30_qualification.json"
D27_OUT = P.ARTIFACTS / "phase17_d27_qualification.json"
REF_CHI2 = 902.4409
RULES = {"repaired": 1e-2, "historical": None}


def run_case(job) -> dict:
    case, rule = job
    import phase17_geo_core as G
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.srp import SRPOptions
    P.K._init()
    ns = D27._namespace(case)
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
    xb, st, s = ns["estimate_two_way_range_bls_lm"](
        scn.t_grid, obs, start, fx.rb.pass_geo, *args, srp=srp, solve_for_k_srp=True, k_srp_initial=K2.K0,
        acceptance_rescore_tolerance_factor=RULES[rule], **kw, **extra)
    el = time.time() - t0
    step, sd, chi2 = B._gauss_newton_at(scn.t_grid, obs, fx.rb.pass_geo, np.asarray(xb), s.k_srp_estimate, srp,
                                        args, kw, {})
    return dict(case=case, rule=rule, status=st, termination_detail=s.termination_detail,
                production_decrement=s.gauss_newton_decrement,
                independent_decrement=D27._decrement_at(scn.t_grid, obs, fx.rb.pass_geo, np.asarray(xb),
                                                        s.k_srp_estimate, srp, args, kw, {}),
                gn_step_over_sigma_max=float(np.max(np.abs(step / sd))), chi2=chi2,
                k=s.k_srp_estimate, x=np.asarray(xb).tolist(), iterations=s.iterations,
                rescore_accepts=s.acceptance_rescore_accepts,
                pos_err_m=float(np.linalg.norm(np.asarray(xb)[:3] - x0[:3])), elapsed_s=el)


def gates(rows: list[dict]) -> dict:
    stored = {r["case"]: r for r in json.loads(D27_OUT.read_text(encoding="utf-8"))["simplified"]["rows"]}
    rep = {r["case"]: r for r in rows if r["rule"] == "repaired"}
    hist = {r["case"]: r for r in rows if r["rule"] == "historical"}
    same = lambda a, b: a["x"] == b["x"] and a["k"] == b["k"]  # noqa: E731
    t, a, bj = rep["truth_start"], rep["apriori"], rep["bad_jacobian"]
    g = dict(
        Q1=t["chi2"] <= REF_CHI2 + 1.0 and t["gn_step_over_sigma_max"] <= 0.25,
        Q2=same(a, stored["apriori"]) or (a["chi2"] <= REF_CHI2 + 0.5 and a["gn_step_over_sigma_max"] <= 0.25),
        Q3=bj["status"] == "DampingLimit" and bj["production_decrement"] >= 100.0,
        Q5=all(same(hist[c], stored[c]) for c in D27.CASES),
    )
    g["Q4_floor"] = {k: rep["floor"][k] for k in ("status", "production_decrement", "chi2", "rescore_accepts")}
    g["Q7_elapsed_s"] = {c: dict(repaired=rep[c]["elapsed_s"], historical=hist[c]["elapsed_s"],
                                 stored_d27=stored[c]["elapsed_s"]) for c in D27.CASES}
    return g


def main():
    global OUT
    if len(sys.argv) > 1:  # e.g. "amended_A1": separate record per predeclared rule version
        OUT = OUT.with_name(f"phase17_d30_qualification_{sys.argv[1]}.json")
    jobs = [(c, r) for r in RULES for c in D27.CASES]
    with ProcessPoolExecutor(len(jobs)) as ex:
        rows = list(ex.map(run_case, jobs))
    out = dict(label="HARDEN-D30 qualification (KP2 cross-check, SIMPLIFIED dynamics, K solve-for)",
               predeclaration="studies/phase17/configs/phase17_d30_predeclaration.json", rows=rows,
               gates=gates(rows))
    OUT.write_text(json.dumps(P.K._clean(out), indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["gates"].items()}, indent=1))
    print(json.dumps([{k: r[k] for k in r if k != "x"} for r in rows], indent=1))


if __name__ == "__main__":
    main()
