"""Phase 17-HARDEN-D28: qualification of the no-explicit-prior posterior contract.

Gates are frozen in artifacts/phase17_d28_predeclaration.json.

Modes:
  --scenarios --code old|new   run the scenario hand-off matrix of tests/test_scenarios.py
                               (position / geometric range-rate / two-way range-rate x BLS / SRIF x
                               cold / hot / formal / sqrt_formal, two arcs, seeded noise) with the
                               pre-D28 estimators (3959f7a, exec'd into lunar_od.estimators BEFORE
                               lunar_od.scenarios binds them) or the current module; writes one JSON each
  --compare                    per-arc differences old vs new -> artifacts/phase17_d28_qualification.json

In a formal / sqrt_formal chain arc 1 runs with no explicit prior and its posterior becomes arc 2's
EXPLICIT prior, so D28 can move arc 2's estimate. That is a downstream consequence of the corrected
arc-1 covariance, quantified here rather than assumed away.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")
ART = ROOT / "artifacts"
OUT = ART / "phase17_d28_qualification.json"
PRE_D28 = "3959f7a"


def _load(code: str):
    import lunar_od.estimators as E
    if code == "old":
        src = subprocess.run(["git", "show", f"{PRE_D28}:lunar_od/estimators.py"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
        exec(compile(src, E.__file__, "exec"), E.__dict__)
        # lunar_od/__init__ has already imported lunar_od.scenarios, which bound the estimator functions
        # by name: rebind every such name in every loaded lunar_od module to the pre-D28 objects
        n = 0
        for name, mod in list(sys.modules.items()):
            if not name.startswith("lunar_od") or mod is E:
                continue
            for attr, val in list(vars(mod).items()):
                if callable(val) and getattr(val, "__module__", None) == "lunar_od.estimators" and attr in E.__dict__:
                    setattr(mod, attr, E.__dict__[attr])
                    n += 1
        assert n > 0 and "_posterior_prior_terms" not in src, "pre-D28 source not in effect"
    return E


def _invariance(code: str) -> list[dict]:
    """G1/G2: every estimator path, no prior and explicit prior, on cheap fixtures."""
    E = _load(code)
    from lunar_od.srp import SRPOptions
    from tests import test_harden_d25_posterior_dynamics_contract as D25
    from tests import test_scenarios as T
    D25.setup_module(None)
    mu = 4902.800066e9
    t_all, x_truth, gep, gsp = T._matrix_truth_history(mu)
    pos_st = (T._synthetic_station(0.0, 0.0, 0.0), T._synthetic_station(0.0, 90.0, 0.0),
              T._synthetic_station(45.0, -30.0, 500.0), T._synthetic_station(-35.0, 150.0, 600.0))
    rr_st = (T._synthetic_rr_station(0.0, 0.0, 0.0), T._synthetic_rr_station(0.0, 90.0, 0.0),
             T._synthetic_rr_station(45.0, -30.0, 500.0), T._synthetic_rr_station(-35.0, 150.0, 600.0))
    pos = T._with_measurement_noise(T._build_position_arc(1, 0, 8, t_all, x_truth, pos_st), seed=101)
    rr = T._with_measurement_noise(T._build_range_rate_arc(1, 0, 8, t_all, x_truth, rr_st), seed=201)
    fx = D25._build_fixture()
    dx = np.array([60.0, -50.0, 35.0, 0.025, -0.020, 0.012])
    p0 = np.diag([1e4] * 3 + [1e-2] * 3)
    cases = []
    for arc, kind in ((pos, "position"), (rr, "range_rate")):
        for est in ("srif", "bls_lm"):
            fn = getattr(E, f"estimate_{kind}_{est}")
            for prior in (None, p0):
                cases.append((f"{kind}_{est}_{'explicit' if prior is not None else 'none'}", fn,
                              (arc.t_pass_s, arc.obs_data, arc.truth_state_history_mci[0] + dx, arc.pass_geo, mu, 0.0,
                               0.0, gep, gsp), dict(max_iter=10, rtol=1e-12, atol=1e-13, prior_covariance=prior)))
    tw = (fx["t_pass"], fx["obs"], fx["x_true0"], fx["pg"], *D25._dyn(fx))
    srp = fx["srp"]
    for prior in (None, p0):
        tag = "explicit" if prior is not None else "none"
        cases.append((f"two_way_bls_six_{tag}", E.estimate_two_way_range_bls_lm, tw,
                      dict(max_iter=3, **D25.TOL, srp=srp, prior_covariance=prior, **fx["harm"])))
        cases.append((f"two_way_bls_k_{tag}", E.estimate_two_way_range_bls_lm, tw,
                      dict(max_iter=3, **D25.TOL, srp=srp, solve_for_k_srp=True, k_srp_initial=0.01,
                           prior_covariance=prior, **fx["harm"])))
        cases.append((f"two_way_srif_six_{tag}", E.estimate_two_way_range_srif, tw,
                      dict(max_iter=3, **D25.TOL, srp=srp, prior_covariance=prior)))
        cases.append((f"two_way_srif_k_{tag}", E.estimate_two_way_range_srif, tw,
                      dict(max_iter=3, **D25.TOL, srp=srp, solve_for_k_srp=True, k_srp_initial=0.01,
                           k_srp_prior_sigma=0.005, prior_covariance=prior)))
    cases.append(("two_way_bls_k_kprior_only", E.estimate_two_way_range_bls_lm, tw,
                  dict(max_iter=3, **D25.TOL, srp=srp, solve_for_k_srp=True, k_srp_initial=0.01,
                       k_srp_prior_sigma=0.005, **fx["harm"])))
    rows = []
    for name, fn, args, kw in cases:
        x, stop, st = fn(*args, **kw, return_posterior=True)
        rows.append(dict(name=name, x=np.asarray(x, float).tolist(), stop=stop, iterations=st.iterations,
                         final_cost=st.final_cost, k=st.k_srp_estimate,
                         info=np.asarray(st.posterior_information).tolist(),
                         cov=np.asarray(st.posterior_covariance).tolist()))
    return rows


def compare_invariance() -> dict:
    old = json.loads((ART / "phase17_d28_invariance_old.json").read_text(encoding="utf-8"))
    new = json.loads((ART / "phase17_d28_invariance_new.json").read_text(encoding="utf-8"))
    rows = []
    for a, b in zip(old, new):
        assert a["name"] == b["name"]
        est = (a["x"] == b["x"] and a["stop"] == b["stop"] and a["iterations"] == b["iterations"]
               and a["final_cost"] == b["final_cost"] and a["k"] == b["k"])
        sa, sb = np.sqrt(np.diag(np.array(a["cov"]))), np.sqrt(np.diag(np.array(b["cov"])))
        rows.append(dict(name=a["name"], estimate_bitwise=bool(est), posterior_bitwise=a["info"] == b["info"]
                         and a["cov"] == b["cov"], sigma_rel_change_max=float(np.max(np.abs(sb / sa - 1)))))
    expect_same_post = [r for r in rows if r["name"].endswith("_explicit") or r["name"].startswith("two_way_srif_k")]
    return dict(rows=rows,
                G1_estimates_unchanged=all(r["estimate_bitwise"] for r in rows),
                G2_explicit_prior_posterior_unchanged=all(r["posterior_bitwise"] for r in expect_same_post),
                no_prior_posteriors_changed=[r["name"] for r in rows if not r["posterior_bitwise"]])


def kp2_materiality() -> dict:
    """KP2 CANON k0 HF K path at the BLS1 production solution: the repaired production design saved by D25
    (artifacts/phase17_d25_posthoc_fd_columns.npz), posterior with the historical default diagonal vs data-only,
    and the frozen KP2 sigma criterion against the KP2 solver's sigma (data-only by construction)."""
    sys.path.insert(0, str(ROOT / "examples"))
    import lunar_od.estimators as E
    import phase17_kp2_core as K2
    d = np.load(ART / "phase17_d25_posthoc_fd_columns.npz")["design"]
    q = json.loads((ART / "phase17_d25_qualification.json").read_text(encoding="utf-8"))["kp2_k_path"]
    sig_hist = np.array(q["sigma_repaired"])
    sd_solver = np.array(q["kp2_solver"]["sigma_ratio_repaired"]) * sig_hist
    w = np.full(d.shape[0], 1.0 / K2.SIGMA_RANGE_M ** 2)
    pi6, _, s6, _, _ = E._prior_information_and_scale(6, E._TWO_WAY_RANGE_NO_BIAS_CFG,
                                                       E._two_way_range_bias_prior_and_scale, None, None)
    scale = np.diag(list(np.diag(s6)) + [1e-2])
    hist = np.zeros((7, 7))
    hist[:6, :6] = pi6
    c_hist, _ = E._square_root_covariance_from_design(d, w, hist, scale)
    c_new, _ = E._square_root_covariance_from_design(d, w, np.zeros((7, 7)), scale)
    s_hist, s_new = np.sqrt(np.diag(c_hist)), np.sqrt(np.diag(c_new))
    return dict(reproduces_d25_sigma=float(np.max(np.abs(s_hist / sig_hist - 1))),
                sigma_change_data_only_vs_historical=(s_new / s_hist - 1).tolist(),
                kp2_solver_over_production_data_only=(sd_solver / s_new).tolist(),
                kp2_solver_over_production_historical=(sd_solver / s_hist).tolist(),
                frozen_sigma_criterion_pass=bool(np.max(np.abs(sd_solver / s_new - 1)) <= 0.01))


def _scenario_matrix(code: str) -> list[dict]:
    E = _load(code)
    from lunar_od.scenarios import run_batch_arc_sequence
    import lunar_od.scenarios as S
    assert (S.estimate_position_srif is E.estimate_position_srif), "scenarios not bound to the module under test"
    from tests import test_scenarios as T

    mu = 4902.800066e9
    t_all, x_truth, gep, gsp = T._matrix_truth_history(mu)
    pos_st = (T._synthetic_station(0.0, 0.0, 0.0), T._synthetic_station(0.0, 90.0, 0.0),
              T._synthetic_station(45.0, -30.0, 500.0), T._synthetic_station(-35.0, 150.0, 600.0))
    rr_st = (T._synthetic_rr_station(0.0, 0.0, 0.0), T._synthetic_rr_station(0.0, 90.0, 0.0),
             T._synthetic_rr_station(45.0, -30.0, 500.0), T._synthetic_rr_station(-35.0, 150.0, 600.0))
    builders = {
        "position": lambda i, s, e: T._with_measurement_noise(T._build_position_arc(i, s, e, t_all, x_truth, pos_st),
                                                              seed=100 + i),
        "geometric_rr": lambda i, s, e: T._with_measurement_noise(
            T._build_range_rate_arc(i, s, e, t_all, x_truth, rr_st), seed=200 + i),
        "two_way_rr": lambda i, s, e: T._with_measurement_noise(
            T._build_two_way_range_rate_arc(i, s, e, t_all, x_truth, rr_st), seed=300 + i),
    }
    bank = (np.array([60.0, -50.0, 35.0, 0.025, -0.020, 0.012]), np.array([80.0, -65.0, 45.0, 0.030, -0.025, 0.016]))
    rows = []
    for case in ("position", "geometric_rr", "two_way_rr"):
        mtype = "position" if case == "position" else "range_rate"
        for est in ("bls_lm", "srif"):
            modes = ("cold", "hot", "formal") if est == "bls_lm" else ("cold", "hot", "formal", "sqrt_formal")
            for mode in modes:
                arcs = (builders[case](1, 0, 8), builders[case](2, 10, 18))
                res = run_batch_arc_sequence(arcs, mtype, mode, est, mu, 0.0, 0.0, gep, gsp, cold_start_bank=bank,
                                             label=f"{case}_{est}_{mode}", max_iter=10, rtol=1e-12, atol=1e-13)
                for k, a in enumerate(res.arc_results):
                    pc = None if a.posterior_covariance is None else np.asarray(a.posterior_covariance).tolist()
                    rows.append(dict(case=case, estimator=est, mode=mode, arc=k + 1, stop=a.stop_reason,
                                     x=np.asarray(a.estimated_state, float).tolist(),
                                     pos_err=float(a.final_position_error_m), vel_err=float(a.final_velocity_error_mps),
                                     posterior_covariance=pc))
    return rows


def compare() -> dict:
    old = json.loads((ART / "phase17_d28_scenarios_old.json").read_text(encoding="utf-8"))
    new = json.loads((ART / "phase17_d28_scenarios_new.json").read_text(encoding="utf-8"))
    out = []
    for a, b in zip(old, new):
        assert (a["case"], a["estimator"], a["mode"], a["arc"]) == (b["case"], b["estimator"], b["mode"], b["arc"])
        xa, xb = np.array(a["x"]), np.array(b["x"])
        r = dict(case=a["case"], estimator=a["estimator"], mode=a["mode"], arc=a["arc"],
                 stop_old=a["stop"], stop_new=b["stop"], estimate_bitwise=bool(np.array_equal(xa, xb)),
                 pos_err_old=a["pos_err"], pos_err_new=b["pos_err"])
        if a["posterior_covariance"] is not None and b["posterior_covariance"] is not None:
            pa, pb = np.array(a["posterior_covariance"]), np.array(b["posterior_covariance"])
            sa, sb = np.sqrt(np.abs(np.diag(pa))), np.sqrt(np.abs(np.diag(pb)))
            r["sigma_rel_change_max"] = float(np.max(np.abs(sb / sa - 1)))
            r["estimate_shift_over_new_sigma_max"] = float(np.max(np.abs(xb - xa)[:6] / sb[:6]))
        r["estimate_shift_m"] = float(np.linalg.norm((xb - xa)[:3]))
        out.append(r)
    return dict(rows=out,
                n_rows=len(out),
                n_estimate_changed=sum(not r["estimate_bitwise"] for r in out),
                max_sigma_rel_change=max((r.get("sigma_rel_change_max", 0.0) for r in out), default=0.0),
                max_estimate_shift_m=max(r["estimate_shift_m"] for r in out),
                max_estimate_shift_over_sigma=max((r.get("estimate_shift_over_new_sigma_max", 0.0) for r in out),
                                                  default=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", action="store_true")
    ap.add_argument("--code", choices=("old", "new"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--kp2", action="store_true")
    ap.add_argument("--compare-invariance", action="store_true")
    a = ap.parse_args()
    if a.scenarios:
        rows = _scenario_matrix(a.code)
        (ART / f"phase17_d28_scenarios_{a.code}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(len(rows), "rows")
    if a.kp2:
        out = kp2_materiality()
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        data["kp2_materiality"] = out
        OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
    if a.invariance:
        rows = _invariance(a.code)
        (ART / f"phase17_d28_invariance_{a.code}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(len(rows), "rows")
    if a.compare_invariance:
        out = compare_invariance()
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        data["invariance"] = out
        OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
    if a.compare:
        out = compare()
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        data["scenario_handoff_matrix"] = out
        OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        for r in out["rows"]:
            print(r["case"], r["estimator"], r["mode"], r["arc"], r["stop_old"], r["stop_new"],
                  "est_same" if r["estimate_bitwise"] else "EST %.3e m / %.3e sig" % (
                      r["estimate_shift_m"], r.get("estimate_shift_over_new_sigma_max", float("nan"))),
                  "dsig %.2e" % r.get("sigma_rel_change_max", float("nan")))


if __name__ == "__main__":
    main()
