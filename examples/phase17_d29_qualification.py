"""Phase 17-HARDEN-D29: qualification of the SRIF K-path prior contract.

Gates are frozen in artifacts/phase17_d29_predeclaration.json.

  --run --code old|new   SRIF / BLS cases on the 3.3 h single-station fixture (SRP-only dynamics, seeded 5 m
                         noise) with the pre-D29 estimators (b5d59b8, exec'd and rebound in every loaded lunar_od
                         module) or the current module -> artifacts/phase17_d29_run_{old,new}.json
  --compare              gates G1-G6 + materiality -> artifacts/phase17_d29_qualification.json
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
OUT = ART / "phase17_d29_qualification.json"
PRE_D29 = "b5d59b8"
K0, SIG_K = 0.01, 0.005
DX = np.array([30.0, -25.0, 20.0, 0.03, -0.02, 0.015])          # second start (the hidden prior's mean moves)
P0 = np.diag([1e4] * 3 + [1e-2] * 3)


def _load(code):
    import lunar_od.estimators as E
    if code == "old":
        src = subprocess.run(["git", "show", f"{PRE_D29}:lunar_od/estimators.py"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
        exec(compile(src, E.__file__, "exec"), E.__dict__)
        for name, mod in list(sys.modules.items()):
            if name.startswith("lunar_od") and mod is not E:
                for attr, val in list(vars(mod).items()):
                    if callable(val) and getattr(val, "__module__", None) == "lunar_od.estimators" \
                            and attr in E.__dict__:
                        setattr(mod, attr, E.__dict__[attr])
        assert "prior_rhs_sqrt" not in src
    return E


def _fixture():
    """The D29 regression-test fixture (built in tests/, where the unchecked generator is allowed)."""
    from tests import test_harden_d25_posterior_dynamics_contract as D25
    from tests import test_harden_d29_srif_k_prior_contract as T
    D25.setup_module(None)
    f = T._build_fixture()
    return dict(f, dyn=D25._dyn(f), tol=D25.TOL)


def run(code):
    E = _load(code)
    fx = _fixture()
    base = (fx["t_pass"], fx["obs"])
    out = []
    starts = {"start_truth": fx["x_true0"], "start_offset": fx["x_true0"] + DX}
    for sname, x0 in starts.items():
        for prior in (None, P0):
            ptag = "explicit_state_prior" if prior is not None else "no_state_prior"
            cases = {
                "srif_k": (E.estimate_two_way_range_srif,
                           dict(solve_for_k_srp=True, k_srp_initial=K0, k_srp_prior_sigma=SIG_K)),
                "bls_k": (E.estimate_two_way_range_bls_lm,
                          dict(solve_for_k_srp=True, k_srp_initial=K0, k_srp_prior_sigma=SIG_K)),
                "srif_six": (E.estimate_two_way_range_srif, dict()),
            }
            for cname, (fn, kw) in cases.items():
                x, stop, st = fn(*base, x0, fx["pg"], *fx["dyn"], max_iter=40, **fx["tol"], srp=fx["srp"],
                                 prior_covariance=prior, return_posterior=True, **kw)
                out.append(dict(case=cname, start=sname, prior=ptag, stop=stop, iterations=st.iterations,
                                x=np.asarray(x, float).tolist(), k=st.k_srp_estimate, final_cost=st.final_cost,
                                cov=np.asarray(st.posterior_covariance).tolist(),
                                info=np.asarray(st.posterior_information).tolist()))
    # independent objective pieces at the new SRIF-K no-prior solutions (data-only H and residual + K prior)
    if code == "new":
        from lunar_od.srp import SRPOptions
        for r in out:
            if r["case"] in ("srif_k", "bls_k") and r["prior"] == "no_state_prior":
                srp = SRPOptions(k_srp_m2_per_kg=r["k"])
                aug = E.propagate_state_with_k_sensitivity(fx["t_pass"], np.array(r["x"]), *fx["dyn"], srp=srp,
                                                           **fx["tol"])
                _, hx = E.two_way_range_nominal_and_initial_jacobian(fx["obs"], fx["pg"], aug)
                h = np.hstack([hx, E._two_way_range_k_srp_column(fx["obs"], aug, hx)[:, None]])
                hist = E.propagate_state(fx["t_pass"], np.array(r["x"]), *fx["dyn"], srp=srp, **fx["tol"])
                res, _ = E.compute_two_way_range_residuals(hist, fx["obs"], fx["pg"])
                r["indep_H"] = h.tolist()
                r["indep_res"] = np.asarray(res).tolist()
                r["w"] = E._two_way_range_weight_diagonal(fx["obs"], fx["pg"]).tolist()
    return out


def _sig(c):
    return np.sqrt(np.diag(np.asarray(c)))


def compare() -> dict:
    old = json.loads((ART / "phase17_d29_run_old.json").read_text(encoding="utf-8"))
    new = json.loads((ART / "phase17_d29_run_new.json").read_text(encoding="utf-8"))
    key = lambda r: (r["case"], r["start"], r["prior"])                      # noqa: E731
    O, N = {key(r): r for r in old}, {key(r): r for r in new}
    full = lambda r: np.array(r["x"] + ([r["k"]] if r["k"] is not None else []))   # noqa: E731
    same = lambda a, b: a["x"] == b["x"] and a["k"] == b["k"] and a["stop"] == b["stop"] \
        and a["iterations"] == b["iterations"] and a["cov"] == b["cov"] and a["info"] == b["info"]  # noqa: E731
    g1 = all(same(O[k], N[k]) for k in N if k[2] == "explicit_state_prior")
    g2 = all(same(O[k], N[k]) for k in N if k[0] == "srif_six")
    # G3: start independence of the SRIF-K no-prior solution
    st = {}
    for code, D in (("old", O), ("new", N)):
        a, b = D[("srif_k", "start_truth", "no_state_prior")], D[("srif_k", "start_offset", "no_state_prior")]
        st[code] = float(np.max(np.abs(full(a) - full(b)) / _sig(N[("srif_k", "start_truth", "no_state_prior")]["cov"])))
    g3 = st["new"] <= 1e-2
    # G4 / G5 at the new solutions (start_truth)
    r = N[("srif_k", "start_truth", "no_state_prior")]
    h, res, w = np.array(r["indep_H"]), np.array(r["indep_res"]), np.array(r["w"])
    sw = np.sqrt(w)
    a = np.vstack([h * sw[:, None], np.r_[np.zeros(6), 1.0 / SIG_K][None, :]])
    b = np.r_[sw * res, (K0 - r["k"]) / SIG_K]
    c = np.linalg.norm(a, axis=0)
    step = np.linalg.lstsq(a / c, b, rcond=None)[0] / c
    rinv = np.linalg.inv(np.linalg.qr(a / c, mode="r"))
    cov_ind = (rinv @ rinv.T) / np.outer(c, c)
    sd = np.sqrt(np.diag(cov_ind))
    g4_val = float(np.max(np.abs(step) / sd))
    g5_val = float(np.max(np.abs(_sig(r["cov"]) / sd - 1)))
    # G6: BLS vs SRIF under the identical declared problem (no state prior, same K prior, same data/start)
    bl = N[("bls_k", "start_truth", "no_state_prior")]
    g6_est = float(np.max(np.abs(full(bl) - full(r)) / sd))
    g6_cov = float(np.max(np.abs(_sig(bl["cov"]) / _sig(r["cov"]) - 1)))
    # POST-HOC (labelled) G6 attribution: declared-problem covariance rebuilt at each estimator's own point
    def declared_cov(rr):
        hh, ww = np.array(rr["indep_H"]), np.sqrt(np.array(rr["w"]))
        aa = np.vstack([hh * ww[:, None], np.r_[np.zeros(6), 1.0 / SIG_K][None, :]])
        cc = np.linalg.norm(aa, axis=0)
        ri = np.linalg.inv(np.linalg.qr(aa / cc, mode="r"))
        return (ri @ ri.T) / np.outer(cc, cc)
    cb, cs = declared_cov(bl), declared_cov(r)
    g6_posthoc = dict(label="POST-HOC: the G6 posterior gap is the covariance's linearisation-point dependence",
                      bls_posterior_vs_declared_at_bls_point=float(np.max(np.abs(_sig(bl["cov"]) / _sig(cb) - 1))),
                      srif_posterior_vs_declared_at_srif_point=float(np.max(np.abs(_sig(r["cov"]) / _sig(cs) - 1))),
                      declared_bls_point_vs_srif_point=float(np.max(np.abs(_sig(cb) / _sig(cs) - 1))))
    # materiality: SRIF-K no-prior change old -> new
    mat = {}
    for s_ in ("start_truth", "start_offset"):
        o, n = O[("srif_k", s_, "no_state_prior")], N[("srif_k", s_, "no_state_prior")]
        mat[s_] = dict(estimate_change_over_new_sigma=(np.abs(full(n) - full(o)) / _sig(n["cov"])).tolist(),
                       position_change_m=float(np.linalg.norm(np.array(n["x"][:3]) - np.array(o["x"][:3]))),
                       k_old=o["k"], k_new=n["k"],
                       sigma_rel_change=(_sig(n["cov"]) / _sig(o["cov"]) - 1).tolist(),
                       stop_old=o["stop"], stop_new=n["stop"], iterations_old=o["iterations"],
                       iterations_new=n["iterations"])
    gates = dict(G1_explicit_state_prior_unchanged=g1, G2_six_state_unchanged=g2,
                 G3_no_state_pull=g3, G4_stationarity=g4_val <= 1e-2, G5_posterior_rebuild=g5_val <= 1e-8,
                 G6_bls_srif_equivalence=(g6_est <= 2e-2 and g6_cov <= 1e-6))
    return dict(gates=gates, g6_posthoc_attribution=g6_posthoc, start_dependence_over_sigma=st, stationarity_max_over_sigma=g4_val,
                posterior_rebuild_sigma_rel=g5_val, bls_srif_estimate_over_sigma=g6_est, bls_srif_sigma_rel=g6_cov,
                bls_stop=bl["stop"], srif_stop=r["stop"], materiality=mat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--code", choices=("old", "new"))
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.run:
        rows = run(a.code)
        (ART / f"phase17_d29_run_{a.code}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(len(rows), "rows")
    if a.compare:
        out = compare()
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
