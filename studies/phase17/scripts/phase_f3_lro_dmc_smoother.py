"""Phase F3: Kalman filter / RTS smoother with dynamic model compensation on the real LRO arc.

Predeclaration: studies/phase17/configs/phase_f3_lro_dmc_smoother_predeclaration.json.

Two steps:
  --reference [--workers N]  one F2 residual(p, partials=True) at F2p over days 260-263 (F2p numerics), plus the
                             reference and LOLA states at F2's 60 s comparison epochs; cached outside git
                             (<workspace>/lro_data/f3_cache/reference.npz); prints the G0 numbers
  --sweep                    the linearised filter + smoother for every sigma, gates, hypotheses, decision
With neither flag both run in order.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # F2p's numerics: legacy 20 s grid, quantised station epochs
import phase_f2k_tight_tolerance as K  # noqa: E402
import phase_f3_core as C  # noqa: E402

PRE = "studies/phase17/configs/phase_f3_lro_dmc_smoother_predeclaration.json"
OUT = HERE.parent / "results" / "phase_f3_lro_dmc_smoother.json"
DAYS = M.F2J_FIT_DAYS + M.F2J_PREDICT_DAYS
DAY263_UTC = "2009-09-20T00:00:00"
SIGMAS = (0.0, 1e-9, 3e-9, 1e-8, 3e-8)
TAU_S = 1800.0
W_DOP = M.SIGMA_M / M.SIGMA_DOP          # Doppler rows are scaled to unit weight
#: constants, in F2's parameter order 6..14: K, range biases 103/105/126, Doppler bias, C_T, S_T, C_N, S_N
CONST_SCALES = [1e-3, 1.0, 1.0, 1.0, 1e-3, 1e-9, 1e-9, 1e-9, 1e-9]
STATE_SCALES = [1.0] * 3 + [1e-3] * 3
W_SCALE = 1e-9
EPOCH_PRIOR = (1e4, 10.0)                # m, m/s
DOPPLER_BIAS_PRIOR = 1.0                 # m/s; F2 has no Doppler-bias prior, so a loose one


def cache_path() -> Path:
    from lunar_od.project_paths import project_root
    path = project_root() / "lro_data" / "f3_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path / "reference.npz"


def f2p():
    d = json.loads(M.OUT_F2P.read_text(encoding="utf-8"))
    return d, d["runs"]["S1"]


def build_reference(workers: int) -> None:
    import spiceypy as spice
    t_start = time.time()
    d, run = f2p()
    M.set_tolerance(*M.F2K_TOL)
    M.set_doppler_edits(d["doppler_edits"])
    p = K.point_vector(run)
    pb = M.Problem(300, doppler=True, days=DAYS, n_emp=len(M.EMP_NAMES))
    assert (pb.grid_s, pb.rtol, pb.atol, pb.emp_fd_step) == (20.0, 1e-12, 1e-13, 1e-9), "not F2p's numerics"
    pb.attach_pool(workers)
    try:
        res, h, hist = pb.residual(p, partials=True)
    finally:
        pb.close_pool()
    nr = len(pb.rows)
    n_fit_r = sum(len(M.F.read_day(pb.data, day)[0]) for day in M.F2J_FIT_DAYS)
    t_r = np.asarray(pb.setup.t3, float)
    t_d = np.array([x["tm"] for x in pb.dop])
    t263 = float(spice.str2et(DAY263_UTC)) - pb.setup.et0
    is263 = np.concatenate([np.arange(nr) >= n_fit_r, t_d >= t263])
    kind = np.concatenate([np.zeros(nr, int), np.ones(t_d.size, int)])
    t_cmp_fit = np.arange(float(t_r[:n_fit_r].min()), float(t_r[:n_fit_r].max()), 60.0)
    t_cmp_263 = np.arange(float(t_r[n_fit_r:].min()), float(t_r[n_fit_r:].max()), 60.0)
    t_cmp = np.concatenate([t_cmp_fit, t_cmp_263])
    ref_cmp = pb.propagate_p(p, t_eval=np.concatenate([[pb.t_epoch], t_cmp]))[1:]
    lola = np.array([spice.spkezr(M.F.LRO, pb.setup.et0 + t, "J2000", "NONE", "MOON")[0] for t in t_cmp]) * 1e3
    np.savez_compressed(cache_path(), p=p, res=res, h=h, hist=hist, grid=pb.grid, t_epoch=pb.t_epoch,
                        t_row=np.concatenate([t_r, t_d]), kind=kind, is263=is263, t_cmp_fit=t_cmp_fit,
                        t_cmp_263=t_cmp_263, ref_cmp=ref_cmp, lola_cmp=lola, elapsed_s=time.time() - t_start)
    print(json.dumps(g0(load()), indent=1))


def load() -> dict:
    with np.load(cache_path()) as z:
        return {k: z[k] for k in z.files}


def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def _resid_stats(r, kind, is263):
    """Unit-weight residuals -> range m / Doppler mm/s RMS on the fit days and on 263."""
    out = {}
    for name, sel in (("fit", ~is263), ("263", is263)):
        out[name] = dict(range_rms_m=_rms(r[sel & (kind == 0)]),
                         doppler_rms_mmps=_rms(r[sel & (kind == 1)] / W_DOP * 1e3),
                         range_mean_m=float(np.mean(r[sel & (kind == 0)])),
                         doppler_mean_mmps=float(np.mean(r[sel & (kind == 1)] / W_DOP * 1e3)))
    return out


def _ric_rms(pos, ref_states):
    ric = M._ric(np.hstack([pos, np.zeros_like(pos)]), ref_states)
    return np.sqrt(np.mean(ric ** 2, axis=0)).tolist()


def g0(z) -> dict:
    _, run = f2p()
    st = _resid_stats(z["res"], z["kind"], z["is263"])
    nf = z["t_cmp_fit"].size
    ric_fit = _ric_rms(z["ref_cmp"][:nf, :3], z["lola_cmp"][:nf])
    ric_263 = _ric_rms(z["ref_cmp"][nf:, :3], z["lola_cmp"][nf:])
    want_fit, want_263 = run["ric_rms_m"], run["prediction"]["ric_rms_m"]
    checks = dict(
        fit_range=abs(st["fit"]["range_rms_m"] / run["postfit_rms_m"] - 1.0) <= 1e-3,
        fit_doppler=abs(st["fit"]["doppler_rms_mmps"] / run["doppler_postfit_rms_mmps"] - 1.0) <= 1e-3,
        p263_range=abs(st["263"]["range_rms_m"] - run["prediction"]["range_rms_m"]) <= 0.01,
        p263_doppler=abs(st["263"]["doppler_rms_mmps"] - run["prediction"]["doppler_rms_mmps"]) <= 0.01,
        ric_fit=max(abs(a - b) for a, b in zip(ric_fit, want_fit)) <= 0.01,
        ric_263=max(abs(a - b) for a, b in zip(ric_263, want_263)) <= 0.01)
    return dict(PASS=all(checks.values()), checks=checks, residuals=st, ric_fit_m=ric_fit, ric_263_m=ric_263,
                f2p=dict(fit_range_m=run["postfit_rms_m"], fit_doppler_mmps=run["doppler_postfit_rms_mmps"],
                         p263_range_m=run["prediction"]["range_rms_m"],
                         p263_doppler_mmps=run["prediction"]["doppler_rms_mmps"], ric_fit_m=want_fit,
                         ric_263_m=want_263),
                n_rows=dict(fit_range=int(np.sum(~z["is263"] & (z["kind"] == 0))),
                            fit_doppler=int(np.sum(~z["is263"] & (z["kind"] == 1))),
                            p263_range=int(np.sum(z["is263"] & (z["kind"] == 0))),
                            p263_doppler=int(np.sum(z["is263"] & (z["kind"] == 1)))))


def reference_of(z) -> C.Reference:
    hist = z["hist"]
    n = hist.shape[0]
    phi = hist[:, 6:42].reshape(n, 6, 6).transpose(0, 2, 1)       # stored column-major
    s = np.zeros((n, 6, 9))
    s[:, :, 0] = hist[:, 42:48]
    for i in range(4):
        s[:, :, 5 + i] = hist[:, 48 + 6 * i:54 + 6 * i]
    return C.Reference(z["grid"], phi, s)


def prior(z):
    d, _ = f2p()
    p = z["p"]
    mean = np.zeros(15)
    sig = np.array([EPOCH_PRIOR[0]] * 3 + [EPOCH_PRIOR[1]] * 3 + [0.0] * 9)
    sig[10] = DOPPLER_BIAS_PRIOR
    for j, (m, s) in d["priors"].items():
        mean[int(j)] = m - p[int(j)]
        sig[int(j)] = s
    assert np.all(sig > 0)
    return mean, np.diag(sig ** 2)


def sweep() -> None:
    z = load()
    g0r = g0(z)
    _, run = f2p()
    ref = reference_of(z)
    t0 = float(z["t_epoch"])
    h, r, t_row, kind, is263 = z["h"], z["res"], z["t_row"], z["kind"], z["is263"]
    nf = z["t_cmp_fit"].size
    t_cmp = np.concatenate([z["t_cmp_fit"], z["t_cmp_263"]])
    epochs = np.unique(np.concatenate([t_row, t_cmp]))
    pos = {t: i for i, t in enumerate(epochs)}
    order = np.lexsort((np.arange(t_row.size), t_row))
    rows = [None] * epochs.size
    upd_kind = []                                         # row kinds in update order (time, then row order)
    for j in order:
        if is263[j]:
            continue
        k = pos[t_row[j]]
        rows[k] = (h[j][None, :], np.array([r[j]])) if rows[k] is None else             (np.vstack([rows[k][0], h[j]]), np.append(rows[k][1], r[j]))
        upd_kind.append(int(kind[j]))
    upd_kind = np.array(upd_kind)
    # local rows of every measurement (physical units, no w), for the residuals at the smoothed state
    phi_r, s_r = ref.at(t_row)
    hx = np.einsum("nij,nj->ni", np.linalg.inv(phi_r).transpose(0, 2, 1), h[:, :6])
    hc = h[:, 6:] - np.einsum("ni,nij->nj", hx, s_r)
    row_epoch = np.array([pos[t] for t in t_row])
    cmp_epoch = np.array([pos[t] for t in t_cmp])
    pm, pc = prior(z)
    oracle_x, oracle_cov = C.batch_oracle(h[~is263], r[~is263], pm, pc)

    results = {}
    for sigma in SIGMAS:
        tic = time.time()
        sg = None if sigma == 0.0 else sigma
        nw = 0 if sg is None else 3
        scales = STATE_SCALES + [W_SCALE] * nw + CONST_SCALES
        out = C.smooth(ref, t0, epochs, rows, pm, pc, sg, TAU_S, scales)
        xs = out["x"][1:]                                 # epochs (index 0 is t0)
        dx = xs[:, :6]
        dc = xs[:, 6 + nw:]
        post = r - np.einsum("ni,ni->n", hx, dx[row_epoch]) - np.einsum("ni,ni->n", hc, dc[row_epoch])
        st = _resid_stats(post, kind, is263)
        chi2_263 = float(np.sum(post[is263] ** 2))
        est = z["ref_cmp"][:, :3] + dx[cmp_epoch, :3]
        ric_fit = _ric_rms(est[:nf], z["lola_cmp"][:nf])
        ric_263 = _ric_rms(est[nf:], z["lola_cmp"][nf:])
        x0 = out["x"][0]
        x0_15 = np.concatenate([x0[:6], x0[6 + nw:]])
        cov0 = out["cov"][t0]
        idx = np.r_[0:6, 6 + nw:6 + nw + 9]
        sig0 = np.sqrt(np.diag(cov0)[idx])
        nis = out["nis"]
        item = dict(sigma_mps2=sigma, tau_s=TAU_S, residuals=st, chi2_263=chi2_263, ric_fit_m=ric_fit,
                    ric_263_m=ric_263, epoch_deviation_over_sigma=(x0_15 / sig0).tolist(),
                    epoch_sigma=sig0.tolist(),
                    doppler_bias_mmps=float((z["p"][10] + dc[0, 4]) * 1e3), doppler_bias_change_mmps=float(dc[0, 4] * 1e3),
                    k_srp=float(z["p"][6] + dc[0, 0]),
                    mean_nis=dict(range=float(nis[upd_kind == 0].mean()), doppler=float(nis[upd_kind == 1].mean())),
                    runtime_s=time.time() - tic)
        if nw:
            w = xs[cmp_epoch, 6:9]
            item["w_ric_rms_mps2"] = _ric_rms(w + z["ref_cmp"][:, :3], z["ref_cmp"])
        else:
            item["batch_oracle_max_dev_over_sigma"] = float(np.max(np.abs(x0_15 - oracle_x) /
                                                               np.sqrt(np.diag(oracle_cov))))
        results[f"{sigma:g}"] = item
        print(f"sigma {sigma:g}: fit RIC {np.round(ric_fit, 2)} 263 RIC {np.round(ric_263, 2)} chi2_263 "
              f"{chi2_263:.1f} NIS {item['mean_nis']} {item['runtime_s']:.0f}s", flush=True)

    c0 = results["0"]
    g1 = dict(epoch_deviation=max(abs(v) for v in c0["epoch_deviation_over_sigma"]) <= 0.05,
              ric_fit=max(abs(a - b) for a, b in zip(c0["ric_fit_m"], run["ric_rms_m"])) <= 0.05,
              ric_263=max(abs(a - b) for a, b in zip(c0["ric_263_m"], run["prediction"]["ric_rms_m"])) <= 0.05)
    gates = dict(G0=g0r["PASS"], G1=all(g1.values()))
    nonzero = [k for k in results if float(k) > 0.0]
    star = min(nonzero, key=lambda k: results[k]["chi2_263"])
    s_ = results[star]
    h1 = s_["ric_fit_m"][1] <= 0.7 * run["ric_rms_m"][1] and s_["ric_fit_m"][2] <= 0.7 * run["ric_rms_m"][2]
    h2 = s_["ric_263_m"][1] <= 5.05 and s_["residuals"]["263"]["range_rms_m"] <= 5.21
    if not all(gates.values()):
        decision = "F3_SETUP_NOT_QUALIFIED"
    else:
        decision = {(True, True): "F3_DMC_IMPROVES", (True, False): "F3_DMC_FIT_ARC_ONLY",
                    (False, True): "F3_DMC_PREDICTION_ONLY", (False, False): "F3_NO_DMC_GAIN"}[(h1, h2)]
    out = dict(predeclaration=PRE, gates=gates, G0=g0r, G1_checks=g1, sigma_star=float(star),
               hypotheses=dict(H1=h1, H2=h2) if all(gates.values()) else "not read (gate FAIL)",
               hypotheses_as_computed=dict(H1=h1, H2=h2), decision=decision, tau_s=TAU_S,
               epoch_prior=dict(position_m=EPOCH_PRIOR[0], velocity_mps=EPOCH_PRIOR[1],
                                doppler_bias_mps=DOPPLER_BIAS_PRIOR),
               n_epochs=int(epochs.size), n_updates=int(upd_kind.size), sweep=results,
               reference_elapsed_s=float(z["elapsed_s"]))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("gates", "G1_checks", "sigma_star", "hypotheses_as_computed",
                                          "decision")}, indent=1))


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    if "--sweep" not in sys.argv:
        build_reference(workers)
    if "--reference" not in sys.argv:
        sweep()
