"""Phase F2: first LRO orbit determination from real USN range, compared with the LOLA precision orbit.

Predeclaration: studies/phase17/configs/phase_f2_lro_od_predeclaration.json.  Reuses F1's reader, stations,
epochs and measurement model (phase_f1_lro_residuals.py); the spacecraft history comes from lunar_od's
propagator instead of the reference SPK.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f1_lro_residuals as F  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2_lro_od.json"
PADS = (103, 105, 126)
K0 = 0.01
SIGMA_M = 1.0
RTOL, ATOL = 1e-10, 1e-11
GRID_S = 20.0
EPOCH_LEAD_S = 120.0
MAX_ITER = 30
OFFSET = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0])
#: A1: central-difference steps for the STM / dx/dK columns (state 10 m, 1 cm/s; K 1e-3 m^2/kg).
FD_STEPS = np.array([10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2, 1e-3])


class Problem:
    def __init__(self, nmax: int):
        import spiceypy as spice
        from lunar_od import relativistic_light_time as R
        from lunar_od.ephemeris import sample_moon_centered_ephemeris
        from lunar_od.gravity_model_loader import load_lunar_gravity_model
        from lunar_od.project_paths import gravity_data_root
        self.data = Path(F.REPO.parent / "lro_data")
        self.rows, _ = F.read_day(self.data)
        self.setup = F.Setup(self.data, self.rows, "DE440")
        s = self.setup
        self.obs = np.array([F.C * r.rtlt_s / 2.0 for _, r in self.rows])
        self.pad_index = np.array([PADS.index(p) for p, _ in self.rows])
        self.t_epoch = float(s.t3.min()) - EPOCH_LEAD_S
        self.t_end = float(s.t3.max()) + 10.0
        self.grid = np.arange(self.t_epoch, self.t_end + GRID_S, GRID_S)
        eph_t = np.arange(self.t_epoch - 600.0, self.t_end + 600.0, 60.0)
        self.eph = sample_moon_centered_ephemeris(s.et0, eph_t)
        self.mu_e = float(spice.bodvrd("399", "GM", 1)[1][0]) * 1e9
        self.mu_s = float(spice.bodvrd("10", "GM", 1)[1][0]) * 1e9
        model = load_lunar_gravity_model(gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab", nmax=nmax, mmax=nmax)
        self.model = dataclasses.replace(model, frame="MOON_PA_DE440")
        self.mu_m = float(self.model.mu_m3_s2)
        self.reference_epoch = np.asarray(spice.spkezr(F.LRO, s.et0 + self.t_epoch, "J2000", "NONE", "MOON")[0]) * 1e3
        self.media = None
        from lunar_od.media import MediaModel
        self.media = MediaModel(uplink_hz=F.UPLINK_HZ, downlink_hz=F.DOWNLINK_HZ)
        self.R = R

    def propagate(self, x0, k, t_eval=None, sensitivity=True):
        from lunar_od.dynamics import PropagationProfile, propagate_state, propagate_state_with_k_sensitivity
        from lunar_od.srp import SRPOptions
        kw = dict(srp=SRPOptions(k_srp_m2_per_kg=float(k)), rtol=RTOL, atol=ATOL, harmonic_model=self.model,
                  harmonic_epoch_et0=self.setup.et0, **PropagationProfile().propagator_kwargs())
        t = self.grid if t_eval is None else t_eval
        args = (t, np.asarray(x0, float), self.mu_m, self.mu_e, self.mu_s, self.eph.earth_position,
                self.eph.sun_position)
        if sensitivity:
            return np.asarray(propagate_state_with_k_sensitivity(*args, harmonic_stm_opt_in=True, **kw))
        return np.asarray(propagate_state(*args, **kw))

    def fd_history(self, x0, k) -> np.ndarray:
        """A1: [x, Phi (column-major), dx/dK] from central differences of the 6-state propagator."""
        base = self.propagate(x0, k, sensitivity=False)
        n = base.shape[0]
        cols = np.zeros((n, 6, 7))
        for j in range(7):
            q = np.concatenate([np.asarray(x0, float), [float(k)]])
            hj = FD_STEPS[j]
            q[j] += hj
            plus = self.propagate(q[:6], q[6], sensitivity=False)
            if j == 6 and float(k) < hj:
                # K must stay non-negative (SRPOptions refuses K < 0): forward difference near the bound
                cols[:, :, j] = (plus - base) / hj
                continue
            q[j] -= 2.0 * hj
            minus = self.propagate(q[:6], q[6], sensitivity=False)
            cols[:, :, j] = (plus - minus) / (2.0 * hj)
        phi = cols[:, :, :6]
        return np.hstack([base, phi.transpose(0, 2, 1).reshape(n, 36), cols[:, :, 6]])

    def computed(self, hist, partials=True, idx=None):
        R, s = self.R, self.setup
        sc = R.SpacecraftHistory(self.grid, hist)
        idx = range(len(self.rows)) if idx is None else idx
        comp, jac = [], []
        for i in idx:
            pad = self.rows[i][0]
            sol = R.solve_two_way(float(s.t3[i]), s.tide[pad], sc, s.eph, R.DSN_C2_LIGHT_TIME_MODEL,
                                  site=s.sites[pad], media=self.media, media_site=s.media_sites[pad])
            comp.append(R.two_way_range_m(sol))
            if partials:
                jac.append(R.two_way_range_partial(sol, sc))
        return np.array(comp), (np.array(jac) if partials else None)

    def residual(self, p, partials=True):
        hist = self.fd_history(p[:6], p[6]) if partials else self.propagate(p[:6], p[6], sensitivity=False)
        comp, jac = self.computed(hist, partials)
        res = self.obs - comp - p[7:10][self.pad_index]
        if not partials:
            return res, None, hist
        h = np.zeros((len(self.rows), 10))
        h[:, :7] = jac
        h[np.arange(len(self.rows)), 7 + self.pad_index] = 1.0
        return res, h, hist


def solve(nmax: int, sign: float) -> dict:
    t0 = time.time()
    pb = Problem(nmax)
    x0 = pb.reference_epoch + sign * OFFSET
    p = np.concatenate([x0, [K0], np.zeros(3)])
    res, _, _ = pb.residual(p, partials=False)
    for j in range(3):
        p[7 + j] = float(np.mean(res[pb.pad_index == j]))
    scale = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1e-3, 1.0, 1.0, 1.0])
    lam, status, history = 1e-3, "MaxIter", []
    res, h, hist = pb.residual(p)
    cost = float(res @ res)
    for it in range(1, MAX_ITER + 1):
        hs = h * scale / SIGMA_M
        a = hs.T @ hs
        g = hs.T @ (res / SIGMA_M)
        accepted = False
        while lam < 1e10:
            step = scale * np.linalg.solve(a + lam * np.diag(np.diag(a)), g)
            pc = p + step
            if pc[6] < 0.0:
                lam *= 10.0
                continue
            rc, _, _ = pb.residual(pc, partials=False)
            cc = float(rc @ rc)
            if cc < cost:
                rel = (cost - cc) / cost
                p, cost = pc, cc
                res, h, hist = pb.residual(p)
                lam = max(lam / 10.0, 1e-9)
                accepted = True
                break
            lam *= 10.0
        history.append(dict(iteration=it, cost=cost, rms_m=float(np.sqrt(cost / len(res))), lam=lam,
                            pos_step_m=float(np.linalg.norm(step[:3])) if accepted else None))
        print(f"[N{nmax} {sign:+.0f}] it {it} rms {history[-1]['rms_m']:.3f} m K {p[6]:.5f} "
              f"step {history[-1]['pos_step_m']} lam {lam:.1e} t {time.time() - t0:.0f}s", flush=True)
        if not accepted:
            status = "DampingLimit"
            break
        if rel < 1e-8 and np.linalg.norm(step[:3]) < 1e-3:
            status = "Converged"
            break
    # formal covariance and comparison with the reference over the observed span
    hs = h / SIGMA_M
    cov = np.linalg.inv(hs.T @ hs)
    t_cmp = np.arange(float(pb.setup.t3.min()), float(pb.setup.t3.max()), 60.0)
    est = pb.propagate(p[:6], p[6], t_eval=np.concatenate([[pb.t_epoch], t_cmp]), sensitivity=False)[1:]
    import spiceypy as spice
    ref = np.array([spice.spkezr(F.LRO, pb.setup.et0 + t, "J2000", "NONE", "MOON")[0] for t in t_cmp]) * 1e3
    ric = []
    for e, r in zip(est, ref):
        rh = r[:3] / np.linalg.norm(r[:3])
        ch = np.cross(r[:3], r[3:]); ch /= np.linalg.norm(ch)
        ah = np.cross(ch, rh)
        d = e[:3] - r[:3]
        ric.append([d @ rh, d @ ah, d @ ch])
    ric = np.array(ric)
    tot = np.linalg.norm(ric, axis=1)
    per_pass = []
    for idx in F.passes(pb.rows, pb.setup.t3):
        per_pass.append(dict(pad=pb.rows[idx[0]][0], n=int(idx.size), rms_m=float(np.sqrt(np.mean(res[idx] ** 2))),
                             mean_m=float(np.mean(res[idx]))))
    return dict(nmax=nmax, sign=sign, status=status, iterations=len(history), history=history,
                postfit_rms_m=float(np.sqrt(cost / len(res))), per_pass=per_pass,
                x0=p[:6].tolist(), k_srp=float(p[6]), k_srp_sigma=float(np.sqrt(cov[6, 6])),
                biases_m=dict(zip(map(str, PADS), map(float, p[7:10]))),
                bias_sigma_m=dict(zip(map(str, PADS), map(float, np.sqrt(np.diag(cov)[7:10])))),
                x0_minus_reference=(p[:6] - pb.reference_epoch).tolist(),
                ric_rms_m=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(), ric_max_m=np.max(np.abs(ric), axis=0).tolist(),
                total_rms_m=float(np.sqrt(np.mean(tot ** 2))), total_max_m=float(tot.max()),
                est_positions=est[:, :3].tolist(), elapsed_s=time.time() - t0)


def g4_partials() -> dict:
    pb = Problem(60)
    p = np.concatenate([pb.reference_epoch, [K0], np.zeros(3)])
    idx = list(np.linspace(0, len(pb.rows) - 1, 10).astype(int))
    hist = pb.fd_history(p[:6], p[6])
    comp, jac = pb.computed(hist, True, idx)
    steps = list(FD_STEPS)
    fd = np.zeros((len(idx), 7))
    for j, hstep in enumerate(steps):
        for sgn in (1.0, -1.0):
            q = p.copy()
            q[j] += sgn * hstep
            hq = pb.propagate(q[:6], q[6], sensitivity=False)
            cq, _ = pb.computed(hq, False, idx)
            fd[:, j] += sgn * cq / (2.0 * hstep)
    rel = np.abs(jac - fd) / np.maximum(np.abs(fd), 1e-12)
    col = [float(np.linalg.norm(jac[:, j] - fd[:, j]) / np.linalg.norm(fd[:, j])) for j in range(7)]
    return dict(column_relative_error=col, max=float(max(col)), PASS=max(col) <= 1e-4,
                worst_element_relative=float(rel.max()))


def main():
    with ProcessPoolExecutor(4) as ex:
        futs = {("N100", 1.0): ex.submit(solve, 100, 1.0), ("N100", -1.0): ex.submit(solve, 100, -1.0),
                ("N60", 1.0): ex.submit(solve, 60, 1.0)}
        fg4 = ex.submit(g4_partials)
        runs = {f"{k[0]}_{'+' if k[1] > 0 else '-'}": f.result() for k, f in futs.items()}
        g4 = fg4.result()
    a, b = runs["N100_+"], runs["N100_-"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    dbias = max(abs(a["biases_m"][k] - b["biases_m"][k]) for k in a["biases_m"])
    gates = dict(G1=a["status"] == "Converged" and b["status"] == "Converged",
                 G2=dpos <= 1e-2 and dbias <= 1e-3, G3=a["postfit_rms_m"] <= 2.0, G4=g4["PASS"])
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2_lro_od_predeclaration.json", gates=gates,
               G2_start_difference=dict(position_max_m=dpos, bias_max_m=dbias), G4=g4, runs=runs)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: {kk: r[kk] for kk in ("status", "iterations", "postfit_rms_m", "k_srp", "k_srp_sigma", "biases_m",
                                       "ric_rms_m", "ric_max_m", "total_rms_m", "total_max_m", "elapsed_s")}
             for k, r in runs.items()}
    print(json.dumps(dict(gates=gates, G2=out["G2_start_difference"], G4=g4, runs=brief), indent=1))


if __name__ == "__main__":
    main()
