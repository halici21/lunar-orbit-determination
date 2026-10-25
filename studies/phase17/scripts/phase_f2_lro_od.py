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


#: Phase F2d: Doppler weight (F1c post-offset scatter 1-3 mm/s).
SIGMA_DOP = 2e-3
#: Phase F2e A1: observed |range rate| above this is physically impossible for LRO seen from Earth.
MAX_DOPPLER_MPS = 3000.0
#: Parallel evaluation: measurements per worker task.
MEASURE_CHUNK = 300

# ----------------------------------------------------------------------
# Parallel evaluation (behaviour-preserving): each worker process holds its own Problem, built once by the
# initializer with the same arguments as the main one, and runs exactly the serial code on its share.
# ----------------------------------------------------------------------
_WORKER: dict = {}


def _worker_init(nmax, doppler, days, n_emp=0):
    import warnings
    warnings.simplefilter("ignore")
    _WORKER["pb"] = Problem(nmax, doppler=doppler, days=days, n_emp=n_emp)


def _worker_propagate(q):
    return _WORKER["pb"].propagate(q[:6], q[6], sensitivity=False, emp=q[7:] if len(q) > 7 else None)


# ----------------------------------------------------------------------
# Phase F2g: empirical once-per-revolution accelerations (study-level model; frozen dynamics.py unchanged)
# ----------------------------------------------------------------------
#: a_emp = (C_T cos u + S_T sin u) T + (C_N cos u + S_N sin u) N, u = argument of latitude in J2000,
#: T = N x R (along-track), N = h / |h| (cross-track).
EMP_NAMES = ("C_T", "S_T", "C_N", "S_N")
EMP_FD_STEP = 1e-9      # m/s^2
EMP_SCALE = 1e-9        # m/s^2, LM column scale
EMP_PRIOR_SIGMA = 1e-8  # m/s^2


def empirical_acceleration(y, emp) -> np.ndarray:
    r, v = np.asarray(y[:3], float), np.asarray(y[3:6], float)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    r_hat = r / np.linalg.norm(r)
    t_hat = np.cross(n_hat, r_hat)
    node = np.cross(np.array([0.0, 0.0, 1.0]), h)
    node /= np.linalg.norm(node)
    cos_u = float(node @ r_hat)
    sin_u = float(np.cross(node, r_hat) @ n_hat)
    return (emp[0] * cos_u + emp[1] * sin_u) * t_hat + (emp[2] * cos_u + emp[3] * sin_u) * n_hat


def _spacecraft_history(grid, hist, n_emp: int):
    """SpacecraftHistory whose position sensitivity also carries the empirical-parameter columns (F2g): the
    light-time partials (_round_trip_partial) are linear in that matrix, so extra columns pass straight through."""
    from lunar_od import relativistic_light_time as R
    if n_emp == 0 or hist.shape[1] <= 48:
        return R.SpacecraftHistory(grid, hist)

    class _WithEmpirical(R.SpacecraftHistory):
        def __init__(self):
            super().__init__(grid, hist[:, :48])
            self.extra = hist[:, 48:48 + 6 * n_emp]

        def position_sensitivity(self, t_s):
            base = super().position_sensitivity(t_s)
            cols = [R._interp_state(self.t_grid, self.extra[:, 6 * i:6 * i + 6], t_s)[:3].reshape(3, 1)
                    for i in range(n_emp)]
            return np.hstack([base] + cols)

    return _WithEmpirical()


def _worker_measure(task):
    kind, hist, lo, hi, partials = task
    pb = _WORKER["pb"]
    if kind == "range":
        return pb.computed(hist, partials, idx=range(lo, hi))
    return pb.computed_doppler(hist, partials, idx=range(lo, hi))


class Problem:
    def __init__(self, nmax: int, doppler: bool = False, days: tuple = (F.DAY_DIR,), n_emp: int = 0):
        import spiceypy as spice
        from lunar_od import relativistic_light_time as R
        from lunar_od import station_time as ST
        from lunar_od.ephemeris import sample_moon_centered_ephemeris
        from lunar_od.gravity_model_loader import load_lunar_gravity_model
        from lunar_od.project_paths import gravity_data_root
        self.data = Path(F.REPO.parent / "lro_data")
        self.nmax = nmax
        self.n_emp = int(n_emp)
        self.pool = None
        self.days = tuple(days)
        self.rows = []
        for day in self.days:                    # F2e: several day directories in one arc
            self.rows += F.read_day(self.data, day)[0]
        self.setup = F.Setup(self.data, self.rows, "DE440")
        s = self.setup
        self.obs = np.array([F.C * r.rtlt_s / 2.0 for _, r in self.rows])
        self.pad_index = np.array([PADS.index(p) for p, _ in self.rows])
        self.use_doppler = doppler
        t_all = [float(s.t3.min()), float(s.t3.max())]
        if doppler:
            import phase_f1c_lro_doppler as D
            from lunar_od.media import MediaModel
            drows = []
            for day in self.days:
                drows += D.read_doppler(self.data, day)
            drows.sort(key=lambda x: (x[0], x[1].utc))
            td = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[p])[0]
                           for p, r in drows])
            self.dop = []
            for i, j in D.intervals(drows, td):
                pad, ra = drows[i]
                rb = drows[j][1]
                a = (rb.doppler_count_raw - ra.doppler_count_raw) / (rb.utc - ra.utc).total_seconds()
                ft = ra.transmit_frequency_hz
                obs = -F.C * (a - D.BIAS_HZ) / (2.0 * ft * D.TURNAROUND * D.M_MULT)
                if abs(obs) > MAX_DOPPLER_MPS:     # F2e A1: physically impossible interval
                    continue
                self.dop.append(dict(pad=pad, tm=0.5 * (td[i] + td[j]), tc=td[j] - td[i], obs=obs,
                                     media=MediaModel(uplink_hz=ft, downlink_hz=ft * D.TURNAROUND)))
            t_all += [min(d["tm"] - d["tc"] for d in self.dop), max(d["tm"] + d["tc"] for d in self.dop)]
        self.t_epoch = min(t_all) - EPOCH_LEAD_S
        self.t_end = max(t_all) + 10.0
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

    def propagate(self, x0, k, t_eval=None, sensitivity=True, emp=None):
        from lunar_od.dynamics import PropagationProfile, propagate_state, propagate_state_with_k_sensitivity
        from lunar_od.srp import SRPOptions
        kw = dict(srp=SRPOptions(k_srp_m2_per_kg=float(k)), rtol=RTOL, atol=ATOL, harmonic_model=self.model,
                  harmonic_epoch_et0=self.setup.et0, **PropagationProfile().propagator_kwargs())
        t = self.grid if t_eval is None else t_eval
        args = (t, np.asarray(x0, float), self.mu_m, self.mu_e, self.mu_s, self.eph.earth_position,
                self.eph.sun_position)
        if emp is not None:
            return self._propagate_empirical(np.asarray(t, float), np.asarray(x0, float), float(k), emp)
        if sensitivity:
            return np.asarray(propagate_state_with_k_sensitivity(*args, harmonic_stm_opt_in=True, **kw))
        return np.asarray(propagate_state(*args, **kw))

    def _propagate_empirical(self, t, x0, k, emp):
        """F2g: production propagate_state's own RHS (dynamics._build_state_rhs) plus the empirical acceleration,
        integrated with production's shadow-event segmentation (DOP853). With emp = 0 this reproduces
        propagate_state(method='DOP853', shadow_events=True) bit for bit (gate G0)."""
        from lunar_od import dynamics as dyn
        from lunar_od.srp import SRPOptions
        srp = SRPOptions(k_srp_m2_per_kg=k)
        rhs, get_sun = dyn._build_state_rhs(t, self.mu_m, self.mu_e, self.mu_s, self.eph.earth_position,
                                            self.eph.sun_position, 0.0, 0.0, "indirect", self.model, None,
                                            self.setup.et0, srp)
        e = np.asarray(emp, float)

        def rhs_e(tt, y):
            out = np.array(rhs(tt, y), dtype=float, copy=True)
            out[3:6] += empirical_acceleration(y, e)
            return out

        cuts = dyn._shadow_contact_times(rhs_e, t, x0, get_sun, srp, RTOL, ATOL)
        return np.asarray(dyn._integrate_segmented(t, x0, rhs_e, RTOL, ATOL, "DOP853", cuts,
                                                   "F2g empirical propagation",
                                                   contact_check=dyn._shadow_contact_check(get_sun, srp)))

    def emp_of(self, p):
        return p[11:11 + self.n_emp] if self.n_emp else None

    def propagate_p(self, p, t_eval=None):
        """6-state history for a full parameter vector (state, K and, F2g, the empirical accelerations)."""
        return self.propagate(p[:6], p[6], t_eval=t_eval, sensitivity=False, emp=self.emp_of(p))

    def fd_history(self, x0, k, emp=None) -> np.ndarray:
        """A1: [x, Phi (column-major), dx/dK (, dx/d emp_i)] from central differences of the 6-state propagator.

        The propagations (base, then plus/minus per parameter; plus only for K at its bound) are
        independent; with a worker pool attached they run in parallel, otherwise in this order serially.
        """
        q0 = np.concatenate([np.asarray(x0, float), [float(k)]] + ([np.asarray(emp, float)] if emp is not None
                                                                  else []))
        steps = list(FD_STEPS) + [EMP_FD_STEP] * (0 if emp is None else len(emp))
        qs, plan = [q0], []
        for j in range(len(steps)):
            hj = steps[j]
            plus = q0.copy()
            plus[j] += hj
            qs.append(plus)
            # K must stay non-negative (SRPOptions refuses K < 0): forward difference near the bound
            forward = j == 6 and float(k) < hj
            if not forward:
                minus = q0.copy()
                minus[j] -= hj
                qs.append(minus)
            plan.append((j, hj, forward))
        if self.pool is not None:
            hists = list(self.pool.map(_worker_propagate, qs))
        else:
            hists = [self.propagate(q[:6], q[6], sensitivity=False, emp=q[7:] if len(q) > 7 else None)
                     for q in qs]
        base = hists[0]
        n = base.shape[0]
        cols = np.zeros((n, 6, len(steps)))
        pos = 1
        for j, hj, forward in plan:
            plus = hists[pos]
            pos += 1
            if forward:
                cols[:, :, j] = (plus - base) / hj
                continue
            minus = hists[pos]
            pos += 1
            cols[:, :, j] = (plus - minus) / (2.0 * hj)
        phi = cols[:, :, :6]
        extra = [cols[:, :, j] for j in range(7, len(steps))]
        return np.hstack([base, phi.transpose(0, 2, 1).reshape(n, 36), cols[:, :, 6]] + extra)

    def attach_pool(self, workers: int) -> None:
        """Parallel evaluation: each worker builds its own identical Problem once (kernels, field, setup)."""
        self.pool = ProcessPoolExecutor(workers, initializer=_worker_init,
                                        initargs=(self.nmax, self.use_doppler, self.days, self.n_emp))

    def close_pool(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def _parallel_measure(self, kind: str, hist, partials: bool, n: int):
        tasks = [(kind, hist, lo, min(lo + MEASURE_CHUNK, n), partials) for lo in range(0, n, MEASURE_CHUNK)]
        parts = list(self.pool.map(_worker_measure, tasks))
        comp = np.concatenate([c for c, _ in parts])
        jac = np.concatenate([j for _, j in parts]) if partials else None
        return comp, jac

    def computed(self, hist, partials=True, idx=None):
        if idx is None and self.pool is not None:
            return self._parallel_measure("range", hist, partials, len(self.rows))
        R, s = self.R, self.setup
        sc = _spacecraft_history(self.grid, hist, self.n_emp)
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

    def computed_doppler(self, hist, partials=True, idx=None):
        """F2d: counted Doppler of every interval [m/s] and its partials (7 columns)."""
        if idx is None and self.pool is not None:
            return self._parallel_measure("doppler", hist, partials, len(self.dop))
        R, s = self.R, self.setup
        sc = _spacecraft_history(self.grid, hist, self.n_emp)
        comp, jac = [], []
        for d in (self.dop if idx is None else [self.dop[i] for i in idx]):
            pad = d["pad"]
            val, part, _ = R.counted_doppler(d["tm"], d["tc"], s.tide[pad], sc, s.eph, R.DSN_C2_LIGHT_TIME_MODEL,
                                             with_partial=partials, site=s.sites[pad], media=d["media"],
                                             media_site=s.media_sites[pad])
            comp.append(val)
            if partials:
                jac.append(part)
        return np.array(comp), (np.array(jac) if partials else None)

    @property
    def n_par(self) -> int:
        """[x0 (6), K, range biases (3)] (+ Doppler bias, F2d) (+ empirical accelerations, F2g at 11..)."""
        if self.n_emp and not self.use_doppler:
            raise ValueError("the empirical-acceleration layout (indices 11..) assumes the Doppler bias at 10")
        return (11 if self.use_doppler else 10) + self.n_emp

    def _dyn_columns(self, hh, jac, w=1.0):
        """Place the dynamical partials: x0 and K at 0..6, the empirical accelerations (F2g) at 11.."""
        hh[:, :7] = jac[:, :7] * w
        if self.n_emp:
            hh[:, 11:11 + self.n_emp] = jac[:, 7:7 + self.n_emp] * w

    def residual(self, p, partials=True):
        """Range rows [m] followed (F2d) by Doppler rows scaled to range units by SIGMA_M / SIGMA_DOP."""
        hist = self.fd_history(p[:6], p[6], self.emp_of(p)) if partials else self.propagate_p(p)
        comp, jac = self.computed(hist, partials)
        res = self.obs - comp - p[7:10][self.pad_index]
        nr = len(self.rows)
        h = None
        if partials:
            h = np.zeros((nr, self.n_par))
            self._dyn_columns(h, jac)
            h[np.arange(nr), 7 + self.pad_index] = 1.0
        if self.use_doppler:
            w = SIGMA_M / SIGMA_DOP
            cd, jd = self.computed_doppler(hist, partials)
            obs_d = np.array([d["obs"] for d in self.dop])
            res = np.concatenate([res, (obs_d - cd - p[10]) * w])
            if partials:
                hd = np.zeros((len(self.dop), self.n_par))
                self._dyn_columns(hd, jd, w)
                hd[:, 10] = w
                h = np.vstack([h, hd])
        return res, h, hist


def solve(nmax: int, sign: float, priors: dict | None = None, tag: str = "", doppler: bool = False,
          days: tuple = (F.DAY_DIR,), x_start=None, predict_days: tuple = (), workers: int = 0,
          n_emp: int = 0) -> dict:
    t0 = time.time()
    pb = Problem(nmax, doppler=doppler, days=days, n_emp=n_emp)
    if workers > 1:
        pb.attach_pool(workers)
    nr = len(pb.rows)
    x0 = pb.reference_epoch + sign * OFFSET if x_start is None else np.asarray(x_start, float)
    p = np.concatenate([x0, [K0], np.zeros(pb.n_par - 7)])
    res, _, _ = pb.residual(p, partials=False)
    for j in range(3):
        p[7 + j] = float(np.mean(res[:nr][pb.pad_index == j]))
    if doppler:
        p[10] = float(np.mean(res[nr:])) * SIGMA_DOP / SIGMA_M
    # F2b: priors as pseudo-observations {parameter index: (mean, sigma)}, weighted against SIGMA_M
    prior_items = sorted((priors or {}).items())
    for j, (mean, _) in prior_items:
        if j >= 7:
            p[j] = mean

    def aug(r, hh, pp):
        if not prior_items:
            return r, hh
        w = np.array([SIGMA_M / s for _, (_, s) in prior_items])
        r_p = np.array([m - pp[j] for j, (m, _) in prior_items]) * w
        r_out = np.concatenate([r, r_p])
        if hh is None:
            return r_out, None
        h_p = np.zeros((len(prior_items), hh.shape[1]))
        for row, (j, _) in enumerate(prior_items):
            h_p[row, j] = w[row]
        return r_out, np.vstack([hh, h_p])

    scale = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1e-3, 1.0, 1.0, 1.0] + ([1e-3] if doppler else [])
                     + [EMP_SCALE] * pb.n_emp)
    lam, status, history = 1e-3, "MaxIter", []
    rel = 1.0
    res, h, hist = pb.residual(p)
    ra, ha = aug(res, h, p)
    cost = float(ra @ ra)
    for it in range(1, MAX_ITER + 1):
        hs = ha * scale / SIGMA_M
        a = hs.T @ hs
        g = hs.T @ (ra / SIGMA_M)
        accepted = False
        while lam < 1e10:
            step = scale * np.linalg.solve(a + lam * np.diag(np.diag(a)), g)
            pc = p + step
            if pc[6] < 0.0:
                lam *= 10.0
                continue
            rc, _, _ = pb.residual(pc, partials=False)
            rca, _ = aug(rc, None, pc)
            cc = float(rca @ rca)
            if cc < cost:
                rel = (cost - cc) / cost
                p, cost = pc, cc
                res, h, hist = pb.residual(p)
                ra, ha = aug(res, h, p)
                lam = max(lam / 10.0, 1e-9)
                accepted = True
                break
            lam *= 10.0
        history.append(dict(iteration=it, cost=cost, rms_m=float(np.sqrt(np.mean(res[:nr] ** 2))), lam=lam,
                            pos_step_m=float(np.linalg.norm(step[:3])) if accepted else None))
        if doppler:
            history[-1]["doppler_rms_mmps"] = float(np.sqrt(np.mean(res[nr:] ** 2)) * SIGMA_DOP / SIGMA_M * 1e3)
        with open(F.REPO.parent / f"f2{tag}_progress_N{nmax}_{'p' if sign > 0 else 'm'}.log", "a", encoding="utf-8") as log:
            log.write(f"it {it} rms {history[-1]['rms_m']:.3f} m K {p[6]:.5f} step {history[-1]['pos_step_m']} "
                      f"lam {lam:.1e} t {time.time() - t0:.0f}s\n")
        if not accepted:
            # F2d: the damping limit after a near-zero relative improvement is a stationary point
            status = "STATIONARY" if (doppler and rel < 1e-6) else "DampingLimit"
            break
        if rel < 1e-8 and np.linalg.norm(step[:3]) < 1e-3:
            status = "Converged"
            break
    # formal covariance and comparison with the reference over the observed span
    hs = ha / SIGMA_M
    cov = np.linalg.inv(hs.T @ hs)
    t_cmp = np.arange(float(pb.setup.t3.min()), float(pb.setup.t3.max()), 60.0)
    est = pb.propagate_p(p, t_eval=np.concatenate([[pb.t_epoch], t_cmp]))[1:]
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
    extra = {}
    if doppler:
        rd = res[nr:] * SIGMA_DOP / SIGMA_M
        extra = dict(doppler_bias_mmps=float(p[10] * 1e3), doppler_bias_sigma_mmps=float(np.sqrt(cov[10, 10]) * 1e3),
                     doppler_postfit_rms_mmps=float(np.sqrt(np.mean(rd ** 2)) * 1e3), n_doppler=int(rd.size))
    return dict(nmax=nmax, sign=sign, status=status, iterations=len(history), history=history, **extra,
                postfit_rms_m=float(np.sqrt(np.mean(res[:nr] ** 2))), per_pass=per_pass,
                x0=p[:6].tolist(), k_srp=float(p[6]), k_srp_sigma=float(np.sqrt(cov[6, 6])),
                biases_m=dict(zip(map(str, PADS), map(float, p[7:10]))),
                empirical_mps2=(dict(zip(EMP_NAMES, map(float, p[11:11 + pb.n_emp]))) if pb.n_emp else None),
                empirical_sigma_mps2=(dict(zip(EMP_NAMES, map(float, np.sqrt(np.diag(cov)[11:11 + pb.n_emp]))))
                                      if pb.n_emp else None),
                bias_sigma_m=dict(zip(map(str, PADS), map(float, np.sqrt(np.diag(cov)[7:10])))),
                x0_minus_reference=(p[:6] - pb.reference_epoch).tolist(),
                ric_rms_m=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(), ric_max_m=np.max(np.abs(ric), axis=0).tolist(),
                total_rms_m=float(np.sqrt(np.mean(tot ** 2))), total_max_m=float(tot.max()),
                est_positions=est[:, :3].tolist(),
                prediction=_predict_and_close(pb, p, predict_days),
                elapsed_s=time.time() - t0)


def _predict_and_close(pb: Problem, p, predict_days: tuple):
    try:
        return predict(pb, p, predict_days) if predict_days else None
    finally:
        pb.close_pool()


def _ric(est, ref):
    out = []
    for e, r in zip(est, ref):
        rh = r[:3] / np.linalg.norm(r[:3])
        ch = np.cross(r[:3], r[3:])
        ch /= np.linalg.norm(ch)
        ah = np.cross(ch, rh)
        d = e[:3] - r[:3]
        out.append([d @ rh, d @ ah, d @ ch])
    return np.array(out)


def predict(pb: Problem, p, predict_days: tuple) -> dict:
    """F2e: propagate the fitted solution over days it did not see and score it there."""
    import spiceypy as spice
    from lunar_od import station_time as ST
    import phase_f1c_lro_doppler as D
    R, s = pb.R, pb.setup
    rows = []
    for day in predict_days:
        rows += F.read_day(pb.data, day)[0]
    t3 = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                   for q, r in rows])
    drows = []
    for day in predict_days:
        drows += D.read_doppler(pb.data, day)
    drows.sort(key=lambda x: (x[0], x[1].utc))
    td = np.array([ST.utc_to_tdb_seconds([r.utc.strftime("%Y-%m-%dT%H:%M:%S.%f")], s.et0, s.sites[q])[0]
                   for q, r in drows])
    grid = np.arange(pb.t_epoch, max(t3.max(), td.max()) + 10.0 + GRID_S, GRID_S)
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    # the fit sampled Earth/Sun over the fit span only; cover the prediction span as well
    pb.eph = sample_moon_centered_ephemeris(s.et0, np.arange(pb.t_epoch - 600.0, grid[-1] + 600.0, 60.0))
    hist = pb.propagate_p(p, t_eval=grid)
    sc = R.SpacecraftHistory(grid, hist)
    res_r = []
    for (q, r), t in zip(rows, t3):
        sol = R.solve_two_way(float(t), s.tide[q], sc, s.eph, R.DSN_C2_LIGHT_TIME_MODEL, site=s.sites[q],
                              media=pb.media, media_site=s.media_sites[q])
        res_r.append(F.C * r.rtlt_s / 2.0 - R.two_way_range_m(sol) - p[7 + PADS.index(q)])
    res_r = np.array(res_r)
    res_d, td_mid = [], []
    for i, j in D.intervals(drows, td):
        q, ra = drows[i]
        rb = drows[j][1]
        a = (rb.doppler_count_raw - ra.doppler_count_raw) / (rb.utc - ra.utc).total_seconds()
        ft = ra.transmit_frequency_hz
        obs = -F.C * (a - D.BIAS_HZ) / (2.0 * ft * D.TURNAROUND * D.M_MULT)
        if abs(obs) > MAX_DOPPLER_MPS:             # F2e A1
            continue
        from lunar_od.media import MediaModel
        val, _, _ = R.counted_doppler(0.5 * (td[i] + td[j]), td[j] - td[i], s.tide[q], sc, s.eph,
                                      R.DSN_C2_LIGHT_TIME_MODEL, site=s.sites[q],
                                      media=MediaModel(uplink_hz=ft, downlink_hz=ft * D.TURNAROUND),
                                      media_site=s.media_sites[q])
        res_d.append(obs - val - (p[10] if pb.use_doppler else 0.0))
        td_mid.append(0.5 * (td[i] + td[j]))
    res_d = np.array(res_d)
    passes = []
    for idx in F.passes(rows, t3):
        passes.append(dict(pad=rows[idx[0]][0], n=int(idx.size), hours_after_fit_end=float(
            (t3[idx].min() - s.t3.max()) / 3600.0), range_mean_m=float(res_r[idx].mean()),
            range_rms_m=float(np.sqrt(np.mean(res_r[idx] ** 2)))))
    t_cmp = np.arange(float(t3.min()), float(t3.max()), 60.0)
    est = pb.propagate_p(p, t_eval=np.concatenate([[pb.t_epoch], t_cmp]))[1:]
    ref = np.array([spice.spkezr(F.LRO, s.et0 + t, "J2000", "NONE", "MOON")[0] for t in t_cmp]) * 1e3
    ric = _ric(est, ref)
    return dict(n_range=int(res_r.size), n_doppler=int(res_d.size),
                range_rms_m=float(np.sqrt(np.mean(res_r ** 2))), range_mean_m=float(res_r.mean()),
                doppler_rms_mmps=float(np.sqrt(np.mean(res_d ** 2)) * 1e3), doppler_mean_mmps=float(res_d.mean() * 1e3),
                passes=passes, ric_rms_m=np.sqrt(np.mean(ric ** 2, axis=0)).tolist(),
                ric_max_m=np.max(np.abs(ric), axis=0).tolist())


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


#: Phase F2b priors (studies/phase17/configs/phase_f2b_lro_od_priors_predeclaration.json): parameter index ->
#: (mean, sigma). Bias means are F1's M_FULL per-station mean residuals on 2009-260.
F2B_PRIORS = {6: (0.010, 0.005), 7: (489.1605710824553, 3.0), 8: (476.89111818207635, 3.0),
              9: (474.5091541284396, 3.0)}
OUT_F2B = HERE.parent / "results" / "phase_f2b_lro_od_priors.json"


def main_f2b():
    with ProcessPoolExecutor(2) as ex:
        fp = ex.submit(solve, 100, 1.0, F2B_PRIORS, "b")
        fm = ex.submit(solve, 100, -1.0, F2B_PRIORS, "b")
        a, b = fp.result(), fm.result()
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    dbias = max(abs(a["biases_m"][k] - b["biases_m"][k]) for k in a["biases_m"])
    gates = dict(G1=a["status"] == "Converged" and b["status"] == "Converged",
                 G2=dpos <= 1.0 and dbias <= 0.1, G3=a["postfit_rms_m"] <= 20.0 and b["postfit_rms_m"] <= 20.0)
    for r in (a, b):
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2b_lro_od_priors_predeclaration.json", gates=gates,
               G2_start_difference=dict(position_max_m=dpos, bias_max_m=dbias), priors=F2B_PRIORS,
               runs={"N100_+": a, "N100_-": b})
    OUT_F2B.write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: {kk: r[kk] for kk in ("status", "iterations", "postfit_rms_m", "k_srp", "k_srp_sigma", "biases_m",
                                       "ric_rms_m", "ric_max_m", "total_rms_m", "total_max_m", "elapsed_s")}
             for k, r in out["runs"].items()}
    print(json.dumps(dict(gates=gates, G2=out["G2_start_difference"], runs=brief), indent=1))


OUT_F2C = HERE.parent / "results" / "phase_f2c_lro_od_degree.json"


def main_f2c():
    """Phase F2c (studies/phase17/configs/phase_f2c_lro_od_degree_predeclaration.json): F2b at N = 200, 300."""
    with ProcessPoolExecutor(3) as ex:
        futs = {"N200_+": ex.submit(solve, 200, 1.0, F2B_PRIORS, "c"),
                "N200_-": ex.submit(solve, 200, -1.0, F2B_PRIORS, "c"),
                "N300_+": ex.submit(solve, 300, 1.0, F2B_PRIORS, "c")}
        runs = {k: f.result() for k, f in futs.items()}
    a, b = runs["N200_+"], runs["N200_-"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    dbias = max(abs(a["biases_m"][k] - b["biases_m"][k]) for k in a["biases_m"])
    gates = dict(G1=a["status"] == "Converged" and b["status"] == "Converged",
                 G2=dpos <= 1.0 and dbias <= 0.1, G3=all(r["postfit_rms_m"] <= 5.0 for r in runs.values()))
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2c_lro_od_degree_predeclaration.json", gates=gates,
               G2_start_difference=dict(position_max_m=dpos, bias_max_m=dbias), priors=F2B_PRIORS, runs=runs)
    OUT_F2C.write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: {kk: r[kk] for kk in ("status", "iterations", "postfit_rms_m", "k_srp", "k_srp_sigma", "biases_m",
                                       "ric_rms_m", "ric_max_m", "total_rms_m", "total_max_m", "elapsed_s")}
             for k, r in runs.items()}
    print(json.dumps(dict(gates=gates, G2=out["G2_start_difference"], runs=brief), indent=1))


OUT_F2D = HERE.parent / "results" / "phase_f2d_lro_od_range_doppler.json"


def main_f2d():
    """Phase F2d (studies/phase17/configs/phase_f2d_lro_od_range_doppler_predeclaration.json)."""
    with ProcessPoolExecutor(2) as ex:
        futs = {"N200_+": ex.submit(solve, 200, 1.0, F2B_PRIORS, "d", True),
                "N200_-": ex.submit(solve, 200, -1.0, F2B_PRIORS, "d", True)}
        runs = {k: f.result() for k, f in futs.items()}
    a, b = runs["N200_+"], runs["N200_-"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    gates = dict(G1=all(r["status"] in ("Converged", "STATIONARY") for r in runs.values()), G2=dpos <= 1.0,
                 G3=all(r["postfit_rms_m"] <= 2.0 and r["doppler_postfit_rms_mmps"] <= 5.0 for r in runs.values()))
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2d_lro_od_range_doppler_predeclaration.json",
               gates=gates, G2_start_difference_m=dpos, priors=F2B_PRIORS, sigma_doppler_mps=SIGMA_DOP, runs=runs)
    OUT_F2D.write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: {kk: r[kk] for kk in ("status", "iterations", "postfit_rms_m", "doppler_postfit_rms_mmps",
                                       "doppler_bias_mmps", "k_srp", "biases_m", "ric_rms_m", "ric_max_m",
                                       "total_rms_m", "total_max_m", "elapsed_s")}
             for k, r in runs.items()}
    print(json.dumps(dict(gates=gates, G2=dpos, runs=brief), indent=1))


OUT_F2E = HERE.parent / "results" / "phase_f2e_lro_two_day_arc.json"


def _workers_arg() -> int:
    """``--workers N`` on the command line: parallel evaluation with N worker processes (0: serial)."""
    if "--workers" in sys.argv:
        return int(sys.argv[sys.argv.index("--workers") + 1])
    return 0
F2E_FIT_DAYS = ("trk/lro_no_01/2009260", "trk/lro_no_01/2009261")
F2E_PREDICT_DAYS = ("trk/lro_no_01/2009262",)


def main_f2e():
    """Phase F2e (studies/phase17/configs/phase_f2e_lro_two_day_arc_predeclaration.json)."""
    pre = json.loads((HERE.parent / "configs" / "phase_f2e_lro_two_day_arc_predeclaration.json").read_text(
        encoding="utf-8"))
    s1 = np.array(json.loads(pre["starts"]["S1"].split(": ", 1)[1]), float)
    s2 = s1 + np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
    workers = _workers_arg()
    if workers > 1:
        # one start after the other, each with the whole worker pool (identical numbers, see
        # phase_f2_parallel_equivalence.py)
        runs = {"S1": solve(200, 1.0, F2B_PRIORS, "e1", True, F2E_FIT_DAYS, s1, F2E_PREDICT_DAYS, workers),
                "S2": solve(200, 1.0, F2B_PRIORS, "e2", True, F2E_FIT_DAYS, s2, (), workers)}
    else:
        with ProcessPoolExecutor(2) as ex:
            futs = {"S1": ex.submit(solve, 200, 1.0, F2B_PRIORS, "e1", True, F2E_FIT_DAYS, s1, F2E_PREDICT_DAYS),
                    "S2": ex.submit(solve, 200, 1.0, F2B_PRIORS, "e2", True, F2E_FIT_DAYS, s2, ())}
            runs = {k: f.result() for k, f in futs.items()}
    a, b = runs["S1"], runs["S2"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    gates = dict(G1=all(r["status"] in ("Converged", "STATIONARY") for r in runs.values()), G2=dpos <= 1.0,
                 G3=all(r["postfit_rms_m"] <= 2.0 and r["doppler_postfit_rms_mmps"] <= 5.0 for r in runs.values()))
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2e_lro_two_day_arc_predeclaration.json",
               gates=gates, G2_start_difference_m=dpos, priors=F2B_PRIORS, fit_days=F2E_FIT_DAYS,
               predict_days=F2E_PREDICT_DAYS, runs=runs)
    OUT_F2E.write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: {kk: r[kk] for kk in ("status", "iterations", "postfit_rms_m", "doppler_postfit_rms_mmps",
                                       "doppler_bias_mmps", "k_srp", "biases_m", "ric_rms_m", "total_rms_m")}
             for k, r in runs.items()}
    print(json.dumps(dict(gates=gates, G2=dpos, runs=brief, prediction=a["prediction"]), indent=1))


OUT_F2F = HERE.parent / "results" / "phase_f2f_lro_degree300_continuation.json"


def _f2f_part_path(name: str) -> Path:
    return HERE.parent / "results" / f"phase_f2f_part_{name}.json"


def main_f2f_part(name: str):
    """One F2f start in its own OS process (CSPICE is not thread-safe, so the starts cannot share a process)."""
    pre = json.loads((HERE.parent / "configs" / "phase_f2e_lro_two_day_arc_predeclaration.json").read_text(
        encoding="utf-8"))
    s1 = np.array(json.loads(pre["starts"]["S1"].split(": ", 1)[1]), float)
    s2 = s1 + np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
    workers = _workers_arg() or 7
    if name == "s1":
        run = solve(300, 1.0, F2B_PRIORS, "f1", True, F2E_FIT_DAYS, s1, F2E_PREDICT_DAYS, workers)
    else:
        one = solve(300, 1.0, F2B_PRIORS, "f2a", True, (F.DAY_DIR,), s2, (), workers)
        run = solve(300, 1.0, F2B_PRIORS, "f2b", True, F2E_FIT_DAYS, np.array(one["x0"]), (), workers)
        one.pop("est_positions")
        run["continuation_stage1"] = one
    _f2f_part_path(name).write_text(json.dumps(run), encoding="utf-8")


def main_f2f():
    """Phase F2f (studies/phase17/configs/phase_f2f_lro_degree300_continuation_predeclaration.json): merge the
    two parts written by ``--f2f-part s1`` and ``--f2f-part s2``."""
    runs = {"S1": json.loads(_f2f_part_path("s1").read_text(encoding="utf-8")),
            "S2": json.loads(_f2f_part_path("s2").read_text(encoding="utf-8"))}
    a, b = runs["S1"], runs["S2"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    gates = dict(G1=all(r["status"] in ("Converged", "STATIONARY") for r in runs.values()), G2=dpos <= 1.0,
                 G3=all(r["postfit_rms_m"] <= 2.0 and r["doppler_postfit_rms_mmps"] <= 5.0 for r in runs.values()))
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2f_lro_degree300_continuation_predeclaration.json",
               gates=gates, G2_start_difference_m=dpos, priors=F2B_PRIORS, fit_days=F2E_FIT_DAYS,
               predict_days=F2E_PREDICT_DAYS, nmax=300, runs=runs)
    OUT_F2F.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(gates=gates, G2=dpos), indent=1))


#: Phase F2g: F2b's priors plus N(0, EMP_PRIOR_SIGMA) on each empirical acceleration (indices 11..14).
F2G_PRIORS = {**F2B_PRIORS, **{11 + i: (0.0, EMP_PRIOR_SIGMA) for i in range(len(EMP_NAMES))}}
OUT_F2G = HERE.parent / "results" / "phase_f2g_lro_empirical.json"


def _f2g_part_path(name: str) -> Path:
    return HERE.parent / "results" / f"phase_f2g_part_{name}.json"


def main_f2g_part(name: str):
    """One F2g start in its own OS process (empirical 1/rev accelerations, N = 300)."""
    pre = json.loads((HERE.parent / "configs" / "phase_f2e_lro_two_day_arc_predeclaration.json").read_text(
        encoding="utf-8"))
    s1 = np.array(json.loads(pre["starts"]["S1"].split(": ", 1)[1]), float)
    s2 = s1 + np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
    workers = _workers_arg() or 7
    ne = len(EMP_NAMES)
    if name == "s1":
        run = solve(300, 1.0, F2G_PRIORS, "g1", True, F2E_FIT_DAYS, s1, F2E_PREDICT_DAYS, workers, ne)
    else:
        one = solve(300, 1.0, F2G_PRIORS, "g2a", True, (F.DAY_DIR,), s2, (), workers, ne)
        run = solve(300, 1.0, F2G_PRIORS, "g2b", True, F2E_FIT_DAYS, np.array(one["x0"]), (), workers, ne)
        one.pop("est_positions")
        run["continuation_stage1"] = one
    _f2g_part_path(name).write_text(json.dumps(run), encoding="utf-8")


def main_f2g_merge():
    runs = {"S1": json.loads(_f2g_part_path("s1").read_text(encoding="utf-8")),
            "S2": json.loads(_f2g_part_path("s2").read_text(encoding="utf-8"))}
    a, b = runs["S1"], runs["S2"]
    dpos = float(np.max(np.linalg.norm(np.array(a["est_positions"]) - np.array(b["est_positions"]), axis=1)))
    gates = dict(G1=all(r["status"] in ("Converged", "STATIONARY") for r in runs.values()), G2=dpos <= 1.0,
                 G3=all(r["postfit_rms_m"] <= 2.0 and r["doppler_postfit_rms_mmps"] <= 5.0 for r in runs.values()))
    checks = json.loads((HERE.parent / "results" / "phase_f2g_checks.json").read_text(encoding="utf-8"))
    gates.update(G0=checks["G0"]["PASS"], G4e=checks["G4e"]["PASS"])
    for r in runs.values():
        r.pop("est_positions")
    out = dict(predeclaration="studies/phase17/configs/phase_f2g_lro_empirical_predeclaration.json",
               gates=gates, G2_start_difference_m=dpos, priors={str(k): v for k, v in F2G_PRIORS.items()},
               fit_days=F2E_FIT_DAYS, predict_days=F2E_PREDICT_DAYS, nmax=300, runs=runs)
    OUT_F2G.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(gates=gates, G2=dpos), indent=1))


if __name__ == "__main__":
    if "--f2g-part" in sys.argv:
        main_f2g_part(sys.argv[sys.argv.index("--f2g-part") + 1])
    elif "--f2g-merge" in sys.argv:
        main_f2g_merge()
    elif "--f2f-part" in sys.argv:
        main_f2f_part(sys.argv[sys.argv.index("--f2f-part") + 1])
    elif "--f2f-merge" in sys.argv:
        main_f2f()
    elif "--f2e" in sys.argv:
        main_f2e()
    elif "--f2b" in sys.argv:
        main_f2b()
    elif "--f2c" in sys.argv:
        main_f2c()
    elif "--f2d" in sys.argv:
        main_f2d()
    else:
        main()
