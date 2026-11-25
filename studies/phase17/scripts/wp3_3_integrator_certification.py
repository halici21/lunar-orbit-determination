"""F3 WP3.3: long-arc integrator certification of every library PropagationProfile (REQ-DYN-006, analysis only).

Predeclaration: studies/phase17/configs/wp3_3_integrator_certification_predeclaration.json (63a82eb, committed
before any number). Production source is not edited: every propagation is a public propagate_state call.

Gates: K_G1 two-body 10-revolution closure (also tests/test_wp3_3_integrator_certification.py), R_G1 reference
self-convergence, R_G2 reference contacts, C_G1 validity range per profile and tolerance, C_G2 diagnostics.

    python wp3_3_integrator_certification.py run [--workers 12]       # full campaign -> results JSON
    python wp3_3_integrator_certification.py smoke [--workers 12]     # 2 h arcs, software check only (no numbers kept)
    python wp3_3_integrator_certification.py kg1                       # K_G1 table only, printed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lunar_od.dynamics import PropagationProfile  # noqa: E402

PRE = ROOT / "studies" / "phase17" / "configs" / "wp3_3_integrator_certification_predeclaration.json"
OUT = ROOT / "studies" / "phase17" / "results" / "wp3_3_integrator_certification.json"

DAY = 86400.0
PROFILES: dict[str, PropagationProfile] = {
    "P_ADAMS": PropagationProfile(method="ADAMS", shadow_events=False),
    "P_DOP853": PropagationProfile(method="DOP853", shadow_events=False),
    "P_D2": PropagationProfile(method="DOP853", shadow_events=True),
    "P_RK45": PropagationProfile(method="RK45", shadow_events=False),
    "P_RK23": PropagationProfile(method="RK23", shadow_events=False),
    "P_RADAU": PropagationProfile(method="Radau", shadow_events=False),
    "P_BDF": PropagationProfile(method="BDF", shadow_events=False),
    "P_LSODA": PropagationProfile(method="LSODA", shadow_events=False),
}
TOLERANCES = {"T_LIB": (1e-11, 1e-12), "T_THESIS": (1e-10, 1e-11)}
ARCS_DAYS = (1, 3, 7, 28)
TIERS = {"MM": 1e-3, "CM": 1e-2, "DM": 1e-1, "M": 1.0}
CONCLUSIVE_FRACTION = 0.1
BUDGET_S = 1800.0
STEP_S = 60.0
DENSE_S = 10.0
REF_TOL = (1e-13, 1e-14)
REF_SELF_TOL = (1e-12, 1e-13)
PRE_TOL = (1e-12, 1e-13)
BISECT_S = 1e-6
R_G1_MAX_M = 1e-4
R_G2_PROBE_S = 1e-3
ORBITS = ("O1_SHADOWED_100KM", "O2_SUNLIT_100KM", "O3_ELLIPTIC_100x3000KM")
K_SRP = 0.010

# K_G1
KEPLER_CASES = ("K_CIRC", "K_ECC")
K_G1_REVOLUTIONS = 10
K_G1_CLOSURE_M = 1e-3
K_G1_ENERGY_REL = 1e-11
K_G1_TOL = TOLERANCES["T_LIB"]
_FAR_EARTH = np.array([3.844e8, 0.0, 0.0])
_FAR_SUN = np.array([0.0, 1.496e11, 0.0])


class BudgetExceeded(RuntimeError):
    pass


class _Deadline:
    """Wraps the Sun-position callable (called at every RHS evaluation) with a wall-clock deadline."""

    def __init__(self, fn, seconds: float | None):
        self.fn = fn
        self.end = None if seconds is None else time.perf_counter() + float(seconds)

    def __call__(self, t):
        if self.end is not None and time.perf_counter() > self.end:
            raise BudgetExceeded("wall-clock budget exceeded")
        return self.fn(t)


# ======================================================================
# K_G1: two-body closure
# ======================================================================
def _mu_moon() -> float:
    from lunar_od.ephemeris_generation import gravitational_parameters
    return float(gravitational_parameters("DE440")[0])


def kepler_initial_state(case: str, mu: float) -> tuple[np.ndarray, float]:
    """(x0, period). Periapsis on +x, orbit plane tilted 60 deg about x."""
    from lunar_od.constants import R_MOON_M
    rp = R_MOON_M + 100e3
    ra = rp if case == "K_CIRC" else R_MOON_M + 3000e3
    if case not in KEPLER_CASES:
        raise ValueError(case)
    a = 0.5 * (rp + ra)
    vp = math.sqrt(mu * (2.0 / rp - 1.0 / a))
    inc = math.radians(60.0)
    x0 = np.array([rp, 0.0, 0.0, 0.0, vp * math.cos(inc), vp * math.sin(inc)])
    return x0, 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def two_body_closure(profile: str, case: str, budget_s: float | None = None) -> dict:
    """K_G1: 10 Keplerian periods with propagate_state under ``profile`` at T_LIB."""
    from lunar_od.dynamics import propagate_state
    mu = _mu_moon()
    x0, period = kepler_initial_state(case, mu)
    t_end = K_G1_REVOLUTIONS * period
    t = np.append(np.arange(0.0, t_end, STEP_S), t_end)
    rtol, atol = K_G1_TOL
    t0 = time.perf_counter()
    x = np.asarray(propagate_state(t, x0, mu, 0.0, 0.0, lambda _t: _FAR_EARTH,
                                   _Deadline(lambda _t: _FAR_SUN, budget_s), rtol=rtol, atol=atol,
                                   **PROFILES[profile].propagator_kwargs()), dtype=float)
    runtime = time.perf_counter() - t0
    energy = 0.5 * np.sum(x[:, 3:6] ** 2, axis=1) - mu / np.linalg.norm(x[:, :3], axis=1)
    e0 = 0.5 * float(np.dot(x0[3:], x0[3:])) - mu / float(np.linalg.norm(x0[:3]))
    return dict(profile=profile, case=case, revolutions=K_G1_REVOLUTIONS, period_s=period,
                closure_m=float(np.linalg.norm(x[-1, :3] - x0[:3])),
                closure_mps=float(np.linalg.norm(x[-1, 3:] - x0[3:])),
                energy_rel_max=float(np.max(np.abs(energy - e0)) / abs(e0)), runtime_s=runtime)


def _k_g1_task(profile: str, case: str) -> dict:
    warnings.simplefilter("ignore")
    try:
        out = two_body_closure(profile, case, BUDGET_S)
        out["status"] = "PASS" if (out["closure_m"] <= K_G1_CLOSURE_M
                                   and out["energy_rel_max"] <= K_G1_ENERGY_REL) else "FAIL"
    except BudgetExceeded:
        out = dict(profile=profile, case=case, status="BUDGET_EXCEEDED")
    except Exception as exc:  # reported as a finding, not hidden
        out = dict(profile=profile, case=case, status="ERROR", error=f"{type(exc).__name__}: {exc}")
    return out


# ======================================================================
# orbits
# ======================================================================
class OrbitContext:
    def __init__(self, orbit: str, horizon_days: float):
        import wp3_1_materiality_survey as S
        from lunar_od.constants import J2_MOON_GRAIL_UNNORMALIZED, R_MOON_M
        from lunar_od.ephemeris import sample_moon_centered_ephemeris
        from lunar_od.ephemeris_generation import gravitational_parameters
        from lunar_od.spice_loader import load_spice_kernels
        from lunar_od.srp import SRPOptions
        load_spice_kernels()
        self.orbit = orbit
        self.et0, self.epoch_info = S.epoch_et0("E1")
        self.mus = tuple(float(m) for m in gravitational_parameters("DE440"))
        t_eph = np.arange(-600.0, horizon_days * DAY + 600.0 + 0.5, 60.0)
        self.eph = sample_moon_centered_ephemeris(self.et0, t_eph)
        sun0 = self.eph.sun_position(0.0)
        if orbit == "O1_SHADOWED_100KM":
            self.x0, info = S.initial_state("O1", self.et0, sun0, self.mus[0])
        elif orbit == "O2_SUNLIT_100KM":
            self.x0, info = S.initial_state("O2", self.et0, sun0, self.mus[0])
        elif orbit == "O3_ELLIPTIC_100x3000KM":
            xc, info = S.initial_state("O1", self.et0, sun0, self.mus[0])
            rp, ra = R_MOON_M + 100e3, R_MOON_M + 3000e3
            a = 0.5 * (rp + ra)
            vp = math.sqrt(self.mus[0] * (2.0 / rp - 1.0 / a))
            self.x0 = np.concatenate([xc[:3], vp * xc[3:] / np.linalg.norm(xc[3:])])
            info = dict(info, periapsis_alt_km=100.0, apoapsis_alt_km=3000.0)
        else:
            raise ValueError(orbit)
        self.info = dict(self.epoch_info, **{k: v for k, v in info.items() if k != "pole"})
        self.srp = SRPOptions(k_srp_m2_per_kg=K_SRP, shadow_model="CONICAL_PENUMBRA")
        self.j2 = J2_MOON_GRAIL_UNNORMALIZED      # amendment A1: production default grail_pa_de440 model

    def propagate(self, t, x0, rtol, atol, *, method="DOP853", shadow_events=False, budget_s=None):
        from lunar_od.dynamics import propagate_state
        return np.asarray(propagate_state(t, x0, *self.mus, self.eph.earth_position,
                                          _Deadline(self.eph.sun_position, budget_s), rtol=rtol, atol=atol,
                                          method=method, shadow_events=shadow_events, srp=self.srp,
                                          j2_moon=self.j2, lunar_j2_epoch_et0=self.et0), dtype=float)

    def illumination_class(self, t: float, r) -> int:
        from lunar_od.srp import _illumination_for
        nu = float(_illumination_for(np.asarray(r, float), np.asarray(self.eph.sun_position(t), float), self.srp))
        return 2 if nu >= 1.0 else (0 if nu <= 0.0 else 1)


@lru_cache(maxsize=4)
def orbit_context(orbit: str, horizon_days: float = float(ARCS_DAYS[-1])) -> OrbitContext:
    return OrbitContext(orbit, horizon_days)


def grid(days: float, step: float = STEP_S) -> np.ndarray:
    return np.arange(0.0, days * DAY + 0.5 * step, step)


# ======================================================================
# reference
# ======================================================================
def segmented(ctx: OrbitContext, t_out: np.ndarray, cuts, rtol: float, atol: float) -> np.ndarray:
    """DOP853 restarted at every cut; each segment one propagate_state call (shadow_events False)."""
    t_out = np.asarray(t_out, float)
    inner_cuts = [float(c) for c in cuts if t_out[0] < c < t_out[-1]]
    bounds = [float(t_out[0]), *inner_cuts, float(t_out[-1])]
    out = np.empty((t_out.size, 6))
    out[0] = ctx.x0
    x = np.asarray(ctx.x0, float)
    i = 1
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = int(np.searchsorted(t_out, b, side="right"))
        inner = t_out[i:j]
        tail = [] if (inner.size and inner[-1] == b) else [b]
        te = np.concatenate([[a], inner, tail])
        seg = ctx.propagate(te, x, rtol, atol)
        out[i:j] = seg[1:1 + (j - i)]
        x = seg[-1]
        i = j
    return out


def _classes(ctx: OrbitContext, t, x) -> np.ndarray:
    return np.array([ctx.illumination_class(float(tk), xk[:3]) for tk, xk in zip(t, x)], dtype=int)


def contacts(ctx: OrbitContext, dense_t, dense_x) -> tuple[list[float], dict]:
    """Class changes on the dense grid, each refined on 101 sub-samples, then bisected to BISECT_S."""
    from lunar_od.radiometrics import _interp_state
    cls = _classes(ctx, dense_t, dense_x)

    def cls_at(m):
        return ctx.illumination_class(m, _interp_state(dense_t, dense_x, m)[:3])

    cuts, double_jumps, brackets = [], 0, []
    for i in np.flatnonzero(cls[1:] != cls[:-1]):
        sub = np.linspace(float(dense_t[i]), float(dense_t[i + 1]), 101)
        scls = [cls[i], *[cls_at(m) for m in sub[1:-1]], cls[i + 1]]
        for k in range(100):
            if scls[k] == scls[k + 1]:
                continue
            if abs(scls[k] - scls[k + 1]) > 1:
                double_jumps += 1
            a, b, ca = float(sub[k]), float(sub[k + 1]), scls[k]
            while b - a > BISECT_S:
                m = 0.5 * (a + b)
                if cls_at(m) == ca:
                    a = m
                else:
                    b = m
            cuts.append(0.5 * (a + b))
            brackets.append(b - a)
    return cuts, dict(dense_class_changes=int(np.sum(np.abs(np.diff(cls)))), double_jumps_in_subgrid=double_jumps,
                      max_bracket_s=float(max(brackets)) if brackets else 0.0)


def build_reference(orbit: str, days: float = float(ARCS_DAYS[-1])) -> dict:
    """Event-segmented DOP853 1e-13 reference on the 60 s grid, its 1e-12 twin, R_G1 and R_G2."""
    warnings.simplefilter("ignore")
    from lunar_od.radiometrics import _interp_state
    t_start = time.perf_counter()
    ctx = orbit_context(orbit, days)
    dense_t, t60 = grid(days, DENSE_S), grid(days)
    stride = int(round(STEP_S / DENSE_S))
    pre = ctx.propagate(dense_t, ctx.x0, *PRE_TOL)
    cuts0, _ = contacts(ctx, dense_t, pre)
    dense1 = segmented(ctx, dense_t, cuts0, *REF_TOL)
    cuts1, _ = contacts(ctx, dense_t, dense1)
    final = segmented(ctx, dense_t, cuts1, *REF_TOL)
    cuts_final_check, info = contacts(ctx, dense_t, final)
    ref13 = final[::stride]
    assert np.array_equal(dense_t[::stride], t60)
    ref12 = segmented(ctx, t60, cuts1, *REF_SELF_TOL)
    d_self = np.linalg.norm(ref13[:, :3] - ref12[:, :3], axis=1)
    d_self_v = np.linalg.norm(ref13[:, 3:] - ref12[:, 3:], axis=1)
    # R_G2 (a): the class changes across every cut on the final reference
    probe_fail = 0
    for c in cuts1:
        lo = ctx.illumination_class(c - R_G2_PROBE_S, _interp_state(dense_t, final, c - R_G2_PROBE_S)[:3])
        hi = ctx.illumination_class(c + R_G2_PROBE_S, _interp_state(dense_t, final, c + R_G2_PROBE_S)[:3])
        probe_fail += int(lo == hi)
    arcs = {}
    for d in ARCS_DAYS:
        if d > days:
            continue
        n = int(np.searchsorted(t60, d * DAY, side="right"))
        arcs[str(d)] = dict(dR_m=float(d_self[:n].max()), dR_v_mps=float(d_self_v[:n].max()),
                            r_g1=bool(d_self[:n].max() <= R_G1_MAX_M),
                            contacts=int(sum(1 for c in cuts1 if c <= d * DAY)))
    shift = (float(np.max(np.abs(np.array(cuts1) - np.array(cuts0))))
             if len(cuts0) == len(cuts1) and cuts1 else None)
    r_g2 = dict(probe_failures=probe_fail, cuts=len(cuts1), dense_class_changes_final=info["dense_class_changes"],
                recount_on_final=len(cuts_final_check), double_jumps_in_subgrid=info["double_jumps_in_subgrid"],
                max_bracket_s=info["max_bracket_s"], cut_shift_iteration_s=shift, cuts_prepass=len(cuts0))
    r_g2["pass"] = bool(probe_fail == 0 and info["dense_class_changes"] == len(cuts1))
    return dict(orbit=orbit, info=ctx.info, x0=ctx.x0.tolist(), arcs=arcs, r_g2=r_g2, cuts=cuts1,
                runtime_s=time.perf_counter() - t_start, ref13=ref13, days=days)


# ======================================================================
# certification cells
# ======================================================================
def run_cell(orbit: str, profile: str, tol: str, ref_states: np.ndarray, days_list=ARCS_DAYS,
             budget_s: float = BUDGET_S) -> dict:
    """One profile x tolerance x orbit: every arc a separate propagate_state call from t0."""
    warnings.simplefilter("ignore")
    ctx = orbit_context(orbit, float(max(days_list)))
    rtol, atol = TOLERANCES[tol]
    kw = PROFILES[profile].propagator_kwargs()
    out, stop = {}, None
    for d in days_list:
        if stop:
            out[str(d)] = dict(status="NOT_ATTEMPTED", e_r_m=None, e_v_mps=None, reason=stop)
            continue
        t = grid(d)
        t0 = time.perf_counter()
        try:
            x = ctx.propagate(t, ctx.x0, rtol, atol, budget_s=budget_s, **kw)
        except BudgetExceeded:
            out[str(d)] = dict(status="BUDGET_EXCEEDED", e_r_m=None, e_v_mps=None,
                               runtime_s=time.perf_counter() - t0)
            stop = f"budget exceeded at {d} d"
            continue
        except Exception as exc:  # a finding, reported as computed
            out[str(d)] = dict(status="ERROR", e_r_m=None, e_v_mps=None, error=f"{type(exc).__name__}: {exc}",
                               runtime_s=time.perf_counter() - t0)
            stop = f"error at {d} d"
            continue
        ref = ref_states[:t.size]
        e_r = np.linalg.norm(x[:, :3] - ref[:, :3], axis=1)
        e_v = np.linalg.norm(x[:, 3:] - ref[:, 3:], axis=1)
        out[str(d)] = dict(status="OK", e_r_m=float(e_r.max()), e_v_mps=float(e_v.max()),
                           e_r_end_m=float(e_r[-1]), runtime_s=time.perf_counter() - t0)
    return dict(orbit=orbit, profile=profile, tol=tol, arcs=out)


def validity_range(cells_by_orbit: dict, ref_self_by_orbit: dict) -> dict:
    """C_G1: per tier the longest arc for which every orbit is OK, within the tier and conclusive.

    ``cells_by_orbit[orbit][arc] = {status, e_r_m}``; ``ref_self_by_orbit[orbit][arc] = dR`` (m).
    Returns {tier: longest arc in days, 'none' or 'inconclusive'}.
    """
    out = {}
    for tier, thr in TIERS.items():
        best: int | str = "none"
        for d in ARCS_DAYS:
            failed = inconclusive = False
            for orbit, cell in cells_by_orbit.items():
                c = cell[str(d)]
                if c["status"] != "OK" or c["e_r_m"] is None:
                    failed = True
                elif ref_self_by_orbit[orbit][str(d)] > CONCLUSIVE_FRACTION * thr:
                    inconclusive = True
                elif c["e_r_m"] > thr:
                    failed = True
            if failed:
                break
            if inconclusive:
                if best == "none":
                    best = "inconclusive"
                break
            best = d
        out[tier] = best
    return out


def production_contacts_diagnostic(orbit: str, ref_cuts) -> dict:
    """C_G2: production shadow_contact_times over day 1 at T_LIB vs the reference cuts."""
    from lunar_od.dynamics import shadow_contact_times
    ctx = orbit_context(orbit)
    t = grid(1)
    prod = np.asarray(shadow_contact_times(t, ctx.x0, *ctx.mus, ctx.eph.earth_position, ctx.eph.sun_position,
                                           srp=ctx.srp, rtol=TOLERANCES["T_LIB"][0], atol=TOLERANCES["T_LIB"][1],
                                           j2_moon=ctx.j2, lunar_j2_epoch_et0=ctx.et0), dtype=float).reshape(-1)
    ref = np.array([c for c in ref_cuts if 0.0 < c < DAY])
    d = dict(production=int(prod.size), reference=int(ref.size))
    if prod.size == ref.size and prod.size:
        d["max_abs_dt_s"] = float(np.max(np.abs(np.sort(prod) - np.sort(ref))))
    return d


# ======================================================================
# driver
# ======================================================================
def _ref_task(orbit: str, days: float):
    ref = build_reference(orbit, days)
    try:
        ref["c_g2"] = production_contacts_diagnostic(orbit, ref["cuts"]) if days >= 1 else None
    except Exception as exc:
        ref["c_g2"] = dict(error=f"{type(exc).__name__}: {exc}")
    return ref


def _meta() -> dict:
    import numpy
    import scipy

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    try:
        import numba
        nb = numba.__version__
    except ImportError:
        nb = None
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", "src")),
                python=platform.python_version(), platform=platform.platform(), numpy=numpy.__version__,
                scipy=scipy.__version__, numba=nb,
                predeclaration_sha256=hashlib.sha256(PRE.read_bytes()).hexdigest())


def run(workers: int, smoke: bool = False) -> dict:
    days_list = (1 / 12,) if smoke else ARCS_DAYS
    horizon = float(max(days_list))
    t_start = time.perf_counter()
    result = dict(study="F3 WP3.3 long-arc integrator certification", meta=_meta(), smoke=smoke,
                  settings=dict(profiles={k: v.propagator_kwargs() for k, v in PROFILES.items()},
                                tolerances=TOLERANCES, arcs_days=list(days_list), tiers=TIERS,
                                budget_s=BUDGET_S, reference_tol=REF_TOL, reference_self_tol=REF_SELF_TOL,
                                k_srp=K_SRP, orekit="out of scope: not installed locally; deferred to WP5.1"))
    refs, k_g1, cells = {}, [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_ref_task, o, horizon): ("ref", o) for o in ORBITS}
        if not smoke:
            futs.update({pool.submit(_k_g1_task, p, c): ("kg1", p) for p in PROFILES for c in KEPLER_CASES})
        cell_futs = {}
        for fut in as_completed(list(futs)):
            kind, key = futs[fut]
            res = fut.result()
            if kind == "kg1":
                k_g1.append(res)
                print(f"K_G1 {res['profile']} {res['case']}: {res['status']}", flush=True)
                continue
            refs[key] = res
            print(f"reference {key}: {res['runtime_s']:.0f} s, {len(res['cuts'])} cuts, R_G2 {res['r_g2']['pass']}",
                  flush=True)
            for p in PROFILES:
                for tol in TOLERANCES:
                    cell_futs[pool.submit(run_cell, key, p, tol, res["ref13"], days_list)] = (key, p, tol)
        for fut in as_completed(list(cell_futs)):
            res = fut.result()
            cells.append(res)
            print(f"cell {res['orbit']} {res['profile']} {res['tol']}: "
                  + ", ".join(f"{d}:{a['status']}" for d, a in res["arcs"].items()), flush=True)
    arc_keys = [str(d) for d in days_list]
    ref_self = {o: {d: refs[o]["arcs"].get(d, {}).get("dR_m", math.inf) for d in arc_keys} for o in ORBITS}
    validity = {}
    if not smoke:
        for p in PROFILES:
            for tol in TOLERANCES:
                by_orbit = {c["orbit"]: c["arcs"] for c in cells if c["profile"] == p and c["tol"] == tol}
                validity[f"{p}/{tol}"] = validity_range(by_orbit, ref_self)
    for r in refs.values():
        r.pop("ref13", None)
    result.update(reference=refs, k_g1=sorted(k_g1, key=lambda r: (r["profile"], r["case"])),
                  cells=sorted(cells, key=lambda c: (c["profile"], c["tol"], c["orbit"])), validity=validity,
                  runtime_s=time.perf_counter() - t_start)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "smoke", "kg1"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.mode == "kg1":
        for p in PROFILES:
            for c in KEPLER_CASES:
                print(json.dumps(_k_g1_task(p, c)), flush=True)
        return 0
    res = run(a.workers, smoke=a.mode == "smoke")
    if a.mode == "smoke":
        print(json.dumps({k: res[k] for k in ("runtime_s",)}, indent=1))
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
