"""F3 WP3.2 S2, exploratory (amendment log): why the L_G6 path differences exceed 1e-5 m with LRP on.

Not a predeclared gate. L_G6 failed as predeclared (results/wp3_2_s2_gate_values.json). Hypothesis, stated
before this run: the C6 discrete rule is not smooth in the spacecraft position (a quadrature node crossing the
terminator or the thermal-floor line puts a kink in the RHS, and switching the seed axis at |z_x| = 0.9
rotates the node grid), so the adaptive DOP853 steps of the 6-, 42- and 48-state propagations accumulate
different errors; the wiring itself is right. Variants, each run through the SAME production wrapper and
propagators (path_differences of the L_G6 test):

- ``c6``: production (repeats the gate numbers);
- ``smooth``: the module functions the wrapper imports are replaced by a smooth surrogate,
  g = C0 (R/|r|)^2 r_hat (1 + 0.5 s_hat . r_hat), with its exact gradient (complex step), C0 = 1e-6 N m^-2;
- ``fine``: C6 with 120 x 144 nodes (4x the nodes, each kink 4x smaller).

For each variant and case the 6-state self floor F = max|p6(rtol 1e-13) - p6(rtol 3e-14)| is also recorded.

Amended gate set L_G6-A, written in docs/assurance/amendment-log.md before this run:
(A1) smooth: in all four cases the 42/48- vs 6-state positions agree to 1e-5 m, Phi(42) vs Phi(48) to 1e-8,
     and the effect exceeds 0.1 m (the predeclared L_G6 thresholds, through the production wiring);
(A2) in every case F(c6) >= 3 F(smooth);
(A3) c6: in every case the 42/48- vs 6-state positions agree to 4 F(c6) and Phi to 1e-8.
``fine`` is information only.

usage: python wp3_2_s2_lg6_floor_diagnostic.py OUT.json
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT, ROOT / "studies" / "phase17" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import lunar_od.dynamics as dyn  # noqa: E402
import lunar_od.lunar_radiation as lr  # noqa: E402
from lunar_od.constants import J2_MOON_UNNORMALIZED, R_MOON_M  # noqa: E402
from lunar_od.lunar_radiation import LunarRadiationOptions  # noqa: E402
from tests import test_wp3_2_lunar_radiation as M  # noqa: E402

G = M.G
C0 = 1e-6
FINE = LunarRadiationOptions(n_theta=120, n_phi=144)


def _smooth(r_sc, r_sun):
    r = np.asarray(r_sc)
    rho = np.sqrt(np.sum(r * r))
    q = r / rho
    s_hat = np.asarray(r_sun, float) / np.linalg.norm(r_sun)
    return C0 * (R_MOON_M / rho) ** 2 * q * (1.0 + 0.5 * np.sum(s_hat * q))


def _smooth_value(r_sc, r_sun, _options=None):
    return np.asarray(_smooth(np.asarray(r_sc, float), r_sun), float)


def _smooth_value_and_gradient(r_sc, r_sun, _options=None):
    r = np.asarray(r_sc, float)
    cols = []
    for k in range(3):
        rc = r.astype(complex)
        rc[k] += 1e-30j
        cols.append(_smooth(rc, r_sun).imag / 1e-30)
    return _smooth_value(r, r_sun), np.stack(cols, axis=1)


@contextmanager
def _patched(variant):
    saved = (lr.lunar_radiation_acceleration_per_k, lr.lunar_radiation_acceleration_and_gradient_per_k)
    if variant == "smooth":
        lr.lunar_radiation_acceleration_per_k = _smooth_value
        lr.lunar_radiation_acceleration_and_gradient_per_k = _smooth_value_and_gradient
    try:
        yield
    finally:
        lr.lunar_radiation_acceleration_per_k, lr.lunar_radiation_acceleration_and_gradient_per_k = saved


def _kw(harmonics, tide, lrp):
    kw = dict(srp=M.NOS, lunar_radiation=lrp, method="DOP853")
    if tide:
        kw["lunar_tide"] = M.TIDE
    if harmonics:
        kw.update(harmonic_model=G._zonal_model(), harmonic_rotation=G._rot_x(0.03))
        return kw, dict(kw, harmonic_stm_opt_in=True)
    kw.update(j2_moon=J2_MOON_UNNORMALIZED)
    return kw, kw


def case(variant, harmonics, tide):
    lrp = FINE if variant == "fine" else M.LRP
    kw, kw_var = _kw(harmonics, tide, lrp)
    tight = dict(rtol=1e-13, atol=1e-14)
    with _patched(variant):
        p6 = dyn.propagate_state(M.T6H, M.X0, *M.BASE, **kw, **tight)
        p6_tighter = dyn.propagate_state(M.T6H, M.X0, *M.BASE, **kw, rtol=3e-14, atol=3e-15)
        p42 = dyn.propagate_augmented_state(M.T6H, G._aug0(M.X0), *M.BASE, **kw_var, **tight)
        p48 = dyn.propagate_state_with_k_sensitivity(M.T6H, M.X0, *M.BASE, **kw_var, **tight)
    p6_free = dyn.propagate_state(M.T6H, M.X0, *M.BASE, **dict(kw, lunar_radiation=None), **tight)
    scale = np.max(np.abs(p48[:, 6:42]))
    return dict(position_vs_6_state_m={"42": float(np.max(np.abs(p42[:, :3] - p6[:, :3]))),
                                       "48": float(np.max(np.abs(p48[:, :3] - p6[:, :3])))},
                phi_42_vs_48_relative=float(np.max(np.abs(p42[:, 6:42] - p48[:, 6:42])) / scale),
                self_floor_6_state_m=float(np.max(np.abs(p6_tighter[:, :3] - p6[:, :3]))),
                lrp_effect_6h_m=float(np.max(np.abs(p6_free[:, :3] - p6[:, :3]))))


def verdicts(runs):
    cases = [f"harmonics_{h}_tide_{t}" for h in (False, True) for t in (False, True)]
    a1 = {c: max(runs[f"smooth_{c}"]["position_vs_6_state_m"].values()) <= 1e-5
          and runs[f"smooth_{c}"]["phi_42_vs_48_relative"] <= 1e-8 and runs[f"smooth_{c}"]["lrp_effect_6h_m"] > 0.1
          for c in cases}
    a2 = {c: runs[f"c6_{c}"]["self_floor_6_state_m"] >= 3.0 * runs[f"smooth_{c}"]["self_floor_6_state_m"]
          for c in cases}
    a3 = {c: max(runs[f"c6_{c}"]["position_vs_6_state_m"].values()) <= 4.0 * runs[f"c6_{c}"]["self_floor_6_state_m"]
          and runs[f"c6_{c}"]["phi_42_vs_48_relative"] <= 1e-8 for c in cases}
    out = {name: ("PASS" if all(v.values()) else "FAIL") for name, v in (("A1", a1), ("A2", a2), ("A3", a3))}
    out["cases"] = {c: {"A1": a1[c], "A2": a2[c], "A3": a3[c]} for c in cases}
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main(path):
    import numba
    import scipy
    out = dict(status="exploratory, not a predeclared gate (docs/assurance/amendment-log.md)",
               commit=_git("rev-parse", "HEAD"), src_status=_git("status", "--porcelain", "--", "src"),
               python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               numba=numba.__version__, smooth_c0_n_m2=C0, fine_nodes=[FINE.n_theta, FINE.n_phi], runs={})
    plan = [(v, h, t) for v in ("smooth", "c6") for h in (False, True) for t in (False, True)]
    plan += [("fine", False, False), ("fine", False, True)]
    for variant, harmonics, tide in plan:
        t0 = time.perf_counter()
        name = f"{variant}_harmonics_{harmonics}_tide_{tide}"
        out["runs"][name] = case(variant, harmonics, tide)
        print(name, json.dumps(out["runs"][name]), f"({time.perf_counter() - t0:.0f} s)", flush=True)
    out["verdicts"] = verdicts(out["runs"])
    print(json.dumps(out["verdicts"]))
    Path(path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
