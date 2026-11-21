"""F3 WP3.2 S1, gate T_G1(b): the default paths are byte-identical before and after the tide change.

Predeclaration: studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53).
``run`` executes a fixed set of production propagations and estimator solves with ``lunar_tide``
omitted (mode ``omit``) or passed as ``None`` (mode ``none``) and writes the SHA-256 of every output
array's bytes. ``compare`` checks that two or more such files agree configuration by configuration.

Synthetic geometry, no SPICE: Earth and Sun on circles (sidereal month and year, 23.44 deg tilt),
a 100 km circular polar orbit with the Sun in its plane (eclipses every orbit).

usage: python wp3_2_s1_default_identity.py run {omit|none} OUT.json
       python wp3_2_s1_default_identity.py compare A.json B.json [...] [--out RESULT.json]
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lunar_od.dynamics as dyn  # noqa: E402
from lunar_od import estimators as E  # noqa: E402
from lunar_od.constants import (  # noqa: E402
    AU_M, J2_EARTH_UNNORMALIZED, J2_MOON_UNNORMALIZED, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M,
)
from lunar_od.gravity_harmonics import SphericalHarmonicGravityModel, cbar_n0_from_unnormalized_jn  # noqa: E402
from lunar_od.measurements import PassGeometry  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402

DAY_S = 86400.0
TILT = math.radians(23.44)
W_EARTH = 2.0 * math.pi / (27.321661 * DAY_S)
W_SUN = 2.0 * math.pi / (365.25636 * DAY_S)
D_EARTH_M = 384_400e3
SUN_PHASE = 1.0


def _circle(radius, omega, phase, t):
    th = omega * np.asarray(t, dtype=float) + phase
    return radius * np.stack([np.cos(th), np.sin(th) * math.cos(TILT), np.sin(th) * math.sin(TILT)], axis=-1)


def earth(t):
    return _circle(D_EARTH_M, W_EARTH, 0.0, t)


def sun(t):
    return _circle(AU_M, W_SUN, SUN_PHASE, t) + earth(t)


def polar_orbit_with_sun_in_plane(altitude_m=100e3):
    a = R_MOON_M + altitude_m
    s = sun(0.0) / np.linalg.norm(sun(0.0))
    z = np.array([0.0, 0.0, 1.0])
    h = np.cross(s, z)
    h /= np.linalg.norm(h)
    v = math.sqrt(MU_MOON_M3S2 / a) * np.cross(h, z)
    return np.concatenate([a * z, v])


def zonal_model():
    cbar = np.zeros((4, 4))
    sbar = np.zeros((4, 4))
    cbar[2, 0] = cbar_n0_from_unnormalized_jn(J2_MOON_UNNORMALIZED, 2)
    cbar[3, 0] = -3.2e-6
    return SphericalHarmonicGravityModel(mu_m3_s2=MU_MOON_M3S2, r_ref_m=1738.0e3, cbar=cbar, sbar=sbar,
                                         nmax=3, mmax=0, frame="MOON_PA_DE440")


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


class _Ephemeris:
    earth_position = staticmethod(earth)
    sun_position = staticmethod(sun)


# ----------------------------------------------------------------------
# hashing
# ----------------------------------------------------------------------
def _leaves(obj, prefix, out):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        out[prefix] = f"{arr.dtype.str}{list(arr.shape)}:{h}"
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            _leaves(getattr(obj, f.name), f"{prefix}.{f.name}", out)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _leaves(v, f"{prefix}[{i}]", out)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _leaves(obj[k], f"{prefix}[{k!r}]", out)
    elif isinstance(obj, (bool, np.bool_)):
        out[prefix] = repr(bool(obj))
    elif isinstance(obj, (int, np.integer)):
        out[prefix] = repr(int(obj))
    elif isinstance(obj, (float, np.floating)):
        out[prefix] = float(obj).hex()
    elif obj is None or isinstance(obj, str):
        out[prefix] = repr(obj)
    else:
        out[prefix] = f"<{type(obj).__name__}>"


def _record(value):
    leaves = {}
    _leaves(value, "out", leaves)
    digest = hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest()
    return {"digest": digest, "leaves": leaves}


# ----------------------------------------------------------------------
# configurations
# ----------------------------------------------------------------------
def configurations(tide_kw):
    x0 = polar_orbit_with_sun_in_plane()
    t6h = np.arange(0.0, 6 * 3600.0 + 1.0, 300.0)
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    pen = SRPOptions(k_srp_m2_per_kg=0.01)
    nos = SRPOptions(k_srp_m2_per_kg=0.01, shadow_model="NO_SHADOW")
    model = zonal_model()
    rot = _rotation_x(0.03)
    base = (MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, earth, sun)
    j2 = dict(j2_moon=J2_MOON_UNNORMALIZED)
    harm = dict(harmonic_model=model, harmonic_rotation=rot)
    cfg = {}

    def add(name, fn):
        cfg[name] = fn

    add("ps_adams_j2", lambda: dyn.propagate_state(t6h, x0, *base, **j2, **tide_kw))
    add("ps_adams_srp_earthj2", lambda: dyn.propagate_state(
        t6h, x0, *base, **j2, srp=pen, j2_earth=J2_EARTH_UNNORMALIZED, **tide_kw))
    add("ps_dop853_srp_events", lambda: dyn.propagate_state(
        t6h, x0, *base, **j2, srp=pen, method="DOP853", shadow_events=True, **tide_kw))
    add("ps_dop853_noshadow", lambda: dyn.propagate_state(
        t6h, x0, *base, **j2, srp=nos, method="DOP853", rtol=1e-13, atol=1e-14, **tide_kw))
    add("ps_reference_srp", lambda: _reference(lambda: dyn.propagate_state(
        t6h, x0, *base, **j2, srp=pen, **tide_kw)))
    add("ps_harmonic", lambda: dyn.propagate_state(t6h, x0, *base, **harm, srp=pen, **tide_kw))
    add("pa_adams_j2", lambda: dyn.propagate_augmented_state(t6h, aug0, *base, **j2, **tide_kw))
    add("pa_adams_srp", lambda: dyn.propagate_augmented_state(t6h, aug0, *base, **j2, srp=pen, **tide_kw))
    add("pa_reference_srp", lambda: _reference(lambda: dyn.propagate_augmented_state(
        t6h, aug0, *base, **j2, srp=pen, **tide_kw)))
    add("pa_dop853_events", lambda: dyn.propagate_augmented_state(
        t6h, aug0, *base, **j2, srp=pen, method="DOP853", shadow_events=True, **tide_kw))
    add("pa_harmonic_events", lambda: dyn.propagate_augmented_state(
        t6h, aug0, *base, **harm, harmonic_stm_opt_in=True, srp=pen, method="DOP853",
        shadow_events=True, **tide_kw))
    add("k48_adams", lambda: dyn.propagate_state_with_k_sensitivity(t6h, x0, *base, srp=pen, **j2, **tide_kw))
    add("k48_dop853_events", lambda: dyn.propagate_state_with_k_sensitivity(
        t6h, x0, *base, srp=pen, **j2, method="DOP853", shadow_events=True, **tide_kw))
    add("k48_harmonic_events", lambda: dyn.propagate_state_with_k_sensitivity(
        t6h, x0, *base, srp=pen, **harm, harmonic_stm_opt_in=True, method="DOP853",
        shadow_events=True, **tide_kw))
    add("contacts", lambda: np.array(dyn.shadow_contact_times(t6h, x0, *base, srp=pen, **j2, **tide_kw)))
    add("truth_profile", lambda: dyn.propagate_truth_with_ephemeris(
        t6h, x0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, _Ephemeris(), **j2, srp=pen,
        propagation_profile=dyn.PropagationProfile(), **tide_kw))
    add("truth_legacy", lambda: dyn.propagate_truth_with_ephemeris(
        t6h, x0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, _Ephemeris(), **j2, **tide_kw))
    for solver in ("bls_lm", "srif"):
        for variant in ("legacy", "profile"):
            add(f"est_position_{solver}_{variant}",
                lambda s=solver, v=variant: _position_solve(s, v, tide_kw))
            add(f"est_range_rate_{solver}_{variant}",
                lambda s=solver, v=variant: _range_rate_solve(s, v, tide_kw))
    return cfg


def _reference(call):
    saved = dyn._FAST_DYNAMICS
    dyn._FAST_DYNAMICS = False
    try:
        return call()
    finally:
        dyn._FAST_DYNAMICS = saved


def _position_solve(solver, variant, tide_kw):
    from tests import test_filters as T
    x0 = polar_orbit_with_sun_in_plane()
    t = np.arange(0.0, 3600.0 + 1.0, 120.0)
    srp = SRPOptions(k_srp_m2_per_kg=0.01)
    truth = dyn.propagate_state(t, x0, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, earth, sun,
                                srp=srp, j2_moon=J2_MOON_UNNORMALIZED, rtol=1e-12, atol=1e-13)
    stations = tuple(T._synthetic_position_station(*q) for q in [(0., 0., 0.), (0., 90., 0.), (45., -30., 500.)])
    pg = PassGeometry(t_s=t, earth_pos_mci_m=np.zeros((t.size, 3)), earth_vel_mci_mps=np.zeros((t.size, 3)),
                      x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], t.size, axis=0), stations=stations,
                      measurement_type="position")
    obs = T._build_clean_position_observations(t, truth, pg)
    start = x0 + np.array([50.0, -40.0, 30.0, 0.05, -0.03, 0.02])
    kw = dict(max_iter=3, return_posterior=True, rtol=1e-12, atol=1e-13, j2_moon=J2_MOON_UNNORMALIZED)
    if variant == "profile":
        kw.update(srp=srp, propagation_profile=dyn.PropagationProfile())
    fn = getattr(E, f"estimate_position_{solver}")
    return fn(t, obs, start, pg, MU_MOON_M3S2, MU_EARTH_M3S2, MU_SUN_M3S2, earth, sun, **kw, **tide_kw)


def _range_rate_solve(solver, variant, tide_kw):
    from tests import test_filters as T
    mu, x0, t, _truth, pg, obs, e, s = T._synthetic_range_rate_case(600.0)
    start = x0 + np.array([20.0, -15.0, 10.0, 0.02, -0.01, 0.015])
    kw = dict(max_iter=3, return_posterior=True, rtol=1e-12, atol=1e-13)
    if variant == "profile":
        kw.update(srp=SRPOptions(k_srp_m2_per_kg=0.01), propagation_profile=dyn.PropagationProfile())
    fn = getattr(E, f"estimate_range_rate_{solver}")
    return fn(t, obs, start, pg, mu, 0.0, 0.0, e, s, **kw, **tide_kw)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def run(mode, out_path):
    if mode not in ("omit", "none"):
        raise SystemExit("mode must be omit or none")
    tide_kw = {} if mode == "omit" else {"lunar_tide": None}
    results = {}
    for name, call in configurations(tide_kw).items():
        try:
            results[name] = _record(call())
        except Exception as exc:  # noqa: BLE001 - recorded, the comparison fails on it
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{name:28s} {results[name].get('digest', results[name].get('error'))[:72]}", flush=True)
    import numba
    import scipy
    payload = {
        "gate": "T_G1(b)", "mode": mode, "commit": _git("rev-parse", "HEAD"),
        "src_status": _git("status", "--porcelain", "--", "src"),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "numba": numba.__version__, "fast_dynamics": bool(dyn._FAST_DYNAMICS), "configs": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def compare(paths, out_path=None):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = sorted(set().union(*(r["configs"] for r in runs)))
    table, failures = {}, []
    for name in names:
        entries = [r["configs"].get(name, {"error": "missing"}) for r in runs]
        digests = [e.get("digest") for e in entries]
        same = all(d is not None and d == digests[0] for d in digests)
        table[name] = {"identical": same, "digests": digests,
                       "errors": [e.get("error") for e in entries if "error" in e]}
        if not same:
            failures.append(name)
    result = {"gate": "T_G1(b)", "inputs": [{"path": str(p), "mode": r["mode"], "commit": r["commit"],
                                             "src_status": r["src_status"]} for p, r in zip(paths, runs)],
              "n_configs": len(names), "n_identical": len(names) - len(failures), "failures": failures,
              "verdict": "PASS" if not failures else "FAIL", "configs": table}
    print(json.dumps({k: result[k] for k in ("n_configs", "n_identical", "failures", "verdict")}, indent=1))
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        args = sys.argv[2:]
        out = None
        if "--out" in args:
            i = args.index("--out")
            out = args[i + 1]
            args = args[:i] + args[i + 2:]
        compare(args, out)
    else:
        raise SystemExit(__doc__)
