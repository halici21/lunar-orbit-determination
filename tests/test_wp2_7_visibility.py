"""F2 WP2.7 (MEAS-14): atan2 elevation and the azimuth wrap, gap bridging on non-uniform grids, light-time
apparent visibility.

Gates G1-G11 in studies/phase17/configs/wp2_7_predeclaration.json. The "before" sha256 values in
tests/fixtures/wp2_7_visibility_before.json were recorded from the functions before the change by
studies/phase17/scripts/wp2_7_record_before.py, which builds its inputs with the helpers below.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest

C = 299792458.0
R_MOON = 1737.4e3
MIN_EL = float(np.deg2rad(10.0))
TWO_PI = 2.0 * np.pi
MAX_GAPS = (0.0, 150.0, 1200.0, 1800.0)
BEFORE_PATH = Path(__file__).parent / "fixtures" / "wp2_7_visibility_before.json"


def sha(array, dtype=np.float64) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array).astype(dtype)).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------- directions (G1-G2)
def d_atan(x):
    """atan of a Decimal: halve the argument until it is small, then the Taylor series."""
    k = 0
    while abs(x) > Decimal("1e-4"):
        x = x / (1 + (1 + x * x).sqrt())
        k += 1
    term, total, n, x2 = x, x, 1, x * x
    while True:
        term = -term * x2
        n += 2
        add = term / n
        if abs(add) < Decimal("1e-80"):
            break
        total += add
    return total * (2 ** k)


def ref_elevation(s, e, z):
    """60-digit elevation atan2(Z, hypot(S, E)) of the exact binary inputs."""
    getcontext().prec = 60
    half_pi = 2 * d_atan(Decimal(1))
    S, E, Z = Decimal(float(s)), Decimal(float(e)), Decimal(float(z))
    h = (S * S + E * E).sqrt()
    if abs(Z) <= h:
        return d_atan(Z / h)
    return (half_pi if Z > 0 else -half_pi) - d_atan(h / Z)


def g1_directions(seed):
    """SEZ components of the survey's construction: zenith, horizon and random directions at 3.8e8 m."""
    rng = np.random.default_rng(seed)
    rho, rows = 3.8e8, []
    for zeta in np.logspace(-12, -1, 45):
        for az in rng.uniform(0.0, TWO_PI, 6):
            rows.append((-rho * np.sin(zeta) * np.cos(az), rho * np.sin(zeta) * np.sin(az), rho * np.cos(zeta)))
    for el in np.concatenate([np.logspace(-12, -1, 45), -np.logspace(-12, -1, 45)]):
        az = rng.uniform(0.0, TWO_PI)
        rows.append((-rho * np.cos(el) * np.cos(az), rho * np.cos(el) * np.sin(az), rho * np.sin(el)))
    for _ in range(300):
        v = rng.standard_normal(3)
        rows.append(tuple(rho * v / np.linalg.norm(v)))
    return rows


def sez_to_ecef_at_origin(s, e, z):
    """At lat = lon = 0 the ECEF -> SEZ rotation is a signed permutation: SEZ = (-z, y, x) exactly."""
    return np.array([z, e, -s], dtype=float)


WEST_OF_NORTH = [np.array([1e3, -(10.0 ** k), 4e8]) for k in range(-14, 1)]
CARDINALS = {"N": np.array([0.0, 0.0, 4e8]), "E": np.array([0.0, 4e8, 0.0]),
             "S": np.array([0.0, 0.0, -4e8]), "W": np.array([0.0, -4e8, 0.0])}


def azel_python(vectors):
    from lunar_od.geometry import ecef2razel_sez
    return np.array([ecef2razel_sez(v, 0.0, 0.0) for v in vectors])


def azel_numba(vectors, lat=0.0, lon=0.0):
    from lunar_od.accelerated import _razel_scalar
    return np.array([[float(x) for x in _razel_scalar(v[0], v[1], v[2], lat, lon, TWO_PI)] for v in vectors])


def old_python_azel(rho_ecef, lat, lon):
    """The pre-WP2.7 ecef2razel_sez body (arcsin elevation, unguarded 2 pi); its sha256 is checked against the
    recorded function output before use."""
    from lunar_od.geometry import ecef2sez_dcm
    rho_sez = ecef2sez_dcm(lat, lon) @ np.asarray(rho_ecef, dtype=float)
    s, e, z = rho_sez
    rng = float(np.linalg.norm(rho_sez))
    el = float(np.arcsin(float(np.clip(z / rng, -1.0, 1.0))))
    az = float(np.arctan2(e, -s))
    if az < 0.0:
        az += TWO_PI
    return az, el, rng


# ------------------------------------------------------------------------------------------- arc
def hermite(t_grid, states, t):
    """Cubic Hermite positions and velocities from a (N, 6) history (the radiometrics._interp_state formula)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    i1 = np.clip(np.searchsorted(t_grid, t, side="right"), 1, t_grid.size - 1)
    i0 = i1 - 1
    h = (t_grid[i1] - t_grid[i0])[:, None]
    s = ((t - t_grid[i0])[:, None]) / h
    r0, v0, r1, v1 = states[i0, :3], states[i0, 3:6], states[i1, :3], states[i1, 3:6]
    pos = (2 * s**3 - 3 * s**2 + 1) * r0 + (s**3 - 2 * s**2 + s) * h * v0 + (-2 * s**3 + 3 * s**2) * r1 \
        + (s**3 - s**2) * h * v1
    vel = (6 * s**2 - 6 * s) * r0 / h + (3 * s**2 - 4 * s + 1) * v0 + (-6 * s**2 + 6 * s) * r1 / h \
        + (3 * s**2 - 2 * s) * v1
    return np.hstack([pos, vel])


def build_arc():
    """1 day, 10 s, 100 km polar lunar orbit from 2027-03-02T00:00 TDB, the three DSN complexes, 10 deg mask."""
    import spiceypy as spice
    from lunar_od.config import range_rate_stations
    from lunar_od.dynamics import propagate_state
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.visibility import VisibilityConfig, sample_j2000_to_itrf93_transforms

    load_spice_kernels(None, clear=False)
    et0 = spice.str2et("2027-03-02T00:00:00")
    t = np.arange(0.0, 86400.0 + 1.0, 10.0)
    moon_eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, t[-1] + 601.0, 60.0))
    gm = gravitational_parameters()
    r0 = R_MOON + 100e3
    v = np.sqrt(gm[0] / r0)
    x0 = np.array([r0, 0.0, 0.0, 0.0, v * np.sin(np.radians(10.0)), v * np.cos(np.radians(10.0))])
    x = np.asarray(propagate_state(t, x0, *gm, moon_eph.earth_position, moon_eph.sun_position,
                                   rtol=1e-12, atol=1e-12), dtype=float)
    if x.shape[0] != t.size:
        x = x.T

    def earth(tt):
        return np.asarray(moon_eph.earth_position(np.atleast_1d(np.asarray(tt, dtype=float))), float).reshape(-1, 3)

    config = VisibilityConfig(r_moon_mean_m=R_MOON, earth_rotation_rad_s=7.292115e-5,
                              epoch_utc="2027-03-02 00:00:00", min_elevation_deg=10.0)
    return {"spice": spice, "et0": et0, "t": t, "x": x, "earth": earth, "config": config,
            "xf": sample_j2000_to_itrf93_transforms(et0, t),
            "stations": [s for s in range_rate_stations() if "DSN" in s.name]}


def uniform_grids(a):
    """name -> (t, states, transforms) for G6/G8: the 10 s arc, every 6th sample, and a linspace grid."""
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    t_l = np.linspace(0.0, 86399.7, 8641)
    return {"uniform_10s": (a["t"], a["x"], a["xf"]),
            "uniform_60s": (a["t"][::6], a["x"][::6], a["xf"][::6]),
            "linspace": (t_l, hermite(a["t"], a["x"], t_l), sample_j2000_to_itrf93_transforms(a["et0"], t_l))}


def visible_rho_ecef(a, mask):
    """Station -> spacecraft ECEF vectors at the visible samples of each station (G3)."""
    sc_ecef = np.einsum("nij,nj->ni", a["xf"][:, :3, :3], a["x"][:, :3] - a["earth"](a["t"]))
    return {st.name: sc_ecef[mask[:, j]] - st.r_ecef_m for j, st in enumerate(a["stations"])}


@pytest.fixture(scope="module")
def arc():
    pytest.importorskip("spiceypy")
    try:
        a = build_arc()
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    from lunar_od.visibility import analyze_visibility_gap_with_transforms
    a["mask"] = analyze_visibility_gap_with_transforms(a["t"], a["x"], a["stations"], a["earth"], a["xf"], 1200.0,
                                                       a["config"])[2]
    return a


@pytest.fixture(scope="module")
def before():
    return json.loads(BEFORE_PATH.read_text())


# ------------------------------------------------------------------------------------------- slice 1
def test_g1_elevation_accuracy_vs_decimal_reference():
    worst = {"python": 0.0, "numba": 0.0}
    for seed in (7, 8):
        rows = g1_directions(seed)
        vectors = [sez_to_ecef_at_origin(*r) for r in rows]
        py, nb = azel_python(vectors), azel_numba(vectors)
        for (s, e, z), el_py, el_nb in zip(rows, py[:, 1], nb[:, 1]):
            ref = ref_elevation(s, e, z)
            worst["python"] = max(worst["python"], abs(float(Decimal(float(el_py)) - ref)))
            worst["numba"] = max(worst["numba"], abs(float(Decimal(float(el_nb)) - ref)))
    assert max(worst.values()) < 4.5e-16, worst


def test_g2_azimuth_contract(before):
    vectors = [sez_to_ecef_at_origin(*r) for seed in (7, 8) for r in g1_directions(seed)]
    for name, fn in (("python", azel_python), ("numba", azel_numba)):
        out = fn(vectors)
        assert np.all((out[:, 0] >= 0.0) & (out[:, 0] < TWO_PI)), name
        assert sha(out[:, 0]) == before["g1_directions"][name]["az"], name
        assert sha(out[:, 2]) == before["g1_directions"][name]["range"], name
        west = fn(WEST_OF_NORTH)
        assert np.all((west[:, 0] >= 0.0) & (west[:, 0] < TWO_PI)), (name, west[:, 0])
        was_two_pi = np.array(before["west_of_north"][name]["returned_two_pi"])
        assert was_two_pi.sum() == 8, name
        assert np.all(west[was_two_pi, 0] == 0.0), (name, west[was_two_pi, 0])
        assert [float(x).hex() for x in west[~was_two_pi, 0]] == \
            [h for h, w in zip(before["west_of_north"][name]["az_hex"], was_two_pi) if not w], name
        cardinal = fn(list(CARDINALS.values()))
        assert [float(x).hex() for x in cardinal[:, 0]] == before["cardinals"][name], name


def test_g3_realistic_elevation_change(arc, before):
    rho = visible_rho_ecef(arc, arc["mask"])
    for st in arc["stations"]:
        vectors = rho[st.name]
        old = np.array([old_python_azel(v, st.lat_rad, st.lon_rad) for v in vectors])
        assert sha(old[:, 1]) == before["realistic"][st.name]["python"]["el"], "reimplementation is not the old body"
        from lunar_od.geometry import ecef2razel_sez
        new = np.array([ecef2razel_sez(v, st.lat_rad, st.lon_rad) for v in vectors])
        nb = azel_numba(vectors, st.lat_rad, st.lon_rad)
        assert np.max(np.abs(new[:, 1] - old[:, 1])) < 1e-15, st.name
        assert np.max(np.abs(nb[:, 1] - new[:, 1])) < 1e-15, st.name
        for name, out in (("python", new), ("numba", nb)):
            assert sha(out[:, 0]) == before["realistic"][st.name][name]["az"], (st.name, name)
            assert sha(out[:, 2]) == before["realistic"][st.name][name]["range"], (st.name, name)


# ------------------------------------------------------------------------------------------- slice 2
def _synthetic():
    t_a = np.concatenate([np.arange(0.0, 3600.0 + 1, 10.0), np.arange(3900.0, 14400.0 + 1, 300.0)])
    vis_a = ~((t_a >= 4000.0) & (t_a <= 5500.0))
    t_b = np.concatenate([np.arange(0.0, 3600.0 + 1, 300.0), np.arange(3610.0, 14400.0 + 1, 10.0)])
    vis_b = ~((t_b >= 5000.0) & (t_b <= 5600.0))
    return (t_a, vis_a), (t_b, vis_b)


def test_g4_nonuniform_synthetic_gap_bridging():
    from lunar_od.visibility import _bridge_network_gaps, _segments_from_mask
    (t_a, vis_a), (t_b, vis_b) = _synthetic()
    seg_a = _segments_from_mask(_bridge_network_gaps(t_a, vis_a[:, None], 1200.0))
    seg_b = _segments_from_mask(_bridge_network_gaps(t_b, vis_b[:, None], 1200.0))
    assert [[t_a[i], t_a[j]] for i, j in zip(*seg_a)] == [[0.0, 3900.0], [5700.0, 14400.0]]   # 1500 s: kept open
    assert [[t_b[i], t_b[j]] for i, j in zip(*seg_b)] == [[0.0, 14400.0]]                    # 600 s: bridged


def test_g5_nonuniform_realistic_matches_full_grid(arc):
    from lunar_od.visibility import analyze_visibility_gap_with_transforms
    t = arc["t"]

    def segments(keep):
        s, e, _, _ = analyze_visibility_gap_with_transforms(t[keep], arc["x"][keep], arc["stations"], arc["earth"],
                                                            arc["xf"][keep], 1200.0, arc["config"])
        return t[keep][s], t[keep][e]

    full_s, full_e = segments(np.ones(t.size, dtype=bool))
    for keep in ((t <= 21600.0) | (np.mod(t, 120.0) == 0.0), (t > 21600.0) | (np.mod(t, 120.0) == 0.0)):
        s, e = segments(keep)
        assert s.size == full_s.size, (s.size, full_s.size)
        spacing = 120.0
        assert np.all(np.abs(s - full_s) <= spacing) and np.all(np.abs(e - full_e) <= spacing)


def test_g6_g8_uniform_grids_and_geometric_mode_bitwise(arc, before):
    from lunar_od.visibility import analyze_visibility_gap, analyze_visibility_gap_with_transforms
    for name, (t, x, xf) in uniform_grids(arc).items():
        for max_gap in MAX_GAPS:
            for kwargs in ({}, {"geometry": "geometric"}):
                out = analyze_visibility_gap_with_transforms(t, x, arc["stations"], arc["earth"], xf, max_gap,
                                                             arc["config"], **kwargs)
                key = f"{name}/{max_gap:g}"
                assert [sha(out[0], np.int64), sha(out[1], np.int64), sha(out[2], np.uint8),
                        sha(out[3], np.uint8)] == before["transforms"][key], (key, kwargs)
            if name != "linspace":
                out = analyze_visibility_gap(t, x, arc["stations"], arc["earth"], max_gap, arc["config"])
                assert [sha(out[0], np.int64), sha(out[1], np.int64), sha(out[2], np.uint8),
                        sha(out[3], np.uint8)] == before["gst"][f"{name}/{max_gap:g}"], name


def test_g7_grid_fail_closed(arc):
    from lunar_od.visibility import analyze_visibility_gap, analyze_visibility_gap_with_transforms
    t, x, xf = arc["t"][:50].copy(), arc["x"][:50], arc["xf"][:50]
    bad_grids = []
    for edit in ("nan", "decreasing", "repeated"):
        tb = t.copy()
        if edit == "nan":
            tb[10] = np.nan
        elif edit == "decreasing":
            tb[10], tb[11] = tb[11], tb[10]
        else:
            tb[11] = tb[10]
        bad_grids.append(tb)
    for tb in bad_grids:
        with pytest.raises(ValueError):
            analyze_visibility_gap_with_transforms(tb, x, arc["stations"], arc["earth"], xf, 1200.0, arc["config"])
        with pytest.raises(ValueError):
            analyze_visibility_gap(tb, x, arc["stations"], arc["earth"], 1200.0, arc["config"])


# ------------------------------------------------------------------------------------------- slice 3
def _segment_clearance(r_a, r_b, moon):
    a, b = r_a - moon, r_b - moon
    ab = b - a
    p = -np.sum(a * ab, axis=1) / np.sum(ab * ab, axis=1)
    d = np.linalg.norm(a + ab * p[:, None], axis=1) - R_MOON
    return np.where((p >= 0.0) & (p <= 1.0), d, np.inf), np.clip(p, 0.0, 1.0)


def _oracle_masks(a, t, states, two_way):
    """Independent light-time masks: SPICE sxform at every epoch (also t1), scipy CubicHermiteSpline for the
    spacecraft, Moon at the closest-point epoch, first-order aberration."""
    from scipy.interpolate import CubicHermiteSpline
    from lunar_od.geometry import ecef2sez_dcm
    spice, et0 = a["spice"], a["et0"]
    spline = CubicHermiteSpline(t, states[:, :3], states[:, 3:6], axis=0, extrapolate=True)

    def station(st, tt):
        xf = np.array([np.asarray(spice.sxform("J2000", "ITRF93", float(et0 + ti))) for ti in tt])
        return xf, np.einsum("nji,j->ni", xf[:, :3, :3], st.r_ecef_m), np.einsum("nji,j->ni", xf[:, 3:6, :3], st.r_ecef_m)

    def elevation(st, xf, los, v, sign):
        u = los / np.linalg.norm(los, axis=1)[:, None]
        beta = sign * v / C
        u = u + beta - np.sum(u * beta, axis=1)[:, None] * u
        sez = np.einsum("ij,nj->ni", ecef2sez_dcm(st.lat_rad, st.lon_rad), np.einsum("nij,nj->ni", xf[:, :3, :3], u))
        return np.arctan2(sez[:, 2], np.hypot(sez[:, 0], sez[:, 1]))

    out = np.zeros((t.size, len(a["stations"])), dtype=bool)
    for j, st in enumerate(a["stations"]):
        xf_t, r_t, v_t = station(st, t)
        tau = np.zeros(t.size)
        for _ in range(6):
            sc = spline(t - tau) - a["earth"](t - tau)
            tau = np.linalg.norm(sc - r_t, axis=1) / C
        t2 = t - tau
        _, p = _segment_clearance(r_t, sc, -a["earth"](t2))
        clear_d, _ = _segment_clearance(r_t, sc, -a["earth"](t - p * tau))
        vis = (elevation(st, xf_t, sc - r_t, v_t, 1.0) > MIN_EL) & (clear_d >= 0.0)
        if two_way:
            tau_u = tau.copy()
            for _ in range(6):
                xf_1, r_1, v_1 = station(st, t2 - tau_u)
                tau_u = np.linalg.norm(sc - r_1, axis=1) / C
            t1 = t2 - tau_u
            _, p = _segment_clearance(r_1, sc, -a["earth"](t2))
            clear_u, _ = _segment_clearance(r_1, sc, -a["earth"](t1 + p * tau_u))
            vis &= (elevation(st, xf_1, sc - r_1, v_1, -1.0) > MIN_EL) & (clear_u >= 0.0)
        out[:, j] = vis
    return out


def _flip_times(t, column):
    k = np.where(column[:-1] != column[1:])[0]
    return 0.5 * (t[k] + t[k + 1])


@pytest.mark.parametrize("mode", ["light_time_downlink", "light_time_two_way"])
def test_g9_light_time_masks_vs_oracle(arc, mode):
    from lunar_od.visibility import analyze_visibility_gap_with_transforms
    t = arc["t"]
    prod = analyze_visibility_gap_with_transforms(t, arc["x"], arc["stations"], arc["earth"], arc["xf"], 1200.0,
                                                  arc["config"], geometry=mode)[2]
    oracle = _oracle_masks(arc, t, arc["x"], mode == "light_time_two_way")
    assert not np.array_equal(prod, arc["mask"]), "light time changed nothing on this arc"
    for j in range(len(arc["stations"])):
        differing = t[prod[:, j] != oracle[:, j]]
        if differing.size:
            boundaries = _flip_times(t, oracle[:, j])
            assert all(np.min(np.abs(boundaries - d)) < 1e-3 for d in differing), (j, differing)


def test_g10_boundary_shifts(arc):
    from lunar_od.geometry import ecef2sez_dcm
    from lunar_od.visibility import analyze_visibility_gap_with_transforms, sample_j2000_to_itrf93_transforms
    t, mask = arc["t"], arc["mask"]
    usable, total, shifts = 0, 0, {"occultation": [], "set": [], "rise_down": [], "rise_two_way": []}
    for j, st in enumerate(arc["stations"]):
        for k in np.where(mask[:-1, j] != mask[1:, j])[0]:
            if k < 2 or k > t.size - 3:
                continue
            total += 1
            rho = np.einsum("nij,nj->ni", arc["xf"][k:k + 2, :3, :3],
                            arc["x"][k:k + 2, :3] - arc["earth"](t[k:k + 2])) - st.r_ecef_m
            sez = np.einsum("ij,nj->ni", ecef2sez_dcm(st.lat_rad, st.lon_rad), rho)
            el = np.arctan2(sez[:, 2], np.hypot(sez[:, 0], sez[:, 1]))
            kind = "elevation" if (el[0] > MIN_EL) != (el[1] > MIN_EL) else "occultation"
            rising = bool(mask[k + 1, j])
            tw = 0.5 * (t[k] + t[k + 1]) + np.arange(-6.0, 6.0 + 1e-9, 0.02)
            xw, xfw = hermite(t, arc["x"], tw), sample_j2000_to_itrf93_transforms(arc["et0"], tw)
            flips = {}
            for mode in ("geometric", "light_time_downlink", "light_time_two_way"):
                m = analyze_visibility_gap_with_transforms(tw, xw, [st], arc["earth"], xfw, 1200.0, arc["config"],
                                                           geometry=mode)[2][:, 0]
                flips[mode] = _flip_times(tw, m)
            if any(v.size != 1 for v in flips.values()):
                continue
            usable += 1
            down = float(flips["light_time_downlink"][0] - flips["geometric"][0])
            two = float(flips["light_time_two_way"][0] - flips["geometric"][0])
            if kind == "occultation":
                shifts["occultation"] += [down, two]
            elif not rising:
                shifts["set"] += [down, two]
            else:
                shifts["rise_down"].append(down)
                shifts["rise_two_way"].append(two)
    assert total > 0 and usable >= 0.9 * total, (usable, total)
    assert all(1.30 <= s <= 1.38 for s in shifts["occultation"]), shifts["occultation"]
    assert all(abs(s) <= 0.25 for s in shifts["set"] + shifts["rise_down"]), (shifts["set"], shifts["rise_down"])
    assert all(2.4 <= s <= 3.1 for s in shifts["rise_two_way"]), shifts["rise_two_way"]
    assert shifts["occultation"] and shifts["rise_two_way"], shifts


def test_g11_fail_closed(arc):
    from lunar_od.visibility import analyze_visibility_gap_with_transforms
    t, x, xf, args = arc["t"][:50], arc["x"][:50], arc["xf"][:50], (arc["stations"], arc["earth"])
    for mode in ("light_time_downlink", "light_time_two_way"):
        with pytest.raises(ValueError):
            analyze_visibility_gap_with_transforms(t, x, *args, xf[:, :3, :3], 1200.0, arc["config"], geometry=mode)
        with pytest.raises(ValueError):
            analyze_visibility_gap_with_transforms(t, x[:, :3], *args, xf, 1200.0, arc["config"], geometry=mode)
    with pytest.raises(ValueError):
        analyze_visibility_gap_with_transforms(t, x, *args, xf, 1200.0, arc["config"], geometry="apparent")
