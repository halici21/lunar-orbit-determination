"""WP2.7 survey before any gate (MEAS-14), no code change: how large is each visibility / az-el defect?

1. Elevation as arcsin(Z/|rho|) (geometry.ecef2razel_sez, accelerated._razel_scalar) vs atan2(Z, hypot(S, E)),
   both against a 60-digit decimal reference, near the zenith, near the horizon and at random directions.
2. Azimuth contract [0, 2 pi): a direction a hair west of north; noisy azimuths near north (informational,
   residuals are wrapped with wrap_to_pi).
3. Gap bridging on a non-uniform grid: visibility._gap_fill_network_visibility counts invisible samples with
   dt = t[1] - t[0]; compared with a time rule (midpoint gap duration <= max_gap_s).
4. Geometric vs light-time apparent visibility on a 3-day 100 km lunar orbit seen from the three DSN complexes,
   10 s grid: mask differences and boundary-time shifts (elevation rise/set, lunar occultation entry/exit) for the
   one-way downlink (spacecraft at t - tau, station at t) and two-way (uplink at t - 2 tau also clear), with the
   diurnal aberration size separately.
5. Terrain sensitivity without a DEM: at each occultation boundary the rate of the ray's clearance above the
   1737.4 km sphere, i.e. the boundary shift per km of limb relief (LOLA relief spans about -9 to +11 km).
Run from the worktree.
"""
import json
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import spiceypy as spice

from lunar_od.accelerated import _razel_scalar, _visibility_mask_ecef_numpy
from lunar_od.config import range_rate_stations
from lunar_od.dynamics import propagate_state
from lunar_od.ephemeris import sample_moon_centered_ephemeris
from lunar_od.ephemeris_generation import gravitational_parameters
from lunar_od.geometry import ecef2razel_sez, ecef2sez_dcm
from lunar_od.spice_loader import load_spice_kernels
from lunar_od.visibility import (VisibilityConfig, _gap_fill_network_visibility, _segments_from_mask,
                                 analyze_visibility_gap_with_transforms, sample_j2000_to_itrf93_transforms)

C = 299792458.0
R_MOON = 1737.4e3
MIN_EL = float(np.deg2rad(10.0))
TWO_PI = 2.0 * np.pi
getcontext().prec = 60


# ---------------------------------------------------------------- 1. elevation numerics
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


D_HALF_PI = 2 * d_atan(Decimal(1))


def ref_elevation(s, e, z):
    S, E, Z = Decimal(float(s)), Decimal(float(e)), Decimal(float(z))
    h = (S * S + E * E).sqrt()
    if abs(Z) <= h:
        return d_atan(Z / h)
    return (D_HALF_PI if Z > 0 else -D_HALF_PI) - d_atan(h / Z)


def elevation_numerics():
    rng = np.random.default_rng(7)
    rho = 3.8e8
    cases = {"zenith": [], "horizon": [], "random": []}
    for zeta in np.logspace(-12, -1, 45):
        for az in rng.uniform(0.0, TWO_PI, 6):
            cases["zenith"].append((-rho * np.sin(zeta) * np.cos(az), rho * np.sin(zeta) * np.sin(az), rho * np.cos(zeta)))
    for el in np.concatenate([np.logspace(-12, -1, 45), -np.logspace(-12, -1, 45)]):
        az = rng.uniform(0.0, TWO_PI)
        cases["horizon"].append((-rho * np.cos(el) * np.cos(az), rho * np.cos(el) * np.sin(az), rho * np.sin(el)))
    for _ in range(300):
        v = rng.standard_normal(3)
        cases["random"].append(tuple(rho * v / np.linalg.norm(v)))
    out = {}
    for name, rows in cases.items():
        err_asin, err_numba, err_atan2, same_numba = [], [], [], True
        for s, e, z in rows:
            ecef = np.array([z, e, -s])             # lat = lon = 0: SEZ = (-z_ecef, y_ecef, x_ecef) exactly
            ref = ref_elevation(s, e, z)
            _, el_code, _ = ecef2razel_sez(ecef, 0.0, 0.0)
            el_nb = _razel_scalar(ecef[0], ecef[1], ecef[2], 0.0, 0.0, TWO_PI)[1]
            el_atan2 = float(np.arctan2(z, np.hypot(s, e)))
            err_asin.append(abs(float(Decimal(el_code) - ref)))
            err_numba.append(abs(float(Decimal(float(el_nb)) - ref)))
            err_atan2.append(abs(float(Decimal(el_atan2) - ref)))
            same_numba &= float(el_nb) == el_code
        out[name] = {"n": len(rows), "max_err_arcsin_rad": max(err_asin), "max_err_numba_arcsin_rad": max(err_numba),
                     "max_err_atan2_rad": max(err_atan2), "numba_equals_python_bitwise": bool(same_numba)}
    # where the arcsin error comes from: zenith distance below which arcsin returns exactly pi/2
    flat = [zeta for zeta in np.logspace(-12, -6, 61)
            if ecef2razel_sez(np.array([3.8e8 * np.cos(zeta), 3.8e8 * np.sin(zeta), 0.0]), 0.0, 0.0)[1] == np.pi / 2]
    out["arcsin_returns_exact_pi_over_2_up_to_zenith_distance_rad"] = max(flat) if flat else None
    return out


# ---------------------------------------------------------------- 2. azimuth contract
def azimuth_contract():
    rows = []
    for k in range(-14, 1):
        east = -(10.0 ** k)
        ecef = np.array([1e3, east, 4e8])          # S = -4e8 (due north), E slightly west, Z = 1 km
        az = ecef2razel_sez(ecef, 0.0, 0.0)[0]
        az_nb = float(_razel_scalar(ecef[0], ecef[1], ecef[2], 0.0, 0.0, TWO_PI)[0])
        rows.append({"east_m": east, "true_az_minus_2pi_rad": float(np.arctan2(east, 4e8)), "az_rad": az,
                     "python_returns_2pi": bool(az == TWO_PI), "numba_returns_2pi": bool(az_nb == TWO_PI)})
    neg_zero = ecef2razel_sez(np.array([1e3, -0.0, 4e8]), 0.0, 0.0)[0]
    return {"tiny_west_of_north_at_4e8_m": rows,
            "largest_west_offset_returning_2pi_rad": max((-r["true_az_minus_2pi_rad"] for r in rows
                                                          if r["python_returns_2pi"]), default=None),
            "east_minus_zero_returns_negative_zero": bool(np.signbit(neg_zero))}


# ---------------------------------------------------------------- 3. gap bridging
def segments_current(t, vis, max_gap_s):
    filled = _gap_fill_network_visibility(np.asarray(vis, bool)[:, None], max_gap_s, float(t[1] - t[0]))
    return _segments_from_mask(filled)


def segments_time_rule(t, vis, max_gap_s):
    vis = np.asarray(vis, bool)
    starts, ends = _segments_from_mask(vis)
    filled = vis.copy()
    for m in range(len(starts) - 1):
        e, s = ends[m], starts[m + 1]
        if 0.5 * ((t[s - 1] + t[s]) - (t[e] + t[e + 1])) <= max_gap_s:
            filled[e + 1:s] = True
    return _segments_from_mask(filled)


def seg_times(t, seg):
    return [[float(t[a]), float(t[b])] for a, b in zip(*seg)]


def synthetic_grid_cases():
    out = {}
    t_a = np.concatenate([np.arange(0.0, 3600.0 + 1, 10.0), np.arange(3900.0, 14400.0 + 1, 300.0)])
    vis_a = ~((t_a >= 4000.0) & (t_a <= 5500.0))      # 1500 s outage on the 300 s part, max gap 1200 s
    t_b = np.concatenate([np.arange(0.0, 3600.0 + 1, 300.0), np.arange(3610.0, 14400.0 + 1, 10.0)])
    vis_b = ~((t_b >= 5000.0) & (t_b <= 5600.0))      # 600 s outage on the 10 s part, max gap 1200 s
    for name, t, vis, truth in (("dense_then_coarse_1500s_outage", t_a, vis_a, "not bridged"),
                                ("coarse_then_dense_600s_outage", t_b, vis_b, "bridged")):
        out[name] = {"max_gap_s": 1200.0, "expected": truth,
                     "current_segments_s": seg_times(t, segments_current(t, vis, 1200.0)),
                     "time_rule_segments_s": seg_times(t, segments_time_rule(t, vis, 1200.0))}
    return out


# ---------------------------------------------------------------- 4. apparent visibility
def hermite(t_grid, states, t):
    t = np.atleast_1d(np.asarray(t, float))
    i1 = np.clip(np.searchsorted(t_grid, t, side="right"), 1, t_grid.size - 1)
    i0 = i1 - 1
    h = (t_grid[i1] - t_grid[i0])[:, None]
    s = ((t - t_grid[i0]) / h[:, 0])[:, None]
    r0, v0, r1, v1 = states[i0, :3], states[i0, 3:6], states[i1, :3], states[i1, 3:6]
    return ((2 * s**3 - 3 * s**2 + 1) * r0 + (s**3 - 2 * s**2 + s) * h * v0
            + (-2 * s**3 + 3 * s**2) * r1 + (s**3 - s**2) * h * v1)


def clearance(r_obs, r_tgt, r_body):
    """Closest distance of the obs->tgt segment to the body centre minus R_MOON (inf if the closest point is
    an endpoint outside [0, 1]); same geometry as visibility.is_occulted_by_moon."""
    a, b = r_obs - r_body, r_tgt - r_body
    ab = b - a
    tau = -np.sum(a * ab, axis=1) / np.sum(ab * ab, axis=1)
    c = a + ab * tau[:, None]
    d = np.linalg.norm(c, axis=1) - R_MOON
    return np.where((tau >= 0.0) & (tau <= 1.0), d, np.inf)


class Arc:
    def __init__(self, days=3.0, step=10.0):
        load_spice_kernels(None, clear=False)
        self.et0 = spice.str2et("2027-03-02T00:00:00")
        self.t = np.arange(0.0, days * 86400.0 + 1.0, step)
        self.t_hist = np.arange(-60.0, self.t[-1] + 61.0, step)
        self.moon_eph = sample_moon_centered_ephemeris(self.et0, np.arange(-600.0, self.t[-1] + 601.0, 60.0))
        gm = gravitational_parameters()
        r0 = R_MOON + 100e3
        v = np.sqrt(gm[0] / r0)
        x0 = np.array([r0, 0.0, 0.0, 0.0, v * np.sin(np.radians(10.0)), v * np.cos(np.radians(10.0))])
        self.x_hist = np.asarray(propagate_state(self.t_hist, x0, *gm, self.moon_eph.earth_position,
                                                 self.moon_eph.sun_position, rtol=1e-12, atol=1e-12))
        if self.x_hist.shape[0] != self.t_hist.size:
            self.x_hist = self.x_hist.T
        self.x = self.x_hist[6:6 + self.t.size]
        assert np.allclose(self.t_hist[6:6 + self.t.size], self.t)
        self.xf = sample_j2000_to_itrf93_transforms(self.et0, self.t)
        self.stations = [s for s in range_rate_stations() if "DSN" in s.name]

    def earth(self, t):
        return np.asarray(self.moon_eph.earth_position(np.atleast_1d(t)), float).reshape(-1, 3)

    def rot(self, t):
        return np.array([np.asarray(spice.pxform("J2000", "ITRF93", float(self.et0 + ti))) for ti in np.atleast_1d(t)])

    def xform6(self, t):
        return np.array([np.asarray(spice.sxform("J2000", "ITRF93", float(self.et0 + ti))) for ti in np.atleast_1d(t)])

    def sc_geo(self, t):
        """spacecraft geocentric, J2000 axes."""
        return hermite(self.t_hist, self.x_hist, t) - self.earth(t)

    def evaluate(self, st, t, xf6=None, mode="geometric", aberration=False):
        """elevation (rad) and clearance (m) for reception at t; mode geometric | down | up."""
        t = np.atleast_1d(np.asarray(t, float))
        xf6 = self.xform6(t) if xf6 is None else xf6
        r_ecef = st.r_ecef_m
        rot = xf6[:, :3, :3]
        r_st = np.einsum("nji,j->ni", rot, r_ecef)
        v_st = np.einsum("nji,j->ni", xf6[:, 3:6, :3], r_ecef)
        if mode == "geometric":
            sc = self.sc_geo(t)
            moon = -self.earth(t)
            rot_view, r_view, v_view, sign = rot, r_st, v_st, 1.0
        else:
            tau = np.zeros(t.size)
            for _ in range(4):
                sc = self.sc_geo(t - tau)
                tau = np.linalg.norm(sc - r_st, axis=1) / C
            t2 = t - tau
            moon = -self.earth(t2)
            rot_view, r_view, v_view, sign = rot, r_st, v_st, 1.0
            if mode == "up":
                tau_u = tau.copy()
                for _ in range(4):
                    xf_u = self.xform6(t2 - tau_u)
                    r_up = np.einsum("nji,j->ni", xf_u[:, :3, :3], r_ecef)
                    tau_u = np.linalg.norm(sc - r_up, axis=1) / C
                rot_view, r_view = xf_u[:, :3, :3], r_up
                v_view, sign = np.einsum("nji,j->ni", xf_u[:, 3:6, :3], r_ecef), -1.0
        los = sc - r_view
        u = los / np.linalg.norm(los, axis=1)[:, None]
        if aberration:   # reception: towards +v; transmission (point-ahead): towards -v
            beta = sign * v_view / C
            u = u + beta - np.sum(u * beta, axis=1)[:, None] * u
        sez = np.einsum("ij,nj->ni", ecef2sez_dcm(st.lat_rad, st.lon_rad), np.einsum("nij,nj->ni", rot_view, u))
        el = np.arctan2(sez[:, 2], np.hypot(sez[:, 0], sez[:, 1]))
        az = np.mod(np.arctan2(sez[:, 1], -sez[:, 0]), TWO_PI)
        return el, clearance(r_view, sc, moon), az


def bisect(f, a, b, iters=60):
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def apparent_visibility(arc):
    t, n = arc.t, arc.t.size
    config = VisibilityConfig(r_moon_mean_m=R_MOON, earth_rotation_rad_s=7.292115e-5,
                              epoch_utc="2027-03-02 00:00:00", min_elevation_deg=10.0)
    _, _, vis_prod, _ = analyze_visibility_gap_with_transforms(t, arc.x, arc.stations, arc.earth, arc.xf, 1200.0, config)
    r_ecef_sc = np.einsum("nij,nj->ni", arc.xf[:, :3, :3], arc.x[:, :3] - arc.earth(t))
    r_ecef_moon = np.einsum("nij,nj->ni", arc.xf[:, :3, :3], -arc.earth(t))
    st_ecef = np.vstack([s.r_ecef_m for s in arc.stations])
    vis_numpy = _visibility_mask_ecef_numpy(r_ecef_sc, r_ecef_moon, st_ecef, np.array([s.lat_rad for s in arc.stations]),
                                            np.array([s.lon_rad for s in arc.stations]), MIN_EL, R_MOON)
    out = {"arc": "2027-03-02T00:00 TDB + 3 d, 10 s grid, 100 km polar lunar orbit, DSN complexes, 10 deg mask",
           "production_numba_equals_numpy_fallback": bool(np.array_equal(vis_prod, vis_numpy))}
    per_station, boundaries = {}, []
    north_counts = {}
    for j, st in enumerate(arc.stations):
        el_g, cl_g, az_g = arc.evaluate(st, t, arc.xf, "geometric")
        el_d, cl_d, _ = arc.evaluate(st, t, arc.xf, "down")
        el_u, cl_u, _ = arc.evaluate(st, t, None, "up")
        el_da, _, _ = arc.evaluate(st, t, arc.xf, "down", aberration=True)
        el_ua, _, _ = arc.evaluate(st, t, None, "up", aberration=True)
        vis_g = (el_g > MIN_EL) & (cl_g >= 0.0)
        vis_d = (el_d > MIN_EL) & (cl_d >= 0.0)
        vis_2w = vis_d & (el_u > MIN_EL) & (cl_u >= 0.0)
        per_station[st.name] = {
            "visible_samples_geometric": int(vis_g.sum()),
            "survey_geometric_equals_production": bool(np.array_equal(vis_g, vis_prod[:, j])),
            "samples_differing_downlink": int(np.sum(vis_g != vis_d)),
            "samples_differing_two_way": int(np.sum(vis_g != vis_2w)),
            "max_abs_downlink_elevation_minus_geometric_rad": float(np.max(np.abs(el_d - el_g)[vis_g])),
            "max_abs_diurnal_aberration_elevation_rad": float(max(np.max(np.abs(el_da - el_d)[vis_g]),
                                                                  np.max(np.abs(el_ua - el_u)[vis_g]))),
        }
        az_vis = az_g[vis_g]
        north_counts[st.name] = int(np.sum(np.minimum(az_vis, TWO_PI - az_vis) < 3.0 * 1e-4))
        flips = np.where(vis_g[:-1] != vis_g[1:])[0]
        for k in flips:
            ta, tb = t[k], t[k + 1]
            el_flip = (el_g[k] > MIN_EL) != (el_g[k + 1] > MIN_EL)
            occ_flip = (cl_g[k] >= 0.0) != (cl_g[k + 1] >= 0.0)
            if el_flip == occ_flip:
                continue
            kind = "elevation" if el_flip else "occultation"
            rising = bool(vis_g[k + 1])
            lo, hi = ta - 10.0, tb + 10.0

            def g(mode, ab=False, kind=kind):
                def f(x):
                    el, cl, _ = arc.evaluate(st, x, None, mode, aberration=ab)
                    return float(el[0] - MIN_EL) if kind == "elevation" else float(min(cl[0], 1e7))
                return f

            def g2w(kind=kind):
                fd, fu = g("down", kind=kind), g("up", kind=kind)
                return lambda x: min(fd(x), fu(x))

            roots = {}
            for label, f in (("geo", g("geometric")), ("down", g("down")), ("two_way", g2w()),
                             ("down_aberr", g("down", True))):
                if np.sign(f(lo)) == np.sign(f(hi)):
                    roots[label] = None
                else:
                    roots[label] = bisect(f, lo, hi)
            if roots["geo"] is None or roots["down"] is None:
                continue
            row = {"station": st.name, "kind": kind, "rising": rising, "t_geo_s": roots["geo"],
                   "shift_down_s": roots["down"] - roots["geo"],
                   "shift_two_way_s": None if roots["two_way"] is None else roots["two_way"] - roots["geo"],
                   "shift_down_aberration_minus_down_s": None if roots["down_aberr"] is None
                   else roots["down_aberr"] - roots["down"]}
            if kind == "occultation":
                fd = g("down")
                rate = (fd(roots["down"] + 0.5) - fd(roots["down"] - 0.5)) / 1.0
                row["clearance_rate_mps"] = rate
                row["shift_per_km_relief_s"] = 1000.0 / abs(rate)
            boundaries.append(row)
    out["per_station"] = per_station
    out["visible_samples_within_3_sigma_1e-4_rad_of_north"] = north_counts

    def stats(vals):
        vals = np.asarray([v for v in vals if v is not None], float)
        if vals.size == 0:
            return None
        return {"n": int(vals.size), "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max())}

    summary = {}
    for kind in ("elevation", "occultation"):
        for rising in (True, False):
            rows = [b for b in boundaries if b["kind"] == kind and b["rising"] == rising]
            key = f"{kind}_{'rise_or_exit' if rising else 'set_or_entry'}"
            summary[key] = {"shift_down_s": stats([b["shift_down_s"] for b in rows]),
                            "shift_two_way_s": stats([b["shift_two_way_s"] for b in rows]),
                            "aberration_extra_shift_s": stats([b["shift_down_aberration_minus_down_s"] for b in rows])}
            if kind == "occultation":
                summary[key]["shift_per_km_relief_s"] = stats([b["shift_per_km_relief_s"] for b in rows])
    out["boundary_shift_summary"] = summary
    out["boundaries"] = boundaries
    return out, vis_prod


def realistic_nonuniform(arc, vis_prod):
    """Network mask of the 3-day arc resampled to a non-uniform grid (10 s for 12 h, then 120 s; and the
    reverse); segments by the current rule, the time rule, and the time rule on the full 10 s grid."""
    t = arc.t
    net = np.any(vis_prod, axis=1)
    full = seg_times(t, segments_time_rule(t, net, 1200.0))
    out = {"full_10s_current_equals_time_rule": seg_times(t, segments_current(t, net, 1200.0)) == full,
           "full_10s_segments": len(full)}
    for name, keep in (("10s_then_120s", (t <= 43200.0) | (np.mod(t, 120.0) == 0.0)),
                       ("120s_then_10s", (t > 43200.0) | (np.mod(t, 120.0) == 0.0))):
        ts, vs = t[keep], net[keep]
        cur = seg_times(ts, segments_current(ts, vs, 1200.0))
        tim = seg_times(ts, segments_time_rule(ts, vs, 1200.0))
        out[name] = {"samples": int(ts.size), "current_segments": len(cur), "time_rule_segments": len(tim),
                     "current_equals_time_rule": cur == tim}
    for dt, mg in ((10.0, 1800.0), (60.0, 1200.0), (60.0, 1800.0)):
        keep = np.mod(t, dt) == 0.0
        ts, vs = t[keep], net[keep]
        out[f"uniform_{int(dt)}s_max_gap_{int(mg)}s_current_equals_time_rule"] = (
            seg_times(ts, segments_current(ts, vs, mg)) == seg_times(ts, segments_time_rule(ts, vs, mg)))
    return out


def main():
    out = {"elevation_numerics": elevation_numerics(), "azimuth_contract": azimuth_contract(),
           "nonuniform_grid_synthetic": synthetic_grid_cases()}
    arc = Arc()
    out["apparent_visibility"], vis_prod = apparent_visibility(arc)
    out["nonuniform_grid_realistic"] = realistic_nonuniform(arc, vis_prod)
    path = Path("studies/phase17/results/wp2_7_survey.json")
    path.write_text(json.dumps(out, indent=1))
    brief = {k: v for k, v in out.items() if k != "apparent_visibility"}
    brief["apparent_visibility"] = {k: v for k, v in out["apparent_visibility"].items() if k != "boundaries"}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
