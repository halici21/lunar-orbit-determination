"""WP2.6 survey before any gate (MEAS-13), no code change: how large is each hidden-module defect?

Landmark optical (lunar_landmark_optical):
  1. landmark lat/lon read in MOON_PA_DE440 vs MOON_ME (the IAU/LOLA catalogue frame): surface displacement;
  2. nadir attitude: the analytic d(u,v)/dr holds the camera DCM fixed; if the DCM is built from the estimated
     position (nadir_pointing_camera_frame(r)), the true derivative also carries dC/dr. Relative difference;
  3. principal point W/2 vs (W-1)/2 (pixel-centre convention): fixed 0.5 px.
DeltaDOR (delta_dor): the module computes D_S with Newtonian light time in the Moon-centred frame and
D_Q = -(r_B - r_A).s/c with a static baseline. Reference: Newtonian light time in the SSB frame for D_S, moved to geocentric time, and the
IERS 2010 consensus geometric delay (eq. 11.9, gravity term omitted) for D_Q. Goldstone-Canberra and
Madrid-Canberra (the Moon is south on this date), both stations above 10 deg, quasar 3 deg from the spacecraft. Run from the worktree.
"""
import json
from pathlib import Path

import numpy as np
import spiceypy as spice

from lunar_od.config import Station
from lunar_od.delta_dor import QuasarDirection, quasar_differential_delay_s, solve_common_transmit_event
from lunar_od.lunar_landmark_optical import (DEFAULT_CAMERA, landmark_inertial_position_m, nadir_pointing_camera_frame,
                                             pinhole_position_jacobian, pinhole_project)
from lunar_od.spice_loader import load_spice_kernels

C = 299792458.0
R_MOON = 1737.4e3
load_spice_kernels(None, clear=False)
ET0 = spice.str2et("2027-03-02T00:00:00")
GOLDSTONE = Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4)
MADRID = Station("Madrid", 40.4314, -4.2481, 834.0, (0, 0, 0), 2.0, 1e-4, 1e-4)
CANBERRA = Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4)


def unit(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def landmark_survey():
    out = {}
    disp = []
    for lat in (-60.0, -20.0, 0.0, 30.0, 70.0):
        for lon in (0.0, 90.0, 180.0, 270.0):
            r_pa = landmark_inertial_position_m(np.radians(lat), np.radians(lon), R_MOON, ET0)
            r_me = np.asarray(spice.pxform("J2000", "MOON_ME", ET0)).T @ (R_MOON * unit(lat, lon))
            disp.append(float(np.linalg.norm(r_pa - r_me)))
    out["pa_vs_me_surface_displacement_m"] = {"max": max(disp), "min": min(disp)}
    out["pa_vs_me_in_pixels_at_100km"] = max(disp) / DEFAULT_CAMERA.ground_sample_distance_m(100e3)

    rel_fixed, rel_coupled = [], []
    rng = np.random.default_rng(6)
    for _ in range(40):
        lm_dir = rng.standard_normal(3)
        lm_dir /= np.linalg.norm(lm_dir)
        r_lm = R_MOON * lm_dir
        tilt = rng.standard_normal(3)
        tilt -= tilt @ lm_dir * lm_dir
        tilt /= np.linalg.norm(tilt)
        r_sc = (R_MOON + 100e3) * lm_dir + rng.uniform(2e3, 14e3) * tilt   # landmark inside the ~9 deg half-FOV
        c0 = nadir_pointing_camera_frame(r_sc)
        if not pinhole_project(r_sc, r_lm, c0).within_fov:
            continue
        analytic = pinhole_position_jacobian(r_sc, r_lm, c0)
        fd_fixed, fd_coupled = np.zeros((2, 3)), np.zeros((2, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.0
            for sign in (1, -1):
                pf = pinhole_project(r_sc + sign * e, r_lm, c0)
                pc = pinhole_project(r_sc + sign * e, r_lm, nadir_pointing_camera_frame(r_sc + sign * e))
                fd_fixed[:, j] += sign * np.array([pf.u_px, pf.v_px]) / 2.0
                fd_coupled[:, j] += sign * np.array([pc.u_px, pc.v_px]) / 2.0
        rel_fixed.append(float(np.linalg.norm(analytic - fd_fixed) / np.linalg.norm(fd_fixed)))
        rel_coupled.append(float(np.linalg.norm(analytic - fd_coupled) / np.linalg.norm(fd_coupled)))
    out["jacobian_fixed_attitude_vs_fd_rel_max"] = max(rel_fixed)
    out["jacobian_vs_nadir_coupled_fd_rel"] = {"max": max(rel_coupled), "median": float(np.median(rel_coupled)),
                                               "n": len(rel_coupled)}
    out["principal_point_offset_px"] = 0.5
    return out


def state(target, observer, et):
    return np.asarray(spice.spkezr(target, et, "J2000", "NONE", observer)[0]) * 1e3


def station(st, et, origin):
    """(position relative to ``origin``, geocentric J2000 state) of a station."""
    sv = np.asarray(spice.sxform("ITRF93", "J2000", et)) @ np.concatenate([st.r_ecef_m, np.zeros(3)])
    return state("EARTH", origin, et)[:3] + sv[:3], sv


def sc_offset(et):
    """Spacecraft relative to the Moon: 100 km circular orbit in the J2000 x-z plane."""
    ph = 2 * np.pi / 7080.0 * (et - ET0)
    return (R_MOON + 100e3) * np.array([np.cos(ph), 0.0, np.sin(ph)])


def sc_ssb(et):
    return state("MOON", "SSB", et)[:3] + sc_offset(et)


def elevation(st, et):
    p, sv = station(st, et, "SSB")
    d = sc_ssb(et) - p
    return np.degrees(np.arcsin(d @ sv[:3] / np.linalg.norm(d) / np.linalg.norm(sv[:3])))


def ddor_survey():
    rows = []
    for (sta, stb), hour in ((pair, h) for pair in ((GOLDSTONE, CANBERRA), (MADRID, CANBERRA))
                             for h in np.arange(0.0, 24.0, 0.25)):
        t_obs = ET0 + hour * 3600.0
        if min(elevation(sta, t_obs), elevation(stb, t_obs)) < 10.0:
            continue
        los = sc_ssb(t_obs) - state("EARTH", "SSB", t_obs)[:3]
        los /= np.linalg.norm(los)
        perp = np.cross(los, [0.0, 0.0, 1.0])
        perp /= np.linalg.norm(perp)
        s_hat = np.cos(np.radians(3.0)) * los + np.sin(np.radians(3.0)) * perp
        quasar = QuasarDirection.from_unit_vector(s_hat)

        # the module, Moon-centred (local time offsets from t_obs)
        ra_m = station(sta, t_obs, "MOON")[0]
        ev = solve_common_transmit_event(0.0, ra_m, lambda dt: station(stb, t_obs + dt, "MOON")[0],
                                         lambda dt: sc_offset(t_obs + dt))
        dq_mod = quasar_differential_delay_s(ra_m, station(stb, t_obs, "MOON")[0], quasar)

        # SSB Newtonian reference for D_S, same transmit event
        ra_s = station(sta, t_obs, "SSB")[0]
        tau_a = 1.3
        for _ in range(20):
            tau_a = np.linalg.norm(ra_s - sc_ssb(t_obs - tau_a)) / C
        t_tx = t_obs - tau_a
        tau_b = tau_a
        for _ in range(20):
            tau_b = np.linalg.norm(station(stb, t_tx + tau_b, "SSB")[0] - sc_ssb(t_tx)) / C
        # BCRS coordinate-time interval -> geocentric (TT-like) interval, first order in V/c (the same
        # transformation that gives eq. 11.9 its -(V.b)/c^2 term): dt_geo = dt_bcrs - V.(x_B(t_B) - x_A(t_A))/c^2
        v_e = state("EARTH", "SSB", t_obs)[3:]
        b_events = station(stb, t_tx + tau_b, "EARTH")[0] - station(sta, t_obs, "EARTH")[0]
        ds_ssb = (tau_b - tau_a) - (v_e @ b_events) / C ** 2

        # IERS 2010 eq. 11.9 geometric delay, gravity omitted
        b = station(stb, t_obs, "EARTH")[0] - station(sta, t_obs, "EARTH")[0]
        wb = station(stb, t_obs, "EARTH")[1][3:]
        v = state("EARTH", "SSB", t_obs)[3:]
        dq_iers = ((-(b @ s_hat) / C * (1 - (v @ v) / (2 * C ** 2) - (v @ wb) / C ** 2)
                    - (v @ b) / C ** 2 * (1 + (s_hat @ v) / (2 * C))) / (1 + s_hat @ (v + wb) / C))
        rows.append({"pair": f"{sta.name}-{stb.name}", "hour": float(hour),
                     "ds_module_minus_ssb_ns": (ev.spacecraft_differential_delay_s - ds_ssb) * 1e9,
                     "dq_module_minus_iers_ns": (dq_mod - dq_iers) * 1e9,
                     "ddor_module_minus_reference_ns": ((ev.spacecraft_differential_delay_s - dq_mod)
                                                        - (ds_ssb - dq_iers)) * 1e9,
                     "dq_iers_ms": dq_iers * 1e3})

    def stat(key):
        vals = np.array([r[key] for r in rows])
        return {"max_abs": float(np.abs(vals).max()), "min": float(vals.min()), "max": float(vals.max())}

    return {"n_epochs": len(rows), "ds_module_minus_ssb_ns": stat("ds_module_minus_ssb_ns"),
            "dq_module_minus_iers_ns": stat("dq_module_minus_iers_ns"),
            "ddor_module_minus_reference_ns": stat("ddor_module_minus_reference_ns"),
            "typical_ddor_precision_ns": 0.1, "rows": rows}


def main():
    out = {"landmark": landmark_survey(), "ddor": ddor_survey()}
    Path("studies/phase17/results/wp2_6_survey.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["landmark"], indent=1))
    print(json.dumps({k: v for k, v in out["ddor"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
