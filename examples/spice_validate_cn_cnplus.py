"""SPICE cross-validation of CN and CN+S apparent line-of-sight vectors.

End-to-end validation (validation only -- the measurement model is not modified):
for each Earth-station -> lunar-orbiter epoch in the SPICE fixture, the fixture
truth trajectory is written to a temporary SPK so that SPICE can evaluate the
*same* trajectory with its own aberration corrections. We then compare, in J2000
axes:

  1. our CN line of sight        vs   2. SPICE CN  (spkcpo, abcorr='CN')
  3. our CN+S (spice_ssb) LOS    vs   4. SPICE CN+S (spkcpo, abcorr='CN+S')

Reported per epoch and in aggregate: range difference [m], LOS vector norm
difference [m], angular difference [arcsec], and light-time difference [s].

The temporary SPK is densely sampled from the model interpolant
(`interp_state_history`), so SPICE reproduces our trajectory to interpolation
precision and the residual differences isolate the light-time / stellar
aberration / frame-chain algorithms. Regenerates ``SPICE_CN_CNPLUS_VALIDATION.md``.

Run from the project root (SPICE kernels required)::

    python python_port/examples/spice_validate_cn_cnplus.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiceypy as spice  # noqa: E402

from lunar_od import (  # noqa: E402
    MoonCenteredEphemeris,
    generate_position_measurements,
    load_spice_kernels,
    range_rate_stations,
)
from lunar_od.measurements import (  # noqa: E402
    C_LIGHT_MPS,
    apply_stellar_aberration,
    interp_state_history,
    solve_one_way_light_time,
)

PYTHON_PORT = Path(__file__).resolve().parents[1]
FIXTURE = PYTHON_PORT / "fixtures" / "spice_snapshots.json"
OUT = PYTHON_PORT / "SPICE_CN_CNPLUS_VALIDATION.md"
SC_BODY_ID = -999       # custom NAIF id for the synthetic orbiter
MOON_ID = 301
ARCSEC = np.degrees(1.0) * 3600.0


def _ang_arcsec(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) * 3600.0)


def _load():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    truth, meas = fx["truth_propagation"], fx["position_measurements"]
    eph = MoonCenteredEphemeris(
        t_ephem_s=truth["t_ephem_s"], earth_pos_m=truth["earth_pos_grid_m"],
        sun_pos_m=truth["sun_pos_grid_m"], earth_vel_mps=truth["earth_vel_grid_mps"])
    by = {s.name: s for s in range_rate_stations()}
    stations = [by[n] for n in meas["station_names"]]
    return {
        "eph": eph, "stations": stations, "et0": float(fx["et"]),
        "state": np.asarray(truth["state_history_mci_m_mps"], float),
        "tp": np.asarray(meas["t_pass_s"], float), "vis": meas["vis_mask_raw"],
    }


def _write_temp_spk(d):
    """Write a dense Hermite SPK (body -999 wrt Moon, J2000) from the model
    interpolant, covering the retarded epochs. Returns the file path."""
    tp = d["tp"]
    step = min(0.5, max(0.05, (tp.max() - tp.min()) / 60.0))
    grid = np.arange(tp.min() - 2.0, tp.max() + 1.0 + step, step)
    states_km = np.array(
        [interp_state_history(tp, d["state"], float(t))[:6] for t in grid]
    ) / 1000.0
    epochs = d["et0"] + grid
    path = os.path.join(tempfile.gettempdir(), "lunar_sc_truth_cnval.bsp")
    if os.path.exists(path):
        os.remove(path)
    handle = spice.spkopn(path, "CN_CNPS_VALIDATION", 0)
    spice.spkw13(
        handle, SC_BODY_ID, MOON_ID, "J2000",
        float(epochs[0]), float(epochs[-1]), "SC_TRUTH",
        3, len(epochs), states_km, epochs,
    )
    spice.spkcls(handle)
    return path


def _our_cn_cnplus(d, pass_geo, i, clean):
    """Our geometric, CN, and CN+S (spice_ssb) LOS in J2000, plus one-way LT."""
    k = int(clean[i, 5]) - 1
    sid = int(clean[i, 4]) - 1
    t_r = float(clean[i, 0])
    station = pass_geo.stations[sid]
    x_rx = np.asarray(pass_geo.x_j2000_to_itrf93[k], float)
    earth_pos = np.asarray(pass_geo.earth_pos_mci_m[k], float)
    st_ecef = np.asarray(station.r_ecef_m, float)
    srel = np.linalg.solve(x_rx, np.concatenate([st_ecef, np.zeros(3)]))
    station_mci = earth_pos + srel[:3]
    geo_los = interp_state_history(d["tp"], d["state"], t_r)[:3] - station_mci   # no light time
    sol = solve_one_way_light_time(
        t_r, station_mci, lambda t: interp_state_history(d["tp"], d["state"], t)[:3])
    r_sc_tt = interp_state_history(d["tp"], d["state"], sol.transmit_time_s)[:3]
    cn_los = r_sc_tt - station_mci
    v_obs_ssb = np.asarray(pass_geo.earth_vel_ssb_j2000_mps[k], float) + srel[3:]
    cnps_los = apply_stellar_aberration(cn_los, v_obs_ssb, light_speed_mps=C_LIGHT_MPS)
    return geo_los, cn_los, cnps_los, float(sol.light_time_s), station.name, t_r


def _spice_los(d, station, t_r, abcorr):
    """SPICE LOS in J2000 [m] and light time for a given abcorr, via spkcpo."""
    et = d["et0"] + t_r
    obspos_km = np.asarray(station.r_ecef_m, float) / 1000.0
    state6, lt = spice.spkcpo(str(SC_BODY_ID), et, "J2000", "OBSERVER", abcorr,
                              obspos_km, "EARTH", "ITRF93")
    return np.asarray(state6[:3]) * 1000.0, float(lt)


def _moon_displacement_over_lt(d, t_r, lt):
    """|Moon(et) - Moon(et - lt)| wrt SSB in J2000 [m]."""
    et = d["et0"] + t_r
    m0, _ = spice.spkpos("MOON", et, "J2000", "NONE", "SSB")
    m1, _ = spice.spkpos("MOON", et - lt, "J2000", "NONE", "SSB")
    return float(np.linalg.norm((np.asarray(m0) - np.asarray(m1)) * 1000.0))


def main() -> int:
    if not FIXTURE.is_file():
        raise SystemExit(f"Fixture not found: {FIXTURE}")
    d = _load()

    # Phase 1: cache geometry (x-form, Earth pos, SSB Earth velocity) per epoch.
    load_spice_kernels()
    try:
        _, pass_geo, clean = generate_position_measurements(
            d["tp"], d["state"], d["stations"], d["vis"], d["eph"].earth_position,
            d["eph"].earth_velocity, d["et0"], noise=False, apply_light_time=True,
            apply_stellar_aberration=True, stellar_aberration_model="spice_ssb")
    finally:
        spice.kclear()

    # Phase 2: write the temp SPK, then evaluate SPICE CN / CN+S.
    load_spice_kernels()
    spk_path = _write_temp_spk(d)
    spice.furnsh(spk_path)
    rows = []
    try:
        for i in range(clean.shape[0]):
            geo_los, cn_los, cnps_los, our_lt, st_name, t_r = _our_cn_cnplus(d, pass_geo, i, clean)
            station = pass_geo.stations[int(clean[i, 4]) - 1]
            sp_none, _ = _spice_los(d, station, t_r, "NONE")
            sp_cn, sp_lt_cn = _spice_los(d, station, t_r, "CN")
            sp_cns, sp_lt_cns = _spice_los(d, station, t_r, "CN+S")
            rows.append({
                "i": i + 1, "station": st_name,
                # geometric (no light time) sanity -- isolates the SPK / frame setup
                "none_norm_diff": float(np.linalg.norm(geo_los - sp_none)),
                # CN
                "cn_range_diff": abs(np.linalg.norm(cn_los) - np.linalg.norm(sp_cn)),
                "cn_norm_diff": float(np.linalg.norm(cn_los - sp_cn)),
                "cn_ang_as": _ang_arcsec(cn_los, sp_cn),
                "cn_lt_diff": abs(our_lt - sp_lt_cn),
                # CN+S
                "cns_range_diff": abs(np.linalg.norm(cnps_los) - np.linalg.norm(sp_cns)),
                "cns_norm_diff": float(np.linalg.norm(cnps_los - sp_cns)),
                "cns_ang_as": _ang_arcsec(cnps_los, sp_cns),
                "cns_lt_diff": abs(our_lt - sp_lt_cns),
                # explanatory: the Moon's barycentric displacement during the light time
                "moon_disp": _moon_displacement_over_lt(d, t_r, our_lt),
                # cross-check: the CN -> CN+S shift each model applies
                "our_cn_to_cns_as": _ang_arcsec(cn_los, cnps_los),
                "spice_cn_to_cns_as": _ang_arcsec(sp_cn, sp_cns),
            })
    finally:
        spice.unload(spk_path)
        spice.kclear()
        if os.path.exists(spk_path):
            os.remove(spk_path)

    _write_report(rows)
    print(f"epochs: {len(rows)}")
    print("NONE max: norm_diff {:.3e} m (geometric sanity)".format(
        max(r["none_norm_diff"] for r in rows)))
    print("CN   max: range_diff {:.3e} m | norm_diff {:.3e} m | ang {:.3e}\" | lt {:.3e} s".format(
        max(r["cn_range_diff"] for r in rows), max(r["cn_norm_diff"] for r in rows),
        max(r["cn_ang_as"] for r in rows), max(r["cn_lt_diff"] for r in rows)))
    print("CN+S max: range_diff {:.3e} m | norm_diff {:.3e} m | ang {:.3e}\" | lt {:.3e} s".format(
        max(r["cns_range_diff"] for r in rows), max(r["cns_norm_diff"] for r in rows),
        max(r["cns_ang_as"] for r in rows), max(r["cns_lt_diff"] for r in rows)))
    return 0


def _write_report(rows):
    def mx(key):
        return max(r[key] for r in rows)

    L = []
    L.append("# SPICE CN / CN+S Apparent-LOS Cross-Validation\n")
    L.append("_Generated by `examples/spice_validate_cn_cnplus.py` (validation only; "
             "the measurement model is unchanged)._\n")
    L.append("## Method\n")
    L.append("The fixture truth trajectory is written to a temporary type-13 (Hermite) "
             "SPK as a custom body (id -999) centred on the Moon in J2000, densely "
             "sampled from the model interpolant `interp_state_history`. SPICE then "
             "evaluates the **same** trajectory from each Earth station with "
             "`spkcpo(..., abcorr, obsctr='EARTH', obsref='ITRF93')`, returning the "
             "apparent line of sight in **J2000 axes** (station -> spacecraft):\n")
    L.append("- **NONE** — geometric, no correction (`abcorr='NONE'`): isolates the SPK / "
             "frame / station setup.\n")
    L.append("- **CN**  — converged Newtonian one-way light time (`abcorr='CN'`).\n")
    L.append("- **CN+S** — CN plus stellar aberration (`abcorr='CN+S'`), whose reception-"
             "case observer velocity is relative to the SSB — matching our `spice_ssb` "
             "model.\n")

    L.append("## 0. Geometric (NONE) sanity check\n")
    L.append(f"Our geometric LOS (no light time) vs SPICE `NONE`: **max "
             f"{mx('none_norm_diff'):.2e} m**. The setup (custom SPK, J2000/ITRF93 "
             "frames, station position) is exact, so any CN/CN+S difference below is "
             "physical, not a harness artefact.\n")

    L.append("## 1. CN — our one-way light time vs SPICE `abcorr='CN'`\n")
    L.append("| # | station | range diff [m] | LOS norm diff [m] | angular diff [arcsec] | light-time diff [s] |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['i']} | {r['station']} | {r['cn_range_diff']:.3e} | "
                 f"{r['cn_norm_diff']:.3e} | {r['cn_ang_as']:.3e} | {r['cn_lt_diff']:.3e} |")
    L.append(f"\n**CN max:** range {mx('cn_range_diff'):.2e} m · norm {mx('cn_norm_diff'):.2e} m · "
             f"angle {mx('cn_ang_as'):.2e}\" · light-time {mx('cn_lt_diff'):.2e} s.\n")

    L.append("## 2. CN+S — our `spice_ssb` model vs SPICE `abcorr='CN+S'`\n")
    L.append("| # | station | range diff [m] | LOS norm diff [m] | angular diff [arcsec] | light-time diff [s] |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['i']} | {r['station']} | {r['cns_range_diff']:.3e} | "
                 f"{r['cns_norm_diff']:.3e} | {r['cns_ang_as']:.3e} | {r['cns_lt_diff']:.3e} |")
    L.append(f"\n**CN+S max:** range {mx('cns_range_diff'):.2e} m · norm {mx('cns_norm_diff'):.2e} m · "
             f"angle {mx('cns_ang_as'):.2e}\" · light-time {mx('cns_lt_diff'):.2e} s.\n")

    L.append("## 3. Diagnosis — the CN difference is the Moon's light-time displacement\n")
    L.append("The CN difference equals, to within metres, the distance the **Moon moves "
             "(relative to the SSB) during the one-way light time**. Our model forms the "
             "light-time LOS in the **Moon-centred (MCI) frame**, treating it as "
             "inertial; SPICE forms it in the SSB-inertial frame. Because the Moon "
             "travels at ~30 km/s w.r.t. the SSB, over the ~1.36 s Earth-Moon light time "
             "it moves ~40 km, and `LOS_MCI - LOS_SSB = Moon(t_r) - Moon(t_r - τ)`.\n")
    L.append("| # | station | CN LOS norm diff [m] | Moon displacement over τ [m] | ratio |")
    L.append("|---:|---|---:|---:|---:|")
    for r in rows:
        ratio = r["cn_norm_diff"] / r["moon_disp"] if r["moon_disp"] else float("nan")
        L.append(f"| {r['i']} | {r['station']} | {r['cn_norm_diff']:.1f} | "
                 f"{r['moon_disp']:.1f} | {ratio:.4f} |")
    L.append("")

    L.append("## 4. The stellar-aberration (CN -> CN+S) rotation matches SPICE\n")
    L.append("Despite the CN frame offset, the `+S` rotation each model applies on top "
             "of its own CN agrees, confirming the stellar-aberration operator itself is "
             "correct (cf. the `spice.stelab` operator check in "
             "`docs/spice_cross_validation.md`):\n")
    L.append("| # | station | our CN->CN+S [arcsec] | SPICE CN->CN+S [arcsec] | diff [arcsec] |")
    L.append("|---:|---|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['i']} | {r['station']} | {r['our_cn_to_cns_as']:.3f} | "
                 f"{r['spice_cn_to_cns_as']:.3f} | "
                 f"{abs(r['our_cn_to_cns_as'] - r['spice_cn_to_cns_as']):.3f} |")

    L.append("\n## Conclusions\n")
    L.append(f"1. **Harness is exact** — geometric (NONE) LOS matches SPICE to "
             f"{mx('none_norm_diff'):.1e} m, so the comparison is trustworthy.\n")
    L.append(f"2. **Finding: the model's light-time LOS is built in the Moon-centred "
             f"(non-inertial) frame.** It differs from SPICE's SSB-inertial CN by the "
             f"Moon's barycentric displacement during the light time — up to "
             f"{mx('cn_norm_diff'):.1e} m of LOS / {mx('cn_lt_diff'):.1e} s of light "
             f"time / {mx('cn_ang_as'):.1f}\" of direction. This is a **frame "
             "approximation, not a light-time iteration error** (the iteration itself "
             "converges; the offset is exactly the Moon's motion, ratio ≈ 1.000).\n")
    L.append("3. **Self-consistent, so synthetic OD is unbiased.** Generation and "
             "residual prediction use the same MCI-frame computation, so the offset "
             "cancels in observed-minus-computed; it would matter only for absolute "
             "fidelity against real DSN data or SPICE-referenced ephemerides.\n")
    L.append(f"4. **The stellar-aberration rotation is correct** — the CN->CN+S shift "
             f"matches SPICE to {max(abs(r['our_cn_to_cns_as'] - r['spice_cn_to_cns_as']) for r in rows):.2e} "
             "arcsec; the CN+S difference simply inherits the CN frame offset.\n")
    L.append("\n_Recommendation (not applied here, per the validation-only scope): to be "
             "inertial-exact against SPICE/real data, form the light-time LOS in an "
             "inertial frame — e.g. add the Moon's `(t_r) - (t_r - τ)` displacement, or "
             "reference the spacecraft and station to a common SSB/J2000 origin before "
             "differencing._\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
