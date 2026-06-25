"""Phase 0 - Existing Force Model Verification (blocks F, G, H: ephemeris layer).

Measurement only: compares the MATLAB PlanetEphemeris DE421 dataset
(python_port/ephemeris_data.mat) against SPICE de421.bsp. Modifies nothing.

This is a SEPARATE ephemeris-source validation layer; any residual is reported as
an ephemeris-source difference, not a force-model error.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import spiceypy as spice  # noqa: E402
from lunar_od import load_spice_kernels  # noqa: E402

MAT = Path(__file__).resolve().parents[1] / "ephemeris_data.mat"
J2000_JD = 2451545.0


def _load_mat():
    f = h5py.File(str(MAT), "r")
    first_jd = float(np.array(f["first_jd"]).ravel()[0])
    dt = float(np.array(f["dt_sec"]).ravel()[0])
    t_vec = np.array(f["t_vec"]).ravel()
    return f, first_jd, dt, t_vec


def _spice_pos(body, et):
    p, _ = spice.spkpos(body, et, "J2000", "NONE", "MOON")
    return np.asarray(p) * 1000.0          # km -> m


def _spice_state(body, et):
    s, _ = spice.spkezr(body, et, "J2000", "NONE", "MOON")
    return np.asarray(s[:3]) * 1000.0, np.asarray(s[3:6]) * 1000.0


def main():
    load_spice_kernels()
    f, first_jd, dt, t_vec = _load_mat()
    print(f"first_jd = {first_jd:.9f}   dt_sec = {dt}   n = {t_vec.size}")

    # ---- G: epoch alignment + time-scale investigation ----------------------
    et_tdb0 = (first_jd - J2000_JD) * 86400.0          # interpret first_jd as TDB JD
    et_utc0 = spice.str2et(f"JD {first_jd:.9f}")        # interpret first_jd as UTC JD (str2et default)
    deltet = spice.deltet(et_tdb0, "ET")
    print("\n=== G. Epoch alignment / time-scale ===")
    print(f"  ET(JD as TDB) = {et_tdb0:.6f}   ET(JD as UTC) = {et_utc0:.6f}")
    print(f"  fark (UTC-TDB yorumu) = {et_utc0 - et_tdb0:.3f} s   (~ET-UTC = {deltet:.3f} s)")
    print(f"  t_vec adimi = {np.mean(np.diff(t_vec[:1000])):.6g} s (= dt_sec? {np.isclose(np.mean(np.diff(t_vec[:50])), dt)})")
    print(f"  utc epoch UTC string: {spice.et2utc(et_utc0, 'C', 3)}")

    # ---- F + H: compare over a 1-day window for BOTH time-scale interpretations
    idx = np.arange(0, 8640 + 1, 360)                  # every hour over day 1 (25 pts)
    rE = np.array(f["rEarth_data"][:, idx]).T          # (k,3) meters
    rS = np.array(f["rSun_data"][:, idx]).T
    vE = np.array(f["vEarth_data"][:, idx]).T
    tsec = t_vec[idx]

    print("\n=== F. PlanetEphemeris(.mat) vs SPICE de421 — 1-gun penceresi (her iki zaman yorumu) ===")
    results = {}
    for label, et0 in (("JD=TDB", et_tdb0), ("JD=UTC", et_utc0)):
        dE = dS = dvE = 0.0; dE_list = []
        for k, ts in enumerate(tsec):
            et = et0 + float(ts)
            spE = _spice_pos("EARTH", et); spS = _spice_pos("SUN", et)
            _, spvE = _spice_state("EARTH", et)
            de = np.linalg.norm(rE[k] - spE); ds = np.linalg.norm(rS[k] - spS); dv = np.linalg.norm(vE[k] - spvE)
            dE = max(dE, de); dS = max(dS, ds); dvE = max(dvE, dv); dE_list.append(de)
        results[label] = (dE, dS, dvE)
        secular = dE_list[-1] - dE_list[0]
        print(f"  [{label}] max Δ: Earth_pos={dE:.3e} m  Sun_pos={dS:.3e} m  Earth_vel={dvE:.3e} m/s  | sekuler(Earth) {secular:+.2e} m")

    best = min(results, key=lambda k: results[k][0])
    dE, dS, dvE = results[best]
    print(f"\n  -> En kucuk fark: {best}")
    print(f"     Earth_pos max = {dE:.3e} m | Sun_pos max = {dS:.3e} m | Earth_vel max = {dvE:.3e} m/s")
    print(f"     ILK ESIK (<1 km, sekuler yok, isaret/birim/epoch hatasi yok): "
          f"{'PASS' if dE < 1000 and dS < 1000 else 'FAIL'}")
    print(f"     NIHAI HEDEF (metre-alti): {'MET' if dE < 1.0 else 'NOT-YET (' + f'{dE:.2e} m' + ')'}")

    # ---- H: unit / frame / target-center direction (at epoch 0, best time scale)
    et0 = et_tdb0 if best == "JD=TDB" else et_utc0
    spE0 = _spice_pos("EARTH", et0)
    matE0 = rE[0]
    cos = float(np.dot(matE0, spE0) / (np.linalg.norm(matE0) * np.linalg.norm(spE0)))
    print("\n=== H. Unit / frame / target-center direction (epoch 0) ===")
    print(f"  |Earth|_mat  = {np.linalg.norm(matE0):.6e} m  (= {np.linalg.norm(matE0)/1000:.1f} km)")
    print(f"  |Earth|_spice= {np.linalg.norm(spE0):.6e} m")
    print(f"  yon dot(u_mat,u_spice) = {cos:.9f}  (>1-1e-6 -> Moon->Earth, ICRF≈J2000) "
          f"-> {'PASS' if cos > 1 - 1e-6 else 'FAIL'}")
    print(f"  buyukluk orani = {np.linalg.norm(matE0)/np.linalg.norm(spE0):.9f} (≈1 -> metre, isaret dogru)")

    f.close()
    spice.kclear()


if __name__ == "__main__":
    main()
