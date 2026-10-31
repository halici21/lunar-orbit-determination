"""Phase F1: real LRO two-way range residuals against the LOLA team's precision orbit (no estimation).

Predeclaration: studies/phase17/configs/phase_f1_lro_residuals_predeclaration.json.  The archive data live
outside the repository (``--data``, default ``<workspace>/lro_data``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "results" / "phase_f1_lro_residuals.json"
DAY_DIR = "trk/lro_no_01/2009260"
SPK = "spk/lro_no_01_grgm900c_l600.bsp"
LRO = "-85"
C = 299792458.0
UPLINK_HZ = 2271.125e6 * 221.0 / 240.0
DOWNLINK_HZ = 2271.125e6
PASS_GAP_S = 600.0
GRID_S = 10.0
#: Station states free of ulp(et0) epoch quantisation (2026-10-27 G4e root cause); False = historical records.
EXACT_EPOCH_STATIONS = True

#: lrors_arch_sis Appendix A, WGS-84 geodetic (deg, min, sec), height [m].
STATIONS = {
    103: ("USN Dongara USPS", (-29, 2, 44.7798), (115, 20, 55.2395), 250.47),
    105: ("USN South Point Hawaii USHS", (19, 0, 50.0562), (204, 20, 12.1155), 385.19),
    126: ("USN Kiruna KU1S", (67, 53, 22.4100), (21, 3, 56.3571), 400.4),
    129: ("USN Weilheim WU2S", (47, 52, 52.3160), (11, 5, 1.0280), 663.37),   # Phase F1b
}


def _deg(dms):
    d, m, s = dms
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def _ecef(pad):
    from lunar_od.geometry import geodetic_to_ecef_wgs84
    _, lat, lon, h = STATIONS[pad]
    x, y, z = geodetic_to_ecef_wgs84(_deg(lat), _deg(lon), h)
    return np.array([float(x), float(y), float(z)]), math.radians(_deg(lat)), math.radians(_deg(lon)), h


def load_kernels(data: Path, ephemeris: str) -> None:
    import spiceypy as spice
    from lunar_od.spice_loader import load_spice_kernels, resolve_kernel_dir
    load_spice_kernels(ephemeris=ephemeris)
    spice.furnsh(str(Path(resolve_kernel_dir()) / "earth_latest_high_prec.bpc"))
    spice.furnsh(str(data / SPK))


def read_day(data: Path, day_dir: str = DAY_DIR) -> tuple[list, dict]:
    from lunar_od.utdf import file_pad_id, read_utdf
    from lro_data_partition import require_readable_day
    require_readable_day(day_dir)
    rows, r1 = [], dict(files=0, records=0, bad=[])
    for f in sorted((data / day_dir).glob("*.trk")):
        r1["files"] += 1
        try:
            recs = read_utdf(f)
        except ValueError as exc:
            r1["bad"].append(f"{f.name}: {exc}")
            continue
        pad = file_pad_id(f)
        for r in recs:
            r1["records"] += 1
            if r.sic != 59 or r.vid != 1 or r.pad_rcv != pad:
                r1["bad"].append(f"{f.name}: sic {r.sic} vid {r.vid} pad {r.pad_rcv}")
            if pad in STATIONS and r.range_valid:
                rows.append((pad, r))
    r1["PASS"] = not r1["bad"]
    return rows, r1


class Setup:
    """Kernels-dependent context: epochs, spacecraft history, station providers."""

    def __init__(self, data: Path, rows, ephemeris: str, grid_s: float = GRID_S):
        import spiceypy as spice
        from lunar_od import relativistic_light_time as R
        from lunar_od import station_time as ST
        load_kernels(data, ephemeris)
        utc = [r.utc for _, r in rows]
        day0 = min(utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.et0 = float(spice.str2et(day0.strftime("%Y-%m-%dT%H:%M:%S")))
        self.sites, self.media_sites, self.r_ecef, self.base, self.tide = {}, {}, {}, {}, {}
        from lunar_od.media import MediaSite
        for pad in STATIONS:
            r, lat, lon, h = _ecef(pad)
            self.r_ecef[pad] = r
            self.sites[pad] = ST.Site.from_itrf_m(r)
            self.media_sites[pad] = MediaSite(lat, lon, h)
            self.base[pad] = self._station(r)
            self.tide[pad] = R.displaced_station_state(self.base[pad], self.et0, r, solid_tide=True)
        tags = [u.strftime("%Y-%m-%dT%H:%M:%S.%f") for u in utc]
        self.t3 = np.array([ST.utc_to_tdb_seconds([tag], self.et0, self.sites[pad])[0]
                            for tag, (pad, _) in zip(tags, rows)])
        t_lo, t_hi = float(self.t3.min()) - 900.0, float(self.t3.max()) + 300.0
        self.grid = np.arange(t_lo, t_hi + grid_s, grid_s)
        hist = np.array([spice.spkezr(LRO, self.et0 + t, "J2000", "NONE", "MOON")[0] for t in self.grid]) * 1e3
        self.sc = R.SpacecraftHistory(self.grid, hist)
        self.eph = R.BarycentricEphemeris(self.et0)
        cover = spice.spkcov(str(data / SPK), int(LRO))
        lo, hi = spice.wnfetd(cover, 0)[0], spice.wnfetd(cover, spice.wncard(cover) - 1)[1]
        self.coverage_ok = bool(self.et0 + t_lo >= lo and self.et0 + t_hi <= hi)

    def _station(self, r):
        """Moon-centred J2000 station state at ``et0 + t``, free of epoch quantisation.

        ``et0 + t`` rounds to ulp(et0) ~ 6e-8 s (2009). SPICE is read at the rounded epoch and the state is moved
        by its own velocity x the exact rounding error (Fast2Sum, |et0| >= |t|), as in
        :func:`lunar_od.relativistic_light_time.exact_epoch_station_state`. Without it the station steps by up to
        ~1e-4 m whenever a light-time event crosses an ulp(et) boundary (G4e: a 1.7e-6 m/s counted-Doppler jump).
        ``EXACT_EPOCH_STATIONS = False`` restores the quantised provider used by the historical F1/F2 records.
        """
        import spiceypy as spice
        et0 = self.et0
        exact = EXACT_EPOCH_STATIONS

        def state(t_s: float) -> np.ndarray:
            t = float(t_s)
            et = et0 + t
            x = np.asarray(spice.sxform("ITRF93", "J2000", et), dtype=float)
            earth = np.asarray(spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")[0], dtype=float) * 1e3
            s = x @ np.concatenate([r, np.zeros(3)]) + earth
            if exact:
                rounding = t - (et - et0)
                if rounding:
                    s[:3] += s[3:6] * rounding
            return s
        return state


def compute(setup: Setup, rows, variant: str) -> tuple[np.ndarray, np.ndarray, int]:
    from lunar_od import relativistic_light_time as R
    from lunar_od.media import MediaModel
    media = None if variant in ("NO_MEDIA", "MOON_NEWTONIAN") else MediaModel(uplink_hz=UPLINK_HZ,
                                                                             downlink_hz=DOWNLINK_HZ)
    model = R.LightTimeModel() if variant == "MOON_NEWTONIAN" else R.DSN_C2_LIGHT_TIME_MODEL
    tide = variant not in ("NO_TIDE", "MOON_NEWTONIAN")
    comp, elev, failures = np.full(len(rows), np.nan), np.full(len(rows), np.nan), 0
    for i, ((pad, rec), t3) in enumerate(zip(rows, setup.t3)):
        st = setup.tide[pad] if tide else setup.base[pad]
        try:
            sol = R.solve_two_way(float(t3), st, setup.sc, setup.eph, model, site=setup.sites[pad],
                                  media=media, media_site=setup.media_sites[pad] if media else None)
        except Exception:  # noqa: BLE001 - counted for gate R4
            failures += 1
            continue
        if sol.two_cycle:
            failures += 1
        comp[i] = R.two_way_range_m(sol)
        los = setup.sc.state(sol.downlink.t_tx_s)[:3] - st(float(t3))[:3]
        import spiceypy as spice
        up = np.asarray(spice.pxform("ITRF93", "J2000", setup.et0 + float(t3))) @ setup.media_sites[pad].up_ecef
        elev[i] = math.degrees(math.asin(float(np.dot(up, los) / np.linalg.norm(los))))
    return comp, elev, failures


def passes(rows, t3) -> list[np.ndarray]:
    out = []
    for pad in STATIONS:
        idx = np.array([i for i, (p, _) in enumerate(rows) if p == pad], dtype=int)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(t3[idx])]
        cuts = np.flatnonzero(np.diff(t3[idx]) > PASS_GAP_S) + 1
        out.extend(np.split(idx, cuts))
    return out


def pass_stats(rows, t3, res, elev) -> list[dict]:
    out = []
    for idx in passes(rows, t3):
        ok = idx[np.isfinite(res[idx])]
        if ok.size < 3:
            continue
        r = res[ok]
        tt = t3[ok] - t3[ok].mean()
        coef = np.polyfit(tt, r, 1)
        out.append(dict(pad=rows[ok[0]][0], station=STATIONS[rows[ok[0]][0]][0], n=int(ok.size),
                        t_start_s=float(t3[ok].min()), duration_s=float(np.ptp(t3[ok])),
                        elev_min_deg=float(np.nanmin(elev[ok])), elev_max_deg=float(np.nanmax(elev[ok])),
                        mean_m=float(r.mean()), debiased_rms_m=float(np.std(r)),
                        detrended_rms_m=float(np.std(r - np.polyval(coef, tt))), slope_m_per_h=float(coef[0] * 3600)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "lro_data"))
    a = ap.parse_args()
    data = Path(a.data)
    t0 = time.time()
    rows, r1 = read_day(data)
    setup = Setup(data, rows, "DE440")
    observed = np.array([C * r.rtlt_s / 2.0 for _, r in rows])
    results, fails = {}, {}
    for v in ("M_FULL", "NO_MEDIA", "NO_TIDE", "MOON_NEWTONIAN"):
        comp, elev, nf = compute(setup, rows, v)
        results[v] = (observed - comp, elev)
        fails[v] = nf
    # R3: 5 s grid, every 10th record, M_FULL
    sub = list(range(0, len(rows), 10))
    setup5 = Setup(data, rows, "DE440", grid_s=5.0)
    rows_sub = [rows[i] for i in sub]
    comp10 = observed[sub] - results["M_FULL"][0][sub]
    setup5.t3 = setup5.t3[sub]
    comp5, _, _ = compute(setup5, rows_sub, "M_FULL")
    r3 = float(np.nanmax(np.abs(comp10 - comp5)))
    # DE421 variant
    setup421 = Setup(data, rows, "DE421")
    comp421, elev421, nf421 = compute(setup421, rows, "M_FULL")
    results["DE421"] = (observed - comp421, elev421)
    fails["DE421"] = nf421
    stats = {v: pass_stats(rows, setup.t3, res, el) for v, (res, el) in results.items()}
    gates = dict(R1=r1["PASS"], R2=setup.coverage_ok, R3=r3 <= 1e-3, R4=all(n == 0 for n in fails.values()))
    full = stats["M_FULL"]
    hyp = dict(H1_all_passes_debiased_rms_le_3m=all(p["debiased_rms_m"] <= 3.0 for p in full))
    out = dict(predeclaration="studies/phase17/configs/phase_f1_lro_residuals_predeclaration.json",
               n_range_records=len(rows), records_per_station={str(p): sum(1 for q, _ in rows if q == p)
                                                                for p in STATIONS},
               R1=r1, R3_grid_10s_vs_5s_max_m=r3, R4_failures=fails, gates=gates, hypotheses=hyp,
               passes=stats, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    brief = {v: [(p["pad"], p["n"], round(p["elev_min_deg"], 1), round(p["mean_m"], 2), round(p["debiased_rms_m"], 3),
                  round(p["detrended_rms_m"], 3)) for p in s] for v, s in stats.items()}
    print(json.dumps(dict(gates=gates, hyp=hyp, n=len(rows), R3=r3, fails=fails, R1_bad=r1["bad"][:5], passes=brief),
                     indent=1))


if __name__ == "__main__":
    main()
