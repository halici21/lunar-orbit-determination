"""Phase F1c: real LRO two-way S-band Doppler against the LOLA precision orbit (no estimation).

Predeclaration: studies/phase17/configs/phase_f1c_lro_doppler_predeclaration.json.  Reuses F1's stations,
kernels and spacecraft history (phase_f1_lro_residuals.Setup).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f1_lro_residuals as F  # noqa: E402

OUT = HERE.parent / "results" / "phase_f1c_lro_doppler.json"
TURNAROUND = 240.0 / 221.0
M_MULT = 1000.0
BIAS_HZ = 2.4e8
MAX_GAP_S = 60.0


def read_doppler(data: Path, day_dir: str = F.DAY_DIR, pads=(103, 105, 126)) -> list:
    from lunar_od.utdf import file_pad_id, read_utdf
    from lro_data_partition import require_readable_day
    require_readable_day(day_dir)
    rows = []
    for f in sorted((data / day_dir).glob("*.trk")):
        pad = file_pad_id(f)
        if pad not in pads:
            continue
        rows.extend((pad, r) for r in read_utdf(f) if r.doppler_valid)
    rows.sort(key=lambda x: (x[0], x[1].utc))
    return rows


def intervals(rows, t):
    out = []
    for i in range(len(rows) - 1):
        (pa, ra), (pb, rb) = rows[i], rows[i + 1]
        dt = t[i + 1] - t[i]
        if pa == pb and 0.0 < dt <= MAX_GAP_S:
            out.append((i, i + 1))
    return out


def main():
    from lunar_od import relativistic_light_time as R
    from lunar_od.media import MediaModel
    t0 = time.time()
    data = Path(F.REPO.parent / "lro_data")
    rows = read_doppler(data)
    setup = F.Setup(data, rows, "DE440")
    t = setup.t3
    ivs = intervals(rows, t)
    obs, comp = {}, {}
    r1_bad, ft_set = [], {}
    for v in ("M_FULL", "NO_MEDIA"):
        obs_v, comp_v, fails = [], [], 0
        for i, j in ivs:
            pad, ra = rows[i]
            rb = rows[j][1]
            dn = rb.doppler_count_raw - ra.doppler_count_raw
            dt_utc = (rb.utc - ra.utc).total_seconds()
            ft = ra.transmit_frequency_hz
            if v == "M_FULL":
                if dn <= 0:
                    r1_bad.append(("dN<=0", pad, str(ra.utc)))
                ft_set.setdefault(pad, set()).add(ft)
            a = dn / dt_utc
            obs_v.append(-F.C * (a - BIAS_HZ) / (2.0 * ft * TURNAROUND * M_MULT))
            media = None if v == "NO_MEDIA" else MediaModel(uplink_hz=ft, downlink_hz=ft * TURNAROUND)
            st = setup.tide[pad]
            try:
                val, _, _ = R.counted_doppler(0.5 * (t[i] + t[j]), t[j] - t[i], st, setup.sc, setup.eph,
                                              R.DSN_C2_LIGHT_TIME_MODEL, site=setup.sites[pad], media=media,
                                              media_site=setup.media_sites[pad] if media else None)
            except Exception:  # noqa: BLE001 - counted for R3
                fails += 1
                val = np.nan
            comp_v.append(val)
        obs[v], comp[v] = np.array(obs_v), np.array(comp_v)
        comp[v + "_fails"] = fails
    # passes: split one station's intervals at gaps > 10 min
    stats = {}
    for v in ("M_FULL", "NO_MEDIA"):
        res = obs[v] - comp[v]
        out = []
        pads = np.array([rows[i][0] for i, _ in ivs])
        tm = np.array([0.5 * (t[i] + t[j]) for i, j in ivs])
        for pad in (103, 105, 126):
            idx = np.flatnonzero(pads == pad)
            if idx.size == 0:
                continue
            cuts = np.flatnonzero(np.diff(tm[idx]) > 600.0) + 1
            for seg in np.split(idx, cuts):
                r = res[seg]
                tt = (tm[seg] - tm[seg].mean()) / 3600.0
                coef = np.polyfit(tt, r, 1)
                out.append(dict(pad=int(pad), n=int(seg.size), t_start_s=float(tm[seg].min()),
                                mean_mmps=float(r.mean() * 1e3), rms_mmps=float(np.sqrt(np.mean(r ** 2)) * 1e3),
                                std_mmps=float(r.std() * 1e3), slope_mmps_per_h=float(coef[0] * 1e3),
                                obs_rms_mps=float(np.sqrt(np.mean(obs[v][seg] ** 2)))))
        stats[v] = out
    full = stats["M_FULL"]
    gates = dict(R1=not r1_bad and all(len(s) == 1 for s in ft_set.values()),
                 R2=all(p["rms_mmps"] * 1e-3 < 0.01 * p["obs_rms_mps"] for p in full),
                 R3=comp["M_FULL_fails"] == 0 and comp["NO_MEDIA_fails"] == 0)
    hyp = dict(H1=all(p["rms_mmps"] <= 5.0 for p in full), H2=all(abs(p["mean_mmps"]) <= 5.0 for p in full))
    out = dict(predeclaration="studies/phase17/configs/phase_f1c_lro_doppler_predeclaration.json",
               n_intervals=len(ivs), transmit_frequencies_hz={str(k): sorted(v) for k, v in ft_set.items()},
               R1_bad=r1_bad[:20], gates=gates, hypotheses=hyp, passes=stats, elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(gates=gates, hyp=hyp, n=len(ivs), ft=out["transmit_frequencies_hz"],
                          passes={v: [(p["pad"], p["n"], round(p["mean_mmps"], 3), round(p["rms_mmps"], 3),
                                       round(p["std_mmps"], 3), round(p["slope_mmps_per_h"], 2)) for p in s]
                                  for v, s in stats.items()}), indent=1))


if __name__ == "__main__":
    main()
