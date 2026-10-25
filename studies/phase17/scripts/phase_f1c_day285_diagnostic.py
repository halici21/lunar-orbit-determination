"""Phase F1c post-hoc diagnostic (not predeclared, not a gate): is the -11 mm/s Doppler offset of 2009-260 the
same on 2009-285 (PAD 103 and 129)?  M_FULL model as in phase_f1c_lro_doppler.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f1_lro_residuals as F  # noqa: E402
import phase_f1c_lro_doppler as D  # noqa: E402

OUT = HERE.parent / "results" / "phase_f1c_day285_diagnostic.json"


def main():
    from lunar_od import relativistic_light_time as R
    from lunar_od.media import MediaModel
    data = Path(F.REPO.parent / "lro_data")
    rows = D.read_doppler(data, "trk/lro_no_01/2009285", pads=(103, 129))
    setup = F.Setup(data, rows, "DE440")
    t = setup.t3
    out = []
    ivs = D.intervals(rows, t)
    res, pads, tm = [], [], []
    for i, j in ivs:
        pad, ra = rows[i]
        rb = rows[j][1]
        a = (rb.doppler_count_raw - ra.doppler_count_raw) / (rb.utc - ra.utc).total_seconds()
        ft = ra.transmit_frequency_hz
        obs = -F.C * (a - D.BIAS_HZ) / (2.0 * ft * D.TURNAROUND * D.M_MULT)
        media = MediaModel(uplink_hz=ft, downlink_hz=ft * D.TURNAROUND)
        val, _, _ = R.counted_doppler(0.5 * (t[i] + t[j]), t[j] - t[i], setup.tide[pad], setup.sc, setup.eph,
                                      R.DSN_C2_LIGHT_TIME_MODEL, site=setup.sites[pad], media=media,
                                      media_site=setup.media_sites[pad])
        res.append(obs - val); pads.append(pad); tm.append(0.5 * (t[i] + t[j]))
    res, pads, tm = np.array(res), np.array(pads), np.array(tm)
    for pad in (103, 129):
        idx = np.flatnonzero(pads == pad)
        cuts = np.flatnonzero(np.diff(tm[idx]) > 600.0) + 1
        for seg in np.split(idx, cuts):
            if seg.size < 3:
                continue
            out.append(dict(pad=int(pad), n=int(seg.size), mean_mmps=float(res[seg].mean() * 1e3),
                            std_mmps=float(res[seg].std() * 1e3)))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for p in out:
        print(p)


if __name__ == "__main__":
    main()
