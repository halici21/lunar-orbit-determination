"""Phase C2 independent time-scale reference: ERFA (SOFA) through pyerfa.

Predeclared in lunar-orbit-determination studies/phase17/configs/phase_c2_time_scale_predeclaration.json.
Reads the survey epochs exported by phase_c2_time_scale_survey.py --export-epochs and writes, per cell,
station and epoch (TDB seconds past et0):

  geo        ERFA dtdb geocentric (u = v = 0), TDB - TT [s]
  topo       ERFA dtdb at the station minus geo [s]
  geo_rate   d(geo)/dt, central difference h = 60 s on the two-part date
  topo_rate  d(topo)/dt, same
  tai_utc    TAI - UTC [s] from ERFA's own leap-second table (erfa.taiutc)
  tdb_utc    TDB - UTC [s]

Two-part Julian dates: date1 = 2451545.0 + whole days of et0, date2 = remaining fraction + t/86400, so the
reference carries no et0 + t quantisation. dtdb's ut argument is UTC (fraction of the day from midnight).

    .venv/Scripts/python.exe -I erfa_reference.py <epochs.json> <reference.json>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import erfa
import numpy as np

DAY = 86400.0
J2000 = 2451545.0
H_RATE_S = 60.0
TT_TAI_S = 32.184


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def two_part(et0: float, t: np.ndarray):
    days = math.floor(et0 / DAY)
    frac = (et0 - days * DAY) / DAY
    return J2000 + days, frac + np.asarray(t, float) / DAY


def site(r_ecef_m):
    x, y, z = (float(v) for v in r_ecef_m)
    return math.atan2(y, x), math.hypot(x, y) / 1e3, z / 1e3


def utc_of_tdb(d1, d2):
    """TDB two-part date -> (tt2, utc1, utc2, ut_fraction); TT = TDB - dtdb, one fixed-point pass is exact
    to < 1e-15 s because d(dtdb)/dt ~ 3e-10."""
    g = erfa.dtdb(d1, d2, 0.0, 0.0, 0.0, 0.0)
    tt2 = d2 - g / DAY
    tai2 = tt2 - TT_TAI_S / DAY
    utc1, utc2 = erfa.taiutc(d1, tai2)
    ut = np.mod(np.mod(utc1, 1.0) - 0.5 + utc2, 1.0)
    return tt2, tai2, utc1, utc2, ut


def dtdb_at(d1, d2, elong, u, v):
    *_, ut = utc_of_tdb(d1, d2)
    return erfa.dtdb(d1, d2, ut, elong, u, v)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ep = json.loads(src.read_text(encoding="utf-8"))
    out = dict(meta=dict(pyerfa=erfa.__version__, erfa=erfa.version.erfa_version,
                         sofa=erfa.version.sofa_version, h_rate_s=H_RATE_S,
                         input=str(src.name), input_sha256=sha256(src),
                         convention="TDB seconds past et0; values in seconds"), cells={})
    for cell, c in ep["cells"].items():
        et0 = float(c["et0"])
        rec = dict(et0=et0, stations={})
        for name, ts in c["stations"].items():
            t = np.asarray(ts, float)
            elong, u, v = site(ep["station_positions_itrf93_m"][name])
            d1, d2 = two_part(et0, t)
            h = H_RATE_S / DAY
            geo = erfa.dtdb(d1, d2, utc_of_tdb(d1, d2)[4], 0.0, 0.0, 0.0)
            top = dtdb_at(d1, d2, elong, u, v)
            geo_p = erfa.dtdb(d1, d2 + h, utc_of_tdb(d1, d2 + h)[4], 0.0, 0.0, 0.0)
            geo_m = erfa.dtdb(d1, d2 - h, utc_of_tdb(d1, d2 - h)[4], 0.0, 0.0, 0.0)
            top_p, top_m = dtdb_at(d1, d2 + h, elong, u, v), dtdb_at(d1, d2 - h, elong, u, v)
            tt2, tai2, utc1, utc2, _ = utc_of_tdb(d1, d2)
            tai_utc = ((d1 - utc1) + (tai2 - utc2)) * DAY
            tdb_utc = ((d1 - utc1) + (d2 - utc2)) * DAY
            rec["stations"][name] = dict(
                t=t.tolist(), geo=geo.tolist(), topo=(top - geo).tolist(),
                geo_rate=((geo_p - geo_m) / (2 * H_RATE_S)).tolist(),
                topo_rate=(((top_p - geo_p) - (top_m - geo_m)) / (2 * H_RATE_S)).tolist(),
                tai_utc=tai_utc.tolist(), tdb_utc=tdb_utc.tolist(),
                site=dict(elong_rad=elong, u_km=u, v_km=v))
        out["cells"][cell] = rec
    dst.write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(json.dumps(dict(input_sha256=out["meta"]["input_sha256"], output=str(dst), output_sha256=sha256(dst),
                          pyerfa=erfa.__version__)))


if __name__ == "__main__":
    main()
