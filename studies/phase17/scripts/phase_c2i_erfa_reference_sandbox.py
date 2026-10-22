"""Phase C2-I gates G2/G3: pyerfa reference values (runs only in agent-sandbox/c2-erfa-reference).

Predeclared in lunar-orbit-determination studies/phase17/configs/phase_c2i_station_time_implementation_
predeclaration.json (G2, G3). Writes one .npz with the exact inputs and pyerfa outputs:

  g2_*   eraDtdb inputs (d1, d2, ut, elong, u, v) and value; set 0 = C2 survey epochs geocentric,
         1 = C2 survey epochs at their stations, 2 = 2,000 epochs 1950-2100 x 10 sites (3 DSN + 7 random)
  g3a_*  (y, m, d, fd) and erfa.dat: +-1 s around every leap second 1972-2017 and 1,000 random epochs 1972-2050
  g3b_*  1,000 random UTC tags 1980-2050 x 3 DSN sites: (y, m, d, sod), site, and the chain
         dtf2d -> utctai -> taitt -> dtdb(tt, ut = sod / 86400, site): tt1, tt2, dtdb
  g3c_*  the C2 survey epochs turned into UTC tags (y, m, d, sod) by ERFA, then the same forward chain

    .venv/Scripts/python.exe -I c2i_reference.py <phase_c2_survey_epochs.json> <out.npz>
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
SEED = 20261022


def two_part(et0, t):
    days = math.floor(et0 / DAY)
    return J2000 + days, (et0 - days * DAY) / DAY + np.asarray(t, float) / DAY


def site(r):
    x, y, z = (float(c) for c in r)
    return math.atan2(y, x), math.hypot(x, y) / 1e3, z / 1e3


def ut_of_tdb(d1, d2):
    g = erfa.dtdb(d1, d2, 0.0, 0.0, 0.0, 0.0)
    tai2 = d2 - g / DAY - 32.184 / DAY
    utc1, utc2 = erfa.taiutc(d1, tai2)
    return np.mod(np.mod(utc1, 1.0) - 0.5 + utc2, 1.0), utc1, utc2


def forward(y, m, d, sod, s):
    h = np.floor(sod / 3600.0).astype(int)
    mi = np.floor((sod - h * 3600.0) / 60.0).astype(int)
    sec = sod - h * 3600.0 - mi * 60.0
    u1, u2 = erfa.dtf2d("UTC", y, m, d, h, mi, sec)
    a1, a2 = erfa.utctai(u1, u2)
    t1, t2 = erfa.taitt(a1, a2)
    return t1, t2, erfa.dtdb(t1, t2, sod / DAY, s[0], s[1], s[2])


def main():
    ep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    rng = np.random.default_rng(SEED)
    pos = ep["station_positions_itrf93_m"]
    dsn = {n: site(r) for n, r in pos.items()}
    names = sorted(dsn)
    r = {}

    # ---------------- G2 ----------------
    cols = {k: [] for k in ("d1", "d2", "ut", "elong", "u", "v", "set")}

    def add(d1, d2, ut, s, tag):
        n = np.size(d2)
        for k, v in zip(("d1", "d2", "ut", "elong", "u", "v", "set"),
                        (np.full(n, d1) if np.ndim(d1) == 0 else d1, d2, ut,
                         np.full(n, s[0]), np.full(n, s[1]), np.full(n, s[2]), np.full(n, tag))):
            cols[k].append(np.asarray(v, float))
    c2_et0, c2_t, c2_st = [], [], []
    for cell, c in ep["cells"].items():
        for name, ts in c["stations"].items():
            t = np.asarray(ts, float)
            d1, d2 = two_part(float(c["et0"]), t)
            ut, _, _ = ut_of_tdb(d1, d2)
            add(d1, d2, ut, (0.0, 0.0, 0.0), 0)
            add(d1, d2, ut, dsn[name], 1)
            c2_et0.append(np.full(t.size, float(c["et0"])))
            c2_t.append(t)
            c2_st.append(np.full(t.size, names.index(name)))
    sites = [dsn[n] for n in names] + [(rng.uniform(-math.pi, math.pi), rng.uniform(0, 6400), rng.uniform(-6400, 6400))
                                       for _ in range(7)]
    jd = rng.uniform(2433282.5, 2488069.5, 2000)
    d1g = np.floor(jd) + 0.5
    d2g = jd - d1g
    utg = rng.uniform(0, 1, 2000)
    for s in sites:
        add(d1g, d2g, utg, s, 2)
    g2 = {k: np.concatenate(v) for k, v in cols.items()}
    g2["val"] = erfa.dtdb(g2["d1"], g2["d2"], g2["ut"], g2["elong"], g2["u"], g2["v"])
    r.update({"g2_" + k: v for k, v in g2.items()})
    r["g2_sites"] = np.asarray(sites, float)

    # ---------------- G3a: leap seconds ----------------
    ya, ma, da, fa = [], [], [], []
    prev = None
    for year in range(1972, 2019):
        for month in range(1, 13):
            v = erfa.dat(year, month, 1, 0.0)
            if prev is not None and v != prev:
                py, pm = (year, month - 1) if month > 1 else (year - 1, 12)
                last = int(erfa.cal2jd(year, month, 1)[1] - erfa.cal2jd(py, pm, 1)[1])
                ya += [py, year]
                ma += [pm, month]
                da += [last, 1]
                fa += [86399.0 / DAY, 1.0 / DAY]
            prev = v
    jr = rng.uniform(2441317.5, 2469807.5, 1000)
    yr, mr, dr, fr = erfa.jd2cal(np.floor(jr - 0.5) + 0.5, jr - (np.floor(jr - 0.5) + 0.5))
    y3 = np.concatenate([np.asarray(ya), yr]).astype(int)
    m3 = np.concatenate([np.asarray(ma), mr]).astype(int)
    d3 = np.concatenate([np.asarray(da), dr]).astype(int)
    f3 = np.concatenate([np.asarray(fa), fr])
    r.update(g3a_y=y3, g3a_m=m3, g3a_d=d3, g3a_fd=f3, g3a_dat=erfa.dat(y3, m3, d3, f3),
             g3a_n_boundary=np.int64(len(ya)))

    # ---------------- G3b: random tags x 3 DSN sites ----------------
    jt = rng.uniform(2444239.5, 2469807.5, 1000)
    yb, mb, db, _ = erfa.jd2cal(np.floor(jt - 0.5) + 0.5, 0.0)
    sod = rng.uniform(0.0, DAY, 1000)
    B = {k: [] for k in ("y", "m", "d", "sod", "site", "tt1", "tt2", "dtdb")}
    for i, n in enumerate(names):
        t1, t2, w = forward(yb, mb, db, sod, dsn[n])
        for k, v in zip(B, (yb, mb, db, sod, np.full(sod.size, i), t1, t2, w)):
            B[k].append(np.asarray(v))
    r.update({"g3b_" + k: np.concatenate(v) for k, v in B.items()})

    # ---------------- G3c: C2 survey epochs as UTC tags ----------------
    et0c, tc, stc = np.concatenate(c2_et0), np.concatenate(c2_t), np.concatenate(c2_st).astype(int)
    d1c, d2c = zip(*(two_part(e, t) for e, t in zip(et0c, tc)))
    d1c, d2c = np.asarray(d1c), np.asarray(d2c)
    _, u1, u2 = ut_of_tdb(d1c, d2c)
    yc, mc, dc, fc = erfa.jd2cal(u1, u2)
    sodc = fc * DAY
    s_arr = np.asarray([dsn[names[i]] for i in stc], float).T
    t1c, t2c, wc = forward(yc, mc, dc, sodc, (s_arr[0], s_arr[1], s_arr[2]))
    r.update(g3c_et0=et0c, g3c_t=tc, g3c_site=stc, g3c_y=yc, g3c_m=mc, g3c_d=dc, g3c_sod=sodc,
             g3c_tt1=t1c, g3c_tt2=t2c, g3c_dtdb=wc)
    r["site_names"] = np.asarray(names)
    r["meta"] = np.asarray(json.dumps(dict(pyerfa=erfa.__version__, erfa=erfa.version.erfa_version,
                                           sofa=erfa.version.sofa_version, seed=SEED)))
    np.savez_compressed(out, **r)
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                          n_g2=int(g2["val"].size), n_g3a=int(y3.size), n_g3b=int(sod.size * 3),
                          n_g3c=int(tc.size), leap_boundaries=len(ya) // 2)))


if __name__ == "__main__":
    main()
