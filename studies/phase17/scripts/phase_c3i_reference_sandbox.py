"""Phase C3-I gates G2/G3: Orekit reference values (runs only in agent-sandbox/orekit-gmat-oracle/harness).

Predeclared in lunar-orbit-determination studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json.

  g2_*  Orekit ITRF 2020 -> ITRF 93 offset (X93 - X2020) [m] at 3 DSN + 7 random sites, 2000.0 ... 2040.0 yearly
        (epoch t = Jan 1 12:00 TT of each year, given as TDB seconds past J2000)
  g3_*  Orekit TidalDisplacement (IERS 2010, removePermanentDeformation = False) in ITRF [m] at every C3 grid
        point (cell, station, t)

    ..\\.venv\\Scripts\\python.exe c3i_reference.py <phase_c3_station_grid.json> <out.npz>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import orekit_env  # noqa: F401
from orekit_env import J2000_TDB, TDB
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.frames import FramesFactory, ITRFVersion
from org.orekit.models.earth.displacement import TidalDisplacement
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions

SEED = 20261023


def main():
    grid = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    conv = IERSConventions.IERS_2010
    itrf = FramesFactory.getITRF(conv, False)
    itrf93 = FramesFactory.getITRF(ITRFVersion.ITRF_1993, conv, False)
    # amendment A1: the explicitly versioned ITRF 2020 frame (getITRF(conv) follows the EOP file's version label)
    itrf2020 = FramesFactory.getITRF(ITRFVersion.ITRF_2020, conv, False)
    tt = TimeScalesFactory.getTT()
    ut1 = TimeScalesFactory.getUT1(conv, False)
    tide = TidalDisplacement(Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS, Constants.JPL_SSD_SUN_EARTH_PLUS_MOON_MASS_RATIO,
                             Constants.JPL_SSD_EARTH_MOON_MASS_RATIO, CelestialBodyFactory.getSun(),
                             CelestialBodyFactory.getMoon(), conv, False)
    args = conv.getNutationArguments(ut1)
    rng = np.random.default_rng(SEED)
    names = sorted(grid["stations"])
    sites = [np.asarray(grid["stations"][n], float) for n in names]
    for _ in range(7):
        lat, lon = math.asin(rng.uniform(-1, 1)), rng.uniform(-math.pi, math.pi)
        rr = 6.371e6 + rng.uniform(-200.0, 3000.0)
        sites.append(np.array([rr * math.cos(lat) * math.cos(lon), rr * math.cos(lat) * math.sin(lon), rr * math.sin(lat)]))
    sites = np.asarray(sites)
    years = np.arange(2000, 2041)
    et_years = np.array([AbsoluteDate(int(y), 1, 1, 12, 0, 0.0, tt).durationFrom(J2000_TDB) for y in years])
    g2 = np.zeros((sites.shape[0], years.size, 3))
    for i, p in enumerate(sites):
        pv = Vector3D(*[float(x) for x in p])
        for j, y in enumerate(years):
            d = AbsoluteDate(int(y), 1, 1, 12, 0, 0.0, tt)
            q = itrf2020.getStaticTransformTo(itrf93, d).transformPosition(pv)
            g2[i, j] = [q.getX() - p[0], q.getY() - p[1], q.getZ() - p[2]]
    arrays = dict(g2_sites=sites, g2_years=years.astype(float), g2_et=et_years, g2_offset=g2,
                  site_names=np.asarray(names))
    for cell, c in grid["cells"].items():
        t = np.arange(c["t_min"] - 600.0, c["t_max"] + 600.0 + 1e-9, 60.0)
        base = AbsoluteDate(J2000_TDB, float(c["et0"]), TDB)
        for k, n in enumerate(names):
            pv = Vector3D(*[float(x) for x in sites[k]])
            disp = np.zeros((t.size, 3))
            for i, ti in enumerate(t):
                d = AbsoluteDate(base, float(ti), TDB)
                v = tide.displacement(args.evaluateAll(d), itrf, pv)
                disp[i] = [v.getX(), v.getY(), v.getZ()]
            arrays[f"g3|{cell}|{n}|t"] = t
            arrays[f"g3|{cell}|{n}|disp"] = disp
        print(cell, flush=True)
    arrays["meta"] = np.asarray(json.dumps(dict(orekit="13.1.9", conventions="IERS_2010", seed=SEED,
                                                grid_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())))
    np.savez_compressed(out, **arrays)
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
