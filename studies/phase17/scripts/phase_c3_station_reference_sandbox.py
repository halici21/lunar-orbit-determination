"""Phase C3 reference: station positions and displacements from Orekit (runs only in this sandbox venv).

Predeclared in lunar-orbit-determination studies/phase17/configs/phase_c3_station_frame_predeclaration.json.
Reads the grid exported by phase_c3_station_frame_survey.py --export-grid and writes, per cell and station,
GCRF Earth-relative positions [m] on the grid and at random check epochs:

  A     the production ECEF vector read as ITRF 2020 (IERS 2010, full EOP)
  B     the same vector read as ITRF 93 (Orekit Helmert chain)
  TIDE  Orekit TidalDisplacement (IERS 2010, removePermanentDeformation = False), rotated to GCRF
  POLE  IERS 2010 pole tide (eqs. 7.24-7.26, 2018 linear secular pole), rotated to GCRF

Epochs are TDB seconds past J2000 as et0 + t, built as AbsoluteDate(AbsoluteDate(J2000_TDB, et0), t) so et0 + t
is never rounded.

    ..\\.venv\\Scripts\\python.exe c3_station_reference.py <grid.json> <out.npz>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import orekit_env  # noqa: F401  (sets up orekit-data and the JVM)
from orekit_env import J2000_TDB, TDB
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.frames import FramesFactory, ITRFVersion
from org.orekit.models.earth.displacement import TidalDisplacement
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions

SEED = 20261023
N_CHECK = 200
ARCSEC = math.pi / (180.0 * 3600.0)


def main():
    grid = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    conv = IERSConventions.IERS_2010
    itrf = FramesFactory.getITRF(conv, False)
    itrf93 = FramesFactory.getITRF(ITRFVersion.ITRF_1993, conv, False)
    gcrf = FramesFactory.getGCRF()
    ut1 = TimeScalesFactory.getUT1(conv, False)
    eop = FramesFactory.getEOPHistory(conv, False)
    tide = TidalDisplacement(Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS, Constants.JPL_SSD_SUN_EARTH_PLUS_MOON_MASS_RATIO,
                             Constants.JPL_SSD_EARTH_MOON_MASS_RATIO, CelestialBodyFactory.getSun(),
                             CelestialBodyFactory.getMoon(), conv, False)
    args = conv.getNutationArguments(ut1)
    rng = np.random.default_rng(SEED)

    def v(x):
        return np.array([x.getX(), x.getY(), x.getZ()])

    def pole_ecef(p, date):
        x, y, z = p
        lam = math.atan2(y, x)
        theta = math.atan2(math.hypot(x, y), z)           # geocentric colatitude
        pc = eop.getPoleCorrection(date)
        years = date.durationFrom(J2000_TDB) / (365.25 * 86400.0)
        xbar, ybar = (55.0 + 1.677 * years) * 1e-3, (320.5 + 3.460 * years) * 1e-3     # arcsec
        m1 = pc.getXp() / ARCSEC - xbar
        m2 = -(pc.getYp() / ARCSEC - ybar)
        c = m1 * math.cos(lam) + m2 * math.sin(lam)
        s_r = -33.0 * math.sin(2 * theta) * c * 1e-3
        s_t = -9.0 * math.cos(2 * theta) * c * 1e-3
        s_l = 9.0 * math.cos(theta) * (m1 * math.sin(lam) - m2 * math.cos(lam)) * 1e-3
        rh = np.array([math.sin(theta) * math.cos(lam), math.sin(theta) * math.sin(lam), math.cos(theta)])
        th = np.array([math.cos(theta) * math.cos(lam), math.cos(theta) * math.sin(lam), -math.sin(theta)])
        lh = np.array([-math.sin(lam), math.cos(lam), 0.0])
        return s_r * rh + s_t * th + s_l * lh

    def evaluate(p_vec, et0, ts):
        base = AbsoluteDate(J2000_TDB, float(et0), TDB)
        res = np.zeros((len(ts), 4, 3))
        for i, t in enumerate(ts):
            d = AbsoluteDate(base, float(t), TDB)
            to_g = itrf.getStaticTransformTo(gcrf, d)
            res[i, 0] = v(to_g.transformPosition(p_vec))
            res[i, 1] = v(itrf93.getStaticTransformTo(gcrf, d).transformPosition(p_vec))
            res[i, 2] = v(to_g.transformVector(tide.displacement(args.evaluateAll(d), itrf, p_vec)))
            pe = pole_ecef(v(p_vec), d)
            res[i, 3] = v(to_g.transformVector(Vector3D(float(pe[0]), float(pe[1]), float(pe[2]))))
        return res

    arrays = {}
    for cell, c in grid["cells"].items():
        t = np.arange(c["t_min"] - 600.0, c["t_max"] + 600.0 + 1e-9, 60.0)
        tc = rng.uniform(c["t_min"], c["t_max"], N_CHECK)
        for name, r in grid["stations"].items():
            p = Vector3D(*[float(x) for x in r])
            key = f"{cell}|{name}"
            arrays[key + "|t"] = t
            arrays[key + "|grid"] = evaluate(p, c["et0"], t)
            arrays[key + "|tc"] = tc
            arrays[key + "|check"] = evaluate(p, c["et0"], tc)
        print(cell, flush=True)
    arrays["meta"] = np.asarray(json.dumps(dict(orekit="13.1.9", conventions="IERS_2010", eop_end=str(eop.getEndDate()),
                                                variants=["A", "B", "TIDE", "POLE"], seed=SEED,
                                                grid_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())))
    np.savez_compressed(out, **arrays)
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
