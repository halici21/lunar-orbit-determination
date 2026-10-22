"""Phase C3 POST-HOC diagnostic (informational): SPICE's ITRF93 Earth PCKs vs Orekit's ITRF93 chain with IERS EOP.

Written after the C3 survey's S2 screen flagged T2B = 112 m at the 2027 survey epochs. Evaluates the Goldstone
ECEF vector of the production catalogue through (a) the production PCK earth_2025_250826_2125_predict.bpc and
(b) earth_latest_high_prec.bpc where it has coverage, against Orekit ITRF 93 (IERS 2010, finals2000A EOP), from
2025 to the survey epoch. Runs in agent-sandbox/orekit-gmat-oracle/harness (copy:
studies/phase17/scripts/phase_c3_eop_divergence_sandbox.py).

    ..\\.venv\\Scripts\\python.exe c3_eop_divergence.py <kernel_dir> <out.json>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import spiceypy as sp

import orekit_env  # noqa: F401
from orekit_env import J2000_TDB, TDB
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory, ITRFVersion
from org.orekit.time import AbsoluteDate
from org.orekit.utils import IERSConventions

DATES = ("2025-03-01", "2025-06-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-12-01", "2026-03-01",
         "2026-06-01", "2026-10-01", "2027-01-01", "2027-03-08", "2027-03-20")
GOLDSTONE = np.array([-2353621.0, -4641341.0, 3677052.0])     # production catalogue, ~m


def main():
    kdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    for k in ("naif0012.tls.txt", "earth_assoc_itrf93.tf.txt"):
        sp.furnsh(str(kdir / k))
    conv = IERSConventions.IERS_2010
    itrf93 = FramesFactory.getITRF(ITRFVersion.ITRF_1993, conv, False)
    gcrf = FramesFactory.getGCRF()
    eop = FramesFactory.getEOPHistory(conv, False)
    rho = math.hypot(GOLDSTONE[0], GOLDSTONE[1])
    omega = 7.292115e-5
    rows = []
    for pck in ("earth_2025_250826_2125_predict.bpc", "earth_latest_high_prec.bpc"):
        sp.furnsh(str(kdir / pck))
        for utc in DATES:
            et = sp.str2et(utc + "T00:00:00")
            d = AbsoluteDate(J2000_TDB, float(et), TDB)
            o = itrf93.getStaticTransformTo(gcrf, d).transformPosition(Vector3D(*GOLDSTONE))
            o = np.array([o.getX(), o.getY(), o.getZ()])
            try:
                s = np.asarray(sp.pxform("ITRF93", "J2000", et)) @ GOLDSTONE
            except Exception:  # noqa: BLE001 - SPICE raises when the PCK has no coverage
                rows.append(dict(pck=pck, utc=utc, covered=False))
                continue
            diff = float(np.linalg.norm(o - s))
            rows.append(dict(pck=pck, utc=utc, covered=True, diff_m=diff, equivalent_ut1_s=diff / (omega * rho),
                             orekit_ut1_minus_utc_s=float(eop.getUT1MinusUTC(d))))
        sp.unload(str(kdir / pck))
    out.write_text(json.dumps(dict(post_hoc=True, station="Goldstone (production catalogue)", rows=rows), indent=1),
                   encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
