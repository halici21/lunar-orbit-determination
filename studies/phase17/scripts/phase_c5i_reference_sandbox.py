"""Phase C5-I gates G2/G3: Orekit reference values (runs only in agent-sandbox/orekit-gmat-oracle/harness).

Predeclared in lunar-orbit-determination studies/phase17/configs/phase_c5i_media_predeclaration.json.

  g2_*  Orekit NiellMappingFunctionModel(UTC).mappingFactors on elevation x latitude x height x 12 dates, with the
        day of year Orekit itself uses (AbsoluteDate.getDayOfYear(UTC))
  g3_*  Orekit SingleLayerModelMappingFunction(h).mappingFactor at h = 350 km and 450 km

    ..\\.venv\\Scripts\\python.exe c5i_reference.py <out.npz>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import orekit_env  # noqa: F401
from org.orekit.bodies import GeodeticPoint
from org.orekit.models.earth.ionosphere import SingleLayerModelMappingFunction
from org.orekit.models.earth.troposphere import NiellMappingFunctionModel
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import TrackingCoordinates

ELEV = np.array([5, 7, 10, 15, 20, 30, 45, 60, 75, 89.9])
LAT = np.array([-80, -60, -45, -30, -15, -5, 0, 5, 15, 30, 45, 60, 75, 80])
HGT = np.array([0.0, 500.0, 1000.0, 2000.0, 3000.0])


def main():
    out = Path(sys.argv[1])
    utc = TimeScalesFactory.getUTC()
    niell = NiellMappingFunctionModel(utc)
    rows = []
    for month in range(1, 13):
        date = AbsoluteDate(2027, month, 1, 6, 0, 0.0, utc)
        doy = float(date.getDayOfYear(utc))
        for el in ELEV:
            for lat in LAT:
                for h in HGT:
                    m = niell.mappingFactors(TrackingCoordinates(0.0, math.radians(el), 1.0e6),
                                             GeodeticPoint(math.radians(lat), 0.3, float(h)), date)
                    rows.append([math.radians(el), math.radians(lat), h, doy, m[0], m[1]])
    rows = np.array(rows)
    el3 = np.radians(np.arange(5.0, 90.01, 0.5))
    g3 = {}
    for hh in (350e3, 450e3):
        f = SingleLayerModelMappingFunction(hh)
        g3[f"g3_{int(hh)}"] = np.array([float(f.mappingFactor(float(e))) for e in el3])
    np.savez_compressed(out, g2=rows, g3_el=el3, **g3,
                        meta=np.asarray(json.dumps(dict(orekit="13.1.9", columns="el,lat,h,doy,mh,mw"))))
    print(json.dumps(dict(output=str(out), n_g2=int(rows.shape[0]), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
