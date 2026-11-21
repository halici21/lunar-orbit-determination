"""F2 WP2.2 gate G10: ERFA (SOFA) refraction constants (runs only in agent-sandbox/c2-erfa-reference).

Predeclared in studies/phase17/configs/wp2_2_predeclaration.json. Writes erfa.refco(phpa, tc, rh, wl) -> (refa,
refb) on a pressure x temperature x humidity x wavelength grid (optical and radio branches).

    .venv/Scripts/python.exe -I wp2_2_erfa_refco_sandbox.py <out.npz>
"""
import hashlib
import json
import sys
from pathlib import Path

import erfa
import numpy as np

PRESSURE_HPA = (0.0, 600.0, 800.0, 900.0, 1013.25, 1050.0)
TEMPERATURE_C = (-40.0, -10.0, 0.0, 10.0, 25.0, 40.0)
HUMIDITY = (0.0, 0.2, 0.5, 0.8, 1.0)
WAVELENGTH_UM = (0.4, 0.55, 1.0, 1.0e4, 1.0e6)     # optical (<= 100 um) and radio


def main():
    out = Path(sys.argv[1])
    rows = []
    for p in PRESSURE_HPA:
        for t in TEMPERATURE_C:
            for rh in HUMIDITY:
                for wl in WAVELENGTH_UM:
                    a, b = erfa.refco(p, t, rh, wl)
                    rows.append([p, t, rh, wl, float(a), float(b)])
    rows = np.array(rows)
    np.savez_compressed(out, refco=rows, meta=np.asarray(json.dumps(
        dict(pyerfa=erfa.__version__, columns="phpa,tc,rh,wl_um,refa,refb"))))
    print(json.dumps(dict(output=str(out), n=int(rows.shape[0]), sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
