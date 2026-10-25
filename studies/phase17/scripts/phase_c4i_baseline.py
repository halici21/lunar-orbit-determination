"""Phase C4-I G1 baseline: Moon-centred Earth from the pre-C4-I loader (run before any edit)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "phase_c4i_baseline.json"
EPOCHS_UTC = [f"{y}-{m:02d}-15T{h:02d}:00:00" for y in (2025, 2026, 2027) for m in (1, 4, 7, 10) for h in (0, 12)][:24] + ["2026-10-23T00:00:00"]


def earth_from_moon(epochs):
    import spiceypy as spice
    return [list(map(float.hex, map(float, spice.spkezr("EARTH", spice.str2et(e), "J2000", "NONE", "MOON")[0])))
            for e in epochs]


if __name__ == "__main__":
    from lunar_od.spice_loader import REQUIRED_KERNELS, load_spice_kernels
    load_spice_kernels()
    OUT.write_text(json.dumps(dict(required_kernels=list(REQUIRED_KERNELS), epochs_utc=EPOCHS_UTC,
                                   earth_from_moon_hex=earth_from_moon(EPOCHS_UTC)), indent=1), encoding="utf-8")
    print(len(EPOCHS_UTC), REQUIRED_KERNELS)
