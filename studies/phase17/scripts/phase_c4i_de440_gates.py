"""Phase C4-I gates G1-G4 (studies/phase17/configs/phase_c4i_de440_predeclaration.json)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE.parent / "results" / "phase_c4i_baseline.json"
OUT = HERE.parent / "results" / "phase_c4i_de440.json"


def _earth(epochs):
    import spiceypy as spice
    return np.array([spice.spkezr("EARTH", spice.str2et(e), "J2000", "NONE", "MOON")[0] for e in epochs], float)


def _loaded_bsp():
    import spiceypy as spice
    return sorted(Path(spice.kdata(i, "SPK")[0]).name for i in range(spice.ktotal("SPK")))


def child(mode: str) -> dict:
    """Run in a fresh process so the environment variable is read at import."""
    from lunar_od.spice_loader import REQUIRED_KERNELS, load_spice_kernels, planetary_ephemeris
    base = json.loads(BASE.read_text(encoding="utf-8"))
    load_spice_kernels(ephemeris="DE421" if mode == "explicit_de421" else None)
    e = _earth(base["epochs_utc"])
    out = dict(mode=mode, selected=planetary_ephemeris("DE421" if mode == "explicit_de421" else None),
               loaded_spk=_loaded_bsp(), module_required_kernels=list(REQUIRED_KERNELS),
               earth_hex=[[float.hex(float(v)) for v in row] for row in e])
    if mode == "explicit_de421":
        from lunar_od.spice_loader import required_kernels
        out["kernel_list_equals_baseline"] = list(required_kernels("DE421")) == base["required_kernels"]
        try:
            planetary_ephemeris("DE999")
            out["unknown_raises"] = False
        except ValueError:
            out["unknown_raises"] = True
    return out


def main():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    runs = {}
    for mode, env in (("explicit_de421", None), ("default", None), ("env_de421", "DE421")):
        e = dict(os.environ)
        e.pop("LUNAR_OD_PLANETARY_EPHEMERIS", None)
        if env:
            e["LUNAR_OD_PLANETARY_EPHEMERIS"] = env
        r = subprocess.run([sys.executable, __file__, "--child", mode], env=e, capture_output=True, text=True,
                           check=True)
        runs[mode] = json.loads(r.stdout.strip().splitlines()[-1])
    b = np.array([[float.fromhex(v) for v in row] for row in base["earth_from_moon_hex"]])
    arr = {m: np.array([[float.fromhex(v) for v in row] for row in r["earth_hex"]]) for m, r in runs.items()}
    d440 = np.linalg.norm((arr["default"] - b)[:, :3], axis=1) * 1e3
    g1 = runs["explicit_de421"]["kernel_list_equals_baseline"] and bool(np.array_equal(arr["explicit_de421"], b))
    g2 = runs["default"]["selected"] == "DE440" and "de440.bsp" in runs["default"]["loaded_spk"] \
        and "de421.bsp" not in runs["default"]["loaded_spk"] and float(d440.max()) <= 5.0
    g3 = bool(np.array_equal(arr["env_de421"], b)) and runs["env_de421"]["selected"] == "DE421" \
        and runs["explicit_de421"]["unknown_raises"]
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                        "tests/test_de440_kernel_contract.py"], cwd=REPO, capture_output=True, text=True)
    g4 = t.returncode == 0
    out = dict(predeclaration="studies/phase17/configs/phase_c4i_de440_predeclaration.json",
               de440_minus_de421_earth_m=dict(max=float(d440.max()), min=float(d440.min())),
               loaded_spk={m: r["loaded_spk"] for m, r in runs.items()},
               G4_output=t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-500:],
               gates=dict(G1=g1, G2=g2, G3=g3, G4=g4))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2])))
    else:
        main()
