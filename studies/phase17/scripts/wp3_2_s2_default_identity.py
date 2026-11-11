"""F3 WP3.2 S2, gate L_G1(b): existing paths are byte-identical before and after the LRP change.

Predeclaration: studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json (da5a4a0).
Reuses the S1 T_G1(b) configuration set (wp3_2_s1_default_identity.configurations) twice: as in S1, and with
the S1 tide on in every call (prefix ``tide_``), 50 configurations. ``run`` writes the SHA-256 of every output
array's bytes with ``lunar_radiation`` omitted (mode ``omit``) or passed as ``None`` (mode ``none``).
``compare`` is the S1 comparison.

usage: python wp3_2_s2_default_identity.py run {omit|none} OUT.json
       python wp3_2_s2_default_identity.py compare A.json B.json [...] [--out RESULT.json]
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import wp3_2_s1_default_identity as S1  # noqa: E402
from lunar_od.lunar_tide import LunarTideOptions  # noqa: E402

TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")


def configurations(lrp_kw):
    cfg = dict(S1.configurations(dict(lrp_kw)))
    cfg.update({f"tide_{name}": call for name, call in S1.configurations({"lunar_tide": TIDE, **lrp_kw}).items()})
    return cfg


def run(mode, out_path):
    if mode not in ("omit", "none"):
        raise SystemExit("mode must be omit or none")
    lrp_kw = {} if mode == "omit" else {"lunar_radiation": None}
    results = {}
    for name, call in configurations(lrp_kw).items():
        try:
            results[name] = S1._record(call())
        except Exception as exc:  # noqa: BLE001 - recorded, the comparison fails on it
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{name:34s} {results[name].get('digest', results[name].get('error'))[:72]}", flush=True)
    import numba
    import scipy
    payload = {
        "gate": "L_G1(b)", "mode": mode, "commit": S1._git("rev-parse", "HEAD"),
        "src_status": S1._git("status", "--porcelain", "--", "src"),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "numba": numba.__version__, "fast_dynamics": bool(S1.dyn._FAST_DYNAMICS), "configs": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        args = sys.argv[2:]
        out = None
        if "--out" in args:
            k = args.index("--out")
            out = args[k + 1]
            args = args[:k] + args[k + 2:]
        result = S1.compare(args)
        result["gate"] = "L_G1(b)"
        if out:
            Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise SystemExit(__doc__)
