"""F3 WP3.2 S4, gate LT_G1(b): existing paths are byte-identical before and after the SRP light-time change.

Predeclaration: studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json (cc7a454).
Runs the 79 S3 E_G1(b) configurations (wp3_2_s3_default_identity.configurations and e2_configurations, the
Earth shadow omitted) plus the four real-E2 configurations with the S3 Earth shadow on (prefix ``es_e2_``):
83 configurations. ``run`` writes the SHA-256 of every output with ``srp_light_time`` omitted (mode ``omit``)
or passed as ``None`` (mode ``none``). Exceptions are outcomes, hashed as in S3; only S3's two accepted
exceptions are accepted.

usage: python wp3_2_s4_default_identity.py run {omit|none} OUT.json
       python wp3_2_s4_default_identity.py compare A.json B.json [...] [--out RESULT.json]
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
import wp3_2_s3_default_identity as S3  # noqa: E402
from lunar_od.earth_shadow import EarthShadowOptions  # noqa: E402
from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402


def run(mode, out_path):
    if mode not in ("omit", "none"):
        raise SystemExit("mode must be omit or none")
    kw = {} if mode == "omit" else {"srp_light_time": None}
    results = {}
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        calls = S3.configurations(kw)
    calls.update(S3.e2_configurations(kw))
    calls.update({f"es_{n}": c for n, c in S3.e2_configurations({"earth_shadow": EarthShadowOptions(), **kw}).items()})
    for name, call in calls.items():
        if name.startswith(("e2_", "es_e2_")):
            results[name] = S3._outcome(name, call)
        else:
            with use_lunar_j2_model("iau_mean_pole_legacy"):
                results[name] = S3._outcome(name, call)
        print(f"{name:36s} {results[name].get('digest', results[name].get('error'))[:72]}", flush=True)
    import numba
    import scipy
    payload = {
        "gate": "LT_G1(b)", "mode": mode, "commit": S1._git("rev-parse", "HEAD"),
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
        result["gate"] = "LT_G1(b)"
        if out:
            Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise SystemExit(__doc__)
