"""F3 WP3.2 S3, gate E_G1(b): existing paths are byte-identical before and after the Earth-shadow change.

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json (a5954f1).
Runs the S1 T_G1(b) configuration set (wp3_2_s1_default_identity.configurations) three times: as in S1, with
the S1 tide on in every call (prefix ``tide_``) and with the S2 lunar radiation pressure on in every call
(prefix ``lrp_``), under the legacy mean-pole lunar J2 (the model S1 and S2 were evaluated on); plus four real-E2
configurations on the default grail_pa_de440 J2 (prefix ``e2_``): 79 configurations. ``run`` writes the SHA-256
of every output with ``earth_shadow`` omitted (mode ``omit``) or passed as ``None`` (mode ``none``).

An exception is an outcome: its type and message are hashed and compared like any output. Only the two
predeclared exceptions are accepted (S1's fail-closed GM check in the four tide-on range-rate solves, S2's
"lunar_radiation needs an active SRPOptions" in the LRP-on configurations without active SRP); any other
exception is recorded as an error and fails the comparison.

usage: python wp3_2_s3_default_identity.py run {omit|none} OUT.json
       python wp3_2_s3_default_identity.py compare A.json B.json [...] [--out RESULT.json]
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import wp3_2_s1_default_identity as S1  # noqa: E402
import wp3_2_s3_geometry as GEO  # noqa: E402
from lunar_od.constants import J2_MOON_GRAIL_UNNORMALIZED  # noqa: E402
from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402
from lunar_od.lunar_radiation import LunarRadiationOptions  # noqa: E402
from lunar_od.lunar_tide import LunarTideOptions  # noqa: E402
from lunar_od.srp import SRPOptions  # noqa: E402

TIDE = LunarTideOptions(static_field_convention="TIDE_FREE")
LRP = LunarRadiationOptions()
ACCEPTED_EXCEPTIONS = (
    ("tide_est_range_rate_", "needs its GM > 0"),
    ("lrp_", "lunar_radiation needs an active SRPOptions"),
)


def _outcome(name, call):
    try:
        value = call()
    except Exception as exc:  # noqa: BLE001 - an outcome; unexpected ones fail the comparison
        text = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ValueError) and any(name.startswith(p) and m in str(exc) for p, m in ACCEPTED_EXCEPTIONS):
            return {"digest": hashlib.sha256(text.encode("utf-8")).hexdigest(), "expected_error": text}
        return {"error": text}
    return S1._record(value)


def e2_configurations(es_kw):
    import lunar_od.dynamics as dyn
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels()
    et0, t, eph, mus, x0 = GEO.e2_case()
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    base = (*mus, eph.earth_position, eph.sun_position)
    kw = dict(srp=SRPOptions(k_srp_m2_per_kg=0.01), method="DOP853", shadow_events=True, rtol=1e-11, atol=1e-12,
              j2_moon=J2_MOON_GRAIL_UNNORMALIZED, lunar_j2_model="grail_pa_de440", **es_kw)
    pre = {k: v for k, v in kw.items() if k not in ("method", "shadow_events")}
    return {
        "e2_ps_grail_events": lambda: dyn.propagate_state(t, x0, *base, **kw),
        "e2_pa_grail_events": lambda: dyn.propagate_augmented_state(t, aug0, *base, **kw),
        "e2_k48_grail_events": lambda: dyn.propagate_state_with_k_sensitivity(t, x0, *base, **kw),
        "e2_contacts_grail": lambda: np.array(dyn.shadow_contact_times(t, x0, *base, **pre)),
    }


def configurations(es_kw):
    cfg = dict(S1.configurations(dict(es_kw)))
    cfg.update({f"tide_{n}": c for n, c in S1.configurations({"lunar_tide": TIDE, **es_kw}).items()})
    cfg.update({f"lrp_{n}": c for n, c in S1.configurations({"lunar_radiation": LRP, **es_kw}).items()})
    return cfg


def run(mode, out_path):
    if mode not in ("omit", "none"):
        raise SystemExit("mode must be omit or none")
    es_kw = {} if mode == "omit" else {"earth_shadow": None}
    results = {}
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        calls = configurations(es_kw)
    calls.update(e2_configurations(es_kw))
    for name, call in calls.items():
        if name.startswith("e2_"):
            results[name] = _outcome(name, call)
        else:
            with use_lunar_j2_model("iau_mean_pole_legacy"):
                results[name] = _outcome(name, call)
        print(f"{name:36s} {results[name].get('digest', results[name].get('error'))[:72]}", flush=True)
    import numba
    import scipy
    payload = {
        "gate": "E_G1(b)", "mode": mode, "commit": S1._git("rev-parse", "HEAD"),
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
        result["gate"] = "E_G1(b)"
        if out:
            Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise SystemExit(__doc__)
