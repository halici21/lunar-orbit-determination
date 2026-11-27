"""WP7.1 diagnostic: the five commit-tier values that differ between the local machine and the CI runner.

Prints, as JSON, the exact quantities the three tests assert on, plus the platform facts that can change them
(CPU, numpy SIMD dispatch, BLAS, Numba target).  Analysis only: it imports the test modules' own helpers, so
it measures exactly what the tests measure, and it changes no gate.

    python studies/phase17/scripts/wp7_1_platform_gate_values.py [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402


def platform_facts() -> dict:
    facts = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
             "processor": platform.processor(), "numpy": np.__version__}
    try:
        from numpy._core._multiarray_umath import __cpu_features__, __cpu_baseline__, __cpu_dispatch__
        facts["numpy_cpu_features_on"] = sorted(k for k, v in __cpu_features__.items() if v)
        facts["numpy_cpu_baseline"] = list(__cpu_baseline__)
        facts["numpy_cpu_dispatch"] = list(__cpu_dispatch__)
    except Exception as exc:  # pragma: no cover
        facts["numpy_cpu_features_error"] = repr(exc)
    try:
        import scipy
        facts["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import numba
        from numba.core import config as nconfig
        import llvmlite.binding as llvm
        facts["numba"] = numba.__version__
        facts["numba_disable_jit"] = bool(nconfig.DISABLE_JIT)
        facts["llvm_host_cpu"] = llvm.get_host_cpu_name()
        facts["numba_cpu_name_env"] = os.environ.get("NUMBA_CPU_NAME")
    except Exception as exc:
        facts["numba_error"] = repr(exc)
    for key in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_CPU_NAME", "NUMBA_DISABLE_JIT"):
        if os.environ.get(key):
            facts.setdefault("env", {})[key] = os.environ[key]
    return facts


def tide_values() -> dict:
    import tests.test_wp3_2_lunar_tide as T
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        cols = T.make_rk4_columns()
        hist = T.dyn.propagate_augmented_state(T.T6H, T._aug0(T.X0), *T.BASE, j2_moon=T.J2_MOON_UNNORMALIZED,
                                               lunar_tide=T.TIDE, **T.TIGHT)
        gated = T._stm_err(cols, T._phi(hist))
        control = T._stm_err(cols, T._phi(T._no_tide_gradient_replica()))
    return {"T_G5a_gated": gated, "T_G5a_limit": 1e-7, "T_G5a_control": control,
            "T_G5a_control_ratio": control / gated}


def lrp_values() -> dict:
    import tests.test_wp3_2_lunar_radiation as L
    out = {}
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        for harmonics in (False, True):
            for tide in (False, True):
                pos, phi, effect = L.path_differences(harmonics, tide)
                out[f"L_G6[{harmonics}-{tide}]"] = {"pos_max_m": max(pos.values()), "pos_limit_m": 1e-5,
                                                    "phi_max": max(phi.values()), "effect_m": effect}
    return out


def dmc_values() -> dict:
    import tests.test_dmc as D
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        x0 = D._state9()
        t = [0.0, 3600.0]
        force = dict(j2_moon=D.J2_MOON, **D.TOL)
        phi = D.propagate_dmc_state(t, x0, D.OPTIONS, D.MU_MOON, D.MU_EARTH, 0.0, D._earth, D._sun,
                                    **force)[-1, 9:].reshape(9, 9, order="F")
        steps = [1.0] * 3 + [1e-3] * 3 + [1e-6] * 3
        rel = []
        for j, h in enumerate(steps):
            plus, minus = x0.copy(), x0.copy()
            plus[j] += h
            minus[j] -= h
            f = [D.propagate_dmc_state(t, x, D.OPTIONS, D.MU_MOON, D.MU_EARTH, 0.0, D._earth, D._sun,
                                       stm=False, **force)[-1] for x in (plus, minus)]
            column = (f[0] - f[1]) / (2.0 * h)
            rel.append(float(np.linalg.norm(column - phi[:, j]) / np.linalg.norm(phi[:, j])))
    return {"dmc_fd_rel_err_per_column": rel, "dmc_limit": 1e-6, "dmc_worst_column": int(np.argmax(rel))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    record = {"facts": platform_facts(), "tide": tide_values(), "lrp": lrp_values(), "dmc": dmc_values()}
    text = json.dumps(record, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
