"""WP7.1 reference platform for SIMD-sensitive gate replays (docs/assurance/test-levels.md).

A few predeclared gates sit within floating-point / integration-floor distance of their
thresholds, and their measured value depends on the CPU instruction set the numerics run
on (measured: studies/phase17/results/wp7_1_platform_gate_values/).  Their recorded
verdicts were produced where numpy dispatches AVX-512 and Numba compiles for an AVX-512
host.  Tests marked ``reference_platform`` (rows in tests/levels.py) replay those verdicts
and are meaningful only there; elsewhere they skip, and under ``--reference-platform``
(tools/run_tier.py reference) they FAIL instead, so the gate is never silently unchecked.
"""
from __future__ import annotations

import os

MARKER = "reference_platform"

#: numpy SIMD features the recorded verdicts used (numpy._core._multiarray_umath.__cpu_features__).
NUMPY_FEATURES: tuple[str, ...] = ("AVX512F", "AVX512_SKX")
#: Environment variables that change the Numba code-generation target away from the host.
NUMBA_TARGET_OVERRIDES: tuple[str, ...] = ("NUMBA_CPU_NAME", "NUMBA_CPU_FEATURES", "NUMBA_DISABLE_JIT")


def reference_platform_problem() -> str | None:
    """None on the reference platform; otherwise why this machine is not it."""
    problems = []
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:  # pragma: no cover - numpy < 2
        from numpy.core._multiarray_umath import __cpu_features__ as features
    missing = [name for name in NUMPY_FEATURES if not features.get(name)]
    if missing:
        problems.append("numpy does not dispatch " + ", ".join(missing))
    overrides = [name for name in NUMBA_TARGET_OVERRIDES if os.environ.get(name)]
    if overrides:
        problems.append("Numba target overridden by " + ", ".join(overrides))
    try:
        import llvmlite.binding as llvm

        host_features = llvm.get_host_cpu_features()
        if not host_features.get("avx512f", False):
            problems.append(f"Numba host CPU {llvm.get_host_cpu_name()!r} has no avx512f")
    except Exception as exc:  # llvmlite missing or API moved: fail closed
        problems.append(f"Numba host target unknown ({exc!r})")
    return "; ".join(problems) or None
