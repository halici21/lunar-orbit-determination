"""F2 WP2.7 G12 diagnosis: how robust is the iteration bound of the SRIF fixed-point test?

tests/test_srif_acceptance_wp4_1.py::test_srif_solution_is_a_fixed_point restarts the SRIF at
its own answer x_est and requires at most 3 iterations.  This script restarts it from x_est
moved by 0 to 3 ulp per component (15 fixed cases) and records the iteration counts.

Run on the current tree, and on the tree with src/lunar_od/accelerated.py taken from the base
commit (f754ab2) through PYTHONPATH:

    python studies/phase17/scripts/wp2_7_g12_fixed_point_sweep.py <label> <out.json> [kind]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lunar_od  # noqa: E402
from tests import test_srif_acceptance_wp4_1 as T  # noqa: E402

TEST_BOUND = 3


def _cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20261107)
    cases = [("none", np.zeros(6, dtype=int))]
    for i in range(3):
        for s in (+1, -1):
            k = np.zeros(6, dtype=int)
            k[i] = s
            cases.append((f"x{i}{'+' if s > 0 else '-'}1ulp", k))
    cases += [(f"rand{j}", rng.integers(-3, 4, size=6)) for j in range(8)]
    return cases


def _moved(x: np.ndarray, ulps: np.ndarray) -> np.ndarray:
    x = x.copy()
    for c, k in enumerate(ulps):
        for _ in range(abs(int(k))):
            x[c] = np.nextafter(x[c], np.inf if k > 0 else -np.inf)
    return x


def main(label: str, out: str, kind: str = "position") -> None:
    x_est, reason0, st0 = T._solve(kind, T.X_TRUE + T.OFFSET, rtol=1e-12, atol=1e-13)
    rows = []
    for name, ulps in _cases():
        x_again, reason, st = T._solve(kind, _moved(x_est, ulps), rtol=1e-12, atol=1e-13)
        rows.append({"case": name, "ulps": ulps.tolist(), "reason": reason,
                     "iterations": int(st.iterations),
                     "restart_position_shift_m": float(np.linalg.norm(x_again[:3] - x_est[:3]))})
        print(rows[-1], flush=True)
    its = [r["iterations"] for r in rows]
    result = {
        "label": label, "kind": kind, "lunar_od": lunar_od.__file__,
        "first_solve": {"reason": reason0, "iterations": int(st0.iterations),
                        "position_error_vs_truth_m": float(np.linalg.norm(x_est[:3] - T.X_TRUE[:3]))},
        "restarts": rows,
        "iterations": its,
        "above_test_bound": sum(i > TEST_BOUND for i in its),
        "max_iterations": max(its),
        "max_restart_position_shift_m": max(r["restart_position_shift_m"] for r in rows),
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{label}: iterations {its}; {result['above_test_bound']} of {len(its)} above {TEST_BOUND}")


if __name__ == "__main__":
    main(*sys.argv[1:])
