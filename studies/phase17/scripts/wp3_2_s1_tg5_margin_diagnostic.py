"""F3 WP3.2 S1, EXPLORATORY (after the T_G5 result): where the T_G5(a) error of 7.4e-8 comes from.

T_G5(a) passed at 7.37e-8 against 1e-7, while the predeclaration expected 1e-9..1e-8. This varies the
oracle only: the RK4 step dt and the central-difference steps h (1 m and 1e-3 m/s times a scale). The
production STM is computed once, exactly as in the test. If the error falls about as 1/h when h grows, it
is floating-point noise in the oracle's differences; if it falls with dt, it is RK4 truncation; if neither
moves it, it is the production STM. Changes no gate, no test and no threshold.

usage: python wp3_2_s1_tg5_margin_diagnostic.py OUT.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lunar_od.dynamics as dyn  # noqa: E402
from lunar_od.constants import J2_MOON_UNNORMALIZED  # noqa: E402
from tests import test_wp3_2_lunar_tide as M  # noqa: E402

CASES = ((5.0, 1.0), (2.5, 1.0), (5.0, 10.0), (2.5, 10.0), (5.0, 0.1))   # (dt in s, scale of h)


def oracle(dt, scale):
    rhs = M._production_rhs6()
    cols = []
    for j in range(6):
        h = (1.0 if j < 3 else 1e-3) * scale
        e = np.zeros(6)
        e[j] = h
        cols.append((M._rk4(rhs, M.X0 + e, M.T6H, dt) - M._rk4(rhs, M.X0 - e, M.T6H, dt)) / (2 * h))
    return np.stack(cols, axis=-1)


def per_column(fd, phi):
    return [float(np.max(np.abs(fd[:, :, j] - phi[:, :, j])) / np.max(np.abs(phi[:, :, j]))) for j in range(6)]


def main(path):
    hist = dyn.propagate_augmented_state(M.T6H, M._aug0(M.X0), *M.BASE, j2_moon=J2_MOON_UNNORMALIZED,
                                         lunar_tide=M.TIDE, **M.TIGHT)
    phi = M._phi(hist)
    fds, rows = {}, []
    for dt, scale in CASES:
        t0 = time.perf_counter()
        fds[(dt, scale)] = fd = oracle(dt, scale)
        cols = per_column(fd, phi)
        rows.append(dict(dt_s=dt, h_scale=scale, columns=cols, worst=max(cols)))
        print(dt, scale, f"{max(cols):.3e}", [f"{c:.2e}" for c in cols], f"({time.perf_counter() - t0:.1f} s)",
              flush=True)
    # oracle against oracle: same h, two steps (truncation), and same step, two h (difference noise)
    between = dict(
        dt5_vs_dt2p5_at_h_scale_10=per_column(fds[(5.0, 10.0)], fds[(2.5, 10.0)]),
        h_scale_1_vs_10_at_dt5=per_column(fds[(5.0, 1.0)], fds[(5.0, 10.0)]),
    )
    out = dict(
        status="EXPLORATORY: after the T_G5 result; no gate, test or threshold changes",
        predeclaration="studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json (bdafb53)",
        gate="T_G5(a): max over columns <= 1e-7 (computed 7.37e-8 at dt 5 s, h scale 1)",
        python=platform.python_version(), numpy=np.__version__, machine=platform.machine(),
        production_stm="propagate_augmented_state, DOP853 rtol 1e-13 atol 1e-14, as in the test",
        cases=rows, oracle_vs_oracle=between,
    )
    Path(path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
