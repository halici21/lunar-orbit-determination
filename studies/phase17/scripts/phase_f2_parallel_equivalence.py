"""Parallel evaluation check (behaviour preservation): the pooled residual and Jacobian must equal the serial ones
bit for bit on the same problem (F2d's one-day range + Doppler arc, N = 200, at the F2d solution).

Run: python studies/phase17/scripts/phase_f2_parallel_equivalence.py [workers]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as F2  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2_parallel_equivalence.json"


def main(workers: int = 8):
    d = json.loads((HERE.parent / "results" / "phase_f2d_lro_od_range_doppler.json").read_text(encoding="utf-8"))
    r = d["runs"]["N200_-"]
    p = np.concatenate([r["x0"], [r["k_srp"]], [r["biases_m"][k] for k in ("103", "105", "126")],
                        [r["doppler_bias_mmps"] * 1e-3]])
    pb = F2.Problem(200, doppler=True)
    t = time.time()
    res_s, h_s, _ = pb.residual(p)
    t_serial = time.time() - t
    pb.attach_pool(workers)
    pb.residual(p, partials=False)            # warm-up: workers build their Problem
    t = time.time()
    res_p, h_p, _ = pb.residual(p)
    t_parallel = time.time() - t
    pb.close_pool()
    out = dict(workers=workers, rows=int(res_s.size), residual_identical=bool(np.array_equal(res_s, res_p)),
               jacobian_identical=bool(np.array_equal(h_s, h_p)),
               max_abs_residual_diff=float(np.max(np.abs(res_s - res_p))),
               max_abs_jacobian_diff=float(np.max(np.abs(h_s - h_p))),
               serial_s=t_serial, parallel_s=t_parallel, speedup=t_serial / t_parallel)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
