"""WP4.2-R G3: on the 24 h arc (seed 2000) the fixed BLS-LM ends at or below the cost at the truth, all three types.

Same arc, data and start as wp4_2r_survey.py; runs the production code as it is (no monkeypatch). Run from the worktree.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import wp2_5_long_arc_compare as L  # noqa: E402
from lunar_od.measurements import compute_position_residuals, compute_range_rate_residuals, measurement_sigma_vector  # noqa: E402
from lunar_od.two_way_range import compute_two_way_range_residuals, generate_two_way_range_measurements  # noqa: E402

OUT = Path(ARGS[0]) if ARGS else Path(__file__).parents[1] / "results" / "wp4_2r_g3_24h.json"


def data(a, kind):
    rng = np.random.default_rng(2000)
    if kind == "position":
        return L.position(a, L.IMPLICIT, True, rng)
    if kind == "range_rate":
        return L.generate_range_rate_measurements(a["t"], a["aug"][:, :6], L.STATIONS, a["vis"], a["eph"].earth_position,
                                                  a["eph"].earth_velocity, L.et0, noise=True, rng=rng, arc_id=1,
                                                  companion_geometry="apparent_one_way")
    return generate_two_way_range_measurements(a["t"], a["aug"][:, :6], L.STATIONS, a["vis"], a["eph"].earth_position,
                                               a["eph"].earth_velocity, L.et0, noise=True, rng=rng, arc_id=1)[:2]


def cost(a, kind, x, obs, geo):
    f = {"position": compute_position_residuals, "range_rate": compute_range_rate_residuals,
         "two_way_range": compute_two_way_range_residuals}[kind]
    r = np.asarray(f(a["propagate"](x), obs, geo)[0]) / measurement_sigma_vector(obs, geo)
    return float(r @ r)


def main():
    a = L.setup(24.0)
    out = {"arc_h": 24.0, "seed": 2000, "gate": "final cost <= cost at the truth", "per_type": {}}
    for kind in ("position", "range_rate", "two_way_range"):
        obs, geo = data(a, kind)
        t0 = time.perf_counter()
        x, status, stats = getattr(L.est, f"estimate_{kind}_bls_lm")(a["t"], obs, L.start, geo, *L.gm,
                                                                    a["eph"].earth_position, a["eph"].sun_position,
                                                                    rtol=1e-12, atol=1e-13)
        row = {"pos_err_m": float(np.linalg.norm(np.asarray(x[:3]) - L.x0[:3])), "status": str(status),
               "iterations": int(stats.iterations), "cost": cost(a, kind, x, obs, geo),
               "cost_truth": cost(a, kind, L.x0, obs, geo), "rows": int(obs.shape[0]),
               "wall_s": time.perf_counter() - t0}
        row["pass"] = row["cost"] <= row["cost_truth"] and row["status"] != "DampingLimit"
        out["per_type"][kind] = row
        print(kind, json.dumps(row), flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    out["pass"] = all(r["pass"] for r in out["per_type"].values())
    OUT.write_text(json.dumps(out, indent=1))
    print("G3", "PASS" if out["pass"] else "FAIL")


if __name__ == "__main__":
    main()
