"""PHASE 17-GEO s14/s24 - attribute the G0 published-vs-consistent difference.

ANALYSIS SPACE ONLY.  Same canonical orbit, same propagation, same stations,
same light-time configuration.  Only the Earth ephemeris the MEASUREMENT model
sees is changed:

  PUBLISHED      phase17_r1m_core.build_range_arc, untouched (the R1M..R1O-R fixture)
  CAMPAIGN_TABLE build_geo_range_arc fed the campaign module's own Earth table
                 -- must reproduce PUBLISHED exactly, or the builder itself differs
  OFFSET_ONLY    SPICE Earth 5.83 d early (the campaign ET0), not clamped
  CLAMP_ONLY     SPICE Earth at the right epoch, frozen after the table's end (4 orbits)
  CONSISTENT     SPICE Earth at the right epoch over the whole arc (the GEO fixture)

This is a diagnostic of the fixture, not a geometry choice: no case in the
predeclared grid is added, moved or removed by it.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARTIFACTS = HERE.parent / "artifacts"


def _init():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)


def run(mode: str) -> dict:
    import phase17_geo_core as G
    from phase17_r1m_core import build_range_arc, campaign_epoch, campaign_initial_state

    et0, T = campaign_epoch()
    x0 = campaign_initial_state()
    D = 15.0 * T
    if mode == "PUBLISHED":
        fx = build_range_arc(0.0, D, label=mode, station_filter=(
            "Goldstone DSN", "Madrid DSN", "Canberra DSN"), cadence_s=90.0)
        h_x0, h_k, w, obs = fx.h_x0, fx.h_k, fx.w, fx.obs
    else:
        arc = G.build_geo_range_arc(x0, et0, D, label=mode, earth_mode=mode)
        h_x0, h_k, w, obs = arc.h_x0, arc.h_k, arc.w, arc.obs
    m = G.information_metrics(h_x0, h_k, w)
    np.savez_compressed(HERE.parent / "results" / "phase17_geo_cache" / f"FIXTURE_{mode}.npz",
                        h_x0=h_x0, h_k=h_k, w=w, obs=obs)
    return dict(mode=mode, n_obs=m["n_obs"], f_perp=m["f_perp"], theta_k_deg=m["theta_k_deg"],
                conditional_k_information=m["conditional_k_information"],
                sigma_k_frac=m["sigma_k_frac"], k_column_norm=m["k_column_norm"])


def main() -> None:
    (HERE.parent / "results" / "phase17_geo_cache").mkdir(parents=True, exist_ok=True)
    modes = ["PUBLISHED", "CAMPAIGN_TABLE", "OFFSET_ONLY", "CLAMP_ONLY", "CONSISTENT"]
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as ex:
        rows = list(ex.map(run, modes))
    cache = HERE.parent / "results" / "phase17_geo_cache"
    a = np.load(cache / "FIXTURE_PUBLISHED.npz")
    b = np.load(cache / "FIXTURE_CAMPAIGN_TABLE.npz")
    same_shape = a["h_x0"].shape == b["h_x0"].shape
    bitwise = bool(same_shape and all(np.array_equal(a[k], b[k]) for k in ("h_x0", "h_k", "w", "obs")))
    max_rel = (float(max(np.max(np.abs(a[k] - b[k])) / max(np.max(np.abs(a[k])), 1e-300)
                         for k in ("h_x0", "h_k"))) if same_shape else float("nan"))
    print(f"{'mode':16s} {'n_obs':>6s} {'f_perp':>9s} {'theta_K':>8s} {'I_K|x':>12s} {'sigK/K':>9s}")
    for r in rows:
        print(f"{r['mode']:16s} {r['n_obs']:6d} {r['f_perp']:9.6f} {r['theta_k_deg']:8.3f} "
              f"{r['conditional_k_information']:12.5e} {100 * r['sigma_k_frac']:8.4f}%")
    print(f"\nbuilder check: CAMPAIGN_TABLE vs PUBLISHED  same shape={same_shape}  "
          f"bitwise={bitwise}  max rel diff={max_rel:.3e}")
    out = dict(rows=rows, builder_reproduces_published_bitwise=bitwise,
               builder_max_rel_diff=max_rel)
    (ARTIFACTS / "phase17_geo_fixture_attribution.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
