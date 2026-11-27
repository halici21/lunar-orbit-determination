"""F5 WP5.1, information only (not a gate): attribute the O_L link differences.

    python studies/phase17/scripts/wp5_1_link_diagnostic.py

For each case, the production ``propagate_state`` (DOP853, rtol 1e-13, atol 1e-9, no step cap) is compared with
(a) the TDB adapter at the SAME uncapped setting (same physics, same tolerances: the remaining difference
is solve_ivp dense output vs. stop-at-every-sample, plus the harmonic rotation grid), and (b) the TDB adapter at
the accepted (self-converged) setting (= the O_L gate). The production path with the exact-epoch DIRECT
rotation is compared with (b) as well. Writes ``studies/phase17/results/wp5_1_link_diagnostic.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))

import compare as C  # noqa: E402
import lunar_od_side as L  # noqa: E402


def _d(a, b):
    return float(np.max(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)))


def main() -> int:
    sc = C.scenario()
    out = {}
    for case in ("D2-50", "D2-100", "D3"):
        s_acc = C.setting(case)
        uncapped = L.Setting(s_acc.rtol, s_acc.atol)
        _, x_unc, _ = L.propagate_aligned(case, sc, time_arg="TDB", setting=uncapped)
        _, x_acc, _ = L.propagate_aligned(case, sc, time_arg="TDB", setting=s_acc)
        x_prod = L.propagate_production(case, sc, setting=uncapped)
        row = {"accepted_setting": s_acc.as_dict(),
               "production_grid_minus_adapter_uncapped_m": _d(x_prod, x_unc),
               "production_grid_minus_adapter_accepted_m (O_L)": _d(x_prod, x_acc),
               "adapter_uncapped_minus_adapter_accepted_m": _d(x_unc, x_acc)}
        if case.startswith("D2"):
            x_dir = L.propagate_production(case, sc, setting=uncapped, rotation="direct")
            row["production_direct_minus_adapter_uncapped_m"] = _d(x_dir, x_unc)
            row["production_direct_minus_production_grid_m"] = _d(x_dir, x_prod)
        out[case] = row
        print(case, json.dumps(row), flush=True)
    path = ROOT / "studies" / "phase17" / "results" / "wp5_1_link_diagnostic.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
