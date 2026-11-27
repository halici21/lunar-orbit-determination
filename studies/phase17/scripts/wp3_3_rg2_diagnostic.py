"""F3 WP3.3 exploratory diagnostic (after the result): why the O2 reference fails R_G2(a).

Not a gate; recorded in docs/assurance/amendment-log.md. The predeclared R_G2 FAIL for O2 stands.
Hypothesis: the reference cut set (found on the iteration-1 trajectory) is not converged against the final
reference: where a cut moved by more than the 1 ms probe between the iteration-1 and the final trajectory, the
probe straddles no boundary. Test: rebuild the O2 reference exactly as the driver does, recount the contacts on
the final trajectory, and compare each failing probe with |recount - cut|.

    python wp3_3_rg2_diagnostic.py   -> studies/phase17/results/wp3_3_rg2_diagnostic.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wp3_3_integrator_certification as C  # noqa: E402

OUT = C.ROOT / "studies" / "phase17" / "results" / "wp3_3_rg2_diagnostic.json"


R_SUN_M = 6.957e8


def _nu(ctx, t, r) -> float:
    from lunar_od.srp import _illumination_for
    return float(_illumination_for(np.asarray(r, float), np.asarray(ctx.eph.sun_position(t), float), ctx.srp))


def _outer_depth(ctx, t, r) -> float:
    """(a_sun + a_occ) - gamma: > 0 inside the outer penumbra contact (geometric, radians)."""
    from lunar_od.constants import R_MOON_M
    r = np.asarray(r, float)
    s = np.asarray(ctx.eph.sun_position(t), float) - r
    m = -r
    a_sun = np.arcsin(R_SUN_M / np.linalg.norm(s))
    a_occ = np.arcsin(R_MOON_M / np.linalg.norm(m))
    gamma = np.arccos(np.clip(np.dot(s, m) / np.linalg.norm(s) / np.linalg.norm(m), -1.0, 1.0))
    return float(a_sun + a_occ - gamma)


def main() -> int:
    warnings.simplefilter("ignore")
    from lunar_od.radiometrics import _interp_state
    orbit, days = "O2_SUNLIT_100KM", 28.0
    ctx = C.orbit_context(orbit, days)
    dense_t = C.grid(days, C.DENSE_S)
    pre = ctx.propagate(dense_t, ctx.x0, *C.PRE_TOL)
    cuts0, _ = C.contacts(ctx, dense_t, pre)
    dense1 = C.segmented(ctx, dense_t, cuts0, *C.REF_TOL)
    cuts1, _ = C.contacts(ctx, dense_t, dense1)
    final = C.segmented(ctx, dense_t, cuts1, *C.REF_TOL)
    recount, _ = C.contacts(ctx, dense_t, final)
    cuts1, recount = np.array(cuts1), np.array(recount)
    shift = np.abs(recount - cuts1) if recount.size == cuts1.size else None
    fails = []
    for k, c in enumerate(cuts1):
        lo = ctx.illumination_class(c - C.R_G2_PROBE_S, _interp_state(dense_t, final, c - C.R_G2_PROBE_S)[:3])
        hi = ctx.illumination_class(c + C.R_G2_PROBE_S, _interp_state(dense_t, final, c + C.R_G2_PROBE_S)[:3])
        if lo == hi:
            probes = {}
            for dt in (1e-3, 1e-2, 1e-1):
                for sgn in (-1, 1):
                    t = c + sgn * dt
                    r = _interp_state(dense_t, final, t)[:3]
                    probes[f"{sgn * dt:+g}"] = dict(one_minus_nu=1.0 - _nu(ctx, t, r), cls=ctx.illumination_class(t, r),
                                                    outer_depth_rad=_outer_depth(ctx, t, r))
            fails.append(dict(index=k, t_s=float(c), day=float(c / C.DAY),
                              shift_to_final_s=None if shift is None else float(shift[k]), probes=probes))
    res = dict(orbit=orbit, cuts=int(cuts1.size), recount=int(recount.size), probe_failures=len(fails),
               failures=fails,
               max_shift_s=None if shift is None else float(shift.max()),
               shift_above_probe=None if shift is None else int(np.sum(shift > C.R_G2_PROBE_S)),
               failing_probes_with_shift_above_probe=sum(1 for f in fails if (f["shift_to_final_s"] or 0) > C.R_G2_PROBE_S),
               max_shift_iteration0_to_1_s=float(np.max(np.abs(np.array(cuts0) - cuts1))) if len(cuts0) == cuts1.size else None)
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
