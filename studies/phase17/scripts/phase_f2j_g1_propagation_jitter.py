"""Phase F2j-G1 amendment A2, step D5: does a 1 um change of the start state move the propagated orbit by
decimetres, and is that the shadow-segment restarts or the integrator's error control?

Predeclaration: studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json (amendment A2).  Propagates
F2j's end-point state with F2g's empirical propagator (Problem._propagate_empirical, rebuilt here so the
shadow cuts can be frozen) in five cases, one process each, and compares positions on the 20 s grid.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records
import phase_f2j_g1_diagnostic as G  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2j_g1_propagation_jitter.json"
DX = 1e-6
TIGHT = (1e-12, 1e-13)


def propagate(pb, x0, k, emp, rtol, atol, cuts=None):
    from lunar_od import dynamics as dyn
    from lunar_od.srp import SRPOptions
    t = pb.grid
    srp = SRPOptions(k_srp_m2_per_kg=k)
    rhs, get_sun = dyn._build_state_rhs(t, pb.mu_m, pb.mu_e, pb.mu_s, pb.eph.earth_position, pb.eph.sun_position,
                                        0.0, 0.0, "indirect", pb.model, None, pb.setup.et0, srp)

    def rhs_e(tt, y):
        out = np.array(rhs(tt, y), dtype=float, copy=True)
        out[3:6] += M.empirical_acceleration(y, emp)
        return out

    if cuts is None:
        cuts = dyn._shadow_contact_times(rhs_e, t, x0, get_sun, srp, rtol, atol)
    hist = np.asarray(dyn._integrate_segmented(t, x0, rhs_e, rtol, atol, "DOP853", cuts, "F2j-G1 D5"))
    return hist, list(cuts)


def case(name: str):
    import warnings
    warnings.simplefilter("ignore")
    p = G.end_point()
    pb = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES))
    x0, k, emp = p[:6].copy(), float(p[6]), p[11:15]
    rtol, atol = (M.RTOL, M.ATOL) if not name.endswith("tight") else TIGHT
    cuts = None
    if name == "plus_frozen_cuts":
        _, cuts = propagate(pb, x0, k, emp, rtol, atol)
    if name.startswith("plus"):
        x0[0] += DX
    hist, used = propagate(pb, x0, k, emp, rtol, atol, cuts)
    return name, hist[:, :3], used


def stats(a, b, ta):
    d = np.linalg.norm(a - b, axis=1)
    return dict(rms_m=float(np.sqrt(np.mean(d ** 2))), max_m=float(d.max()), end_m=float(d[-1]),
                t_of_max_h=float((ta[int(np.argmax(d))] - ta[0]) / 3600.0))


def main():
    names = ("base", "plus", "plus_frozen_cuts", "base_tight", "plus_tight")
    with ProcessPoolExecutor(len(names)) as ex:
        res = {n: (h, c) for n, h, c in ex.map(case, names)}
    pb_grid = M.Problem(300, doppler=True, days=M.F2J_FIT_DAYS, n_emp=len(M.EMP_NAMES)).grid
    h = {n: v[0] for n, v in res.items()}
    cuts = {n: v[1] for n, v in res.items()}
    out = dict(predeclaration="studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json (A2, D5)",
               dx0_m=DX, tight_rtol_atol=TIGHT,
               n_cuts={n: len(c) for n, c in cuts.items()},
               max_cut_shift_s=float(np.max(np.abs(np.array(cuts["plus"]) - np.array(cuts["base"]))))
               if len(cuts["plus"]) == len(cuts["base"]) else None,
               plus_vs_base=stats(h["plus"], h["base"], pb_grid),
               plus_frozen_cuts_vs_base=stats(h["plus_frozen_cuts"], h["base"], pb_grid),
               plus_tight_vs_base_tight=stats(h["plus_tight"], h["base_tight"], pb_grid),
               base_vs_base_tight=stats(h["base"], h["base_tight"], pb_grid))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
