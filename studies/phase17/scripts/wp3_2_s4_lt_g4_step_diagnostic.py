"""F3 WP3.2 S4: exploratory diagnostic of the LT_G4(a)/(d) FAIL (written AFTER the result; not a gate).

LT_G4(a) and (d) failed as computed at the predeclared steps (h = 10 m and 100 m). This script asks whether the
analytic gradients or the finite-difference oracle carry the error. For each gated sample it evaluates the same
fourth-order central difference at several steps and splits the error into the production geometric part
(srp_acceleration_position_gradient against an FD of srp_acceleration_with_lunar_shadow, code S4 did not touch)
and the S4 increment part. A gradient defect gives an error that does not fall with any step; FD truncation
falls as h^4 and FD round-off grows as 1/h. Records the per-sample minimum over the steps.

usage: python wp3_2_s4_lt_g4_step_diagnostic.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_od import earth_shadow as es  # noqa: E402
from lunar_od import srp_light_time as lt  # noqa: E402
from lunar_od.lunar_j2_model import use_lunar_j2_model  # noqa: E402
from lunar_od.srp import srp_acceleration_position_gradient, srp_acceleration_with_lunar_shadow  # noqa: E402
from tests import test_wp3_2_srp_light_time as M  # noqa: E402

STEPS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def part_a(sm):
    out = {"CONICAL_PENUMBRA": [], "NO_SHADOW": []}
    for i in range(sm["r"].shape[0]):
        r, v, s, vs, vm = sm["r"][i], sm["v"][i], sm["s"][i], sm["vs"][i], sm["vm"][i]
        for srp in (M.PEN01, M.NOS01):
            if srp is M.PEN01:
                nu = lt.light_time_illumination(r, s, vs, vm, srp)
                if not (sm["band"][i] and 0.0 < nu < 1.0):
                    continue
                if not M._stencil_clear(lambda x: M._lunar_contacts(x, s, vs, vm), r, 10.0):
                    continue
            _, g_r, _, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, srp)
            g_p = srp_acceleration_position_gradient(r, s, srp)
            per_h = {}
            for h in STEPS:
                fd_inc = M._fd4(lambda x: M.WD.full_ray_crossing(x, v, s, vs, vm, srp), r, h)
                fd_p = M._fd4(lambda x: srp_acceleration_with_lunar_shadow(x, s, srp), r, h)
                per_h[h] = dict(total=_rel(g_r + g_p, fd_inc + fd_p), production=_rel(g_p, fd_p),
                                increment=_rel(g_r, fd_inc))
            out[srp.shadow_model].append(per_h)
    return out


def part_d(n=2000, seed=20261125):
    from tests.test_wp3_2_earth_shadow import make_samples as s3_samples
    r_all, s_all, e_all = s3_samples(n, seed)
    rng = np.random.default_rng(20261127 + 3)
    v_all = M.G._unit(rng, n) * rng.uniform(0.5e3, 2.5e3, size=n)[:, None]
    vm_all = M.G._unit(rng, n) * rng.uniform(28.5e3, 31.5e3, size=n)[:, None]
    vs_all = M.G._unit(rng, n) * rng.uniform(0.0, 16.0, size=n)[:, None]
    rows = []
    for i in range(n):
        r, s, e, v, vs, vm = r_all[i], s_all[i], e_all[i], v_all[i], vs_all[i], vm_all[i]
        if not 0.0 < es.earth_illumination(r, s, e, M.ES) < 1.0:
            continue

        def contacts(x):
            return (*es.earth_shadow_contact_functions(x, s, e, M.ES), *M._lunar_contacts(x, s, vs, vm))
        if not M._stencil_clear(contacts, r, 100.0):
            continue
        da, g_r, _, _ = lt.srp_light_time_increment_and_jacobians(r, v, s, vs, vm, M.PEN01)
        nu, grad_nu = es.earth_illumination_and_gradient(r, s, e, M.ES)
        got = nu * g_r + np.outer(da, grad_nu)
        per_h = {}
        for h in STEPS:
            fd = M._fd4(lambda x: es.earth_illumination(x, s, e, M.ES) * M.WD.full_ray_crossing(x, v, s, vs, vm,
                                                                                               M.PEN01), r, h)
            fd_nu = M._fd4(lambda x: np.atleast_1d(es.earth_illumination(x, s, e, M.ES)), r, h)[0]
            per_h[h] = dict(total=_rel(got, fd), earth_nu_gradient=_rel(grad_nu, fd_nu))
        rows.append(per_h)
    return rows


def summarise(rows, keys):
    out = {}
    for key in keys:
        by_h = {str(h): max(r[h][key] for r in rows) for h in STEPS}
        best = [min(r[h][key] for h in STEPS) for r in rows]
        out[key] = dict(max_over_samples_by_step=by_h, max_over_samples_of_min_over_steps=max(best),
                        n_samples_min_le_1e6=int(sum(b <= 1e-6 for b in best)), n=len(rows))
    return out


def main(out_path):
    with use_lunar_j2_model("iau_mean_pole_legacy"):
        a = part_a(M.make_samples())
        d = part_d()
    payload = dict(study="F3 WP3.2 S4 LT_G4 step diagnostic (exploratory, after the result)", steps_m=list(STEPS),
                   a_conical=summarise(a["CONICAL_PENUMBRA"], ("total", "production", "increment")),
                   a_no_shadow=summarise(a["NO_SHADOW"], ("total", "production", "increment")),
                   d_composed=summarise(d, ("total", "earth_nu_gradient")))
    Path(out_path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
