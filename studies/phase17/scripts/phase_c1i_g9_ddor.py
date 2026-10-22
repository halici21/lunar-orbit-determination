"""Phase C1-I gate G9: delta-DOR Jacobians in the barycentric frame vs the authoritative R1O-D2 f_perp.

Predeclared in studies/phase17/configs/phase_c1i_light_time_implementation_predeclaration.json (G9).
Writes studies/phase17/results/phase_c1i_g9_ddor.json.

R1O-D2 / F1 canonical geometry (CANON, Goldstone-Canberra, production visibility rows, dense angular 5 nrad):
the authoritative range + delta-DOR f_perp is 0.172343. The same design is recomputed with
  MOON        the production delta-DOR rows (must reproduce the authoritative value),
  SSB_DDOR    delta-DOR rows from relativistic_light_time (DSN model), range rows unchanged  [the G9 rule],
  SSB_BOTH    delta-DOR and two-way range rows both from relativistic_light_time            [informational].
Rule: |delta f_perp| >= 0.01 or an absolute-class change opens a delta-DOR requalification phase.

    python studies/phase17/scripts/phase_c1i_g9_ddor.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

OUT = HERE.parents[2] / "studies" / "phase17" / "results" / "phase_c1i_g9_ddor.json"
GRID = HERE.parents[2] / "studies" / "phase17" / "configs" / "phase17_r1od2_predeclared_grid.json"
AUTHORITATIVE_F_PERP = 0.172343


def main():
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)
    import phase17_r1od2_core as D2
    D2.install_legacy_earth_guard()
    from lunar_od import relativistic_light_time as R
    from lunar_od.radiometrics import _interp_state
    from lunar_od.two_way_range import _station_state_providers_for_pass

    t0 = time.time()
    g = next(x for x in json.loads(GRID.read_text(encoding="utf-8"))["geometries"] if x["case_id"] == "CANON")
    scn = D2.Scenario(g["case_id"], g["family"], float(g["et0"]), tuple(g["x0"]), float(g["duration_s"]),
                      float(g["k_srp"]), g["shadow_model"])
    nom = D2.propagate_nom48(scn)
    rb = D2.build_range_block(scn, nom)
    rows = D2.build_ddor_rows(scn, rb, "GC", D2.ROW_RULE_PRODUCTION_VISIBILITY)
    w5 = D2.ddor_weights(rows, D2.SEMANTICS_ANGULAR, 5e-9)
    m_range = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w)])
    m_moon = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w), (rows.h_x0, rows.h_k, w5)])

    t_grid = rb.t_grid
    model = R.DSN_LIGHT_TIME_MODEL
    eph = R.BarycentricEphemeris(float(scn.et0))
    sc = R.SpacecraftHistory(t_grid, nom)                      # 48 columns: x0 and K partials
    b_fn = lambda t: _interp_state(t_grid, rows.hist_b, t)    # noqa: E731
    dd = np.zeros((rows.n, 7))
    for i, t_obs in enumerate(rows.t_used):
        a_state = _interp_state(t_grid, rows.hist_a, float(t_obs))
        dd[i] = R.delta_dor_partial(R.solve_delta_dor(float(t_obs), a_state, b_fn, sc, eph, model), sc)
    m_ssb_ddor = D2.info_metrics([(rb.h_x0, rb.h_k, rb.w), (dd[:, :6], dd[:, 6], w5)])

    pg = rb.pass_geo
    st = [R.exact_epoch_station_state(lambda t, p=p: np.asarray(p.state(t), float), pg.et0_s, pg.t_s,
                                      pg.earth_pos_mci_m, pg.earth_vel_mci_mps)
          for p in _station_state_providers_for_pass(pg)]
    hr = np.zeros((rb.obs.shape[0], 7))
    for i in range(rb.obs.shape[0]):
        sol = R.solve_two_way(float(rb.obs[i, 0]), st[int(rb.obs[i, 2]) - 1], sc, eph, model)
        hr[i] = R.two_way_range_partial(sol, sc)
    m_ssb_both = D2.info_metrics([(hr[:, :6], hr[:, 6], rb.w), (dd[:, :6], dd[:, 6], w5)])
    m_ssb_range = D2.info_metrics([(hr[:, :6], hr[:, 6], rb.w)])

    def pick(m):
        return {k: m[k] for k in ("f_perp", "theta_k_deg", "sigma_k_frac")}
    d = m_ssb_ddor["f_perp"] - m_moon["f_perp"]
    cls_moon, cls_ssb = D2.absolute_class(m_moon["f_perp"]), D2.absolute_class(m_ssb_ddor["f_perp"])
    out = dict(
        n_range=int(rb.obs.shape[0]), n_ddor=int(rows.n),
        authoritative_f_perp=AUTHORITATIVE_F_PERP,
        range_only_moon=pick(m_range), range_only_ssb=pick(m_ssb_range),
        moon=pick(m_moon), ssb_ddor=pick(m_ssb_ddor), ssb_both=pick(m_ssb_both),
        moon_reproduces_authoritative=bool(abs(m_moon["f_perp"] - AUTHORITATIVE_F_PERP) < 5e-7),
        delta_f_perp_ssb_ddor=float(d), absolute_class=dict(moon=cls_moon, ssb_ddor=cls_ssb),
        jacobian_row_relative_change=dict(
            ddor_max=float(np.max(np.linalg.norm(dd[:, :6] - rows.h_x0, axis=1) / np.linalg.norm(rows.h_x0, axis=1))),
            range_max=float(np.max(np.linalg.norm(hr[:, :6] - rb.h_x0, axis=1) / np.linalg.norm(rb.h_x0, axis=1)))),
        requalification_opened=bool(abs(d) >= 0.01 or cls_moon != cls_ssb),
        elapsed_s=time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
