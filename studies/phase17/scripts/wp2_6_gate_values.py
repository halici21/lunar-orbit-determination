"""WP2.6 slice 2: the measured values behind gates G7-G11 (the tests assert them; this records them).

Uses the arc of tests/test_wp2_6_delta_dor_ssb.py. Also records two informational numbers: delta_dor_ssb minus
the legacy Moon-centred observable, and delta-DOR with the same-model quasar term minus delta-DOR with the IERS
consensus quasar term (the inconsistency the same-model design removes). Run from the worktree.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_wp2_6_delta_dor_ssb as T  # noqa: E402
from lunar_od.delta_dor import QuasarDirection, delta_dor_observable_s, quasar_differential_delay_s, \
    solve_common_transmit_event  # noqa: E402
from lunar_od.station_time import tdb_minus_tt  # noqa: E402


def main():
    a = inspect.unwrap(T.arc)()
    R, eph, C = a["R"], a["eph"], T.C
    out = {"arc": "2027-03-02T16:00 TDB + 1 h, 100 km lunar orbit, quasar 3 deg from the spacecraft"}

    g7 = {}
    for st_a, st_b in ((T.GOLDSTONE, T.CANBERRA), (T.MADRID, T.CANBERRA)):
        fa, fb = a["station"](st_a), a["station"](st_b)
        sites = (a["site"](st_a), a["site"](st_b))
        model = R.LightTimeModel(frame="SSB", interval_time_scale="TT", station_time="FULL_TOPOCENTRIC")
        rows = []
        for t in np.linspace(0.0, 3600.0, 10):
            k = T._quasar(a, t)
            diff = (R.quasar_delay_station_time(t, fa, fb, k, eph, model, sites=sites).delay_s
                    - R.quasar_delay_consensus(t, fa, fb, k, eph, gravity_bodies=()).delay_s)
            geo = [f(t)[:3] + eph.moon_state(t)[:3] - eph.body_position("EARTH", t) for f in (fa, fb)]
            v = eph.body_velocity("EARTH", t)
            topo = [float(tdb_minus_tt(a["et0"], t, s)) - float(tdb_minus_tt(a["et0"], t)) for s in sites]
            rows.append({"diff_ns": diff * 1e9, "dtdb_error_ns": ((topo[1] - topo[0]) - v @ (geo[1] - geo[0]) / C ** 2) * 1e9})
        g7[f"{st_a.name}-{st_b.name}"] = {"max_abs_diff_ns": max(abs(r["diff_ns"]) for r in rows),
                                         "max_abs_diff_plus_dtdb_error_ns": max(abs(r["diff_ns"] + r["dtdb_error_ns"]) for r in rows)}
    out["G7_same_model_minus_consensus"] = g7

    fa, fb = a["station"](T.GOLDSTONE), a["station"](T.CANBERRA)
    sites = (a["site"](T.GOLDSTONE), a["site"](T.CANBERRA))
    rows = []
    for t in np.linspace(300.0, 3300.0, 6):
        k = T._quasar(a, t)
        ssb = R.delta_dor_ssb(t, fa, fb, a["sc"], eph, k, sites=sites)
        consensus = R.quasar_delay_consensus(t, fa, fb, k, eph)
        ev = solve_common_transmit_event(t, fa(t)[:3], lambda x: fb(x)[:3], lambda x: a["sc"].state(x)[:3])
        legacy = delta_dor_observable_s(ev, quasar_differential_delay_s(fa(t)[:3], fb(t)[:3],
                                                                        QuasarDirection.from_unit_vector(k)))
        rows.append({"t_obs": float(t), "ssb_minus_legacy_ns": (ssb.value_s - legacy) * 1e9,
                     "ddor_same_model_minus_ddor_consensus_ns": (consensus.delay_s - ssb.quasar.delay_s) * 1e9,
                     "quasar_gravitational_ns": ssb.quasar.gravitational_s * 1e9})
    out["informational_goldstone_canberra"] = rows
    path = Path("studies/phase17/results/wp2_6_gate_values.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
