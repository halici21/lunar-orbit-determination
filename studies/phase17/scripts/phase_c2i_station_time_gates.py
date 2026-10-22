"""Phase C2-I gates G2 (TDB-TT series port) and G3 (UTC chain) for src/lunar_od/station_time.py.

Predeclared in studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json (G2, G3).
The pyerfa reference (inputs and outputs) is studies/phase17/results/phase_c2i_erfa_reference.npz, produced in
agent-sandbox/c2-erfa-reference by phase_c2i_erfa_reference_sandbox.py. Writes
studies/phase17/results/phase_c2i_g2_g3.json.

    python studies/phase17/scripts/phase_c2i_station_time_gates.py --reference-sha256 <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "studies" / "phase17" / "results" / "phase_c2i_erfa_reference.npz"
EPOCHS = ROOT / "studies" / "phase17" / "results" / "phase_c2_survey_epochs.json"
OUT = ROOT / "studies" / "phase17" / "results" / "phase_c2i_g2_g3.json"
G2_VALUE_LIMIT_S = 1e-15
G2_RATE_REL, G2_RATE_ABS = 1e-6, 1e-19
G3_LIMIT_S = 1e-10
H_S = 1.0
DJ00, DAY = 2451545.0, 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-sha256", required=True)
    a = ap.parse_args()
    from lunar_od import station_time as S
    from lunar_od.spice_loader import load_spice_kernels
    load_spice_kernels(None, clear=True)

    sha = hashlib.sha256(REF.read_bytes()).hexdigest()
    r = np.load(REF, allow_pickle=False)
    names = [str(x) for x in r["site_names"]]
    out = dict(reference=dict(file=REF.name, sha256=sha, expected=a.reference_sha256, meta=json.loads(str(r["meta"]))))

    # ---------------- G2 value ----------------
    port = S.dtdb(r["g2_d1"], r["g2_d2"], r["g2_ut"], r["g2_elong"], r["g2_u"], r["g2_v"])
    diff = np.abs(port - r["g2_val"])
    by_set = {lab: float(diff[r["g2_set"] == k].max()) for k, lab in
              ((0, "c2_epochs_geocentric"), (1, "c2_epochs_at_station"), (2, "grid_1950_2100_10_sites"))}
    g2_value = dict(n=int(diff.size), max_abs_s=float(diff.max()), by_set_max_abs_s=by_set,
                    limit_s=G2_VALUE_LIMIT_S, pass_=bool(diff.max() <= G2_VALUE_LIMIT_S))

    # ---------------- G2 rate: analytic vs central difference on the two-part epoch ----------------
    ep = json.loads(EPOCHS.read_text(encoding="utf-8"))
    pos = ep["station_positions_itrf93_m"]
    worst = 0.0
    n_rate = 0
    worst_case = None
    for cell, c in ep["cells"].items():
        et0 = float(c["et0"])
        for name, ts in c["stations"].items():
            t = np.asarray(ts, float)
            for site in (None, S.Site.from_itrf_m(pos[name])):
                an = S.tdb_minus_tt_rate(et0, t, site)
                fd = (S.tdb_minus_tt(et0, t + H_S, site) - S.tdb_minus_tt(et0, t - H_S, site)) / (2 * H_S)
                ratio = np.abs(an - fd) / (G2_RATE_REL * np.abs(an) + G2_RATE_ABS)
                n_rate += t.size
                i = int(np.argmax(ratio))
                if ratio[i] > worst:
                    worst = float(ratio[i])
                    worst_case = dict(cell=cell, station=name, site=site is not None, t=float(t[i]),
                                      analytic=float(an[i]), fd=float(fd[i]), diff=float(an[i] - fd[i]))
    g2_rate = dict(n=n_rate, h_s=H_S, worst_ratio_to_limit=worst, worst=worst_case,
                   limit=f"|analytic - fd| <= {G2_RATE_REL} |rate| + {G2_RATE_ABS}", pass_=bool(worst <= 1.0))

    # ---------------- G3a: leap seconds ----------------
    mjd = np.array([S._mjd(int(y), int(m), int(d)) for y, m, d in zip(r["g3a_y"], r["g3a_m"], r["g3a_d"])], float)
    utc_cal = (mjd - 51544.0) * DAY - 43200.0 + r["g3a_fd"] * DAY
    mod = S.tai_minus_utc(utc_cal)
    bad = np.flatnonzero(mod != r["g3a_dat"])
    g3a = dict(n=int(mod.size), n_boundary_instants=int(r["g3a_n_boundary"]), mismatches=int(bad.size),
               first_mismatch=(None if not bad.size else dict(y=int(r["g3a_y"][bad[0]]), m=int(r["g3a_m"][bad[0]]),
                                                              d=int(r["g3a_d"][bad[0]]), fd=float(r["g3a_fd"][bad[0]]),
                                                              module=float(mod[bad[0]]),
                                                              erfa=float(r["g3a_dat"][bad[0]]))),
               pass_=bool(bad.size == 0))

    # ---------------- G3b / G3c: UTC tags -> TDB seconds past et0 ----------------
    def chain(prefix, et0_per_tag):
        y, m, d, sod = (r[prefix + k] for k in ("y", "m", "d", "sod"))
        site_idx = r[prefix + "site"].astype(int)
        res = np.zeros(sod.size)
        for k, name in enumerate(names):
            sel = np.flatnonzero(site_idx == k)
            site = S.Site.from_itrf_m(pos[name])
            for e0 in np.unique(et0_per_tag[sel]):
                rows = sel[et0_per_tag[sel] == e0]
                tags = [(int(y[i]), int(m[i]), int(d[i]), float(sod[i])) for i in rows]
                res[rows] = S.utc_to_tdb_seconds(tags, float(e0), site)
        ref = ((r[prefix + "tt1"] - DJ00) * DAY - et0_per_tag) + r[prefix + "tt2"] * DAY + r[prefix + "dtdb"]
        return res, ref

    mjd_b = np.array([S._mjd(int(y), int(m), int(d)) for y, m, d in zip(r["g3b_y"], r["g3b_m"], r["g3b_d"])], float)
    et0_b = (mjd_b - 51544.0) * DAY - 43200.0                                  # the tag's UTC midnight
    got_b, ref_b = chain("g3b_", et0_b)
    db = np.abs(got_b - ref_b)
    got_c, ref_c = chain("g3c_", r["g3c_et0"])
    dc = np.abs(got_c - ref_c)
    roundtrip = np.abs(got_c - r["g3c_t"])
    g3b = dict(n=int(db.size), max_abs_s=float(db.max()), limit_s=G3_LIMIT_S, pass_=bool(db.max() <= G3_LIMIT_S))
    g3c = dict(n=int(dc.size), max_abs_s=float(dc.max()), limit_s=G3_LIMIT_S, pass_=bool(dc.max() <= G3_LIMIT_S),
               informational_roundtrip_to_c2_epoch_max_abs_s=float(roundtrip.max()))

    # informational: what the SPICE chain would give on the same tags (the C2 T2 + T3 epoch parts)
    spice_c = np.zeros(dc.size)
    for e0 in np.unique(r["g3c_et0"]):
        rows = np.flatnonzero(r["g3c_et0"] == e0)
        tags = [(int(r["g3c_y"][i]), int(r["g3c_m"][i]), int(r["g3c_d"][i]), float(r["g3c_sod"][i])) for i in rows]
        spice_c[rows] = S.spice_utc_chain_tdb_seconds(tags, float(e0))
    out.update(
        G2_value=g2_value, G2_rate=g2_rate, G3a_leap_seconds=g3a, G3b_utc_chain_random=g3b, G3c_utc_chain_c2_epochs=g3c,
        informational=dict(module_minus_spice_chain_us=dict(min=float((got_c - spice_c).min() * 1e6),
                                                             max=float((got_c - spice_c).max() * 1e6))),
        reference_ok=bool(sha == a.reference_sha256))
    out["G2"] = bool(g2_value["pass_"] and g2_rate["pass_"] and out["reference_ok"])
    out["G3"] = bool(g3a["pass_"] and g3b["pass_"] and g3c["pass_"] and out["reference_ok"])
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
