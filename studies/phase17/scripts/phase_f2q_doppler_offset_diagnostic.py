"""Phase F2q: where the 2009-263 counted-Doppler mean offset comes from (diagnostic, no fit).

Predeclaration: studies/phase17/configs/phase_f2q_doppler_mean_offset_predeclaration.json.  F2o-edited Doppler,
days 260..263, scored against F2p's orbit and the LOLA ephemeris, both with F2p's measurement parameters, through
phase_f2l_lola_scoring.score().  The hypothesis rules are the predeclared ones, applied as written.
"""
from __future__ import annotations

import functools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2l_lola_scoring as L  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2q_doppler_offset_diagnostic.json"
ORBITS = ("F2p", "LOLA")
CASES = [(o, d) for o in ORBITS for d in L.DAYS]
BAND_K = 3.0       # fit-day pass-mean band: median +- 3 x 1.4826 MAD
PAD_MMPS = 3.0     # H_PAD difference
FIT = [d.rsplit("/", 1)[1] for d in L.DAYS[:3]]
DAY = L.DAYS[-1].rsplit("/", 1)[1]


def _stats(r):
    r = np.asarray(r, float)
    return dict(n=int(r.size), mean=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))), std=float(r.std()))


def passes(ivs):
    """Per pass: PAD, start UTC, residual array, transmit frequencies (edited intervals already removed)."""
    out = []
    for n in sorted({x["pass_index"] for x in ivs}):
        g = sorted((x for x in ivs if x["pass_index"] == n), key=lambda x: x["t_mid_s"])
        out.append(dict(pad=g[0]["pad"], utc_start=g[0]["utc"], r=np.array([x["resid_mmps"] for x in g]),
                        ft=sorted({round(x["transmit_hz"], 3) for x in g}), keys=[(x["pad"], x["utc"]) for x in g]))
    return out


def main():
    run = json.loads(M.OUT_F2P.read_text(encoding="utf-8"))["runs"]["S1"]
    edits = {(p, u) for p, u in json.loads(M.OUT_F2P.read_text(encoding="utf-8"))["doppler_edits"]}
    with ProcessPoolExecutor(len(CASES)) as ex:
        res = dict(ex.map(functools.partial(L.case, intervals=True, run=run, freq=True), CASES))
    ps = {}
    for o in ORBITS:
        for d in L.DAYS:
            ivs = [x for x in res[(o, d)]["doppler_intervals"] if (x["pad"], x["utc"]) not in edits]
            ps[(o, d.rsplit("/", 1)[1])] = passes(ivs)
    # the two orbits see the same intervals
    for d in FIT + [DAY]:
        assert [p["keys"] for p in ps[("F2p", d)]] == [p["keys"] for p in ps[("LOLA", d)]]

    table = []
    for d in FIT + [DAY]:
        for a, b in zip(ps[("F2p", d)], ps[("LOLA", d)]):
            table.append(dict(day=d, pad=a["pad"], utc_start=a["utc_start"], n=int(a["r"].size),
                              F2p=_stats(a["r"]), LOLA=_stats(b["r"]), transmit_hz=a["ft"]))
    fit_means = np.array([t["LOLA"]["mean"] for t in table if t["day"] in FIT])
    med = float(np.median(fit_means))
    mad = 1.4826 * float(np.median(np.abs(fit_means - med)))
    band = (med - BAND_K * mad, med + BAND_K * mad)
    inside = lambda m: band[0] <= m <= band[1]  # noqa: E731
    for t in table:
        t["LOLA_mean_inside_fit_band"] = inside(t["LOLA"]["mean"])

    day = [t for t in table if t["day"] == DAY]
    rF = np.concatenate([p["r"] for p in ps[("F2p", DAY)]])
    rL = np.concatenate([p["r"] for p in ps[("LOLA", DAY)]])
    mean_F, mean_L = float(rF.mean()), float(rL.mean())
    rL_pass = [p["r"] for p in ps[("LOLA", DAY)]]

    # H_ORBIT
    orbit_part = float((rL - rF).mean())
    h_orbit = abs(orbit_part) >= 0.5 * abs(mean_F)
    # H_DAY
    h_day = all(np.sign(t["LOLA"]["mean"]) == np.sign(mean_L) and not t["LOLA_mean_inside_fit_band"] for t in day)
    # H_PASS: drop one pass, or the worst two (largest |pass mean|, LOLA)
    drop_one = [float(np.concatenate([r for j, r in enumerate(rL_pass) if j != i]).mean())
                for i in range(len(rL_pass))]
    worst2 = sorted(range(len(rL_pass)), key=lambda i: -abs(rL_pass[i].mean()))[:2]
    drop_two = float(np.concatenate([r for j, r in enumerate(rL_pass) if j not in worst2]).mean()) \
        if len(rL_pass) > 2 else float("nan")
    h_pass = any(inside(m) for m in drop_one) or inside(drop_two)
    # H_PAD
    pad_263 = {q: float(np.concatenate([p["r"] for p in ps[("LOLA", DAY)] if p["pad"] == q]).mean())
               for q in sorted({p["pad"] for p in ps[("LOLA", DAY)]})}
    pad_all = {q: float(np.median([t["LOLA"]["mean"] for t in table if t["pad"] == q]))
               for q in sorted({t["pad"] for t in table})}
    d263 = max(pad_263.values()) - min(pad_263.values()) if len(pad_263) > 1 else 0.0
    dall = max(pad_all.values()) - min(pad_all.values())
    h_pad = d263 > PAD_MMPS and dall > PAD_MMPS
    # H_FREQ
    f263 = {f for t in day for f in t["transmit_hz"]}
    ffit = {f for t in table if t["day"] in FIT for f in t["transmit_hz"]}
    disjoint = not (f263 & ffit)
    sharing = [t for t in table if t["day"] in FIT and set(t["transmit_hz"]) & f263]
    h_freq = disjoint and all(not t["LOLA_mean_inside_fit_band"] for t in sharing)

    out = dict(predeclaration="studies/phase17/configs/phase_f2q_doppler_mean_offset_predeclaration.json",
               edits_excluded=len(edits), fit_band_LOLA_mmps=dict(median=med, mad_scaled=mad, low=band[0],
                                                                   high=band[1]),
               day263=dict(mean_F2p=mean_F, mean_LOLA=mean_L, orbit_part=orbit_part, drop_one_means=drop_one,
                           worst_two=[day[i]["utc_start"] for i in worst2], drop_two_mean=drop_two,
                           pad_means_263=pad_263, pad_medians_all_days=pad_all,
                           transmit_hz_263=sorted(f263), transmit_hz_shared_with_fit=sorted(f263 & ffit),
                           fit_passes_sharing_263_frequency=[(t["day"], t["pad"], t["utc_start"], t["LOLA"]["mean"])
                                                             for t in sharing]),
               hypotheses=dict(H_ORBIT=h_orbit, H_DAY=h_day, H_PASS=h_pass, H_PAD=h_pad, H_FREQ=h_freq),
               day_means={d: {o: _stats(np.concatenate([p["r"] for p in ps[(o, d)]])) for o in ORBITS}
                          for d in FIT + [DAY]},
               passes=table)
    out["reading"] = "not located" if not any(out["hypotheses"].values()) else \
        ", ".join(k for k, v in out["hypotheses"].items() if v)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("hypotheses", "reading", "fit_band_LOLA_mmps", "day263")}, indent=1))
    for t in table:
        print(t["day"], t["pad"], t["utc_start"][11:16], t["n"], round(t["F2p"]["mean"], 2), round(t["LOLA"]["mean"], 2),
              round(t["LOLA"]["std"], 2), t["LOLA_mean_inside_fit_band"], t["transmit_hz"][:3], len(t["transmit_hz"]))


if __name__ == "__main__":
    main()
