"""Phase F2o: per-interval counted-Doppler outlier editing on 2009-260..263.

Predeclaration: studies/phase17/configs/phase_f2o_doppler_interval_editing_predeclaration.json.  An interval is
edited when |r - running median(9, within the pass)| > 5 x max(1.4826 MAD, SIGMA_DOP).  Applied with F2k's
orbit; the LOLA ephemeris is the cross-check.  Residuals come from phase_f2l_lola_scoring.score().
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

OUT = HERE.parent / "results" / "phase_f2o_doppler_editing.json"
WINDOW, K_SIGMA = 9, 5.0
FLOOR_MMPS = M.SIGMA_DOP * 1e3
PASS_FACTOR = 5.0   # F2m's pass rule, reported after editing
CASES = [(o, d) for o in ("F2k", "LOLA") for d in L.DAYS]


def edit(ivs):
    """Flags per interval (same order) and the per-pass thresholds."""
    flags = np.zeros(len(ivs), bool)
    passes = []
    for n in sorted({x["pass_index"] for x in ivs}):
        idx = np.array([i for i, x in enumerate(ivs) if x["pass_index"] == n])
        idx = idx[np.argsort([ivs[i]["t_mid_s"] for i in idx])]
        r = np.array([ivs[i]["resid_mmps"] for i in idx])
        h = WINDOW // 2
        med = np.array([np.median(r[max(0, k - h):k + h + 1]) for k in range(r.size)])
        dev = r - med
        sig = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
        thr = K_SIGMA * max(sig, FLOOR_MMPS)
        bad = np.abs(dev) > thr
        flags[idx[bad]] = True
        passes.append(dict(pass_index=int(n), pad=ivs[idx[0]]["pad"], utc_start=ivs[idx[0]]["utc"], n=int(r.size),
                           sigma_rob_mmps=sig, threshold_mmps=thr, n_edited=int(bad.sum())))
    return flags, passes


def _stats(r):
    r = np.asarray(r, float)
    return dict(n=int(r.size), rms=float(np.sqrt(np.mean(r ** 2))), mean=float(r.mean()), std=float(r.std()))


def main():
    with ProcessPoolExecutor(len(CASES)) as ex:
        res = dict(ex.map(functools.partial(L.case, intervals=True), CASES))
    out = dict(predeclaration="studies/phase17/configs/phase_f2o_doppler_interval_editing_predeclaration.json",
               window=WINDOW, k_sigma=K_SIGMA, floor_mmps=FLOOR_MMPS, orbits={})
    keys = {}
    for o in ("F2k", "LOLA"):
        days, edited_all, pass_scatter = {}, [], []
        for d in L.DAYS:
            ivs = res[(o, d)]["doppler_intervals"]
            flags, passes = edit(ivs)
            day = d.rsplit("/", 1)[1]
            r = np.array([x["resid_mmps"] for x in ivs])
            edited = [dict(day=day, **ivs[i], threshold_mmps=next(p["threshold_mmps"] for p in passes
                                                                   if p["pass_index"] == ivs[i]["pass_index"]))
                      for i in np.flatnonzero(flags)]
            edited_all += edited
            keys.setdefault(o, set()).update((day, e["pad"], e["utc"]) for e in edited)
            for p in passes:
                m = np.array([x["pass_index"] == p["pass_index"] for x in ivs]) & ~flags
                pass_scatter.append(dict(day=day, pad=p["pad"], utc_start=p["utc_start"], std=float(r[m].std())))
            days[day] = dict(before=_stats(r), after=_stats(r[~flags]), n_edited=int(flags.sum()), passes=passes,
                             range_m=res[(o, d)]["range_m"])
        med = float(np.median([s["std"] for s in pass_scatter]))
        out["orbits"][o] = dict(days=days, edited=edited_all,
                                f2m_rule_after_editing=dict(median_mmps=med, threshold_mmps=PASS_FACTOR * med,
                                                            rejected=[s for s in pass_scatter
                                                                      if s["std"] > PASS_FACTOR * med]))
    fit = {d.rsplit("/", 1)[1] for d in L.DAYS[:3]}
    out["fit_day_intervals_edited"] = any(e["day"] in fit for e in out["orbits"]["F2k"]["edited"])
    out["edited_sets_agree"] = keys.get("F2k", set()) == keys.get("LOLA", set())
    out["only_in_F2k"] = sorted(keys.get("F2k", set()) - keys.get("LOLA", set()))
    out["only_in_LOLA"] = sorted(keys.get("LOLA", set()) - keys.get("F2k", set()))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(fit_day_intervals_edited=out["fit_day_intervals_edited"],
                          edited_sets_agree=out["edited_sets_agree"],
                          only_in_F2k=out["only_in_F2k"], only_in_LOLA=out["only_in_LOLA"],
                          per_day={o: {d: dict(n_edited=v["n_edited"], before=v["before"], after=v["after"])
                                       for d, v in out["orbits"][o]["days"].items()} for o in out["orbits"]},
                          edited_F2k=[(e["day"], e["pad"], e["utc"], round(e["resid_mmps"], 1),
                                       round(e["threshold_mmps"], 1)) for e in out["orbits"]["F2k"]["edited"]],
                          f2m_after={o: out["orbits"][o]["f2m_rule_after_editing"] for o in out["orbits"]}),
                     indent=1))


if __name__ == "__main__":
    main()
