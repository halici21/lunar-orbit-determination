"""Phase F2m: per-pass Doppler screening (reject a pass whose scatter about its mean exceeds 5 x the median),
applied to every day 2009-260..263 with F2k's orbit; LOLA's orbit as a cross-check.

Predeclaration: studies/phase17/configs/phase_f2m_doppler_pass_screening_predeclaration.json.  The residuals come
from phase_f2l_lola_scoring.score() (predict()'s measurement code); totals after screening are recombined from
the per-pass counts, RMS and means.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2l_lola_scoring as L  # noqa: E402

OUT = HERE.parent / "results" / "phase_f2m_doppler_screening.json"
FACTOR = 5.0
CASES = [(o, d) for o in ("F2k", "LOLA") for d in L.DAYS]


def _combine(passes):
    n = np.array([q["n"] for q in passes], float)
    if n.sum() == 0:
        return dict(n=0, rms=float("nan"), mean=float("nan"))
    return dict(n=int(n.sum()), rms=float(np.sqrt(np.sum(n * np.array([q["rms"] for q in passes]) ** 2) / n.sum())),
                mean=float(np.sum(n * np.array([q["mean"] for q in passes])) / n.sum()))


def _screen(res, kind):
    """Rejected passes of every day for one orbit: about-mean RMS > FACTOR x median over all passes."""
    allp = [(d, q) for d, v in res.items() for q in v[kind]]
    med = float(np.median([q["rms_about_mean"] for _, q in allp]))
    bad = [dict(day=d.rsplit("/", 1)[1], pad=q["pad"], t_start_h=q["t_start_h"], n=q["n"],
                rms_about_mean=q["rms_about_mean"]) for d, q in allp if q["rms_about_mean"] > FACTOR * med]
    return med, bad


def main():
    with ProcessPoolExecutor(len(CASES)) as ex:
        res = dict(ex.map(L.case, CASES))
    orbits = {o: {d: res[(o, d)] for d in L.DAYS} for o in ("F2k", "LOLA")}
    out = dict(predeclaration="studies/phase17/configs/phase_f2m_doppler_pass_screening_predeclaration.json",
               factor=FACTOR, screening={}, days={})
    for o, r in orbits.items():
        med, bad = _screen(r, "doppler_passes")
        med_r, bad_r = _screen(r, "range_passes")
        out["screening"][o] = dict(doppler_median_mmps=med, doppler_threshold_mmps=FACTOR * med, doppler_rejected=bad,
                                   range_median_m_reported_only=med_r, range_would_reject_reported_only=bad_r)
    rule = out["screening"]["F2k"]["doppler_rejected"]           # the rule screens with our own orbit
    rejected = {(b["day"], b["pad"], round(b["t_start_h"], 6)) for b in rule}
    for o, r in orbits.items():
        for d, v in r.items():
            day = d.rsplit("/", 1)[1]
            keep = [q for q in v["doppler_passes"] if (day, q["pad"], round(q["t_start_h"], 6)) not in rejected]
            out["days"].setdefault(day, {})[o] = dict(range_m=v["range_m"], doppler_before_mmps=v["doppler_mmps"],
                                                      doppler_after_mmps=_combine(keep),
                                                      doppler_passes=v["doppler_passes"], range_passes=v["range_passes"])
    fit_days = {L.DAYS[i].rsplit("/", 1)[1] for i in range(3)}
    out["fit_day_pass_rejected"] = any(b["day"] in fit_days for b in rule)
    out["rejected_sets_agree"] = ({(b["day"], b["pad"]) for b in rule}
                                  == {(b["day"], b["pad"]) for b in out["screening"]["LOLA"]["doppler_rejected"]})
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(screening=out["screening"], fit_day_pass_rejected=out["fit_day_pass_rejected"],
                          rejected_sets_agree=out["rejected_sets_agree"],
                          after={d: {o: v[o]["doppler_after_mmps"] for o in v} for d, v in out["days"].items()}),
                     indent=1))


if __name__ == "__main__":
    main()
