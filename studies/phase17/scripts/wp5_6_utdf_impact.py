"""F5 WP5.6: UTDF hardening on the LRO development days (U_G7(b), U_G8(a), U_G8(c)).

Predeclaration: studies/phase17/configs/wp5_6_utdf_hardening_predeclaration.json.

* U_G8(a): every record decodes to the same values with this branch's reader as with main's (953f567) in
  every field main's UTDFRecord has.
* U_G7(b): on the Doppler intervals F1c forms (phase_f1c_lro_doppler.read_doppler / intervals: doppler_valid,
  PADs 103/105/126, same PAD, 0 < dt <= 60 s), the new record-derived range rate equals F1c's expression
  -C (dn/dt - BIAS_HZ) / (2 f_T TURNAROUND M_MULT) to <= 4 ulp.
* U_G8(c), information: the F1c intervals check_doppler_counts flags, by code, each RATE_DISCONTINUITY with
  its UTC and deviation, and whether it is among F2o's edited intervals.

Only development days are read, each through lro_data_partition.require_readable_day; the hold-out is never
opened. No study script calls the new checks, so nothing here moves a study result.
"""
from __future__ import annotations

import collections
import dataclasses
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import phase_f1c_lro_doppler as D  # noqa: E402
import phase_f1_lro_residuals as F  # noqa: E402
from lro_data_partition import require_readable_day  # noqa: E402

from lunar_od import utdf as U  # noqa: E402

PREDECLARATION = "studies/phase17/configs/wp5_6_utdf_hardening_predeclaration.json"
OUT = HERE.parent / "results" / "wp5_6_utdf_impact.json"
MAIN_SHA = "953f567"
DAYS = ("2009260", "2009261", "2009262", "2009263", "2009285")
F2O = HERE.parent / "results" / "phase_f2o_doppler_editing.json"
EPS = 2.220446049250313e-16
MAX_DOPPLER_MPS = 3000.0   # phase_f2_lro_od.MAX_DOPPLER_MPS (F2e A1), copied to avoid importing the F2 driver


def _main_reader():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{MAIN_SHA}:src/lunar_od/utdf.py"],
                         check=True, capture_output=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "utdf_main.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("utdf_main", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["utdf_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    data = REPO.parent / "lro_data"
    old = _main_reader()
    old_fields = [f.name for f in dataclasses.fields(old.UTDFRecord)]
    f2o = json.loads(F2O.read_text(encoding="utf-8"))
    edited = {(e["day"], e["pad"], e["utc"]) for e in f2o["orbits"]["F2k"]["edited"]}

    g8a = dict(records=0, files=0, mismatches=[])
    g7b = dict(intervals=0, exact=0, max_rel=0.0, max_ulp=0.0, missing=0)
    days = {}
    for day in DAYS:
        day_dir = f"trk/lro_no_01/{day}"
        require_readable_day(day_dir)
        for f in sorted((data / day_dir).glob("*.trk")):
            g8a["files"] += 1
            new_recs, old_recs = U.read_utdf(f), old.read_utdf(f)
            for rn, ro in zip(new_recs, old_recs, strict=True):
                g8a["records"] += 1
                for name in old_fields:
                    if getattr(rn, name) != getattr(ro, name):
                        g8a["mismatches"].append(f"{f.name}: {name}")

        rows = D.read_doppler(data, day_dir)                       # F1c's rows: (pad, record), sorted
        t = [(r.utc - rows[0][1].utc).total_seconds() for _, r in rows]
        f1c = D.intervals(rows, t)
        per_pad = collections.defaultdict(list)
        for k, (pad, r) in enumerate(rows):
            per_pad[pad].append(k)
        checked = {}
        for pad, ks in per_pad.items():
            for iv in U.check_doppler_counts([rows[k][1] for k in ks], file_pad=pad):
                checked[(ks[iv.index_a], ks[iv.index_b])] = iv
        codes = collections.Counter()
        disc, flagged_edited, untested = [], 0, 0
        # the studies' own interval filter (phase_f2_lro_od: F2e A1 |obs| > MAX_DOPPLER_MPS, F2p DOPPLER_EDITS)
        filt = collections.Counter()
        for i, j in f1c:
            pad, ra = rows[i]
            rb = rows[j][1]
            iv = checked.get((i, j))
            if iv is None:
                g7b["missing"] += 1
                continue
            dn = rb.doppler_count_raw - ra.doppler_count_raw
            a = dn / (rb.utc - ra.utc).total_seconds()
            ref = -F.C * (a - D.BIAS_HZ) / (2.0 * ra.transmit_frequency_hz * D.TURNAROUND * D.M_MULT)
            g7b["intervals"] += 1
            if iv.range_rate_mps == ref:
                g7b["exact"] += 1
            rel = abs(iv.range_rate_mps - ref) / abs(ref)
            if U.COUNT_ROLLOVER not in iv.codes:
                g7b["max_rel_excluding_rollover"] = max(g7b.get("max_rel_excluding_rollover", 0.0), rel)
            g7b["max_rel"] = max(g7b["max_rel"], rel)
            g7b["max_ulp"] = max(g7b["max_ulp"], rel / EPS)
            for c in iv.codes:
                codes[c] += 1
            study_drop = abs(ref) > MAX_DOPPLER_MPS
            count_coded = bool(set(iv.codes) - {U.RATE_DISCONTINUITY, U.GAP})
            filt[("study_drops" if study_drop else "study_keeps") + "|" +
                 ("count_or_record_coded" if count_coded else "not_count_coded")] += 1
            if not iv.codes and iv.deviation_mps != iv.deviation_mps:
                untested += 1
            key = (day, pad, ra.utc.isoformat())
            if iv.codes and key in edited:
                flagged_edited += 1
            if U.RATE_DISCONTINUITY in iv.codes:
                disc.append(dict(pad=pad, utc=ra.utc.isoformat(), deviation_mmps=iv.deviation_mps * 1e3,
                                 in_f2o_edited=key in edited))
        days[day] = dict(f1c_intervals=len(f1c), flagged=sum(1 for i, j in f1c if checked.get((i, j)) and
                                                          checked[(i, j)].codes),
                         codes=dict(codes), rate_discontinuities=disc, untested_for_discontinuity=untested,
                         study_filter_vs_codes=dict(filt),
                         f2o_edited_on_day=sum(1 for e in edited if e[0] == day),
                         f2o_edited_also_flagged=flagged_edited)
    g8a["PASS"] = not g8a["mismatches"] and g8a["records"] > 0
    g7b["note"] = ("intervals that differ are listed by code in U_G8c (COUNT_ROLLOVER intervals are unwrapped "
                   "by the new reader; F1c's expression uses the wrapped count)")
    g7b["PASS"] = g7b["intervals"] > 0 and g7b["missing"] == 0 and g7b["max_rel"] <= 4 * EPS
    out = dict(predeclaration=PREDECLARATION, main_reader=MAIN_SHA, days_read=list(DAYS),
               U_G8a_decode_identity=dict(g8a, mismatches=g8a["mismatches"][:50]),
               U_G7b_f1c_range_rate=g7b, U_G8c_impact=days)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(U_G8a=g8a["PASS"], records=g8a["records"], U_G7b=g7b,
                          impact={d: dict(n=v["f1c_intervals"], flagged=v["flagged"], codes=v["codes"],
                                          disc=[(x["pad"], x["utc"], round(x["deviation_mmps"], 1),
                                                 x["in_f2o_edited"]) for x in v["rate_discontinuities"]])
                                  for d, v in days.items()}), indent=1))


if __name__ == "__main__":
    main()
