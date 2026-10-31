"""Phase F2t: where in 2009-09-17..18 F2s-1's cross-track misfit sits, and whether C_N/S_N absorb it.

Predeclaration: studies/phase17/configs/phase_f2t_short_windows_predeclaration.json.  Analysis only: F2s-1's
window fit (phase_f2s_lola_arc_structure.fit_window), no production code changes.

Modes (parts under results/phase_f2t_parts/; --summary merges them):
  --fit-all [--shard i/n] [--workers N]   12 h window fits, V0 (C_N/S_N fixed) and V1 (free)
  --summary                               decision as predeclared
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
import phase_f2r_postfit_growth as R  # noqa: E402
import phase_f2s_lola_arc_structure as S  # noqa: E402

S.PARTS = HERE.parent / "results" / "phase_f2t_parts"
OUT = HERE.parent / "results" / "phase_f2t_short_windows.json"
WIN_S = 12 * 3600.0
B_STARTS_H = (0, 6, 12, 18, 24, 30, 36)
C_STARTS_H = (60, 72)
HIGH_FACTOR, HIGH_FLOOR_M = 2.0, 1.0
ABSORB_FACTOR, NOT_ABSORB_FRAC = 1.5, 0.67
CONTROL_MAX_M = 1.0


def windows(F0) -> list:
    out = []
    for v in ("V0", "V1"):
        for grp, starts in (("B", B_STARTS_H), ("C", C_STARTS_H)):
            for h in starts:
                ws = F0 + h * 3600.0
                out.append(dict(name=f"{v}_{grp}{h:02d}", variant=v, group=grp, start_h=h, ws=ws, we=ws + WIN_S,
                                free_cn=v == "V1"))
    return out


def main_fit_all(workers):
    _, spans = R.problem()
    wins = windows(spans["F"][0])
    i, n = (map(int, sys.argv[sys.argv.index("--shard") + 1].split("/")) if "--shard" in sys.argv else (0, 1))
    for k, w in enumerate(wins):
        if k % n == i and not (S.PARTS / f"{w['name']}.json").exists():
            S.fit_window(w, workers)


def main_summary():
    _, spans = R.problem()
    wins = {w["name"]: S.read_part(w["name"]) for w in windows(spans["F"][0])}
    N = lambda name: wins[name]["fit_rms_RTN_m"][2]
    names = lambda v, g, st: [f"{v}_{g}{h:02d}" for h in st]
    b0, b1, c0n = names("V0", "B", B_STARTS_H), names("V1", "B", B_STARTS_H), names("V0", "C", C_STARTS_H)
    ok = lambda v: [x for x in wins if x.startswith(v) and wins[x]["status"] == "STATIONARY"]
    n_v = lambda v: len([x for x in wins if x.startswith(v)])
    c0 = float(np.median([N(x) for x in c0n]))
    thr = max(HIGH_FACTOR * c0, HIGH_FLOOR_M)
    high = [h for h, x in zip(B_STARTS_H, b0) if N(x) >= thr]
    res = dict(c0=c0, high_threshold_m=thr, high_start_h=high, n_stationary_V0=len(ok("V0")),
               n_stationary_V1=len(ok("V1")), control_N=[N(x) for x in c0n])
    if 2 * len(ok("V0")) < n_v("V0") or 2 * len(ok("V1")) < n_v("V1") or max(N(x) for x in c0n) > CONTROL_MAX_M:
        loc, emp = "F2T_INCONCLUSIVE", "F2T_INCONCLUSIVE"
    else:
        idx = [B_STARTS_H.index(h) for h in high]
        contiguous = bool(idx) and idx == list(range(idx[0], idx[0] + len(idx)))
        if 1 <= len(high) <= 3 and contiguous and len(B_STARTS_H) - len(high) >= 3:
            loc = "LOCALIZED"
        elif len(high) >= 5:
            loc = "SPREAD"
        elif not high:
            loc = "NOT_REPRODUCED"
        else:
            loc = "MIXED_PATTERN"
        if high:
            m0 = float(np.median([N(f"V0_B{h:02d}") for h in high]))
            m1 = float(np.median([N(f"V1_B{h:02d}") for h in high]))
            res.update(median_V0_high_N=m0, median_V1_high_N=m1)
            if m1 <= max(ABSORB_FACTOR * c0, HIGH_FLOOR_M):
                emp = "EMP_ABSORBS"
            elif m1 >= NOT_ABSORB_FRAC * m0:
                emp = "EMP_DOES_NOT_ABSORB"
            else:
                emp = "EMP_PARTIAL"
        else:
            emp = "DESCRIPTIVE_ONLY"
    out = dict(phase="F2t", predeclaration="configs/phase_f2t_short_windows_predeclaration.json",
               location_class=loc, empirical_class=emp, decision_inputs=res,
               windows=[{x: y for x, y in w.items() if x != "iterations"} | dict(n_iter=len(w["iterations"]))
                        for w in wins.values()])
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(location_class=loc, empirical_class=emp, **res), indent=1))
    for w in out["windows"]:
        print(w["name"], w["status"], "fit", np.round(w["fit_rms_RTN_m"], 2).tolist(), "E2",
              np.round(w["e2_restricted_rms_RTN_m"], 2).tolist(), "K", round(w["k_srp"], 4),
              "CN/SN", f"{w['C_N']:.2e} {w['S_N']:.2e}")


if __name__ == "__main__":
    if "--fit-all" in sys.argv:
        main_fit_all(M._workers_arg() or 8)
    elif "--summary" in sys.argv:
        main_summary()
