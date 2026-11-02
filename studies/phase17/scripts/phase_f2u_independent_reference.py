"""Phase F2u: is F2t's 09-17/18 cross-track misfit specific to LOLA? F2t's V0 window fits against the FDF orbit.

Predeclaration: studies/phase17/configs/phase_f2u_independent_reference_predeclaration.json.  Analysis only: F2s-1's
window fit with the reference switched to the LRO project's definitive orbit (NAIF lrorg_2009169_2010001_v01.bsp),
no production code changes.

Modes (parts under results/phase_f2u_parts/; --summary merges them):
  --diff                                  steps A0 + A: FDF segments, LOLA minus FDF per window (descriptive)
  --fit-all [--shard i/n] [--workers N]   step B: V0 window fits against FDF
  --summary                               decision as predeclared
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as M  # noqa: E402
M.use_historical_numerics()  # merge of main (G4e-R/TRF): keep the numerics of this phase's records
import phase_f2r_postfit_growth as R  # noqa: E402
import phase_f2s_lola_arc_structure as S  # noqa: E402
import phase_f2t_short_windows as T  # noqa: E402

S.PARTS = HERE.parent / "results" / "phase_f2u_parts"
OUT = HERE.parent / "results" / "phase_f2u_independent_reference.json"
FDF = M.F.REPO.parent / "lro_data" / "spk" / "lrorg_2009169_2010001_v01.bsp"
E_L = 1.69 - 0.28    # F2t: LOLA median B minus median C N RMS (m), as predeclared
SHARED_FRAC, SPECIFIC_FRAC, N_HIGH_MIN = 0.5, 0.25, 5
_lola = R.lola


def ref_state(et0, t, which):
    """LRO state (m, m/s) from LOLA or FDF; reloading a kernel puts it on top of SPICE's priority list."""
    import spiceypy as spice
    if which == "FDF":
        spice.furnsh(str(FDF))
    else:
        spice.unload(str(FDF))
    return _lola(et0, t)


R.lola = lambda et0, t: ref_state(et0, t, "FDF")


def v0_windows(F0):
    return [w for w in T.windows(F0) if w["variant"] == "V0"]


def segments():
    import spiceypy as spice
    h = spice.dafopr(str(FDF))
    spice.dafbfs(h)
    out = []
    while spice.daffna():
        dc, ic = spice.dafus(spice.dafgs(), 2, 6)
        out.append((float(dc[0]), float(dc[1]), int(ic[0]), int(ic[1]), int(ic[2]), int(ic[3])))
    spice.dafcls(h)
    return out


def seams_in(win, et0, seg):
    """Interior FDF segment boundaries (s from et0) inside a window."""
    b = sorted({s[0] - et0 for s in seg} | {s[1] - et0 for s in seg})
    return [x for x in b if win["ws"] < x < win["we"]]


def main_diff():
    import spiceypy as spice
    pb, spans = R.problem()
    et0 = pb.setup.et0
    seg = segments()
    out = dict(fdf_file=FDF.name, fdf_sha256=hashlib.sha256(FDF.read_bytes()).hexdigest(),
               n_segments=len(seg), segment_kinds=sorted({s[2:] for s in seg}), windows=[])
    for w in v0_windows(spans["F"][0]):
        ts = np.arange(w["ws"], w["we"] + 1e-6, R.DT_S)
        lo, fd = ref_state(et0, ts, "LOLA"), ref_state(et0, ts, "FDF")
        d = S.to_rtn(lo[:, :3] - fd[:, :3], lo)
        # argument of latitude from LOLA: angle of r in the orbit plane from the ascending node
        r, v = lo[:, :3], lo[:, 3:]
        h = np.cross(r, v)
        node = np.cross([0.0, 0.0, 1.0], h)
        node /= np.linalg.norm(node, axis=1)[:, None]
        u = np.arctan2(np.sum(np.cross(node, r) * h, axis=1) / np.linalg.norm(h, axis=1), np.sum(node * r, axis=1))
        A = np.column_stack([np.ones_like(u), np.cos(u), np.sin(u)])
        c = np.linalg.lstsq(A, d[:, 2], rcond=None)[0]
        out["windows"].append(dict(name=w["name"][3:], seams_utc=[spice.et2utc(et0 + x, "ISOC", 0)
                                                                for x in seams_in(w, et0, seg)],
                                   rms_RTN_m=np.sqrt(np.mean(d ** 2, axis=0)).tolist(),
                                   n_fit_c_a_b_m=c.tolist(), n_1rev_amp_m=float(np.hypot(c[1], c[2]))))
    S.write_part("A", out)
    print(json.dumps({x: out[x] for x in ("fdf_sha256", "n_segments", "segment_kinds")}))
    for w in out["windows"]:
        print(w["name"], "seams", w["seams_utc"], "LOLA-FDF RTN", np.round(w["rms_RTN_m"], 2).tolist(),
              "N 1/rev", round(w["n_1rev_amp_m"], 2))


def main_fit_all(workers):
    _, spans = R.problem()
    i, n = (map(int, sys.argv[sys.argv.index("--shard") + 1].split("/")) if "--shard" in sys.argv else (0, 1))
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for k, w in enumerate(v0_windows(spans["F"][0])):
        if (w["name"] == only if only else k % n == i) and not (S.PARTS / f"{w['name']}.json").exists():
            S.fit_window(w, workers)


def classify(N, b, c):
    c_F = float(np.median([N[x] for x in c]))
    m_F = float(np.median([N[x] for x in b]))
    E_F = m_F - c_F
    n_high = sum(N[x] >= max(2 * c_F, 1.0) for x in b)
    if c_F > E_L:
        cls = "F2U_UNRESOLVED"
    elif E_F >= SHARED_FRAC * E_L and n_high >= N_HIGH_MIN:
        cls = "H_SHARED"
    elif E_F <= SPECIFIC_FRAC * E_L:
        cls = "H_LOLA_SPECIFIC"
    else:
        cls = "MIXED"
    return cls, dict(c_F=c_F, m_F=m_F, E_F=E_F, E_L=E_L, n_high=int(n_high))


def main_summary():
    pb, spans = R.problem()
    et0 = pb.setup.et0
    seg = segments()
    ws = v0_windows(spans["F"][0])
    wins = {w["name"]: S.read_part(w["name"]) for w in ws}
    N = {x: wins[x]["fit_rms_RTN_m"][2] for x in wins}
    b = [w["name"] for w in ws if w["group"] == "B"]
    c = [w["name"] for w in ws if w["group"] == "C"]
    n_ok = sum(wins[x]["status"] == "STATIONARY" for x in wins)
    if 2 * n_ok < len(wins):
        cls, res = "F2U_INCONCLUSIVE", {}
    else:
        cls, res = classify(N, b, c)
    seamy = {w["name"] for w in ws if seams_in(w, et0, seg)}
    bs, cs = [x for x in b if x not in seamy], [x for x in c if x not in seamy]
    seam = dict(windows_with_seam=sorted(seamy), kept_B=bs, kept_C=cs)
    if cls != "F2U_INCONCLUSIVE" and bs and cs:
        seam["class"], seam["inputs"] = classify(N, bs, cs)
        seam["SEAM_SENSITIVE"] = seam["class"] != cls
    out = dict(phase="F2u", predeclaration="configs/phase_f2u_independent_reference_predeclaration.json",
               reference=FDF.name, n_stationary=n_ok, f2u_class=cls, decision_inputs=res, seam_check=seam,
               step_A=S.read_part("A"),
               windows=[{x: y for x, y in w.items() if x != "iterations"} | dict(n_iter=len(w["iterations"]))
                        for w in wins.values()])
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(f2u_class=cls, **res, seam_check=seam), indent=1))
    for w in out["windows"]:
        print(w["name"], w["status"], "fit vs FDF", np.round(w["fit_rms_RTN_m"], 2).tolist(), "E2 vs FDF",
              np.round(w["e2_restricted_rms_RTN_m"], 2).tolist(), "K", round(w["k_srp"], 4))


if __name__ == "__main__":
    if "--diff" in sys.argv:
        main_diff()
    elif "--fit-all" in sys.argv:
        main_fit_all(M._workers_arg() or 8)
    elif "--summary" in sys.argv:
        main_summary()
