"""Compare the TRF (G4e-R numerics, exact-epoch stations) F2g/h/i refits with the historical legacy records.

Historical: studies/phase17/results/phase_f2g_lro_empirical.json, phase_f2h_lro_alongtrack_empirical.json (legacy LM,
rtol 1e-10, empirical FD step 1e-9, quantised station epochs). F2i has no historical record.
Refit: outputs/numerical-core/trf/<same names> and phase_f2i_lro_basin_robustness.json.

usage: phase_f2_refit_compare.py   (prints a Markdown table and writes outputs/numerical-core/trf/phase_f2_refit_compare.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase_f2_lro_od as F2  # noqa: E402

PAIRS = [("F2g S1", "phase_f2g_lro_empirical.json", "S1", "S1"),
         ("F2g S2", "phase_f2g_lro_empirical.json", "S2", "S2"),
         ("F2h S1", "phase_f2h_lro_alongtrack_empirical.json", "S1", "S1"),
         ("F2i S1R", None, None, "S1R"),
         ("F2i S2", None, None, "S2")]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _brief(r):
    if r is None:
        return None
    p = r.get("prediction") or {}
    return dict(status=r["status"], iterations=r.get("iterations"), range_rms_m=r["postfit_rms_m"],
                doppler_rms_mmps=r["doppler_postfit_rms_mmps"], k_srp=r["k_srp"], k_srp_sigma=r.get("k_srp_sigma"),
                empirical_mps2=r.get("empirical_mps2"), biases_m=r.get("biases_m"),
                fit_ric_rms_m=r.get("ric_rms_m"), fit_total_rms_m=r.get("total_rms_m"),
                pred_range_rms_m=p.get("range_rms_m"), pred_doppler_rms_mmps=p.get("doppler_rms_mmps"),
                pred_ric_rms_m=p.get("ric_rms_m"), elapsed_s=r.get("elapsed_s"))


def main():
    hist_dir, trf_dir = HERE.parent / "results", F2._output_dir("trf")
    out = {}
    for label, name, hkey, tkey in PAIRS:
        tname = name or "phase_f2i_lro_basin_robustness.json"
        h = _load(hist_dir / name) if name else None
        t = _load(trf_dir / tname)
        run = t["runs"][tkey] if t else None
        if run is None and name is None:   # F2i unmerged: S1R not run (on TRF it repeats F2h S1), read the part
            run = _load(trf_dir / f"phase_f2i_part_{tkey.lower()}.json")
        out[label] = dict(historical=_brief(h["runs"][hkey]) if h else None,
                          refit=_brief(run),
                          historical_gates=h["gates"] if h else None, refit_gates=t["gates"] if t else None,
                          historical_G2_m=h.get("G2_start_difference_m") if h else None,
                          refit_G2_m=t.get("G2_start_difference_m") if t else None)
    (trf_dir / "phase_f2_refit_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def f(v, fmt):
        return "—" if v is None else (" / ".join(format(x, fmt) for x in v) if isinstance(v, list) else format(v, fmt))
    rows = [("status", "status", "s"), ("range post-fit [m]", "range_rms_m", ".3f"),
            ("Doppler post-fit [mm/s]", "doppler_rms_mmps", ".2f"), ("K_SRP", "k_srp", ".4f"),
            ("K_SRP σ", "k_srp_sigma", ".1e"), ("fit vs LOLA R/A/C [m]", "fit_ric_rms_m", ".1f"),
            ("pred. range RMS [m]", "pred_range_rms_m", ".2f"), ("pred. Doppler RMS [mm/s]", "pred_doppler_rms_mmps", ".2f"),
            ("pred. vs LOLA R/A/C [m]", "pred_ric_rms_m", ".1f"), ("elapsed [h]", "elapsed_s", ".1f")]
    for label, c in out.items():
        print(f"\n### {label}\n\n| | historical (legacy) | refit (TRF, G4e-R) |\n|---|---|---|")
        for title, key, fmt in rows:
            hv = (c["historical"] or {}).get(key)
            tv = (c["refit"] or {}).get(key)
            if key == "elapsed_s":
                hv, tv = (None if hv is None else hv / 3600), (None if tv is None else tv / 3600)
            print(f"| {title} | {f(hv, fmt)} | {f(tv, fmt)} |")
        for side in ("historical", "refit"):
            e = (c[side] or {}).get("empirical_mps2")
            if e:
                print(f"| empirical ({side}) | " + ", ".join(f"{k} {v:+.2e}" for k, v in e.items()) + " | |")
        print(f"| gates | {c['historical_gates']} (G2 {f(c['historical_G2_m'], '.1f')} m) "
              f"| {c['refit_gates']} (G2 {f(c['refit_G2_m'], '.1f')} m) |")


if __name__ == "__main__":
    main()
