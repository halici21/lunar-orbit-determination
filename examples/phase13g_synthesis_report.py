"""Phase 13G-d — gravity model synthesis and truth/estimator recommendation.

Reads the CUMULATIVE Phase 13G store (results/phase13g/phase13g_store.json)
produced by the committed campaign script and generates a deterministic
synthesis: evidence-labelled claims, truth-model recommendation, estimator
physics/software recommendations, threshold policy, seven-day decision matrix
and the thesis model hierarchy.

DESIGN GUARANTEES (structural, tested by an import-surface scan):
- stdlib only — no ``lunar_od`` import, no SPICE, no gravity data, and no
  import of the campaign scripts (no hidden dependency);
- no propagation: this script only reads JSON and writes MD/JSON/CSV;
- deterministic: the report timestamp is the STORE's ``generated_utc`` (never
  ``now()``), and the store file's SHA-256 is recorded as provenance;
- single data source: the store JSON only (the campaign CSVs are derived
  artifacts and are deliberately not read).

EVIDENCE POLICY (per the accepted 13G-d plan corrections):
- every major claim carries one of six evidence levels;
- non-adjacent truth-relative errors are NEVER presented as measured: they
  are linear sums of successive nested-model differences and are labelled
  "derived conservative envelope from successive nested-model differences"
  (triangle inequality); RSS aggregation is not used anywhere;
- the 10 m/day truncation threshold is an engineering screening heuristic,
  not a validated OD requirement;
- seven-day runtime figures are rough linear projections from one-day
  measurements, not measured values;
- diagnostic / intentionally-wrong rows never feed physics recommendations.

Outputs (git-ignored):
  results/phase13g/phase13g_synthesis_report.md
  results/phase13g/phase13g_model_recommendations.json
  results/phase13g/phase13g_recommendation_tables.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "results" / "phase13g" / "phase13g_store.json"
DEFAULT_OUT = ROOT / "results" / "phase13g"

MD_NAME = "phase13g_synthesis_report.md"
JSON_NAME = "phase13g_model_recommendations.json"
CSV_NAME = "phase13g_recommendation_tables.csv"

# ---------------------------------------------------------------------------
# Evidence levels and science-review labels (fixed vocabulary)
# ---------------------------------------------------------------------------
EV_VERIFIED = "verified implementation result"
EV_LEVEL1 = "Level-1 internal model-vs-model validation result"
EV_THRESHOLD = "threshold demonstrated"
EV_RECOMMEND = "recommendation from available evidence"
EV_HYPOTHESIS = "working hypothesis"
EV_NOT_VALIDATED = "not yet validated"
EVIDENCE_LEVELS = (EV_VERIFIED, EV_LEVEL1, EV_THRESHOLD, EV_RECOMMEND,
                   EV_HYPOTHESIS, EV_NOT_VALIDATED)

ENVELOPE_LABEL = ("derived conservative envelope from successive "
                  "nested-model differences")

SR_TRUTH = ("Accept with caveats — primary truth recommendation from "
            "available Level-1 internal model-vs-model evidence")
SR_EST_PHYSICS = ("Provisional recommendation — filter-in-the-loop "
                  "validation required")
SR_EST_SOFTWARE = "Not yet ready for harmonics-enabled production scenarios"

THRESHOLD_M_PER_DAY = 10.0
LADDER = (8, 16, 32, 64, 128)
CLOSURE_QUALIFICATION = "matched per-case 128->256 closure not run"

LEVEL1_DISCLAIMER = (
    "All numbers in this synthesis are Level-1 internal model-vs-model "
    "comparisons under one shared integrator, one shared DE421-derived .mat "
    "translational ephemeris and identical tolerances; nothing here is an "
    "accuracy claim against external truth."
)

THRESHOLD_POLICY_TEXT = (
    "10 m/day is an engineering screening heuristic selected to compare "
    "truncation levels consistently across the Phase 13G campaign. It is "
    "not a validated OD requirement, estimator-error threshold, or mission "
    "navigation requirement. A final accuracy threshold can only be defined "
    "together with measurement noise, tracking geometry, arc duration, the "
    "required navigation accuracy, and observed estimator behavior."
)

SEVEN_DAY_WINDOW_TEXT = (
    "The one-day ladder results suggest that the truncation ordering is "
    "likely to remain useful over longer windows, but accumulated "
    "along-track phase, long-period element evolution, and possible beating "
    "with the body-fixed gravity field may change the quantitative "
    "separation. This requires a selected multi-day confirmation. The "
    "expectation that the nmax ranking is preserved over seven days is a "
    "working hypothesis, not a demonstrated result."
)

RUNTIME_PROJECTION_LABEL = ("rough linear runtime projection from one-day "
                            "measurements; approximate, campaign not run")


class StoreError(RuntimeError):
    """Clear, contextual error for any store access/validation problem."""


# ---------------------------------------------------------------------------
# Store access / validation layer (no bare dictionary digging elsewhere)
# ---------------------------------------------------------------------------
STAGE_COMMANDS = {
    "baseline": "python examples/phase13g_gravity_orbit_effects.py --baseline",
    "compare": "python examples/phase13g_gravity_orbit_effects.py --compare",
    "frames": "python examples/phase13g_gravity_orbit_effects.py --frames",
    "sensitivity": "python examples/phase13g_gravity_orbit_effects.py --sensitivity",
}


def load_store(path: Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_file():
        raise StoreError(
            f"Phase 13G store not found: {path}. Run the campaign stages "
            f"first (see examples/phase13g_gravity_orbit_effects.py); this "
            f"synthesis never runs propagation itself."
        )
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        store = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"store {path} is not valid JSON: {exc}") from exc
    if not isinstance(store, dict):
        raise StoreError(f"store {path} must contain a JSON object.")
    return store, sha256


def require_stage(store: dict, stage: str) -> dict:
    data = store.get(stage)
    if not data:
        raise StoreError(
            f"required stage '{stage}' is missing from the store; "
            f"regenerate it with: {STAGE_COMMANDS.get(stage, '?')} "
            f"(NOT run automatically by this synthesis)."
        )
    return data


def get_window(stage_data: dict, stage: str, window: str,
               case_id: str | None = None) -> dict:
    if case_id is not None:
        cases = stage_data.get("cases") or {}
        case = cases.get(case_id)
        if case is None:
            raise StoreError(
                f"case '{case_id}' missing from stage '{stage}' "
                f"(available: {sorted(cases)})."
            )
        windows = case.get("windows") or {}
    else:
        windows = stage_data.get("windows") or {}
    wdata = windows.get(window)
    if wdata is None:
        raise StoreError(
            f"window '{window}' missing from stage '{stage}'"
            + (f", case '{case_id}'" if case_id else "")
            + f" (available: {sorted(windows)})."
        )
    return wdata


def _context(stage, window, case_id, name):
    where = f"stage '{stage}', window '{window}'"
    if case_id:
        where += f", case '{case_id}'"
    return f"{where}, comparison '{name}'"


def find_comparison(rows: list, name: str, *, stage: str, window: str,
                    case_id: str | None = None,
                    allow_diagnostic: bool = False) -> dict:
    matches = [r for r in rows if r.get("comparison") == name]
    ctx = _context(stage, window, case_id, name)
    if not matches:
        raise StoreError(f"comparison not found: {ctx} "
                         f"(available: {sorted({r.get('comparison') for r in rows})}).")
    if len(matches) > 1:
        raise StoreError(f"duplicate comparison rows ({len(matches)}) for {ctx}.")
    row = matches[0]
    if (row.get("diagnostic") or row.get("intentionally_wrong")) and not allow_diagnostic:
        raise StoreError(
            f"{ctx} is flagged diagnostic/intentionally_wrong and must not "
            f"feed a physics recommendation (pass allow_diagnostic=True only "
            f"for the diagnostics section)."
        )
    value = row.get("final_dpos_m")
    if value is None or not math.isfinite(float(value)):
        raise StoreError(f"non-finite or missing final_dpos_m in {ctx}: {value!r}.")
    return row


def maybe_comparison(rows, name, **kw):
    try:
        return find_comparison(rows, name, **kw)
    except StoreError:
        return None


def find_run(rows: list, run: str, *, stage: str, window: str,
             case_id: str | None = None) -> dict:
    matches = [r for r in rows if r.get("run") == run]
    ctx = _context(stage, window, case_id, f"run '{run}'")
    if not matches:
        raise StoreError(f"run not found: {ctx}.")
    if len(matches) > 1:
        raise StoreError(f"duplicate run rows for {ctx}.")
    return matches[0]


def find_element(rows: list, run: str, element: str, *, stage: str,
                 window: str, case_id: str | None = None) -> dict:
    matches = [r for r in rows if r.get("run") == run and r.get("element") == element]
    ctx = _context(stage, window, case_id, f"element '{element}' of run '{run}'")
    if not matches:
        raise StoreError(f"element row not found: {ctx}.")
    if len(matches) > 1:
        raise StoreError(f"duplicate element rows for {ctx}.")
    return matches[0]


# ---------------------------------------------------------------------------
# Claims, nmax recommendation and the conservative envelope
# ---------------------------------------------------------------------------
def claim(text: str, value, unit: str, level: str, source: str,
          case_id: str | None = None, window: str | None = None,
          comparison: str | None = None, qualification: str | None = None,
          limitations: str | None = None) -> dict:
    if level not in EVIDENCE_LEVELS:
        raise ValueError(f"unknown evidence level {level!r}.")
    out = {"claim": text, "value": value, "unit": unit,
           "evidence_level": level, "source": source}
    for key, val in (("case_id", case_id), ("window", window),
                     ("comparison", comparison), ("qualification", qualification),
                     ("limitations", limitations)):
        if val is not None:
            out[key] = val
    return out


def ladder_step(rows, lo: int, hi: int, *, stage, window, case_id=None):
    row = maybe_comparison(rows, f"ladder_{lo}_vs_{hi}", stage=stage,
                           window=window, case_id=case_id)
    return None if row is None else float(row["final_dpos_m"])


def nmax_recommendation(rows, *, stage: str, window: str,
                        case_id: str | None = None,
                        threshold: float = THRESHOLD_M_PER_DAY) -> dict:
    """Smallest ladder N whose N->2N step is below the screening threshold.

    The step full@2N - full@N is the truncation-error proxy of full@N.  When
    no step closes the threshold, nmax=128 is a RECOMMENDATION from available
    evidence with the explicit qualification that the matched per-case
    128->256 closure was not run.
    """
    steps = {}
    for lo, hi in zip(LADDER[:-1], LADDER[1:]):
        step = ladder_step(rows, lo, hi, stage=stage, window=window, case_id=case_id)
        if step is not None:
            steps[(lo, hi)] = step
            if step < threshold:
                return {"case_id": case_id, "nmax": lo,
                        "label": EV_THRESHOLD,
                        "evidence_step": f"{lo}->{hi}",
                        "evidence_step_m_per_day": step,
                        "qualification": None, "steps": steps}
    last = steps.get((64, 128))
    return {"case_id": case_id, "nmax": LADDER[-1],
            "label": EV_RECOMMEND,
            "evidence_step": "64->128",
            "evidence_step_m_per_day": last,
            "qualification": CLOSURE_QUALIFICATION, "steps": steps}


def conservative_envelope(rows, from_n: int, *, stage: str, window: str,
                          case_id: str | None = None,
                          extra_base_m: float = 0.0) -> dict:
    """Linear sum of successive nested-model differences from ``from_n`` up
    to nmax=128 (triangle inequality => a conservative upper bound for the
    same case/epoch/metric).  This is NOT a measured truth-relative
    difference and is labelled accordingly.  No RSS aggregation is used:
    root-sum-square would require an independence/orthogonality assumption
    that has not been demonstrated, and it is not a mathematical bound.
    """
    idx = LADDER.index(from_n)
    parts, total = [], float(extra_base_m)
    for lo, hi in zip(LADDER[idx:-1], LADDER[idx + 1:]):
        step = ladder_step(rows, lo, hi, stage=stage, window=window, case_id=case_id)
        if step is None:
            raise StoreError(
                f"cannot build conservative envelope for nmax={from_n}: "
                f"ladder step {lo}->{hi} missing in "
                f"{_context(stage, window, case_id, 'ladder')}."
            )
        parts.append({"step": f"{lo}->{hi}", "m_per_day": step})
        total += step
    return {"from_nmax": from_n, "to_nmax": LADDER[-1],
            "envelope_m_per_day": total, "parts": parts,
            "label": ENVELOPE_LABEL}


# ---------------------------------------------------------------------------
# Synthesis assembly
# ---------------------------------------------------------------------------
SENSITIVITY_CASE_META = {
    "S1_alt100_i45": ("altitude", "100 km circular, i=45 deg"),
    "S2_alt200_i45": ("altitude", "200 km circular, i=45 deg"),
    "S3_alt500_i45": ("altitude", "500 km circular, i=45 deg"),
    "S4_alt100_i0": ("inclination", "100 km circular, equatorial"),
    "S5_alt100_i30": ("inclination", "100 km circular, i=30 deg"),
    "S6_alt100_i60": ("inclination", "100 km circular, i=60 deg"),
    "S7_alt100_i90": ("inclination", "100 km circular, polar"),
    "S8_ecc80x500_i45": ("eccentric", "80x500 km eccentric, i=45 deg"),
}
GL_CASES = ("S1_alt100_i45", "S7_alt100_i90", "S8_ecc80x500_i45")


def build_synthesis(store: dict, store_sha256: str, store_path: Path) -> dict:
    baseline = require_stage(store, "baseline")
    sensitivity = require_stage(store, "sensitivity")
    compare = store.get("compare")          # optional stages: sections shrink
    frames = store.get("frames")

    b_day1 = get_window(baseline, "baseline", "day1")
    b_comp = b_day1["comparisons"]

    def bval(name, **kw):
        return float(find_comparison(b_comp, name, stage="baseline",
                                     window="day1", **kw)["final_dpos_m"])

    claims: list[dict] = []
    baseline_numbers = {
        "j2_effect": bval("j2_effect"),
        "j2only_model_error": bval("j2only_model_error"),
        "c20_bridge": bval("c20_bridge"),
        "c22_effect": bval("c22_effect"),
        "tesseral_full_minus_zonal": bval("tesseral_from_full_minus_zonal"),
        # the direct tesseral-only run is a DIAGNOSTIC decomposition (flagged
        # in the campaign store); it is quoted descriptively for the coupling
        # statement and never feeds a recommendation
        "tesseral_direct": bval("tesseral_direct", allow_diagnostic=True),
        "ladder_64_vs_128": bval("ladder_64_vs_128"),
    }
    claims.append(claim(
        "J2-only vs full@64 trajectory separation, baseline LLO (day 1)",
        round(baseline_numbers["j2only_model_error"], 1), "m/day", EV_LEVEL1,
        "baseline/day1/j2only_model_error", window="day1",
        comparison="j2only_model_error"))
    claims.append(claim(
        "C20-only vs classical-J2 separation (C20/J2 value, reference-radius "
        "and GM pairing differences; numerical proximity to the independent "
        "mean-pole frame diagnostic is coincidental and must not be treated "
        "as mutual validation)",
        round(baseline_numbers["c20_bridge"], 1), "m/day", EV_LEVEL1,
        "baseline/day1/c20_bridge", window="day1", comparison="c20_bridge"))

    # element physics (baseline day1, J2-only vs full@64)
    b_elem = b_day1["elements"]
    argp_j2 = find_element(b_elem, "v1_j2", "argp_rad", stage="baseline", window="day1")
    argp_full = find_element(b_elem, "v6_full64", "argp_rad", stage="baseline", window="day1")
    ecc_j2 = find_element(b_elem, "v1_j2", "e", stage="baseline", window="day1")
    ecc_full = find_element(b_elem, "v6_full64", "e", stage="baseline", window="day1")
    element_findings = {
        "argp_drift_j2_rad_per_day": argp_j2["drift_per_day"],
        "argp_drift_full64_rad_per_day": argp_full["drift_per_day"],
        "argp_sign_reversal": (argp_j2["drift_per_day"] * argp_full["drift_per_day"]) < 0,
        "e_drift_j2_per_day": ecc_j2["drift_per_day"],
        "e_drift_full64_per_day": ecc_full["drift_per_day"],
        "inclination_note": (
            "inclination drift was nearly common between the J2-only and "
            "full-harmonics cases, consistent with a shared non-harmonic "
            "contribution such as third-body dynamics; this attribution was "
            "not isolated with a third-body-on/off experiment"),
        "inclination_evidence": EV_HYPOTHESIS,
    }

    # per-case sensitivity table --------------------------------------------
    case_rows = []
    for case_id, (axis, label) in SENSITIVITY_CASE_META.items():
        w = get_window(sensitivity, "sensitivity", "day1", case_id=case_id)
        rows = w["comparisons"]
        rec = nmax_recommendation(rows, stage="sensitivity", window="day1",
                                  case_id=case_id)
        j2row = find_comparison(rows, "j2only_model_error", stage="sensitivity",
                                window="day1", case_id=case_id)
        entry = {
            "case_id": case_id, "axis": axis, "description": label,
            "j2only_error_m_per_day": float(j2row["final_dpos_m"]),
            "recommended_nmax": rec["nmax"],
            "evidence_level": rec["label"],
            "evidence_step": rec["evidence_step"],
            "evidence_step_m_per_day": rec["evidence_step_m_per_day"],
            "qualification": rec["qualification"],
        }
        if case_id == "S8_ecc80x500_i45":
            entry["perilune_ratio_high_degree"] = j2row.get("perilune_ratio")
        case_rows.append(entry)
        claims.append(claim(
            f"candidate truth degree for {label}",
            rec["nmax"], "nmax", rec["label"],
            f"sensitivity/{case_id}/day1/ladder_{rec['evidence_step']}",
            case_id=case_id, window="day1",
            comparison=f"ladder_{rec['evidence_step']}",
            qualification=rec["qualification"]))

    # supporting 128->256 closures: selected cross-checks only ---------------
    supporting_closures = []
    if compare is not None:
        c_day1 = get_window(compare, "compare", "day1")
        row = maybe_comparison(c_day1["comparisons"], "g1800_ladder_128_vs_256",
                               stage="compare", window="day1")
        if row is not None:
            supporting_closures.append({
                "source": "compare/day1/g1800_ladder_128_vs_256 (baseline orbit, GL1800F)",
                "value_m_per_day": float(row["final_dpos_m"]),
                "role": "selected cross-check only",
            })
        xm = maybe_comparison(c_day1["comparisons"], "cross_model_nmax64",
                              stage="compare", window="day1")
        cross_model_baseline = None if xm is None else float(xm["final_dpos_m"])
    else:
        cross_model_baseline = None
    s8_rows = get_window(sensitivity, "sensitivity", "day1",
                         case_id="S8_ecc80x500_i45")["comparisons"]
    s8_row = maybe_comparison(s8_rows, "g1800_ladder_128_vs_256",
                              stage="sensitivity", window="day1",
                              case_id="S8_ecc80x500_i45")
    if s8_row is not None:
        supporting_closures.append({
            "source": "sensitivity/S8/day1/g1800_ladder_128_vs_256 (GL1800F)",
            "value_m_per_day": float(s8_row["final_dpos_m"]),
            "role": "selected cross-check only",
        })

    cross_model = {"baseline_m_per_day": cross_model_baseline, "cases": {}}
    for case_id in GL_CASES:
        rows = get_window(sensitivity, "sensitivity", "day1", case_id=case_id)["comparisons"]
        row = maybe_comparison(rows, "cross_model_nmax64", stage="sensitivity",
                               window="day1", case_id=case_id)
        if row is not None:
            cross_model["cases"][case_id] = float(row["final_dpos_m"])
    cross_model["interpretation"] = (
        "metre-per-day agreement between GRGM660PRIM (DE421 PA) and GL1800F "
        "(DE440 PA) supports robustness of the conclusions against the "
        "gravity-model choice; it is a selected cross-check, not external "
        "validation.")

    # frame diagnostics (allow_diagnostic; never feeds recommendations) ------
    diagnostics = []
    if frames is not None:
        f_day1 = get_window(frames, "frames", "day1")
        for name in ("frame_wrongpair_n64", "frame_frozen_n64", "frame_meanpole_n64"):
            row = maybe_comparison(f_day1["comparisons"], name, stage="frames",
                                   window="day1", allow_diagnostic=True)
            if row is not None:
                diagnostics.append({"comparison": name,
                                    "value_m_per_day": float(row["final_dpos_m"]),
                                    "intentionally_wrong": True})

    # estimator physics table (measured + conservative envelopes) ------------
    est_cases = ("S1_alt100_i45", "S7_alt100_i90", "S8_ecc80x500_i45")
    estimator_physics = []
    for case_id in est_cases:
        w = get_window(sensitivity, "sensitivity", "day1", case_id=case_id)
        rows = w["comparisons"]
        j2 = float(find_comparison(rows, "j2only_model_error",
                                   stage="sensitivity", window="day1",
                                   case_id=case_id)["final_dpos_m"])
        step_64_128 = ladder_step(rows, 64, 128, stage="sensitivity",
                                  window="day1", case_id=case_id)
        entries = [{
            "candidate": "J2-only",
            "vs": "full@64 (measured) + 64->128 step (envelope to 128)",
            "measured_vs_full64_m_per_day": j2,
            "envelope_vs_128_m_per_day": (None if step_64_128 is None
                                          else j2 + step_64_128),
            "envelope_label": ENVELOPE_LABEL,
        }]
        for cand in (8, 16, 32, 64):
            env = conservative_envelope(rows, cand, stage="sensitivity",
                                        window="day1", case_id=case_id)
            entries.append({
                "candidate": f"nmax={cand}",
                "vs": "full@128",
                "measured_vs_full64_m_per_day": None,
                "envelope_vs_128_m_per_day": env["envelope_m_per_day"],
                "envelope_label": ENVELOPE_LABEL,
                "parts": env["parts"],
            })
        estimator_physics.append({"case_id": case_id, "rows": entries})

    # seven-day decision matrix ----------------------------------------------
    seven_day_cases = []
    for case_id in ("S1_alt100_i45", "S7_alt100_i90", "S8_ecc80x500_i45",
                    "S3_alt500_i45"):
        w = get_window(sensitivity, "sensitivity", "day1", case_id=case_id)
        runs = w["runs"]
        rt128 = None
        for r in runs:
            if r.get("run") == "v6_full128":
                rt128 = r.get("runtime_s")
        min_alt = min(float(r["min_altitude_m"]) for r in runs
                      if r.get("min_altitude_m") not in (None, ""))
        seven_day_cases.append({
            "case_id": case_id,
            "one_day_evidence": "nmax ordering and J2-only inadequacy measured",
            "secular_ambiguity": ("argp/e secular claims benefit from a longer "
                                  "window" if case_id != "S3_alt500_i45"
                                  else "low - high-altitude control"),
            "runtime_projection_s": (None if rt128 is None else round(7.0 * rt128, 1)),
            "runtime_projection_label": RUNTIME_PROJECTION_LABEL,
            "surface_margin_km": round(min_alt / 1e3, 2),
            "info_gain": ("checks whether the nmax ranking and secular element "
                          "trends persist over seven days"),
        })
    seven_day = {
        "candidates": seven_day_cases,
        "decision": "recommended confirmation",
        "decision_options": ("required before final truth recommendation",
                             "recommended confirmation",
                             "optional thesis appendix", "not justified"),
        "rationale": [
            "not required to form the first truth recommendation (the "
            "truncation ordering is already measured at one day)",
            "valuable to strengthen the secular/long-period separation of "
            "the element-evolution findings",
            "supports the thesis defence of the argp/e evolution results",
            "checks whether the nmax ranking is preserved over the longer "
            "window (working hypothesis: it is; not demonstrated)",
        ],
        "window_dependence_text": SEVEN_DAY_WINDOW_TEXT,
        "auto_run": False,
    }

    # truth-model framework ---------------------------------------------------
    truth = {
        "candidates": [
            {"id": "T1", "model": "GRGM660PRIM nmax=64 + MOON_PA_DE421",
             "pros": "DE421 pipeline consistency; low runtime; threshold "
                     "evidence at 200 km, i=60 deg and the eccentric case",
             "cons": "64->128 steps at 100 km/polar remain tens of m/day; "
                     "weak as a low-LLO truth"},
            {"id": "T2", "model": "GRGM660PRIM nmax=128 + MOON_PA_DE421",
             "pros": "translational + PA pipeline consistency; conservative "
                     "for low-LLO/polar; modest runtime; supported by the "
                     "selected high-degree closures",
             "cons": "matched per-case 128->256 closure not run for every "
                     "sensitivity case; cannot be called threshold-"
                     "demonstrated everywhere"},
            {"id": "T3", "model": "GL1800F nmax=128 + MOON_PA_DE440",
             "pros": "independent high-resolution JPL solution; DE440 PA "
                     "frame-exact; strong selected cross-model agreement",
             "cons": "translational ephemeris remains DE421-derived; no "
                     "production kernel profile; not run on the full matrix"},
            {"id": "T4", "model": "GL1800F nmax=256 + MOON_PA_DE440",
             "pros": "selected high-resolution confirmation",
             "cons": "not a default-truth candidate; selected cases only"},
        ],
        "primary": {
            "model": "GRGM660PRIM nmax=128 + MOON_PA_DE421",
            "evidence_level": EV_RECOMMEND,
            "science_review": SR_TRUTH,
            "explicit_negatives": [
                "NOT threshold demonstrated in every case",
                "NOT universally converged",
                "NOT externally validated",
                "NOT mission-grade truth",
            ],
        },
        "confirmation": {
            "model": "GL1800F nmax=128 (+ nmax=256 on selected cases) + MOON_PA_DE440",
            "role": ("selected high-resolution cross-check and independent "
                     "gravity-solution robustness check; not external validation"),
        },
        "regime_dependent_lower_degrees": {
            "values": {"200 km": 64, "500 km": 32, "eccentric 80x500 km": 64,
                       "i=60 deg (100 km)": 64},
            "presentation": ("operational/runtime optimization, estimator "
                             "candidates and sensitivity conclusions — NOT a "
                             "requirement to lower the canonical truth degree"),
        },
        "decision_forms_considered": (
            "single canonical thesis truth",
            "regime-dependent truth",
            "primary truth + independent confirmation model (chosen)"),
    }

    # estimator recommendation -------------------------------------------------
    estimator = {
        "physics_statement": (
            "Available orbit-difference evidence indicates that estimator "
            "dynamics should include harmonics beyond J2. Degrees in the "
            "32-64 range are reasonable candidates for the first "
            "filter-in-the-loop experiments, but no estimator-optimal nmax "
            "has yet been demonstrated."),
        "physics_evidence_level": EV_RECOMMEND,
        "physics_science_review": SR_EST_PHYSICS,
        "mismatch_note": (
            "the structured J2-only mismatch signature (argument-of-perilune "
            "drift sign reversal, ~80x eccentricity-drift difference) makes "
            "it a working hypothesis that plain process noise cannot absorb "
            "the model error as white noise; this requires an explicit "
            "experiment and is not assumed solved"),
        "mismatch_evidence_level": EV_HYPOTHESIS,
        "software": {
            "J2-only": "operational today; physically weak in all tested LLO regimes",
            "harmonics + UKF": ("six-state propagation technically compatible; "
                                "first executable candidate AFTER scenario-runner "
                                "threading (Phase 13B2b); not yet run"),
            "harmonics + BLS-LM/SRIF": ("harmonic gradient/STM absent; currently "
                                        "unsupported; needs gradient or an "
                                        "alternative sensitivity strategy"),
            "truth-harmonics / estimator-J2 mismatch": EV_NOT_VALIDATED,
        },
        "software_science_review": SR_EST_SOFTWARE,
        "validation_plan": ("filter-in-the-loop experiments with residual "
                            "analysis, NIS/NEES, state error, covariance "
                            "consistency and process-noise sensitivity"),
    }

    # thesis hierarchy ----------------------------------------------------------
    s1_rows = get_window(sensitivity, "sensitivity", "day1",
                         case_id="S1_alt100_i45")["comparisons"]
    zonal_s1 = float(find_comparison(s1_rows, "zonal_beyond_c20",
                                     stage="sensitivity", window="day1",
                                     case_id="S1_alt100_i45")["final_dpos_m"])
    hierarchy = [
        {"layer": 1, "model": "Moon point mass + Earth/Sun third bodies",
         "added_physics": "central + third-body dynamics",
         "observed_difference_m_per_day": None,
         "evidence_level": EV_VERIFIED,
         "thesis_role": "dynamic baseline (Kepler-only energy sanity 1.5e-10/orbit)"},
        {"layer": 2, "model": "classical Moon J2",
         "added_physics": "dominant oblateness",
         "observed_difference_m_per_day": round(baseline_numbers["j2_effect"], 0),
         "evidence_level": EV_LEVEL1,
         "thesis_role": "historical reference dynamics"},
        {"layer": 3, "model": "real C20-only bridge",
         "added_physics": "GRAIL C20 vs constants J2 pairing",
         "observed_difference_m_per_day": round(baseline_numbers["c20_bridge"], 0),
         "evidence_level": EV_LEVEL1,
         "thesis_role": "consistency bridge (not a frame diagnostic)"},
        {"layer": 4, "model": "C20+C22",
         "added_physics": "largest sectoral term",
         "observed_difference_m_per_day": round(baseline_numbers["c22_effect"], 0),
         "evidence_level": EV_LEVEL1,
         "thesis_role": "first longitude-dependent signal"},
        {"layer": 5, "model": "zonal-only @64",
         "added_physics": "higher zonals (C30+)",
         "observed_difference_m_per_day": round(zonal_s1, 0),
         "evidence_level": EV_LEVEL1,
         "thesis_role": "axisymmetric secular driver (S1 source)"},
        {"layer": 6, "model": "full low-degree (nmax 8/16/32)",
         "added_physics": "low-degree tesserals",
         "observed_difference_m_per_day": None,
         "evidence_level": EV_LEVEL1,
         "thesis_role": "estimator candidates (ladder steps per regime)"},
        {"layer": 7, "model": "operational high-degree (nmax 64)",
         "added_physics": "mid-degree field",
         "observed_difference_m_per_day": round(baseline_numbers["ladder_64_vs_128"], 1),
         "evidence_level": EV_LEVEL1,
         "thesis_role": "threshold-demonstrated regimes (200 km, i=60, eccentric)"},
        {"layer": 8, "model": "low-LLO truth candidate (nmax 128)",
         "added_physics": "high-degree field",
         "observed_difference_m_per_day": None,
         "evidence_level": EV_RECOMMEND,
         "thesis_role": f"primary truth candidate ({CLOSURE_QUALIFICATION})"},
        {"layer": 9, "model": "GL1800F 128/256 (selected)",
         "added_physics": "independent high-resolution solution",
         "observed_difference_m_per_day": cross_model_baseline,
         "evidence_level": EV_LEVEL1,
         "thesis_role": "selected cross-check only (not external validation)"},
    ]

    return {
        "provenance": {
            "store_path": str(store_path),
            "store_sha256": store_sha256,
            "store_generated_utc": store.get("generated_utc"),
            "campaign_commits": ["a081dcb (13G-b)", "5401db8 (13G-c1)"],
            "context_commit": "18d1900 (PROJECT_CONTEXT.md)",
        },
        "level1_disclaimer": LEVEL1_DISCLAIMER,
        "evidence_levels": list(EVIDENCE_LEVELS),
        "claims": claims,
        "baseline_numbers": baseline_numbers,
        "element_findings": element_findings,
        "case_recommendations": case_rows,
        "supporting_closures": supporting_closures,
        "cross_model": cross_model,
        "frame_diagnostics": diagnostics,
        "estimator_physics_tables": estimator_physics,
        "threshold_policy": {
            "value_m_per_day": THRESHOLD_M_PER_DAY,
            "classification": "engineering screening heuristic",
            "text": THRESHOLD_POLICY_TEXT,
        },
        "seven_day": seven_day,
        "truth": truth,
        "estimator": estimator,
        "thesis_hierarchy": hierarchy,
    }


# ---------------------------------------------------------------------------
# Known-value consistency checks (real store only; opt-in via CLI flag)
# ---------------------------------------------------------------------------
KNOWN_VALUE_BANDS = {
    ("baseline", "j2only_model_error"): (12_800.0, 15_000.0),
    ("S1_alt100_i45", "ladder_64_vs_128"): (70.0, 100.0),
    ("S7_alt100_i90", "j2only_model_error"): (48_000.0, 58_000.0),
    ("S8_ecc80x500_i45", "ladder_64_vs_128"): (1.5, 3.5),
}


def verify_known_values(synth: dict) -> None:
    base = synth["baseline_numbers"]["j2only_model_error"]
    lo, hi = KNOWN_VALUE_BANDS[("baseline", "j2only_model_error")]
    if not (lo <= base <= hi):
        raise StoreError(f"known-value check failed: baseline j2only "
                         f"{base:.1f} m/day outside [{lo}, {hi}].")
    by_case = {c["case_id"]: c for c in synth["case_recommendations"]}
    for (case_id, which), (lo, hi) in KNOWN_VALUE_BANDS.items():
        if case_id == "baseline":
            continue
        entry = by_case[case_id]
        value = (entry["evidence_step_m_per_day"] if which.startswith("ladder")
                 else entry["j2only_error_m_per_day"])
        if value is None or not (lo <= value <= hi):
            raise StoreError(f"known-value check failed: {case_id}/{which} "
                             f"= {value} outside [{lo}, {hi}].")
    print("[synthesis] known-value consistency checks PASS")


# ---------------------------------------------------------------------------
# Writers (deterministic: timestamp = store generated_utc)
# ---------------------------------------------------------------------------
def write_json(synth: dict, out_dir: Path) -> Path:
    path = out_dir / JSON_NAME
    path.write_text(json.dumps(synth, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")
    return path


def write_csv(synth: dict, out_dir: Path) -> Path:
    path = out_dir / CSV_NAME
    fields = ["table", "case_id", "axis", "description",
              "j2only_error_m_per_day", "recommended_nmax", "evidence_level",
              "evidence_step", "evidence_step_m_per_day", "qualification",
              "layer", "model", "added_physics",
              "observed_difference_m_per_day", "thesis_role"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in synth["case_recommendations"]:
            writer.writerow({"table": "case_recommendations", **row})
        for row in synth["thesis_hierarchy"]:
            writer.writerow({"table": "thesis_hierarchy",
                             "evidence_level": row["evidence_level"], **row})
    return path


def _fmt(value, digits=1):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_md(synth: dict, out_dir: Path) -> Path:
    p = synth["provenance"]
    t = synth["truth"]
    e = synth["estimator"]
    sd = synth["seven_day"]
    lines = [
        "# Phase 13G-d — Gravity Model Synthesis and Truth/Estimator Recommendation",
        "",
        f"- source store: `{Path(p['store_path']).name}` "
        f"(sha256 `{p['store_sha256'][:16]}...`), campaign store timestamp "
        f"{p['store_generated_utc']} (this report is deterministic: no "
        "generation-time clock is used)",
        f"- campaign provenance: {', '.join(p['campaign_commits'])}; canonical "
        f"context {p['context_commit']}",
        "",
        "## 1. Executive summary",
        "",
        "- Classical J2-only lunar dynamics is inadequate in every tested LLO "
        f"regime ({_fmt(min(c['j2only_error_m_per_day'] for c in synth['case_recommendations'])/1e3)}"
        f"-{_fmt(max(c['j2only_error_m_per_day'] for c in synth['case_recommendations'])/1e3)} km/day "
        "vs the full field).",
        "- Primary thesis-truth candidate: **GRGM660PRIM nmax=128 + "
        "MOON_PA_DE421** (recommendation from available evidence; "
        f"{CLOSURE_QUALIFICATION}); independent confirmation: GL1800F 128/256 "
        "on selected cases (selected cross-check, not external validation).",
        "- Estimator dynamics need harmonics beyond J2; degrees 32-64 are "
        "reasonable first filter-in-the-loop candidates; no estimator-optimal "
        "nmax has been demonstrated.",
        f"- Seven-day campaign decision: **{sd['decision']}** (no run started).",
        "",
        "## 2. Scope and evidence levels",
        "",
        f"- {synth['level1_disclaimer']}",
        "- Evidence vocabulary: " + "; ".join(synth["evidence_levels"]) + ".",
        "- Non-adjacent truth-relative errors are linear sums of successive "
        "nested-model differences (triangle inequality) and are labelled "
        f"'{ENVELOPE_LABEL}'. RSS aggregation is not used: it is not a "
        "mathematical upper bound and its independence assumption has not "
        "been demonstrated.",
        "",
        "## 3. Verification vs validation vs estimation performance",
        "",
        "- Verification (implementation correctness) and Level-1 validation "
        "(physical adequacy of models against richer internal models) are "
        "complete for the gravity chain; estimation performance is NOT.",
        "- trajectory separation != estimator error;",
        "- final position difference != measurement residual;",
        "- one-day model difference != covariance-consistency requirement.",
        "- No filter/OD performance with harmonics has been validated.",
        "",
        "## 4. Data sources and campaign provenance",
        "",
        "- Single input: the cumulative Phase 13G JSON store (campaign CSVs "
        "are derived artifacts and were not read).",
        f"- store sha256: `{p['store_sha256']}`.",
        "",
        "## 5. Gravity-model and frame strategy",
        "",
        "- GRGM660PRIM (DE421 PA, frame-exact with the default kernel chain) "
        "is the primary science model; GL1800F (DE440 PA) is the selected "
        "high-resolution cross-check; GRGM1200L stays loader-sanity only.",
        "- Explicit MOON_PA_DE421 / MOON_PA_DE440 names everywhere; bare "
        "MOON_PA is forbidden (kernel-pool alias trap).",
        "- The translational ephemeris remains DE421-derived in all runs; its "
        "contribution is expected small for the tested third-body terms but "
        "was not independently isolated.",
        "",
        "## 6. Dominant physical contributions (baseline LLO, day 1)",
        "",
        f"- J2 effect over point-mass: {_fmt(synth['baseline_numbers']['j2_effect']/1e3, 1)} km/day.",
        f"- J2-only vs full@64: {_fmt(synth['baseline_numbers']['j2only_model_error']/1e3, 2)} km/day (headline).",
        f"- C22 effect: {_fmt(synth['baseline_numbers']['c22_effect']/1e3, 2)} km/day; tesseral "
        f"contribution {_fmt(synth['baseline_numbers']['tesseral_full_minus_zonal']/1e3, 1)} km/day "
        f"(difference-based; direct diagnostic {_fmt(synth['baseline_numbers']['tesseral_direct']/1e3, 1)} "
        "km/day, coupling ~4%).",
        f"- C20 bridge: {_fmt(synth['baseline_numbers']['c20_bridge'], 0)} m/day — reflects C20/J2 "
        "value, reference-radius and GM pairing differences; its numerical "
        "proximity to the independent mean-pole frame diagnostic is "
        "coincidental and must not be treated as mutual validation.",
        "- Element physics: argument-of-perilune drift sign reverses between "
        "J2-only and the full field; eccentricity drift grows ~80x (LLO "
        f"lifetime driver). {synth['element_findings']['inclination_note']} "
        f"(evidence: {synth['element_findings']['inclination_evidence']}).",
        "",
        "## 7-9. Altitude / inclination / eccentric sensitivity",
        "",
        "| case | axis | J2-only m/day | recommended nmax | evidence | step (m/day) | qualification |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in synth["case_recommendations"]:
        lines.append(
            f"| {c['case_id']} | {c['axis']} | {_fmt(c['j2only_error_m_per_day'], 0)} "
            f"| {c['recommended_nmax']} | {c['evidence_level']} "
            f"| {c['evidence_step']} = {_fmt(c['evidence_step_m_per_day'], 1)} "
            f"| {c['qualification'] or '-'} |")
    lines += [
        "",
        "- Supporting 128->256 closures (selected cross-check only; NOT "
        "substitutes for the missing per-case runs): "
        + "; ".join(f"{s['source']}: {_fmt(s['value_m_per_day'], 1)} m/day"
                    for s in synth["supporting_closures"]) + ".",
        "- The polar result is consistent with broad longitude sampling and "
        "strong tesseral exposure (not an isolated causal proof); the "
        "200-vs-100 km non-monotonicity has a working hypothesis "
        "(accumulated along-track phase beating with the rotating "
        "body-fixed field), not an isolated mechanism.",
        "- Eccentric case: perilune-window ratio ~"
        + _fmt(next((c.get('perilune_ratio_high_degree') for c in
                     synth['case_recommendations']
                     if c['case_id'] == 'S8_ecc80x500_i45'), None), 2)
        + " on the J2-only comparison; apolune dwell filters high degrees.",
        "",
        "## 10. Cross-model consistency (GL1800F)",
        "",
        f"- baseline: {_fmt(synth['cross_model']['baseline_m_per_day'], 2)} m/day; "
        + "; ".join(f"{k}: {_fmt(v, 2)} m/day"
                    for k, v in synth["cross_model"]["cases"].items()) + ".",
        f"- {synth['cross_model']['interpretation']}",
        "",
        "## 11. Truncation convergence and threshold policy",
        "",
        f"- {synth['threshold_policy']['text']}",
        "- The 10 m/day value is used only as a model-selection screening "
        "metric in this synthesis.",
        "",
        "## 12. Truth-model recommendation",
        "",
        f"- PRIMARY: **{t['primary']['model']}** — evidence level: "
        f"{t['primary']['evidence_level']}.",
        "- Explicitly NOT claimed: " + "; ".join(t["primary"]["explicit_negatives"]) + ".",
        f"- Confirmation: {t['confirmation']['model']} — {t['confirmation']['role']}.",
        "- Regime-dependent lower degrees ("
        + ", ".join(f"{k}: {v}" for k, v in
                    t["regime_dependent_lower_degrees"]["values"].items())
        + f") are presented as {t['regime_dependent_lower_degrees']['presentation']}.",
        f"- Science review: {t['primary']['science_review']}.",
        "",
        "## 13. Estimator-physics recommendation",
        "",
        f"- {e['physics_statement']}",
        f"- Mismatch note: {e['mismatch_note']} (evidence: {e['mismatch_evidence_level']}).",
        f"- Science review: {e['physics_science_review']}.",
        "",
        "### Conservative-envelope tables (per selected case)",
        "",
    ]
    for tbl in synth["estimator_physics_tables"]:
        lines += [f"#### {tbl['case_id']}", "",
                  "| candidate | measured vs full@64 (m/day) | envelope vs "
                  "full@128 (m/day) | note |", "|---|---|---|---|"]
        for row in tbl["rows"]:
            lines.append(
                f"| {row['candidate']} | {_fmt(row['measured_vs_full64_m_per_day'], 1)} "
                f"| {_fmt(row['envelope_vs_128_m_per_day'], 1)} | {row['envelope_label']} |")
        lines.append("")
    lines += [
        "## 14. Current software capability",
        "",
    ]
    for key, val in e["software"].items():
        lines.append(f"- {key}: {val}.")
    lines += [
        f"- Science review: {e['software_science_review']}.",
        f"- Validation plan: {e['validation_plan']}.",
        "",
        "## 15. Seven-day campaign decision",
        "",
        f"- DECISION: **{sd['decision']}** (options considered: "
        + "; ".join(sd["decision_options"]) + "). No run was started by this synthesis.",
        "- Rationale: " + " ".join(f"({i+1}) {r};" for i, r in
                                   enumerate(sd["rationale"])),
        f"- {sd['window_dependence_text']}",
        "",
        "| case | runtime projection (s) | note | surface margin (km) |",
        "|---|---|---|---|",
    ]
    for c in sd["candidates"]:
        lines.append(f"| {c['case_id']} | {_fmt(c['runtime_projection_s'], 0)} "
                     f"| {c['runtime_projection_label']} | {c['surface_margin_km']} |")
    lines += [
        "",
        "## 16. Thesis model hierarchy",
        "",
        "| layer | model | added physics | observed diff (m/day) | evidence | thesis role |",
        "|---|---|---|---|---|---|",
    ]
    for h in synth["thesis_hierarchy"]:
        lines.append(f"| {h['layer']} | {h['model']} | {h['added_physics']} "
                     f"| {_fmt(h['observed_difference_m_per_day'], 1)} "
                     f"| {h['evidence_level']} | {h['thesis_role']} |")
    lines += [
        "",
        "## 17. Limitations",
        "",
        "- One-day windows dominate the evidence base; seven-day behavior is "
        "a working hypothesis pending the recommended confirmation.",
        "- All conclusions are Level-1 (internal model-vs-model); no external "
        "flight-dynamics tool has validated these trajectories.",
        "- Harmonic gradient/STM is not implemented; BLS-LM/SRIF cannot use "
        "harmonics; no filter/OD performance with harmonics has been "
        "validated.",
        "- The DE421-derived translational ephemeris + DE440 PA rotation mix "
        "was declared but not independently isolated.",
        "- Frame diagnostics (frozen/mean-pole/wrong pairing) are "
        "intentionally wrong configurations and never feed recommendations.",
        "",
        "## 18. Recommended next phases",
        "",
        "1. Phase 13G-c2 (recommended confirmation): selected seven-day runs "
        "(S1, S7, S8, S3) after separate approval.",
        "2. Phase 13B2b: scenario-runner threading; first executable "
        "harmonics estimator path (UKF, six-state).",
        "3. Phase 13B2c: optional production kernel profiles.",
        "4. Filter-in-the-loop mismatch experiments only after the above.",
        "",
    ]
    path = out_dir / MD_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-known-values", action="store_true",
                        help="assert campaign numbers sit in the known bands "
                             "(use on the real store, not on test fixtures)")
    args = parser.parse_args(argv)

    store, sha = load_store(args.store)
    synth = build_synthesis(store, sha, args.store)
    if args.verify_known_values:
        verify_known_values(synth)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in (write_json(synth, args.out), write_csv(synth, args.out),
                 write_md(synth, args.out)):
        print(f"[synthesis] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
