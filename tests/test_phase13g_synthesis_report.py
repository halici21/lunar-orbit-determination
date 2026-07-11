"""Phase 13G-d -- synthesis-report tests (fixture store; no campaign data).

Exercises the store access/validation layer, the evidence-label logic, the
conservative-envelope policy, the mandated report language (threshold policy,
seven-day decision, C20-bridge and GL1800F wording, science-review labels),
determinism, and the structural no-dependency guarantee (import-surface
scan).  A smoke test against the real store runs only when it exists.
"""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import phase13g_synthesis_report as syn  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture store builder (synthetic; values chosen to exercise both label paths)
# ---------------------------------------------------------------------------
def comp(name, value, diagnostic=False, wrong=False, **extra):
    return {"comparison": name, "final_dpos_m": value, "rms_dpos_m": value,
            "final_radial_m": 0.0, "final_along_m": value,
            "final_cross_m": 0.0, "diagnostic": diagnostic,
            "intentionally_wrong": wrong, **extra}


def ladder(steps):
    return [comp(f"ladder_{lo}_vs_{hi}", val) for (lo, hi), val in steps.items()]


def elem(run, element, drift):
    return {"run": run, "element": element, "drift_per_day": drift,
            "initial": 0.0, "final": 0.0, "min": 0.0, "max": 0.0,
            "short_period_amp": 0.0, "diagnostic": False}


def runs_block():
    return [{"run": "v1_j2", "runtime_s": 0.1, "min_altitude_m": 99_000.0},
            {"run": "v6_full128", "runtime_s": 10.0, "min_altitude_m": 95_000.0}]


def case_window(j2only, steps, gl=False, s8=False):
    comps = [comp("j2only_model_error", j2only,
                  **({"perilune_ratio": 0.84} if s8 else {})),
             comp("c20_bridge", 1000.0), comp("c22_effect", 4000.0),
             comp("zonal_beyond_c20", 200.0),
             comp("tesseral_contribution", 12_000.0)] + ladder(steps)
    if gl:
        comps.append(comp("cross_model_nmax64", 1.0))
    if s8:
        comps.append(comp("g1800_ladder_128_vs_256", 0.8))
    return {"t_end_s": 86400.0, "n_epochs": 721, "period_s": 7068.0,
            "runs": runs_block(), "comparisons": comps, "elements": []}


REC_STEPS = {(8, 16): 3000.0, (16, 32): 6000.0, (32, 64): 900.0, (64, 128): 80.0}
THR64_STEPS = {(8, 16): 2000.0, (16, 32): 1900.0, (32, 64): 100.0, (64, 128): 2.0}
THR32_STEPS = {(8, 16): 400.0, (16, 32): 40.0, (32, 64): 0.5, (64, 128): 0.1}


def build_fixture_store():
    baseline_day1 = {
        "t_end_s": 86400.0, "n_epochs": 721,
        "runs": runs_block(),
        "comparisons": [
            comp("j2_effect", 65_000.0), comp("c20_bridge", 1018.0),
            comp("c22_effect", 3950.0),
            comp("tesseral_from_full_minus_zonal", 13_000.0),
            # diagnostic in the real store too (13G-b decomposition flag)
            comp("tesseral_direct", 12_500.0, diagnostic=True),
            comp("j2only_model_error", 13_870.0),
            comp("ladder_64_vs_128", 64.6),
        ],
        "elements": [elem("v1_j2", "argp_rad", +0.022),
                     elem("v6_full64", "argp_rad", -0.162),
                     elem("v1_j2", "e", -1.6e-5),
                     elem("v6_full64", "e", -1.32e-3)],
    }
    cases = {
        "S1_alt100_i45": case_window(13_500.0, REC_STEPS, gl=True),
        "S2_alt200_i45": case_window(15_100.0, THR64_STEPS),
        "S3_alt500_i45": case_window(6_400.0, THR32_STEPS),
        "S4_alt100_i0": case_window(11_300.0, REC_STEPS),
        "S5_alt100_i30": case_window(6_100.0, REC_STEPS),
        "S6_alt100_i60": case_window(34_600.0, THR64_STEPS),
        "S7_alt100_i90": case_window(52_900.0, REC_STEPS, gl=True),
        "S8_ecc80x500_i45": case_window(12_800.0, THR64_STEPS, gl=True, s8=True),
    }
    return {
        "generated_utc": "2026-07-11T00:00:00+00:00",
        "baseline": {"windows": {"day1": baseline_day1}},
        "sensitivity": {"cases": {cid: {"spec": {}, "windows": {"day1": w}}
                                  for cid, w in cases.items()}},
        "compare": {"windows": {"day1": {
            "runs": [], "elements": [],
            "comparisons": [comp("cross_model_nmax64", 3.5),
                            comp("g1800_ladder_128_vs_256", 2.2)]}}},
        "frames": {"windows": {"day1": {
            "runs": [], "elements": [],
            "comparisons": [
                comp("frame_wrongpair_n64", 0.9, diagnostic=True, wrong=True),
                comp("frame_frozen_n64", 1700.0, diagnostic=True, wrong=True),
                comp("frame_meanpole_n64", 28_000.0, diagnostic=True, wrong=True),
            ]}}},
    }


def build_synth(store=None):
    store = store or build_fixture_store()
    return syn.build_synthesis(store, "f" * 64, Path("fixture_store.json"))


def render(tmpdir, store=None):
    synth = build_synth(store)
    out = Path(tmpdir)
    syn.write_json(synth, out)
    syn.write_csv(synth, out)
    syn.write_md(synth, out)
    md = (out / syn.MD_NAME).read_text(encoding="utf-8")
    js = (out / syn.JSON_NAME).read_text(encoding="utf-8")
    return synth, md, js


class AccessLayerTests(unittest.TestCase):
    def test_missing_baseline_stage(self):
        store = build_fixture_store()
        del store["baseline"]
        with self.assertRaisesRegex(syn.StoreError, "--baseline"):
            build_synth(store)

    def test_missing_sensitivity_stage(self):
        store = build_fixture_store()
        del store["sensitivity"]
        with self.assertRaisesRegex(syn.StoreError, "--sensitivity"):
            build_synth(store)

    def test_missing_comparison_names_context(self):
        store = build_fixture_store()
        rows = store["sensitivity"]["cases"]["S2_alt200_i45"]["windows"]["day1"]["comparisons"]
        rows[:] = [r for r in rows if r["comparison"] != "j2only_model_error"]
        with self.assertRaisesRegex(
                syn.StoreError, "S2_alt200_i45.*j2only_model_error"):
            build_synth(store)

    def test_duplicate_comparison_error(self):
        store = build_fixture_store()
        rows = store["baseline"]["windows"]["day1"]["comparisons"]
        rows.append(comp("j2only_model_error", 1.0))
        with self.assertRaisesRegex(syn.StoreError, "duplicate"):
            build_synth(store)

    def test_non_finite_metric_error(self):
        store = build_fixture_store()
        store["baseline"]["windows"]["day1"]["comparisons"][5]["final_dpos_m"] = float("nan")
        with self.assertRaisesRegex(syn.StoreError, "non-finite"):
            build_synth(store)

    def test_diagnostic_rows_blocked_from_physics(self):
        rows = [comp("frame_meanpole_n64", 28_000.0, diagnostic=True, wrong=True)]
        with self.assertRaisesRegex(syn.StoreError, "diagnostic"):
            syn.find_comparison(rows, "frame_meanpole_n64",
                                stage="frames", window="day1")
        # and: diagnostics live only in their own section of the synthesis
        synth = build_synth()
        json_wo_diag = json.dumps({k: v for k, v in synth.items()
                                   if k != "frame_diagnostics"})
        self.assertNotIn("frame_meanpole", json_wo_diag)
        self.assertTrue(all(d["intentionally_wrong"]
                            for d in synth["frame_diagnostics"]))


class LabelAndEnvelopeTests(unittest.TestCase):
    def test_recommendation_and_threshold_labels(self):
        synth = build_synth()
        by_case = {c["case_id"]: c for c in synth["case_recommendations"]}
        self.assertEqual(by_case["S1_alt100_i45"]["recommended_nmax"], 128)
        self.assertEqual(by_case["S1_alt100_i45"]["evidence_level"], syn.EV_RECOMMEND)
        self.assertEqual(by_case["S1_alt100_i45"]["qualification"],
                         syn.CLOSURE_QUALIFICATION)
        self.assertEqual(by_case["S7_alt100_i90"]["evidence_level"], syn.EV_RECOMMEND)
        self.assertEqual(by_case["S2_alt200_i45"]["recommended_nmax"], 64)
        self.assertEqual(by_case["S2_alt200_i45"]["evidence_level"], syn.EV_THRESHOLD)
        self.assertIsNone(by_case["S2_alt200_i45"]["qualification"])
        self.assertEqual(by_case["S3_alt500_i45"]["recommended_nmax"], 32)
        self.assertEqual(by_case["S8_ecc80x500_i45"]["recommended_nmax"], 64)

    def test_conservative_envelope_is_linear_sum(self):
        rows = ladder(REC_STEPS)
        env = syn.conservative_envelope(rows, 32, stage="sensitivity",
                                        window="day1", case_id="X")
        self.assertAlmostEqual(env["envelope_m_per_day"], 900.0 + 80.0)
        self.assertEqual(env["label"], syn.ENVELOPE_LABEL)

    def test_envelope_missing_step_errors(self):
        rows = ladder({(32, 64): 900.0})     # 64->128 missing
        with self.assertRaisesRegex(syn.StoreError, "64->128"):
            syn.conservative_envelope(rows, 32, stage="sensitivity",
                                      window="day1", case_id="X")

    def test_j2only_envelope_adds_measured_base(self):
        synth = build_synth()
        s1 = next(t for t in synth["estimator_physics_tables"]
                  if t["case_id"] == "S1_alt100_i45")
        j2row = next(r for r in s1["rows"] if r["candidate"] == "J2-only")
        self.assertAlmostEqual(j2row["envelope_vs_128_m_per_day"],
                               13_500.0 + 80.0)


class MandatedLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        cls.synth, cls.md, cls.js = render(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # 1-2: RSS never an upper bound; envelope is a linear sum ----------------
    def test_rss_never_called_upper_bound(self):
        self.assertNotIn("rss", self.js.lower())
        for line in self.md.splitlines():
            if "RSS" in line:
                self.assertIn("not used", line)
                self.assertNotIn("upper bound for", line)
        self.assertIn(syn.ENVELOPE_LABEL, self.md)
        self.assertIn("linear sums", self.md)

    # 3: threshold policy -----------------------------------------------------
    def test_threshold_policy_language(self):
        self.assertIn("engineering screening heuristic", self.md)
        self.assertIn("not a validated OD requirement", self.md)
        self.assertNotIn("target OD accuracy", self.md)

    # 4: trajectory difference is not estimator error --------------------------
    def test_trajectory_not_estimator_error(self):
        self.assertIn("trajectory separation != estimator error", self.md)
        self.assertIn("No filter/OD performance with harmonics has been "
                      "validated", self.md)

    # 5: seven-day decision -----------------------------------------------------
    def test_seven_day_decision(self):
        self.assertEqual(self.synth["seven_day"]["decision"],
                         "recommended confirmation")
        self.assertFalse(self.synth["seven_day"]["auto_run"])
        self.assertIn("working hypothesis, not a demonstrated result", self.md)
        self.assertNotIn("weakly dependent", self.md)

    # 6: runtime projection label -------------------------------------------------
    def test_runtime_projection_label(self):
        self.assertIn(syn.RUNTIME_PROJECTION_LABEL, self.md)
        for c in self.synth["seven_day"]["candidates"]:
            self.assertEqual(c["runtime_projection_label"],
                             syn.RUNTIME_PROJECTION_LABEL)

    # closure claims ---------------------------------------------------------------
    def test_no_universal_closure_claim(self):
        self.assertIn(syn.CLOSURE_QUALIFICATION, self.md)
        self.assertIn("NOT threshold demonstrated in every case", self.md)
        self.assertNotIn("threshold demonstrated everywhere", self.md)
        # "universally converged" may appear ONLY inside its explicit negation
        self.assertEqual(self.md.count("universally converged"),
                         self.md.count("NOT universally converged"))
        for s in self.synth["supporting_closures"]:
            self.assertEqual(s["role"], "selected cross-check only")

    # C20 bridge wording --------------------------------------------------------
    def test_c20_bridge_not_frame_validation(self):
        self.assertIn("coincidental", self.md)
        self.assertIn("must not be treated as mutual validation", self.md)
        self.assertNotIn("reproduces", self.md)

    # GL1800F wording ---------------------------------------------------------------
    def test_gl1800f_selected_cross_check(self):
        self.assertIn("selected cross-check", self.md)
        self.assertIn("not external validation", self.md)
        self.assertNotIn("externally validated the", self.md)

    # disclaimers / labels ------------------------------------------------------------
    def test_level1_disclaimer_present(self):
        self.assertIn("Level-1 internal model-vs-model", self.md)
        self.assertIn(syn.LEVEL1_DISCLAIMER, self.js)

    # 10: three distinct science-review labels ------------------------------------------
    def test_science_review_labels_distinct(self):
        labels = {syn.SR_TRUTH, syn.SR_EST_PHYSICS, syn.SR_EST_SOFTWARE}
        self.assertEqual(len(labels), 3)
        for label in labels:
            self.assertIn(label, self.md)
        self.assertEqual(self.synth["truth"]["primary"]["science_review"],
                         syn.SR_TRUTH)
        self.assertEqual(self.synth["estimator"]["physics_science_review"],
                         syn.SR_EST_PHYSICS)
        self.assertEqual(self.synth["estimator"]["software_science_review"],
                         syn.SR_EST_SOFTWARE)

    def test_estimator_language_bounds(self):
        self.assertIn("no estimator-optimal nmax has yet been demonstrated",
                      self.md)
        self.assertIn("working hypothesis that plain process noise cannot "
                      "absorb", self.md)

    def test_claims_have_evidence_fields(self):
        for c in self.synth["claims"]:
            self.assertIn(c["evidence_level"], syn.EVIDENCE_LEVELS)
            for key in ("claim", "value", "unit", "source"):
                self.assertIn(key, c)


class DeterminismAndHygieneTests(unittest.TestCase):
    def test_outputs_deterministic(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d1, \
                tempfile.TemporaryDirectory() as d2:
            _, md1, js1 = render(d1)
            _, md2, js2 = render(d2)
            csv1 = (Path(d1) / syn.CSV_NAME).read_bytes()
            csv2 = (Path(d2) / syn.CSV_NAME).read_bytes()
        self.assertEqual(md1, md2)
        self.assertEqual(js1, js2)
        self.assertEqual(csv1, csv2)

    def test_import_surface_is_stdlib_only(self):
        # scan IMPORT statements (the guarantee is about dependencies; module
        # names may legitimately appear in docstrings and error messages)
        source = (ROOT / "examples" / "phase13g_synthesis_report.py").read_text(
            encoding="utf-8")
        import re
        imports = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", source,
                             flags=re.MULTILINE)
        for module in imports:
            root = module.split(".")[0]
            self.assertNotIn(root, ("lunar_od", "spiceypy",
                                    "phase13g_gravity_orbit_effects",
                                    "phase12b_real_grail_validation"),
                             f"forbidden dependency import {module!r}")

    def test_unknown_evidence_level_rejected(self):
        with self.assertRaises(ValueError):
            syn.claim("x", 1, "m", "somewhere-in-between", "src")


@unittest.skipUnless(syn.DEFAULT_STORE.is_file(),
                     "real Phase 13G store not available")
class RealStoreSmokeTests(unittest.TestCase):
    def test_real_store_synthesis_with_known_values(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rc = syn.main(["--out", tmp, "--verify-known-values"])
            self.assertEqual(rc, 0)
            for name in (syn.MD_NAME, syn.JSON_NAME, syn.CSV_NAME):
                self.assertTrue((Path(tmp) / name).is_file())


if __name__ == "__main__":
    unittest.main()
