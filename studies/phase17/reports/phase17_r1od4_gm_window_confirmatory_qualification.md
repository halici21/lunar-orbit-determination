# Phase 17-R1O-D4 — Goldstone–Madrid Narrow-Window Confirmatory Robustness Qualification

Truth level: **SYNTHETIC_CAMPAIGN_TRUTH** (K_truth = 0.01 m²/kg). No DSN allocation or mission performance is claimed.

Artifacts: `artifacts/phase17_r1od4_*`. Campaign: `examples/phase17_r1od4_campaign.py`. Predeclaration:
`artifacts/phase17_r1od4_predeclaration.json` (commit 91e899d, before any D4 K metric). Tests:
`tests/test_r1od4_gm_confirmation.py`.

## 1. Executive Summary

**The D3 secondary Goldstone–Madrid (GM) signal does not confirm under the primary (high-fidelity) dynamics.**

- **Simplified dynamics:** the canonical orbit reproduces a 30° window of STRONG FIRST_SESSION gain, 85°–115°
  (7 points, 2.46 days).
- **High-fidelity dynamics:** the same cell has **no** window of 7 or more points.
  - The longest STRONG run is 60°–80° (5 points).
  - No calibration case, from C0_IDEAL through C2_FREE, produces one.
  - **None of the five predeclared orbits** has a HIGH_FIDELITY C1_NOMINAL window (0/5).
- **Decision:** the predeclared decision tree is applied in the fixed order A→E. It stops at **Case C:
  `PRIMARY_CLASS = GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE`**, because the simplified and high-fidelity models disagree
  on whether the window exists.
- **Consequence:** `NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED = NO`.
- **Mechanism of the fidelity dependence** (descriptive, not a decision input): the fidelity change moves the
  **range-only** baseline f_perp (RMS 0.030). It barely moves the **combined** range + ΔDOR f_perp (RMS 0.007). The
  change in Δf_perp correlates with −Δ(range f_perp) at r = 0.976.
  - So the simplified-model "window" is mostly the simplified model's range-only baseline being weaker there. It is
    not ΔDOR geometry that holds across fidelity.
- **Verification:** all verification passed.
  - 50/50 D3 anchor values reproduced exactly.
  - 2200 independent rebuilds agree to 1.4e-14.
  - Every contract passed, and the predeclaration hash never changed.

## 2. Why D4 Exists

D3 closed its primary question: ΔDOR tracking gain is confined to narrow geometric windows. It also recorded one
secondary, exploratory observation: GM FIRST_SESSION was STRONG at D3 k = 5, 6, 7 (75°/90°/105°) under C1 nominal
calibration and simplified dynamics. D3 recommended a predeclared robustness test of that region before any
nonlinear work. D4 is that test.

## 3. Exploratory vs Confirmatory Evidence

- **The D3 observation was exploratory.** It was 1 of 13 schedule patterns, on a 15° grid, with one orbit and one
  dynamics model, so it carries multiple-comparison exposure.
- **D4 is confirmatory.** Everything below was frozen and hash-committed before any D4 K-information metric existed:
  - the schedule (FIRST_SESSION);
  - the phase interval (60°–165°) and the 5° grid;
  - the 30° window criterion;
  - the orbit set;
  - the dynamics hierarchy (HF primary);
  - the calibration cases;
  - the decision tree.

## 4. Entering Authoritative State

| item | value |
|---|---|
| branch | `feature/phase17-r-k-srp-estimation` |
| HEAD | `89942a618731ab44151a4b9c3396540f1949105a` |
| tree | `9e55303a5f12d089222a61eb713aa69d02f9fd65` |
| main / origin/main | `fea476f81dad07b3914e53eab709e10fa6e9d10b` |
| entry suite | 1417 collected / 1388 passed / 0 failed / 0 errors / 29 skipped |
| gates | HARDEN PASS, F1 PASS, D3 PASS (PRIMARY_CLASS narrow windows) |

## 5. Literature Status

`D4_LITERATURE_STATUS = D3_CONTRACT_RECONFIRMED_NO_REVISION`.

- The DSN 810-005 index still lists 210F (the revision D3 used) and 107F.
- The re-downloaded 107F PDF has sha256 `3a8da003…1d1d`, byte-identical to D3's copy.
- The values carried over, with their labels:
  - random ΔDOR delay σ = **0.048 ns** (DSN-210F);
  - nominal session-bias prior = **0.0317 ns**. This is REPOSITORY arithmetic from 210F's systematic rows, not a
    DSN Gaussian claim.

## 6. D3 Hypothesis Being Tested

- **H1:** GM FIRST_SESSION has a STRONG window (Δf_perp > 0.05 over ≥ 30° of synodic phase) overlapping 75°–150°,
  under canonical HF dynamics with C1 nominal calibration.
- **H2:** it is orbit-robust (at least 4 of 5 orbits).
- **H3:** it survives realistic calibration and dynamics fidelity.

## 7. Predeclaration

- **File:** `artifacts/phase17_r1od4_predeclaration.json`.
- **Hash:** sha256 **`3a48bc212f2b5b0b26f15953dfd6c2e2cc230a2fdf042cb9cd1e77772c8b9084`**, taken over the canonical
  JSON body (sorted keys, compact separators).
- **Commit:** 91e899d, before any D4 metric.
- **Enforcement:** the campaign pins the hash and refuses to run on a mismatch. The test suite checks that tampering
  with the window size, the threshold or the grid is refused.
- **Contents:**
  - the full matrix and its fallback rule (the fallback was not needed);
  - the classes and the decision tree;
  - the mechanism rule;
  - the verification list A–G.

## 8. Dense Epoch Grid

- **Grid:** 60°–165° in 5° steps, 22 points. It includes the D3 anchors 75/90/105/135/150°.
- **Epoch convention:** `et0 = 857806356.99998 + (phase/15)·S/24`, with S = 2 551 442.8896 s. This is identical to
  D3's k·S/24 at phase = 15k; `test_phase_epoch_convention_equals_d3_k_at_multiples_of_15_deg` checks both.
- **Independent check:** the ET was also reconstructed as phase/360·S. The maximum difference is 0.0 s.
- **Spacing:** 5° is 0.41 days.
- **Fixed and not searched:** the 15-orbit arc, the 90 s grid and K = 0.01.

## 9. Window-Width Equivalence Rule

- **Criterion:** D3 required 3 consecutive 15° points, which is a 30° endpoint span. On the 5° grid the same physical
  width is **7 consecutive STRONG points** (30°, 2.46 days).
- **Regression test:** `test_regression_three_consecutive_5deg_points_do_not_qualify` confirms that 3 points, 3
  isolated anchors, or 6 points do not qualify.
- **Other rules:**
  - no wraparound;
  - an infeasible cell (no qualifying GM session) is not STRONG;
  - a window overlaps the candidate region if [start, end] intersects [75°, 150°].

## 10. GM Session Contract

The session construction is D3's (`phase17_r1od3_core`), unchanged:

- **Dense rows:** production-visibility ΔDOR rows on the Goldstone–Madrid pair.
- **Windows:** a window is a contiguous dual-visible run of at least 20 minutes.
- **Sessions:** at most 60 minutes, centred in the window.
- **Normal points:** 3 per session, at 1/6, 1/2 and 5/6 of the session's rows.

Every one of the 220 geometries had at least one qualifying GM session, so there were no infeasible cells. Qualifying
GM dual visibility ranged from 0.9 to 5.6 hours per arc.

## 11. FIRST_SESSION Primary Design

FIRST_SESSION is the chronologically earliest qualifying GM session, chosen without looking at K information. It
gives 3 normal points and one bias column (`test_first_session_is_the_chronologically_earliest_qualifying_window`).

## 12. ALL_FEASIBLE Reference

ALL_FEASIBLE_SESSIONS stacks every qualifying GM session. It is the D3 primary pattern and gives continuity with D3.
The canonical HF session counts range from 1 to 6 across the phase grid.

## 13. Calibration Model

| D4 case | D3 core name | bias model |
|---|---|---|
| C0_IDEAL | C0_IDEAL | no bias nuisance |
| C1_HALF | C1_x0.5 | per-session bias, prior 0.0159 ns |
| **C1_NOMINAL** | C1_x1.0 | per-session bias, prior 0.0317 ns |
| C1_DOUBLE | C1_x2.0 | per-session bias, prior 0.0634 ns |
| C2_FREE | C2_FREE | free per-session bias |

- **Data weight:** the random 0.048 ns delay σ per normal point in every case, with no systematic term in the weight.
- **Bias parameters:** one bias column per session, nuisance scale 1e-10 s.

## 14. Orbit Set

The initial states are frozen by sha256 of the D2 predeclared-grid x0 (`test_orbit_initial_states_are_frozen`).

| orbit | source | role |
|---|---|---|
| CANON | campaign x0 (checked equal to `campaign_initial_state()`) | canonical |
| O1_LOW_BETA_ECLIPSE_RICH | GEO G1_b00 | eclipse-rich, low beta |
| O2_FULL_SUN_HIGH_BETA | GEO G1_b80 | full sun, high beta |
| O3_HIGH_ALTITUDE_500KM | GEO G2_alt500_b20_fixD | high altitude |
| O5_FAVOURABLE_GEO_MAX_FPERP | GEO G3_i26.7_bcan | previously favourable |

## 15. Dynamics Fidelity Model

- **HIGH_FIDELITY (primary):** `phase17_r1od2_fidelity.propagate_fidelity`.
  - Force model: GL1800F 50×50 (MOON_PA_DE440), Earth and Sun point-mass third bodies (SPICE), and production SRP with
    a conical penumbra.
  - STM: F1-qualified harmonic STM, rtol 1e-12.
  - This is the D3 Stage C configuration.
  - Run time: about 330 s per geometry (median 337 s).
- **SIMPLIFIED (comparison):** `propagate_nom48`, with a lunar point mass, lunar J2 and SRP.
- **Matrix:** the full predeclared matrix ran, 5 orbits × 22 phases × 2 models = 220 geometries, in 2344 s on 15
  workers. The fallback was not invoked.
- **HF path check:** the HF range-only f_perp at CANON 75° is 0.21182734886671062, **bitwise equal** to D3's HF2
  value (`artifacts/phase17_r1od3_high_fidelity.json`).

## 16. D3 Anchor Reproduction

Verification item G compared canonical SIMPLIFIED at 75/90/105/135/150°, for FIRST and ALL, under all five
calibrations. That is 50 values against `artifacts/phase17_r1od3_primary_results.csv`.

- Maximum |difference| = **0.0**, so `D3_ANCHORS_REPRODUCED = YES`.
- The pytest test `test_d3_anchor_90deg_reproduces_exactly` repeats this at 90°.

## 17. Canonical Simplified Result

- **FIRST_SESSION, C1_NOMINAL:** STRONG at 75°, then **85°–115° (7 points)**.
- **The window:** 85°–115°, 30° wide, 2.46 days. Max Δ = 0.0663, median Δ = 0.0565. It overlaps [75°, 150°].
- **Reading:** D3's k = 5–7 anchors are embedded in a genuine 30° STRONG run under the simplified model. Seen alone,
  H1 would look confirmed there.

## 18. Canonical High-Fidelity Result

- **STRONG points (FIRST_SESSION, C1_NOMINAL):** 60°, 65°, 70°, 75° and 80°, with Δ = 0.060, 0.076, 0.073, 0.085 and
  0.054, plus an isolated 140° (0.0545).
- **Longest run:** 5 points (20°). No window.
- **Maximum HF Δ:** 0.0847, at 75°.
- **Across the old window region:** from 85° to 120°, HF Δ falls to 0.006–0.041. These cells are MAGNITUDE_ONLY or
  MODEST.
- **So `R0` fails:** there is no ≥ 30° STRONG window under canonical HF at C1_NOMINAL.

## 19. Dense Phase Continuity

- **Simplified:** the dense grid is smooth between the 15° anchors, with no single-point spikes making the window.
- **HF:** Δ is also smooth, but its STRONG run sits at the **lower edge** of the predeclared interval (60°–80°).
- **Unresolved edge:** whether HF gain continues below 60° is untested. The interval was fixed before results and
  was not redefined (see §36).
- **ALL_FEASIBLE:** noisier. It has isolated high cells at 135° (0.116) and 150° (0.086), and its longest HF STRONG
  run is 4 points.

## 20. Orbit Robustness

R1 is HIGH_FIDELITY, C1_NOMINAL, FIRST_SESSION, with a window of ≥ 7 points overlapping [75°, 150°]:

| orbit | window | longest HF STRONG run | longest SIMPLIFIED run |
|---|---|---|---|
| CANON | none | 5 (60–80°) | 7 (85–115°) |
| O1 | none | 1 | 1 |
| O2 | none | 1 | 1 |
| O3 | none | 1 | 1 |
| O5 | none | 6 (60–85°) | 7 (70–100°) |

- **Result:** `ORBIT_WINDOW_COUNT = 0/5`, so `ORBIT_ROBUSTNESS_CLASS = NO_CONFIRMED_WINDOW`.
- **O1–O3:** Δ swings by up to ±0.2 within 5–10° (for example O2: +0.219 at 135°, −0.111 at 140°). Such rapid
  swings make a contiguous 30° window structurally impossible for these orbits.
- **Simplified dynamics:** even there, only 2/5 orbits (CANON, O5) have a window.

## 21. Calibration Robustness

R2 on canonical HF, FIRST_SESSION: none of the five cases has a window. The STRONG runs are:

| case | STRONG runs |
|---|---|
| C0_IDEAL | 60–80° (5), plus 140° |
| C1_HALF | 60–80° (5), plus 140° |
| C1_NOMINAL | 60–80° (5), plus 140° |
| C1_DOUBLE | 60–80° (5) |
| C2_FREE | 65–75° (3) |

- **Result:** `CALIBRATION_ROBUSTNESS_CLASS = NO_WINDOW_ANY_CALIBRATION`.
- **Why no fate table was produced:** the predeclared R2 procedure tracks the fate of each nominally confirmed window,
  or the C0/C1_HALF windows if there are none. HF produced none of either, so `R2_window_fates` is empty by
  construction.
- **Descriptive fates of the simplified-only windows** (not decision inputs):
  - canonical 85–115°: PERSISTS under C0, C1_HALF and C1_NOMINAL; FRAGMENTS under C1_DOUBLE (85–100° and 115°);
    DISAPPEARS under C2_FREE.
  - O5 70–100°: PERSISTS under C0, C1_HALF and C1_NOMINAL; FRAGMENTS under C1_DOUBLE and C2_FREE.
  - So even in the simplified model, the window would have been CALIBRATION_SENSITIVE.
- **Fate labels** (`r2_status`, unit-tested):
  - PERSISTS: every window point is still STRONG.
  - SHRINKS: a window of ≥ 7 points still intersects the original but no longer covers it.
  - FRAGMENTS: no such window, but ≥ 2 STRONG points remain inside.
  - DISAPPEARS: otherwise.

  The label names were predeclared. Their operational definitions were written in code before the first D4 result
  was read.

## 22. Fidelity Robustness

R3 compares canonical SIMPLIFIED and HIGH_FIDELITY under C1_NOMINAL, FIRST_SESSION:

- **Existence:** the window exists in SIMPLIFIED (85–115°) and not in HF, so **`FIDELITY_CLASS = FIDELITY_CHANGES_EXISTENCE`**.
- **Class changes:** 14 of 22 phases change class. At 90–110°, STRONG becomes MAGNITUDE_ONLY. At 65–75°, MODEST
  becomes STRONG. At 140°, MODEST becomes STRONG.
- **Fidelity sensitivity by orbit:**

  | orbit | max \|Δ(HF) − Δ(SIMPLIFIED)\| | class changes |
  |---|---|---|
  | CANON | 0.057 | 14 |
  | O5 | 0.077 | 13 |
  | O1 | 0.0085 | 0 |
  | O2 | 0.017 | 0 |
  | O3 | 0.0008 | 0 |

  The sensitivity is strong for the two low-altitude canonical-family orbits and negligible for O1–O3.
- **Decomposition** (descriptive, `R3_fidelity.descriptive_decomposition_canon_first_c1` in the summary). Going from
  SIMPLIFIED to HF:
  - the range-only f_perp changes by RMS **0.030**;
  - the combined range + ΔDOR f_perp changes by RMS only **0.007**;
  - Δ(Δf_perp) correlates with −Δ(range f_perp) at **r = 0.976**.

  Example, 90°:

  | model | range f_perp | combined f_perp |
  |---|---|---|
  | SIMPLIFIED | 0.248 | 0.312 |
  | HF | 0.281 | 0.300 |

- **Interpretation:** the simplified window mostly reflects where the simplified range-only baseline is weakest. The
  ΔDOR contribution itself is fidelity-stable to about 0.01. Because the relative-gain metric is a difference, a
  fidelity change in the range baseline moves the window.

## 23. FIRST vs ALL Mechanism

This uses the predeclared rule, on canonical HF at C1_NOMINAL, over the 19 phases where ALL has more than one
session:

| quantity | value |
|---|---|
| median d_diff | +0.0007 |
| median I_kk ratio | 1.118 |
| median added-session orthogonal fraction | 0.083 |
| median first-session orthogonal fraction | 0.078 |

- **Result:** `FIRST_VS_ALL_MECHANISM_CLASS = NO_SYSTEMATIC_DIFFERENCE`.
- **Phase dependence** (descriptive; see `phase17_r1od4_first_vs_all_mechanism.png`):
  - 60–90°: later sessions dilute the direction (d_diff down to −0.041 at 75°).
  - 125–150°: they add direction (+0.069 at 135°, +0.065 at 150°).
  - The two effects cancel in the median.

## 24. Literature Geometry Overlay

The overlay uses DSN "good GM geometry" as context, not as a validity rule: Sun–Earth–spacecraft angle > 10° and
declination > −15° at the FIRST midpoint.

- All **22/22** canonical HF phases satisfy it.
- Ranges across the phases:
  - Sun–Earth–spacecraft angle: 65°–175°.
  - Spacecraft declination: −8.6° to +27.5°.
  - Lunar declination: −6.9° to +27.6°.
- The 6 STRONG phases are a subset of the 22, so the literature criterion does not discriminate the gain at all.

## 25. Radio-Source Coordinate Overlay

The overlay uses the DSN-107F Table 3 coordinates, parsed with the D3 parser: 4488 sources, PDF sha256 `3a8da003…`.

- A catalog source lies within 0.1 rad of the geocentric spacecraft direction at **220/220** FIRST_SESSION midpoints.
- Nearest-source separation: median 1.36°, max 2.56°.
- So `REAL_SOURCE_COORDINATE_COVERAGE = COORDINATE_SOURCE_AVAILABLE`. This establishes coordinate proximity only; it
  makes no claim about flux, structure or scheduling.

## 26. Absolute K Separation

Across all 440 feasible C1_NOMINAL cells (both schedules, both models):

- 397 are WEAK and 43 are MODERATE. None is STRONG_K_SEPARATION.
- The maximum combined f_perp is 0.464 (O2, SIMPLIFIED, 135°, FIRST).
- So `ABSOLUTE_K_IDENTIFIABILITY_CLASS = MODERATE_K_SEPARATION`, as the best case only.
- Every canonical HF cell is below 0.35 (WEAK). σK/K for canonical HF FIRST sits at 0.07–0.08.

## 27. Relative Gain

C1_NOMINAL, FIRST_SESSION class counts:

| cell | STRONG | MODEST | MAGNITUDE_ONLY | LOSS |
|---|---|---|---|---|
| CANON HF | 6 | 7 | 9 | 0 |
| CANON SIMPLIFIED | 8 | 10 | 4 | 0 |
| O5 HF | 8 | 3 | 11 | 0 |
| O1 HF | 2 | 4 | 14 | 2 |
| O2 HF | 3 | 2 | 13 | 4 |
| O3 HF | 2 | 3 | 14 | 3 |

STRONG cells occur, but they do not form the required contiguous 30° run under HF.

## 28. Independent Reconstruction

| check | result |
|---|---|
| A: predeclaration hash | pinned; campaign refuses a mismatch; tamper tests pass |
| B: phase → ET | reconstructed independently as phase/360·S; max diff 0.0 s |
| C: visibility | every GM row dual-visible for all 220 geometries; every normal point dual-visible; Earth-epoch audit passed |
| D/E: ownership | normal-point parent rows equal the session's `np_index`; H rows equal the parent rows; each bias column is owned by exactly its session |
| F: rebuild | independent `lstsq` reconstruction for every cell × schedule × calibration (2200): max \|f_perp difference\| **1.35e-14** (gate 1e-10) |
| G: D3 anchors | 50/50, exact |

## 29. Regression Tests

`tests/test_r1od4_gm_confirmation.py` has 13 tests, all passing:

- predeclaration hash and tamper refusal;
- schedule and calibration mapping;
- the 22-point grid with anchors;
- the epoch convention against D3's k;
- the 7-point window;
- the regression that **3 consecutive 5° points do not qualify**;
- boundary, infeasible and overlap rules;
- R2 fate labels;
- FIRST chronological selection and ALL continuity;
- orbit x0 freeze;
- HF configuration;
- the 90° D3 anchor, exact;
- anchor-cell contracts and rebuild.

The D3, F1, HARDEN, D2 and measurement tests ran unchanged inside the full suite.

## 30. Full Suite

`python -m pytest -o addopts="" -p no:cacheprovider`:

| | collected | passed | failed | errors | skipped |
|---|---|---|---|---|---|
| entry | 1417 | 1388 | 0 | 0 | 29 |
| D4 | **1430** | **1401** | **0** | **0** | **29** |

- The 13 new tests are exactly the D4 tests.
- The 29 skips are the same opt-in slow regressions as at entry (`LUNAR_OD_RUN_SLOW_TESTS`).
- `NEW_SCIENTIFIC_REGRESSIONS = 0`, `UNKNOWN_NONPASSES = 0`.
- No production path was modified, so no slow tests needed re-running.

## 31. Final Diff Review

**CLEAN.** Relative to the entering HEAD 89942a6, the only changed paths are the D4 files:

- `examples/phase17_r1od4_predeclare.py`
- `examples/phase17_r1od4_campaign.py`
- `tests/test_r1od4_gm_confirmation.py`
- this report
- `artifacts/phase17_r1od4_*`

What did not change:

- production code (`lunar_od/`), the measurement model, the physical model and the estimator;
- the D3 core and the D2/F1 code.

There are no debug leftovers. `$TEMP/d4` (the catalog PDF and logs) was removed at closure.

## 32. Confirmatory Decision Tree

The predeclared order is A→E.

| case | condition | result |
|---|---|---|
| A | all 7 success items | **no**: items 1, 2, 3, 5 fail; items 4, 6, 7 pass |
| B | R0 true | **no**: no canonical HF window |
| C | SIMPLIFIED and HF disagree on existence | **yes**: 85–115° vs none |
| D | not reached | — |
| E | not reached | — |

**`PRIMARY_CLASS = GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE`.** Case C sits before D and E in the fixed order. Note that
the facts that would have triggered E also hold: there is no canonical HF C1_NOMINAL window. The class reported is the
first one reached in the predeclared order.

## 33. Scientific Interpretation

- **Exploratory → confirmatory:** the D3 feature is real *in the simplified model*. Dense sampling makes it a clean
  30° run. It does not survive moving to the primary high-fidelity dynamics, and it was never orbit-robust (2/5 orbits
  even when simplified).
- **Why it fails:** the fidelity decomposition locates the cause. Δf_perp is a difference between two f_perp values.
  The range-only term is sensitive to the gravity field and third-body modelling at these phases (±0.04). The combined
  ΔDOR-inclusive term is not (±0.01). A window defined on the difference therefore moves when the baseline moves.
- **What this means for ΔDOR:** ΔDOR adds a roughly fidelity-stable amount of K direction. Whether that amount clears
  the 0.05 threshold depends on the range-only baseline, and the baseline depends on the force model.
- **Conclusion:** the GM secondary signal is best described as non-replicating under the primary dynamics. It should
  not be optimized further.

## 34. Operational Interpretation

A usable ΔDOR opportunity must survive nearby dates, realistic session placement, calibration uncertainty, orbit
geometry and dynamics fidelity. This one fails on two of those:

- **orbit geometry:** at most 2/5 orbits have a window, and only under simplified dynamics;
- **dynamics fidelity:** existence flips between models.

It also survives calibration only in a sensitive way, even in the simplified model (FRAGMENTS at C1_DOUBLE). There is
no scheduling opportunity to recommend. All results remain SYNTHETIC_CAMPAIGN_TRUTH.

## 35. Remaining Limitations

- Linear information analysis only. No estimator was run, as the brief required.
- One K formulation: a constant scalar K.
- Five representative orbits, not a population.
- The HF model is a point-mass Earth/Sun third body plus a 50×50 lunar field. Higher-order effects (for example Earth
  J2 on the spacecraft, albedo, thermal) are absent from both models.
- The window threshold (0.05) and the width (30°) are predeclared conventions inherited from D3.
- The calibration prior is repository arithmetic, not a DSN Gaussian.

## 36. Remaining Unknowns

- **The HF STRONG run is censored at the predeclared lower edge.** Canonical HF has 60–80° (5 points) and O5 HF has
  60–85° (6 points).
  - Whether a ≥ 30° HF window exists at phases below 60° is **untested**. The interval was fixed before results, and
    extending it post hoc is forbidden.
  - D3's simplified map had MODEST gain at 45° and 60°.
  - Any follow-up would need a new predeclaration. It would also inherit the same range-baseline fidelity sensitivity
    found here.
- Why the fidelity sensitivity is concentrated in the low-altitude canonical-family orbits (CANON, O5) and absent for
  O1–O3 was not decomposed by force term.
- The phase-dependent FIRST-vs-ALL behaviour (dilution at 60–90°, addition at 125–150°) is descriptive only.

## 37. What This Phase Actually Established

- The D3 GM anchors reproduce exactly, and the D4 machinery is contract-valid and independently reconstructed.
- Under simplified dynamics, canonical GM FIRST_SESSION has a 30° STRONG window (85–115°).
- Under the primary HF dynamics:
  - no orbit has a qualifying window (0/5);
  - no calibration case produces one for the canonical orbit.
- The HF/simplified disagreement comes mostly from the range-only baseline, not from the ΔDOR geometry (r = 0.976).
- No cell reaches absolute STRONG_K_SEPARATION; the best case is 0.464.

## 38. What This Phase Did NOT Establish

- Any nonlinear estimation result, K recovery or Monte Carlo performance.
- Anything about phases outside 60–165°, baselines other than GM, or schedules other than FIRST and ALL.
- Real DSN schedulability, source flux or allocation.
- That no ΔDOR design could ever help K: only this formulation, this design family and these orbits were tested.

## 39. Final Gate

`PHASE17_R1O_D4_GATE = PASS`. This means the confirmatory characterization was completed correctly. It does not mean
the hypothesis was confirmed.

## 40. Whether Nonlinear Confirmation Is Justified

**NO.** Only Case A justifies it, and D4 reached Case C. Following the brief, D4 does not proceed to nonlinear
estimation. It interpreted the fragility instead: the gain is a range-baseline-difference artifact of the simplified
dynamics.

## 41. Whether ΔDOR Tracking Design Is Closed

`DDOR_TRACKING_DESIGN_BRANCH_STATUS = NOT_CONFIRMED_FRAGILITY_DOCUMENTED`.

- **Why not formally closed:** the predeclared closure label (CLOSED_FOR_CURRENT_K_SRP_FORMULATION) is attached to
  Case E. The predeclared order resolved to C, so the formal label is not applied.
- **What that means in practice:** no ΔDOR tracking-design result in D3 or D4 supports nonlinear confirmation.
  - The primary D3 conclusion: narrow windows.
  - The secondary D4 conclusion: not replicated under HF, and not orbit-robust.
- The branch offers no remaining predeclared hypothesis.

## 42. Exact Next Action

**Hard stop after D4.** Do not start nonlinear estimation. The recommended next research direction, which needs
separate authorization, is **alternative observables and/or alternative K_SRP parameterizations**, not further ΔDOR
schedule, epoch or baseline search.

The only ΔDOR question D4 leaves open is the HF behaviour below 60°. It is not recommended to pursue it, because it is
subject to the same range-baseline fidelity sensitivity. It would need a fresh predeclaration.
