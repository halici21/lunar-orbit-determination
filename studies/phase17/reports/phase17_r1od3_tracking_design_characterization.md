# PHASE 17-R1O-D3 — PREDECLARED ΔDOR TRACKING-DESIGN CHARACTERIZATION

> **The question, answered first.** Can a predeclared operational tracking strategy create a reproducible and
> reasonably broad region of useful K_SRP direction gain? The strategy could vary the baseline, the lunar-phase epoch,
> session placement and count, dual-baseline stacking, and calibrated-bias knowledge.
>
> **No. The gain stays confined to narrow geometric windows.**
> - **No tracking window.** Across 24 uniform synodic epochs (15° apart), under operational sessions with the DSN
>   random delay weight and the nominal calibrated session bias, no primary strategy reaches STRONG relative gain at
>   ≥ 3 consecutive epochs.
> - **The canonical D2 cell weakens.** It falls from STRONG to **MODEST** under operational sampling (Δf_perp +0.0328).
> - **Goldstone–Madrid** gives the largest gains. They are confined to lunar phase ≈ 135–150°, where its own ΔDOR
>   separability peaks.
> - **Calibration is a first-order limiter.** STRONG cells number 9 (ideal) → 5 (nominal prior) → 0 (free session
>   biases).
> - **Orbit and dynamics fidelity** move cells across the threshold as much as the epoch does.
> - **K is never absolutely separable.** Every primary cell is WEAK_K_SEPARATION.
>
> `PRIMARY_CLASS = DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS`. `PHASE17_R1O_D3_GATE` is in §48.
> All numbers are `SYNTHETIC_CAMPAIGN_TRUTH`.

---

## 1. Executive Summary

D3 characterised ΔDOR tracking design on the qualified post-F1 production chain. Everything was predeclared in
c860759 before any D3 K metric was computed: 24 epochs × {GC_ONLY, GM_ONLY, GC_PLUS_GM} × 13 chronological schedule
patterns × 5 calibration treatments.

**Checks:**
- Every observation obeys the elevation + occultation contract.
- Every ephemeris stayed in coverage.
- Bias columns are provably owned by their sessions.
- 292 independent rebuilds agree with the reported metrics to ≤ 4.1e-15.
- The canonical authoritative values reproduce exactly.

**Findings:**
- **No robust or merely consecutive window exists** under the primary patterns.
- **Nominal STRONG epochs are isolated:**
  - GM_ONLY at k = 6, 9, 10 (the run 9–10 is only 2 long);
  - GC_PLUS_GM at 9–10;
  - GC_ONLY at none.
- **Calibration limits the gain strongly** (H3 supported).
- **The joint schedule "broadens" only by the literal predeclared rule.** Its STRONG epochs are GM's standalone
  strength, and it never beats the better single baseline, so H2 is not supported mechanistically.
- **No geometric predictor is adequate.** The best pooled |ρ| is 0.47.
- **High fidelity changes the class of the near-threshold case.** It moves GC at k = 5 from MODEST to STRONG, which
  shows the epoch map is fidelity-sensitive near the line.

**Nonlinear estimator confirmation is not justified.**

## 2. Scientific Question

Is ΔDOR complementarity for K_SRP **schedulable**, meaning found in a predictable, multi-day region of epoch and
baseline space? Or is it **encountered**, as isolated favourable geometries? The quantity is Δf_perp: the gain in the
fraction of K's measurement signature orthogonal to the six-state signature.

## 3. Entering Authoritative State

Verified:

| field | value |
|---|---|
| branch | `feature/phase17-r-k-srp-estimation` |
| HEAD | `d071d4c09d511de9eaa6e36b6ee32ba9b4056ba5` |
| tree | `b7d573ba9dc5ef6ca15eb1784b099a4049f2670f` |
| main = origin/main | `fea476f81dad07b3914e53eab709e10fa6e9d10b` |
| entering gates | HARDEN PASS, F1 PASS, infrastructure QUALIFIED_FOR_NEW_SCIENCE, SRP variational QUALIFIED, tracking design safe |
| entering suite (re-run) | 1406 collected / 1377 passed / 0 failed / 0 errors / 29 skipped |

The authoritative values (`studies/phase17/reports/phase17_authoritative_status.md`) are range 0.091502 and GC 5 nrad combined 0.172343.
D3 reproduced both at k = 0 to < 1e-6 (0.0915024 and 0.1723430).

## 4. F1 / HARDEN Dependencies

D3 uses:
- the F1 variational equations, through `propagate_nom48` (48-state);
- the checked canonical Earth, through `measurement_contracts`;
- the production visibility contract;
- production common-transmit ΔDOR rows;
- F-order STMs.

None of the forbidden paths was touched: legacy Earth, the elevation-only rule, frozen-Earth bridges, the pre-F1
model. `install_legacy_earth_guard()` is active in every worker. No production file changed.

## 5. Current Literature Contract

See `studies/phase17/reports/phase17_r1od3_literature_contract.md`, which cites:
- DSN-210 **Rev. F** (2026-07-02), verified from the source;
- DSN-107 Rev. F;
- CCSDS 500.1-G-2 (still current).

Every statement is labelled LITERATURE_RESULT, REPOSITORY_HYPOTHESIS or REPOSITORY_RESULT.

## 6. DSN Rev. F Update

The Rev. F change log lists only Figure 2 and Table 1, both band capabilities. All D3-relevant passages are identical
to Rev. E:
- the GM/GC geometry statements;
- the two-baseline requirement;
- the ~1 h pass of 5–10 scans;
- Tables 8–10;
- "only random error for the data weight when systematics are estimated".

One D2 arithmetic slip was found and corrected. The RSS of the Table 8 systematic rows is 0.0317 ns, not 0.033 ns.

## 7. Repository Measurement Contract

- **Observable.** ΔDOR D_S is the production common-transmit differential delay. Its state and K rows are
  quasar-independent.
- **Visibility.** Each row needs both stations visible: elevation ≥ 10° and no lunar occultation.
- **Range.** Production two-way range from DSN-3, 90 s cadence, recomputed at every epoch and orbit.

## 8. Predeclared Grid

`studies/phase17/configs/phase17_r1od3_predeclared_grid.json`: canonical SHA-256 `a651c4de…0003`, committed in c860759 before any
metric. The campaign and tests refuse a mismatching grid (`GridHashMismatch`). The grid was **not modified**.

**One `IMPLEMENTATION_BUG_FIX` (D3-IBF-01, recorded in the task state).** D2's `pair_geometry()` overwrote the
diagnostic qualifying-session count with its raw window count.
- **Affected:** the session-count predictor and two columns of the epoch-geometry CSV.
- **Not affected:** any schedule, metric, class, window or decision.

The campaign was re-run end to end. Every summary and decision field was identical, and only that predictor changed
(pooled ρ 0.02 → 0.17, class NONE both times).

## 9. Epoch Construction

et0_k = 857806356.99998 + k · 2551442.8896/24 s, for k = 0…23: 1.23 d and 15° of synodic phase apart.

- **Canonical epoch.** k = 0 is the canonical epoch exactly.
- **D2 epochs.** k = 6, 12, 18 are D2's E1–E3 (checked to 1 ms).
- **Held fixed.** The same J2000/LME state vector, 15 orbits, 90 s grid, and K. This is the D2 epoch method.
- **Recomputed at each epoch.** Sun, Earth, Earth orientation, stations and visibility.
- **Wraparound.** None.

## 10. Baseline Strategies

- **GC_ONLY**, **GM_ONLY**, and **GC_PLUS_GM** (stacked, no simultaneity).
- Madrid–Canberra is excluded; it is not a DSN baseline.

GC is feasible at all 24 epochs. GM has no qualifying session at k = 13–19, where the Moon's geocentric declination is
−18° to −27°. At k = 13 and 19 GM has only a dual-visibility window shorter than 20 min. That is consistent with DSN-210F's "declination above −15 deg … Goldstone-Madrid" condition. GC_PLUS_GM
is feasible where both are (17 epochs).

## 11. Session Definition

- **Window.** A contiguous run of dual-visible production rows.
- **Qualifying window.** ≥ 20 min.
- **Session.** ≤ 60 min, centred in the window (DSN-210F: a pass of about 1 h).
- **Normal points.** Three, at 1/6, 1/2 and 5/6 of the session's rows (the D2/CB13 abstraction). Each normal point lies
  inside its session and its window (tested).

## 12. Schedule Patterns

- **Single baseline:** FIRST, MID (window midpoint closest to the arc midpoint), LAST, FIRST_PLUS_LAST, ALL_FEASIBLE.
- **Joint:** FIRST_GC+FIRST_GM, FIRST_LAST_GC+FIRST_LAST_GM, ALL_GC+ALL_GM.

Patterns are chosen by chronology only; no K input. The **primary** patterns (ALL) drive the decision tree; the others
are secondary.

## 13. Resource-Conflict Model

The rule is Goldstone complex-level no-overlap, with GC priority. Antenna identity is not represented, which is a
labelled limitation.

**Result: `RESOURCE_CONFLICT = NO` at every epoch and for every joint pattern.** GC and GM dual-visibility windows never
overlap in time: a Goldstone–Canberra overlap and a Goldstone–Madrid overlap occur at different hours of the Earth's
rotation. No session was dropped, and no primary joint schedule double-books.

## 14. Noise Semantics

- **Primary:** constant delay, σ = 0.048 ns per normal point (DSN-210F Table 9).
- **Continuity diagnostic only:** the historical angular 5 nrad, used for the canonical reproduction.

## 15. Calibration / Bias Model

| case | treatment |
|---|---|
| C0 | ideal |
| C1 | one delay bias per session, prior σ_b = {0.5, 1, 2} × 0.0317 ns (nominal ×1) |
| C2 | free session biases |

The data weight is always the random 0.048 ns, so there is no double counting. The Gaussian session-bias model is a
REPOSITORY choice. Bias ownership is machine-audited, including a permutation regression.

## 16. Quasar / Catalog Treatment

The formal K design is quasar-independent. No quasar sweep was run.

**Operational overlay:** the **DSN-107F** ICRF3-SX coordinates (4488 parsed; 3C 273 matched to 0.6 mas). All **202**
primary sessions have a catalog source within 0.1 rad of the geocentric spacecraft direction: median nearest 1.39°,
maximum 3.49°.

`REAL_SOURCE_AVAILABILITY = COORDINATE_AVAILABLE_ALL_SESSIONS`. Flux, SNR and structure were not assessed.

## 17. Primary Epoch Geometry

`studies/phase17/results/phase17_r1od3_epoch_geometry.csv`:
- The Sun–Moon–Earth angle sweeps 7.7°–174.7°.
- The lunar declination spans −27.4° to +27.6°.

**Caveat (LITERATURE_RESULT).** DSN-210F calls geometry "good" only when the Sun–Earth–spacecraft angle exceeds 10°.
Near k = 23, the lunar elongation is about 5°, so real performance there would be worse than the modelled random
error.

## 18. Range-Only Baseline

Range-only f_perp varies strongly with epoch, from 0.064 to 0.275. That is why Δf_perp is always taken against the same
epoch's range baseline.

## 19. Goldstone–Canberra Results

Primary GC_ONLY (ALL sessions) under C1 nominal:

| class | epochs |
|---|---:|
| STRONG | 0 |
| MODEST | 7 |
| MAGNITUDE_ONLY | 14 |
| DIRECTION_LOSS | 3 |

The largest Δ is +0.0486, at k = 5.

**Canonical k = 0:**

| treatment | Δf_perp | class |
|---|---:|---|
| C0 | +0.059 | STRONG |
| C1 nominal | +0.0328 | MODEST |
| C2 free | −0.027 | — |

The D2 dense-row STRONG result (+0.0808) needed every dual-visible row and an ideal calibration.

## 20. Goldstone–Madrid Results

GM_ONLY is STRONG at k = 6, 9, 10 (+0.050, +0.078, +0.086). It is MODEST at 3 epochs and never DIRECTION_LOSS. The
region k ≈ 5–10 (phase 75–150°) is GM's favourable region. There its dense ΔDOR-only f_perp reaches 0.32–0.38, against
≤ 0.11 elsewhere.

This revises D2's "GM never strongly complementary", which D2 tested only at k = 0, 6, 12, 18.

## 21. Joint-Baseline Results

`studies/phase17/results/phase17_r1od3_joint_baseline_results.csv`. GC_PLUS_GM is STRONG at k = 9, 10 (+0.069, +0.087), where GM
alone is already STRONG. Across epochs, the median (joint − GC) under C1 is +0.0002. The joint never exceeds the better
single baseline.

**`JOINT_BASELINE_EFFECT = BROADENS_WINDOW`.** This is the literal predeclared rule: joint STRONG count (2) ≥ GC STRONG
count (0) + 2. **Mechanistically it is not synergy.** The broadening is GM's standalone strength imported into the GC
schedule.

The free-bias retention difference, joint minus GC, has median −0.079. Adding GM sessions does **not** help calibration
robustness. H2 is not supported as a mechanism.

## 22. Session-Placement Results

C1 nominal, 24 epochs (`summary.placement_and_count`):

| pattern | strong epochs | median Δ |
|---|---:|---:|
| GC FIRST | 2 | +0.021 |
| GC MID | 2 | +0.017 |
| GC LAST | 0 | +0.012 |
| GM FIRST | 3 | +0.019 |
| GM FIRST_PLUS_LAST | 4 | +0.032 |

**Secondary observation, multiple-comparison-exposed.** GM FIRST_SESSION, and the joint FIRST+FIRST pattern, are STRONG
at **k = 5, 6, 7**: three consecutive epochs, with small margins of +0.053 to +0.069. This comes from 1 of 13 patterns
examined, is not a primary pattern, and was not robustness-tested. It is recorded as a hypothesis for a future
predeclared phase, not as a D3 window.

## 23. Session-Count Results

More sessions is not better for direction:

| GC pattern | median Δ |
|---|---:|
| FIRST (1 session) | +0.021 |
| FIRST_PLUS_LAST (2) | +0.012 |
| ALL (median 2.5) | +0.010 |

Additional sessions add K information that the state can mimic, which raises ‖b_K‖ without raising the orthogonal
part. σ_K/K still improves with more sessions (magnitude, not direction).

## 24. Calibration-Prior Results

STRONG counts summed over the three primary strategies:

| treatment | STRONG cells |
|---|---:|
| C0 | 9 |
| C1 ×0.5 | 7 |
| C1 ×1 | 5 |
| C1 ×2 | 0 |
| C2 free | 0 |

- **IDEAL_VS_NOMINAL:** NOMINAL_RETAINS_MOST (median Δ_C1/Δ_C0 = 0.87, over GC cells with Δ_C0 > 0.02).
- **NOMINAL_VS_FREE:** FREE_RETAINS_MOST (median Δ_C2/Δ_C1 = 0.79).

The medians look mild, but the STRONG cells are exactly the ones that lose the most. The ratio labels summarise many
small-Δ cells. The class counts show the strong-gain population is calibration-limited: doubling the prior already
removes every STRONG cell.

**Bias posteriors (GC ALL):**

| treatment | σ_b | max \|corr(K, bias)\| |
|---|---:|---:|
| C1 | 0.026 ns | 0.12 |
| C2 | 0.059 ns | 0.28 |

H3 is supported at class level.

## 25. Tracking-Cost Results

Median session-hours:

| pattern | session-hours |
|---|---:|
| single session | 0.95–1.0 |
| GC ALL | 2.0 |
| joint ALL | 4.7 |

`studies/phase17/results/phase17_r1od3_delta_vs_session_hours.png` shows no monotone Δ gain with hours. Station-hours are 2 × session-hours.

## 26. Favorable-Window Analysis

Predeclared definition: STRONG under C1 nominal for ≥ 3 consecutive epochs, primary pattern.

**Windows found: none**, for any primary strategy. There are none under C0 (ideal) either, and none under C2.

## 27. Window Width and Continuity

The longest nominal STRONG run under a primary pattern is 2 epochs (GM and joint, k = 9–10), about 2.5 days. GC_ONLY
has no STRONG epoch.

## 28. Robustness to Adjacent Epochs

There is no window, so this does not apply. Isolated STRONG epochs fail the adjacent-epoch criterion by definition.

## 29. Candidate Geometric Predictors

Predeclared, Spearman ρ against the C1 nominal Δ (primary pattern):

| predictor | ρ GC | ρ GM | ρ pooled | strength |
|---|---:|---:|---:|---|
| ΔDOR-only dense f_perp | +0.30 | +0.47 | +0.39 | WEAK |
| Sun–Moon–Earth angle | −0.30 | −0.63 | −0.47 | WEAK |
| range LOS vs SRP direction | +0.30 | +0.63 | +0.46 | WEAK |
| \|cos(baseline, K plane-of-sky)\| | −0.47 | +0.65 | −0.12 | NONE |
| range-only f_perp | +0.15 | +0.17 | +0.14 | NONE |
| projected baseline | +0.09 | −0.23 | −0.16 | NONE |
| qualifying session count | +0.11 | +0.22 | +0.17 | NONE |
| normal-point count | +0.11 | +0.22 | +0.17 | NONE |

**`BEST_SUPPORTED_PREDICTOR = NONE_ADEQUATE`.** The best pooled |ρ| is 0.47, which is WEAK.

## 30. DDOR-Only Separability as Predictor

For GM, the ΔDOR-only separability is the most interpretable signal: GM's STRONG epochs are exactly where its
ΔDOR-only f_perp is 0.32–0.38. For GC, ΔDOR-only f_perp peaks at k = 0, 6, 11, 16, 21, 22 (≈ 0.13–0.16) without
matching gains. A block that is K-separable on its own is necessary-looking but not sufficient: its orthogonal part
must also be orthogonal to what **range** already spans.

## 31. Physical Interpretation

ΔDOR adds plane-of-sky angular information along its baseline. Whether that information is K-specific depends on the
relative orientation of three things:
- the baseline;
- the spacecraft's plane-of-sky K displacement;
- the subspace already spanned by the range state signature.

All three rotate with lunar phase, lunar declination and orbit geometry. The favourable alignments are short: 1–3 grid
steps, about 1–4 days. Session biases absorb precisely the low-frequency part of the ΔDOR signal that carries the
K-specific component.

## 32. Orbit-Geometry Robustness

`studies/phase17/results/phase17_r1od3_orbit_robustness.json`. There is no primary window, so
`ORBIT_ROBUSTNESS_CLASS = NO_WINDOW_TO_TEST`. The descriptive results at k = 0, 6, 12, 18 are still informative:
- **k = 0, GC_ONLY:** STRONG for 4 of 5 representative orbits (O2–O5), MODEST for O1, and MODEST for the canonical
  orbit itself.
- **k = 6:** GC STRONG for O1, O2, O3 and O5.
- **k = 12 and 18:** mostly DIRECTION_LOSS or MAGNITUDE_ONLY.

Orbit geometry moves cells across the threshold as much as epoch does. The canonical orbit is not a favourable
representative.

## 33. High-Fidelity Dynamics Check

`studies/phase17/results/phase17_r1od3_high_fidelity.json`. The force model is GL1800F 50×50 with Earth and Sun third body, SRP, and
the F1 STM. Cases were chosen by rule:

| case | simplified | high fidelity |
|---|---|---|
| HF1 (k = 0 GC) | MODEST +0.033 | MODEST +0.027 |
| **HF2** (fallback rule: GC epoch with largest Δ, k = 5) | MODEST +0.0486 | **STRONG +0.0852** |
| HF3 (k = 2 GC) | DIRECTION_LOSS | DIRECTION_LOSS |
| HF4 (k = 0 joint) | MODEST | MODEST |

**`TRACKING_DESIGN_HIGH_FIDELITY_CLASS = CHANGES_PRIMARY_INTERPRETATION`.** This is the predeclared rule: HF2 crosses a
class boundary by more than 0.005.

HF2 was selected precisely because it sat just below the threshold. Fidelity moved it above. A single epoch changing
class does not create a window, and the grid forbids further escalation. It does show that **the exact location of
favourable epochs is sensitive to dynamics fidelity** near the threshold, which reinforces the "narrow and fragile"
conclusion.

## 34. Real-Source Availability Overlay

See §16. The overlay covers coordinates only, and every session passes. Actual quasar suitability (flux ≥ correlated
flux needed, structure) is **not established**.

## 35. Operational Limitations

- No DSN scheduler: no antenna identity, slewing, data-return latency or allocation competition.
- The Gaussian session-bias abstraction.
- The Sun-elongation degradation near k = 23 is not modelled.
- X-band random budget only.
- 90 s row grid; normal points snapped to rows.

## 36. Absolute K Identifiability

Every primary cell under C1 nominal (58 feasible) is **WEAK_K_SEPARATION**. The best combined f_perp is 0.299 over
primary cells, and 0.328 over any pattern or calibration.

The weakest right-singular vector's K component has median 0.036 and minimum 0.0026. These are small because the
scaled design weights K differently; K is not separated. `ABSOLUTE_K_IDENTIFIABILITY_CLASS = WEAK_K_SEPARATION`.

## 37. Relative Gain vs Absolute Separation

Even the STRONG relative cells (Δ up to +0.087) leave K absolutely WEAK. A relative gain here means "ΔDOR adds some
K-specific direction". It never means "K becomes identifiable".

## 38. Regression Tests

`tests/test_r1od3_tracking_design.py` has 11 tests:
- grid hash, and refusal of a tampered grid;
- the uniform epoch grid, with the canonical and D2 epochs included;
- sessions: minimum window, centring, normal points;
- deterministic patterns;
- the Goldstone rule, including a closed-interval touch;
- row ownership;
- bias ownership;
- the session-label permutation regression;
- delay weights and priors;
- canonical reproduction under F1;
- the full contract set at the canonical epoch.

## 39. Full Suite

The normal suite ran on the committed tree (50a728f):

| collected | passed | failed | errors | skipped | subtests |
|---:|---:|---:|---:|---:|---:|
| 1417 | **1388** | **0** | **0** | 29 | 146 passed |

- **Count.** The 11 extra tests versus entry are the new D3 tests.
- **Skips.** They are the same 29 qualified opt-in/data skips.
- **Slow tests.** Not re-run: D3 changes no production path they exercise.
- **Test groups.** HARDEN, F1, D2 and measurement tests all run inside the full suite.

`NEW_SCIENTIFIC_REGRESSIONS = 0`, `UNKNOWN_NONPASSES = 0`.

## 40. Final Diff Review

`git diff d071d4c..HEAD`: 20 files, **all additive, all D3**:
- the grid writer, core, campaign and tests;
- the literature contract and this report;
- the `phase17_r1od3_*` artifacts and two analytical figures.

No `lunar_od/` production file, no existing test and no other document changed. The scratch download of DSN-107F was
removed; its SHA-256 is recorded in the summary.

`FINAL_DIFF_REVIEW = PASS`.

## 41. Academic Interpretation

The literature establishes that ΔDOR measures plane-of-sky angle and that baseline orientation matters. The open
question was whether the resulting K_SRP signature can be **scheduled** outside the range-generated state subspace.

For this lunar campaign the answer is: only in short, geometry-specific windows. They are sensitive to calibration,
orbit geometry and force fidelity, and no simple pre-observation geometric quantity predicts them. Complementarity is
real but **encountered, not reliably designable**, with these tools and baselines.

That is a stronger and more useful negative result than one favourable case.

## 42. Operational Interpretation

A navigation team choosing when and where to request ΔDOR could not, from this analysis, count on a multi-day window of
K-specific information:
- Goldstone–Madrid around lunar phase 135–150° from this epoch, and the canonical GC epoch, are the only places where
  gains appear.
- Calibration of the session biases is as important as geometry.

The information that ΔDOR does add is mostly magnitude (σ_K improves), not direction.

## 43. Remaining Limitations

- Simplified dynamics for the primary sweep; high fidelity only at 4 rule-selected cases.
- One synodic month, one orbit for the sweep.
- A schedule abstraction.
- Coordinate-only catalog check.
- Linear information metrics only, with no estimator run.

## 44. Remaining Unknowns

- Whether the k = 5–7 GM secondary run is robust. It would need its own predeclared test.
- Whether the high-fidelity map differs systematically from the simplified one away from the tested cases.

Neither is an unknown in the D3 characterization itself: both are explicitly outside what D3's predeclared grid tests.

## 45. What This Phase Actually Established

- A dense, predeclared, fully contract-verified map of ΔDOR K-direction gain versus epoch, baseline, schedule and
  calibration.
- No robust or consecutive window under primary operational schedules.
- Calibration-limited strong gains.
- GM gains around phase 135–150°.
- No adequate geometric predictor.
- Fidelity-sensitive threshold cases.
- A corrected DSN-210F literature contract.

## 46. What This Phase Did NOT Establish

- Real spacecraft K identifiability.
- DSN schedule availability.
- Quasar suitability beyond coordinates.
- Flight or mission covariance.
- Any estimator result.
- The robustness of the secondary k = 5–7 GM run.

## 47. Decision Tree

Applied in the predeclared order A → B → C → D:
- **A** requires a robust window; there are no windows. Not met.
- **B** requires C1-nominal STRONG at ≥ 2 epochs: there are 5 STRONG cells (GM 3, joint 2) with no robust window.
  **B matches.**
- C and D were not reached. For completeness, there are no C0 windows either.

## 48. Final Verdict

The characterisation is complete. Every strict criterion is met:
- the entering gates and authoritative baselines were verified and reproduced;
- the Rev. F contract was refreshed;
- the grid was committed before any metric, with a uniform 24-point epoch grid;
- visibility, coverage, session, ownership and weight contracts were verified;
- GC, GM and GC+GM were all characterised, with no double-booking;
- range was recomputed at every geometry;
- all metrics and both label sets were produced;
- the window rule was applied unedited;
- orbit robustness and high fidelity were resolved;
- the independent rebuild agreed to 4e-15;
- the tests and full suite pass, and the diff is clean.

**`PHASE17_R1O_D3_GATE = PASS`** (characterisation complete).
**`PRIMARY_CLASS = DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS`**: a negative tracking-design
result.

## 49. Whether Nonlinear Estimator Confirmation Is Justified

**NO.** The gain is narrow, calibration-fragile and fidelity-sensitive, and it is never absolutely identifying. A
nonlinear K-recovery campaign would test a geometry that D3 shows is not reliably schedulable.

A narrowly scoped confirmation is not scientifically justified by D3's primary results. If pursued at all, it should
first be preceded by a predeclared robustness test of the secondary GM k = 5–7 run.

## 50. Exact Next Action

**Stop (D3 hard stop).** Options for a future, separately authorised phase:
- a predeclared robustness test of the GM phase-75–150° region, covering orbit, fidelity and calibration;
- an assessment of alternative observables or parameterisations for K.

Nonlinear estimation is not recommended on the present evidence.
