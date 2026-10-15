# PHASE 17-FXR — EARTH-EPOCH FIXTURE ERRATUM AND HISTORICAL BASELINE REQUALIFICATION

> ### ERRATUM (Phase 17-R1O-D2, enforced by Phase 17-HARDEN): lunar occultation
>
> **0.3204 is superseded.** The range + production ΔDOR 5 nrad value 0.3204 (and its surrogate twin 0.3203) is labelled
> **`EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`**. FXR corrected its Earth epoch, but
> its ΔDOR rows were still accepted on station elevation alone, without a lunar-occultation test. As a result,
> **93 of 263 rows (35%) had the spacecraft behind the Moon** for at least one station.
>
> **Authoritative values** (Phase 17-R1O-D2, canonical Goldstone–Canberra, 5 nrad, both stations visible under the
> production elevation + occultation contract):
>
> | quantity | value |
> |---|---:|
> | range-only `f_perp` | **0.091508** |
> | range + ΔDOR `f_perp` | **0.172351** |
> | gain Δ`f_perp` | **+0.0808** (STRONG_DIRECTION_GAIN) |
> | σ_K/K | **7.46%** |
>
> **R1O surrogates under the same contract** (Phase 17-HARDEN, `studies/phase17/results/phase17_harden_visibility_requalification.json`):
>
> | surrogate | 5 nrad `f_perp` | 1 nrad `f_perp` | rows |
> |---|---:|---:|---|
> | ΔDOR-like | 0.1722 | 0.1381 | 170 kept of 263 |
> | Earth LOS | 0.0919 | — | 360 kept; 230 of 590 were occulted |
>
> The R1O-D-era bridge surrogate uses an Earth frozen at t = 0 and is historical only. On the legacy (elevation-only) path, every FXR number
> still reproduces bitwise. The single current status is in `studies/phase17/reports/phase17_authoritative_status.md`. The defect is
> HARDEN-D04 in `studies/phase17/reports/phase17_harden_defect_ledger.md`. The text below is kept unchanged as the historical record.


> **The question, answered first.** After the Earth-position epoch inconsistency is corrected,
> which R1M–R1O-R results are superseded, which conclusions survive, and what is the canonical
> K_SRP range baseline?
>
> - **Superseded:** every absolute number computed on the published range fixture (§28).
> - **Survive:**
>   - all mathematical and software qualifications (R1COV QR covariance, R1O-R column-major STM,
>     production ΔDOR and optical models);
>   - R1M's core finding: range-only multi-arc processing does not separate K;
>   - R1O-OPT's marginal optical gain;
>   - the Earth-LOS negative control.
> - **Changed:** the ΔDOR conclusion. Production ΔDOR was "magnitude only" on the published
>   fixture. On the consistent geometry it is a **strong direction gain**:
>   - `f_perp` goes 0.0915 → 0.3204 at 5 nrad;
>   - `sigma_K/K` goes 25.4% → 4.0%;
>   - the cause is isolated to the 5.83-day epoch offset.
>
>   R1O-R's corrected surrogate conclusion ("no candidate improves on range") changes with it.
> - **Canonical baseline from now on:** 15 orbits, DSN-3, 90 s cadence, 960 observations,
>   `f_perp = 0.091508`, `theta_K = 5.25°`, `I(K|x) = 1.5497e+05`, `sigma_K/K = 25.40%`
>   (`SYNTHETIC_CAMPAIGN_TRUTH`).

The central erratum map is `studies/phase17/reports/phase17_fxr_earth_epoch_fixture_erratum.md`.

---

## 1. Executive Summary

Phase 17-GEO found that the canonical range-analysis fixture mixed two epochs. FXR proves the
defect from the code and the data files, repairs it without touching production physics, keeps the
old fixture reproducible on request, and re-runs every affected historical analysis twice (old and
corrected fixture) through the same historical code.

- **The defect.** `examples/phase17_r1m_core.build_range_arc` fed the production two-way range model
  an Earth table, `od_gravity_covariance_campaign.get_earth_pos`, with these properties:
  - its t = 0 is ET 857302356.99998;
  - that is **504000 s = 5.8333 days** before the epoch the same call used for the Earth rotation,
    the Sun and the spacecraft (857806356.99998, 2027-03-08T19:31:27.814 UTC);
  - `numpy.interp` holds it constant, position *and* velocity, after 28260 s (3.9986 orbits).

  The Moon→Earth line of sight was therefore 71°–83° wrong. On the published 15-orbit arc,
  590 of the 782 observations were formed after the clamp.
- **Attribution.** Four arms share an identical truth trajectory; only the Earth given to the
  measurement model differs. Repairing the epoch moves `f_perp` by −0.115 (no clamp) or −0.167
  (clamp present). Repairing the clamp moves it by +0.006 or −0.046. **The epoch offset dominates.**
- **Reproduction.** The legacy path reproduces:
  - every committed historical artifact: bit for bit, or to 1 ulp where the historical script
    itself disagreed with itself by 1 ulp;
  - GEO's cached design matrices for all four arms, bitwise.
- **Requalification outcomes:**
  - R1COV: algorithm valid, numbers superseded.
  - R1O-R: STM repair valid, context numbers superseded.
  - R1O-OPT: class confirmed (MARGINAL_DIRECTION_GAIN).
  - R1M: PARTIAL.
    - Confirmed: K is not separated by range-only multi-arc processing; nothing is practically
      identifiable; `MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = NO`.
    - Not confirmed: the ordering "M3-linked beats contiguous G3" (780 vs 1856).
  - **R1O-D: CHANGED**, and R1O's corrected-surrogate conclusion with it.
- **Protection.** `tests/test_fxr_earth_epoch_fixture.py` (10 tests) compares the Earth states the
  production model actually received against SPICE at the rotation epoch, and pins both baselines.

`PHASE17_FXR_GATE = PASS`. `PRIMARY_CLASS = HISTORICAL_EARTH_EPOCH_FIXTURE_DEFECT_REPAIRED_AND_PRIOR_BASELINES_REQUALIFIED`.
The ΔDOR reclassification puts the project on **decision-tree Case B** (§44).

## 2. Why FXR Exists

Six reports (R1M, R1COV, R1O, R1O-D, R1O-OPT, R1O-R) quote numbers built on one fixture, which GEO
showed to be physically inconsistent. FXR does not look for new K_SRP physics. It decides, from
reruns rather than assumptions, which of those numbers and conclusions still stand.

## 3. What a Fixture Is

A fixture is analysis-side scenario setup. It covers the epoch, the ephemerides handed to the
models, the stations, the tracking schedule, the cadence, the noise, the force-model switches and
the arc duration. Here the production models are correct. The fixture handed them inputs from
inconsistent epochs.

## 4. Physical Meaning of the Earth-Epoch Error

Two-way range measures the spacecraft position projected on the Earth–spacecraft line of sight. It
therefore sees a K_SRP-driven trajectory perturbation only through that projection.

At the correct epoch the Sun–Moon–Earth angle is **174.4°**: the Earth is almost exactly
anti-sunward from the Moon (near new Moon). The SRP acceleration is also roughly anti-sunward, so
the range line of sight lies close to the SRP direction. An initial-state correction can then
imitate most of the K signature: `f_perp = 0.0915`, `theta_K = 5.25°`.

The published fixture put the Earth where it was 5.83 days earlier: Sun–Moon–Earth angle 114.2°,
with the line of sight rotated by 71°–83°. That projection happens to be more favourable to K
(`f_perp = 0.2523`), but it is not the geometry of the scenario.

## 5. Entering Repository State

| field | value |
|---|---|
| `ACTIVE_WORKTREE` | `C:/Users/erayh/Documents/Python/lunar-orbit-determination/python_port_phase17r0` |
| `START_BRANCH` | `feature/phase17-r-k-srp-estimation` |
| `START_HEAD` | `40bc1533fb3056b626b3132bbd71a153456c16da` ("chore(repo): audit and clean local repository artifacts") |
| `START_TREE` | `121fe64ffb5b35399bd16c7427d0d3f7f0f42c56` |
| `MAIN_HEAD` = `ORIGIN_MAIN_HEAD` | `fea476f81dad07b3914e53eab709e10fa6e9d10b` (as expected; unchanged) |
| `WORKTREE_STATUS` | clean apart from untracked files |
| `UNTRACKED_FILES` | `examples/phase17_r1od_fd_convergence.py`, `examples/phase17_r1od_qualification.py` (R1O-D orphans; left untouched) |

`FXR_INPUT_GATE = PASS`. The `Grad` path used by the analysis scripts is the documented hidden
junction to `lunar-orbit-determination` (hygiene report §334). Runs used
`PYTHONPATH=<worktree>;<od_covariance_campaign>/06_scripts`.

## 6. Graphify Refresh and Discovery

`graphify update .` was run at `40bc153`: graphify 0.9.65; 6795 nodes and 18476 edges; the GEO
scripts were now included. It was refreshed again after the FXR edits: 6896 nodes. The Earth
ephemeris is now a node (`measurement_earth_ephemeris`) wired into `build_range_arc`, so future
impact queries can see this dependency.

- `affected "build_range_arc"` listed 25 caller scripts.
- `path` confirmed that the fixture wires `generate_two_way_range_measurements` and
  `_two_way_range_k_srp_column`.
- **`explain "get_earth_pos"` found only five test-local helpers.** The campaign table is not a
  node: it lives in a separate repository and enters as a runtime callable. Its direct consumers
  were found by Grep.

Details: `studies/phase17/results/phase17_fxr_graphify_discovery.md`. `GRAPHIFY_DISCOVERY_GATE = PASS`.

## 7. Manual Runtime Configuration Audit

Every geometric epoch was traced in source and then verified at runtime on the built arcs
(`studies/phase17/results/phase17_fxr_epoch_audit.json`, `runtime_audit`):

| quantity | epoch (published fixture) | epoch (canonical) |
|---|---|---|
| spacecraft state, propagation t = 0 | `phase7_et0` (`campaign_initial_state` = manifest `x0_truth_phase7`, bitwise) | same |
| Sun (SRP) | `phase7_et0 + t` | same |
| Earth rotation (`sxform`, visibility transforms) | `pass_geo.et0_s + t` = `phase7_et0 + t_start + t_local` | same |
| **Earth position/velocity (station centre, visibility, light time)** | **`C.ET0 + t_local` = `phase7_et0 − 504000 s + t_local`, held after 28260 s** | **`pass_geo.et0_s + t_local`** |
| range receive tag t3; t1, t2u, t2d | `pass_geo.et0_s + obs[:,0]`, solved through the same station providers | same |
| Earth third-body force | none (`mu_earth = 0`; constant placeholder vector) | same |

Measured:
- **canonical:** Earth origin − rotation origin = 0 s; stored Earth states equal SPICE at the
  rotation epoch exactly (max difference 0 m, 0 m/s).
- **legacy:** Earth origin − rotation origin = −504000 s; stored Earth states equal SPICE 504000 s
  early to 0.97 m (linear interpolation of the 60 s table); against the measurement epoch they are
  off by up to 3.9e8 m.

`FXR_RUNTIME_EPOCH_AUDIT = PASS`.

## 8. Exact Defect Location

```
DEFECT_FILE      = studies/phase17/scripts/phase17_r1m_core.py
DEFECT_SYMBOL    = build_range_arc   (visibility and generate_two_way_range_measurements were
                   given od_gravity_covariance_campaign.get_earth_pos / get_earth_vel)
DEFECT_MECHANISM = an untagged Earth callable whose t = 0 is the campaign ET0 (ephemeris_data.mat
                   first_jd) was passed next to et0 = phase7_et0 = campaign ET0 + 504000 s; the
                   callable interpolates a 0..28260 s table with numpy.interp, which holds its end
                   values. The production model trusts that callable's t to share et0's origin.
```

The campaign repository knew about the offset: its own `phase8_setup.py` shifts the table
(`gep(t) = interp(t + OFF)`). The Phase 17 fixture imported the module's un-shifted
`get_earth_pos` instead.

Secondary Earth consumers that read the same table directly:
- the R1O surrogate ΔDOR (`phase17_r1o_core.build_ddor_arc`);
- the R1O-R Earth-LOS control and bridge;
- the R1O-D production-ΔDOR campaign, which **froze** `get_earth_pos(0.0)` for the whole arc;
- the R1O window probe and celestial control;
- three R1O-D qualification scripts.

Production code is not implicated.

## 9. Epoch-Offset Reconstruction

Both epochs were recovered from the files the code reads:

| field | value | source |
|---|---|---|
| `SCENARIO_ET` | 857806356.99998 | `phase7_measurement_geometry_manifest.json:phase7_et0` via `campaign_epoch()` |
| `HISTORICAL_EARTH_TABLE_START_ET` | 857302356.99998 | `(ephemeris_data.mat:first_jd − 2451545)·86400` = `od_gravity_covariance_campaign.ET0` (equal) |
| manifest `canonical_et0` / `offset_s` | 857302356.99998 / 504000.0 | same manifest |
| `DELTA_ET_SECONDS` | **504000.0** (exact; equals the manifest offset) | |
| `DELTA_ET_DAYS` | **5.833333** | |

The offset enters the geometry as a Moon→Earth direction error:
- 70.8° at t = 0;
- 71.1° at the clamp;
- 82.6° at the end of the arc.

`EARTH_EPOCH_OFFSET_REPRODUCTION_GATE = PASS`.

## 10. Earth-Table Clamp Characterization

| field | value |
|---|---|
| table | 472 nodes, 60 s, `T_OBS = arange(0, 4·T + 30, 60)` |
| `TABLE_DURATION` | 28260 s = 3.9986 orbits (not exactly 4 T) |
| `ARC_DURATION` | 106011.9 s = 15 orbits (grid end 106020 s) |
| `CLAMP_START_ET` | 857330616.99998 (table-epoch ET of the frozen Earth); at scenario t = 28260 s, i.e. scenario ET 857834616.99998 |
| `CLAMP_START_ORBIT_NUMBER` | 3.9986 (inside the 4th orbit) |
| behaviour after the end / before the start | **held constant** (last / first node). Not extrapolated, not repeated. Verified |
| velocity after the clamp | also held (967 m/s), so the frozen position and the non-zero velocity are inconsistent |
| side effect in production interpolation | the Hermite station interpolation between frozen nodes wobbles by up to 8.4 km |
| published W15 observations after the clamp | 590 / 782 (75%); 73% of the arc |

`EARTH_TABLE_CLAMP_CHARACTERIZATION_GATE = PASS`.

## 11. Historical Fixture Reproduction

The legacy arm (A) reproduces the published W15 baseline exactly, against committed
`studies/phase17/results/r1od_campaign_summary.json`:

- n_obs 782;
- `f_perp` 0.2522919965229832;
- `theta_K` 14.613182283011557°;
- `I(K|x)` 1162817.0907819087;
- `sigma_K/K` 0.09273513250655349.

Its h_x0, h_k, w and obs are bitwise equal to GEO's cached `FIXTURE_PUBLISHED`. Every phase task's
legacy arm also reproduces its phase's committed artifact (§16–§21).
`HISTORICAL_FIXTURE_REPRODUCTION_GATE = PASS`.

## 12. Four-Way Epoch/Clamp Attribution

Configuration: 15 orbits, Goldstone + Madrid + Canberra, 90 s cadence. The truth trajectory
(nom48) is identical in all arms (same SHA-256), and each arm is bitwise equal to the corresponding
GEO cache.

| arm | obs | `f_perp` | `theta_K` | `I(K\|x)` | `sigma_K` | `sigma_K/K` | `‖b_K‖` | max\|corr(K,x)\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A offset + clamp (published) | 782 | 0.252292 | 14.613° | 1.162817e+06 | 9.2735e-04 | 9.27% | 4274.2 | 0.337 |
| B offset only | 768 | 0.206667 | 11.927° | 7.290480e+05 | 1.1712e-03 | 11.71% | 4131.5 | 0.569 |
| C clamp only | 968 | 0.085728 | 4.918° | 1.233540e+05 | 2.8472e-03 | 28.47% | 4096.9 | 0.393 |
| D canonical | 960 | 0.091508 | 5.250° | 1.549742e+05 | 2.5402e-03 | 25.40% | 4302.0 | 0.646 |

Change in `f_perp` caused by repairing each defect:

| effect | value |
|---|---:|
| total (D − A) | −0.160784 |
| epoch repair, no clamp (D − B) = `EPOCH_EFFECT_DELTA_F_PERP` | **−0.115159** |
| epoch repair, clamp present (C − A) | −0.166564 |
| clamp repair, correct epoch (D − C) = `CLAMP_EFFECT_DELTA_F_PERP` | **+0.005780** |
| clamp repair, offset epoch (B − A) | −0.045625 |
| interaction | +0.051405 |
| Shapley main effects: epoch / clamp | −0.140862 / −0.019923 |

GEO's reading, "the epoch offset causes most of the error; the clamp is secondary", is confirmed.

## 13. Corrected Canonical Fixture

`build_range_arc(..., earth_fixture=EARTH_FIXTURE_CANONICAL)` is the default. The measurement Earth
is an `EarthEphemeris` with:
- `et_origin = pass_geo.et0_s` (the production rotation origin);
- explicit coverage `[0, t_local[-1]]`;
- frame J2000, centre MOON, units m and m/s;
- exact SPICE `spkezr(EARTH, et_origin + t, J2000, NONE, MOON)`;
- **no clamp**: any request outside the coverage raises `EarthEphemerisCoverageError`.

Windowed arcs use the window's own origin. `earth_position_mci` and `earth_velocity_mci` give the
same Earth, in campaign-epoch seconds, to the analysis helpers that place stations outside the range
model. The arc records the ephemeris it used (`ArcFixture.earth`).

## 14. Legacy Reproduction Path

`EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED`:
- is never a default;
- emits `SupersededEarthFixtureWarning`;
- delegates to the campaign table exactly as before, so it is bitwise-reproducing.

The two single-defect diagnostics (`DIAGNOSTIC_OFFSET_EPOCH_UNCLAMPED`,
`DIAGNOSTIC_CORRECT_EPOCH_CLAMPED`) are attribution-only and restricted to arcs that start at the
campaign epoch. GEO's two published-fixture arms now request the legacy fixture explicitly.
`LEGACY_REPRODUCTION_PATH_AVAILABLE = YES`.

## 15. Permanent Regression Protection

`tests/test_fxr_earth_epoch_fixture.py` has 10 tests. It skips when the frozen campaign data is
absent.

- **Behavioural contract:**
  - the canonical `pass_geo.earth_pos/vel` equal SPICE at `pass_geo.et0_s + t` bitwise;
  - `arc.earth.et_origin == pass_geo.et0_s == campaign ET0`, including for a window starting at
    2.5 orbits;
  - coverage spans the arc, `clamp_after = inf`, and the Earth keeps moving after 28260 s;
  - requests outside coverage raise;
  - the canonical fixture is the default; legacy warns; an unknown fixture raises;
  - the station-placement helper uses the measurement epoch;
  - the truth trajectory does not depend on the fixture.
- **Legacy provenance:** the offset is exactly 504000 s; the node value equals SPICE 5.83 d early;
  the Earth is frozen after 28260 s; published baseline 782 obs with schedule SHA-256 `7358d8d3…`,
  `f_perp` 0.25229, `sigma_K/K` 9.2735% (rel 1e-9).
- **Canonical baseline:** 960 obs with schedule SHA-256 `40c64341…`, `f_perp` 0.091508,
  `sigma_K/K` 25.402% (rel 1e-9).

`EARTH_EPOCH_FIXTURE_REGRESSION_GATE = PASS` (10/10).

## 16. R1M Requalification

Same code (`phase17_r1m_multi_arc_study`, `phase17_r1m_addendum` helpers) and same predeclared
schedules. The legacy arm reproduces `r1m_multi_arc_results.json` and `r1m_model_a_fairness.csv`
exactly.

**Single arcs** (default single station, 60 s):

| case | published: obs, station, `f_perp`, `I(K\|x)`, σ_K/K | corrected: obs, station, `f_perp`, `I(K\|x)`, σ_K/K |
|---|---|---|
| G0 1.3 orb | 82, Canberra, 0.00889, 1.224, 9040% | 108, Goldstone, 0.05083, 26.91, 1928% |
| G1 2.0 orb | 144, Canberra, 0.2246, 2659, 194% | 146, Goldstone, 0.06733, 163.9, 781% |
| G2 3.0 orb | 215, Canberra, 0.2434, 9355, 103% | 218, Goldstone, 0.07305, 615.1, 403% |
| G3 5.0 orb | 307, Canberra, 0.2690, 35203, 53% | 291, Canberra, 0.05936, 1856, 232% |

**Multi-arc** (DSN-3):

| case | published `I(K\|x)` / `f_perp` / σ_K/K | corrected `I(K\|x)` / `f_perp` / σ_K/K | corrected obs per window (Model A) |
|---|---|---|---|
| M1 A / B | 2.6e-3 / 0.008 / — ; 3479 / 0.199 / 170% | 3.4e-2 / 0.0075 / — ; 116.4 / 0.030 / 927% | 41, 72 |
| M2 A / B | 4.0e-3 / 0.008 ; 36588 / 0.300 / 52% | 3.4e-2 / 0.0066 ; 564.2 / 0.037 / 421% | 41, 72, 35 |
| M3 A / B | 6.5e-3 / 0.008 ; **49027 / 0.290 / 45%** | 6.0e-2 / 0.0067 ; **780.4 / 0.036 / 358%** | 41, 73, 72, 35, 35 |
| F2 A / B | 2754 / 0.206 ; 40319 / 0.277 / 50% | 466.6 / 0.077 ; 2768 / 0.073 / 190% | 258, 152 |
| F3 A / B | **5536** / 0.257 ; 158043 / 0.279 / **25%** | **570.4** / 0.076 ; 12498 / 0.088 / **89%** | 258, 152, 142 |

Linked-to-contiguous controls, published → corrected: M1 6517 → 319; M2 108.7 → 10.4; M3 10.2 → 1.64.
The Schur sum rule holds in every case (≤ 8.7e-9).

| R1M conclusion | corrected result | status |
|---|---|---|
| range-only multi-arc accumulates information but never separates K | normal-matrix rank 6/7 everywhere; weakest mode ≥ 0.9999999993 K; `f_perp` ≤ 0.088 | **CONFIRMED** |
| pure multi-arc is worse than contiguous tracking for similar duration | F3-A 570 (6 orbits, 552 obs) < G3 1856 (5 orbits) | **CONFIRMED** |
| continuity-linked beats pure multi-arc | B/A = 3405, 16480, 12967, 5.9, 21.9 | **CONFIRMED** |
| linked beats contiguous with the same observation time | 319×, 10.4×, 1.64× | **CONFIRMED** (margin much smaller for M3) |
| M3 Model B is best and beats contiguous G3 | M3-B 780 < G3 1856; F3-B 12498 is best | **CHANGED** |
| best case falls short of practical identifiability | best σ_K/K is 89% (F3-B) | **CONFIRMED** (a fortiori) |
| `f_perp` saturates at 0.28–0.30 | saturates at ≤ 0.09 | **number superseded** |
| `MULTI_ARC_PRODUCTION_IMPLEMENTATION_JUSTIFIED = NO` | both grounds hold | **CONFIRMED** |

Caveat: the M-window anchors were chosen from visibility on the defective fixture. FXR reran them
as predeclared and did not re-anchor them, because re-anchoring would be a new campaign.

```
R1M_ABSOLUTE_NUMBERS_STATUS          = SUPERSEDED
R1M_QUALITATIVE_CONCLUSION_AFTER_FXR = PARTIAL
```

## 17. R1COV Requalification

The algorithm was not re-proved (s28), because it did not change. It was run on the corrected
designs and compared with R1COV's exact rational oracle:

| case / prior | old floored σ_K | qualified QR σ_K | oracle σ_K | rel. error | old overconfidence |
|---|---:|---:|---:|---:|---:|
| G0 data-only | 2.945e-03 | 1.927648e-01 | 1.927648e-01 | 3.4e-13 | ×65.5 |
| G0 broad (σ_K,prior = 1) | 2.945e-03 | 1.892802e-01 | same | 3.3e-13 | ×64.3 |
| G0 moderate (0.01) | 2.945e-03 | 9.986571e-03 | same | 8.7e-16 | ×3.4 |
| G1 data-only | 1.736e-03 | 7.810870e-02 | same | 6.2e-15 | ×45.0 |
| G3 data-only | 4.570e-04 | 2.321057e-02 | same | 8.7e-15 | ×50.8 |
| M3 Model B | 4.795e-04 | 3.579759e-02 (358%) | 3.579759e-02 | — | ×74.7 |

Further checks on the corrected designs:
- the legacy arm reproduces `r1cov_requalification.csv` and `.json` exactly;
- cond(information)/cond(design)² = 1.0000 for G0–G3;
- the design is full rank 7 while the normal matrix is rank 6, so R1COV's reinterpretation holds;
- the floor still binds on exactly one mode;
- the prior diagnostic still responds (broad and moderate priors differ).

```
R1COV_ALGORITHM_QUALIFICATION_STATUS    = UNCHANGED
R1COV_CANONICAL_NUMERIC_BASELINE_STATUS = REQUALIFIED  (G0 0.193, G1 0.0781, G3 0.0232, M3-B 0.0358 m²/kg)
```

## 18. R1O Requalification

The original ranking is not resurrected (s29). Using R1O-R's corrected-STM analysis with the same
arc definition, landmark set, pair, noise and weights, the legacy arm reproduces
`r1o_r_requalification.json` exactly:

| case | R1O-R corrected (published fixture) | FXR (consistent fixture) | gain vs range (FXR) |
|---|---:|---:|---:|
| range only | 0.252292 (782 obs) | **0.091508** (960) | — |
| landmark surrogate 1 µrad | 0.252964 | 0.252362 | **+0.161** |
| ΔDOR-like surrogate 1 nrad | 0.070144 | 0.361853 | **+0.270** |
| ΔDOR-like surrogate 5 nrad | 0.206980 | 0.320311 | **+0.229** |
| Earth-centre LOS 1 µrad (negative control) | 0.257829 | 0.092011 | +0.0005 |
| R1O-D bridge, reduced surrogate 5 nrad | 0.252403 | 0.091513 | +0.00001 |

Mechanism. The landmark block's own `f_perp` is 0.2529 on both fixtures: the rows do not involve
the Earth and are bitwise identical. Its 1500× information magnitude pulls the combined system
toward that direction. On the published fixture the range baseline happened to sit at the same
0.25, so the landmark looked direction-neutral; that was a coincidence of the wrong geometry.

The ΔDOR-like surrogate changes intrinsically, and flips only when the epoch is repaired:
offset-only −0.005, clamp-only +0.260.

```
R1O_CORRECTED_CONCLUSION_AFTER_FXR = CHANGED — on the consistent geometry both the ΔDOR-like and the
  landmark surrogates raise f_perp well above the corrected range baseline (0.09 -> 0.25-0.36);
  R1O-R's "neither candidate improves on the range baseline" is superseded. The Earth-LOS negative
  control stays null. R1O's original ranking remains invalid (STM defect).
```

## 19. R1O-D Production ΔDOR Requalification

Same production ΔDOR code, Goldstone–Canberra pair, quasar (RA 2.1, Dec −0.3 rad), 90 s grid, 10°
mask, noise levels and B⊥-based weighting. Three arms:

- **published** (legacy fixture, Earth frozen at t = 0 as R1O-D did): reproduces
  `r1od_campaign_summary.json`, the systematic-residual CSV and the noise CSV exactly. One exception:
  the 5 nrad noise row differs by 2.8e-17. The historical script computed that number with two
  different float expressions, and its two published copies already differ by that ulp; the
  headline value is bitwise.
- **epoch repaired, frozen anchor kept.**
- **fully consistent** (Earth at each measurement epoch).

| quantity | published | epoch repaired, frozen Earth | **fully consistent** |
|---|---:|---:|---:|
| dual-visible / converged ΔDOR rows | 143 | 306 | 263 |
| range-only `f_perp` / `theta_K` / σ_K/K | 0.2523 / 14.61° / 9.27% | 0.0915 / 5.25° / 25.40% | 0.0915 / 5.25° / 25.40% |
| ΔDOR-only `f_perp` (5 nrad) | 0.0186 | 0.3837 | 0.3575 |
| range + ΔDOR 5 nrad `f_perp` / `theta_K` / σ_K/K | 0.2121 / 12.25° / 8.67% | 0.3366 / 19.67° / 3.55% | **0.3204 / 18.69° / 3.97%** |
| `f_perp` gain | −0.0401 | +0.2451 | **+0.2289** |
| `I(K\|x)` gain ratio | ×1.14 | ×51.2 | ×40.9 |
| class (R1O-D rule: > 0.05 strong, > 0.02 modest) | MAGNITUDE_ONLY | STRONG | **STRONG_DIRECTION_GAIN** |
| 1 nrad `f_perp` / σ_K/K | 0.0738 / 7.82% | 0.3830 / 0.72% | 0.3622 / 0.83% |
| 100 nrad `f_perp` | 0.2641 | 0.0999 | 0.0992 |
| Case B: K shift per 1 ns session bias | 22.5% of K | 0.7% | 2.5% |
| Case C inflation (bias solved) | ×1.005 | ×1.000 | ×1.000 |

**Attribution** (Earth at the measurement epoch in every arm):

| arm | gain | ΔDOR-only `f_perp` | class |
|---|---:|---:|---|
| legacy | −0.045 | 0.019 | MAGNITUDE_ONLY |
| offset only | −0.005 | 0.020 | MAGNITUDE_ONLY |
| clamp only | +0.260 | 0.385 | STRONG |
| canonical | +0.229 | 0.357 | STRONG |

Effects on the gain:
- epoch repair: +0.234 (no clamp), +0.306 (with clamp);
- clamp repair: −0.031 / +0.040;
- frozen-anchor convention: +0.005 (legacy), +0.016 (canonical).

**The reclassification is caused by the 5.83-day offset alone.** The independent R1O ΔDOR-like
surrogate (FD Jacobians, corrected STM) lands on the same answer: 0.320311 vs production 0.320440 at
5 nrad.

Two further observations:
- The noise trend reverses. Precision now *raises* the direction gain (1 nrad 0.362 > 5 nrad 0.320 >
  100 nrad 0.099), where the published geometry showed the opposite.
- The normal-matrix rank stays 6/7 and the weakest scaled mode stays K-aligned (to 8 digits). The
  gain is a large rotation and magnitude gain, not a rank change.

The R1O-R provenance note said "below the range-only baseline at every tested precision". That was
already inexact on the published fixture: 30 and 100 nrad gave 0.2636 and 0.2641, above 0.2523.

```
R1OD_RANGE_F_PERP_CORRECTED       = 0.091508
R1OD_DDOR_F_PERP_CORRECTED        = 0.320440   (range + production ΔDOR, 5 nrad, fully consistent)
R1OD_RELATIVE_CONCLUSION_AFTER_FXR = CHANGED — MAGNITUDE_ONLY (gain −0.040) -> STRONG_DIRECTION_GAIN
  (gain +0.229; sigma_K/K 25.4% -> 4.0%); cause isolated to the Earth-epoch offset.
```

## 20. R1O-OPT Production Optical Requalification

Same production optical model, camera contract, landmark rule and 10 s headline cadence. The
optical rows are fixture-independent (bitwise-identical SHA on both fixtures). Only the range block
changes: on the consistent geometry the default single station becomes Goldstone rather than
Canberra.

| quantity | published | consistent |
|---|---:|---:|
| range only (10 s): obs, `f_perp`, `theta_K`, σ_K/K | 3934, 0.294990, 17.16°, 3.495% | 3394, **0.086117**, 4.94°, 12.82% |
| optical only (0.25 px, 100 rows) `f_perp` | 0.261735 | **0.261735** |
| combined 0.1 / 0.25 / 1.0 / 2.5 px `f_perp` | 0.3193 / 0.3253 / 0.3246 / 0.3179 | **0.1405** / 0.1027 / 0.0895 / 0.0870 |
| combined σ_K/K, same order | 2.97 / 3.13 / 3.17 / 3.24% | 7.16 / 10.57 / 12.32 / 12.68% |
| unreachable 0.001 px `f_perp` | 0.2618 | 0.2617 |
| operational improvement factor | ×1.175 | ×1.790 |
| class (step: ≥ ×3 and `f_perp` ≥ 0.60; marginal: any measurable gain) | MARGINAL_DIRECTION_GAIN | **MARGINAL_DIRECTION_GAIN** |

The legacy arm reproduces `r1o_opt_campaign.json` exactly.

```
R1O_OPT_RANGE_F_PERP_CORRECTED        = 0.086117
R1O_OPT_OPTICAL_ONLY_F_PERP_CORRECTED = 0.261735
R1O_OPT_COMBINED_F_PERP_CORRECTED     = 0.140466 (best achievable-noise case, 0.1 px; 0.102725 at 0.25 px)
R1O_OPT_RELATIVE_CONCLUSION_AFTER_FXR = CONFIRMED — MARGINAL_DIRECTION_GAIN (improvement ×1.79, far
  from the ×3 / 0.60 step criterion)
```

## 21. R1O-R Requalification

- The STM oracle tests pass (`tests/test_r1o_stm_layout.py`, 5/5).
- The historical defective arithmetic is reproducible bitwise on **both** fixtures (maximum state and
  K differences 0.0).
- nom48 is bitwise identical across fixtures, because `mu_earth = 0` makes the Earth irrelevant to
  the propagation. The column-major STM result therefore cannot depend on the Earth geometry.
- Only R1O-R's context numbers (range 0.2523, corrected surrogate table) are superseded (§18).

`R1O_R_STM_REPAIR_STATUS = VALID`.

## 22. GEO Canonical Cross-Check

FXR arm D is bitwise equal to GEO's cached `FIXTURE_CONSISTENT`. Its metrics match GEO's
`G0_consistent` row exactly: 960 obs, 0.09150755457311016, 25.402136%. GEO's diagnostics B and C
also match bitwise. GEO's canonical fixture is consistent, and its science class is untouched.
`FXR_VS_GEO_CANONICAL_BASELINE_GATE = PASS`.

## 23. G5 K-Magnitude Implementation Audit

Code path: `phase17_geo_grid` → `run_case` → `build_geo_range_arc(k_srp=K)` →
`propagate48(..., k_srp)`, followed by `information_metrics(k_truth=K)`. **Each K was propagated
independently.** A re-run of K = 0.0025 and K = 0.04 gives different nom48 and h_K, and reproduces
the committed values bitwise.

Across K = 0.0025–0.04:
- `f_perp` runs from 0.0915053 to 0.0915166 (1.2e-4 relative);
- absolute σ_K varies by 1.3e-4 relative;
- ‖b_K‖ varies by 7e-6 relative;
- mean altitude differs in the 4th decimal (km).

GEO's "invariant/exactly" wording is correct to about 1e-4, and its §28 table prints the K = 0.01
value on every row. The interpretation (linear SRP, so K magnitude is a relative-precision effect)
stands, with the caveat that end-to-end sensitivity depends weakly on the K-dependent nominal
trajectory. `GEO_G5_IMPLEMENTATION_CLASS = PHYSICAL_K_TRUTH_TRAJECTORY_SWEEP`.
Evidence: `studies/phase17/results/phase17_fxr_g5_audit.json`.

## 24. Historical vs Corrected Cross-Phase Table

See the erratum §3. It is the authoritative table, with columns Phase / historical / fixture /
corrected / absolute changed / relative changed / status / action, and rows R1M, R1COV, R1O,
R1O-D, R1O-OPT, R1O-R, GEO.

## 25. Mathematical Results That Remain Valid

- R1COV's QR / square-root covariance recovery and its exact-oracle agreement: re-verified on
  corrected designs at ≤ 3.4e-13.
- cond(HᵀWH) = cond(A)².
- The floored normal path's scale-dependent overconfidence.
- The Schur identity `I(K|x) = ‖b_K,⊥‖²` and `f_perp = sqrt(I(K|x)/I_KK)`.
- R1M's Model-A Schur sum rule (≤ 8.7e-9 on corrected designs).

## 26. Software Results That Remain Valid

- R1O-R's column-major STM contract (fixture-independent by construction; tests pass).
- R1O-D's production ΔDOR event solver, implicit Jacobian and K chain.
- R1O-OPT's production optical model (fixture-independent rows).
- The production two-way range model, which was never the defect.

## 27. Synthetic Science Results That Remain Valid

- R1M: range-only multi-arc does not separate K; linked beats pure multi-arc; nothing is practically
  identifiable.
- R1O-OPT: marginal optical direction gain.
- The Earth-LOS negative control.
- GEO's range-only geometry classification.
- R1O-D's methodological point, that an unmodelled session bias acts on the point estimate rather
  than the formal covariance. Its magnitude shrinks from 22.5% to 2.5% of K per ns.

## 28. Superseded Numerical Baselines

All of these are `HISTORICAL_NUMERIC_BASELINE, SUPERSEDED`:

- **Canonical range** (quoted by R1O, R1O-D, R1O-R, GEO-published): `f_perp` 0.2523 → **0.0915**;
  `theta_K` 14.61° → 5.25°; `I(K|x)` 1.163e6 → 1.550e5; σ_K/K 9.27% → 25.40%; 782 → 960 obs.
- **R1M:**
  - G0–G3 `f_perp`: 0.009 / 0.225 / 0.243 / 0.269 → 0.051 / 0.067 / 0.073 / 0.059;
  - G0–G3 `I(K|x)`: 1.22 / 2659 / 9355 / 35203 → 26.9 / 164 / 615 / 1856;
  - multi-arc `I(K|x)` and σ_K: see §16;
  - `f_perp` saturation 0.28–0.30 → ≤ 0.09;
  - "6.4× less information from 26% more data" → F3-A 570 vs G3 1856 (3.3×).
- **R1COV:** G0 σ_K 0.904 → 0.193; G1 0.0194 → 0.0781; G3 0.00533 → 0.0232; M3-B 0.00452 (45%) →
  0.0358 (358%); old-path overconfidence ×261 / ×11.8 / ×10.2 → ×65 / ×45 / ×51.
- **R1O-R corrected table:** landmark 0.2530 → 0.2524; ΔDOR-like 1 / 5 nrad 0.0701 / 0.2070 →
  0.3619 / 0.3203; Earth LOS 0.2578 → 0.0920.
- **R1O-D:** combined 5 nrad 0.2121 → 0.3204; noise sweep 0.074 … 0.264 → 0.362 … 0.099;
  Case B 22.5% → 2.5% per ns.
- **R1O-OPT:** range 0.2950 → 0.0861 (σ_K/K 3.50% → 12.82%); best combined 0.3253 → 0.1405;
  improvement ×1.18 → ×1.79.

## 29. Known Range-Jacobian Limitation

`RANGE_JACOBIAN_STATUS = KNOWN_UNRESOLVED_SYSTEMATIC_DIFFERENCE`. GEO measured a
tolerance-independent analytic-vs-FD state-Jacobian difference of about 1e-6 to 3e-5 relative.
FXR did not investigate its root cause (s25).

## 30. Materiality of That Limitation for FXR

GEO's measured end-to-end effects are |Δ`f_perp`| ≤ 3.77e-6 and Δσ_K/K ≤ 1.6e-5 relative. FXR used
the production analytic design for comparability. A classification is accepted only if its margin
exceeds **100×** that bound, i.e. 3.8e-4 in `f_perp` or 1.6e-3 relative in σ_K. The safety factor is
there because the single-station, 10 s and 60 s arcs were not themselves FD-checked.

Every FXR classification clears it by at least 50×:

| classification | margin |
|---|---|
| R1O-D | gain +0.229 against the 0.05 strong threshold: 0.18 |
| R1O-OPT | 0.171 in `f_perp`; 63% in the improvement factor; far below ×3 |
| R1O surrogates | ≥ 0.14 |
| Earth-LOS control | 0.0195 |
| R1M ordering | 2.4× in information |

No result is `JACOBIAN_LIMITED`. The GEO Jacobian FAIL stands as recorded; FXR did not convert it
into a PASS. `RANGE_JACOBIAN_MAX_KNOWN_MATERIALITY_ON_F_PERP = 3.77e-6`.

## 31. Repository Changes

- **Commit a108f55: fixture repair, legacy path, helpers, test.**
  - `studies/phase17/scripts/phase17_r1m_core.py`: `EarthEphemeris`, fixture constants, coverage error,
    superseded warning, `measurement_earth_ephemeris`, `earth_position_mci`, `earth_velocity_mci`,
    the `build_range_arc(earth_fixture=...)` argument, `ArcFixture.earth`.
  - `phase17_r1o_core.build_ddor_arc`, and `earth_los_rows` / `bridge_reduced_rows` in the R1O-R
    requalification script: now take `earth_fixture`.
  - `phase17_r1od_campaign`: explicit `earth_anchor`; shared `ddor_delay_weights`.
  - Five auxiliary scripts: Earth follows the arc's fixture.
  - GEO published arms: pinned to the legacy fixture.
  - `tests/test_fxr_earth_epoch_fixture.py`.
- **Commit 3928224: evidence scripts and artifacts.** `studies/phase17/scripts/phase17_fxr_epoch_audit.py`,
  `phase17_fxr_requalification.py`, `phase17_fxr_g5_audit.py`; `artifacts/phase17_fxr_*.json`;
  the Graphify discovery note.
- **Final commit: documents.** This report, the central erratum, erratum blocks in eight existing
  documents, and `studies/phase17/results/phase17_fxr_manifest.json`.

## 32. Production Code Impact

`git diff 40bc153 -- lunar_od/` is empty. The production range, SPICE loaders, SRP, shadow,
gravity, STM, K sensitivity, estimators, R1COV covariance, ΔDOR and optical code are untouched. The
stale docstring in `src/lunar_od/lunar_landmark_optical.py` is left as it is (s50).
`PRODUCTION_CODE_CHANGED = NO`.

## 33. Historical Report Errata

Additive blocks were inserted directly under the title. No line was deleted, and all historical
numbers are preserved.

- `phase17_r1m_multi_arc_identifiability.md`: absolute numbers superseded, conclusion PARTIAL.
- `phase17_r1cov_square_root_covariance.md`: algorithm valid, numbers superseded.
- `phase17_r1o_additional_observable_feasibility.md`: second erratum (Earth epoch), R1O-R
  conclusion changed.
- `phase17_r1od_production_ddor_qualification.md`: classification CHANGED.
- `phase17_r1o_opt_production_optical_qualification.md`: classification CONFIRMED.
- `phase17_r1o_r_stm_layout_repair.md`: STM repair valid, baseline superseded.
- `phase17_r1o_scientific_erratum.md`: second erratum.
- `phase17_geo_geometry_identifiability.md`: baseline confirmed; G5 wording note.

`HISTORICAL_REPORTS_ANNOTATED = YES (8)`, `HISTORICAL_REPORT_NUMBERS_DELETED = NO`.

## 34. Academic / Methodological Interpretation

Orbit-determination observability studies are sensitive to more than force models and measurement
equations. They are also sensitive to the consistency of the scenario fixture.

Here an ephemeris epoch that was physically inconsistent by 5.83 days rotated the range measurement
projection by 71°–83° without changing the dynamics at all. The truth trajectory, STM and K
sensitivity were bitwise identical. The resulting "identifiability" was wrong:
- the direction metric was wrong by a factor of 2.8;
- the formal σ_K was wrong by a factor of 2.7;
- the relative verdict on a whole measurement type (ΔDOR) was reversed.

Controlled comparisons did not help. Every arm shared the same wrong Earth, so the shared error
did not cancel: it redefined the geometry being compared. The same lesson was learned in R1O-R for
a shared STM defect.

Static code-dependency analysis cannot prove this kind of configuration correct. The defective
datum entered as a runtime callable from another repository, and its epoch was a property of a
number, not of an edge. Graphify saw every call and none of the problem.

This finding has methodological and reproducibility value. It is not a new astrodynamics result.

## 35. Operational Interpretation

A navigation team forms each tracking observable from several time-tagged quantities:

- the spacecraft state at its event epochs;
- the station state (Earth orientation at the transmit/receive epochs);
- the Earth's and the Moon's barycentric positions;
- the light-time solution linking them.

All of them must refer to the same event epochs, in one declared time scale. A mismatch of a few
days, or an interpolation table that silently holds its last value, does not raise an error. It
produces well-formed observables of a *different* tracking geometry. The formal observability and
covariance computed from them look healthy and are wrong.

Practical guards:
- ephemeris objects carry their epoch origin and coverage;
- out-of-coverage requests fail loudly;
- a regression test checks stored geometry against an independent ephemeris query at the
  measurement epoch.

This phase added all three for the analysis fixture.

## 36. Limitations

- One canonical orbit, one epoch and one arc family.
- One ΔDOR quasar and one station pair.
- Formal covariances with no process noise.
- The R1M window anchors were inherited from the defective visibility and not re-anchored.
- The GEO Jacobian characterization was measured on 4-revolution DSN-3 arcs and extrapolated, with
  a 100× safety factor, to the other arc types.
- The orphan R1O-D scripts (`phase17_r1od_fd_convergence.py`, `phase17_r1od_qualification.py`)
  still call `C.get_earth_pos` directly. If run now, they would pair canonical range arcs with the
  legacy Earth. They were left untouched by instruction (s48).

## 37. Simplified-Dynamics Boundary

`FXR_DYNAMICS_SCOPE = INHERITED_CANONICAL_SIMPLIFIED_DYNAMICS`: Moon point mass, lunar J2 and
cannonball SRP with conical penumbra. `mu_earth = mu_sun = 0`, and no high-order lunar gravity.
These were deliberately not activated, so that the fixture repair stays the only change.
Production-fidelity behaviour, including whether the ΔDOR gain survives third-body gravity, is
unknown.

## 38. Synthetic-Truth Boundary

`PROVENANCE = SYNTHETIC_CAMPAIGN_TRUTH`. K = 0.01 m²/kg is a campaign setting, not a spacecraft
property. No result here is mission-qualified observability.

## 39. Regression Tests

| suite | result |
|---|---|
| `test_fxr_earth_epoch_fixture.py` (new) | 10 passed |
| `test_r1o_stm_layout.py` | 5 passed |
| `test_two_way_range.py`, `test_two_way_range_integration.py` | passed |
| `test_srp_k_sensitivity.py`, `test_k_sensitivity_fd_harness.py` | passed |
| `test_k_srp_square_root_covariance.py` | passed |
| `test_delta_dor.py`, `test_lunar_landmark_optical.py` | passed |
| `test_k_srp_bls_solve_for.py`, `test_k_srp_srif_solve_for.py` | passed |
| targeted total | **170 passed, 0 failed** |

## 40. Full Suite

Command: `python -m pytest tests -o addopts="" -q -rfE`. The workspace `pytest.ini` already adds
`-q`, so with a second `-q` pytest prints no count line; `-o addopts=""` restores it. Run time 230 s.

```
TOTAL_TESTS_COLLECTED = 1475   (1329 tests + 146 subtests)
TESTS_PASSED          = 1434   (1288 + 146 subtests)
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

The GEO baseline was 1465 / 1424 / 2 / 10 / 29. The whole difference is the 10 new FXR tests, and
the non-pass set is identical:

| non-pass | class |
|---|---|
| `test_fa06_pytest_session_imports_isolated_repository`: the workspace-root `pytest.ini` injects the sibling worktree onto `pythonpath` | KNOWN_PREEXISTING_ENVIRONMENT |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change`: R2 byte gate on `src/lunar_od/dynamics.py`, awaiting owner re-freeze. `lunar_od/` is untouched by FXR (empty diff) | KNOWN_PREEXISTING_PROVENANCE |
| 10 errors in `test_r2_measurement_fidelity.py`, downstream of that gate | KNOWN_PREEXISTING_PROVENANCE |

```
FXR_INTRODUCED_NONPASSES   = 0
NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES          = 0
```

## 41. What This Phase Actually Established

1. The canonical fixture's Earth was 504000 s early and clamped after 28260 s. This is proven from
   the manifest, the ephemeris file, the campaign module and runtime inspection.
2. The epoch offset dominates the error; the clamp is secondary, with an interaction between them.
3. There is a consistent canonical fixture that fails loudly outside its coverage, has behavioural
   regression protection, and keeps an explicit, bitwise legacy path.
4. The canonical range baseline is `f_perp = 0.0915`, σ_K/K = 25.4%, the same as GEO.
5. Every historical phase was requalified through its own code, with exact legacy reproduction.
6. On the consistent geometry, **production ΔDOR is a strong K-direction and K-information gain
   over range**, not magnitude-only, and the cause is the epoch offset. R1M's range-only negative
   result, R1O-OPT's marginal optical class, R1COV's algorithm and R1O-R's STM contract all survive.
7. GEO's G5 was a physical K-truth sweep, invariant to about 1e-4.

## 42. What This Phase Did NOT Establish

- That ΔDOR makes K_SRP identifiable in operations: FXR used one quasar, one pair, formal
  covariance and simplified dynamics.
- Anything about third-body or high-order-gravity dynamics.
- The root cause of the range-Jacobian discrepancy.
- Re-anchored R1M multi-arc schedules.
- Estimator behaviour: no nonlinear estimation, no Monte Carlo.

## 43. What Remains Unknown

- Whether the ΔDOR direction gain is robust across quasars, station pairs, epochs, orbit geometries
  and session-bias models.
- Whether it survives Earth/Sun third-body gravity and high-order lunar gravity.
- Whether it depends on the near-anti-sunward Earth geometry of this particular epoch.
- The source of the 1e-6–3e-5 range-Jacobian difference.

## 44. Decision Tree From Here

- **Case A** (everything survives): does not apply.
- **Case B** (R1O-D or R1O-OPT relative class changes): **applies.** R1O-D moved from
  MAGNITUDE_ONLY to STRONG_DIRECTION_GAIN; R1O-OPT is unchanged. A focused follow-up on production
  ΔDOR on the consistent fixture comes before any K parameterization work (static / consider /
  stochastic K).
- **Case C** (R1M changes materially): does not apply. R1M is PARTIAL, but its conclusions that
  drive the research narrative (range-only multi-arc does not separate K; no practical
  identifiability) are confirmed.
- **Case D** (FXR and GEO disagree): does not apply. They agree bitwise.

The range-Jacobian qualification phase (Case A's next action) stays pending. It does not block,
because no classification is `JACOBIAN_LIMITED`.

## 45. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 40bc1533fb3056b626b3132bbd71a153456c16da
START_TREE   = 121fe64ffb5b35399bd16c7427d0d3f7f0f42c56
FINAL_HEAD   = (the FXR documentation commit; see git log)
FINAL_TREE   = (that commit's tree)
MAIN_HEAD        = fea476f81dad07b3914e53eab709e10fa6e9d10b
ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b
ACTIVE_WORKTREE  = C:/Users/erayh/Documents/Python/lunar-orbit-determination/python_port_phase17r0

FXR_INPUT_GATE          = PASS
GRAPHIFY_VERSION        = graphify 0.9.65
GRAPHIFY_REFRESHED      = YES (and again after the FXR edits)
GRAPHIFY_REFRESH_HEAD   = 40bc1533fb3056b626b3132bbd71a153456c16da
GRAPHIFY_DISCOVERY_GATE = PASS
FXR_RUNTIME_EPOCH_AUDIT = PASS

DEFECT_FILE      = studies/phase17/scripts/phase17_r1m_core.py
DEFECT_SYMBOL    = build_range_arc (measurement Earth = od_gravity_covariance_campaign.get_earth_pos/_vel)
DEFECT_MECHANISM = untagged Earth callable with origin phase7_et0 − 504000 s passed beside et0 = phase7_et0;
                   numpy.interp holds it after 28260 s

SCENARIO_ET                     = 857806356.99998
HISTORICAL_EARTH_TABLE_START_ET = 857302356.99998
DELTA_ET_SECONDS                = 504000.0
DELTA_ET_DAYS                   = 5.833333
TABLE_DURATION                  = 28260 s (3.9986 orbits)
ARC_DURATION                    = 106011.9 s (15 orbits)
CLAMP_START_ET                  = 857330616.99998 (frozen-Earth ET) at scenario ET 857834616.99998
CLAMP_START_ORBIT_NUMBER        = 3.9986

EARTH_EPOCH_OFFSET_REPRODUCTION_GATE    = PASS
EARTH_TABLE_CLAMP_CHARACTERIZATION_GATE = PASS
HISTORICAL_FIXTURE_REPRODUCTION_GATE    = PASS
EARTH_EPOCH_FIXTURE_REGRESSION_GATE     = PASS
LEGACY_REPRODUCTION_PATH_AVAILABLE      = YES
FXR_VS_GEO_CANONICAL_BASELINE_GATE      = PASS

PHASE17_FXR_GATE = PASS
PRIMARY_CLASS    = HISTORICAL_EARTH_EPOCH_FIXTURE_DEFECT_REPAIRED_AND_PRIOR_BASELINES_REQUALIFIED
```

The remaining fields are in §9–§30 and in `studies/phase17/results/phase17_fxr_manifest.json`.

## 46. Exact Next Action

Owner review of this erratum. After that, per Case B, a **focused production-ΔDOR requalification
on the consistent fixture**, before any K-parameterization study. It should test robustness of the
ΔDOR direction gain across:
- quasar direction and station pair;
- session-bias and calibration models;
- epoch / Earth–Sun geometry;
- the GEO geometry families.

It should also run a ΔDOR end-to-end FD-design materiality check. The range-Jacobian root-cause
phase remains queued. Not started here.
