# F3 WP3.2 S5 — Predeclared hold-out replication of the WP3.1 materiality classification (REQ-DYN-007)

- **Date:** 2026-11-27
- **Predeclaration:** `studies/phase17/configs/wp3_2_s5_replication_predeclaration.json`, committed in `10cf133`
  before any S5 code, test, draw or number. One erratum before any draw (`a287144`, in the file's
  `errata_before_any_result`): the nominal epochs are also kept > 3 days from every catalogue eclipse, which gate
  S5_G6 needs.
- **Order of commits:** predeclaration `10cf133` → failing tests `5aec1e9` → population script and driver `a287144`
  → population `a4fdad4` (geometry only, S5_G6 PASS) → driver fix found by the 1 h smoke run `17ea28a` → survey →
  failing floor-gate test `da0bf3b` → result `364e944`.
- **Code:** population `scripts/wp3_2_s5_population.py`, driver `scripts/wp3_2_s5_replication.py`, tests
  `tests/test_wp3_2_s5_replication.py` (26 tests).
- **Results:** `results/wp3_2_s5_population.json`, `results/wp3_2_s5_replication.json`.
- **Production code changed:** none. No protected file changed, so no re-freeze, path-migration or history-chain
  change applies.
- **Amendment:** one, after the result, with no effect on any gate, class or verdict (section 6;
  `docs/assurance/amendment-log.md`).

## 0. Final status

```
S5 (as predeclared)    = every gate PASS
  S5_G1 unit           = 6 test files pass (212 tests; the 13 skips are the S2/S3/S4 strict xfails)
  S5_G2 null           = every production term with its option None and every study term at scale 0
                         reproduces the baseline bit for bit, 18/18 cells
  S5_G3 floor          = baseline floor 1.0e-5 to 4.0e-5 m; term-run floors <= 1.4e-4 m (LRP); 48-state vs
                         6-state <= 6.6e-5 m (all <= 1e-3 m)
  S5_G4 design         = H3xS1 worst column 1.09e-5 (<= 1e-3; also <= 1e-4)
  S5_G5 eclipse        = H3/H4 Moon centre in umbra; H5/H6 partial (min nu_E 0.994, 0.081); Earth shadow
                         non-zero on every eclipse epoch, exactly zero on H1/H2
  S5_G6 independence   = PASS (epochs 2027-08 .. 2029-12; no WP3.1 epoch, eclipse or altitude)
Replication (hold-out max response; A2 value in brackets)
  TIDE                 = REPLICATED, IMPLEMENT       762    [230]
  LRP                  = REPLICATED, IMPLEMENT       181    [165]
  EARTH_SHADOW         = REPLICATED, IMPLEMENT       290    [200]  (eclipse epochs)
  SRP_LIGHT_TIME       = REPLICATED, IMPLEMENT       0.167  [0.109 ray crossing]
  EARTH_J2             = MARGINAL_NOT_REPLICATED     0.079  [0.108]
  PN                   = REPLICATED, DOCUMENT_ONLY   0.013  [0.029]
  PLANETS              = REPLICATED, DOCUMENT_ONLY   0.040  [0.0062]
  EARTH_J2_ITRF93      = reported only               0.078  (production identity frame: 0.079)
REQ-DYN-007            = passed by the predeclared rule (every gate PASS; the five non-marginal terms
                         REPLICATED; EARTH_J2 MARGINAL_NOT_REPLICATED, which the rule allows because Earth J2
                         is already an opt-in production term)
Not changed            = S2, S3 and S4 stay NOT QUALIFIED as predeclared, with their strict xfails
```

"Response" is WP3.1's: the largest |δθ_i / σ_i| over (x0, K_SRP) in some cell, design and noise level
(OD_MATERIAL at ≥ 0.1).

## 1. Question and design

Does each term's A2 class (WP3.1 report section 4) reproduce on a fresh, independent population, under the D2
profile and with the production opt-in terms? WP3.1 section 9 decided that REQ-DYN-007 closes only on this
confirmatory result.

- **Unchanged from WP3.1:** the materiality rule (0.1), the response, the OD-impact fit (x0 prior
  diag(100 m, 0.1 m/s)², K_SRP free, QR), the two designs (range; range + range rate), the two noise levels
  (PROJECT 5 m / 1e-4 m/s; DSN_REALISTIC 1 m / 3e-5 m/s), 1-day arcs at 60 s, Goldstone, Madrid and Canberra
  above 10°, K_SRP = 0.010 m²/kg with the conical lunar shadow, DOP853 at (1e-12, 1e-13).
- **Changed from WP3.1:**
  - the terms are the production opt-in terms (S1 tide, S2 LRP, S3 Earth shadow, S4 SRP light-time package,
    the existing Earth J2 option). Only PN and PLANETS, which production lacks, are WP3.1's study terms;
  - every propagation uses production's event segmentation (`shadow_events=True`). Each production term cuts its
    own contacts (Earth contacts for S3, light-time contacts for S4), which WP3.1 A2 could not do;
  - the usable-cell rule counts the term run's own floor too: a cell can make a term material only if
    max |δr| ≥ 10 × (baseline floor + term floor).
- **Population (predeclared rule, seed 20261127):** 6 epochs × 3 beta strata = 18 cells.
  - Epochs: two nominal epochs drawn from 2027-04 to 2029-12; two of the three total lunar eclipses of that window;
    two of its three non-total eclipses. Each eclipse arc starts 6–18 h (drawn) before greatest eclipse.
  - Orbits: circular, altitude drawn in 50–200 km; |β| drawn in [0, 30), [30, 60) or [60, 90) degrees, with
    random sign, node angle and argument of latitude. WP3.1 used 100 km and β = 0, 90 and equatorial.
  - No LRO data of any day was read. S5 is synthetic only; the sealed WP5.4 days stay sealed.

| epoch | kind | arc start (UTC) | greatest eclipse (UTC) | Moon centre min ν_E |
|---|---|---|---|---|
| H1 | nominal | 2027-09-08T18:08:08 | – | 1 |
| H2 | nominal | 2028-04-05T15:07:12 | – | 1 |
| H3 | total | 2029-06-25T18:56:23 | 2029-06-26T03:22:46 | 0 |
| H4 | total | 2029-12-20T08:58:22 | 2029-12-20T22:42:46 | 0 |
| H5 | non-total (penumbral) | 2027-08-16T16:11:28 | 2027-08-17T07:14:28 | 0.994 |
| H6 | non-total (partial) | 2028-07-06T07:24:24 | 2028-07-06T18:20:32 | 0.081 |

| cell | β (deg) | altitude (km) | lunar umbra (% of day) | Earth shadow on the spacecraft (% of day) |
|---|---|---|---|---|
| H1xS1 / S2 / S3 | 15.9 / 41.3 / −76.8 | 69.7 / 134.4 / 143.2 | 41.5 / 33.5 / 0 | 0 |
| H2xS1 / S2 / S3 | 19.1 / 57.1 / 73.8 | 66.3 / 57.4 / 78.3 | 42.0 / 35.5 / 0 | 0 |
| H3xS1 / S2 / S3 | 21.9 / 36.7 / −83.4 | 82.1 / 138.6 / 189.5 | 38.9 / 34.6 / 0 | 19.8 / 20.5 / 17.9 |
| H4xS1 / S2 / S3 | −5.7 / −59.1 / 65.4 | 146.6 / 104.1 / 184.7 | 37.8 / 27.2 / 0 | 22.9 / 17.2 / 18.9 |
| H5xS1 / S2 / S3 | 22.4 / −33.2 / 89.2 | 172.5 / 129.8 / 139.1 | 33.4 / 35.7 / 0 | 4.4 / 4.2 / 6.5 |
| H6xS1 / S2 / S3 | −12.2 / −39.4 / 82.6 | 148.9 / 116.4 / 142.7 | 36.6 / 34.6 / 0 | 17.0 / 15.4 / 14.1 |

The eclipse catalogue the rule built (9 full moons within 2° of the anti-Sun point) matches the published list for
the window: penumbral 2027-08-17, partial 2028-01-12 and 2028-07-06, total 2028-12-31, 2029-06-26 and 2029-12-20;
the 2027-07-18 event leaves the Moon centre outside the geometric penumbra and is not an eclipse by the rule.

## 2. Gates

| gate | result | values |
|---|---|---|
| S5_G1_UNIT | PASS | S5 26, WP3.1 terms 18, S1 40, S2 46 (3 xfail), S3 37 (7 xfail), S4 45 (3 xfail); 0 failures |
| S5_G2_NULL | PASS | (a) 5 production terms, (b) PN, PLANETS, EARTH_J2_ITRF93: bit for bit in 18/18 cells |
| S5_G3_FLOOR | PASS | baseline 1.0e-5 to 4.0e-5 m; term floors: LRP 1.4e-4, others ≤ 4.0e-5 m; 48-state ≤ 6.6e-5 m |
| S5_G4_DESIGN | PASS | H3xS1: worst column 1.09e-5 relative (threshold 1e-3; the 1e-4 information level also passes) |
| S5_G5_ECLIPSE | PASS | the 12 checks of the predeclaration all true |
| S5_G6_INDEPENDENCE | PASS | 7 checks; computed with the population, before the survey |

## 3. Results

Response per cell (largest over designs and noise levels), with max |δr| in metres; * = BELOW_FLOOR (max |δr| below
10 × the cell's baseline + term floors):

| cell | TIDE | LRP | EARTH_SHADOW | SRP_LIGHT_TIME | EARTH_J2 | PN | PLANETS |
|---|---|---|---|---|---|---|---|
| H1xS1 | 586 [45] | 69.1 [3.8] | 0* | 0.032 [2.6e-3] | 0.028 [9.7e-4] | 3.7e-3 [0.026] | 3.0e-3* |
| H1xS2 | 177 [16] | 83.1 [3.3] | 0* | 0.041 [2.6e-3] | 0.020 [2.3e-3] | 2.3e-3 [0.025] | 8.4e-5* |
| H1xS3 | 48.1 [26] | 25.5 [0.96] | 0* | 0.069 [3.0e-3] | 3.4e-3 [2.6e-3] | 4.0e-3 [0.025] | 3.8e-5* |
| H2xS1 | 67.5 [81] | 148 [3.9] | 0* | 0.061 [2.1e-3] | 7.3e-3 [1.6e-3] | 7.5e-3 [0.026] | 3.9e-4* |
| H2xS2 | 762 [87] | 35.5 [2.2] | 0* | 0.019 [2.1e-3] | 0.011 [1.3e-3] | 4.2e-3 [0.026] | 6.3e-3* |
| H2xS3 | 608 [54] | 26.0 [1.2] | 0* | **0.167** [2.3e-3] | **0.079** [9.2e-4] | 4.1e-3 [0.025] | 5.3e-3* |
| H3xS1 | 449 [72] | 124 [3.9] | 129 [5.5] | 0.070 [2.4e-3] | 0.021 [2.2e-3] | **0.013** [0.025] | 3.0e-4* |
| H3xS2 | 311 [36] | 86.0 [2.9] | 90.2 [1.9] | 0.044 [1.9e-3] | 0.042 [3.8e-3] | 0.012 [0.025] | 3.0e-3* |
| H3xS3 | 487 [67] | 7.61 [0.52] | 9.80 [1.0] | 0.014 [4.2e-3] | 0.057 [1.8e-3] | 3.2e-3 [0.023] | 1.3e-3* |
| H4xS1 | 11.1 [13] | **181** [3.9] | 178 [5.0] | 0.021 [4.4e-3] | 3.3e-3 [1.0e-3] | 0.011 [0.025] | 2.0e-3 [1.1e-3] |
| H4xS2 | 176 [41] | 51.9 [2.2] | 101 [3.2] | 0.038 [3.1e-3] | 0.030 [1.7e-3] | 1.1e-3 [0.025] | 1.4e-3 [4.2e-4] |
| H4xS3 | 326 [41] | 35.8 [1.6] | 74.9 [2.5] | 0.045 [2.7e-3] | 0.078 [2.6e-3] | 3.4e-3 [0.023] | **0.040** [1.2e-3] |
| H5xS1 | 139 [17] | 125 [3.3] | 41.1 [1.4] | 0.067 [2.2e-3] | 4.7e-3 [9.9e-4] | 6.3e-3 [0.024] | 1.6e-4* |
| H5xS2 | 180 [14] | 104 [3.6] | 21.2 [1.7] | 0.046 [4.1e-3] | 0.012 [1.0e-3] | 0.011 [0.025] | 2.0e-4* |
| H5xS3 | 150 [57] | 0.718 [0.26] | 1.04 [0.06] | 8.9e-3 [3.3e-3] | 0.024 [1.3e-3] | 2.4e-3 [0.025] | 7.5e-4* |
| H6xS1 | 81.4 [62] | 147 [3.5] | **290** [8.2] | 0.077 [3.1e-3] | 8.9e-3 [1.3e-3] | 7.6e-3 [0.024] | 3.7e-3 [3.1e-4] |
| H6xS2 | 311 [29] | 98.3 [3.3] | 91.3 [6.1] | 0.029 [2.6e-3] | 0.022* | 0.010 [0.025] | 8.3e-3* |
| H6xS3 | 698 [59] | 9.94 [0.60] | 3.94 [1.8] | 0.015 [2.6e-3] | 0.036 [2.2e-3] | 4.4e-3 [0.025] | 0.030* |

Largest response over the 18 cells, by design and noise level:

| term | range, PROJECT | range, DSN | range + RR, PROJECT | range + RR, DSN | A2 (range + RR, DSN) |
|---|---|---|---|---|---|
| TIDE | 5.89 | 29.5 | 229 | 762 | 230 |
| LRP | 1.40 | 7.00 | 54.2 | 181 | 165 |
| EARTH_SHADOW (H3–H6) | 2.26 | 11.4 | 86.8 | 290 | 200 |
| SRP_LIGHT_TIME | 7.8e-4 | 3.9e-3 | 0.050 | 0.167 | 0.109 (ray crossing) |
| EARTH_J2 | 5.1e-4 | 2.6e-3 | 0.024 | 0.079 | 0.108 |
| PN | 9.2e-5 | 4.5e-4 | 4.0e-3 | 0.013 | 0.029 |
| PLANETS | 2.8e-4 | 1.4e-3 | 0.012 | 0.040 | 6.2e-3 |
| EARTH_J2_ITRF93 (reported) | 5.1e-4 | 2.6e-3 | 0.024 | 0.078 | – |

Every class is set by the range + range-rate design at the DSN-realistic level, as in WP3.1. With range alone at
the project noise level only TIDE, LRP and EARTH_SHADOW reach 0.1.

## 4. What the numbers mean

- **TIDE, LRP, EARTH_SHADOW.** Material by two to four orders of magnitude in every cell of their domain, as in A2.
  The spread follows geometry:
  - LRP is smallest in the high-β cells, where the spacecraft never enters the lunar shadow (0.72 in H5xS3);
  - EARTH_SHADOW follows the time spent in the Earth's shadow. Even the shallow penumbral eclipse H5
    (Moon centre ν_E ≥ 0.994) gives 21–41, because the spacecraft at 1.9 Mm from the Moon's centre passes deeper.
- **SRP_LIGHT_TIME** replicates at 0.167 in H2xS3 (β 73.8°, no lunar shadow). This is the clean case of WP3.1
  section 5: the velocity terms of the Poynting–Robertson force alone, as in A2's O2×E1 (0.109). In the shadowed
  cells the aberrated force and the displaced shadow largely cancel (0.01–0.08), as WP3.1 section 6.2 found with the
  ray-crossing occulter.
- **EARTH_J2** reaches 0.079 (H2xS3), against A2's 0.108. The 18 cells do not include A2's O1×E1 geometry (β = 0,
  Sun in a polar plane). The study term with the ITRF93 pole gives 0.078 in the same cell, so production's identity
  frame is not the cause (DYN-19 remains immaterial at this level). The predeclared rule classes this as
  MARGINAL_NOT_REPLICATED: of the order of the threshold, but below it everywhere on the hold-out.
- **PN** ≤ 0.013 (A2 ≤ 0.029). Its along-track signature is 2.3–2.6 cm in every cell, as WP3.1's closed form
  predicts; the fit absorbs it into x0.
- **PLANETS** reaches 0.040 in H4xS3 (signal 1.2 mm), 6.5× A2's maximum but 2.5× below the threshold. In 14 of 18
  cells the signal is below 10× the floors. It stays DOCUMENT_ONLY, now with a measured bound of 0.04 rather than
  0.006.

## 5. REQ-DYN-007

The predeclared closure rule asks for every gate PASS, the five non-marginal terms REPLICATED, and neither marginal
term CONTRADICTED. All three hold, so the requirement is **passed** as predeclared.

- The force-model terms that are material on this orbit class are in production as opt-in terms: tide (S1), LRP (S2),
  Earth shadow (S3), the SRP light-time package (S4) and Earth J2 (existing option). The terms left out (PN, Venus
  and Jupiter) stay below 0.1 on the hold-out.
- This closes the completeness question only. S2, S3 and S4 remain NOT QUALIFIED as predeclared: their failed gates
  are numerical floors, an oracle precondition, control ratios and FD steps. S5 used only their value path, whose
  parity with the WP3.1 terms passed; their merge status and fixes are separate owner decisions.
- EARTH_J2 is marginal: A2 0.108 in one cell, hold-out ≤ 0.079 in 18 cells. Whether Earth J2 joins the recommended
  configuration is unchanged (WP3.1 decision 2: on in the recommended real-data configuration, default off).

## 6. Deviations and amendment

- **Erratum before any draw** (`a287144`): the nominal epochs are kept > 3 days from every catalogue eclipse. As
  first written, the rule drew them before the eclipse epochs and could not guarantee S5_G6.
- **Driver fix before the survey** (`17ea28a`): `wp3_1_materiality_survey.closure` no longer matches
  `dynamics._build_state_rhs` on main (two positional arguments added since WP3.1). The S5 driver builds the closure
  as `propagate_state` does; S5_G2(b) confirms it reproduces the production baseline bit for bit. Found by the
  1 h smoke run (software check, no response read).
- **After the result (amendment log 2026-11-27):** the first combine evaluated S5_G3 per cell from the baseline
  floor and the 48-state agreement only; the predeclared gate also includes each term run's own floor. The code
  now evaluates the gate as predeclared (failing test `da0bf3b` first). The term floors were already in the cell
  files (≤ 1.4e-4 m), so the recomputed result has the same gates, classes, verdicts and cell data. No threshold,
  rule or population changed.

## 7. Limits

- Synthetic arcs only: circular 50–200 km orbits, 1-day arcs, three DSN stations, (x0, K_SRP) solve-for. The
  conclusion covers this class. Longer arcs make the drag-like terms (SRP light time) and the secular terms grow;
  an empirical-acceleration solve-for absorbs more.
- One term at a time, as WP3.1; combined-term runs are out of scope.
- The β strata do not include exactly β = 0 with a polar orbit; EARTH_J2's A2 maximum came from that geometry.
- Linear OD impact on a noise-free signature (WP3.1's method); no Monte Carlo of the estimator.
- Inherited model limits of S2–S4 (LRP overestimated during umbral eclipse phases; no atmosphere in the Earth's
  shadow; the doubled Sun SSB velocity in the S4 aberration, ≤ 5.2e-8 rad).

## 8. Regression

- Full suite at `2fc5fd8` (later commits change only this section), run with
  `QT_QPA_PLATFORM=offscreen python -m pytest -p no:cacheprovider tests --junitxml=...`: JUnit header
  `tests="3447" failures="0" errors="0" skipped="71"`, pytest exit 0, 3013 s. The 3299 `testcase` elements split into
  3228 passed, 49 skipped and 22 strict xfails. As at S4, the header counts 148 more tests than the elements; the
  reason is BİLİNMİYOR (not investigated).
- The 22 xfails are the ones on main before S5: S3 seven, WP3.3 K_G1 six, S2 L_G6 three, S4 three, EST-01 two, the
  external-oracle `cd_ssb` case one. No strict xfail passed.
- `tools/check_repository.py` passes (597 migrated paths, 409 preserved byte identities, 563 local links).
- `test_no_study_script_or_config_names_a_holdout_day` passes.
