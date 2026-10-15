# PHASE 17-R1O-D2 — CORRECTED-GEOMETRY PRODUCTION ΔDOR ROBUSTNESS QUALIFICATION

> **The question, answered first.** Is the strong production-ΔDOR K_SRP direction gain found after
> FXR a robust consequence of complementary angular tracking geometry, or a special result of one
> quasar, one baseline, one epoch and one simplified orbit?
>
> **It is geometry-dependent. It is not general.**
>
> - **Quasar:** not the cause. ΔDOR design rows do not depend on the quasar; this was checked
>   bitwise by FD.
> - **Occultation:** part of FXR's canonical gain came from rows formed while the spacecraft was
>   behind the Moon. The honest canonical value is `f_perp` 0.0915 → **0.1724** (gain **+0.081**),
>   not 0.3204.
> - **Where the gain lives:** on the canonical epoch it survives every representative orbit
>   geometry, a high-fidelity dynamics spot check and operational session sampling, but **only on
>   the Goldstone–Canberra baseline**. It is **absent at the other three lunar-phase epochs** and
>   **absent on Goldstone–Madrid**.
> - **Calibration:** unconstrained per-session calibration biases **remove it everywhere**. A
>   DSN-level bias prior keeps about half.
> - **Absolute identifiability:** K stays the weakest design direction in every case.
>
> Predeclared outcome: `DDOR_K_INFORMATION_GAIN_IS_STRONGLY_GEOMETRY_DEPENDENT`.

All numbers are `SYNTHETIC_CAMPAIGN_TRUTH` (K_truth = 0.01 m²/kg is a campaign value). Evidence:
`artifacts/phase17_r1od2_*.json|csv|png`. Grid: `studies/phase17/configs/phase17_r1od2_predeclared_grid.json`
(SHA-256, LF-normalized, `2a99c50e…c07c11`), committed in `1ca5356` **before** any headline metric.

---

## 1. Executive Summary

The D2 campaign re-used FXR's canonical setup and varied, one factor at a time and then jointly:
- quasar geometry (40 synthetic directions per session);
- station pair (Goldstone–Canberra, Goldstone–Madrid, Madrid–Canberra);
- epoch (4 lunar phases);
- orbit geometry (5 GEO families);
- ΔDOR precision and semantics;
- sampling realism (dense rows vs operational sessions);
- calibration nuisance treatment;
- dynamics fidelity.

**Six findings.**

1. **FXR's canonical number contained impossible observations.** The R1O-D/FXR ΔDOR rule tested
   only elevation. 93 of its 263 canonical rows had the spacecraft behind the Moon for at least one
   station. With the production visibility test (elevation plus lunar occultation, the same one the
   range arc uses), the canonical Goldstone–Canberra result is `f_perp` **0.1724** (gain +0.081,
   σ_K/K 7.46%), not 0.3204. That is still a STRONG relative gain under the R1O-D threshold.
2. **The gain is geometry-dependent.** Primary population: 9 geometries × {GC, GM}, 5 nrad.
   - STRONG 6/18: all six are Goldstone–Canberra at the canonical epoch.
   - MODEST 2/18.
   - MAGNITUDE_ONLY 8/18, including one pair with no dual visibility.
   - DIRECTION_LOSS 2/18.

   Goldstone–Canberra at the canonical epoch is STRONG in all six orbit variants. No case at the
   other three lunar-phase epochs is STRONG. Goldstone–Madrid is never STRONG. Madrid–Canberra is
   never dual-visible.
3. **Quasar geometry does not change the formal information.** The production ΔDOR rows are
   quasar-independent: `D_Q` does not depend on the spacecraft, and the FD derivative of the full
   `D_S − D_Q` is bitwise identical for three quasars.
   - The quasar changes only systematic susceptibility.
   - The per-session quasar-coordinate error (0.75 nrad) dominates, at 4.7–5.0% of K, independent
     of separation.
   - Station-location and Earth-orientation terms grow linearly with separation but stay below
     0.6% of K.
   - Position angle changes the total by less than 0.4% of K.
4. **The gain is calibration-fragile.** One free delay bias per session (B3) removes the gain in
   18/18 dense cases, and all but 1 operational case (per the predeclared survival rule). With a
   DSN-level prior of 0.033 ns on those biases (B4), roughly half the gain survives. Canonical
   Goldstone–Canberra keeps +0.046 dense and +0.036 operational.
5. **The production ΔDOR design is FD-qualified.**
   - All six state columns match independent re-propagation to ≤ 2.4e-5 on every row of three
     cases.
   - The K column is resolution-limited by the predeclared oracle (rtol 1e-12, halving floor
     2–4e-2). With a rtol 1e-14 oracle it matches to ≤ 2.0e-4.
   - The FD design moves `f_perp` by at most 3.7e-4.
6. **Higher-fidelity dynamics preserve every class.** GL1800F 50×50 plus Earth and Sun third-body
   gravity moves the truth by 15–46 km over 15 orbits, yet every tested class is unchanged.

**Absolute identifiability is unchanged in kind.** The weakest direction of the whitened, scaled
design is K-aligned (|v_K| = 1.00000000) in every case. 16 of the 17 cases with ΔDOR data stay
`WEAK_K_SEPARATION` (f_perp < 0.35). Only O5 Goldstone–Canberra reaches 0.605.

## 2. What FXR Reversed

On the published (defective) Earth fixture, production ΔDOR at 5 nrad lowered `f_perp` from 0.2523
to 0.2121 (MAGNITUDE_ONLY). On the epoch-consistent fixture it raised it from 0.0915 to 0.3204
(STRONG). FXR isolated the cause to the 5.83-day Earth-epoch offset. It did not test robustness,
and, as D2 found, it inherited R1O-D's elevation-only ΔDOR row rule.

## 3. Why ΔDOR Can Complement Range

Two-way range sees the trajectory along the Earth–spacecraft line of sight. At the canonical epoch
that line lies 12.5° from the SRP (anti-Sun) direction: the Sun–Moon–Earth angle is 167.5°. The
K-driven displacement therefore projects onto range in a way an initial-state change can largely
imitate (f_perp 0.0915).

ΔDOR measures a differential delay between two stations. That is the spacecraft's plane-of-sky
position projected on the baseline (CCSDS eq. 11–12). It can see displacement components that range
cannot. Whether those components carry K-specific information depends on the baseline's projected
orientation relative to the K signature, and on how the state basis maps onto the same projection.

## 4. Strong Relative Gain vs Absolute Identifiability

Two separate labels are kept throughout:

- **Relative gain** `Δf = f_perp(range+ΔDOR) − f_perp(range)` on the same geometry. It uses the
  R1O-D thresholds (STRONG > 0.05, MODEST > 0.02) plus a symmetric DIRECTION_LOSS below −0.02.
- **Absolute separation** uses the GEO gates on the combined `f_perp`: STRONG ≥ 0.50,
  MODERATE ≥ 0.35, else WEAK.

Canonical Goldstone–Canberra is a **STRONG relative gain** (+0.081) that leaves K **WEAK** in
absolute terms (0.172, θ_K = 9.9°). A large relative gain is not identifiability.

## 5. Literature Contract

See `studies/phase17/reports/phase17_r1od2_literature_contract.md`. It inspects:
- DSN 810-005 Module 210 Rev. E (2023/24);
- CCSDS 500.1-G-2 (2019);
- Curkendall & Border (2013).

Every statement is tagged. `D2_LITERATURE_CONTRACT_GATE = PASS`.

## 6. DSN / CCSDS ΔDOR Geometry

`LITERATURE_RESULT`:
- ΔDOR needs two stations at different complexes (DSN-210 pp. 7, 9).
- DSN measurements use Goldstone–Madrid (east–west, RA-sensitive near the ecliptic) or
  Goldstone–Canberra (canted, splits RA/Dec) (DSN-210 p. 13).
- A single measurement constrains the spacecraft only in the baseline–spacecraft plane.
- Two baselines are needed for both angular components (CCSDS §4.1).

`REPOSITORY_RESULT`:
- Madrid–Canberra is never dual-visible in any of the 9 geometries (0 candidate epochs).
- Goldstone–Madrid has no dual visibility at E3.

## 7. Quasar Calibration Context

`LITERATURE_RESULT`:
- Quasar–spacecraft–quasar scans common-mode out clock, instrument and platform errors. Nearer
  quasars cancel more (CB13 pp. 15–21).
- Baseline and media errors scale with separation (DSN-210 p. 18; CCSDS eq. 25–26).
- The quasar-coordinate error is `ε_θ B_p/c` (eq. 32).
- The random budget is 0.048 ns and the systematic RSS is 0.033 ns (DSN-210 Tables 8–9).
- When systematics are modelled explicitly, only the random part weights the data (DSN-210 p. 20).

## 8. Entering Repository State

| field | value |
|---|---|
| `START_BRANCH` | `feature/phase17-r-k-srp-estimation` |
| `START_HEAD` / `START_TREE` | `a4304fc35a27b212fcab1eaf6f936e3d45fe56c2` / `b6f5d9ef114ede86f029e2ed081d48447a391639` |
| `MAIN_HEAD` = `ORIGIN_MAIN_HEAD` | `fea476f81dad07b3914e53eab709e10fa6e9d10b` (unchanged) |
| `ACTIVE_WORKTREE` | `C:/Users/erayh/Documents/Python/lunar-orbit-determination/python_port_phase17r0` |
| `WORKTREE_STATUS` | clean apart from the two untracked R1O-D orphan scripts, which were not run, modified or committed |

`D2_INPUT_GATE = PASS`.

## 9. Graphify / Manual Dependency Audit

The graph was current for the entering HEAD. The only change since FXR's refresh was inside a
function body, so D2 did not refresh before discovery; it did refresh after adding its modules.
Queries run:
- `affected` on `delta_dor_spacecraft_sensitivity_full` and `solve_common_transmit_event`: callers
  are R1O-D scripts and `tests/test_delta_dor.py`.
- `explain` on `quasar_differential_delay_s` and `QuasarDirection`.
- `path build_ddor_rows ↔ analyze_visibility_gap_with_transforms`.

Verified in source:
- The ΔDOR sensitivity has **no** quasar input.
- `phase17_r1od_campaign.build_ddor_rows` accepts a `quasar` argument it never uses.
- The only link from `build_ddor_rows` to the production visibility function is a 3-hop path
  through module imports. Source confirms the R1O-D row rule never calls it, which is the
  occultation defect.

`GRAPHIFY_DISCOVERY_GATE = PASS`. NO GRAPHIFY EDGE ≠ NO DEPENDENCY: every claim above was checked in
source.

## 10. Corrected Earth-Fixture Gate

- Every D2 arc uses `measurement_earth_ephemeris(EARTH_FIXTURE_CANONICAL, et0, t_max)`.
- ΔDOR stations sit on the same ephemeris (position and velocity) at each measurement epoch.
- Every worker calls `install_legacy_earth_guard()`, which makes the campaign table raise
  `LegacyEarthLeakError`. No worker raised.
- Per-geometry runtime audit (`earth_audit` in `phase17_r1od2_results.json`), all 9 PASS:
  - Earth origin = rotation origin = scenario epoch;
  - Earth position and velocity equal SPICE at the measurement epoch (maximum difference 0 m, 0 m/s);
  - `clamp_after = ∞`;
  - the Sun is on the same scenario epoch;
  - the quasar frame contract is explicit (fixed J2000 vector).

`D2_EARTH_FIXTURE_GATE = PASS`. `D2_EPOCH_CONSISTENCY_GATE = PASS`.

## 11. Canonical FXR Reproduction

D0 used FXR's own row rule (elevation-only, Earth at the measurement epoch). Range, ΔDOR rows,
weights and the surrogate are **bitwise** equal to the FXR / R1O-D / R1O builders
(`tests/test_r1od2_ddor_contracts.py`).

| quantity | FXR | D2 D0 |
|---|---:|---:|
| range `f_perp` | 0.091508 | 0.0915076 |
| range + ΔDOR 5 nrad `f_perp` | 0.320440 | 0.3204398 |
| combined σ_K/K | 3.97% | 3.974% |
| 1 nrad / 100 nrad `f_perp` | 0.3622 / 0.0992 | 0.36220 / 0.09921 |
| independent surrogate | 0.320311 | 0.3203109 |

`D2_CANONICAL_REPRODUCTION_GATE = PASS`.

**The D2 reference row rule changes the canonical answer.** Requiring both stations to see the
spacecraft (elevation + lunar occultation) leaves 170 of the 263 rows, in 5 sessions.

| canonical Goldstone–Canberra | f_perp | gain | σ_K/K |
|---|---:|---:|---:|
| FXR rows | 0.3204 | +0.229 | 3.97% |
| visible rows | **0.1724** | **+0.081** | **7.46%** |

Removing the 93 occulted rows lowers the canonical combined `f_perp` by 0.148. The class is unchanged (STRONG). The row-rule choice was fixed in the grid from a row count
alone, before any K metric was computed on it.

## 12. Independent Surrogate Cross-Check

The R1O surrogate uses a geometric differential range at the grid epoch, FD position Jacobian,
column-major STM and its own mean-LOS weight. It was run on the same epochs as production for all
17 primary cases with data. The maximum |Δf_perp| between surrogate and production is **2.9e-4**.
No divergence needed explanation. (FXR's "1223 surrogate rows" was the *combined* count: 960 range + 263 ΔDOR.)

## 13. Predeclared Robustness Grid

The grid is `phase17_r1od2_predeclared_grid.json`, written by `phase17_r1od2_grid.py` and committed
first. It freezes:

| item | frozen content |
|---|---|
| geometries (9) | CANON; E1/E2/E3 = the same inertial state vector at +¼, ½, ¾ synodic month; O1–O5 = GEO cases `G1_b00`, `G1_b80`, `G2_alt500_b20_fixD`, `G3_i90.0_bcan`, `G3_i26.7_bcan` |
| pairs | GC, GM, MC |
| realizations | dense angular 1–100 nrad; dense delay 0.024–1 ns; OPS_EVERY_WINDOW; OPS_TWO_SESSIONS (3 normal points per session, 0.048 ns) |
| quasar grid | 5 separations × 8 position angles |
| bias cases | B0–B4 |
| FD cases | FD1–FD3, with steps and gates |
| fidelity cases | the three D7 cases, including the rule-selected weakest |
| rules | all thresholds and the outcome rules (D → C → B → A) |

The campaign scripts refuse to run on a hash mismatch.

`D2_GRID_PREDECLARED = YES`.

## 14. Quasar Separation Results

**Formal information:** invariant. The rows have no quasar input, and FD of the full observable is
bitwise identical for three quasars (`phase17_r1od2_fd.json: quasar_invariance`).

**Systematic K bias:** 1-σ RSS over components, canonical Goldstone–Canberra, dense 5 nrad.

| separation | total | station location | quasar coordinate |
|---|---:|---:|---:|
| 0.02 rad | 4.91% | 0.04% | 4.90% |
| 0.1 rad (DSN nominal) | 4.91% | 0.19% | 4.86% |
| 0.3 rad | 4.94% | 0.57% | 4.70% |

Formal σ_K/K for comparison is 7.46%, so the systematic bias is 0.66 σ. Goldstone–Madrid: total
1.2–1.9%, with the station term rising 0.07 → 0.92%.

`QUASAR_SEPARATION_EFFECT_CLASS = FORMAL_DIRECTION_INVARIANT__SYSTEMATIC_BIAS_WEAKLY_SEPARATION_DEPENDENT`.

## 15. Quasar Position-Angle Results

At fixed separation, the 8 position angles change the total K bias by at most 0.36% of K (0.3 rad)
or 0.11% (0.1 rad) on Goldstone–Canberra. The formal information is invariant.

`QUASAR_POSITION_ANGLE_EFFECT_CLASS = FORMAL_DIRECTION_INVARIANT__SYSTEMATIC_BIAS_NEGLIGIBLY_PA_DEPENDENT`.

The brief's hypothesis, that two quasars at the same separation but different position angle give
very different *geometric sensitivity*, is **false for this observable**. Only the
calibration-error projection depends on it, and weakly.

## 16. Real-Catalog Check

No authoritative radio-source catalog is available locally. All quasars are
`GEOMETRY_ONLY_SYNTHETIC_QUASAR`, so `OPERATIONAL_SOURCE_AVAILABILITY = UNRESOLVED`. Because the
formal rows are quasar-independent, this affects only the systematic-bias numbers.

## 17. Station-Pair Results

Dense 5 nrad; range recomputed per geometry.

| geometry | range f | GC rows / sessions | GC Δf | GM rows / sessions | GM Δf |
|---|---:|---:|---:|---:|---:|
| CANON | 0.0915 | 170 / 5 | **+0.081** | 70 / 3 | −0.013 |
| E1 (SME 75°) | 0.2476 | 181 / 5 | +0.024 | 130 / 3 | +0.036 |
| E2 (SME 23°) | 0.1347 | 95 / 2 | +0.010 | 22 / 1 | +0.012 |
| E3 (SME 107°) | 0.2369 | 96 / 2 | −0.005 | 0 | — |
| O1 low-beta | 0.0784 | 172 / 5 | **+0.075** | 69 / 3 | −0.000 |
| O2 full-Sun | 0.1142 | 232 / 3 | **+0.157** | 111 / 2 | −0.043 |
| O3 500 km | 0.1165 | 177 / 5 | **+0.128** | 99 / 2 | +0.009 |
| O4 polar | 0.1057 | 150 / 4 | **+0.146** | 93 / 2 | +0.007 |
| O5 GEO-max | 0.2814 | 144 / 3 | **+0.324** | 92 / 2 | −0.081 |

- Madrid–Canberra: 0 dual-visible epochs in every geometry.
- Goldstone–Canberra: baseline length 10 589 km; projected plane-of-sky baseline ≈ 10 300–10 410 km.
- Goldstone–Madrid: baseline length 8394 km; projected ≈ 7 900–8 380 km.

`BEST_STATION_PAIR = Goldstone–Canberra` (median Δf +0.081). `WORST_STATION_PAIR = Goldstone–Madrid`
(median +0.003, two DIRECTION_LOSS); Madrid–Canberra is unusable.
`STATION_PAIR_EFFECT_CLASS = DOMINANT`.

## 18. Projected-Baseline Interpretation

Baseline length is not the explanation: the projected Goldstone–Canberra baseline is only about 1.3×
Goldstone–Madrid's. The discriminating quantity is **the ΔDOR block's own K separability**, i.e.
`f_perp` of ΔDOR alone:

| pair | ΔDOR-only `f_perp` |
|---|---|
| Goldstone–Madrid | 0.000–0.047 in every geometry |
| Goldstone–Canberra, canonical-epoch geometries | 0.086–0.487 |
| Goldstone–Canberra, other epochs | 0.010–0.14 |

A ΔDOR block whose K signature a state change can reproduce cannot add direction. With high
weight it can only pull the combined direction toward its own low `f_perp`. That is the
DIRECTION_LOSS mechanism on O2 and O5 Goldstone–Madrid.

`REPOSITORY_HYPOTHESIS`, consistent with DSN-210 p. 13: the canted Goldstone–Canberra baseline
carries the declination-like plane-of-sky component, and at the canonical epoch that component of
the K displacement is not reproducible by the state.

A simpler geometric descriptor does **not** predict the gain: the median angle between the projected
baseline and the plane-of-sky dr/dK (Figure `delta_vs_baseline_angle`). For example, O1 GC at 66° is
STRONG while CANON GM at 44° is not. The mechanism lives in the state-orthogonal part of the
signature, not in the raw displacement.

## 19. Epoch Results

Same inertial orbit; Goldstone–Canberra, dense 5 nrad.

| epoch | Sun–Moon–Earth angle | range f | Δf |
|---|---:|---:|---:|
| CANON | 167.5° | 0.0915 | +0.081 |
| E1 | 75.1° | 0.248 | +0.024 |
| E2 | 22.8° | 0.135 | +0.010 |
| E3 | 106.5° | 0.237 | −0.005 |

Goldstone–Madrid at the same epochs: −0.013, +0.036, +0.012, and no rows at E3.

`EPOCH_EFFECT_CLASS = DOMINANT`: the strong gain is confined to the canonical epoch.

## 20. Earth/Sun/LOS Interpretation

At the canonical epoch the range line of sight is 12.5° from the SRP direction, and range alone is
weak (0.0915). At E1 and E3 the line of sight is 105° / 74° from SRP. Range alone is already
stronger there (0.24–0.25), and ΔDOR has little direction left to add.

E2 also has a line of sight near the SRP axis (157°), with range 0.135. Yet ΔDOR adds only +0.010:
it has only 2 Goldstone–Canberra sessions, and a ΔDOR-only `f_perp` of 0.010.

Line-of-sight/SRP alignment is therefore **necessary context but not a sufficient predictor**. The
epoch matters through the complete Earth–Moon–Sun–baseline geometry, which only the ΔDOR-only
separability summarizes (§18).

## 21. Representative Orbit-Geometry Results

At the canonical epoch, every orbit variant keeps a STRONG Goldstone–Canberra gain:

| case | Δf |
|---|---:|
| O1 low-beta | +0.075 |
| O2 full-Sun | +0.157 |
| O3 500 km | +0.128 |
| O4 polar | +0.146 |
| O5 GEO-max | +0.324 (combined 0.605, the only STRONG_K_SEPARATION case) |

- O2 (full Sun) keeps the direction gain but, as GEO found, has almost no K magnitude: σ_K/K is
  201% for range and 35% combined.
- Goldstone–Madrid is MAGNITUDE_ONLY or DIRECTION_LOSS in all five.

`ORBIT_GEOMETRY_EFFECT_CLASS = CLASS_PRESERVED_ON_GC_AT_CANONICAL_EPOCH__MAGNITUDE_VARIES_4X`.

## 22. ΔDOR Precision Results

The gain is **non-monotonic in precision**. The combined system moves from range's direction toward
ΔDOR's own direction as ΔDOR weight grows, and can exceed both constituents in between.

| case | 1 nrad | 2 nrad | 5 nrad | 10 nrad | 30 nrad | 100 nrad |
|---|---:|---:|---:|---:|---:|---:|
| CANON Goldstone–Canberra | +0.047 | +0.057 | **+0.081** | +0.072 | +0.026 | +0.006 |
| E1 Goldstone–Canberra | **−0.088** | −0.050 | +0.024 | +0.053 | +0.055 | +0.037 |
| O5 Goldstone–Madrid | −0.209 | … | … | … | … | +0.013 |

FXR's "higher precision → larger gain" was a property of one geometry and one row set.
`NOISE_EFFECT_CLASS = NON_MONOTONIC_GEOMETRY_DEPENDENT`.

Against the DSN random budget (0.048 ns delay, dense rows):
- class counts: 6 STRONG, 0 MODEST, 4 MAGNITUDE_ONLY, 7 DIRECTION_LOSS;
- canonical Goldstone–Canberra is +0.051, only just STRONG.

Operational sampling (OPS_EVERY_WINDOW, 3 normal points per window, 0.048 ns):
- class counts: 7 STRONG, 1 MODEST, 8 MAGNITUDE_ONLY, 2 DIRECTION_LOSS;
- canonical Goldstone–Canberra is +0.067 from 15 normal points.

OPS_TWO_SESSIONS (6 normal points) gives canonical Goldstone–Canberra +0.059.

## 23. Delay / Angular Noise Semantics

The R1O-D/FXR parameter is an **angle**. Each row converts it as `σ_τ = max(B⊥,1)·σ_θ/c`, where
`B⊥` is the baseline component perpendicular to the station-A→spacecraft line of sight
(`ddor_weights`, bitwise equal to R1O-D). So 5 nrad corresponds to:
- **0.164–0.177 ns** of delay on Goldstone–Canberra;
- **0.114–0.140 ns** on Goldstone–Madrid;

in both cases about 3× the DSN random budget.

DSN-210 characterizes ΔDOR error in delay units. D2 therefore reports both semantics. The
cross-pair and cross-epoch comparisons are made under both, and the operational and bias cases use
constant delay σ (0.048 ns random only). Nothing is double-counted: B3/B4 solve the systematic and
weight with the random part only, as DSN-210 p. 20 prescribes.

`DDOR_NOISE_SEMANTICS_GATE = PASS`.

## 24. Session-Bias Results

B1 injects a common unestimated bias on all rows; B2 injects independent per-session biases
(σ 0.033 ns, RSS K shift).

| case | B1 per 0.033 ns | B1 per 1 ns (stress) | B2 |
|---|---:|---:|---:|
| CANON Goldstone–Canberra | 0.6% of K | 18% | 6.3% |
| CANON Goldstone–Madrid | 1.5% | 45% | 2.0% |
| O2 Goldstone–Canberra | 7.6% | 231% | 52% |
| O2 Goldstone–Madrid | 68% | 2068% | 79% |

`BIAS_EFFECT_CLASS = CALIBRATION_FRAGILE` (details in §25).

## 25. Nuisance-Bias Solve

One delay bias per session: B3 unconstrained, B4 with a 0.033 ns prior. The data are weighted with
the random component only (DSN-210 p. 20).

| case | B0 Δf | B3 Δf | B4 Δf | σ_K/K B0 → B3 → B4 |
|---|---:|---:|---:|---|
| CANON Goldstone–Canberra, dense | +0.081 | −0.023 | **+0.046** | 7.46 → 18.8 → 9.37% |
| CANON Goldstone–Canberra, OPS | +0.067 | −0.024 | **+0.036** | 7.65 → 18.0 → 9.51% |
| O4 polar Goldstone–Canberra, dense | +0.146 | +0.017 | +0.082 | 15.5 → 31.8 → 20.8% |
| O5 GEO-max Goldstone–Canberra, dense | +0.324 | −0.044 | +0.169 | 7.96 → 20.3 → 10.7% |

**Predeclared survival rule** (B3 keeps Δf ≥ 0.02 and ≥ 50% of B0): **0/18 dense, 1/18 OPS**
(E1 Goldstone–Madrid). Under B3 the max |corr(K, session bias)| is 0.57–0.68 on the
canonical-epoch Goldstone–Canberra cases.

A per-session constant absorbs exactly the slowly varying plane-of-sky offset that carries ΔDOR's K
information. Only a calibrated prior keeps part of it.

## 26. Quasar-Coordinate Systematic

The quasar-coordinate systematic was run as part of D1 (§14): 0.75 nrad per component, one
independent quasar per session. It is **the dominant calibration systematic for K** in this
geometry: ≈4.9% of K on Goldstone–Canberra (0.66 σ formal) and ≈1.2% on Goldstone–Madrid. The
mapping is `δτ = −B·δŝ_q/c`, with B the J2000 baseline per row; it is transparent and checked by
construction.

## 27. End-to-End State FD Qualification

Setup:
- cases FD1 CANON Goldstone–Canberra (170 rows), FD2 E2 Goldstone–Canberra (95 rows), FD3 O4 polar
  Goldstone–Madrid (93 rows);
- the nominal row set is frozen, and no row was excluded;
- the oracle is an independent 6-state `propagate_state` re-propagation, plus the production
  `solve_common_transmit_event` at the same receive tags.

Maximum best-step relative error (Frobenius over all rows) per state column:

| case | max error |
|---|---:|
| FD1 | 2.39e-5 |
| FD2 | 2.04e-5 |
| FD3 | 1.57e-5 |

`DDOR_STATE_JACOBIAN_FD_GATE = PASS` (gate 1e-4). `MAX_DDOR_STATE_FD_RELERR = 2.39e-5`.

## 28. End-to-End K FD Qualification

At the predeclared oracle tolerance (rtol 1e-12) the K sweep is erratic:

| case | step 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---:|---:|---:|---:|
| FD1 | 0.51 | 3.2e-3 | 0.031 | 3.9e-3 |
| FD2 | 0.22 | 0.15 | 0.010 | 0.028 |
| FD3 | 0.50 | 0.027 | 0.063 | 0.024 |

The oracle's own step-halving floor is 1.9–3.9e-2. So
`DDOR_K_JACOBIAN_FD_GATE = FAIL_NOT_MATERIAL` at the predeclared oracle (maximum 2.36e-2). Its
`f_perp` effect is at most 3.7e-4.

The floor-aware follow-up keeps the same steps and the same gate, and tightens only the independent
oracle's integrator:

| oracle rtol | best K error (FD1 / FD2 / FD3) |
|---|---|
| 1e-13 | 2.5e-4 / 1.0e-3 / 6.1e-4 |
| 1e-14 | **2.1e-5 / 2.0e-4 / 4.1e-5** |

The errors fall with the oracle's noise, so the first result was oracle resolution, not a model
error. **K gate with refined oracle: PASS** (≤ 1e-3). `MAX_DDOR_K_FD_RELERR = 2.0e-4` (refined) /
2.36e-2 (predeclared oracle). Both are reported, and the gate was not relaxed.

## 29. FD Floor / Step-Size Interpretation

- **Position and velocity columns** show the classic V shape: truncation dominates at 300–1000 m
  and 0.3–1 m/s, and noise at 10 m and 0.01 m/s. The best steps are 100 m and 0.1 m/s, where error
  sits at the 1e-5 model/oracle floor.
- **The K signal per step is small**, and the integrator's adaptive-step noise at rtol 1e-12 is
  comparable to it at the small steps. That noise fixes the predeclared floor, and it falls about
  100× at rtol 1e-14. This matches the error falling with oracle tolerance (§28).
- **Materiality.** Replacing the whole analytic ΔDOR block with the best-step FD block changes the
  combined `f_perp` by at most 3.7e-4. That is below the 1e-3 materiality line and about 200× below
  the canonical Δf.

## 30. Production vs Surrogate Agreement

The largest disagreement over the 17 primary cases with data is |Δf_perp| = 2.9e-4 (§12). The
canonical historical rows give 0.320311 vs 0.320440. The two independent derivations of the ΔDOR
design agree on every geometry, including the DIRECTION_LOSS cases.

## 31. Production-Fidelity Dynamics Spot Check

Force model: GL1800F 50×50 in MOON_PA_DE440, plus Earth and Sun point-mass third body (SPICE), plus
the same SRP. The harmonic-STM opt-in is used, and no new force model was implemented.

| case | simplified range / Δf | fidelity range / Δf | class | max \|Δr\| |
|---|---|---|---|---:|
| CANON Goldstone–Canberra | 0.0915 / +0.081 | 0.0943 / +0.073 | STRONG → STRONG | 46.4 km |
| O2 full-Sun Goldstone–Canberra | 0.1142 / +0.157 | 0.1105 / +0.153 | STRONG → STRONG | 15.2 km |
| O5 Goldstone–Madrid (rule-selected weakest) | 0.2814 / −0.081 | 0.3107 / −0.111 | LOSS → LOSS | 21.5 km |

The OPS classes are also preserved. Under fidelity dynamics the B4 values are:
- CANON Goldstone–Canberra: +0.039;
- O2 Goldstone–Canberra: +0.071;
- O5 Goldstone–Madrid: −0.115.

`PRODUCTION_FIDELITY_DYNAMICS_EFFECT = PRESERVES_CLASS`. Three cases only; this is a spot check.

## 32. Information Direction

Over the 17 primary cases with ΔDOR data:

| quantity | min | median | max |
|---|---:|---:|---:|
| combined `f_perp` | 0.071 | 0.172 | 0.605 |
| Δf | −0.081 | +0.012 | +0.324 |

Including the one no-data case (Δf = 0), the median over all 18 is +0.010.

## 33. Information Magnitude

ΔDOR always adds K information where it has rows. For canonical Goldstone–Canberra, `I(K|x)` goes
from 1.55e5 to 1.80e6 and σ_K/K from 25.4% to 7.5%. In the full-Sun case, magnitude stays poor
(σ_K/K 35% combined). Magnitude and direction are separate: E1 Goldstone–Canberra gains little
direction (+0.024) yet still improves σ_K/K from 9.1% to 6.6%.

## 34. Singular-Mode Diagnostics

Every metric comes from the SVD of the whitened, scaled **design**:
- design rank is 7 in every case, with condition 1.0e8–2.2e9;
- the weakest right-singular vector has |K component| = 1.00000000 in every case.

ΔDOR rotates K's signature partly out of the state span but never removes K as the least-determined
direction.

`NORMAL_MATRIX_RANK_USED_AS_OBSERVABILITY_GATE = NO`.

## 35. Relative Gain Classification

Primary population (18 cases):

| class | count |
|---|---:|
| STRONG | 6 |
| MODEST | 2 |
| MAGNITUDE_ONLY | 8 (including E3 Goldstone–Madrid, no dual visibility) |
| DIRECTION_LOSS | 2 |

`DIRECTION_GAIN_CLASS = GEOMETRY_DEPENDENT`: STRONG only on Goldstone–Canberra at the canonical
epoch, 6/6 there and 0/12 elsewhere.

## 36. Absolute Identifiability Classification

Of the 17 cases with ΔDOR data:
- WEAK_K_SEPARATION: 16;
- MODERATE_K_SEPARATION: 0;
- STRONG_K_SEPARATION: 1 (O5 Goldstone–Canberra, 0.605).

`ABSOLUTE_K_IDENTIFIABILITY_CLASS = WEAK_K_SEPARATION_IN_16_OF_17` (K remains the weakest design
direction everywhere).

## 37. Robustness Across Quasar Geometry

Robust by construction: the formal information is quasar-invariant. The systematic bias varies by
less than 1% of K across the tested separations and position angles.

## 38. Robustness Across Station Pairs

**Not robust.**
- Goldstone–Canberra: STRONG 6/9.
- Goldstone–Madrid: STRONG 0/8, with 2 losses.
- Madrid–Canberra: unusable.

## 39. Robustness Across Epochs

**Not robust.** STRONG only at the canonical epoch (0/7 STRONG at E1–E3, both pairs).

## 40. Robustness Across Orbit Geometry

**Robust on Goldstone–Canberra at the canonical epoch** (5/5 orbit variants STRONG). Not on
Goldstone–Madrid.

## 41. Robustness to Calibration Freedom

**Not robust to free session biases:** 0/18 dense and 1/18 OPS survive, by the predeclared rule.
It is **partially robust to calibrated biases** (B4 keeps ~50–60% on the canonical-epoch
Goldstone–Canberra cases).

## 42. Physical Mechanism

ΔDOR complements range only when its own measurement projection (plane-of-sky position along the
**projected baseline**) contains a K-induced displacement component that the six-state basis cannot
reproduce. In this campaign that happens for the canted Goldstone–Canberra baseline in the
near-new-Moon geometry, where range is at its weakest (line of sight 12.5° from SRP). When the
ΔDOR block's own K signature is state-reproducible, heavy ΔDOR weight drags the combined direction
toward that low separability. This is the case for Goldstone–Madrid everywhere and for
Goldstone–Canberra at the other epochs, and it produces MAGNITUDE_ONLY or DIRECTION_LOSS.

Two further mechanisms act on top of this:
- the gain varies with precision in a hump (§22), not monotonically;
- a free per-session calibration bias can mimic most of the slowly varying K offset (§25).

## 43. What Changed in the Repository

| commit | content |
|---|---|
| `1ca5356` | literature contract; predeclared grid (+ writer); `phase17_r1od2_core.py` |
| `89d151d` | campaign, FD (+ K refinement), fidelity, figures and summary scripts; `tests/test_r1od2_ddor_contracts.py` (7 tests); all `phase17_r1od2_*` artifacts |
| final | this report and `studies/phase17/results/phase17_r1od2_manifest.json` |

**Changes made to the D2 analysis code during the phase, disclosed.** All three are metric or
bookkeeping defects in D2's own new code, not threshold changes:

- **Before the first run:** a scaled design for the bias least-squares, and parent-session
  preservation in operational subsets.
- **After the first run:** B4 came out identical to B3, which revealed that the prior did not enter
  the direction projector. It now does. `f_perp`, B0 and B3 were unaffected.

## 44. What Did Not Change

`lunar_od/` has an empty diff: `PRODUCTION_CODE_CHANGED = NO`. Also unchanged:
- the physical, measurement and estimator models;
- R1COV's covariance method;
- historical scripts and reports;
- the two R1O-D orphans (not run, not modified, not committed).

## 45. Academic Interpretation

**Complementarity between tracking data types is a property of geometry, not of the data type.** The
same production ΔDOR observable gives:
- a strong K-direction gain on one baseline at one lunar phase;
- nothing on another baseline;
- a *loss* when it is weighted heavily and its own signature is state-reproducible.

Three methodological points follow.

- **Relative and absolute identifiability must be kept apart.** A +0.08 direction gain on a weak
  baseline still leaves K the weakest design direction.
- **Measurement-schedule realism is part of fixture consistency.** An elevation-only co-visibility
  rule silently admitted 35% impossible (occulted) observations and nearly doubled the published
  gain.
- **An FD oracle has a resolution floor.** It must be characterized before a "FAIL" is read as a
  model error; here the K column needed a 100× tighter integrator to resolve 1e-3.

## 46. Operational Interpretation

For a synthetic LLO K_SRP problem like this one, ΔDOR is not a generic fix. A navigation team
considering it would need:

- the **canted Goldstone–Canberra** baseline, with its declination-like sensitivity (DSN-210 p. 13);
- sessions scheduled in the Earth–Sun geometry where range is weakest;
- **calibrated** session systematics. Free per-pass biases erase the gain, and the quasar-coordinate
  error (0.75 nrad) alone biases K by about 5% of its value, 0.66 σ formal.

Whether real mission geometry, real catalogs and real calibration budgets allow this is not
established here.

## 47. Limitations

- One epoch family per orbit case: the O cases are all at the canonical epoch, so orbit and epoch
  effects are not crossed beyond CANON × E1–E3.
- 4 epochs only, and one arc length (15 orbits, ≈29 h).
- Synthetic quasars; no flux or SNR model.
- Formal covariances only; no nonlinear estimation; no Monte Carlo.
- The session model is one bias per contiguous dual-visibility window.
- The fidelity check covers 3 cases at 50×50.
- The range Jacobian keeps its known 1e-6–3e-5 systematic. It is carried with GEO's measured
  |Δf_perp| ≤ 3.8e-6 and not investigated.

The smallest classification margins:
- O1 Goldstone–Madrid, dense 0.048 ns delay: −0.0210 against the −0.02 threshold, 1.0e-3 margin;
- canonical Goldstone–Canberra, dense 0.048 ns: +0.0512 against 0.05, 1.2e-3 margin.

Both still exceed the 100× range-Jacobian bound (3.8e-4) and the ΔDOR FD materiality (3.7e-4), so
no classification is `JACOBIAN_LIMITED`.

## 48. Synthetic-Truth Boundary

`SYNTHETIC_CAMPAIGN_TRUTH`: K = 0.01 m²/kg, the campaign orbit, and synthetic quasars and
error budgets.

## 49. Mission-Truth Boundary

No result here is a statement about a real spacecraft's K_SRP observability, a real DSN schedule or
a real radio source.

## 50. Regression Tests

Targeted run: **184 passed, 0 failed.** It covered:
- FXR fixture tests (10) and D2 contract tests (7);
- R1O-R STM layout (5);
- production ΔDOR;
- two-way range and range integration;
- SRP K sensitivity and the K-FD harness;
- R1COV square-root covariance;
- production optical;
- BLS / SRIF / SR-UKF K solve-for default parity.

## 51. Full Suite

Command: `python -m pytest tests -o addopts="" -q -rfE`.

```
TOTAL_TESTS_COLLECTED = 1482   (1336 tests + 146 subtests)
TESTS_PASSED          = 1441   (1295 + 146 subtests)
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

The entering FXR baseline was 1475 / 1434 / 2 / 10 / 29. The whole difference is the 7 new D2
contract tests. The non-pass set is identical:

| non-pass | class |
|---|---|
| `test_fa06_pytest_session_imports_isolated_repository` | KNOWN_PREEXISTING_ENVIRONMENT |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change`: R2 byte gate on `src/lunar_od/dynamics.py`, unchanged by D2 | KNOWN_PREEXISTING_PROVENANCE |
| 10 downstream `test_r2_measurement_fidelity.py` errors | KNOWN_PREEXISTING_PROVENANCE |

`D2_INTRODUCED_NONPASSES = 0`, `NEW_SCIENTIFIC_REGRESSIONS = 0`, `UNKNOWN_NONPASSES = 0`.

## 52. What This Phase Actually Established

1. FXR's canonical ΔDOR number is reproduced exactly. It included 93 occulted rows. The honest
   canonical gain is +0.081 (STRONG, relative), not +0.229.
2. The production ΔDOR design is FD-qualified end to end: state columns ≤ 2.4e-5; K ≤ 2.0e-4 with a
   resolving oracle.
3. ΔDOR's formal K information is exactly quasar-independent. Quasar geometry matters only through
   systematics, dominated by the quasar-coordinate term.
4. The strong direction gain is confined to Goldstone–Canberra at the canonical epoch. It survives
   orbit geometry and higher-fidelity dynamics there, but not other epochs or Goldstone–Madrid.
5. It is erased by free per-session calibration biases and halved by calibrated ones.
6. K remains the weakest design direction in every case. Absolute separation stays weak in 16/17.

## 53. What This Phase Did NOT Establish

- K identifiability in a real mission.
- Nonlinear estimator recovery of K.
- Performance with real catalog sources.
- The optimal tracking schedule: none was searched, by design.
- The behaviour over other epochs crossed with other orbits.
- The range-Jacobian root cause.

## 54. What Remains Unknown

- The precise geometric predictor, beyond "ΔDOR-only separability".
- Whether multi-baseline ΔDOR (Goldstone–Canberra + Goldstone–Madrid jointly) or longer arcs widen
  the favourable set.
- How calibrated-bias priors should be set for this problem.
- Whether a tracking-design phase can find the favourable geometry deliberately.

## 55. Decision Tree

- **A (robust): no.** 33% STRONG (< 75%), and 0% B3 survival.
- **C (calibration-fragile) as the primary class: no.** Its precondition (A's geometry test) fails.
  Calibration fragility is nevertheless a **confirmed secondary finding**.
- **B (geometry-dependent): yes.** 6/18 STRONG, all in one pair × epoch cell.
- **D (does not generalize): no.** 33% ≥ 25%, and the gain survives orbit and fidelity changes
  within its cell.

Consequence (brief s39): the next question is **tracking-design characterization**, not K
parameterization, and not yet a nonlinear-estimator campaign on one favourable geometry.

## 56. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = a4304fc35a27b212fcab1eaf6f936e3d45fe56c2
START_TREE   = b6f5d9ef114ede86f029e2ed081d48447a391639
MAIN_HEAD = ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b
D2_INPUT_GATE = PASS
D2_EARTH_FIXTURE_GATE = PASS
D2_EPOCH_CONSISTENCY_GATE = PASS
GRAPHIFY_DISCOVERY_GATE = PASS
D2_LITERATURE_CONTRACT_GATE = PASS
D2_GRID_PREDECLARED = YES
D2_CANONICAL_REPRODUCTION_GATE = PASS
DDOR_NOISE_SEMANTICS_GATE = PASS
DDOR_STATE_JACOBIAN_FD_GATE = PASS
DDOR_K_JACOBIAN_FD_GATE = PASS with a resolving oracle (FAIL_NOT_MATERIAL at the predeclared oracle)
PRODUCTION_FIDELITY_DYNAMICS_EFFECT = PRESERVES_CLASS
NORMAL_MATRIX_RANK_USED_AS_OBSERVABILITY_GATE = NO
ROBUST_STRONG_DIRECTION_GAIN_FOUND = NO
PHASE17_R1O_D2_GATE = PASS
PRIMARY_CLASS = DDOR_K_INFORMATION_GAIN_IS_STRONGLY_GEOMETRY_DEPENDENT
```

## 57. Exact Next Action

1. Owner review of this report, including the occultation correction to FXR's ΔDOR headline.
2. Then a **predeclared ΔDOR tracking-design characterization phase**:
   - Goldstone–Canberra vs Goldstone–Madrid vs joint two-baseline ΔDOR;
   - a denser epoch sweep across the synodic month;
   - realistic session scheduling;
   - calibrated-bias priors;

   with the ΔDOR-only K separability as the candidate predictor. Only if a robust favourable design
   emerges should a nonlinear-estimator confirmation follow.

Not started here: K parameterization, stochastic K, consider-parameter estimation, empirical
acceleration, nonlinear recovery, Monte Carlo, the range-Jacobian root cause.
