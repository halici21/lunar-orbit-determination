# Phase 17-KP1 — Alternative K_SRP Parameterization Characterization

Truth level: **SYNTHETIC_CAMPAIGN_TRUTH**. K_truth = 0.01 m²/kg is constant. No time-varying truth was fabricated.
This is linear information geometry at the truth: there is no nonlinear recovery, no Monte Carlo and no use of
Doppler values.

- **Code:** `examples/phase17_kp1_{core,predeclare,campaign}.py`.
- **Artifacts:** `artifacts/phase17_kp1_*`.
- **Tests:** `tests/test_kp1_alternative_k.py`.
- **Predeclaration:** `artifacts/phase17_kp1_predeclaration.json`, canonical sha256 `4f7f2632…e495`. It was committed
  in `90bec18` before any separation metric existed.

## 1. Result

**`PRIMARY_CLASS = ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS`** (Case A, by the predeclared tree).
The recommended parameterization is `M3_THREE_SEGMENT`.

The headline needs its qualifier: **what becomes distinguishable is the *time variation* of K, not its level.**

- **Variation is separable.**
  - Under the primary case (high fidelity, range + counted Doppler, one free bias per pass), K's variation directions
    separate from the six-state initial condition *and* from the constant K.
  - M3 reaches MODERATE or better in **57.5 %** of 120 cells (rule ≥ 50 %), with internal correlations ≤ 0.90 in every
    cell.
  - M1 (two halves) is PARTIAL at 25.8 %. M2 (linear drift) carries no new information: 5.8 %.
- **The level is not.**
  - In every model, the block's weakest combination is close to the common mode (mean K).
  - Its principal sine is *below* the scalar f_perp: median 0.084 for M3 and 0.089 for M1, against 0.133 for M0.
  - The constant-K weakness found throughout Phase 17 therefore stays. It is a measurement limitation of the K level.
    The scalar parameterization did not create it.
- **Answer to the phase question: both, in different directions.**
  - The available measurements cannot separate the level of K from the state.
  - The global-constant model hides real information: piecewise variation of K is observable when present.
- **It is orbit-dependent.**
  - M3's variation separates in 62–88 % of epochs on CANON, O1, O2 and O5, but in **0 / 24** on the 500 km orbit (O3).
  - O3 is the one orbit where M1 does best (50 %).
- **It is fidelity-robust, unlike D4 and CD1.**
  - Simplified dynamics reproduce 11 of 12 case × model classes: all 3 primary and 8 of 9 secondary. The exception
    is M2 under ideal Doppler, PARTIAL at 0.108 against 0.092.
  - M3's per-cell MODERATE+ decision agrees in 120 / 120 cells (RMS Δ 0.018).

## 2. Question and Scope

Is the weakness seen throughout Phase 17 a limitation of the measurements, a consequence of forcing K_SRP into one
global constant, or both? KP1 answers with linear, local distinguishability only. It does not start nonlinear
estimation, stochastic K, empirical accelerations or a ΔDOR search.

## 3. Entering State (verified)

| item | value |
|---|---|
| branch / HEAD / tree | `feature/phase17-r-k-srp-estimation` / `12839fe8…` / `3176b345…` |
| main = origin/main | `fea476f8…` |
| entry suite (plain invocation) | 1460 collected / 1431 passed / 0 failed / 0 errors / 29 skipped |
| gates | HARDEN, F1, D3, D4, CD1 all PASS |

## 4. Candidate Discovery

- **K semantics.** a_SRP = K·P(d)·ν·r̂, with K = C_R·A/m constant (`lunar_od/srp.py`). dx/dK solves the static-parameter
  variational equation with the F1 A matrix (`propagate_state_with_k_sensitivity`).
- **LITERATURE_RESULT** (sources read in full text in the GEO contract):
  - S1 (LRO OD, §6.3): SRP variations from variable spacecraft area are *time-dependent* dynamical errors.
  - S2 (LRO EKF): operations model the SRP scale factor as a *stochastic* process (Vasicek better than Gauss–Markov).
  - S1 §6.2: cannonball C_R estimation fails in full-Sun periods.
  - No literature statement is used as evidence of KP1 identifiability.
- **REPOSITORY_HYPOTHESIS.** A deterministic low-dimensional K(t) carries measurement directions that the six-state
  initial condition and the constant K cannot reproduce. Such a K(t) is the batch analogue of the stochastic scale
  factor.

| model | definition | parameters | physical role |
|---|---|---|---|
| M0_SCALAR | K(t) = p0 | 1 | baseline (CD1 reproduced) |
| M1_TWO_SEGMENT | constants on [0, T/2), [T/2, T] | 2 | arc-wise scale factor, ~14.7 h correlation (S2) |
| M2_LINEAR_DRIFT | p0 + p1·s(t), s = (t − T/2)/(T/2), 90 s staircase | 2 | first-order term of any slow K(t) (S1) |
| M3_THREE_SEGMENT | constants on equal thirds | 3 | complexity escalation |

**Rejected:**
- once-per-rev attitude modulation: it needs an attitude or multi-plate model, i.e. new physics;
- per-orbit K: 15 parameters;
- splines and high-order polynomials;
- stochastic Gauss–Markov or Vasicek K: deferred by the brief;
- shadow-regime K: unphysical.

## 5. Mathematical Contract

- **Parameters.** p_j in m²/kg, K(t) = Σ p_j b_j(t). At truth every model equals the constant K_truth exactly: segments
  all K_truth, drift 0.
- **Sensitivities are exact compositions of production quantities.** There is no new integration and no production
  change. With J(t) = Φ(t)⁻¹·dx/dK(t):
  - a window is S_[a,b](t) = Φ(t)(J(clip(t, a, b)) − J(a));
  - the drift staircase is Φ(t) Σᵢ sᵢ (J_{i+1} − J_i);
  - all boundaries lie on grid nodes.
- **Design columns.** They come from the production range (`_two_way_range_k_srp_column`) and counted-Doppler
  Jacobians, with each sensitivity in the STM slot (the HARDEN/F1 construction). Five bank columns are packed into one
  call, which is linear in the slot (bitwise check).
- **Scaling** (declared before results):
  - state: [1e6 m ×3, 1e3 m/s ×3], the D2 convention;
  - parameters: fractional K (× K_truth); the drift unit is a K_truth change from mid-arc to the arc end.
- **Metrics** (whitened design, no normal matrix, no rank gate):
  - **s_min / s_max:** principal sines between span(H_p) and span(state [+ nuisance]). They are basis-invariant. For
    M0, s = f_perp.
  - **s_var_min:** the smallest non-zero principal sine against span(state [+ nuisance] + h_K), i.e. the separation of
    the *variation* directions from state and constant K.
  - **Conditioned block B = P⊥H_p:** singular values, condition number, weakest combination, and σ and correlations
    from the QR of B.

## 6. Experiment (predeclared)

- **Population:** CD1's 5 orbits × 24 D3 epochs (15° spacing, canonical included), 15-orbit arcs. **High-fidelity
  dynamics are primary** (GL1800F 50×50, Earth + Sun third body, production SRP + F1). Simplified dynamics are a
  diagnostic only.
- **Measurements:** the CD1 range block (σ 5 m) and CD1 counted-Doppler rows (ALL_VISIBLE, Tc 60 s, structural σ 1e-4
  m/s).
- **Cases:**
  - `RANGE`;
  - `RPD_D1FREE` (**primary**, one free bias per pass);
  - `RPD_D0`;
  - `RPD_D1P1E-4`.
- **Classification:**
  - Thresholds STRONG ≥ 0.50 and MODERATE ≥ 0.35, as in the Phase-17 absolute classes. The internal threshold is
    max|corr| ≤ 0.90.
  - Per model: VAR = fraction of cells with s_var_min ≥ 0.35; INT = fraction of cells with max|corr| ≤ 0.90.
  - DISTINGUISHABLE if VAR ≥ 0.50 and INT ≥ 0.50; PARTIAL if VAR ≥ 0.10; else NO_NEW_INFORMATION.
  - Tree: A (any DISTINGUISHABLE; recommend the lowest-dimension one) → B (any PARTIAL) → C.

## 7. Results (high fidelity, 120 cells, medians)

| case | model | s_min | s_max | s_var_min | VAR | INT | cond | σ/K_truth per parameter | class |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| RANGE | M0 | 0.126 | — | — | — | — | — | 32.4 % | — |
| RANGE | M1 | 0.097 | 0.642 | 0.460 | 0.81 | 1.00 | 2.0 | 34 / 38 % | DISTINGUISHABLE |
| RANGE | M2 | 0.050 | 0.609 | 0.206 | 0.16 | 1.00 | 2.8 | 34 / 70 % | PARTIAL |
| RANGE | M3 | 0.101 | 0.967 | 0.542 | 0.80 | 1.00 | 2.2 | 53 / 44 / 45 % | DISTINGUISHABLE |
| RPD_D0 | M0 | 0.210 | — | — | — | — | — | 0.50 % | — |
| RPD_D0 | M1 | 0.133 | 0.642 | 0.397 | 0.73 | 1.00 | 1.3 | 0.66 / 0.71 % | DISTINGUISHABLE |
| RPD_D0 | M2 | 0.071 | 0.640 | 0.183 | 0.11 | 1.00 | 3.2 | 0.51 / 1.6 % | PARTIAL |
| RPD_D0 | M3 | 0.160 | 0.975 | 0.560 | 0.80 | 1.00 | 1.8 | 1.2 / 0.89 / 1.0 % | DISTINGUISHABLE |
| **RPD_D1FREE** | M0 | 0.133 | — | — | — | — | — | 0.92 % | — |
| **RPD_D1FREE** | M1 | 0.089 | 0.419 | 0.280 | 0.26 | 1.00 | 1.5 | 1.3 / 1.3 % | **PARTIAL** |
| **RPD_D1FREE** | M2 | 0.058 | 0.424 | 0.162 | 0.06 | 1.00 | 2.4 | 0.93 / 2.0 % | **NO_NEW_INFORMATION** |
| **RPD_D1FREE** | M3 | 0.085 | 0.782 | 0.412 | **0.575** | 1.00 | 2.2 | 1.9 / 1.4 / 1.7 % | **DISTINGUISHABLE** |

`RPD_D1P1E-4` gives the same classes as `RPD_D1FREE` (M3 VAR 0.575).

### 7.1 Model complexity: new information or a split weak direction?

- **New information, but only in variation.**
  - M3's two variation sines (median 0.51 and 0.78) are well separated from state + constant K.
  - Its third sine (0.084) is the weak common mode.
  - Internal correlations stay moderate (median max|corr| 0.47; condition ≈ 2).
  - So the extra freedom is not a split of the weak scalar direction into more-degenerate copies.
- **It does not repair the level.**
  - Every block's worst combination is weaker than the scalar's. This must be so: the constant lies inside every block.
  - The weakest conditioned combination is common-mode-like: M3 ≈ [0.73, −0.26, 0.55]; M1 ≈ [0.68, 0.42].
- **Window placement matters more than count.**
  - M1 and M3 are not nested: the M1 boundary at T/2 is not an M3 boundary.
  - M3's median s_var_min (0.41) exceeds M1's (0.28). The predeclared `SPLITS_WEAK_DIRECTION` test is therefore
    **false**.
  - After a window closes, its signature is exactly state-like (Φ(t)·c). Its distinct content is what accumulates
    inside the window, so separation depends on where windows fall relative to the tracking and eclipse geometry.
  - This is an interpretation, not a tested hypothesis.
- **Smooth drift is mimicked.** M2's drift direction has median s_var_min 0.16. A linear K trend over the arc is
  largely reproduced by the state plus the constant K.

### 7.2 Nuisance and measurement impact

- **The Doppler per-pass bias costs variation separation.**
  - M1: VAR 0.73 → 0.26 and median s_var_min 0.40 → 0.28.
  - M3: VAR 0.80 → 0.575 and median s_var_min 0.56 → 0.41.
  - This is the same mechanism CD1 found for the scalar.
- **Range alone already separates the variation** (M1 VAR 0.81, M3 0.80), but with σ of 34–53 % of K_truth.
  - Doppler adds *magnitude*: σ falls to 1.3–1.9 %.
  - Under the free bias, Doppler lowers the *direction* metric. A sine is not monotone in added data, and the precise
    Doppler rows re-weight the geometry.

### 7.3 Orbit dependence (primary case, VAR)

| orbit | M1 | M2 | M3 | scalar f_perp |
|---|---:|---:|---:|---:|
| CANON | 0.33 | 0.04 | 0.62 | 0.104 |
| O1 low β, eclipse-rich | 0.25 | 0.25 | 0.71 | 0.102 |
| O2 full Sun, high β | 0.17 | 0.00 | 0.67 | 0.120 |
| O3 500 km | 0.50 | 0.00 | **0.00** | 0.181 |
| O5 favourable geometry | 0.04 | 0.00 | 0.88 | 0.186 |

- **Population margin.** M3 passes with 69 / 120 cells against the 60 required. The closest cells sit at 0.340, 0.353
  and 0.355.
- **Orbit structure.** The result is a population statement with a clear orbit structure. It is not a property of
  every orbit.

### 7.4 Fidelity

- **Classes.** Simplified dynamics give identical classes in the primary case: M1 PARTIAL, M2 NO_NEW_INFORMATION and
  M3 DISTINGUISHABLE (VAR 0.575). They also match in 8 of 9 secondary case × model classes. The exception is M2 under
  `RPD_D0`: PARTIAL at VAR 0.108 in high fidelity, NO_NEW_INFORMATION at 0.092 in simplified dynamics. It sits at the
  0.10 boundary and does not affect the decision.
- **Per cell.** M3's per-cell MODERATE+ decision agrees in all 120 cells; the RMS of the s_var_min difference is 0.018.
- **Contrast with earlier phases.** Unlike D4 and CD1, the KP1 verdict does not hinge on the range-baseline fidelity
  effect.

## 8. Verification

| check | result | gate |
|---|---|---|
| M0 reproduces CD1 committed f_perp (RANGE, RPD_D0, RPD_D1FREE; 240 cells) | max 1.7e-14 | ≤ 1e-10 |
| bank K column == CD1 production h_K (range and Doppler) | bitwise, every cell | bitwise |
| exact composition vs closed-form double integrator (windows, drift) | ≤ 1e-12 relative | test |
| **FD oracle:** segmented high-fidelity repropagation, K(t) = K_truth + ε b(t), restarts at boundaries / every node, ε ∈ {1e-3, 3e-3}; SEG2_A, SEG3_B, DRIFT on CANON k0 and O2 k12 | worst 7.85e-6 | ≤ 1e-4 |
| FD control: per-node-restarted vs single propagation, same K | 1.05e-6 m | (noise floor) |
| independent rebuild (lstsq residual + generalized eigenproblem), every high-fidelity cell × case × model | max 5.6e-14 | ≤ 1e-8 |
| exact rational oracle (Fraction Schur complement, exact characteristic polynomial, 60-digit Newton), CANON k = 0/6/12/18, primary, all models | max 1.1e-15 | ≤ 1e-12 |
| packed vs single-slot Doppler partial | bitwise | — |
| predeclaration hash pinned, tamper refused | test | — |

- **Values not used.** No counted-Doppler value or residual is used anywhere. CD1's closure showed HARDEN-D23 moves
  partial-based metrics by ≤ 4e-10.
- **FD oracle.** The FD oracle differences repropagated *states* only.

## 9. Limitations

- **Linear and local.** The analysis is linear and local at a constant truth. It establishes that variation
  *directions* are distinguishable, not that a varying K would be recovered.
- **Doppler σ.** The Doppler σ is the CD1 structural value. Absolute σ values scale with it; principal sines do not.
- **Arc and stations.** One 15-orbit arc length and DSN3 only. The segment boundaries are fixed fractions of that arc.
- **Spacecraft model.** Cannonball SRP and the spherical-Moon penumbra are used; there is no attitude or area model.
- **Doppler values.** HARDEN-D23 (~1.8e-3 m/s Doppler value offset) still blocks any value-based use.

## 10. Final Status

```text
PHASE17_KP1_GATE = PASS

PRIMARY_CLASS = ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS
                (variation directions only; the K level stays WEAK)

RECOMMENDED_K_PARAMETERIZATION = M3_THREE_SEGMENT (piecewise-constant K on equal thirds of the arc)

SCALAR_K_BASELINE_STATUS = REPRODUCED (CD1 to 1.7e-14); scalar K level remains WEAK
                           (primary median f_perp 0.133); every richer block's weakest combination is weaker (0.08-0.09)

STATE_PARAMETER_SEPARATION_STATUS = VARIATION_SEPARATED_LEVEL_WEAK
                                    (M3 variation sines median 0.51 / 0.78; common-mode sine 0.084)
INTERNAL_PARAMETER_SEPARATION_STATUS = INTERNALLY_SEPARATED (max|corr| <= 0.90 in 120/120 cells for M1-M3;
                                       median 0.47 for M3; condition ~2)

COUNTED_DOPPLER_NUISANCE_IMPACT = MATERIAL (free per-pass bias: M3 VAR 0.80 -> 0.575, M1 0.73 -> 0.26)

NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED = NOT_YET (Case A permits it, but it needs counted-Doppler VALUES:
                                             D23 must be repaired/qualified first; the K level stays WEAK)

STOCHASTIC_K_FOLLOWUP_JUSTIFIED = YES (predeclared rule: variation directions with VAR >= 0.10;
                                  physically motivated by S2; after D23)

D23_REPAIR_REQUIRED_BEFORE_NEXT_PHASE = YES (any next phase that fits K - segmented or stochastic - uses Doppler values)

PRODUCTION_CODE_CHANGED = NO
PHYSICAL_MODEL_CHANGED = NO
ESTIMATOR_CHANGED = NO

TOTAL_TESTS_COLLECTED = 1477
TESTS_PASSED = 1448
TESTS_FAILED = 0
TESTS_ERRORS = 0
TESTS_SKIPPED = 29 (the same opt-in/data-dependent set as at entry)

NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0

FINAL_HEAD = the KP1 results commit that adds this report (a commit cannot record its own hash); its parent is
             predeclaration commit 90bec18
FINAL_TREE = the tree of that commit

PUSH = NONE
MERGE = NONE
MAIN_CHANGED = NO
ORIGIN_MAIN_CHANGED = NO

NEXT_ACTION = hard stop. Recommended, separately authorised, in order:
              (1) HARDEN-D23 repair/qualification of the counted-Doppler Earth interpolation;
              (2) only then a nonlinear or stochastic-K phase that uses the M3 segmentation (or a
                  Vasicek/Gauss-Markov scale factor) with the D1 per-pass bias as reference nuisance,
                  and without claiming the K level is identifiable.
```

Suite counts are from the supported plain invocation (`python -m pytest -o addopts="" -p no:cacheprovider`, no
PYTHONPATH) on the final tree.
