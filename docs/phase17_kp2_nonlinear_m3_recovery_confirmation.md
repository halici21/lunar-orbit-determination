# Phase 17-KP2 — Nonlinear M3 K_SRP Recovery Confirmation

- **Truth level:** **SYNTHETIC_CAMPAIGN_TRUTH**. The common level is K0 = 0.01 m²/kg in every truth.
- **Data:** counted-Doppler values are used on the D23-qualified path.
- **Code:** `examples/phase17_kp2_{core,predeclare,campaign}.py`.
- **Evidence:** `artifacts/phase17_kp2_*`.
- **Tests:** `tests/test_kp2_m3_recovery.py`.
- **Predeclaration:** `artifacts/phase17_kp2_predeclaration.json`, canonical sha256 `111856fb…`, committed in `e2012b1`
  before any high-fidelity recovery result.

## 1. Result

**`PRIMARY_CLASS = M3_VARIATION_PARTIALLY_RECOVERABLE`** (predeclared Case B).

| predeclared figure | value | rule |
|---|---:|---|
| R_sep: recovered T1/T2 trials in KP1-separable cells | **8 / 12 = 0.667** | A needs ≥ 0.80; B ≥ 0.30 |
| FP: false variation detections on the constant truth T0 | **0 / 10** | ≤ 0.20 |
| CONS: χ²_err ≤ 13.82 among converged trials | **28 / 28** | ≥ 0.80 |
| NEES₉ ≤ 27.88 among converged trials | **28 / 28** | ≥ 0.9 → CONSISTENT |
| recovered in KP1-weak cells (contrast) | **7 / 8** | reported |
| all variation trials | 15 / 20 recovered; 28 / 30 trials converged | — |

- **Where it works.** The estimator recovers the *time variation* of K wherever the data carry it. The estimates are
  statistically consistent and never invent a variation.
- **Why it is only partial.** All but one of the misses are in the full-Sun orbit O2 (§4).
- **The common level stays not claimed.** It is formally constrained, but KP1 showed it is state-degenerate.

## 2. Design (frozen in `e2012b1`)

- **Model.**
  - M3: K(t) = K_j on equal thirds of the 15-orbit arc.
  - It is realised exactly by segmented production propagation (`propagate_state_with_k_sensitivity` restarted at the
    boundaries).
  - The STM is chained, Φ(t, t₀) = Φ_j(t, t_j) Φ(t_j, t₀), with exact per-segment dx/dK_j.
  - No production code is changed.
- **Truths.** Fractional segment deviations: T0 (0, 0, 0); T1 (+0.10, −0.10, 0); T2 (−0.05, +0.10, −0.05). All are
  zero-mean, so the level is K0 in every case.
  - Doppler bias: N(0, 1e-4 m/s) per pass.
  - White noise: 5 m range and 1e-4 m/s Doppler.
  - A priori: N(0, diag(100 m, 0.1 m/s)²).
  - All draws are seeded.
- **Cells.** 5 orbits × D3 epochs k ∈ {0, 12}, high fidelity (GL1800F 50×50 + Earth + Sun + production SRP/F1). One
  seed per (cell, truth) gives 30 trials.
- **Estimator.** Analysis-layer batch Gauss–Newton on production range and counted-Doppler **values** and production
  Jacobians.
  - Parameters: x₀ (6), K₁..K₃, and one free bias per Doppler pass (CD1 D1).
  - Staged start: x₀ + biases first, with K held at K0. Production estimators have no joint range + Doppler solve, no
    per-pass bias and no M3 K, and are unchanged.
  - Convergence: every step < 1e-2 σ.
  - Covariance: QR of the whitened design (no normal-matrix inverse).
- **Recovery of a trial:** converged ∧ χ²_err(variation) ≤ 13.82 ∧ χ²_detect(variation) ≥ 13.82, both on 2 degrees of
  freedom at 0.999.

## 3. Recovered Variation Is Tied to the Injected Variation

- **Consistent with truth.** Every converged trial has χ²_err ≤ 10.5, against the 13.82 gate. The estimated variation
  sits within its own uncertainty of the injected one, and T1 and T2 come out in their own directions:
  - CANON k0 T1: v̂ (0.123, 0.005) against v (0.141, 0).
  - CANON k0 T2: (−0.119, 0.068) against (−0.106, 0.061).
- **No false variation.** On T0, χ²_detect is ≤ 2.9 in all 10 trials. With no injected variation, the estimator does
  not manufacture one.
- **Not traded against state, level or biases:**
  - NEES over (x₀, f) is consistent in 28 of 28 converged trials.
  - Within the block, max |corr| among segments is ≤ 0.72 in every converged trial.
  - Per-pass bias estimates match truth to an rms of 2.3e-5 m/s, below the 1e-4 truth σ.
  - The level error is consistent: |z| ≤ 2.4 in every converged trial (median 0.86).
  - Whitened residual rms is 1.00 for range and 0.99 for Doppler.

## 4. Why Five Variation Trials Were Not Recovered

| trial | mechanism | evidence |
|---|---|---|
| O2 k0 T1, O2 k0 T2 | **insufficient information** | σ_v ≈ 0.08 against a 0.10–0.14 truth; consistent (χ²_err 0.9, 0.7) but undetectable (χ²_detect 1.3, 4.9) |
| O1 k12 T2 | **insufficient information** (marginal) | σ_v ≈ 0.033; χ²_detect 12.4, just below 13.82; consistent (χ²_err 0.2) |
| O2 k12 T1 | **convergence floor, estimate correct** | f̂ (0.142, −0.083, 0.023) against (0.10, −0.10, 0); χ²_err 0.63, χ²_detect 47; last step 0.0198 σ (criterion 0.01 σ); later steps rejected by ~1e-3 σ integrator cost noise |
| O2 k12 T2 | **estimator basin / iteration budget** | a priori error 254 m / 0.195 m/s (first cost 3e13); stage 1 did not settle within 4 evaluations and the solve diverged |

- **Basin check (post-hoc, labelled; `artifacts/phase17_kp2_posthoc_basin.json`).** The same O2 k12 T2 data and a
  priori, re-solved with 10 stage-1 and 20 stage-2 evaluations, **converges and recovers the variation**: χ²_err 1.2,
  χ²_detect 23.9, NEES 9.5, residual rms 0.99 / 0.98. The frozen result is unchanged.
- **The failure classes.**
  - None is nuisance absorption or a false signal.
  - O2 (full Sun, high β, no eclipses) carries the least variation information. Its σ_v is 2–9× the other orbits'.
  - The two O2 k12 non-convergences are a convergence-criterion artefact and a basin effect, respectively.

## 5. Linear Direction Does Not Predict Nonlinear Detectability

- **The contrast.** KP1-weak cells recovered 7 of 8, KP1-separable cells 8 of 12.
- **What predicts recovery.** The variation **magnitude** σ_v does: median 0.011 in recovered trials against 0.033 in
  missed ones. The KP1 direction metric s_var_min does not: corr(σ_v, s_var_min) = 0.23.
- **Interpretation.** A direction well separated from the state is not enough if the information along it is small,
  as in O2. Conversely, a weakly separated direction can still be detected when the Doppler information is large.
- **Consequence.** KP1's linear classification is necessary context, not a recovery predictor. That supports the
  predeclared rule's use of nonlinear χ² outcomes rather than the KP1 class.

## 6. Doppler Bias Impact (secondary, predeclared)

- **Setup.** T1 at k = 0 on all 5 orbits, re-solved with no truth bias and no bias parameters (D0).
- **Result.** Free per-pass biases inflate σ_v by a median factor of **1.24**. Recovery is identical: 4 of 5 in both
  D1 and D0; O2 is missed in both.
- **Class.** `DOPPLER_BIAS_IMPACT = MINOR` for *variation* recovery.
- **Contrast with CD1.** CD1 found the free bias destroys the *scalar* K's direction gain. The M3 variation directions
  are largely orthogonal to a per-pass constant.

## 7. Common K Level

`COMMON_LEVEL_STATUS = NOT_CLAIMED`.

- **What the numbers show.** The level is formally constrained under truth = model (median σ 1.0 % of K0) and its
  errors are consistent (median |z| 0.86).
- **Why it is not claimed.** KP1 showed the common mode is the weakest, state-degenerate direction (block sine
  ≈ 0.085). Its estimate therefore rests on the model being exactly right, and KP2 does not test robustness to model
  error.

## 8. Verification

| check | result | gate |
|---|---|---|
| KP1 anchor: the segmented design reproduces KP1 committed M3 σ and s_var_min at each T0 truth (10 cells) | max **8.8e-8** relative | ≤ 1e-5 — PASS |
| noise-free CANON k0 T1 recovery | f within **0.0022 σ**; x₀ within 0.0022 σ; 7 evaluations | ≤ 0.02 σ — PASS |
| M0 baseline: formal σ_K vs CD1 committed (CANON k0, O5 k0) | 0.0086203 vs 0.0086203; 0.0092162 vs 0.0092162 | ≤ 5 % — PASS |
| closed-form chaining (unit test, double integrator) | ≤ 1e-9 relative | test — PASS |
| FD: segmented repropagation vs chained dx/dK_j (CANON k0, O2 k12, 2 orbits) | CANON ≤ 9.1e-6; O2 segments 1–2 ≤ 6.7e-6; **O2 segment 3 5.3e-4** (1.86e-5 at the KP1 oracle tolerance) | ≤ 1e-4 — **FAIL as frozen**, adjudicated (§8.1) |
| production BLS cross-check (CANON k0, T0, range only, scalar K) | production stopped with `DampingLimit` and drove K to 2.9e-7 (boundary); KP2 converged (K̂ 0.0054); warm-started production agrees to 8e-4 σ | ≤ 0.02 σ — **FAIL as frozen**, adjudicated (§8.2; HARDEN-D25) |
| job errors | none | PASS |

### 8.1 FD exceedance

- **Frozen result.** O2 k12 segment 3 reached 5.3e-4 against the 1e-4 gate. Every other segment, and CANON throughout,
  is ≤ 9.1e-6.
- **Post-hoc check (labelled; `artifacts/phase17_kp2_posthoc_verification.json`).** The same FD job was rerun with
  nominal and perturbed propagations at rtol 1e-13 / atol 1e-14, the KP1 FD-oracle tolerance. Segment errors fall to
  5.2e-6 / 6.8e-6 / **1.86e-5**, all ≤ 1e-4.
- **Interpretation.** The exceedance was the FD oracle's own integration floor on the shortest-lived sensitivity, not
  an error in the chained dx/dK₃. The KP1 anchor (8.8e-8) and the closed-form chaining test confirm the construction
  independently.

### 8.2 Production BLS cross-check

- **Frozen result.** Production `estimate_two_way_range_bls_lm` stopped with `DampingLimit` and drove K to the zero
  boundary (2.9e-7). This is the long-arc range-only instability R1M recorded. The KP2 solver converged
  (K̂ = 0.00541). The frozen comparison therefore fails.
- **Stationarity check (post-hoc, labelled).** Production BLS was warm-started at the KP2 solution, on the same data
  and dynamics.
  - It **converged and stayed**: every estimate is within **8e-4 σ** (K 0.0054130 vs 0.0054110), so the two
    estimators share the least-squares solution.
  - The state σ agree within 0.1–2.8 %. Production's **σ_K is about 9 % larger**.
- **Root cause, from code inspection.** `_two_way_range_posterior_design_with_k` propagates the posterior design
  **without the harmonic gravity model** that the solve uses. Production's reported covariance is therefore
  linearised on a different force model.
  - It is recorded as ledger item **HARDEN-D25** and not fixed here, since an estimator change is out of scope.
  - KP2's covariances come from the QR of its own design, which reproduces CD1's committed σ to 1e-6.

## 9. Final Status

```text
PHASE17_KP2_GATE = FAIL on two frozen verification criteria (FD 5.3e-4 > 1e-4; production-BLS comparison
                   impossible because production did not converge), both adjudicated post-hoc: FD at the KP1 oracle
                   tolerance 1.86e-5 <= 1e-4; production BLS warm-started at the KP2 solution agrees to 8e-4 sigma,
                   with its sigma offset traced to HARDEN-D25. The scientific result does not depend on either. PASS on
                   owner acceptance of the adjudication (as for D23 and CD1)

PRIMARY_CLASS = M3_VARIATION_PARTIALLY_RECOVERABLE

M3_VARIATION_RECOVERY_STATUS = PARTIAL: 15/20 variation trials recovered (8/12 in KP1-separable cells, 7/8 in
                               KP1-weak cells); consistent everywhere it converges; 0/10 false detections; misses are
                               O2 information-limited (3) and two O2 k12 estimator-convergence cases (one recovered
                               post-hoc with a longer initialisation)
M3_COMMON_LEVEL_RECOVERY_STATUS = NOT_CLAIMED (formally ~1 % and consistent under truth = model; state-degenerate per KP1)
DOPPLER_BIAS_IMPACT = MINOR for variation recovery (sigma_v x1.24; recovery identical, 4/5 vs 4/5)
NONLINEAR_ESTIMATOR_CONSISTENCY_STATUS = CONSISTENT (chi2_err 28/28, NEES9 28/28 among converged trials)

SCALAR_K_BASELINE = REPRODUCED (M0 sigma = CD1 sigma to 1e-6 relative)
D23_VALUE_PATH = PRESERVED (no production change; D23 regression in the full suite)
CD1 / KP1 PROVENANCE = CONSISTENT (KP1 anchor 8.8e-8; the KP2 finding refines, not contradicts, KP1: direction is not
                       detectability)

PRODUCTION_CODE_CHANGED = NO
PHYSICAL_MODEL_CHANGED = NO
ESTIMATOR_CHANGED = NO (analysis-layer batch solver on production models; production estimators untouched)
K_PARAMETERIZATION_CHANGED = NO (M3 as recommended by KP1)

TOTAL_TESTS_COLLECTED = 1495
TESTS_PASSED = 1466
TESTS_FAILED = 0
TESTS_ERRORS = 0
TESTS_SKIPPED = 29 (the same opt-in/data-dependent set as at entry)
(clean checkout of the KP2 tree; the shared worktree also held another session's uncommitted edit to the
 R2-byte-protected examples/run_scenario_config.py, which alone causes 1 failure + 10 errors there)

PUSH = NONE
MERGE = NONE
MAIN_CHANGED = NO
ORIGIN_MAIN_CHANGED = NO

NEXT_ACTION = hard stop. Not started (separately authorised): a stochastic-K phase, and an estimator-robustness
              improvement (Levenberg-Marquardt / longer staged start) for large a priori errors in full-Sun geometry.
```

## 10. Limitations

- **Small sample.** One noise seed per (cell, truth): 30 trials. The rates are indicative, not Monte Carlo statistics.
- **Fixed amplitude and arc.** One amplitude (10 %), one arc length (15 orbits), DSN3 only.
- **Truth equals model.** No SRP, attitude or area-model error is present. The level's formal precision does not
  carry over to real spacecraft.
- **Estimator scope.** It is an analysis-layer batch solver.
  - The production BLS did not converge on this arc from the a priori start; it agrees only when warm-started (§8.2).
  - Its posterior covariance carries HARDEN-D25.

## 11. Addendum — Phase 17-HARDEN-BLS1 (2026-10-10)

The production-BLS non-convergence in §8.2 was an estimator defect (HARDEN-D26), now repaired. Run unchanged
against the repaired estimator from the declared a priori, the frozen `job_bls` reaches this phase's solution to
≤ 6.2e-5 σ. The PRODUCTION_BLS gate's estimate criterion therefore passes as computed. Its σ criterion still fails,
and the whole failure is HARDEN-D25 (σ ratio 1.0000 once the harmonic field is included). Nothing in §1–§9 changes:
every decision quantity here comes from this phase's own solver. This section's frozen artifacts are not
regenerated. See `docs/phase17_harden_bls1_production_bls_recovery.md`.

## 12. Addendum — Phase 17-HARDEN-D25 (2026-10-11)

HARDEN-D25 is repaired: the production posterior is now linearised on the solve's own dynamics. The frozen
`job_bls`, rerun unchanged end to end with the BLS1 + D25 estimator, passes **both** PRODUCTION_BLS criteria
as computed: estimate ≤ 5.7e-5 σ, σ ratios 1.000000–1.000001. The FD gate remains as adjudicated in §8.
Nothing in §1–§9 changes. See `docs/phase17_harden_d25_posterior_covariance_consistency.md`.
