# Phase 17-HARDEN-BLS1 — Production BLS nonlinear recovery hardening

- **Date:** 2026-10-10
- **Branch:** `feature/phase17-r-k-srp-estimation` (worktree `python_port_phase17r0`), base `46e21c8` (KP2)
- **Scope:** `estimate_two_way_range_bls_lm(solve_for_k_srp=True)` only. The six-state path, SRIF, SR-UKF, the
  measurement models, the dynamics and every KP2 assumption are unchanged.

## 0. Final status

```
BLS_ROOT_CAUSE_STATUS              = IDENTIFIED_AND_REPAIRED
                                     three defects in the K solve-for LM loop (scaling/numerical,
                                     convergence control, Jacobian-tolerance contract), each shown
                                     necessary by ablation; not an identifiability limit
PRODUCTION_BLS_M3_RECOVERY_STATUS  = QUALIFIED  (Outcome A) for the problem production BLS solves:
                                     scalar K, range only. From the declared a priori, with no seeding,
                                     the frozen KP2 job reaches the KP2 solution to <= 6e-5 sigma
                                     (was 347 sigma); independent stationarity <= 4.1e-3 sigma.
                                     Production has no M3 / counted-Doppler / per-pass-bias capability.
                                     Its posterior sigma is not qualified until D25 is fixed, and its
                                     status label is not a convergence indicator (D27).
KP2_VARIATION_RESULT_STATUS        = UNCHANGED: M3_VARIATION_PARTIALLY_RECOVERABLE (the analysis-layer
                                     solver and every KP2 decision input are untouched)
D25_KP2_MATERIALITY                = NOT_MATERIAL to every KP2 classification; MATERIAL ONLY to the
                                     PRODUCTION_BLS gate's sigma criterion, which it explains completely
                                     (sigma ratio 0.9171 -> 1.0000 with the solve's harmonic field)
KP2_GATE_STATUS                    = FAIL as computed, adjudicated; owner accepted 2026-10-15 (gate PASS).
                                     FD: oracle floor (unchanged).
                                     PRODUCTION_BLS: estimate criterion now PASS (<= 6e-5 sigma vs 0.02);
                                     sigma criterion FAIL (K 0.917, z 0.972), wholly HARDEN-D25
                                     (repaired since: D25 end-to-end rerun passes both criteria).
```

The common K level is not claimed.

## 1. Question

KP2's frozen production-BLS verification job failed: from the declared a priori (CANON k0, high fidelity, T0,
range only, scalar K, no prior; `rng([98, 0, 0])`) production BLS ended `DampingLimit` with K driven to
2.9e-7 and the state up to 347 σ from the KP2 least-squares solution. Started *at* that solution it stayed
there to 8e-4 σ. So the least-squares solution exists and production accepts it; production cannot reach it.
BLS1 asks whether that is an implementation / scaling / numerical / convergence-control / contract defect
(repair and qualify) or a genuine limitation (confirm).

Production BLS has no M3, counted-Doppler or per-pass-bias capability. "Production BLS M3 recovery" is
therefore answered on the problem production does solve, and the one the frozen gate tested: the scalar-K,
range-only sub-problem of the KP2 cross-check.

## 2. Diagnosis

All diagnosis ran first in simplified dynamics on the same problem (same a priori, same noise draw), where the
failure reproduces in ~2 minutes: the old code stalled at χ² = 1.28e6 (184 m RMS) against an optimum of 902.4
(truth 948), with K walked to ~0 and a false `Converged` label (step norm < 1e-8 under λ ≈ 1e8).

What was ruled out, by measurement:

| hypothesis | test | result |
|---|---|---|
| inconsistent K Jacobian | FD of the acceptance residual vs `_two_way_range_k_srp_column` at the stall point | K column 0.9 %, state columns ≤ 3e-3: correct |
| data / model mismatch | cost at truth with the estimator's own propagation | χ² 948 for 960 rows: consistent |
| strong nonlinearity | undamped Gauss–Newton step from the stall point | one step to χ² 904 (optimum 902.4), K = 0.0050 |
| genuine identifiability limit | equilibrated design spectrum | cond(H) ≈ 1.3e4 (normal 1.6e8): well posed |

The problem is nearly linear and well posed; the LM loop could not take the step. Three defects in the K
solve-for path, each found by measurement:

**D-a. Normal matrix formed in bookkeeping units (scaling / numerical).** `_lm_step` forms Aᵀ A + λ D with the
columns in the estimator's bookkeeping units (1e6 m, 1e3 m/s, `k_srp_scale` 1e-2). Measured condition
8.5e16 against 1.6e8 for the same matrix column-equilibrated. Its `cond > 1e15` guard therefore called every
step with λ ≲ 1 singular, and the retry loop raised λ to ~1. The weak state/K direction has normalised
eigenvalue 4.3e-8 and needs λ ≲ 1e-8, so the correct step direction was unreachable. This is the R1M/R1COV
squaring defect, previously removed from the covariance but still present in the step, and it violates the
s19/s20 contract that `k_srp_scale` is numerical only.

**D-b. Negative-K trial handling (convergence control).** A trial with K < 0 discarded the whole step,
restored the incumbent and raised λ ×10. On this coupled problem a larger λ makes the step's K component *more*
negative (λ = 1e-2: dK = −0.57), so the orbit correction was thrown away on every iteration while accepted
steps walked K to the bound.

**D-c. Loosened Jacobian tolerance (contract).** `_adaptive_tol` integrates the augmented pass at rtol 1e-8
for the first 40 % of iterations. The Gauss–Newton fixed point is H̃ᵀ W r = 0 with the H actually used, and
the Jacobian error is amplified ~1/s_min along the weak direction. With D-a and D-b repaired, every step from
χ² 1405 was rejected up to the damping guard and the solve reported a false `Converged`.

## 3. Repair (lunar_od/estimators.py, K solve-for path only)

1. `_lm_step_square_root`: the **same** damped system (same D = diag(AᵀA), same λ, same prior, same schedule)
   written as the stacked least-squares problem [A; √(λD); U] d = [b; 0; U d̄] and solved by QR after column
   equilibration. Its normal equations are exactly `_lm_step`'s. The singularity guard keeps its threshold and
   meaning (cond of the normal matrix > 1e15), now evaluated as cond(R)² in the equilibrated basis, so it no
   longer depends on units.
2. Bound-active step: a trial with K < 0 is still never evaluated and never clipped (s18). Instead of being
   discarded, it is replaced by the damped step in the feasible face, the same normal equations with K held
   at its current value. K is re-freed on the next iteration. The objective and the constraint are unchanged.
   Counted in `EstimatorStats.k_srp_bound_active_steps`.
3. The K path's augmented pass runs at the caller's `rtol`/`atol` in every iteration.

The six-state path keeps `_lm_step`, `_adaptive_tol` and the discard path bit for bit
(`test_default_six_state_bls_is_bitwise_unaffected_by_new_kwargs` passes). Not changed: convergence
criteria, λ0 and its schedule, the 1e12 guard, priors, tolerances, SRP model, D25.

A fourth candidate change (taking the step's right-hand side from the acceptance residual) was tried and
**removed**. Its apparent necessity came from scoring with the augmented propagator, which differs from the
acceptance propagator by Δχ² ≈ 4 at the same point. Scored correctly, the solve converges without it.

## 4. Ablation (simplified dynamics, KP2 cross-check problem, same draws)

`examples/phase17_bls1_qualification.py --ablation` → `artifacts/phase17_bls1_qualification.json`
(`simplified_ablation`). Variants are source-level copies exec'd into a private namespace; the production
module is untouched.

| variant | status | χ² | independent GN remainder (max \|step\|/σ) | position error | K |
|---|---|---|---|---|---|
| **final** | Converged | **902.44** | 0.15 | 6.5 m | 0.00513 |
| without square-root step | MaxIter | 1.28e6 | 350 | 157 m | 0.0012 |
| without bound-active step | DampingLimit | 1.30e6 | 351 | 158 m | 5.7e-5 |
| without caller-tolerance Jacobian | Converged (false) | 1405 | 5.3 | 9.3 m | 0.0041 |

Each change is necessary. The final 0.15 σ remainder sits at this case's objective noise floor (rtol 1e-11):
taking that Gauss–Newton step *raises* the scored cost (902.46 → 902.68).

## 5. High-fidelity qualification (the frozen KP2 job, unchanged)

`examples/phase17_bls1_qualification.py --hf` runs `phase17_kp2_campaign.job_bls` unmodified. The job uses
CANON k0, high fidelity (GL1800F field, Earth/Sun third body, conical-penumbra SRP), rtol 1e-12, T0, range
only, scalar K and no prior. It starts from the declared a priori x0 + dx (dx = −109, −124, +56 m;
+0.069, +0.043, +0.107 m/s), K0 = 0.01. The job also runs the KP2 solver on the same data and compares.
Production is never started from the KP2 solution.

| quantity | KP2 (frozen, old code) | BLS1 (repaired) | frozen criterion |
|---|---|---|---|
| production status | DampingLimit | DampingLimit (`LAMBDA_HARD_LIMIT`, after convergence, see §8) | — |
| K: production / KP2 solver | 2.9e-7 / 0.0054110 | **0.0054110** / 0.0054110 | — |
| max \|estimate difference\| / σ | 347 | **6.2e-5** | ≤ 0.02 → **PASS** |
| σ ratio (KP2 / production) | 0.917 … 1.006 | 0.917 (K), 0.972 (z), others ≤ 0.6 % | within 1 % → **FAIL** |
| state error vs truth (MCI) | — | \|Δr\| 6.4 m (−0.08, −5.05, −3.94 m); \|Δv\| 7.0 mm/s | — |
| iterations / negative-K trials / bound-active steps | — | 40 / 5 / 5 | — |
| production wall time | ≈ 2.3 h (estimated: job total 10 976 s minus ~45 min KP2 solver) | 2.6 h (9334 s) | — |

**Diagnostics** at the production solution (labelled, not part of the frozen decision):

- **Stationarity.** The undamped Gauss–Newton remainder, harmonic field included and computed by column-normalised
  lstsq independent of the LM code, is at most **4.1e-3 σ** (χ² 902.98). Production stopped at the minimiser;
  the `DampingLimit` label is D27, not a failure to converge.
- **D25 isolation.** The posterior σ was recomputed at the same point with the solve's own harmonic field. The KP2 / that σ
  ratio is **1.0000 for all seven parameters** (to 5 digits). The whole σ-criterion failure is HARDEN-D25:
  `_two_way_range_posterior_design_with_k` linearises the covariance without the harmonic field.

The repaired path is roughly 10–15 % slower in high fidelity (against the estimated old time) because the Jacobian is integrated at the caller's
tolerance from the first iteration (D-c). That is the price of a correct fixed point.

## 6. Regression protection

`tests/test_harden_bls1_k_srp_recovery.py` (5 tests, ~95 s):

- the square-root step equals `_lm_step` on well-conditioned systems, with and without a prior (data-only K row);
- on the production structure (weak K column, production units) the old solve reports singular at
  cond 3.4e15, while the square-root solve does not (cond 7e7), is unit-independent to 1e-10 and tends to
  Gauss–Newton as λ → 0;
- end to end, from the KP2 a-priori offset on a 20 000 s fixture with 5 m noise: `Converged`, independent GN
  remainder ≤ 1e-2 σ, same solution as a solve started from the truth within 2e-2 σ (HEAD: MaxIter, 123 m,
  58 σ remainder);
- truth K = 0, so the optimum lies on the bound: the bound-active step is used, K → (0, 1e-6) never clipped,
  position < 1 cm (HEAD: 122 m).

## 7. HARDEN-D25 materiality to KP2

**None for the KP2 classifications.** Every KP2 decision quantity (recovery, χ²_var, NEES, false detections,
level z, σ_v) uses the covariance of the KP2 solver's own QR (`examples/phase17_kp2_core.py` `_gauss_newton`,
lines 247–254, `sol["cov"]`). The production posterior is read in exactly two places, `job_bls` and
`posthoc_bls_stationarity` (`examples/phase17_kp2_campaign.py` lines 171 and 240), and only for the
PRODUCTION_BLS verification gate's σ criterion. D25 is **material to that criterion and fully explains its
failure**: recomputed with the harmonic field, the σ ratio is 1.0000 (§5). It is not repaired here. That would be a separate, scoped change with its own posterior re-qualification.

## 8. New observation (ledger HARDEN-D27, open)

At a verified stationary point, the solve can end `DampingLimit` rather than `Converged`. Started at the
optimum, no trial step beats the objective's integration noise, so λ climbs to the guard. Measured on the
regression fixture: from the truth start the solution is identical to the converged one (0 σ) but labelled
`DampingLimit`. The label is not a reliable convergence indicator near the noise floor; qualification here
uses an independent stationarity check. Not changed: a new acceptance rule would be a convergence-criterion
change.

## 9. Verification

- Focused: `test_k_srp_*`, `test_estimators`, `test_estimator_lm_consistency`,
  `test_two_way_range_integration`: 97 passed; `test_harden_mutants` + `test_harden_boundaries`: 48 passed.
- Full suite on a clean detached worktree of `46e21c8` plus only the BLS1 files
  (`LUNAR_OD_PROJECT_ROOT` = workspace root): **1471 passed, 29 skipped, 0 failed**. This is KP2's 1466 + 5 new.
  The shared worktree also carries another session's uncommitted desktop-app and `examples/run_scenario_config.py`
  edits, which are not part of this phase.

## 10. Not done (out of scope)

Stochastic K, Gauss–Markov K, new Monte Carlo science, the D25 repair, the D27 label, and production M3 /
counted-Doppler / per-pass-bias capability. The common K level is not claimed.
