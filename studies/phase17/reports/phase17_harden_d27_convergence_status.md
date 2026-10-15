# Phase 17-HARDEN-D27 — BLS-LM convergence / status contract at damping exhaustion

- **Date:** 2026-10-12
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`), on top of D29 `e35e245`
- **Code:** `lunar_od/estimators.py` (`_gauss_newton_decrement`, `_damping_exhaustion_status`,
  `STATIONARITY_DECREMENT_TOL`, `EstimatorStats.gauss_newton_decrement`), `lunar_od/diagnostics.py`
  (`converged_by_stationarity`, category `converged_stationary`)
- **Evidence:** `artifacts/phase17_d27_qualification.json`, `examples/phase17_d27_qualification.py`
- **Tests:** `tests/test_harden_d27_convergence_status.py` (11), `tests/test_estimator_lm_consistency.py` (2 updated)

## 0. Final status

```
PHASE17_HARDEN_D27_GATE        = PASS (qualified; one conservative limitation and one new open item, s4)
CONVERGENCE_STATUS_CONTRACT    = three outcomes, decided by stationarity evidence, never by lambda:
                                   Converged / STEP_NORM                     accepted step below tol_step_norm (unchanged)
                                   Converged / STATIONARY_AT_DAMPING_LIMIT   lambda > 1e12 AND Gauss-Newton decrement <= 0.1
                                   DampingLimit / LAMBDA_HARD_LIMIT          lambda > 1e12 AND decrement > 0.1 (or not finite)
DAMPING_LIMIT_SEMANTICS        = DampingLimit now means "damping exhausted at a point NOT certified stationary":
                                 a genuine non-convergence, or a noise floor coarser than 0.1 posterior sigma
STATIONARY_SOLUTION_STATUS     = certified: frozen high-fidelity KP2 job (decrement 1.9e-3); exact-optimum fixtures
                                 (two-way range, position BLS)
GENUINE_FAILURE_STATUS         = preserved: corrupted K Jacobian (decrement 2511), forced bad steps (unit tests),
                                 truth-start stall (decrement 3.68; s4, HARDEN-D30)
BLS1_REQUALIFICATION_STATUS    = PASS: a-priori-start estimate bitwise identical to BLS1 (K = 0.00513051097781201);
                                 the BLS1 high-fidelity production solution would now be reported
                                 Converged / STATIONARY_AT_DAMPING_LIMIT
D25 / D28 / D29                = PRESERVED (their regression files pass; posterior code untouched)
```

Estimates are never changed by this rule; only the label and the reported decrement are. The common K level is not
claimed.

## 1. Diagnosis

`DampingLimit` was set whenever λ passed 1e12, i.e. whenever ~14 consecutive trial steps were rejected. Rejection
compares two integrated costs. Near a minimiser the predicted gain of any step falls below the cost's integration
noise, so rejection is *expected* there, and λ exhaustion carries no information about whether the point is a
minimiser. The step-norm and J-Stab tests run only on *accepted* steps, so a solve sitting on its minimiser could
never report convergence. That is a convergence-contract defect: a status was assigned from the damping state rather
than from evidence about the objective.

## 2. Repair

At λ > 1e12, the estimator evaluates the **undamped Gauss-Newton decrement** of the objective it scores, at the
incumbent (the returned) point:

```
delta^2 = || [A; U] d_GN ||^2 = g^T N^+ g,   A = sqrt(W) H,  U^T U = explicit prior information
```

δ² is the χ² reduction the full Gauss-Newton step predicts (the squared Newton decrement, Boyd & Vandenberghe 2004
§9.5.1). δ is the length of the remaining correction in the posterior metric, so every parameter is within δ
posterior σ of the linearised minimiser. It is unit-free (scale-invariant, tested) and needs no damping. It is
computed by column-equilibrated least squares, so an unobservable direction contributes nothing rather than failing.

- **Threshold.** `STATIONARITY_DECREMENT_TOL = 0.1` posterior σ (the whole remaining correction ≤ 0.1 σ), declared in
  code before any case was scored. It does not depend on λ and does not weaken the step-norm or cost tests.
- **Objective.** The prior rows are exactly the scored objective's: the explicit state prior and the explicit K prior
  in the two-way path (`prior_active`), and the explicit prior in the position / range-rate paths. The
  default 15 km / 5 m/s step regularizer is never included (D28/D29 contract).
- **Linearisation.** It is made at the caller's tolerance. The six-state paths re-linearise at `rtol/atol` when
  `_adaptive_tol` had loosened the last pass; the K path already uses the caller's tolerance (BLS1 D-c). A loose
  Jacobian amplifies the gradient error along weak directions by ~1/s_min.
- **Scope.** The rule is general: position, range-rate and two-way-range BLS-LM, six-state and K paths. The
  negative-K damping branch (`LAMBDA_HARD_LIMIT_NEGATIVE_K`, an active bound) is not certified, since a bound-active
  point would need a projected test. SRIF has no damping and is unaffected.
- **Diagnostics.** `analyze_convergence` reports `converged_by_stationarity` and category `converged_stationary`.
  Scenario `algorithmic_success` therefore counts a certified stationary point as converged. `operational_success`
  never depended on the label.

## 3. Qualification (`examples/phase17_d27_qualification.py`)

KP2 cross-check problem (CANON k0, range only, scalar K, no prior, the frozen job's a priori and noise draws).

| case | status / detail | production δ | independent δ | GN step max / σ | note |
|---|---|---:|---:|---:|---|
| simplified, declared a priori | Converged / STEP_NORM | — | 0.159 | 0.149 | K **bitwise** = BLS1 `final` row |
| simplified, step/J-Stab tests off ("floor") | DampingLimit / LAMBDA_HARD_LIMIT | 0.159 | 0.159 | 0.149 | same point; δ above 0.1 (s4) |
| simplified, truth start | DampingLimit / LAMBDA_HARD_LIMIT | 3.68 | 3.68 | 1.95 | genuine stall (s4, D30) |
| simplified, K column sign-flipped | DampingLimit / LAMBDA_HARD_LIMIT | 2511 | 2511 | 605 | genuine failure, 164 m off |
| **high fidelity, stored BLS1 production solution** | was DampingLimit → **Converged / STATIONARY_AT_DAMPING_LIMIT** | **1.9e-3** | — | ≤ 4.1e-3 (BLS1) | data rebuilt exactly (a priori matches) |
| high fidelity, declared a priori | (not a termination point) | 1.1e4 | — | — | contrast |

The production decrement equals the independently assembled one to 1e-12. The high-fidelity case is the D27 defect
itself: the same solution, previously labelled a failure, is now certified with two orders of margin.

Unit and end-to-end tests (`tests/test_harden_d27_convergence_status.py`) cover the following:

- **Decrement algebra.** δ equals √(gᵀN⁻¹g) with and without a prior, is scale-invariant, is 0 at a zero gradient,
  is finite for a rank-deficient design, and is never certified for a non-finite design. The threshold partitions the
  status at 0.99 / 1.01 × tol.
- **Position BLS from 150 m.** With the step tests disabled, the solve descends to its noise floor and is certified
  (δ ≈ 1e-5). The certificate re-linearised at the caller's tolerance: the last propagation is at rtol 1e-12, while
  the loop was still loosened.
- **Forced bad steps from 150 m.** The status stays DampingLimit with δ > 100 and the estimate untouched.
- **Ordinary convergence.** It keeps STEP_NORM.
- **Classification.** The three diagnostics outcomes are distinct.

In `tests/test_estimator_lm_consistency.py`, two tests encoded D27 and are updated with their reasons in the
docstrings:

- The exact-optimum test now expects the stationary certificate.
- The forced-bad-step test now starts 25 m off the optimum, so it tests a genuine failure. Started at the truth, it
  was testing the minimiser.

## 4. Limitation and new observation

- **Conservative at a coarse noise floor.** At rtol 1e-11 in simplified dynamics the "floor" case stops at δ = 0.159.
  BLS1 showed that this remainder is at the objective's noise floor: the GN step raises the scored cost. Under the
  declared 0.1 rule it stays `DampingLimit`. The threshold is **not** relaxed post hoc. Users get δ in
  `stats.gauss_newton_decrement` and can judge it. At the high-fidelity tolerance (1e-12) the floor is far below
  the rule.
- **HARDEN-D30 (new, OPEN_RECOMMENDED_FOLLOWUP).** From the truth start the K solve stalls 1.95 σ from the minimiser.
  A line search at the stall point locates the cause: the full GN step lowers the scored cost by 9.0 χ² (half step
  10.6), but steps of 25 %, 10 % and 1 % *raise* it by 5.0, 3.7 and 6.4 χ². The acceptance cost (6-state
  propagation at rtol 1e-11) carries ~5 χ² of integration noise, so damped steps below that noise are rejected and λ
  escalates. D27 correctly refuses to certify this point (δ 3.68). No authoritative result is affected: KP2 uses its
  own solver, and the frozen production job converges from its declared a priori. A noise-aware acceptance rule is an
  estimator-design change outside this closeout.

## 5. Verification

- Focused: `test_harden_d27_convergence_status`, `test_estimator_lm_consistency`, `test_estimators`,
  `test_harden_bls1_k_srp_recovery`, `test_harden_d25/d28/d29`, `test_k_srp_bls_solve_for`, `test_scenarios`: pass.
- Full suite on a clean detached worktree of `fd24147`: 1497 passed, 29 skipped, 0 failed (`artifacts/phase17_closeout_verification.txt`).
