# Phase 17-HARDEN-D30 — BLS-LM acceptance noise: rescore-on-reject

- **Date:** 2026-10-21
- **Predeclaration:** `studies/phase17/configs/phase17_d30_predeclaration.json`.
  - Committed in `6b32230` before any number.
  - Amendment A1 committed in `ae7b887`, before any Q1–Q5 number was read.
- **Evidence:**
  - `studies/phase17/results/phase17_d30_qualification_amended_A1.json` (primary)
  - `studies/phase17/results/phase17_d30_qualification_original_rule.json` (pre-A1 rule)
  - `studies/phase17/scripts/phase17_d30_qualification.py`
- **Code:** `src/lunar_od/estimators.py` (`estimate_two_way_range_bls_lm`, `_predicted_decrease`)
- **Tests:** `tests/test_harden_d30_acceptance_rescore.py`

## 0. Final status

```
PHASE17_HARDEN_D30_OUTCOME   = D30_REPAIRED (predeclared rule, amendment A1; Q1 Q2 Q3 Q5 Q6 PASS)
ORIGINAL_RULE (pre-A1)       = would be D30_NOT_REPAIRED: Q2 FAIL (a priori chi2 903.17 > 902.94), Q6 FAIL (exact-optimum test)
DEFAULT                      = acceptance_rescore_tolerance_factor = 1e-2 (on); None = historical rule, bitwise (Q5)
SCOPE                        = two-way range BLS-LM only (six-state and K_SRP paths); SRIF, SR-UKF and the
                               position / range-rate LM loops unchanged
AUTHORITATIVE NUMBERS        = none superseded (KP2 uses its own solver; the frozen production job converges
                               from its declared a priori under both rules)
```

## 1. Defect

The two-way range BLS-LM scores the incumbent and the candidate with a 6-state propagation at the caller's tolerance.

- On the KP2 cross-check (simplified dynamics, rtol 1e-11) that score carries about 5 χ² of integration noise.
- D27's line search at the stall point showed the effect. The full Gauss-Newton step lowers the cost by 9.0 χ², but
  25 %, 10 % and 1 % steps *raise* it by 5.0, 3.7 and 6.4 χ².
- Once λ grows, every damped step is smaller than the noise. Each one is rejected, λ grows further, and the solve
  stalls.
- From the truth start the solve ended `DampingLimit` at χ² 916.82, 1.95 σ from the minimiser. D27 correctly
  refused to certify that point (decrement 3.68).

## 2. Repair

The repair is **rescore-on-reject**. When the caller-tolerance score rejects a candidate, both the incumbent and the
candidate are rescored before λ is raised:

- The rescore uses the same acceptance path at `rtol × f` and `atol × f`, with `f = 1e-2`.
- If the tighter score prefers the candidate, the step is accepted.
- The incumbent's tight cost is cached per incumbent state.
- `stats.acceptance_rescore_accepts` counts these acceptances.

**Amendment A1.** A rejected step is rescored only when its predicted decrease of the linearised scored objective is
at least `STATIONARITY_DECREMENT_TOL² = 0.01` χ².

- **Trigger:** the existing exact-optimum test, which starts on the minimiser. There the tight score accepted a
  7.9e-8 m move, chasing the tight integrator's own noise.
- **Why 0.01:** a move of 0.1 posterior σ changes χ² by 0.01. That is D27's certification level, so a smaller step
  cannot matter.

**Unchanged by construction:**
- Every step the historical rule accepts is accepted identically. The new branch runs only after a historical
  rejection.
- The objective, the step, the weights, the prior, the Jacobian, the damping schedule on a confirmed rejection, D27's
  status rule and the posterior are all unchanged.

## 3. Qualification (KP2 cross-check, SIMPLIFIED dynamics, rng [98, 0, 0], K solve-for)

Reference minimiser: the stored D27 a priori row, χ² 902.4409, GN remainder 0.149 σ.

| case | rule | status | χ² | GN remainder [σ] | K | rescore accepts | iterations | elapsed [s] |
|---|---|---|---:|---:|---:|---:|---:|---:|
| apriori | historical | Converged STEP_NORM | 902.4409 | 0.149 | 0.0051305 | 0 | 30 | 168 |
| apriori | **A1** | Converged STEP_NORM | **902.4206** | **0.072** | 0.0049800 | 1 | 33 | 186 |
| truth_start | historical | DampingLimit | 916.8221 | 1.946 | 0.0077595 | 0 | 18 | 101 |
| truth_start | **A1** | **Converged STATIONARY_AT_DAMPING_LIMIT** (δ 0.0015) | **902.4554** | **0.056** | 0.0049802 | 4 | 33 | 191 |
| floor | historical | DampingLimit (δ 0.159) | 902.4409 | 0.149 | 0.0051305 | 0 | 35 | 197 |
| floor | **A1** | **Converged STATIONARY_AT_DAMPING_LIMIT** (δ 0.0056) | 902.4206 | 0.072 | 0.0049800 | 1 | 38 | 209 |
| bad_jacobian | historical | DampingLimit (δ 2511) | 6305552 | 604.6 | 1.5424 | 0 | 31 | 188 |
| bad_jacobian | **A1** | DampingLimit (δ 2510) | 6302894 | 607.1 | 1.5488 | 3 | 47 | 260 |

| gate | rule | A1 | original rule |
|---|---|---|---|
| Q1 truth start reaches the minimiser | χ² ≤ 903.44 and GN ≤ 0.25 σ | **PASS** | PASS (903.01, 0.095 σ, but `MaxIter` after 80 iterations) |
| Q2 a priori not degraded | bitwise, or χ² ≤ 902.94 and GN ≤ 0.25 σ | **PASS** | **FAIL** (903.17) |
| Q3 genuine failure kept | `DampingLimit`, δ ≥ 100 | **PASS** | PASS |
| Q4 floor (reported) | — | `Converged`, δ 0.0056 (was uncertified at 0.159) | `MaxIter`, χ² 905.03 |
| Q5 switch off is historical | 4 stored D27 rows bitwise | **PASS** | PASS |
| Q6 full suite | no new failure | **PASS** (1842 passed, 29 skipped, 0 failed) | FAIL (exact-optimum test) |
| Q7 runtime (reported) | — | truth start 101 → 191 s; others +5 % to +39 % | +70 % to +360 % |

**Reading:**
- With A1 the truth start now reaches the same minimiser as the a priori start: χ² within 0.035, K within 2e-7.
  Both starts are certified stationary.
- The a priori solution itself also improved, from 0.149 σ to 0.072 σ GN remainder. D27 report §4 had recorded the
  historical rule's stop there as "at the noise floor". This repair shows it was noise in the acceptance score, not
  in the objective.
- The **original rule** kept accepting noise-level moves (17–21 rescues). It wandered to χ² 903–905 and stopped at
  `MaxIter`. The A1 gate on predicted gain is what makes the rule work.
- `bad_jacobian` stays a certified non-convergence; its 3 rescues are large genuine steps.

## 4. Effect on earlier results

- **K estimate on this problem:** the minimiser's K is 0.004980. The historical rule's a priori stop gave 0.005131.
  The whole remaining GN step at that stop was 0.149 σ in its largest component, so the K difference is below that.
- **Authoritative numbers:** none change.
  - KP2 classifications use the KP2 solver, not this production loop.
  - The BLS1 frozen production job converged from its declared a priori under the historical rule. It is not re-run
    here.
- **Earlier claim superseded:** D27 report §4 attributed the uncertified δ 0.159 floor stop to the objective's noise
  floor. It was acceptance noise. With A1 the same solve certifies at δ 0.0056.

## 5. Not claimed

- Any behaviour of SRIF, SR-UKF or the position / range-rate LM loops. They have the same acceptance structure but
  were not changed or tested here.
- High-fidelity (GL1800F) runtime. The tight rescore costs one extra 6-state propagation per rejected step, plus one
  per incumbent.
- That f = 1e-2 is optimal. It is the predeclared value.
