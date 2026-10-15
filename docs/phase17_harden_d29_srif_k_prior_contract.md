# Phase 17-HARDEN-D29 — SRIF K-path hidden state-prior contract

- **Date:** 2026-10-12
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`), on top of D28 `b5d59b8`
- **Predeclaration:** `artifacts/phase17_d29_predeclaration.json`, written after the provenance audit and
  before any production change or measured estimate change
- **Evidence:** `artifacts/phase17_d29_qualification.json`, `artifacts/phase17_d29_run_{old,new}.json`,
  `examples/phase17_d29_qualification.py`, `tests/test_harden_d29_srif_k_prior_contract.py`

## 0. Final status

```
PHASE17_HARDEN_D29_GATE                    = FAIL as computed -> ADJUDICATED -> PASS (owner accepted 2026-10-15)
                                             G1 G2 G3 G4 G5 G7 PASS; G6 (BLS/SRIF equivalence) FAIL on its
                                             posterior sub-criterion: 1.44e-6 vs 1e-6, wholly the covariance's
                                             dependence on the linearisation point (two solutions 4.3e-4 sigma
                                             apart); each posterior equals the declared problem's covariance at
                                             its own point to 2e-15 / 6e-15 (§3)
SRIF_K_NO_EXPLICIT_STATE_PRIOR_SEMANTICS   = data-only state + the REQUIRED explicit K prior:
                                             J = r^T W r + (K - K0)^2 / sigma_K^2   (repaired)
SRIF_K_EXPLICIT_STATE_PRIOR_SEMANTICS      = MAP with the supplied state prior + the K prior (unchanged, bitwise)
SRIF_K_EXPLICIT_K_PRIOR_SEMANTICS          = required (s31); in objective and posterior (unchanged)
DEFAULT_STATE_DIAGONAL_ROLE                = numerical step regularizer only (r_bar rows, zero right-hand
                                             side) in every path, the SRIF K path included
SRIF_K_ESTIMATE_STATUS                     = CHANGED only without an explicit state prior, by <= 2.1e-4 sigma
                                             (3 cm) on the test arc; requalified (stationary for J to 3.7e-5 sigma;
                                             start-independent to 1.3e-5 sigma)
SRIF_K_POSTERIOR_STATUS                    = QUALIFIED: = (H^T W H + K-prior row)^-1 at the solution to 6e-15
BLS_SRIF_EQUIVALENCE_STATUS                = SAME DECLARED PROBLEM: estimates agree to 4.3e-4 sigma (was a
                                             different problem: posteriors 1.3e-4 apart, now 1.4e-6 / 3.8e-7,
                                             linearisation-point only)
D28_CONTRACT_STATUS                        = PRESERVED and completed (its one exception, the SRIF K path, is gone)
D25_POSTERIOR_CONSISTENCY_STATUS           = PRESERVED (D25 tests pass)
KP2_D29_MATERIALITY                        = NONE: KP2 never calls the SRIF K path; no authoritative result runs
                                             SRIF K without an explicit state prior
```

No change here bears on K-level identifiability. The common K level is not claimed.

## 1. The question and the provenance

`estimate_two_way_range_srif(solve_for_k_srp=True)` built its prior right-hand side as

```
z_bar = prior_sqrt_info @ (x_prior_full - x_current_full)   if (has_explicit_prior or solve_for_k_srp)
```

with the full 7×7 prior square root. With no explicit state prior, its state block is the 15 km / 5 m/s diagonal,
so the SRIF K estimate was pulled toward its starting state. The six-state SRIF and both BLS paths never do this.

| evidence | what it says |
|---|---|
| SRIF docstring | "same three independent switches as `estimate_two_way_range_bls_lm` (s13), **and the same contract**" |
| BLS K path | state prior in the objective only if explicit; the default diagonal is a step regularizer (D28) |
| s31 rationale | K prior required because "an UNREGULARIZED **K column** here would frequently leave `r_hat` singular": numerical, K-specific, silent about the state |
| Phase 17-R report (external, read-only) | runs without an explicit state prior are "no state prior" / "the data-only regime" |
| six-state SRIF | prior right-hand side only for an explicit prior (D28 contract, MATLAB `P_bar_inv` heritage) |
| origin | `b8da5e6` (2026-09-14 transplant): the clause that switches on the required K row multiplies the *full* prior square root, so it also switches on the default state rows. No test or document asserts that state pull |

One authoritative contract, the BLS one, exists, and the SRIF K behaviour departs from it by an implementation
accident. There were not two competing intents, so no stop was needed.

## 2. Repair (`lunar_od/estimators.py`, SRIF only)

- `prior_rhs_sqrt`: a copy of the prior square root whose state block is zeroed when K is solved for without an
  explicit state prior. `z_bar` uses it. `r_bar`, the step regularizer, keeps the full rows, so the numerical
  safeguard that s31 relies on is untouched.
- The SRIF K-path posterior carries the explicit state prior (if any) and the K prior, never the default
  diagonal. This is the rule of the BLS K path (D28) and reuses `_posterior_prior_terms`.

Unchanged: the K prior, explicit-prior behaviour, the six-state SRIF, the acceptance cost, the negative-K
backtracking, dynamics and measurement models, D25, D28 and BLS1.

## 3. Qualification (3.3 h single-station fixture, SRP-only dynamics, seeded 5 m noise)

| gate | result | value |
|---|---|---|
| G1 explicit state prior unchanged | **PASS** | SRIF K, BLS K and SRIF six-state with P0, two starts: estimate, posterior, iterations bitwise vs `b5d59b8` |
| G2 six-state unchanged | **PASS** | SRIF six-state, with and without P0: bitwise |
| G3 no state pull | **PASS** | two starts (offset 30 m / 0.03 m/s): solutions agree to 1.3e-5 σ (pre-D29 1.0e-4 σ) |
| G4 stationarity | **PASS** | independent Gauss–Newton step of J (data + K prior) at the solution: 3.7e-5 σ |
| G5 posterior rebuild | **PASS** | SRIF K posterior vs independent (H^T W H + K row)⁻¹: 5.8e-15 in σ |
| G6 BLS/SRIF equivalence | estimates **PASS**, posteriors **FAIL as computed** | estimates 4.3e-4 σ (≤ 2e-2); posterior σ 1.44e-6 (> 1e-6) |
| G7 regression and neighbours | **PASS** | D29 (5) + D28 + D25 + BLS1 + `test_k_srp_*` (incl. Phase 17-R's BLS/SRIF common-posterior test): 58 passed; full suite §5 |

**G6 adjudication** (post-hoc, labelled; the predeclared result is unchanged):

| comparison | max σ difference |
|---|---|
| BLS posterior vs declared covariance at the BLS point | 2.3e-15 |
| SRIF posterior vs declared covariance at the SRIF point | 5.8e-15 |
| declared covariance at the BLS point vs at the SRIF point | 1.44e-6 |

Both estimators solve the same problem exactly. The 1.44e-6 is the covariance's dependence on where it is
linearised: the two solutions are 4.3e-4 σ apart because SRIF ends at its stationarity floor (MaxIter,
3.7e-5 σ) while BLS stops on J-Stab. From the offset start the gap is 3.8e-7, inside the criterion. For contrast,
before D29 the two posteriors differed by 1.3e-4 because the problems differed. The 1e-6 criterion was set
without allowing for the linearisation-point term; it is not relaxed.

### Making the historical defect testable

The real diagonal is weak, so pre-D29 its pull is only 1e-4 σ on this arc, too small for a robust test.
Strengthened 10⁴× inside a test (150 m, 5 cm/s), it separates the contracts 400× (labelled diagnostic):

| code | start dependence of the SRIF K solution |
|---|---|
| D29 (step regularizer only) | 6.2e-4 σ |
| pre-D29 (used as a state prior) | 0.27 σ |

`test_a_strengthened_step_regularizer_still_does_not_pull_the_estimate` pins this.

## 4. Materiality

| consumer | effect |
|---|---|
| KP2 | none: KP2 and its production cross-check use the KP2 solver and the BLS, never the SRIF K path |
| Phase 17-R final synthesis (SRIF K, case C) | none: it supplies an explicit `PRIOR_COV` (G1 bitwise) |
| scenarios / desktop / campaigns | none call the SRIF K path |
| test arc, no state prior | SRIF K estimate change ≤ 2.1e-4 σ (3.2 cm, ΔK 8e-8); posterior σ change ≤ 1.3e-4 |

No authoritative number is superseded.

## 5. Verification

- Focused: 58 passed (D29 5, D28 6, D25 4, BLS1 5, `test_k_srp_*`).
- Full suite on a clean detached worktree of `b5d59b8` plus the D29 files (`LUNAR_OD_PROJECT_ROOT` = workspace root): **1486 passed, 29 skipped, 0 failed** (D28 1481 + 5).

## 6. Not done

D27, stochastic K, new science. The SRIF acceptance cost is data-only in every mode, including with explicit
priors. That pre-existing behaviour is not part of D29's prior contract and is left as is.
