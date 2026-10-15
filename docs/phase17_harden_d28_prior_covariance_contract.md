# Phase 17-HARDEN-D28 — Implicit prior / posterior covariance contract

- **Date:** 2026-10-12
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`), on top of D25 `3959f7a`
- **Predeclaration:** `artifacts/phase17_d28_predeclaration.json`, written after the source/heritage audit and
  before any production change or materiality number
- **Evidence:** `artifacts/phase17_d28_qualification.json` (+ `phase17_d28_invariance_{old,new}.json`,
  `phase17_d28_scenarios_{old,new}.json`), `examples/phase17_d28_qualification.py`,
  `tests/test_harden_d28_posterior_prior_contract.py`

## 0. Final status

```
PHASE17_HARDEN_D28_GATE                  = PASS (G1-G6, G8 pass; G7 quantified)
NO_EXPLICIT_PRIOR_OBJECTIVE_SEMANTICS    = DATA-ONLY weighted least squares (unchanged; now documented)
NO_EXPLICIT_PRIOR_COVARIANCE_SEMANTICS   = DATA-ONLY posterior H^T W H (repaired; was + hidden 15 km / 5 m/s)
EXPLICIT_PRIOR_SEMANTICS                 = MAP objective and posterior carry the supplied prior (unchanged, bitwise)
DEFAULT_PRIOR_STATUS                     = NOT A PRIOR: numerical step regularizer only (MATLAB P_bar_inv
                                           heritage); never enters the posterior. Exception by design: the SRIF
                                           K path's objective pulls toward it, so its posterior keeps it
D25_POSTERIOR_CONSISTENCY_STATUS         = QUALIFIED (owner-adjudicated PASS 2026-10-12; D25 tests pass with
                                           the K-path rebuild moved to the data-only contract)
BLS1_ESTIMATE_STATUS                     = QUALIFIED (estimates bitwise unchanged; BLS1 tests pass)
KP2_D28_MATERIALITY                      = NOT MATERIAL: sigma change <= 1.5e-6; production vs KP2 solver now
                                           agrees to 5e-8 (was 1.5e-6, which was D28 itself); KP2 headline
                                           classifications unchanged
```

The common K level is not claimed.

## 1. The question and the evidence

With no explicit prior, `_prior_information_and_scale` supplies a 15 km / 5 m/s state diagonal (plus the bias
configuration's diagonal when biases are solved for). Two semantics were possible: an intentional default prior,
or something that is not a prior at all. The repository answers it:

| evidence | what it says |
|---|---|
| MATLAB origin (`graduation_old_version/last_versions/utils/bls_estimator.m`) | `P_bar_inv = diag([1/15000² ×3, 1/5² ×3])`, commented "A priori covariance inverse (**Regularization**) … Tuning for Lunar Orbit". It is used only in the LM step `(AtWA + λ diag(AtWA) + P_bar_inv) \ AtWb`, with no prior pull, and MATLAB computed no posterior covariance |
| all six Python estimators | the prior right-hand side (pull toward `x_prior`) is applied **only** `if has_explicit_prior`; with none, every iteration's right-hand side is zero, so the diagonal damps the step and cannot move the fixed point (H^T W r = 0) |
| BLS acceptance objective (Phase 6.2 `_acceptance_cost`) | prior term only when `prior_active` (explicit state prior or an explicit K prior) |
| K path | `k_srp_prior_status = "DATA_ONLY"`; "omit it for a data-only solve (s23)" |
| R1COV / R1M / FXR | call the no-explicit-prior covariances "data-only" |
| posterior helpers | add `prior_inv` (or prior √-rows) unconditionally, with no branch and no stated intent |

The fitted problem was therefore always data-only, and the default diagonal was a step regularizer. Only the
posterior treated it as a prior, so the reported covariance belonged to a regularized problem no estimator
solved. **D28 is a defect in the posterior.** It is not an intended contract, so there were no two competing
intents to resolve.

### One estimator whose objective does carry it

`estimate_two_way_range_srif` with `solve_for_k_srp=True` builds its right-hand side as
`prior_sqrt_info @ (x_prior − x)` whenever `has_explicit_prior or solve_for_k_srp`. Here the default state rows
**are** in the objective, a sequential MAP with the required K prior (s31). Its posterior already matched its
objective and is left unchanged. Whether that objective should keep the default state rows is a question about
the estimate, outside this covariance phase. It is recorded in §6.

## 2. Repair (`lunar_od/estimators.py`, posterior blocks only)

`_posterior_prior_terms(prior_inv, prior_sqrt_info, has_explicit_prior)` returns the prior terms a posterior may
carry: the explicit prior, or zero. The rule for each site is that a posterior carries exactly the prior terms
of its own objective:

| estimator / path | posterior prior term after D28 |
|---|---|
| position SRIF / BLS, range-rate SRIF / BLS | explicit prior, else none (information and √-information) |
| two-way BLS, six-state | explicit prior, else none |
| two-way BLS, K path | explicit state prior if given; explicit K prior if given; never the default diagonal |
| two-way SRIF, six-state | explicit prior, else none |
| two-way SRIF, K path | unchanged: the full prior its objective uses |

Untouched: the LM normal matrix and SRIF a-priori rows (the step regularizer), the right-hand sides, the
acceptance objective, the covariance algebra, the D25 dynamics contract and BLS1.

## 3. Qualification (gates in the predeclaration)

| gate | result | value |
|---|---|---|
| G1 estimates unchanged | **PASS** | 17 cases (8 estimator paths × no prior / explicit prior, plus K-prior-only), pre-D28 `3959f7a` vs current: estimate, status, iterations, cost, K bitwise identical |
| G2 explicit prior unchanged | **PASS** | every explicit-prior case, and both SRIF K-path cases: information and covariance bitwise identical |
| G3 no hidden prior | **PASS** | no-prior posteriors equal an independently rebuilt H^T W H: two-way BLS / SRIF six-state ≤ 1e-8 in implied σ; position SRIF / BLS bitwise; SRIF R^T R ≤ 1e-8; the historical construction is measurably different on the same fixtures |
| G4 independent covariance | **PASS** | two-way BLS K path vs an independent column-normalised QR: ≤ 1e-8 in σ, also with a K prior alone (only the K row enters). No rank-deficient data-only case occurred in any test, scenario or fixture |
| G5 D25 preserved | **PASS** | `tests/test_harden_d25_posterior_dynamics_contract.py` passes; its K-path rebuild previously mirrored the hidden prior and is now data-only |
| G6 BLS1 preserved | **PASS** | `tests/test_harden_bls1_k_srp_recovery.py`, `test_k_srp_*` pass |
| G8 regression | **PASS** | `tests/test_harden_d28_posterior_prior_contract.py` (6 tests); full suite on a clean worktree (§5) |

### G7 materiality (quantified, not gated)

| consumer | effect of D28 |
|---|---|
| KP2 CANON k0 HF K path (BLS1 solution, D25 design) | σ change ≤ 1.5e-6; KP2 solver / production = 0.99999995–1.0000000016 (was 1.0000002–1.0000015); frozen 1 % criterion PASS |
| regression fixture (3.3 h, one station, weakest arc tested) | no-prior σ change: two-way BLS six-state 2.7e-4, K path 1.4e-4, SRIF six-state 3.4e-4 |
| scenario hand-off matrix (`tests/test_scenarios.py`, 42 arcs, all estimators × cold / hot / formal / sqrt_formal) | arc-1 posterior σ ≤ 3.7e-8; arc-2 estimates move only through the corrected hand-off prior, by ≤ 3.5e-8 m (≤ 6.6e-7 σ); cold and hot chains bitwise identical |
| scenario fixtures, position / range-rate (`invariance`) | σ ≤ 3e-9 |
| gravity campaign stages F/G/I, D25 requalification | unaffected: they use an explicit P0 (G2) |
| R1COV / R1M / FXR "data-only" σ | they called these data-only; D28 makes that literally true. The R1COV report already showed the data-only Schur σ_K agreeing with production to 3e-8, so no reported number moves materially |

**No authoritative number is superseded by D28.** Its effect is below every reported precision except on
deliberately weak single-pass arcs (≤ 3.4e-4). The D25 report's note that rebuilds include the default prior is
superseded by this contract.

## 4. Permanent regression protection

`tests/test_harden_d28_posterior_prior_contract.py`:

- the rule itself;
- two-way BLS six-state: data-only without a prior, explicit prior carried exactly;
- two-way BLS K path: data-only (`DATA_ONLY` status), and with a K prior alone only the K row enters;
- two-way SRIF: six-state data-only; the K path keeps its objective's full prior;
- position SRIF and BLS: information bitwise equal to the data-only helper; historical − data-only = the default
  diagonal; SRIF R^T R consistent.

Each test also checks that the historical construction is measurably different on the same fixture.

## 5. Verification

- Focused: D28 (6) + D25 (4) + BLS1 (5) + `test_k_srp_*`: 49 passed.
- Slow estimator and scenario suites (`LUNAR_OD_RUN_SLOW_TESTS=1`: `test_scenarios`, `test_estimators`,
  `test_r1_covariance_observability_parity`, `test_reporting`, `test_force_model_parity`,
  `test_two_way_range_integration`): 123 passed.
- Full suite on a clean detached worktree of `3959f7a` plus the D28 files (`LUNAR_OD_PROJECT_ROOT` = workspace root): **1481 passed, 29 skipped, 0 failed** (D25 1475 + 6).

## 6. Open, out of scope

- **SRIF K-path objective.** It pulls toward the default state diagonal (§1). That is consistent with its
  posterior, but it is a hidden prior in an estimate. Changing it would change estimates, which is a separate,
  scoped decision.
- **Rank-deficient data-only problems.** Without the hidden prior, a genuinely unobservable no-prior problem is
  reported as such: six-state through the existing floored inverse (R1M caveat), the K path through the QR rank
  test. None occurred here.
- D27 and stochastic K are not started.

## 7. Addendum — HARDEN-D29 (2026-10-12)

The one exception above, the SRIF K path whose objective carried the default state rows, was a defect: it
departed from the BLS contract its docstring declares. HARDEN-D29 removed it. The SRIF K objective is now data-only
state + the required K prior, and its posterior carries exactly those. The D28 contract now holds in every path:
the 15 km / 5 m/s diagonal is a step regularizer only. See `docs/phase17_harden_d29_srif_k_prior_contract.md`.
