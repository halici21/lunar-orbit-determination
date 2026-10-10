# PHASE 17-R0 — CANONICAL WIP TRANSPLANT AND REVALIDATION

## Executive summary

The preserved mixed WIP branch was not merged because its single preservation commit also contained historical Phase 14–17A-R provenance deletions and stale long-arc/R2 test changes. A clean branch was created from canonical `main`, and only the genuine Phase 17-R estimator delta was transplanted: `lunar_od/estimators.py`, `lunar_od/filters.py`, and the three K_SRP solve-for tests.

The transplant is one local commit, `b8da5e6db8e10ded08237d9ef5e1e8f13dcd4ae8`, on `feature/phase17-r-k-srp-estimation`. No push was performed. Canonical `main` and `origin/main` remain at `fea476f81dad07b3914e53eab709e10fa6e9d10b`.

Dedicated Phase 17-R tests and the focused canonical regression set pass. The full suite has only the already-known FA-06 environment assertion, Phase16 protected-hash assertion, and ten R2 closure setup errors; no transplant-introduced or unknown scientific failures were observed.

## 1. Starting-state verification

Purpose: prove the controlled inputs have not drifted. Setup: fetched `main`, `origin/main`, local WIP, and remote WIP. Method: exact ref/tree comparisons. Results: canonical main and origin are both `fea476f...` with tree `a96e5dc...`; local and remote WIP are both `13e75f...f7d`; both worktrees were clean at their controlled starts. Verdict: `CANONICAL_BASE_GATE=PASS`, `WIP_INPUT_GATE=PASS`.

## 2. Canonical-vs-WIP delta forensics

The raw WIP-vs-main difference contained 13 paths. Five were genuine estimator work; the remaining paths were historical/provenance or stale test material:

| File/category | Classification | Action |
|---|---|---|
| `lunar_od/estimators.py` | PHASE17R_PRODUCTION | TRANSPLANT |
| `lunar_od/filters.py` | PHASE17R_PRODUCTION | TRANSPLANT |
| three `tests/test_k_srp_*_solve_for.py` files | PHASE17R_TEST | TRANSPLANT |
| P1/P2 manifests and reports | HISTORICAL_PROVENANCE_NOISE | EXCLUDE |
| old long-arc example/test | HISTORICAL_PROVENANCE_NOISE | EXCLUDE |
| modified R2 and counted-Doppler tests | STALE/UNRELATED WIP | EXCLUDE |

No differences were found in `dynamics.py`, `two_way_range.py`, `radiometrics.py`, Model-S, SRP, or reference configuration. `UNKNOWN_CRITICAL_DIFFS=0`; `WIP_DELTA_FORENSICS_GATE=PASS`.

## 3. Controlled transplant

Purpose: preserve canonical physics and import only estimator integration. Method: file/patch-level application of the five approved paths onto a fresh worktree based on `main`; no merge, rebase, cherry-pick, or preservation-commit import was used. Resulting commit: `b8da5e6`.

Production behavior added is explicitly opt-in through `solve_for_k_srp=False` by default. BLS/SRIF use a separate K scalar and a seven-parameter augmented solve internally; SR-UKF uses a seven-element sigma-point state. K is `K_SRP=C_R A/m` in m²/kg. The qualification path uses static `K` propagation and zero K process noise; no reference spacecraft K value is imported. Direct measurement K dependence is absent: the measurement column is built through the already-qualified trajectory sensitivity chain.

Negative K initial/trial/sigma-point conditions fail closed or backtrack explicitly; there is no silent clipping or reset. Numerical `k_srp_scale` is a solver scaling factor, not a prior.

## 4. Estimator revalidation

Purpose: prove default invariance and common estimator mathematics before broader science. Setup: the three new solve-for test modules and existing estimator/filter suites. Method: bitwise default comparisons, linear-Gaussian posterior checks, nonlinear recovery, covariance checks, zero-information and informative-sensitivity controls.

Results:

- BLS default parity, Gaussian oracle, noise-free recovery, noisy recovery, and failure semantics: PASS.
- SRIF default parity, Gaussian oracle, nonlinear recovery: PASS.
- SR-UKF default parity, Gaussian oracle, per-sigma-point K propagation, static-K propagation, negative-K rejection, and nonlinear recovery: PASS.
- Common Gaussian posterior and covariance sanity: PASS.
- Zero-sensitivity and informative-sensitivity controls: PASS.

These results establish that the three estimator paths can consume the already-qualified K sensitivity and that enabling solve-for is opt-in. They do not yet establish observability, prior sensitivity, correlations, or holdout superiority; those belong to Phase 17-R1.

## 5. Phase 17A-R regression protection

The focused regression set passed for P21, Model-S, long-arc qualification, event conditioning, SRP production, force/trajectory K sensitivity, range sensitivity, counted-Doppler sensitivity, and the derivative chain. Event-epoch binade dependence remains removed and the old quantization signature is absent. No measurement-stack or event-solver file changed in the feature diff.

## 6. Full-suite results and classification

The feature worktree executed 1220 tests: 1180 passed, 2 failed, 10 errors, and 28 skipped. The two failures are the pre-existing Phase16 protected dynamics hash and FA-06 workspace-root pytest-path assertion. The ten setup errors are the pre-existing R2 `CLOSURE_CURRENT_TREE_SHA256` errors. No transplant-introduced non-pass and no unknown non-pass occurred.

## 7. What changed and what did not

Changed: estimator capability, behind an explicit opt-in, plus its three dedicated tests. Git provenance: one local feature commit. Not changed: force model, SRP law, shadow model, measurement physics, local-delay event solver, Model-S, reference configuration, canonical `main`, remote `main`, P21, or any protected pin. External `od_covariance_campaign` artifacts were inspected as context only and not imported.

## 8. Limitations and next action

The historical WIP report/checkpoints remain external evidence, not canonical qualification artifacts. The current tests demonstrate implementation and regression safety but do not close full Phase 17-R observability, covariance inflation, prior sensitivity, wrong-fixed-K, holdout, or three-estimator scientific interpretation.

`PHASE17_R0_GATE=PASS` and `PRIMARY_CLASS=PHASE17_R_WIP_CLEANLY_TRANSPLANTED_ON_CANONICAL_MAIN_AND_REVALIDATED`.

The next authorized phase is Phase 17-R1: K_SRP observability, prior sensitivity, and holdout qualification. Do not push or merge this branch automatically.
