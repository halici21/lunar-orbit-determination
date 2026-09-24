# GIT PHASE P3 — MAIN CANONICALIZATION AND LOCAL REPOSITORY CLOSURE

## Executive summary

Before P3, local `main` still pointed at the old `3867105` baseline while P2
had already qualified `integration/phase17-canonical@fea476f`.  P1 separated
the Phase 17-R WIP; P2 integrated the long-arc history and the qualified
Phase 14–17A-R state.  P3 moved only the local `main` ref, fast-forward-only,
to that exact P2 commit.  No production bytes, physics, numerical
implementation, remote refs, PRs, tags, or branches were changed.

## P2 final-head reconciliation

`2b4be83` is an ancestor of `fea476f`.  The only commit after the manifest’s
recorded head is `fea476f`, and its sole changed path is
`08_manifests/phase17_canonical_integration_manifest.txt`.  It records the
later final identity; no production or scientific test file changed.  The
discrepancy is therefore provenance-only and historically truthful.

## Input and ancestry gates

After `git fetch --prune origin`, `origin/main` remained
`386710590df9b7d767418973715753735f3f18eb`; no remote drift occurred.  The
controlled integration, qualified, and WIP refs matched P2.  All four
long-arc commits, `f454d59a`, and `9833bd6` are reachable from `fea476f`.
The WIP preservation commit `13e75f1` is not reachable, and no production
estimator file contains Phase 17-R solve-for functionality.

## Worktree safety and canonicalization

`main` was not checked out in the previously dirty primary worktree; that
worktree is on `feature/lunar-j2-force-models`.  A clean dedicated main
worktree was created at
`C:\Users\erayh\Documents\Python\Grad\python_port_main_canonical`.
It advanced from `3867105` to `fea476f` across 62 commits with
`git merge --ff-only`.  No new commit was created by the main move.  The main
worktree is clean and its HEAD/tree exactly equal integration.

The raw `dynamics.py` working bytes initially reflected Windows
`core.autocrlf` normalization after checkout.  The already-qualified raw bytes
were restored from the clean integration worktree and `git add --renormalize`
left the index/ref unchanged and status clean.  This was a working-tree byte
normalization issue, not a scientific or Git-tree change.

## Critical identity checks

All six P2 hashes match on canonical main:

* dynamics `49bd030e...`
* two-way range `bddff533...`
* radiometrics `a41e276e...`
* Model-S `8e21b334...`
* estimators `94146c90...`
* filters `43dc95e2...`

P21 remains PASS with no pin change.  Model-S remains independent and local
delay-based.  The event-epoch binade dependence remains removed; the measured
minimum range increment remains `6.656729789611404e-08 m`.

## Canonical-main qualification

The focused canonical-main suite passed for the long-arc harness, P21/Model-S,
range, four-event counted Doppler, K-sensitivity, SRP, force-gradient parity,
and reference configuration.  The long-arc harness result was 18 passed, 0
failed, 1 skipped.  The full suite collected 1,178 tests: 1,138 passed, 2
failed, 10 errors, and 28 skipped.

The nonpasses are classified, not ignored:

* one pre-existing Phase-16 protected dynamics-hash mismatch;
* one pre-existing R2 current-tree protection mismatch;
* ten pre-existing R2 `CLOSURE_CURRENT_TREE_SHA256` errors;
* the pre-existing workspace-root FA-06 Python-path assertion.

`NEW_SCIENTIFIC_REGRESSIONS = 0` and `UNKNOWN_NONPASSES = 0`.

Estimator defaults remain pre-Phase-17-R; BLS, SRIF, and SR-UKF K solve-for
support is absent.  Reference configuration keeps K_SRP UNKNOWN and does not
use the screening envelope as a prior.

## Local versus remote state

`LOCAL_MAIN = fea476f81dad07b3914e53eab709e10fa6e9d10b`.
`ORIGIN_MAIN = 386710590df9b7d767418973715753735f3f18eb`.
Local main is 62 commits ahead; origin is 0 commits ahead.  No push occurred.

## Historical reachability and cleanup plan

All historical committed development branches audited are fully contained by
local main.  The sole intentionally non-contained branch is
`wip/phase17-r-estimator-integration`, which must be retained.  Preserve
`feature/phase14-17ar-qualified` and `integration/phase17-canonical` until
remote publication is verified.

PR #2/#3/#4 are superseded by the canonical local history, but were not closed.

Recommended future tags (plan only):

* `phase17a-r-qualified` → `0e785f6`, reconstructed qualified boundary;
* `r1-r4-long-arc-conditioned` → `a347397`, long-arc qualified tip;
* `phase17-canonical-pre-estimator` → `fea476f`, integrated canonical state.

Potentially removable after P4 verifies remote publication: the old stacked
branches and the long-arc source branches.  Must keep until then: WIP,
qualified reconstruction, and integration refs.  No deletion occurred in P3.

## What changed and what did not

`GIT_REF_CHANGE`: local `main` moved by fast-forward.

`GIT_HISTORY_CHANGE`: no new commit or merge commit created by canonicalization.

`PRODUCTION_BYTE_CHANGE`: none; six critical hashes and tree identity match.

`PHYSICAL_MODEL_CHANGE`: none.

`NUMERICAL_MODEL_CHANGE`: none.

`TEST_CHANGE`: none in P3.

`REMOTE_REPOSITORY_CHANGE`: none.

Remaining issues are the Phase-16 protected-hash mismatch, R2 closure
provenance failures, FA-06 environment assertion, 83 external campaign
artifacts, and isolated Phase 17-R WIP.  P3 does not repair any of them.

## Verdict

`P2_HEAD_RECONCILIATION_GATE = PASS`

`P2_FINAL_TREE_GATE = PASS`

`P3_INPUT_GATE = PASS`

`MAIN_WORKTREE_GATE = PASS`

`MAIN_FAST_FORWARD_GATE = PASS`

`MAIN_TREE_IDENTITY_GATE = PASS`

`CRITICAL_HASH_GATE = PASS`

`P21_MAIN_GATE = PASS`

`MODEL_S_MAIN_GATE = PASS`

`LONG_ARC_MAIN_GATE = PASS`

`DERIVATIVE_CHAIN_MAIN_GATE = PASS`

`ESTIMATOR_DEFAULT_MAIN_GATE = PASS`

`FULL_SUITE_MAIN_GATE = PASS WITH CLASSIFIED PRE-EXISTING NONPASSES`

`HISTORICAL_BRANCH_REACHABILITY_GATE = PASS`

`REPORT_COMPLETENESS_GATE = PASS`

`P3_LOCAL_CANONICALIZATION_GATE = PASS`

`PRIMARY_CLASS = LOCAL_MAIN_CANONICALIZED_TO_PHASE17A_R_QUALIFIED_INTEGRATION`

Next action: GIT PHASE P4 — remote publication, tagging, and branch/PR cleanup.
Do not execute P4 in this phase.
