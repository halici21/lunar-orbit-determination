# GIT PHASE P2 — CANONICAL INTEGRATION AND FULL SCIENTIFIC QUALIFICATION

## Executive summary

P2 combined the committed high-order history, the divergent long-arc history,
and the reconstructed Phase 14–17A-R state in a clean branch.  The long-arc
commits and harness are preserved in ancestry, while the later Phase 17C/RF
local-delay implementation remains the production authority.  No Phase 17-R
estimator work was merged.  All six P1-qualified critical production/baseline
hashes remain byte-identical.

## Controlled inputs and environment

The supplied P1 tips resolved exactly, except that the WIP hash in the prompt
was a unique 39-character prefix of the actual 40-character object ending in
`f7d`; this was not input drift.  A clean worktree was created from `main` at
`C:\Users\erayh\Documents\Python\Grad\python_port_phase17_canonical`.
`main` was not checked out or modified.

## History integration

`main` fast-forwarded to `fc13cc4` with no reconciliation.  The long-arc line
was merged with history preserved as `f454d59a`; commits `3c6f3b0`, `47c289d`,
`274a489`, and `a347397` are reachable.  The qualified overlay was merged as
`9833bd6`.  No squash, rebase, or cherry-pick was used.

## Scientific reconciliation

Three textual conflicts occurred in `radiometrics.py`, `two_way_range.py`, and
`test_od_contracts.py`.  Five files had semantic overlap, including the R2 and
measurement-safety tests.  The final choice was not “newer wins”: the long-arc
fix and Phase 17C repair are the same numerical family.  Both stop deriving
small light-time intervals by subtracting large arc-relative epochs.  Phase 17C
also completes the repair by assembling the reported round-trip observable from
the converged local downlink/uplink delays and supplies independent-oracle
qualification.

The long-arc fix remains scientifically valid: it identified the loss of local
delay precision (`~1.5e-11 s` versus `~4.4e-16 s` at a one-day relative epoch),
which made the `1e-11 s` residual gate fail after long arcs.  Phase 17C
partially subsumes that implementation because it applies the same local-delay
association to the complete measurement stack; it does not erase the long-arc
history or its harness.

The old P08 assertion requiring production to beat Model-S in every fixture was
superseded as an accuracy authority.  It is now characterization-only; the
independent local-delay oracle remains the accuracy gate.  A residual helper
from the long-arc line was likewise replaced with the already-qualified
local-delay residual recomputation.  No physical threshold was loosened.

## Production identity and event conditioning

Final working-tree SHA256 values match P1 exactly for dynamics, range,
radiometrics, Model-S, estimators, and filters.  P21 remains pinned to
`bddff533...`; Model-S remains independent and uses local-delay assembly.
Historical evidence reports the absolute-ET regression PASS, event-epoch binade
dependence REMOVED, and a flat post-repair minimum range increment of
`6.656729789611404e-08 m` (range value ULP `5.960464477539063e-08 m`).
The old quantization signature (`c*ulp(t_event)/2`, binade doubling) is absent.

## Qualification results

The deterministic long-arc harness passed (`18 passed, 1 skipped`).  The
integrated focused science suites passed for long-arc, P21/Model-S, range,
four-event counted Doppler, K-sensitivity, SRP, force-gradient parity, and
reference configuration.  The full suite collected 1,178 tests and produced
1,138 passes, 2 failures, 10 errors, and 28 skips.  The known non-passes are:

* the pre-existing Phase-16 protected dynamics hash mismatch;
* the pre-existing workspace-root FA-06 isolation assertion;
* ten pre-existing R2 campaign `CLOSURE_CURRENT_TREE_SHA256` errors.

No new scientific regression was observed.  The R2 dynamics mismatch is not
repaired or re-frozen.

## Derivative-chain and estimator state

The Phase 17A-R force, trajectory, range composition/E2E, and counted-Doppler
composition/E2E sensitivity gates pass in the integrated focused suites.
`MEASUREMENT_K_SENSITIVITY = PASS` and `DERIVATIVE_CHAIN_GATE = PASS`.
Production BLS, SRIF, and SR-UKF remain six-state/default behavior; no
`solve_for_k_srp` token or Phase 17-R estimator augmentation exists in the
integration tree.  Reference configuration keeps K_SRP unknown and does not
use a screening envelope as a prior.

## Provenance artifacts

No external campaign artifacts were imported.  The separate
`od_covariance_campaign` repository remains outside the primary tree with its
83 fingerprinted files and its own `.git` boundary.  The new P2 manifest and
this report provide the authoritative links without rewriting historical
reports.

## Final interpretation

The integrated branch preserves the long-arc scientific invariants and history
while retaining the later, more strongly qualified local-delay production
implementation.  The result is qualified through Phase 17A-R and explicitly
excludes Phase 17-R.

## Verdict

`P2_STARTING_STATE_GATE = PASS`

`HOG_INTEGRATION_GATE = PASS`

`LONG_ARC_HISTORY_MERGE_GATE = PASS`

`SEMANTIC_RECONCILIATION_GATE = PASS`

`QUALIFIED_PRODUCTION_IDENTITY_GATE = PASS`

`LONG_ARC_QUALIFICATION_GATE = PASS`

`DERIVATIVE_CHAIN_REQUALIFICATION_GATE = PASS`

`FULL_SCIENTIFIC_REGRESSION_GATE = PASS WITH CLASSIFIED PRE-EXISTING NONPASSES`

`CANONICAL_INTEGRATION_GATE = PASS`
