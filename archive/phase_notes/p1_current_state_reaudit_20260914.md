# GIT PHASE P1 — CURRENT-STATE RE-AUDIT

This supplement records the state after the P1 preservation/reconstruction
commits and verifies that no follow-up operation changed them.

## Current state

The preserved target worktree is clean at
`wip/phase17-r-estimator-integration@13e75f111dbb13bd6ee57d405a66e15c4d909f7`.
The reconstructed worktree is clean at
`feature/phase14-17ar-qualified@0e785f624a4e512177b3e7b5c0adc00d2c809e34`.
The original dirty-state inventory remains in
`p1_dirty_state_inventory.txt` and records 17 modified tracked files and 31
untracked target files before preservation.  The primary checkout's unrelated
README/doc dirt was not touched.

## External artifact repository

`od_covariance_campaign` is a separate Git repository, `master` at
`1d24e1278434b2c9ecc5a5661cb7eb7194823504`, with no remote.  It currently has
83 untracked artifacts: 28 results, 13 reports, 11 manifests, 18 checkpoints,
and 13 other phase-support files.  Every file is SHA256-fingerprinted in
`p1_external_artifact_inventory_20260914.txt`.  The repository's `.git`
metadata was not imported into the target.

Canonical provenance retained externally includes all phase baseline manifests,
the Phase 17C-RF refreeze manifest, Phase 17A-R artifacts, and Phase 17-R WIP
reports/checkpoints.  Temporary/intermediate outputs remain separate pending
owner-approved canonical placement.

## Gate results

`CURRENT_STATE_INVENTORY_GATE = PASS`

All 17 tracked modifications and 31 target untracked files were previously
hashed and classified; all 83 external artifacts are now independently
fingerprinted and classified. `UNCLASSIFIED_CRITICAL_FILES = 0`.

`PRESERVATION_GATE = PASS`

The 48-file preservation snapshot was compared with the original inventory:
zero tracked mismatches, zero untracked scientific mismatches, and zero lost
files. The preserved snapshot is explicitly WIP and carries no Phase 17-R
qualification claim.

`QUALIFIED_RECONSTRUCTION_GATE = PASS`

Six authoritative Phase 17A-R hashes match; P21 and Model-S checks pass; the
qualified branch contains no `solve_for_k_srp` production/test additions.

## Scientific requalification evidence

Focused suites on the reconstructed branch passed for K-sensitivity,
derivative-chain contracts, SRP production, force-gradient parity, event
conditioning, range, and counted-Doppler paths. The full suite collected 1,178
tests: 1,138 passed, 2 failed, 10 errors, and 28 skipped. The failures were
the expected Phase-16 protected dynamics hash and the workspace-root FA-06
isolation assertion; the ten errors were the known R2 campaign
`CLOSURE_CURRENT_TREE_SHA256` errors. No new scientific regression was
observed. The focused groups were 31, 80, 65, and 86 passing tests
respectively (the smaller groups overlap the larger group).

## Interpretation

Phase 14–17A-R was historically qualified in a dirty worktree based on
`fc13cc4`. The qualified branch is a later provenance-preservation
reconstruction of those already-qualified bytes, not a fabricated historical
commit history. Phase 17-R remains only on the WIP branch and external WIP
artifacts.

`MAIN_CHANGED = NO`; `LONG_ARC_BRANCH_MERGED = NO`; `OPEN_PRS_MODIFIED = NO`;
`PUSH = NONE`.
