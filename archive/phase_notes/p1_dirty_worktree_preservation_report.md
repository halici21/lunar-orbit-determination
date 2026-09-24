# GIT PHASE P1 — DIRTY WORKTREE PRESERVATION AND PHASE 17A-R QUALIFIED OVERLAY RECONSTRUCTION

## 1. Executive Summary

The mixed dirty canonical worktree was preserved byte-for-byte before any
partitioning.  A WIP branch and preservation commit now contain the complete
dirty state, including the started Phase 17-R estimator work.  A separate
qualified branch was reconstructed from `fc13cc4` using phase manifests and
authoritative file hashes; it contains the qualified Phase 14–17A-R overlay and
the pre-Phase-17-R estimator/filter baselines, but none of the identified
solve-for-K_SRP additions.

## 2. Why This Phase Was Necessary

The repository's committed canonical base ended at `fc13cc4`, while later
qualified science was still uncommitted in the worktree.  The same worktree
already contained unqualified Phase 17-R estimator changes.  Preservation had
to precede cleanup or reconstruction so qualification claims could not be
silently mixed.

## 3. Starting Repository State

The target scientific worktree was the high-order branch at `fc13cc4` with 17
modified tracked files and 31 untracked files.  The primary checkout's
unrelated README/doc dirt was not touched.  The starting inventory is recorded
in `p1_dirty_state_inventory.txt`.

## 4. Dirty Worktree Inventory

The inventory records each tracked file's HEAD blob, current SHA256, size,
line-ending form, and diff stat; and every untracked file's SHA256, size, and
classification.  Production, test, documentation, and Phase 17-R checkpoint
artifacts are included.  Ignored bytecode was inventoried as project-relevant
but not committed.

## 5. Nested Artifact Repository Analysis

`od_covariance_campaign` is a separate Git repository (`master`, HEAD
`1d24e127...`, no remote) containing 83 untracked report/manifest/result
artifacts at recheck.  Its `.git` internals were not added to the target.  The
artifact repository remains recoverable in place; canonical artifact placement
is deferred to owner-approved integration.

## 6–9. Preservation Strategy, Branch, Snapshot, and Gate

Branch created from the exact `fc13cc4` ancestry:
`wip/phase17-r-estimator-integration`.

Preservation commit:
`13e75f111dbb13bd6ee57d405a66e15c4d909f7` — “wip: preserve mixed
Phase 14-17A-R and Phase 17-R work”.  It is explicitly WIP and does not qualify
Phase 17-R.  The staged snapshot contains 48 files.  Comparison against the
pre-preservation inventory found zero mismatches and zero missing paths.

**PRESERVATION_GATE = PASS**

Physical model change: none.  Git changed only by adding the preservation
branch/commit; no history was rewritten.

## 10–12. Phase 17A-R Boundary Evidence

The boundary is identified by `phase14_base_freeze`, phase 15/16/17/17A
manifests, `phase17ar_manifest.txt`, the recovery snapshot, and the exact
qualified hashes.  Hashes outrank timestamps.  Historical manifests remain
unchanged and still truthfully describe their original uncommitted state.

## 13–18. File-by-File Partition

| File/category | Phase 17A-R state | Current Phase 17-R change | Target |
|---|---|---|---|
| `lunar_od/dynamics.py` | authoritative `49bd...` | no solve-for change | QUALIFIED_OVERLAY |
| `lunar_od/two_way_range.py` | authoritative `bddf...` | no solve-for change | QUALIFIED_OVERLAY |
| `lunar_od/radiometrics.py` | authoritative `a41e...` | no solve-for change | QUALIFIED_OVERLAY |
| `lunar_od/two_way_counted_doppler_reference.py` | authoritative `8e21...` | no solve-for change | QUALIFIED_OVERLAY |
| `lunar_od/estimators.py` | recovery snapshot `94146...` | current `93e9...` solve-for augmentation | QUALIFIED_OVERLAY / WIP_ONLY diff |
| `lunar_od/filters.py` | Phase 17C-RF clean `43dc...` | current `1e5f...` solve-for augmentation | QUALIFIED_OVERLAY / WIP_ONLY diff |
| `tests/test_k_srp_{bls,srif,srukf}_solve_for.py` | absent | newly started Phase 17-R tests | WIP_ONLY |
| other phase 14–17A-R production/tests | manifest-supported | no identified Phase 17-R ownership | QUALIFIED_OVERLAY |
| campaign reports/checkpoints | separate artifact repo | some Phase 17-R WIP | EXTERNAL / OWNER_REVIEW |

## 19–21. Reconstruction and Hash Verification

Branch `feature/phase14-17ar-qualified` was created from `fc13cc4` and
committed as `0e785f624a4e512177b3e7b5c0adc00d2c809e34`, with the new
`08_manifests/phase14_17ar_reconstruction_manifest.txt`.  The manifest records
the later reconstruction, sources, exclusions, ambiguity, and authorization;
it does not fabricate historical phase commits.

Authoritative SHA256 checks: six checked, zero mismatches:

* dynamics `49bd030e...`
* two-way range `bddff533...`
* radiometrics `a41e276e...`
* counted-Doppler reference `8e21b334...`
* estimators `94146c90...`
* filters `43dc95e2...`

P21 pin matches `bddff533...`.  Model-S uses independently solved downlink and
uplink light times and returns `downlink_lt + uplink_lt`, not a regressed
`t3-t1` shortcut.

## 22. Phase 16 Provenance Issue

The known pre-Phase-16 dynamics hash `64411c...` remains distinct from the
qualified Phase 17A-R hash `49bd...`.  No refreeze was performed; the later SRP
and gravity evolution is retained.

## 23. Scientific Requalification

`pytest -q tests/test_k_sensitivity_fd_harness.py tests/test_od_contracts.py`
on the reconstructed branch: **31 passed**.  This rechecks measurement K
sensitivity, derivative-chain contracts, P21 pinning, and Model-S contracts.
The previously recorded broader phase suite remains the source for the known
pre-existing failures (one Phase 16 protected-current-tree mismatch and ten R2
protected closure errors); this P1 run introduced no new failure in the focused
requalification.

## 24. WIP-vs-Qualified Difference

The deterministic commit comparison is recorded in `p1_wip_vs_qualified.txt`.
It identifies exactly two production diffs (`estimators.py`, `filters.py`),
three Phase 17-R solve-for tests, and the qualified-only reconstruction
manifest.  `solve_for_k_srp` is absent from the qualified branch and present in
the WIP branch.  Phase 17-R remains WIP regardless of local individual-gate
signals.

## 25–29. Git/Physical Changes and Partition Gate

Git changes are the WIP preservation commit and qualified reconstruction
commit.  No physical model change was made; this is provenance preservation.
No long-arc branch was merged, `main` was untouched, and open PRs were not
modified.  No file or artifact was lost.  The qualified branch contains no
solve-for-K_SRP WIP additions.

**QUALIFIED_PARTITION_GATE = PASS**

## 30. Exact Inputs for Next Integration Phase

The next phase may use `main`,
`feature/lunar-high-order-gravity-closure`,
`fix/r1-r4-long-arc-time-conditioning`, and
`feature/phase14-17ar-qualified` as controlled inputs.  It must perform the
canonical integration and full regression; this phase does not.

## 31. Final Verdict

The repository now has two scientifically honest recoverable states:
qualified through Phase 17A-R and Phase 17-R work in progress.  The qualified
branch is a later versioning/reconstruction action, not a claim that those
phases were historically committed.

**REPORT_COMPLETENESS_GATE = PASS**

