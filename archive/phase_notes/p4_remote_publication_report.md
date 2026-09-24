# GIT PHASE P4 — REMOTE PUBLICATION, PROVENANCE TAGGING, AND REPOSITORY CLEANUP

## Executive summary

P3 had already established a scientifically qualified local canonical state through Phase 17A-R. P4 published that exact commit to GitHub, preserved the separate Phase 17-R estimator WIP remotely, added three minimal immutable provenance tags, closed PRs #2–#4 as superseded, and removed only remote branches whose tips were already reachable from canonical `main`.

No production source, scientific test, physical model, numerical implementation, or commit history was rewritten. Remote `main` now exactly equals local `main` at `fea476f81dad07b3914e53eab709e10fa6e9d10b`, with tree `a96e5dc28a51d1bc9a6ecfbe713b6361db897959`.

## Starting-state and publication gates

After `git fetch --prune origin`, origin/main was still `386710590df9b7d767418973715753735f3f18eb`, while local main was `fea476f81dad07b3914e53eab709e10fa6e9d10b`; local was 62 commits ahead and zero behind. The canonical worktree was clean. All six P2/P3 critical SHA256 values matched before publication.

Purpose: ensure no remote drift or local scientific-byte drift exists. Setup: clean canonical main worktree and fetched origin refs. Method: compare commit/tree IDs, ancestry, status, and file SHA256 values. Result: `P4_INPUT_GATE=PASS`, `REMOTE_MAIN_DRIFT_GATE=PASS`, `PRE_PUBLICATION_IDENTITY_GATE=PASS`.

## Main publication and remote identity

Purpose: publish the already-qualified state without introducing a merge or rewriting history. Method: ordinary `git push origin main`, followed by a fresh fetch and independent remote ref/tree verification. Result: fast-forward `3867105..fea476f`; no force option was used. `origin/main` and local `main` now share both commit and tree exactly.

Git ref change: remote main advanced. Commit graph change: no new commit. File-byte change: none. Physical/scientific change: none. Limitation: this proves publication identity, not repair of pre-existing provenance/environment failures.

## Phase 17-R WIP preservation

The local WIP tip `13e75f111dbb13bd6ee57d405a66e15c4d909f7d` was absent remotely before P4. It was pushed normally as `origin/wip/phase17-r-estimator-integration` and re-verified at the exact same SHA. It remains explicitly unqualified, unmerged, and outside canonical main.

`PHASE17R_WIP_REMOTE_GATE=PASS` and `PHASE17R_WIP_PRESERVED_SEPARATELY=YES`.

## Provenance tags

Three annotated tags were created and pushed explicitly; no `--tags` operation was used:

- `high-order-gravity-qualified` → `fc13cc4000767ac07abeb3856ec18521bfbc0f76`, the committed high-order-gravity base.
- `r1-r4-long-arc-conditioned` → `a347397e89f0c7b921eaa66c3f94909f75113bd7`, the divergent long-arc qualification tip.
- `phase17a-r-canonical` → `fea476f81dad07b3914e53eab709e10fa6e9d10b`, the canonical pre-Phase-17-R scientific boundary.

The optional reconstruction-only tag was not created; the branch remains available locally as a forensic checkpoint. No WIP tag was created.

## PR supersession and closure

Before closure, each PR head was independently proven reachable from published `origin/main`:

- PR #2 head `543b6b9...`: reachable.
- PR #3 head `9dd19c3...`: reachable.
- PR #4 head `6e88015...`: reachable.

All three were closed, not merged, with a provenance-safe note that their commits remain preserved in canonical history. Final states: #2 CLOSED, #3 CLOSED, #4 CLOSED; `MERGED_NOW=NO` for all.

## Remote branch reachability and cleanup

Every deleted remote branch had zero commits outside `origin/main`, no remaining open PR reference, and a covered provenance role. Deleted refs were:

`feature/lunar-high-order-gravity-closure`, `feature/lunar-j2-force-models`, `feature/r0b-force-contract-parity`, `feature/r1-covariance-observability-parity`, `feature/r2-measurement-fidelity-parity`, `feature/r3-exact-station-transform`, `feature/r4-four-event-counted-doppler`, `fix/history-domain-enforcement`, `fix/r0a-force-config-ukf-parity`, `fix/r1-r4-long-arc-time-conditioning`, and `validation/r1-r4-long-arc-qualification`.

The high-order and long-arc tips are additionally protected by immutable tags. After pruning, the only remote branches are `main` and `wip/phase17-r-estimator-integration`. All deleted scientific tips remain reachable through canonical main or approved tags; `LOST_REMOTE_SCIENTIFIC_COMMITS=0`.

## Local branch/worktree policy

No local branches were deleted. Existing historical branches are checked out by numerous user worktrees, while `main`, the qualified reconstruction, integration checkpoint, and WIP remain useful forensic/recovery refs. Removing them would require separate worktree-owner review and is outside this publication phase.

## Final canonical scientific identity

The final six critical SHA256 values are unchanged from P2/P3:

```text
dynamics.py = 49bd030ec0aaa4b210ab47eac374fce32c60342e16f4b6644cd694ff0a9324ab
two_way_range.py = bddff533c4a1773ba80dd8f96cbb6933baa516b40c0e71f82bb56c927880a995
radiometrics.py = a41e276e5331134d5bfc877b5cc20b1a875a3638ed6dd87fb0b37d2320f2faf5
two_way_counted_doppler_reference.py = 8e21b3346470ab2d21d96e7d543575c804e25e27f4d4f49d6e6d7f5deb5eb2fa
estimators.py = 94146c906686ea7c08a671a8db2976d9d6191d85efc259fbf2aa415d260a58cd
filters.py = 43dc95e22847fbb855981d8c6eb5f65ba73b2e2e39ea0222b95d4822f4ab0045
```

P21, Model-S local-delay independence, long-arc qualification, and derivative-chain gates remain PASS. No Phase 17-R solve-for functionality is reachable from main.

## Remaining issues intentionally unchanged

- Phase 16 protected dynamics hash mismatch remains a separate provenance issue.
- R2 current-tree closure/hash errors remain pre-existing provenance failures.
- FA-06 remains a pre-existing workspace-root environment assertion.
- The 83-artifact `od_covariance_campaign` repository remains separate and untouched.
- Phase 17-R remains preserved WIP and is not qualified.

## Final verdict

`P4_REPOSITORY_CLOSURE_GATE=PASS`.

The repository now has remotely published canonical Phase 17A-R science, explicit immutable boundary tags, closed superseded PRs, no lost scientific commits, and a separately recoverable remote Phase 17-R WIP branch. The next authorized operation is Phase 17-R0 canonical WIP transplant and revalidation; it is not started here.
