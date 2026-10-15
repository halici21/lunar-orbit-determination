# Branch consolidation — 2026-10-10

This report supersedes earlier branch-layout instructions. Historical scientific
reports retain their original meaning and acceptance criteria.

## Integration

Starting main: b12bd7349d48f827eb82fb732ab7d7c8f3ce23fe (PR #7).
Twenty original local branch tips were audited; the initial remote branches
were main, codex/lunar-od-workspace, feature/phase17-r-k-srp-estimation,
wip/phase17-r-estimator-integration, and archive/worktree-snapshots-2026-09-24.

| Original branch | Tip | Commits missing from main | Decision |
| --- | --- | ---: | --- |
| archive/worktree-snapshots-2026-09-24 | a25d6fa | 3 | Merge archive only |
| codex/lunar-od-workspace | a9f9e56 | 0 | Already included; preserve tag and retire branch |
| feature/lunar-high-order-gravity-closure | fc13cc4 | 0 | Already included; preserve tag and retire branch |
| feature/lunar-j2-force-models | 632560d | 0 | Already included; preserve tag and retire branch |
| feature/phase14-17ar-qualified | 0e785f6 | 0 | Already included; preserve tag and retire branch |
| feature/phase17-r-k-srp-estimation | 46e21c8 | 56 | Merge research history |
| feature/r0b-force-contract-parity | 9dd19c3 | 0 | Already included; preserve tag and retire branch |
| feature/r1-covariance-observability-parity | 6e88015 | 0 | Already included; preserve tag and retire branch |
| feature/r2-measurement-fidelity-parity | 5ca652a | 0 | Already included; preserve tag and retire branch |
| feature/r3-exact-station-transform | d03a913 | 0 | Already included; preserve tag and retire branch |
| feature/r4-four-event-counted-doppler | 5bf0c54 | 0 | Already included; preserve tag and retire branch |
| fix/history-domain-enforcement | da28e5a | 0 | Already included; preserve tag and retire branch |
| fix/light-time-nonconvergence | e572a64 | 0 | Already included; preserve tag and retire branch |
| fix/measurement-model-safety | 3265dbe | 0 | Already included; preserve tag and retire branch |
| fix/r0a-force-config-ukf-parity | 543b6b9 | 0 | Already included; preserve tag and retire branch |
| fix/r1-r4-long-arc-time-conditioning | a347397 | 0 | Already included; preserve tag and retire branch |
| integration/phase17-canonical | fea476f | 0 | Already included; preserve tag and retire branch |
| main | fea476f | 0 | Already included; preserve tag and retire branch |
| validation/r1-r4-long-arc-qualification | 3c6f3b0 | 0 | Already included; preserve tag and retire branch |
| wip/phase17-r-estimator-integration | 13e75f1 | 1 | Merge ancestry; retain current qualified code in all conflicts |

The research merge imports 56 commits, including production DeltaDOR and lunar
landmark optical models, K_SRP estimators and observability studies, square-root
covariance work, the SRP variational gradient, physical contract hardening,
counted-Doppler Earth interpolation, and their committed evidence.

The archive merge imports three preservation commits. These files are historical
source and evidence, not executable production modules.

The old WIP commit was a snapshot predating subsequent qualification. Nine files
conflicted. The current research versions were retained after reviewing the
changes: the WIP would remove the SRP position gradient from the variational
equations, restore superseded estimator/covariance code, and reintroduce fixed
machine paths. Exact WIP versions and SHA-256 hashes are preserved in
[the resolution manifest](../../../archive/branch_consolidation_2026-10-10/wip_resolution.json).
No independent nonconflicting WIP change remained to transplant.

Integration invariants verified with Git:
- lunar_od/ is identical to research tip 46e21c80c7ee2ff3caf2b650d2c5e8fc3fc01fc7.
- desktop_app/ and examples/run_scenario_config.py are identical to initial main.
- All 20 original branch tips are ancestors of the integration commit.
- No numerical tolerances or algorithms were changed.

## Desktop bridge provenance integration fix

The first combined suite found that the R2 current-tree gate still expected the
historical run_scenario_config.py hash. PR #7 had already changed this script
through the owner-approved opt-in desktop bridge. This was an integration defect:
the old table was unaware of the merged UI change.

An additive DESKTOP_WORKSPACE_REFROZEN_CURRENT_TREE_SHA256 table now accepts
only the exact PR #7 bytes. Historical hash tables, protected-path membership and
all numerical thresholds remain unchanged. The script itself is unchanged.
[The scope and old/new hashes](desktop_refreeze.json) are recorded explicitly.

tests/test_desktop_bridge_provenance.py verifies the accepted hash against the
filtered Git contents at the approved commit and mutates a copied script to
verify that a later edit still fails. The existing R2 campaign and compatibility
tests qualify the combined tree.

## Pending work preservation

Three dirty worktrees were snapshotted under
[the consolidation archive](../../../archive/branch_consolidation_2026-10-10/README.md).
Their original files remain in place. The snapshots include unfinished BLS1
physics work and a newer workflow document; these are not activated in main.
Snapshots carry file hashes and base commits. The phase17r0 patch was verified
with git apply --reverse --check against its source working tree.

Old branch tips are kept as annotated tags:
branch-history/2026-10-10/<original-branch-name>.
Restoring a historical branch: git switch -c <new-branch> <tag>.
Restore pending work in a separate checkout at the recorded base commit, apply
changes.patch, then copy the corresponding untracked files.

## Validation and publication

Final full suite: **1745 passed, 29 skipped, 0 failures, 0 errors** (1774 tests).
The skip reasons are optional GRGM660PRIM/Phase 13G data and explicitly slow
regressions. Real BLS and UKF desktop jobs both exited successfully, wrote
recorded artifacts and opened their results while the UI event loop continued.
The deliberately short BLS run reported operational_success=false, which the UI
preserved; process completion was not presented as estimator convergence.


The final test command, dependency versions and counts are recorded in
[validation.json](validation.json). The final cleanup receipt is retained locally
as outputs/branch-consolidation-completion.json in the Codex workspace.
External gravity coefficients, SPICE kernels and historical campaign data remain
outside Git. Archives contain legacy paths in historical documents; absolute
paths in newly generated worktree metadata are stored in local-only receipts.

A single main branch is the active development target after publication. Existing
dirty checkouts are detached at their original commit rather than overwritten.
The clean canonical checkout is python_port_main_canonical.
