# Consolidation preservation archive — 2026-10-10

Files here preserve historical or unfinished work. They are not imported by the
library or collected by the normal tests/ suite.

- wip_conflicts/: exact versions of the nine conflicting files at WIP commit
  13e75f111dbb13bd6ee57d405a66e15c4d909f7d.
- wip_resolution.json: file hashes and conflict decisions.
- pending_worktrees/: tracked diffs and unique untracked files from three dirty
  worktrees; meta.json records their bases and file hashes.

The BLS1 estimator changes, qualification script/test and diagnostic JSON are
unfinished snapshots. Their preservation does not constitute scientific
acceptance. Activate them only through a separate reviewed change.

Original worktrees are retained. Local-only receipts map source labels to
absolute paths. All files under archive/ use -text attributes to preserve bytes.

See [the branch audit](../../docs/branch_consolidation_2026-10-10/README.md)
for integration decisions and validation results.
