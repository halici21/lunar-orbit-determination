# Repository professionalization — 2026-10-15

This migration starts at `f27db9d` (the merged Phase 17 main baseline).
It changes repository architecture, packaging and path resolution. Scientific
thresholds, constants, estimator/model implementations, frozen source hashes,
fixtures, predeclarations and result values are preserved.

## Audit and target

The starting tree had 991 tracked files: 55 in the root `lunar_od` package,
136 example files, 117 test/support files, 108 documentation files, 300
artifacts, 97 archive files, 74 desktop files, and four loose manifests.
There was no `pyproject.toml` or citation metadata. Durable guidance, legacy UI
documents and research reports were mixed at the root and in `docs/`.

The target is described in [repository-layout.md](repository-layout.md).
[path-migration.json](path-migration.json) maps every moved path and records
its original checkout SHA-256. The integrity checker enforces byte identity
for moved numerical modules and scientific records; the three edited package
modules contain repository/data-location helpers only.

| Before | After |
| --- | --- |
| `lunar_od/` | `src/lunar_od/`, installable as `lunar-orbit-determination` |
| `examples/phase17*.py` | `studies/phase17/scripts/` |
| Mixed `artifacts/` | Study `configs/` and `results/`; hygiene evidence in development docs |
| `docs/phase17*.md` | `studies/phase17/reports/` |
| `08_manifests/` | `studies/phase17/provenance/` |
| Flat durable docs and root reports | `docs/theory`, `validation`, `research`, `development` |
| Hygiene utilities inside artifacts | `tools/repository_hygiene/` |

No committed artifact was deleted. Raw records up to approximately 4.3 MB
remain because reports/tests use them as scientific evidence. Runtime outputs,
Numba caches, package metadata and local study output are ignored.

## Safety and compatibility

Five pre-existing worktrees were inspected before editing. The canonical
`main` had three modified desktop/test files; the Phase 17 worktree had
substantial uncommitted desktop work; the primary detached checkout had an
untracked research document. Their files were not edited, staged or stashed.
A separate untouched baseline worktree and a dedicated migration worktree
were created under this task's `work/` directory.

The desktop directory stays at `desktop_app/` to avoid relocation conflicts
with active work. Baseline examples retain their public imports and protected
historical drivers. The test suite retains its paths and discovery settings.
Archived snapshots, JSON/CSV receipts and scientific artifacts preserve
historical paths; the manifest supplies current navigation. R2 historical
hash keys and `git cat-file` lookups retain their original identities; only
current source lookup maps into `src/`.

## Verification

Baseline: `python -m pytest -p tests.desktop_preferences --junitxml=baseline-tests.xml -ra`
with `LUNAR_OD_PROJECT_ROOT` pointing at the existing external campaign data
and `QT_QPA_PLATFORM=offscreen`.

Result: **1,690 passed, 29 skipped, 146 subtests passed**, no failures/errors,
596.98 seconds. This is 1,719 test functions/items, plus their recorded subtests.
Historical headline check counts should not be compared without separating
pytest items from subtests.

Post-migration verification is recorded below after the complete checks.

## Deferred scope

- Move `desktop_app/` to `apps/desktop/` after the owners of active worktrees
  integrate their changes.
- Further split legacy/baseline research drivers only after auditing their
  externally imported helpers and frozen source identities.
- Re-run long scientific campaigns or real-mission cross-validation only as
  separately declared scientific work; this migration claims no new science.
- Do not update a dirty local `main` checkout during synchronization. The
  dedicated migration checkout can be synchronized with remote main safely.
