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

The current-main station-layout commit `6f70e29` arrived during the migration.
It was merged into the migration without changing its desktop implementation
or test content. Its new report and receipt were placed in development docs.
The initially modified canonical-main files had been independently committed
and pushed by their owning session before this integration; no dirty work was
staged, reset or overwritten by the migration.

Post-migration validation used a fresh local clone at `58caeff`, an actual
`tools/setup_windows.ps1` installation, and the same external-data overrides.
The source/runtime/test changes were final at that commit; subsequent changes
record this evidence and broaden documentation checking only.

| Check | Result |
| --- | --- |
| Full clean-clone regression | **1,694 passed, 146 subtests passed, 29 skipped**, no failures/errors, 494.91 s |
| Discovery identity comparison | No baseline test identities removed; four station-layout cases added by `6f70e29` |
| Public gate with isolated external inputs | **279 passed**, no failures/errors/skips |
| Scientific preservation | 375 byte-identity gates; all 991 original tracked files retained |
| Numeric-literal audit against integrated main | No scientific or test numeric literals changed by the migration |
| Repository documentation checker | 86 current Markdown documents, 141 local links, all valid |
| Packaging | Editable install and wheel build pass; all 55 library modules present; no study/app modules in wheel |
| Independent wheel environment | Installed wheel imports outside the source checkout; `pip check` passes |
| Development entry points | Example schema, minimal geometry example and Phase 17 CLI help pass |
| Source hygiene | Compileall and Git whitespace checks pass; generated data/caches remain ignored |
| Dependency graph | Refreshed successfully as an ignored per-worktree discovery artifact |

Commands included `python -m pytest -p tests.desktop_preferences -ra`,
`python tools/run_public_checks.py`, `python tools/check_repository.py`,
`python -m pip wheel --no-deps .`, `python -m pip check`, and
`python -m compileall -q src examples studies/phase17/scripts tools tests`.
The [machine-readable verification receipt](repository-professionalization-verification.json)
records exact counts, runtime versions, test-identity differences and JUnit
SHA-256 values. JUnit aggregate counts include subtests; they are not additional
named pytest items.

The validation repairs changed path lookup and historical/current identity
resolution only. The first migration run exposed stale package-root lookups
and an external snapshot lookup that would have introduced four extra skips;
those were corrected. The final clean run retains the baseline's 29 skips.
No numerical expectation, tolerance or scientific decision was relaxed.

**Scientific behavior expected: NONE changed.** Numerical implementation and
scientific-result impact are **NONE**. Existing synthetic qualification tests
passed; no long campaign, new baseline, Monte Carlo study or real-mission
accuracy claim was introduced. Historical frozen-source comments, archives
and immutable machine-readable records retain their original paths through
the migration manifest rather than receiving new scientific identities.

## Deferred scope

- Move `desktop_app/` to `apps/desktop/` after the owners of active worktrees
  integrate their changes.
- Further split legacy/baseline research drivers only after auditing their
  externally imported helpers and frozen source identities.
- Re-run long scientific campaigns or real-mission cross-validation only as
  separately declared scientific work; this migration claims no new science.
- Do not update a dirty local `main` checkout during synchronization. The
  dedicated migration checkout can be synchronized with remote main safely.
