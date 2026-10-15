# Repository architecture and artifact policy

Only the importable `lunar_od` package belongs under `src/`. A normal editable
install supplies imports to examples, study drivers and tests. The wheel
contains the library and license; research evidence, fixtures and desktop
resources are distributed through the source repository.

The existing flat `tests/` suite remains intact: tests share helpers and have
longstanding node identities. [Test categories](../../tests/README.md) document
the semantic groups without introducing discovery churn.

`desktop_app/` remains a separate application. Two active worktrees contained
uncommitted desktop edits during this migration, so its move to `apps/desktop/`
was deferred. Its launchers, runtime probes and package paths support the new
library installation. Its established internal top-level imports are retained.

`examples/` contains baseline reproduction drivers and user-facing entry
points. Some historical drivers have protected source hashes or external
campaign imports. The Phase 17 drivers have their own study namespace; further
splitting baseline examples requires a separate compatibility audit.

Durable documentation is split into `docs/theory/`, `docs/validation/`,
`docs/research/`, and `docs/development/`. Phase-specific reports stay with the
study. Desktop screenshots remain shared documentation assets.

The roadmap, changelog and MkDocs configuration also live in `docs/`.
Contribution guidance is in `.github/CONTRIBUTING.md`; project instructions
are in `.claude/CLAUDE.md`. Pytest options share `pyproject.toml` with package
metadata, preserving worktree-local discovery without a second root config.
The [root cleanup guide](root-layout-cleanup.md) records the current paths and
the ten retained root files.

Optional dependency profiles are grouped under `requirements/`. The standard
root `requirements.txt`, editable installation and Windows setup/launch entry
points remain in place. See the [profile index](../../requirements/README.md).
Earlier Tkinter/HTTP implementations and web assets live under `legacy/`;
[their command mapping](../../legacy/README.md) documents the changed paths.

## Artifact classes

| Class | Policy | Location |
| --- | --- | --- |
| Source/config/predeclaration | Track; preserve pinned rules and hashes | `src/`, study `scripts/` and `configs/` |
| Authoritative scientific evidence | Track; preserve values and provenance | study `results/`, `reports/`, `provenance/`, `fixtures/`, `tests/fixtures/` |
| Large reproducible output | Ignore by default; promote only cited/required evidence | root `results/`, `outputs/`, study `generated/` |
| Cache/log/scratch/build | Ignore | local environments, Python/Numba caches, logs, build/dist |

Several existing Phase 17 raw result records are a few megabytes and support
published reports and regression checks. They remain tracked: a filename
containing `raw` does not establish that a record is disposable. No committed
scientific record or regression fixture was deleted by this migration.

Original repository-hygiene inventories are immutable development evidence in
`docs/development/repository_hygiene/`; their utilities live in
`tools/repository_hygiene/`. Preserved archives and their embedded paths are
historical records. Resolve former locations using [path-migration.json](path-migration.json).

`data/` documents external inputs; kernels, gravity fields and campaign source
data are not redistributed. [Data policy](../../data/README.md) lists overrides.

## Provenance gates

R2 historical path keys, filtered-checkout hashes and commit lookups remain
unchanged. Current-tree source lookups map `lunar_od/...` to `src/lunar_od/...`.
This is a location translation, not a scientific re-freeze. Protected dynamics,
measurements, estimator code and approved desktop bridge retain their bytes.
Report paths embedded in frozen source docstrings are historical qualification
references; resolve them through the manifest rather than modifying the source
identity to refresh a comment.
Use the reference checkout line-ending profile (`core.autocrlf=true`) when
running historical filtered-checkout SHA-256 gates.
