# Changelog

## Unreleased

- Mission map preserves its 2:1 equirectangular proportions when resized.
  Bounded station labels avoid collisions and support exact name selection;
  explicit selection is retained for subpixel marker pairs.
- Orbit legends, interaction instructions and recorded frame/epoch labels use
  separate wrapped bands. Enlarged orbit geometry is clipped away from text;
  playback retains its cached drawing and nearest-record explanation.
- First-run mission plots now explain how to open a run; missing-data messages
  appear only for loaded results. The same contextual explanations are exposed
  through Qt accessibility descriptions.
- Mission view now exposes recorded process/arc-success status and explicitly
  marks convergence as unverified. A direct action opens the loaded result's
  inspector without reopening the source files.
- Linked station dropdown/map/Gantt row selection with keyboard access and
  explicit current-epoch measurement status. Mission plots stack on narrow
  screens, avoiding the previous fixed 960-pixel page width.
- Mission overview with recorded orbit playback, clickable Earth stations and a
  synchronized measurement Gantt. Station, visibility and surface-track pages
  are now primary navigation tools with visible purpose descriptions.
- New native runs persist hashed station/measurement geometry; legacy runs keep
  explicit missing-data states. Orbital descriptors use only recorded lunar GM.
- Expandable orbital help, cached drawing and scrollable small-window controls.
- Cancel the owned startup timer on window close, preventing delayed preflight
  callbacks after closure.


- Asynchronous startup environment report with runtime import/profile checks and
  canonical SPICE kernel inventory/load/sample-epoch validation.
- Persistent desktop kernel selection applied to subsequent child processes.
- Scenario Builder checks readiness before creating a job; failed probes open the
  report without starting computation. Timeout, shutdown and inconsistent reports
  cannot trigger a pending solver launch.
- Fourteen readiness/GUI/process regression cases added to the public gate.
- [Readiness implementation and verification](docs/environment-readiness_2026-10-10.md).

These changes are on main; the existing 0.1.0-alpha.1 source asset is immutable
and predates this feature.

## 0.1.0-alpha.1 — 2026-10-10

First documented Windows source prerelease. Research qualification reports keep
their original version/phase identities; this version describes the desktop
source distribution.

- Native scenario, process monitor, recorded results, archive, derivation,
  comparison/export/reopen and normal-close session workflow.
- Modern and Orchid appearance with expandable advanced tools.
- Empty first-run archive guidance.
- Pinned Windows/Python 3.13 reference environment.
- Portable setup_windows.cmd and launch_lunar_od.cmd entry points.
- GitHub public regression gate using committed fixtures without private data.
- Short root README, documentation index, contribution guidance and legacy labels.
- Historical branches consolidated into main and retained through annotated tags.

Known limits: Python must be installed separately; SPICE kernels and historical
campaign/real gravity data are external. Short workflow runs do not qualify
estimator convergence. Long real-run desktop acceptance and standalone EXE
packaging remain outstanding.

## Historical research

Phase-specific scientific changes and validation are documented in docs/ and
artifacts/. [The branch consolidation report](docs/branch_consolidation_2026-10-10/README.md)
records imported history and preserved pending work.
