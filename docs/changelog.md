# Changelog

## Unreleased

### Scientific model changes (roadmap phases C1–F2, PR #46, #48, #55)

- **Production planetary ephemeris is now DE440** (Phase C4-I, `C4I_QUALIFIED`,
  owner-accepted 2026-10-25). Select DE421 with `ephemeris="DE421"` or
  `LUNAR_OD_PLANETARY_EPHEMERIS=DE421`; DE421 reproduces every earlier record bit
  for bit. Record-reproducing test modules are pinned with the `de421_ephemeris`
  fixture. **Numbers computed before this change were computed on DE421.**
- Barycentric light time, Sun and Earth Shapiro delay and the TT interval for
  two-way observables (C1-I, `C1I_QUALIFIED`).
- Station time and a station-clock consider parameter (C2-I, `C2I_QUALIFIED`).
- Station coordinate realization and solid Earth tide (C3-I, `C3I_QUALIFIED`).
- Media calibration interface: troposphere (Niell mapping, IERS zenith
  hydrostatic delay) and ionosphere delays; ramped uplink stays a real-data
  convention (C5-I, `C5I_QUALIFIED`).
- Opt-in `PropagationProfile` with shadow events and estimator
  `propagation_profile` (D2, `D2_QUALIFIED`, owner-accepted 2026-10-25). With the
  profile BLS-LM closes 0.041 mm and SRIF 0.044 mm from the truth; SRIF still
  reports `MaxIter` because it has no stationarity exit after a rejected step.
  The default profile is unchanged.
- UTDF tracking reader (`lunar_od.utdf`) and first real LRO studies: two-way range
  and Doppler residuals against the LOLA precision orbit (F1, F1b, F1c) and a
  first 12 h range-only orbit determination (F2, `F2_NOT_QUALIFIED` as computed).
- Materiality surveys and characterizations that did not change production code:
  C1, C2, C3, C4, C5, D1. Reports and predeclarations are in
  `studies/phase17/reports/phase_*.md` and `studies/phase17/configs/`.

### Workspace and repository

- Replace the owner's absolute Windows paths in current documents with
  `<workspace>`, `<kernel-dir>` and `<python>` placeholders; the D25 qualification
  script resolves the campaign data through `campaign_root()`. Archived records
  under `archive/` and byte-preserved scripts keep their original text.
- Add an index of the `archive/` folders and of the tags that keep former branch
  heads; add `.mailmap` so both commit identities of the author show as one.
- GitHub keeps only `main`: finished branches are merged and deleted, and pull
  requests use merge commits only, so the commit identities cited by study
  reports stay reachable from `main`.

- Synchronize the QProcess liveness test with child ready/release signals, retaining
  five observed timer callbacks and the final partial-line flush assertion.

- Capture ground-track request settings, explain edited versus displayed results,
  and prevent analytical generation from overtaking a pending SPICE calculation.

- Retain matching ground-track data, canvas, exports and paused playback position
  after a detected render failure; explain when the previous result is retained.

- Keep the initially hidden orbit designer idle; pause ground-track playback on
  page/tab changes and window close, retaining the position for manual Play.

- Distinguish uncomputed analytical ground-track visibility from a calculated zero;
  identify SPICE versus approximate analytical sources and retain fractional days.

- Stack visibility settings and the Gantt plot in small windows; retain full
  controls and lower-axis access with vertical scrolling and no outer panning.

- Show the visibility result’s submitted settings and example-source scope; mark
  edits, lock pending requests and retain matching CSV/PNG data on failure.
- Keep catalogue map names inside the world map without overlapping other names
  or markers; retain multiword cities and support exact station selection by name.
- Sort station coordinates and uncertainties numerically at stored precision;
  preserve undefined versus zero values and add unit-bearing precision tooltips.
- Keep station identities and coordinate cells together when sorting/filtering
  the catalogue. Edit/delete/preset actions and map picks resolve the displayed
  station; rebuilding the table clears an earlier selection.
- Reduce root files from 16 to 10: group roadmap/changelog in `docs/`,
  contributing guidance in `.github/`, project instructions in `.claude/`,
  and the MkDocs configuration in `docs/`. Consolidate pytest options into
  `pyproject.toml`, preserving test discovery and worktree import isolation.
- Pause routine dependency update PRs and remove the three unmerged bot
  branches; retain dependency alerts, required review and all current pins.

## 0.1.0-alpha.4 — 2026-10-16

- Exclude generated Python/Numba caches from portable content identities and ZIPs;
  verify immutable files again after ordinary imports and desktop execution.
- Exercise the actual relocated `LunarOD.cmd --check` launcher, preserve literal
  Windows command-file line endings and quote the bundle directory safely.
- The alpha.3 source tag was retained as an unpublished candidate; its binary
  publication was cancelled after the cache-integrity defect was reproduced.

- Add a Python-bundled Windows x64 portable application and a real, data-free
  desktop tutorial with execution, cancellation and recorded-comparison checks.
- Record the actual 60-second desktop workflow and publish its source/result identities.
- Generate public API signatures/docstrings from source and document numerical units,
  array shapes, reference epochs, priors and returned covariance contracts.
- Qualify library wheels on Windows/Linux and Python 3.10–3.13; keep kernel-backed
  integration checks in their explicit external-data tier.
- Publish the predeclared 64-trial / 192-fit BLS-LM/SRIF/SR-UKF comparison, full raw
  inputs and distribution/consistency diagnostics; report SRIF MaxIter outcomes.
- Add controlled Dependabot updates, PR dependency review and isolated release
  builds that qualify source/wheel/portable assets before tagged publication.
- Normalize equivalent Windows source paths when recording tutorial identities.

- Move catalogue network colors into a wrapping key below the map, preventing
  the legend from covering station markers. Explain custom stars separately
  from network colors and keep unknown networks visible in the key.
- Group optional/development/docs/Windows dependency profiles under `requirements/`
  while retaining the standard core `requirements.txt` and Windows launch commands.
- Move the earlier Tkinter/HTTP tools and web assets to `legacy/`; document their
  new command paths and keep result/test/experiment paths in the current checkout.
- Preserve dependency specifiers, pinned versions and scientific implementation;
  add asset-serving, legacy path and relocated preflight-profile regressions.

## 0.1.0-alpha.2 — 2026-10-15

- Add a data-free, seeded BLS-LM/SRIF orbit-fit tutorial with observations,
  trajectories, residuals, PNG/SVG figures, an animated plot and source identities.
- Publish a source-linked results gallery and estimator/model support matrix;
  add searchable documentation with pull-request builds and main-only Pages deployment.
- Add bug/scientific issue forms, a PR template, roadmap and explicit source/wheel/DOI procedures.
- Add deterministic-noise and noiseless-recovery tutorial checks to the public gate.
- Distribute an installable library wheel alongside the Windows source preview.
- Preserve scientific implementation, numerical tolerances and historical study bytes.

- Add an installable `src/lunar_od` package; consolidate Phase 17 study
  scripts, configs, evidence, reports and provenance; organize durable docs
  by purpose. Preserve numerical implementation, scientific record bytes,
  pinned hashes, test discovery and historical traceability.
- Add citation metadata, artifact policy and repository integrity checks.


- Station catalogue actions stack into two groups on narrow windows, keeping
  every action accessible without horizontal page scrolling. The catalogue map
  resizes with the viewport and preserves equal longitude/latitude degree scales.
- Station catalogue search now reports visible/total and selected rows, explains
  empty results and supports Ctrl+F plus a clear button. CSV exports only the
  filtered table; an empty filter cannot export hidden catalogue entries.
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
- [Readiness implementation and verification](../docs/development/environment-readiness_2026-10-10.md).

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

Phase-specific scientific changes and validation are documented in
`studies/phase17/` and the durable `docs/` guides. [The branch consolidation report](../docs/development/branch_consolidation_2026-10-10/README.md)
records imported history and preserved pending work.
