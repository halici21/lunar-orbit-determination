# Development documentation

Procedures for working on the repository, followed by dated records. The records are kept
under their original names because tests, study reports and frozen records cite them;
read the dated files as history, not as current instructions.

## Current procedures

| Document | Use it for |
|---|---|
| [Repository architecture and artifact policy](repository-layout.md) | What lives where, what is committed and what is generated |
| [Path manifest](path-migration.json) | Current location and preserved byte identity of every migrated file |
| [Releases and DOI](releases.md) | Release gate, automated publication, Zenodo account step |
| [Maintenance and verification automation](maintenance.md) | Recurring checks and dependency upkeep |
| [Root layout cleanup](root-layout-cleanup.md) | Old root commands and their current equivalents |
| [Performance optimization plan](plan.md) | Open performance items |

## Desktop application

- [Desktop result workflow](../../desktop_app/RESULT_WORKFLOW.md): the six pages and how earlier page names map to them.
- Design briefs from the first desktop iterations: [PyQt5 brief](PYQT5_DESKTOP_APP_BRIEF.md),
  [desktop app brief](PYQT5_DESKTOP_APP_README.md), [interface design notes (Türkçe)](UI_README.md).

### Desktop change records

Each record explains one UI change and has a machine-readable `*-receipt*.json` beside it
with the commands, screenshots and checks that verified it.

| Date | Area | Record |
|---|---|---|
| 2026-10-25 | Orbit preview | [Pause the orbit camera for inspection](orbit-camera-control_2026-10-25.md) |
| 2026-10-23 | Orbit preview | [Skip camera frames outside the viewport](orbit-preview-visibility_2026-10-23.md) |
| 2026-10-22 | Orbit designer | [Visible keyboard focus on sliders](orbit-slider-focus_2026-10-22.md) |
| 2026-10-22 | Orbit designer | [Parameter names stay visible with keyboard focus](orbit-designer-context_2026-10-22.md) |
| 2026-10-21 | Orbit designer | [Complete keyboard focus](orbit-designer-keyboard_2026-10-21.md) |
| 2026-10-21 | Orbit designer | [Clear analytical inputs](orbit-designer-guidance_2026-10-21.md) |
| 2026-10-21 | Ground track | [Keyboard access to SPICE surface-track settings](ground-track-keyboard_2026-10-21.md) |
| 2026-10-20 | Ground track | [First-use guidance and time labels](ground-track-guidance_2026-10-20.md) |
| 2026-10-20 | Ground track | [Compact controls](ground-track-density_2026-10-20.md) |
| 2026-10-19 | Ground track | [Plot legibility](ground-track-legibility_2026-10-19.md) |
| 2026-10-19 | Ground track | [Responsive workspace](ground-track-responsive_2026-10-19.md) |
| 2026-10-18 | Ground track | [Request context and model ordering](ground-track-request-context_2026-10-18.md) |
| 2026-10-18 | Ground track | [Result consistency on render failure](ground-track-result-pair_2026-10-18.md) |
| 2026-10-18 | Ground track | [Animation timer lifecycle](ground-track-timers_2026-10-18.md) |
| 2026-10-18 | Ground track | [Model and visibility labels](ground-track-models_2026-10-18.md) |
| 2026-10-17 | Visibility | [Responsive workspace](visibility-responsive_2026-10-17.md) |
| 2026-10-16 | Visibility | [Result context and consistent exports](visibility-context_2026-10-16.md) |
| 2026-10-16 | Stations | [Map labels and exact name selection](station-map-labels_2026-10-16.md) |
| 2026-10-16 | Stations | [Numeric ordering](station-numeric-sort_2026-10-16.md) |
| 2026-10-16 | Stations | [Row identity](station-sorting_2026-10-16.md) |
| 2026-10-15 | Stations | [Map key](station-legend_2026-10-15.md) |
| 2026-10-15 | Stations | [Layout](station-layout_2026-10-15.md) |
| 2026-10-13 | Stations | [Search and export](station-filters_2026-10-13.md) |
| 2026-10-12 | Mission view | [Map proportions and labels](map-layout_2026-10-12.md) |
| 2026-10-12 | Mission view | [Orbit label readability](orbit-labels_2026-10-12.md) |
| 2026-10-12 | Mission view | [First-run guidance](mission-onboarding_2026-10-12.md) |
| 2026-10-11 | Mission view | [Status iteration](mission-status_2026-10-11.md) |
| 2026-10-11 | Mission view | [Usability iteration](mission-usability_2026-10-11.md) |
| 2026-10-11 | Mission view | [Overview](mission-overview_2026-10-11.md) |

The 2026-10-23 to 2026-10-25 simplification (six Turkish pages, Modern palette, archive tabs)
is described in the [changelog](../changelog.md) and the desktop result workflow above.

## Historical repository records

| Date | Record |
|---|---|
| 2026-10-25 | [Retired branch-history tags](branch_consolidation_2026-10-10/retired_tags_2026-10-25.json) (every tip is in `main`) |
| 2026-10-15 | [Repository professionalization](repository-professionalization.md) and its [verification record](repository-professionalization-verification.json) |
| 2026-10-10 | [Branch consolidation](branch_consolidation_2026-10-10/README.md), [main workspace readiness](main_workspace_readiness_2026-10-10.md), [public repository readiness](repository-readiness_2026-10-10.md), [desktop environment readiness](environment-readiness_2026-10-10.md) |
| 2026-09-24 | [Local repository hygiene](local_repository_hygiene_status_report.md) with its inventories in `repository_hygiene/` |
| 2026-09-14 | [Git phase P2 canonical integration](git_phase_p2_canonical_integration_report.md) |
| 2026-07-11 | [Project context](PROJECT_CONTEXT.md) (early workspace notes) |
