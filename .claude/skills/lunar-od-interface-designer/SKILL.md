---
name: lunar-od-interface-designer
description: >-
  Use when designing Lunar OD desktop UI/UX in python_port/desktop_app/:
  dashboard, scenario builder, dynamics / station-visibility / measurement /
  estimator configuration pages, run monitor, result browser, estimator
  comparison, and diagnostics views — plus layout, visual hierarchy, scientific
  usability, and screenshot review. This skill is for design and UX decisions; for
  PyQt5 implementation (controllers, services, workers, QThread/QProcess, .ui
  integration) use lunar-od-desktop-app. Do not modify implementation files unless
  explicitly asked.
---

# Lunar OD Interface Designer

UI/UX design for the Lunar OD desktop application and related visual workflows.

## K-Dense inspiration (design only)
- `infographics` — visual hierarchy / information density.
- `scientific-schematics` — scientific workflow diagrams.
- `scientific-visualization` — plot readability.
- `scientific-slides` / `pptx` — only for presentation outputs, not app screens.
Do **not** use AIDA-style marketing structure for core app screens. AIDA may be
used only for public README / profile / landing-page material.

## Key files
`python_port/desktop_app/ui/`, `controllers/`, `widgets/`, `styles/`, `services/`,
and `python_port/results/`.

## Shared references
- Read `../_shared/ui-design-system.md` for visual style.
- Read `../_shared/desktop-ux-workflow.md` for user flow.
- Read `../_shared/qt-interface-guidelines.md` when the design must map to PyQt5.
- Read `../_shared/ui-review-checklist.md` when reviewing a screen / screenshot.
- Read `../_shared/result-artifact-map.md` for result browser / diagnostics pages.
- Read `../_shared/experiment-reproducibility-checklist.md` when UI changes affect
  scenario creation or run reproducibility.

## Output format (when designing a screen)
1. Screen purpose  2. Target user workflow  3. Layout structure  4. Primary actions
5. Secondary actions  6. Required state indicators  7. Required scientific metadata
8. Empty/loading/running/success/warning/failure states  9. Traceability elements
10. Plot/table/log hierarchy  11. PyQt5 implementation notes (only if useful).

## Critical rules
- The UI must support OD workflows clearly and reproducibly; it must not become a
  decorative dashboard that hides assumptions or provenance.
- A screen is not acceptable only because it looks modern — it must make scientific
  state, assumptions, execution status, diagnostics, and provenance easier to grasp.
- Propose structure/layouts/workflows/review comments only; do not modify PyQt5
  implementation files unless the user explicitly asks for implementation.

## Scope Boundary
Primary for UI/UX design, layout, visual hierarchy, and screenshot review.

## Do Not Use For
- PyQt5 implementation wiring (lunar-od-desktop-app)
- backend scientific logic (domain skills)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
