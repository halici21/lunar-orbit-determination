---
name: lunar-od-desktop-app
description: >-
  Use when implementing the Lunar OD PyQt5 desktop app in python_port/desktop_app/:
  UI pages, controllers, services, workers, QThread/QProcess, scenario builder,
  result browser, plotting integration, and QSS/theme consistency. Trigger:
  wiring a designed screen into PyQt5, connecting controllers/services/workers,
  keeping execution non-blocking, or integrating .ui files with backend services.
  For layout / visual hierarchy / UX / screenshot review, use
  lunar-od-interface-designer first.
---

# Lunar OD Desktop App

PyQt5 desktop application architecture and implementation.

## Relationship to interface design
Use `lunar-od-interface-designer` first for layout, visual hierarchy, screen
design, workflow ergonomics, scientific usability, and screenshot review. Use this
skill to implement that design in PyQt5, connect controllers/services/workers,
preserve non-blocking execution, and integrate `.ui` files with backend services.

## Key files
`python_port/desktop_app/main.py`, `controllers/`, `services/`, `workers/`,
`ui/`, `styles/`.

## Shared references
- Read `../_shared/repo-map.md` when locating UI/backend boundaries.
- Read `../_shared/test-impact-map.md` when UI changes affect scenario execution.
- Read `../_shared/experiment-reproducibility-checklist.md` when desktop workflows
  generate experiments / results.
- Read `../_shared/result-artifact-map.md` when result-browser behavior affects
  traceability.
- Read `../_shared/software-quality-checklist.md` for UI/backend coupling and
  worker-design maintainability.
- Read `../_shared/ui-design-system.md`, `../_shared/desktop-ux-workflow.md`,
  `../_shared/qt-interface-guidelines.md`, `../_shared/ui-review-checklist.md`
  when implementation must preserve the design system, workflow, Qt structure, or
  pass screen review.

## Responsibilities
- preserve UI-to-backend scenario reproducibility.
- prevent result-browser changes from breaking traceability.
- keep long-running scientific work in workers / QThread / QProcess, not the UI thread.

## Critical rules
- Long-running scientific tasks must never block the UI thread.
- Desktop workflows must remain testable without full manual UI interaction.
