# Qt Interface Guidelines

Use when a UI/UX design must map cleanly to PyQt5, Qt Designer, controllers,
services, workers, widgets, or QSS.

## Structure rules
- separate UI layout from controller logic; keep controllers thin.
- keep scientific execution **off** the UI thread (workers / QThread / QProcess).
- group controls by workflow.
- avoid hidden UI state that cannot be reconstructed from a scenario config;
  keep UI-created scenarios reproducible.

## Layout rules
Use: tabs for major result categories; splitters for resizable analysis areas;
tables (sortable/filterable) for structured results; status bars / log panes for
run feedback; left or top navigation for major sections; consistent panel titles
and action placement.
Avoid: modal-heavy frequent workflows; deeply nested panels; hidden critical
settings; long ungrouped scrolling forms; manual-only workflows not reproducible
from saved config.

## Result browser rules
Support: locating output artifacts; opening generated plots; viewing CSV/table
summaries; comparing estimators; tracing a result back to config/script/artifact;
showing run timestamp, seed, and estimator/measurement/station/force-model
summaries where useful.

## Critical rule
A PyQt5 UI is not complete just because widgets exist; the screen must preserve
scientific workflow clarity, nonblocking execution, and result traceability.
