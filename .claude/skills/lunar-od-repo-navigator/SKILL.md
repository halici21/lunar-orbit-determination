---
name: lunar-od-repo-navigator
description: >-
  Use to locate code, trace data flow, explain architecture, map a result back to
  the script that produced it, or identify which modules/tests a change will
  affect — across python_port/lunar_od/, examples/, tests/, desktop_app/, and
  results/. Trigger phrases: "where is", "which file", "what calls", "which script
  produced this figure/CSV", "what does this module do", "what will this change
  affect". Use this before proposing edits, to find the right module, function,
  test, and result artifact.
---

# Lunar OD Repo Navigator

Repository awareness and architecture navigation for the Lunar OD project.

## Use for
- locating code and functions
- tracing data flow (`config -> scenario -> measurements -> estimation -> results`)
- explaining architecture and module boundaries
- mapping a result artifact back to its producing script / config
- identifying affected modules and tests before an edit

## Must know
- `python_port/lunar_od/` — core scientific package
- `python_port/examples/` — reproducible experiment scripts
- `python_port/tests/` — unit / regression tests
- `python_port/desktop_app/` — PyQt5 interface
- `python_port/results/` — generated artifacts (not hand-authored)

## Shared references
- Read `../_shared/repo-map.md` when locating files or explaining module ownership.
- Read `../_shared/result-artifact-map.md` when tracing a result to its source.

## Critical rule
Do not suggest edits before locating the relevant module, function, test, and (if
applicable) result artifact.
