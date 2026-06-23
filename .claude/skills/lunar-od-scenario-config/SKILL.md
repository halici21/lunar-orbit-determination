---
name: lunar-od-scenario-config
description: >-
  Use when defining or editing Lunar OD scenarios and their JSON/config workflow
  in python_port/lunar_od/scenario_config.py and scenarios.py, run via
  examples/run_scenario_config.py: scenario configs, station networks, measurement
  selection, estimator settings, start modes, noise settings, and UI-to-backend
  scenario mapping. Trigger: creating/validating a scenario config, or ensuring a
  UI-created scenario is reproducible. Tests: test_scenario_config.py.
---

# Lunar OD Scenario Config

Scenario reproducibility and the JSON / config workflow.

## Key files
- `python_port/lunar_od/scenario_config.py`, `scenarios.py`
- `python_port/examples/run_scenario_config.py`
- `python_port/tests/test_scenario_config.py`

## Shared references
- Read `../_shared/experiment-reproducibility-checklist.md` for required fields.
- Read `../_shared/numerical-contract.md` for fair-comparison / seed rules.
- Read `../_shared/test-impact-map.md` to pick targeted tests after config changes.
- Read `../_shared/baseline-registry.md` when a scenario change affects a
  protected baseline.

## Responsibilities
- ensure scenario configs are JSON-serializable.
- ensure physical and statistical assumptions are explicit.
- ensure seeds are explicit when stochastic measurements are generated.
- ensure UI-created scenarios can be reproduced from saved config files.

## Critical rule
Scenarios must be serializable, reproducible, and explicit about physical and
statistical assumptions.

## Scope Boundary
Primary for scenario / config definition, serialization, and reproducibility.

## Do Not Use For
- estimator internals (lunar-od-estimator-engineering)
- result interpretation (lunar-od-result-validator)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
