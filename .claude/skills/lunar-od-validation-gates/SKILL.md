---
name: lunar-od-validation-gates
description: >-
  Use to plan scientific/numerical validation for a Lunar OD change: choose
  targeted tests, recommend full regression, identify impacted metrics and
  baselines, assess numerical drift, and judge whether a tolerance update is
  justified. Trigger: any code or scientific-model change, tolerance update,
  regression planning, or writing a validation summary. For software-engineering
  quality (lint, types, CI, coverage, dependency hygiene, flaky tests) use
  lunar-od-software-quality-gates instead.
---

# Lunar OD Validation Gates

Scientific and numerical validation planning and reporting.

## Use for
code changes, scientific-model changes, tolerance updates, regression planning,
validation summaries, numerical-drift analysis, baseline comparisons.

## Shared references
- Read `../_shared/test-impact-map.md` to choose targeted tests.
- Read `../_shared/validation-report-template.md` to write the summary.
- Read `../_shared/numerical-contract.md` for tolerance / fair-comparison rules.
- Read `../_shared/baseline-registry.md` when numerical outputs may change.
- Read `../_shared/science-review-checklist.md` when validating a scientific
  conclusion.

## Must provide
- targeted test command
- full regression command where appropriate (`python -m pytest python_port/tests/ -q`)
- impacted metrics and impacted baseline scenarios
- numerical risks / expected sensitivity
- a validation summary using the template

## Critical rule
Never recommend a scientific code change without identifying the required
regression tests, and never weaken a tolerance without documented justification.

## Scope Boundary
Primary when accepting a code/science change and selecting the required tests and baseline checks.

## Do Not Use For
- small explanations or lookups (answer directly)
- software-engineering quality (lunar-od-software-quality-gates)
- designing the tests themselves (lunar-od-test-strategist)
- interpreting results (lunar-od-result-validator)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
