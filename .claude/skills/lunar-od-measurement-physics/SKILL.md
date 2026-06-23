---
name: lunar-od-measurement-physics
description: >-
  Use when reviewing or modifying Lunar OD measurement models in
  python_port/lunar_od/measurements.py and radiometrics.py (and
  measurement_ingestion.py): range, azimuth/elevation, range-rate, two-way counted
  Doppler, light-time correction, stellar aberration, station motion, residuals,
  and analytic Jacobians. Tests: test_measurements.py,
  test_doppler_model_review.py, test_stellar_aberration_vv.py. Trigger: changing a
  measurement/observable, residual, or Jacobian, or validating measurement physics.
---

# Lunar OD Measurement Physics

Measurement-model correctness.

## Key files
- `python_port/lunar_od/measurements.py`, `radiometrics.py`,
  `measurement_ingestion.py`
- `python_port/tests/test_measurements.py`, `test_doppler_model_review.py`,
  `test_stellar_aberration_vv.py`

## Shared references
- Read `../_shared/measurement-physics-checklist.md` for the per-change checklist.
- Read `../_shared/numerical-contract.md` for tolerance / fair-comparison rules.
- Read `../_shared/cross-validation-contract.md` when validating measurements
  against external software or an independent implementation.

## Responsibilities
- verify sign, frame, timing, light-time, aberration, and unit consistency.
- distinguish instantaneous range-rate from two-way counted Doppler.
- verify physical equivalence across implementations / fast paths.
- check finite-difference agreement for Jacobians where possible.

## Critical rule
Measurement generation, prediction h(x), residual computation, and Jacobian logic
must use consistent physical assumptions.

## Scope Boundary
Primary for measurement / observable / residual / Jacobian models.

## Do Not Use For
- dynamics or frames (lunar-od-dynamics-spice)
- estimator internals (lunar-od-estimator-engineering)
- interpreting a finished result (lunar-od-result-validator)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
