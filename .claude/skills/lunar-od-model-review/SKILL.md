---
name: lunar-od-model-review
description: >-
  Critically review lunar OD physical assumptions, estimator consistency,
  Jacobians, covariance, and numerical risks. Verifies that measurement
  generation, predicted h(x), residual computation, and Jacobians use the same
  model; flags synthetic-data bias, missing or frame-inconsistent light-time,
  simplified Doppler, weak observability, covariance mistuning, and invalid
  estimator comparisons. Use when reviewing OD code or results for correctness.
---

# Lunar OD Model Review

Find concrete, severity-ranked problems in the physics, estimation, and numerics.

## Core consistency question (check this first)
Do **generation**, **prediction h(x)**, **residual computation**, and the
**Jacobian** use the same model? Trace the path:
- generation: `lunar_od/measurements.py`
  (`generate_position_measurements`, `generate_range_rate_measurements`),
- prediction / residuals: `compute_position_residuals[_analytic]`,
  `compute_range_rate_residuals[_analytic]`,
- range-rate physics: `lunar_od/radiometrics.py`,
- estimator Jacobians: `lunar_od/estimators.py`; filters: `lunar_od/filters.py`.

A generation/prediction model mismatch is the highest-severity finding.

## Risk checklist
- **Synthetic-data bias** — truth generated and estimated with the identical
  model hides modeling error; flag results that are only self-consistent.
- **Light-time / aberration** — present? converged? frame-consistent? (see
  `SPICE_CN_CNPLUS_VALIDATION.md`: MCI-frame light-time differs from
  SSB-inertial by the Moon's light-time displacement.)
- **Doppler realism** — instantaneous vs averaged counted Doppler; turnaround,
  transponder, and clock terms (see `DOPPLER_RANGE_RATE_MODEL_REVIEW.md`).
- **Jacobian** — analytic vs finite-difference; are neglected terms documented
  and bounded?
- **Observability / conditioning** — weak arcs, near-singular information,
  large covariance condition number.
- **Covariance tuning** — process / measurement noise, NIS/NEES consistency,
  gating.
- **Comparison validity** — same seed / network / arcs / start mode across the
  estimators being compared.

## Output format
For each finding:
- **[Severity: high / medium / low]** one-line statement,
- a `file.py:function` reference and why it is a problem,
- a **minimal, testable fix** plus the test to add or run
  (`tests/test_*.py`, then the full suite).

Be specific; avoid vague criticism. If something is correct, say so briefly.

## Do not
- Speculate without locating the code.
- Recommend a large rewrite when a small, verifiable change suffices.
- Modify source code as part of the review (review only, unless asked to fix).
