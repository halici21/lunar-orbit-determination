---
name: lunar-od-estimator-engineering
description: >-
  Use when working on Lunar OD estimators in python_port/lunar_od/estimators.py
  (BLS-LM, SRIF), filters.py (SR-UKF), scenarios.py, and observability.py: STM
  mapping, covariance propagation, LM damping, UKF sigma points, NIS/NEES,
  observability, arc handoff / start modes, and estimator-comparison validity.
  Tests: test_estimators.py, test_filters.py, test_scenarios.py. Trigger: editing
  an estimator/filter, or judging whether a BLS-LM vs SRIF vs SR-UKF comparison is
  fair.
---

# Lunar OD Estimator Engineering

Estimator implementation integrity and comparison validity.

## Key files
- `python_port/lunar_od/estimators.py`, `filters.py`, `scenarios.py`,
  `observability.py`
- `python_port/tests/test_estimators.py`, `test_filters.py`, `test_scenarios.py`

## Shared references
- Read `../_shared/estimator-comparison-contract.md` before comparing estimators.
- Read `../_shared/numerical-contract.md` for tolerance / fair-comparison rules.
- Read `../_shared/monte-carlo-guidelines.md` when comparing robustness or
  campaign-level behavior.
- Read `../_shared/truth-model-hierarchy.md` when estimator accuracy is reported
  against a truth trajectory.

## Responsibilities
- verify estimator-comparison validity (aligned truth / measurements / seed / etc.).
- identify estimator-specific biases, observability limits, covariance concerns.
- distinguish estimator improvement from scenario, geometry, or noise changes.
- identify whether a conclusion needs Monte Carlo support.

## Critical rule
Do not compare estimators unless truth trajectory, measurement schedule, station
network, visibility windows, noise, random seed, initial state error, and
covariance assumptions are explicitly aligned (or are the controlled variable).
