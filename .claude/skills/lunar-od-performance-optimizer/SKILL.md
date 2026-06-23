---
name: lunar-od-performance-optimizer
description: >-
  Use for behavior-preserving computational optimization of Lunar OD:
  vectorization, Numba acceleration, STM caching, parallel arc execution,
  profiling, and runtime comparison, including performance items from
  python_port/plan.md. File areas: plan.md, lunar_od/dynamics.py, accelerated.py,
  measurements.py, estimators.py, filters.py, scenarios.py. Trigger: making code
  faster without changing physics. Always compare optimized output against
  baseline before claiming behavior preservation.
---

# Lunar OD Performance Optimizer

Behavior-preserving computational optimization.

## Key files
- `python_port/plan.md`
- `python_port/lunar_od/dynamics.py`, `accelerated.py`, `measurements.py`,
  `estimators.py`, `filters.py`, `scenarios.py`

## Shared references
- Read `../_shared/numerical-contract.md` — optimization must preserve behavior.
- Read `../_shared/test-impact-map.md` to pick the regression tests to run.
- Read `../_shared/baseline-registry.md` when optimization may affect numerical
  outputs or runtime baselines.
- Read `../_shared/software-quality-checklist.md` when optimization affects
  maintainability.

## Responsibilities
- identify a behavior-preserving optimization path.
- compare optimized output against baseline before claiming behavior preservation.
- avoid mixing performance changes with physical-model changes.

## Critical rules
- Performance changes must preserve scientific behavior unless the user explicitly
  asks for a physical / algorithmic model change.
- Optimization must not make code untestable, unmaintainable, or dependent on
  hidden global state. Do not run long runtime benchmarks by default.
