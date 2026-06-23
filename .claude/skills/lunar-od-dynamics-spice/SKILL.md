---
name: lunar-od-dynamics-spice
description: >-
  Use when working on lunar dynamics, propagation, SPICE, or reference frames:
  Moon point-mass and Earth/Sun third-body gravity, J2/Jn, STM propagation, Numba
  acceleration of dynamics, SPICE kernels and ephemerides, MCI / body-fixed /
  ECEF / J2000 / ITRF93 frame logic, and external dynamics cross-validation. File
  areas: python_port/lunar_od/dynamics.py, accelerated.py, ephemeris.py,
  spice_loader.py and tests test_dynamics.py, test_ephemeris.py,
  test_spice_loader.py. Trigger: editing force models, propagation, STM, frames,
  or SPICE handling.
---

# Lunar OD Dynamics & SPICE

Dynamics, propagation, SPICE, reference frames, force models, STM.

## Key files
- `python_port/lunar_od/dynamics.py`, `accelerated.py`, `ephemeris.py`,
  `spice_loader.py`
- `python_port/tests/test_dynamics.py`, `test_ephemeris.py`,
  `test_spice_loader.py`

## Shared references
- Read `../_shared/numerical-contract.md` for fair-comparison and tolerance rules.
- Read `../_shared/dynamics-regression-checklist.md` before/after force-model or
  propagation changes.
- Read `../_shared/cross-validation-contract.md` when comparing dynamics against
  GMAT / Orekit / Tudat / MONTE / Basilisk / SPICE utilities.
- Read `../_shared/truth-model-hierarchy.md` when discussing propagation truth or
  accuracy.

## Responsibilities
- support external dynamics cross-validation workflows.
- identify force-model alignment requirements (frames, epochs, time scales,
  constants, ephemeris sources).
- separate physics changes from implementation (e.g. Numba) changes.

## Critical rule
Do not modify force-model behavior without identifying the regression tests and
the numerical comparison metrics that protect it.

## Scope Boundary
Primary when the task concerns dynamics, propagation, STM, reference frames, or SPICE.

## Do Not Use For
- measurement models (lunar-od-measurement-physics)
- estimator internals (lunar-od-estimator-engineering)
- plain file lookup (lunar-od-repo-navigator)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
