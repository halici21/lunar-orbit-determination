---
name: lunar-od-software-quality-gates
description: >-
  Use for software-engineering quality of the Lunar OD repo (not the science):
  linting, formatting, type checking, coverage review, CI workflow design, flaky
  test diagnosis, test isolation, dependency hygiene, import/package validation,
  maintainability/complexity review, pre-commit/pre-push checks, and deciding
  whether a change is software-safe. File areas: python_port/lunar_od/, tests/,
  examples/, desktop_app/, and the requirements files. Use together with
  lunar-od-validation-gates (which covers scientific/numerical validation).
---

# Lunar OD Software Quality Gates

General software quality, maintainability, test quality, CI readiness, dependency
hygiene. (For scientific/numerical validation, use lunar-od-validation-gates.)

## Shared references
- Read `../_shared/software-quality-checklist.md` for general code-quality review.
- Read `../_shared/ci-quality-gates.md` when recommending CI / pre-push gates.
- Read `../_shared/test-design-guidelines.md` when reviewing or creating tests.
- Read `../_shared/dependency-hygiene-checklist.md` for dependencies / packaging /
  public-repo readiness.
- Read `../_shared/test-impact-map.md` when a change needs targeted tests.
- Read `../_shared/numerical-contract.md` when a quality change may affect numbers.

## Responsibilities
- identify affected files and tests, and available quality tools.
- determine whether a change is behavior-preserving (and flag numerical-behavior risk).
- recommend fast quality gates; recommend optional tools only when useful.
- separate style-only changes from scientific-behavior changes.

## Critical rules
- Software-quality changes must not alter scientific behavior unless the user
  explicitly requests a scientific/algorithmic change.
- Do not add new tooling configuration files (pyproject.toml, .github/workflows/,
  .pre-commit-config.yaml, ruff/mypy/coverage config) unless explicitly asked.

## Must report
software-quality issue · impacted files · suggested fix · required tests ·
optional quality tools · risk of numerical behavior change · whether safe to merge.

## Scope Boundary
Primary for lint, format, type checking, dependency hygiene, CI, maintainability, and packaging.

## Do Not Use For
- scientific/numerical validation (lunar-od-validation-gates)
- test design (lunar-od-test-strategist)
- small explanations or lookups

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
