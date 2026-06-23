---
name: lunar-od-test-strategist
description: Use when designing, reviewing, or improving Lunar OD tests in python_port/tests/ and related source areas, including unit tests, regression tests, finite-difference checks, fixture comparisons, scenario smoke tests, estimator fairness tests, statistical consistency tests, numerical tolerances, and test-oracle selection for dynamics, measurements, estimators, filters, scenarios, desktop workflows, and performance changes.
---

# Lunar OD Test Strategist

## Purpose

Use this skill to design tests before writing or modifying them.

This skill answers:

- What behavior should be protected?
- What type of test is appropriate?
- What is the test oracle?
- What tolerance is justified?
- What fixture or baseline is needed?
- What should run in fast CI, and what should be slow/manual?

This skill complements:

- `lunar-od-validation-gates`, which decides validation gates after a change.
- `lunar-od-software-quality-gates`, which reviews software quality and CI readiness.
- `lunar-od-continuous-verification`, which assigns tests to recurring verification tiers.

## Shared References

Read:

- `../_shared/test-taxonomy.md` when choosing the test type.
- `../_shared/test-oracle-patterns.md` when choosing the expected result source.
- `../_shared/numerical-tolerance-policy.md` when selecting tolerances.
- `../_shared/fixture-baseline-policy.md` when adding or updating fixtures.
- `../_shared/flaky-test-policy.md` when diagnosing flaky tests.
- `../_shared/continuous-verification-matrix.md` when deciding whether a test belongs in fast, nightly, manual, release, or thesis-freeze verification.

If these existing shared files are available, also read them when relevant:

- `../_shared/test-design-guidelines.md`
- `../_shared/test-impact-map.md`
- `../_shared/numerical-contract.md`
- `../_shared/baseline-registry.md`
- `../_shared/measurement-physics-checklist.md`
- `../_shared/dynamics-regression-checklist.md`
- `../_shared/estimator-comparison-contract.md`
- `../_shared/ci-quality-gates.md`

## Test Strategy Procedure

Before proposing a test, identify:

1. behavior under test
2. source file and function
3. scientific risk
4. correct test type
5. oracle
6. tolerance and unit
7. fixture or baseline need
8. randomness and seed handling
9. fast vs slow tier
10. required command
11. remaining risk

## Recommended Test Categories

Use the smallest sufficient test first:

- unit test for isolated deterministic logic
- finite-difference test for Jacobians, STMs, and sensitivities
- analytical reference test when a closed-form value exists
- fixture comparison test for known external or legacy values
- fast-path vs slow-path test for optimized implementations
- scenario smoke test for end-to-end workflow health
- estimator fairness test for BLS-LM, SRIF, and SR-UKF comparisons
- statistical consistency test for NIS, NEES, or Monte Carlo behavior
- artifact schema test for CSV, PNG, and result output contracts
- desktop smoke test for UI workflow health

## Critical Rules

- Do not write tests that only prove code runs.
- Do not duplicate implementation logic exactly as the oracle.
- Do not loosen tolerances to hide failures.
- Do not use uncontrolled randomness.
- Do not run long campaigns as default tests.
- Do not claim scientific protection from a smoke test.
- Do not add or update baselines without explaining the source and reason.
- Do not hide model changes as floating-point differences.

## Output Format

When used, report:

1. Changed behavior or feature
2. Scientific/software risk
3. Proposed tests
4. Test type for each test
5. Oracle for each test
6. Tolerance and unit
7. Fixture/baseline need
8. Random seed handling
9. CI/verification tier
10. Commands to run
11. What not to test
12. Remaining risk

## Scope Boundary
Primary when designing, reviewing, or repairing tests (type, oracle, tolerance, fixture).

## Do Not Use For
- scheduling/tiering tests (lunar-od-continuous-verification)
- post-change validation gating (lunar-od-validation-gates)
- software-quality-only checks (lunar-od-software-quality-gates)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
