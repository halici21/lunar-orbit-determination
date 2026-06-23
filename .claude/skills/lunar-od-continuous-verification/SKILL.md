---
name: lunar-od-continuous-verification
description: Use when planning recurring test execution for the Lunar OD project, including every-commit tests, pre-push checks, pull-request gates, nightly scientific regression, manual long campaigns, release validation, thesis-freeze verification, SPICE-dependent tests, desktop smoke tests, baseline comparisons, and external cross-validation scheduling.
---

# Lunar OD Continuous Verification

## Purpose

Use this skill to decide which tests should run, when they should run, and what they prove.

This skill manages verification tiers:

- every commit
- pre-push
- pull request
- nightly
- manual scientific campaign
- release
- thesis freeze
- external validation

This skill complements:

- `lunar-od-test-strategist`, which designs individual tests.
- `lunar-od-validation-gates`, which validates scientific/numerical changes.
- `lunar-od-software-quality-gates`, which reviews CI and software-quality checks.

## Shared References

Read:

- `../_shared/continuous-verification-matrix.md` when assigning tests to tiers.
- `../_shared/slow-test-policy.md` when long campaigns or benchmarks are involved.
- `../_shared/flaky-test-policy.md` when recurring tests may be unstable.
- `../_shared/test-taxonomy.md` when categorizing tests.
- `../_shared/fixture-baseline-policy.md` when recurring tests depend on fixtures or baselines.

If these existing shared files are available, also read them when relevant:

- `../_shared/ci-quality-gates.md`
- `../_shared/test-impact-map.md`
- `../_shared/baseline-registry.md`
- `../_shared/experiment-reproducibility-checklist.md`
- `../_shared/cross-validation-contract.md`
- `../_shared/dependency-hygiene-checklist.md`

## Verification Tier Procedure

For each test or test group, determine:

1. purpose
2. runtime cost
3. required artifacts
4. SPICE/kernel requirement
5. GUI/headless requirement
6. randomness and seed requirement
7. baseline requirement
8. whether it is deterministic
9. appropriate tier
10. command or manual procedure
11. failure meaning

## Tier Guidance

Every commit:

- fast unit tests
- deterministic regression tests
- no long campaigns
- no private artifacts

Pre-push:

- full fast pytest suite
- targeted tests for changed areas
- optional quality checks if configured

Pull request:

- fast full regression
- targeted scientific checks
- artifact schema checks
- no private local paths

Nightly:

- reduced Monte Carlo
- reduced long-duration scenario
- fragmented-visibility smoke
- Doppler smoke
- selected baseline comparisons

Manual scientific campaign:

- full Monte Carlo
- full 28-day campaign
- thesis figure regeneration
- external data-heavy workflows

Release:

- full fast regression
- registered baseline comparisons
- public examples
- packaging/import checks
- README/result traceability

Thesis freeze:

- all thesis numbers traced to artifacts
- all final figures regenerated or verified
- final scenario configs saved
- final baseline status documented

External validation:

- GMAT/Orekit/Tudat/MONTE/Basilisk comparisons
- only after assumptions are explicitly aligned

## Critical Rules

- Do not put slow campaigns in default CI.
- Do not require private local paths in recurring tests.
- Do not require unavailable SPICE kernels unless the test skips gracefully.
- Do not mark scientific drift as acceptable without baseline comparison.
- Do not treat nightly/manual tests as substitutes for fast regression tests.
- Do not schedule flaky tests repeatedly without addressing the cause.

## Output Format

When used, report:

1. Test or test group
2. Purpose
3. Recommended verification tier
4. Required command or manual action
5. Expected runtime class
6. Required data/kernels/tools
7. Determinism and seed handling
8. Baseline/artifact needs
9. What failure means
10. Whether it belongs in CI, nightly, manual, release, or thesis-freeze

## Scope Boundary
Primary when scheduling recurring tests / CI / nightly / release / thesis-freeze tiers.

## Do Not Use For
- designing individual tests (lunar-od-test-strategist)
- one-off validation of a single change (lunar-od-validation-gates)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
