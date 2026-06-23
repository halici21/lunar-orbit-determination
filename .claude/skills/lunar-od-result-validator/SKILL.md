---
name: lunar-od-result-validator
description: >-
  Use for scientific interpretation and acceptance review of Lunar OD results:
  residual analysis (bias / trend / autocorrelation / whiteness), convergence,
  covariance credibility, NIS/NEES interpretation, estimator-comparison review,
  operational-success review, and the final science-review decision. Trigger:
  judging whether a result is plausible, fair, statistically defensible, and what
  truth-model level its accuracy claim uses.
---

# Lunar OD Result Validator

Scientific interpretation and acceptance review of results.

## Shared references
- Read `../_shared/estimator-comparison-contract.md` when reviewing a comparison.
- Read `../_shared/numerical-contract.md` for fair-comparison rules.
- Read `../_shared/science-review-checklist.md` for the acceptance questions.
- Read `../_shared/truth-model-hierarchy.md` for the accuracy claim's level.
- Read `../_shared/monte-carlo-guidelines.md` for stochastic / campaign results.

## Must evaluate
residual bias / trend / autocorrelation / whiteness; covariance credibility;
NIS / NEES consistency; station-geometry, measurement-diversity, visibility, and
observability limitations; numerical conditioning; statistical significance;
Monte Carlo survivability.

## Must answer
Is the result plausible? reproducible? Is the comparison fair? Is the covariance
credible? Is the claimed improvement real and statistically defensible? Would it
survive Monte Carlo? What truth-model level supports the accuracy claim? Is the
tail behavior explained by geometry, conditioning, measurement count, visibility,
or estimator design? Conclude with a `science-review-checklist.md` label.

## Scope Boundary
Primary when interpreting or accepting numerical results, residuals, NIS/NEES, or a comparison claim.

## Do Not Use For
- tasks with no numerical result to interpret
- tracing provenance (lunar-od-result-reproducer)
- choosing/designing tests (lunar-od-validation-gates / test-strategist)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
