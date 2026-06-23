# Test Design Guidelines

Test quality and maintainability.

## Good tests are
deterministic, isolated, fast by default, meaningful, traceable to behavior, and
explicit about units, tolerance, and physical assumptions.

## Scientific test types
- unit tests for small functions
- regression tests for protected numerical behavior
- finite-difference checks for Jacobians / STMs
- accelerated vs non-accelerated consistency tests
- scenario tests for end-to-end workflows
- desktop-app smoke tests
- artifact-reproducibility tests when practical

## Avoid
- dependence on local absolute paths
- long campaigns required by default
- uncontrolled randomness
- exact floating-point equality without reason
- tests that only check that code runs
- broad tolerances that hide failures
- duplicating implementation logic exactly

## Critical rule
Tests should protect scientific behavior, not merely increase coverage numbers.
