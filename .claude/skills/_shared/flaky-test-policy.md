# Flaky Test Policy

How to identify and fix flaky tests.

## Common causes
uncontrolled randomness; missing seed; time / order dependency; platform-specific
floating-point behavior; SPICE kernel availability; local absolute paths; external
network dependency; parallel-execution assumptions; GUI timing assumptions; overly
tight or overly broad tolerance; hidden global state.

## Required response
- reproduce if possible
- identify the nondeterminism source
- isolate the seed or external state
- remove local-path assumptions
- separate slow / external tests from fast tests
- fix the cause instead of suppressing the test

## Rules
- Do not delete flaky tests just because they are annoying.
- Do not skip tests without documenting the condition.
- Do not broaden tolerances before understanding the failure.
- Do not hide scientific instability as "flakiness".

## Critical rule
A flaky scientific test may reveal a real numerical or reproducibility problem.
