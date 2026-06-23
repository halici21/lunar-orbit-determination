---
name: lunar-od-result-reproducer
description: >-
  Use to trace any Lunar OD result back to its source. Trigger questions: "which
  test produced this?", "which script produced this figure?", "where does this
  slide/table/README number come from?", or verifying thesis figures,
  presentation claims, README metrics, and protected baseline results. Maps a
  claim to script -> config -> artifact (under python_port/results/ and
  examples/) -> seed -> commit.
---

# Lunar OD Result Reproducer

Trace every result back to its source.

## Use for
- "which test/script produced this figure or number?"
- README metrics, thesis figures, presentation claims
- protected baseline results

## Shared references
- Read `../_shared/result-artifact-map.md` for the traceability chain.
- Read `../_shared/experiment-reproducibility-checklist.md` for the fields a
  reproducible experiment must record.
- Read `../_shared/baseline-registry.md` when the result belongs to a protected
  baseline scenario.

## Responsibilities
- trace result to script, config, artifact, and commit when possible.
- classify the result as protected baseline, thesis figure, README metric, or
  exploratory output.

## Critical rule
Every result should be traceable `claim -> figure/table -> artifact -> config ->
script -> seed -> commit` whenever possible. State explicitly any link that is
unknown; do not invent it.
