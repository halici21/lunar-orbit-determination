# UI Review Checklist

Use when reviewing a Lunar OD desktop screen, screenshot, wireframe, or UI proposal.

## Screen clarity
- Is the page purpose clear within ~3 seconds?
- Are primary actions visible and secondary actions less dominant?
- Is the workflow stage obvious?
- Does the page show which scenario / result / run it belongs to?

## Scientific usability
Are these visible or reachable: units; reference frames; time systems / epochs;
estimator assumptions; measurement settings; station / visibility assumptions;
scenario assumptions; result-provenance fields?

## Plot & table readability
Readable plots; scannable tables; visible axis labels and units; readable legends;
non-overloaded numeric summaries; clear comparison views; diagnostics prioritized
over decoration.

## State handling
Are empty, loading, running, success, warning, and failure states handled? Are
logs / error details available?

## Layout robustness
Is long text clipped/overflowing? Usable on smaller screens? Usable with many
results or missing artifacts? Controls grouped by task?

## Traceability
Can the user find the producing config, the output artifact, and the seed? Can
they tell which estimator/settings produced the result, and whether outputs are
exploratory, baseline, thesis, README, or presentation?

## Critical rule
A UI is not complete just because widgets exist; it must support the scientific
workflow clearly.
