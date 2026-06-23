# Desktop UX Workflow

Use when designing Lunar OD desktop workflows, page flow, navigation, interaction.

## Primary workflows
open dashboard -> define scenario -> configure dynamics -> configure stations &
visibility -> configure measurements -> select estimator -> run scenario -> monitor
execution -> inspect results -> compare estimators -> inspect diagnostics -> export
or locate artifacts.

## Workflow rules
Important workflows need minimal navigation. The user should always know: scenario
state, run state, estimator selected, measurement configuration, station network,
output location, and result provenance.

## Run feedback
Long-running tasks must show progress, run status, logs, current stage, failure
reason (if failed), and artifact path (if completed).

## Result pages (expose when available)
scenario config path, producing script/workflow, output artifact path, run
timestamp, random seed, and summaries of estimator / measurement / station / force
model settings.

## Estimator comparison pages (expose when available)
aligned scenario/config summary, measurement count, visibility summary, final and
RMS position/velocity error, residual RMS and mean/bias, NIS/NEES summary,
convergence and failure status, runtime, and the assumptions (initial state error,
covariance). Make explicit whether compared estimators used aligned truth,
measurements, visibility, noise, seed, initial error, and covariance.

## Error handling
Errors explain what failed, why, what to do next, and whether partial results exist.

## Critical rule
The user must never have to guess which scenario, config, seed, estimator, or
artifact produced a displayed result.
