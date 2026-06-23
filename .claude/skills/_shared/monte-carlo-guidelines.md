# Monte Carlo Guidelines

Rules for scientifically valid Monte Carlo studies.

> Running a Monte Carlo campaign is a **long campaign** — do not run it by
> default. Plan, specify, and report only; run only when the user explicitly asks.

## Setup to specify
seed strategy; number of trials; truth-generation method; measurement-generation
method; visibility-generation method; noise model; station network; force model;
measurement model; estimator settings; initial-state-error distribution; initial
covariance assumptions.

## Fairness (when comparing estimators)
Use identical truth trajectories, measurement schedules, visibility windows,
noise realizations, initial uncertainty, and force model (unless testing model
mismatch), with identical convergence / failure criteria.

## Required reporting
median, mean, standard deviation, min / max, percentiles (especially 5th / 50th /
95th), convergence rate, failure rate, outlier rate, and runtime distribution
when relevant.

## Interpretation
- distinguish deterministic vs stochastic effects.
- distinguish geometry-driven vs estimator-design effects.
- distinguish numerical-conditioning vs physics effects.
- do not claim robust improvement from a single run.
- do not hide failed / diverged trials unless the exclusion rule was documented
  before analysis.
