# Baseline Registry

Tracks protected scientific baselines used for regression validation and
long-term numerical consistency.

> No numerical expected metrics are filled in here. Every metric is `to be
> registered` until verified from an actual result artifact. **Do not invent
> expected numbers, seeds, commits, or artifacts.**

## Entry template (per baseline)
- baseline scenario name:
- purpose:
- producing script: (under `examples/`)
- scenario configuration:
- random seed:
- force model:
- measurement model:
- estimator settings:
- expected output artifacts: (under `results/`)
- expected metrics: `to be registered`
- tolerance guidance: `to be registered`
- source commit: `unknown / verify`

## Protected baseline categories (all `to be registered`)
- 1-day orbit determination
- 7-day orbit determination
- 28-day orbit determination
- range-only campaign
- range + angles campaign
- range-rate campaign
- two-way Doppler campaign
- Monte Carlo benchmark campaign
- desktop-app scenario execution baseline

## Comparison metrics (when available)
final / RMS position & velocity error, residual RMS, residual mean / bias,
covariance trace, NIS, NEES, convergence rate, failure rate, runtime.

## Critical rule
Do not treat a changed numerical result as acceptable unless a baseline
comparison explains whether the change is intentional, expected, and
scientifically justified.
