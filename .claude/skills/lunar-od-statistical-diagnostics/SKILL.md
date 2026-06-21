---
name: lunar-od-statistical-diagnostics
description: >-
  Analyze and interpret lunar OD estimator results and consistency diagnostics:
  median / p95 / max / RMS position and velocity error, runtime, operational
  success fraction, NIS, NEES, covariance condition number, residual
  distributions, correlations, seeded Monte Carlo, and long-duration arc
  campaigns. Use when the user wants to interpret result CSVs, compare BLS-LM
  vs SR-UKF statistically, or assess tail behavior and operational robustness.
---

# Lunar OD Statistical Diagnostics

Turn result tables into careful orbit-determination conclusions.

## Source-first rule
Locate the data before analyzing it: aggregate / arc-summary / trials CSVs under
`python_port/results/`, or the producing script in `python_port/examples/`. Load
with pandas and report N (number of arcs or seeds) explicitly.

## Metrics and their OD meaning
- **median position error** — typical accuracy.
- **p95 / max** — tail behavior; the error tail usually drives operational risk.
- **RMS residual / whitened residual** — measurement-fit quality.
- **operational success fraction** — converged + finite final cost + acceptable
  conditioning + (for UKF) covariance stability.
- **NIS** — innovation consistency (per measurement); **NEES** — state-error
  consistency (needs truth + covariance). Compare each to the chi-square band
  for the correct dof.
- **covariance condition number** — observability / numerical health; very large
  values flag weak observability, not necessarily divergence.

## How to operate
- Report three things separately and do not collapse them: **typical** (median),
  **tail** (p95 / max), and **robustness** (success fraction, conditioning).
- Monte Carlo: summarize the distribution across seeds (median + spread); never
  conclude from a single seed.
- Long arcs: examine arc-by-arc trends and boundary handoff, not only the
  aggregate.
- Always state the sample size; mark small-N results as indicative, not
  conclusive, and prefer median/quantiles to the mean when the distribution is
  skewed.

## Plot choice (hand off to lunar-od-figures)
- distribution across seeds → boxplot / violin; one quantity vs scenario →
  grouped bars; residual shape → histogram + Q-Q; sequential or per-arc →
  time series; relationship (e.g. condition number vs error) → scatter.

## Do not
- Overclaim from few seeds or arcs.
- Treat synthetic success as operational / flight performance.
