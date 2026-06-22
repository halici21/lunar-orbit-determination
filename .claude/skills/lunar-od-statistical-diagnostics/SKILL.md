---
name: lunar-od-statistical-diagnostics
description: >-
  Analyze and interpret lunar OD estimator performance and statistical
  consistency: median / p95 / max / RMS position and velocity error, runtime,
  operational success fraction, NIS, NEES, residual distributions, covariance
  conditioning, correlation between arc geometry and error, seeded Monte Carlo,
  and long-duration / fragmented-visibility campaigns. Use when the user wants to
  interpret result CSVs or compare BLS-LM vs SR-UKF statistically.
metadata:
  version: "2.0"
  adapted-from: K-Dense statistical-analysis, exploratory-data-analysis
---

# Lunar OD Statistical Diagnostics

Turn result tables into careful orbit-determination conclusions, and end with a
short structured **findings + recommendations** summary.

## Source-first rule
Locate the data first: aggregate / arc-summary / trials CSVs under
`python_port/results/`, or the producing script in `python_port/examples/`. Load
with pandas and **report N (arcs or seeds) explicitly**.

## Metrics and their OD meaning
- **median** — typical accuracy; **p95 / max** — tail, which drives operational
  risk.
- **RMS residual / whitened residual** — measurement-fit quality.
- **operational success fraction** — converged + finite cost + acceptable
  conditioning + (UKF) covariance stability.
- **NIS** — innovation consistency (per measurement); **NEES** — state-error
  consistency (needs truth + covariance). Compare to the chi-square band for the
  correct dof.
- **covariance condition number** — observability / numerical health.
- **arc-geometry vs error correlation** — e.g. visibility span or elevation vs
  final error; report the coefficient and a scatter, not a causal claim.

## How to operate
- Report **typical** (median), **tail** (p95 / max), and **robustness** (success,
  conditioning) separately; do not collapse them.
- Monte Carlo: summarize the across-seed distribution (median + spread); never
  conclude from a single seed.
- Long arcs: examine arc-by-arc trends and boundary handoff, not just aggregates.
- State sample size; mark small-N results as indicative; prefer quantiles to the
  mean for skewed distributions.

## Plot choice (hand off to lunar-od-figures)
distribution across seeds -> boxplot / violin; quantity vs scenario -> grouped
bars; residual shape -> histogram + Q-Q; sequential / per-arc -> time series;
relationship -> scatter.

## Removed from the K-Dense base
The 200+ file-format detection and chemistry / bioinformatics / imaging format
handling of the EDA skill — not relevant; here the inputs are this project's CSVs.

## Do not
- Overclaim from few seeds or arcs, or treat synthetic success as operational /
  flight performance.
