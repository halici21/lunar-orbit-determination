---
name: lunar-od-figures
description: >-
  Publication-quality matplotlib figures for lunar OD results: residual plots,
  NIS/NEES time histories, station-visibility Gantt charts, BLS-LM vs SR-UKF
  comparison plots, runtime plots, Monte Carlo summaries, and arc-by-arc
  diagnostics, for the thesis, GitHub README, or presentation. Use when the user
  asks for a plot, figure, chart, or diagram from result CSVs or reproducible
  experiment data.
metadata:
  version: "2.0"
  adapted-from: K-Dense scientific-visualization, matplotlib
---

# Lunar OD Figures

Generate clear, reproducible matplotlib figures from this project's real outputs.

## Source-first rule (mandatory)
Never invent or extrapolate data. First locate the actual source: a CSV/PNG under
`python_port/results/`, a generating script in `python_port/examples/` (e.g.
`plot_aberration_corrections.py`, `campaign_diagnostic_plots.py`), or freshly
generated experiment data produced with a recorded seed. If it cannot be located
or reproduced, say so and stop.

## Conventions (see `references/figure-style-guide.md`)
- `matplotlib.use("Agg")`; `constrained_layout=True`;
  `savefig(..., dpi=300, bbox_inches="tight")`.
- **Colorblind-safe**: Okabe-Ito baseline; keep consistent estimator colors
  (BLS-LM navy, SR-UKF gold). Sequential -> viridis; diverging -> coolwarm; never
  jet/rainbow.
- Axis labels carry **units in parentheses**; include a legend, title, and a
  one-line caption.
- Remove unnecessary spines / gridlines; label multi-panel figures (A, B, ...);
  **test in grayscale**.
- Log scale for decade-spanning quantities (position error, covariance condition).
- Export **vector PDF/SVG for the thesis**, 300-dpi PNG for slides / README.

## Figure recipes
- **Residuals**: observed-computed per channel (range [m], az/el [arcsec],
  range-rate [m/s]) vs time; mark arc boundaries.
- **NIS / NEES**: time history with the chi-square acceptance band; state dof.
- **Visibility Gantt**: per-station rows plus a network row.
- **Estimator comparison**: grouped bars for median / p95 / max position error
  and runtime, with a compact table.
- **Monte Carlo**: boxplot / violin across seeds — never a single-seed bar.
- **Arc-by-arc**: per-arc final-error markers (BLS discrete, UKF continuous).

## Removed from the K-Dense base
Journal-specific templates (Nature/Science/Cell), biological figure conventions
(gene-expression heatmaps, treatment groups), and graphical-abstract requirements.

## Do not
- Fabricate or relabel results, or overwrite committed result files without
  saying so; do not modify project source code.

## Scope Boundary
Primary for producing matplotlib figures from real result data.

## Do Not Use For
- inventing data (never)
- statistical interpretation (lunar-od-statistical-diagnostics)
- prose/captions (lunar-od-thesis-writing)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
