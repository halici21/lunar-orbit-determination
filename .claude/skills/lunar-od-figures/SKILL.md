---
name: lunar-od-figures
description: >-
  Create publication-quality matplotlib figures for lunar OD results: residual
  plots, NIS/NEES time histories, station-visibility Gantt charts, BLS-LM vs
  SR-UKF comparison bars/tables, runtime plots, Monte Carlo box/violin
  summaries, and arc-by-arc diagnostics. Use when the user asks for a plot,
  figure, chart, or diagram from result CSVs or experiment data for the thesis,
  GitHub README, or presentation slides.
---

# Lunar OD Figures

Generate clear, reproducible matplotlib figures from this project's real outputs.

## Source-first rule (mandatory)
Never invent data. First locate the actual source:
- a result CSV/PNG under `python_port/results/`,
- a generating script in `python_port/examples/` (e.g.
  `plot_aberration_corrections.py`, `campaign_diagnostic_plots.py`, the
  report/campaign scripts), or
- freshly generated experiment data produced with a recorded seed.
If the data cannot be located or generated, say so and stop — do not fabricate.

## Conventions (see `references/plot_conventions.md`)
- `matplotlib.use("Agg")`; save PNG at dpi 150 into `python_port/docs/figures/`
  (documentation) or `python_port/results/` (experiment outputs).
- Colorblind-safe palette with consistent estimator colors (BLS-LM vs SR-UKF).
- Every figure: axis labels **with units**, a legend, a title, and a one-line
  caption.
- Log scale for quantities spanning decades (position error, covariance
  condition number). Readability over decoration — no chartjunk, no 3-D bars.

## Figure recipes
- **Residuals**: observed−computed per channel (range [m], az/el [arcsec],
  range-rate [m/s]) vs time; mark arc boundaries.
- **NIS / NEES**: time history with the chi-square acceptance band; state dof.
- **Visibility Gantt**: per-station rows plus a network row; optional time cursor.
- **Estimator comparison**: grouped bars for median / p95 / max position error
  and runtime, paired with a compact table.
- **Monte Carlo**: boxplot or violin across seeds — never a single-seed bar.
- **Arc-by-arc**: per-arc final-error markers (BLS discrete, UKF continuous).

## Output
- A runnable snippet, or a small reusable script under `examples/` only if it is
  clearly reusable; plus the saved figure path and the numeric summary it shows.
- Recommend a layout per target: thesis (single-column, large fonts), README
  (wide, self-explanatory), slides (minimal text, high contrast).

## Do not
- Fabricate or extrapolate results, or relabel one scenario as another.
- Modify project source code, or overwrite committed result files without saying so.
