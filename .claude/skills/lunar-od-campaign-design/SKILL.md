---
name: lunar-od-campaign-design
description: >-
  Design controlled lunar OD experiments and scenario matrices for a fair
  BLS-LM vs SR-UKF comparison: shared truth trajectory, measurement realization,
  noise seed, station network, arc duration, visibility constraints, and initial
  uncertainty. Covers baseline, 28-day arc-by-arc, fragmented-visibility,
  two-way-Doppler, and covariance-handoff campaigns. Use when the user wants to
  plan an experiment, set up a scenario config, or choose metrics before running.
---

# Lunar OD Campaign Design

Design reproducible, controlled experiments before anything is run.

## Fair-comparison controls
Hold these IDENTICAL across the estimators being compared, unless the variable
under study is exactly one of them:
- truth trajectory and dynamics model,
- measurement realization and **noise seed**,
- station network and visibility / elevation-mask / occultation settings,
- arc segmentation (duration, stride, minimum samples, gap stitching),
- initial uncertainty / start mode (cold / hot / formal / sqrt_formal),
- measurement type and range-rate physics (geometric vs two-way Doppler).

Changing more than one of these at once invalidates attribution — say so
explicitly whenever you do.

## Where this lives in the repo
- Scenario schema: `lunar_od/scenario_config.py` (`ScenarioConfig`).
- Runner: `examples/run_scenario_config.py`.
- Arc builder: `lunar_od/scenarios.py` (`build_measurement_arcs`).
- Frozen thesis cases: `lunar_od/thesis_matrix.py`.
- Desktop presets: `desktop_app/models/scenario_model.py`.

## Campaign templates
- **1-day Gaussian baseline** — prescribed arcs, cold start, multi-station.
- **28-day arc-by-arc** — long-duration stability, formal handoff.
- **Fragmented visibility** — real SPICE visibility, gap stitching, UKF history
  vs BLS arc-end estimates.
- **Two-way Doppler** — `range_rate_physics="two_way_counted_doppler"`; cost vs
  accuracy.
- **Covariance handoff** — formal / sqrt_formal start; verify prior propagation.
- **Measurement realism** — toggle light-time / stellar aberration (position).

## Procedure
1. State the hypothesis and the single controlled variable.
2. Fix the shared controls; record the seed(s).
3. **Choose metrics up front** (median / p95 / max error, success fraction,
   NIS/NEES, runtime) — hand off to lunar-od-statistical-diagnostics.
4. Define the scenario config and dry-run validate it before the full run.

## Do not
- Compare estimators under different physical or statistical conditions silently.
- Run unseeded Monte Carlo, or report results without recording the seed.
