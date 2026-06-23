---
name: lunar-od-campaign-design
description: >-
  Design fair, reproducible lunar OD experiments and scenario matrices for
  BLS-LM vs SR-UKF: shared truth trajectory, shared measurement realization and
  noise seed, same station visibility / occultation, matched initial uncertainty,
  a clear scenario matrix, and predefined success metrics. Covers baseline,
  28-day arc-by-arc, fragmented-visibility, two-way-Doppler, and
  covariance-handoff campaigns. Use when planning an experiment or a scenario
  config before running.
metadata:
  version: "2.0"
  adapted-from: K-Dense experimental-design
---

# Lunar OD Campaign Design

Design before you run; frame each campaign like a designed experiment.

## Frame it
- **Research question** and the single **controlled variable** (estimator, start
  mode, measurement type, network, arc length, noise level, ...).
- **Experimental unit**: the arc (or the seeded trial) — define independence at
  that level; do not treat measurements within one arc as independent trials.
- **Response variable(s)** chosen up front: final position / velocity error, NIS,
  success fraction, runtime.

## Fair-comparison controls (hold IDENTICAL unless under study)
Truth trajectory and dynamics; measurement realization and **noise seed**;
station network and visibility / elevation / occultation; arc segmentation
(duration, stride, min samples, gap stitching); initial uncertainty / start mode;
measurement type and range-rate physics. Changing more than one at once
invalidates attribution — say so explicitly.

## Design patterns (engineering transfer)
- **Factorial** screening of estimator parameters (start mode x measurement type
  x network).
- **Response-surface** for tuning continuous knobs (UKF Q/R scales, LM damping).
- **Blocking** on known numerical-noise sources (integrator tolerance, grid step).
- **Seeded** allocation; record every seed for auditability and reproducibility.
- **Match the analysis to the design** — pair estimators on the same arcs / seeds.

## Where it lives
Scenario schema `lunar_od/scenario_config.py` (`ScenarioConfig`); runner
`examples/run_scenario_config.py`; arc builder `lunar_od/scenarios.py`
(`build_measurement_arcs`); frozen cases `lunar_od/thesis_matrix.py`; desktop
presets `desktop_app/models/scenario_model.py`.

## Campaign templates
1-day Gaussian baseline · 28-day arc-by-arc (formal handoff) · fragmented SPICE
visibility (UKF history vs BLS arc-ends) · two-way Doppler (cost vs accuracy) ·
covariance handoff (formal / sqrt_formal) · measurement realism (light-time /
stellar aberration).

## Procedure
State the hypothesis -> fix shared controls and record seeds -> **predefine
metrics** (hand off to lunar-od-statistical-diagnostics) -> define the scenario
config and dry-run validate it before the full run.

## Removed from the K-Dense base
Wet-lab / clinical framing: plate layouts, biosample batches, cluster / patient
randomization, blinding / vehicle controls, pseudoreplication.

## Do not
- Compare estimators under different physical / statistical conditions silently,
  or run unseeded Monte Carlo. Do not launch long campaigns without being asked.

## Scope Boundary
Primary for planning experiments and scenario matrices before running.

## Do Not Use For
- running long campaigns (do not)
- interpreting results (lunar-od-statistical-diagnostics / result-validator)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
