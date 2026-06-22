---
name: lunar-od-model-review
description: >-
  Critically review lunar OD model assumptions, estimator consistency, Jacobians,
  covariance, frames, light-time / Doppler / visibility, and numerical risks.
  Verifies that measurement generation, predicted h(x), residual computation, and
  Jacobians use the same model; flags synthetic-data bias, missing or
  frame-inconsistent light-time, simplified Doppler, weak observability,
  covariance mistuning, and invalid estimator comparisons. Use when reviewing OD
  code or results for correctness.
metadata:
  version: "2.0"
  adapted-from: K-Dense scientific-critical-thinking, peer-review
---

# Lunar OD Model Review

Find concrete, severity-ranked problems in the physics, estimation, and numerics.
Distinguish observation from interpretation, and match criticism to importance.

## Check first: model consistency
Do **generation**, **prediction h(x)**, **residual computation**, and the
**Jacobian** use the same model? Trace:
- generation `lunar_od/measurements.py` (`generate_position_measurements`,
  `generate_range_rate_measurements`),
- prediction / residuals `compute_position_residuals[_analytic]`,
  `compute_range_rate_residuals[_analytic]`,
- range-rate physics `lunar_od/radiometrics.py`,
- dynamics / STM `lunar_od/dynamics.py`; estimator Jacobians
  `lunar_od/estimators.py`; filters `lunar_od/filters.py`; visibility
  `lunar_od/visibility.py`; diagnostics `lunar_od/diagnostics.py`.

A generation/prediction mismatch is the highest-severity finding. See
`references/repo-map.md` for the full path map and the project validation rules.

## Risk checklist
- **Synthetic-data bias** — truth and estimator share one model; flag results
  that are only self-consistent, not evidence of fidelity.
- **Light-time / aberration** — present? converged? frame-consistent? (cf.
  `SPICE_CN_CNPLUS_VALIDATION.md`: MCI-frame light-time vs SSB-inertial).
- **Doppler realism** — instantaneous vs averaged counted Doppler; turnaround /
  transponder / clock terms (cf. `DOPPLER_RANGE_RATE_MODEL_REVIEW.md`).
- **Jacobian** — analytic vs finite-difference; are neglected terms documented
  and bounded?
- **Observability / conditioning** — weak arcs, near-singular information.
- **Covariance tuning** — process / measurement noise; NIS/NEES consistency;
  gating.
- **Comparison validity** — same seed / network / arcs / start mode across the
  estimators compared.

## Output (peer-review structure, by severity)
- **Major** [high] — one-line claim, `file.py:function`, why it is wrong, a
  **minimal testable fix** plus the test to add or run
  (`pytest tests/test_<area>.py -k <name>`, then the full suite).
- **Minor** [medium / low] — same format, lower impact.
- **Questions** — where intent is unclear.

Identify the claim type ("is this causal evidence of correctness, or only
self-consistent?"); separate data from interpretation; be specific (no vague
criticism); if something is correct, say so briefly.

## Removed from the K-Dense base
IRB / human-subjects / IACUC / HIPAA, clinical-vs-statistical significance,
GRADE, and journal-venue expectations.

## Do not
- Speculate without locating the code; recommend a rewrite where a small,
  verifiable change suffices; modify source as part of the review (review only,
  unless asked to fix); or weaken test tolerances without justification.
