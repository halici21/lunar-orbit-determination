# Baseline research guide

This guide carries the baseline scientific overview and experiment catalog from
the earlier README. Current qualified extensions and their limits are documented
in [the authoritative Phase 17 status](phase17_authoritative_status.md) and the
individual qualification reports. The baseline overview is not a complete
inventory of every optional force or measurement model.

Commands below run from the repository root. SPICE-backed examples require the
[kernel setup](windows-quickstart.md#spice-kernels). CSV/PNG outputs go to results/.
Long campaigns are separate from the public CI gate.

## Scientific Model

### State Definition

The spacecraft state is represented in a Moon-centered, J2000-aligned inertial
frame:

```text
x = [rx, ry, rz, vx, vy, vz]^T
```

### Baseline Dynamics

The nominal force model includes:

- Lunar point-mass gravity
- Differential third-body gravity from Earth
- Differential third-body gravity from the Sun
- Optional lunar J2 perturbation

The state transition matrix is propagated through an augmented 42-state ODE,
allowing analytical measurement Jacobians to be mapped from the initial arc
state to each measurement epoch.

### Measurements

The measurement layer supports:

- Range
- Azimuth
- Elevation
- Geometric range-rate
- Simplified two-way counted Doppler
- Converged two-way range (`measurement_type: two_way_range`): explicit
  t1/t2u/t2d/t3 event chain with fixed transponder delay, exact event-epoch
  `sxform` station states, and an implicit 3x3 event-system Jacobian
  (BLS-LM/SRIF only; see `two_way_range.md`)

Measurements are generated only when the spacecraft satisfies station
visibility constraints and is not occulted by the Moon.

#### Aberration corrections (optional)

Range/azimuth/elevation generation and residuals accept two opt-in corrections
(both default **off**, so the geometric instantaneous model is unchanged):

- **`apply_light_time=True`** — converged Newtonian one-way light time (CN). The
  spacecraft is evaluated at the transmit time `t_t = t_r − τ` (τ from a
  contraction-mapping iteration) while the station, Earth, and topocentric frame
  are evaluated at the receive time `t_r`. This accounts for **target motion
  during the light travel time**.
- **`apply_stellar_aberration=True`** (requires light time) — reception-case
  Newtonian stellar aberration, applied *after* the CN solution as a pure
  Rodrigues rotation of the inertial line of sight toward the observer velocity
  (`sin φ = (v/c) sin w`). It accounts for the **observer's velocity aberration**
  and shifts azimuth/elevation while leaving range unchanged (rotation preserves
  the vector norm). Two frames are selectable via `stellar_aberration_model`:
  - `"spice_ssb"` — observer velocity relative to the **Solar System Barycentre**
    (J2000 axes; MCI is J2000-aligned so no extra rotation is needed). This is the
    SPICE-like `CN+S` choice; the correction is of order `|v_⊕,SSB|/c ≈ 1e-4 rad`.
  - `"local_mci"` *(default)* — observer velocity relative to the Moon-centred
    frame. Internally consistent but **not** identical to SPICE `CN+S` (it omits
    the Moon's barycentric velocity, giving a correction ~20× smaller).

The same correction settings are carried on `PassGeometry` and reused in
generation, prediction, and residual computation. Jacobian behavior is selected
independently. The default light-time Jacobian remains the backward-compatible
first-order approximation. With `jacobian_model="implicit_light_time"`, the CN
range, azimuth, and elevation rows include implicit `dτ/dx`, transmit-epoch STM,
LOS-normalization, and receive-epoch SEZ chain-rule terms. For CN+S, the range
row remains the same implicit CN result while the angle rows additionally apply
a local tangent-space central finite-difference Jacobian of the production
stellar-aberration transform. The resulting method is hybrid, not fully
analytic: analytic implicit CN sensitivity, local numerical aberration
sensitivity, and analytic receive-frame/SEZ angle mapping. The legacy
first-order mode remains unchanged.
See `one_way_light_time_jacobians.md` for equations and singularity policy.

### Estimators

**BLS-LM** processes all measurements in a completed tracking arc and solves a
damped nonlinear least-squares problem.

**SR-UKF** processes measurements sequentially. It propagates sigma points,
updates the square-root covariance factor, and supports consistency monitoring
through NIS/NEES-style diagnostics.

---

## Main Experiment Families

The `examples/` directory contains scripts used to reproduce the main
comparison studies:

| Script | Description |
| --- | --- |
| `baseline_bls_ukf_comparison.py` | Controlled BLS-LM vs SR-UKF baseline |
| `baseline_bls_ukf_matrix_campaign.py` | Extended comparison matrix |
| `sequential_tracking_comparison.py` | Fragmented visibility case |
| `visibility_28day_dsn_itu_gantt.py` | Multi-station visibility analysis |
| `two_way_doppler_bls_ukf_comparison.py` | Two-way counted Doppler comparison |
| `thesis_factorial_report.py` | Factorial thesis campaign summary |
| `campaign_diagnostic_plots.py` | Diagnostic figure generation |
| `run_scenario_config.py` | Run a JSON-defined scenario |

---

## Scenario Configuration

Scenarios can be defined programmatically or loaded from JSON through
`ScenarioConfig`.

Example fields include:

```text
measurement_type: position | range_rate
estimator_type: bls_lm | srif | ukf
start_mode: cold | hot | formal | sqrt_formal
network: single | multi
duration_h
sample_step_s
min_elevation_deg
noise
j2_moon
range_rate_physics
```

This makes experiment setup reproducible and suitable for regression testing.

---
