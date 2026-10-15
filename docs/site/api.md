# Python API guide

Install the checkout with `python -m pip install -e .`, then import `lunar_od`.
This guide combines explicit numerical contracts with signatures and docstrings
generated from the checked-out source during every site build.

| Task | Public symbols | Implementation |
|---|---|---|
| Coordinate geometry | `geodetic_to_ecef_wgs84`, `Station` | [Geometry](../../src/lunar_od/geometry.py), [stations](../../src/lunar_od/config.py) |
| Propagate state and STM | `propagate_augmented_state` | [Dynamics](../../src/lunar_od/dynamics.py) |
| Prepare observation geometry | `PassGeometry`, `compute_position_residuals_analytic` | [Measurements](../../src/lunar_od/measurements.py) |
| Fit a batch initial state | `estimate_position_bls_lm`, `estimate_position_srif` | [Estimators](../../src/lunar_od/estimators.py) |
| Sequential estimation | `run_lunar_ukf`, `UnscentedTransformConfig` | [Filters](../../src/lunar_od/filters.py) |
| Run prepared arcs | `PreparedArc`, `run_batch_arc_sequence` | [Scenarios](../../src/lunar_od/scenarios.py) |
| Load scenario settings | `ScenarioConfig`, `scenario_config_from_mapping` | [Scenario config](../../src/lunar_od/scenario_config.py) |

The [complete tutorial driver](../../examples/quickstart_od.py) demonstrates
propagation, seeded observation construction, both batch fits and residual export
without hidden local files.

## Array and epoch conventions

- Dynamic states have six components: position in m, velocity in m/s.
- The augmented STM block uses column-major (`order="F"`) storage.
- Position-observation rows contain time, range, azimuth, elevation, station ID,
  epoch index and optionally arc ID; IDs and epoch indices are one-based.
- Time arguments are elapsed seconds in the selected scenario's uniform time basis.
  UTC/SPICE conversions belong at the documented ephemeris boundaries.
- Check the [support matrix](../validation/support-matrix.md) before combining
  optional profiles and estimators.

## Parameters and returned arrays

| Parameter / result | Contract |
|---|---|
| `t_pass_s` | Increasing one-dimensional epoch grid, seconds; the first sample defines the batch state's reference epoch |
| `x_nominal0`, `x0_mci` | Six-element Moon-centred inertial state `[x,y,z,vx,vy,vz]`, m and m/s; optional bias states follow the selected bias mode |
| `obs_data` | Two-dimensional rows `[t_s, range_m, az_rad, el_rad, station_id, epoch_index, ...]` for the geometric position profile; station and epoch indices are one-based |
| `pass_geo` | Geometry on the same time grid, with Earth states and full 6×6 J2000→ITRF93 transforms; station sigmas define the measurement weighting |
| `mu_*_m3_s2` | Gravitational parameters in m³/s²; set a body's value to zero to disable its third-body acceleration |
| `get_earth_pos`, `get_sun_pos` | Callbacks returning Moon-centred inertial positions in m on the requested epochs; match the geometry and force model |
| `rtol`, `atol` | Integration tolerances; no automatic physical error guarantee follows from their values |
| `prior_covariance`, `p0` | Symmetric positive-definite state covariance at the initial epoch, in squared/mixed state units; use the same prior mean and covariance for paired comparisons |
| `process_noise` | UKF Q; its units and time scaling follow `process_noise_model`; `None` disables process noise |
| `return_posterior=True` | Batch returns `(estimated_state, stop_reason, EstimatorStats)` with covariance at the batch reference epoch, **not** at the final observation |
| `LunarUKFResult` | `state_estimates` and `covariances` are indexed by observation update; `final_state` and `final_covariance` refer to the last processed observation |

Range-rate and two-way profiles have different row/model contracts. Read their
theory pages before passing such observations to a position-only function.

## Small examples

```python
import numpy as np
from lunar_od import geodetic_to_ecef_wgs84

# Latitude/longitude are degrees, height and returned Earth-fixed vector are m.
ecef_m = geodetic_to_ecef_wgs84(0.0, 0.0, 0.0)
assert np.isclose(np.linalg.norm(ecef_m), 6378137.0)
```

The [complete executable example](../../examples/quickstart_od.py) constructs
geometry and observations, propagates a six-state orbit plus a column-major STM,
fits the initial state and exports unit-labelled residuals. The
[paired campaign](../../examples/benchmark_estimators.py) additionally demonstrates
matched explicit priors, final-epoch covariance propagation and square-root UKF
settings. These examples require no private data or SPICE downloads.
