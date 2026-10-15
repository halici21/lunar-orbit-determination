# Python API guide

Install the checkout with `python -m pip install -e .`, then import `lunar_od`.
This is a curated map into the implementation; the linked signatures and
docstrings are authoritative.

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
