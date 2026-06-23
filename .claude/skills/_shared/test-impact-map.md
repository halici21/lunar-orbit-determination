# Test Impact Map

Map a changed code area to the targeted test command(s) to run **first**, before
the full regression suite. Commands assume the working directory is the parent of
`python_port/`; if running from inside `python_port/`, drop the `python_port/`
prefix (use `tests/...`).

| Changed area | Targeted command |
|---|---|
| Dynamics / propagation / STM | `python -m pytest python_port/tests/test_dynamics.py -q` |
| Ephemeris / SPICE | `python -m pytest python_port/tests/test_ephemeris.py python_port/tests/test_spice_loader.py -q` |
| Measurements / residuals / Jacobians | `python -m pytest python_port/tests/test_measurements.py -q` |
| Doppler / radiometrics | `python -m pytest python_port/tests/test_doppler_model_review.py python_port/tests/test_stellar_aberration_vv.py -q` |
| Estimators (BLS-LM / SRIF) | `python -m pytest python_port/tests/test_estimators.py -q` |
| Filters (SR-UKF) | `python -m pytest python_port/tests/test_filters.py -q` |
| Observability | `python -m pytest python_port/tests/test_observability.py -q` |
| Scenarios / config | `python -m pytest python_port/tests/test_scenarios.py python_port/tests/test_scenario_config.py -q` |
| Desktop app | desktop smoke tests **if present**; otherwise import-level checks |
| Full regression | `python -m pytest python_port/tests/ -q` |

Rule: run the targeted command first to localize impact, then the full suite
before declaring a change validated. Do not run long campaigns as part of this.
