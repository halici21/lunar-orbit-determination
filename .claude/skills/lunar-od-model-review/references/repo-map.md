# Repo map for lunar OD reviews & campaigns

| Path | Role |
|---|---|
| `lunar_od/dynamics.py` | Moon-centred dynamics, Earth/Sun third-body perturbations, STM propagation |
| `lunar_od/measurements.py` | range / az / el and range-rate generation, residuals, optional light-time / stellar aberration |
| `lunar_od/radiometrics.py` | range-rate physics: instantaneous geometric and two-way counted Doppler, light-time |
| `lunar_od/estimators.py` | BLS-LM and SRIF; analytic Jacobians |
| `lunar_od/filters.py` | SR-UKF / square-root sequential filtering |
| `lunar_od/scenarios.py` | arc building (`build_measurement_arcs`), arc / scenario runners |
| `lunar_od/scenario_config.py` | `ScenarioConfig` schema + validation |
| `lunar_od/visibility.py` | elevation mask, lunar occultation, pass stitching |
| `lunar_od/diagnostics.py` | residuals, chi-square, NIS / NEES, consistency metrics |
| `lunar_od/reporting.py` | CSV / PNG output helpers |
| `examples/` | experiment & report scripts; `run_scenario_config.py` is the runner |
| `tests/` | pytest suite (`test_measurements.py`, `test_estimators.py`, `test_filters.py`, ...) |
| `results/` | generated CSV / PNG outputs (not authored by hand) |
| `desktop_app/` | PyQt5 UI (controllers, models, services) |

## Validation rules (apply in every review and campaign)
- Inspect the relevant source file(s) **before** suggesting any change.
- Preserve numerical correctness; **do not weaken tolerances without
  justification**.
- Recommend a targeted test first (`pytest tests/test_<area>.py -k <name>`), then
  the full suite (`pytest -q`).
- **Never invent results**; locate the script / CSV / plot / test first.
- Keep synthetic-experiment conclusions separate from operational navigation
  claims.
- Keep generated outputs, SPICE kernels, caches, virtual environments, and
  private thesis files out of GitHub unless explicitly requested.
