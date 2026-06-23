# Repository Map

Repository-local map of ownership and data flow for the Lunar OD project.
All paths are under `python_port/`.

## Top-level areas
- `lunar_od/` — core scientific package.
- `examples/` — reproducible experiment / report scripts.
- `tests/` — unit and regression tests (pytest).
- `results/` — generated CSV / PNG / artifacts (not authored by hand).
- `desktop_app/` — PyQt5 interface (controllers, services, workers, widgets, ui, styles, models).

## Core module ownership (`lunar_od/`)
- `dynamics.py` — force models, propagation, STM.
- `accelerated.py` — optional fast numerical kernels (Numba).
- `ephemeris.py`, `spice_loader.py` — SPICE kernels and ephemeris handling.
- `visibility.py` — station visibility and lunar occultation.
- `measurements.py` — synthetic measurements, residuals, analytic Jacobians.
- `radiometrics.py` — range-rate / two-way counted Doppler / light-time models.
- `estimators.py` — BLS-LM and SRIF.
- `filters.py` — SR-UKF.
- `scenarios.py` — arc-by-arc campaign construction and runners.
- `scenario_config.py` — JSON-serializable scenario configuration.
- `diagnostics.py`, `observability.py`, `reporting.py` — analysis and output helpers.
- `measurement_ingestion.py` — external observation ingestion.
- `thesis_matrix.py` — frozen thesis comparison cases / constants.

## Data flow
`scenario_config → scenarios (arcs) → measurements → estimators / filters → diagnostics → reporting → results/`

## Notes
- All module/test/dir paths above were confirmed present when this package was created.
- Packaging / CI config (`pyproject.toml`, `setup.cfg`, `.github/workflows/`,
  `.pre-commit-config.yaml`) is **expected / verify in repo** — not present at creation time.
