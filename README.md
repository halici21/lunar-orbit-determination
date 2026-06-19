# Lunar Orbit Determination Framework

A Python research framework and desktop application for lunar orbit determination
experiments. The project combines a scientific backend for trajectory
propagation, measurement simulation, estimator comparison, diagnostics, and
report generation with a native PyQt5 interface for running and reviewing
experiments.

The main research focus is the comparison of two estimation approaches under
lunar tracking conditions:

- **BLS-LM**: batch least-squares with Levenberg-Marquardt damping
- **SR-UKF**: square-root unscented Kalman filter for sequential estimation

The framework was developed for thesis-scale algorithm research and controlled
synthetic experiments. It is not intended to be used as flight software.

---

## Highlights

- Moon-centered lunar orbit propagation
- Earth and Sun third-body perturbations
- Optional lunar J2 perturbation support
- SPICE-based ephemeris and frame handling
- Ground-station visibility with elevation masks and lunar occultation
- Synthetic range, azimuth, elevation, range-rate, and two-way counted Doppler
  measurements
- Batch and sequential orbit determination workflows
- BLS-LM, SRIF, and SR-UKF estimator implementations
- Arc-by-arc campaign execution with cold, hot, formal, and square-root formal
  handoff modes
- Monte Carlo, long-duration, fragmented-visibility, and Doppler comparison
  scripts
- Diagnostics for residuals, NIS, NEES, observability, covariance conditioning,
  convergence, and runtime
- Native PyQt5 desktop application for local scenario exploration and result
  review

---

## Repository Structure

```text
.
|-- lunar_od/                 # Core orbit determination library
|-- desktop_app/              # PyQt5 desktop application
|-- examples/                 # Reproducible experiment scripts
|-- tests/                    # Unit and regression tests
|-- fixtures/                 # Reference data used by tests
|-- results/                  # Generated experiment outputs
|-- docs/                     # Supplementary notes
|-- requirements.txt          # Core scientific dependencies
|-- requirements-dev.txt      # Test/development dependencies
`-- requirements-accelerated.txt
```

### Core Library

The scientific code lives in `lunar_od/`.

| Module | Purpose |
| --- | --- |
| `geometry.py` | WGS84, ECEF, SEZ, azimuth/elevation geometry |
| `orbit.py` | Classical orbital elements and Cartesian state utilities |
| `dynamics.py` | Equations of motion, STM propagation, gravity gradients |
| `ephemeris.py` | SPICE sampling and interpolation |
| `visibility.py` | Elevation masks, lunar occultation, tracking arcs |
| `measurements.py` | Synthetic observations, residuals, analytic Jacobians |
| `radiometrics.py` | Simplified two-way counted Doppler model |
| `estimators.py` | BLS-LM and SRIF estimators |
| `filters.py` | SR-UKF prediction, update, gating, and diagnostics |
| `scenarios.py` | Arc-by-arc campaign execution |
| `observability.py` | Rank, condition number, Fisher information diagnostics |
| `diagnostics.py` | Residual, consistency, and convergence analysis |
| `scenario_config.py` | JSON-serializable scenario configuration |
| `monte_carlo.py` | Seeded Monte Carlo campaign support |
| `reporting.py` | CSV and plot generation |

### Desktop Application

The desktop app lives in `desktop_app/` and is organized around Qt Designer
`.ui` files, page controllers, services, background workers, reusable widgets,
and a dark QSS theme.

Key screens include:

- Mission Dashboard
- Results Browser
- Run Monitor
- Comparison
- Scenario Builder
- Dynamics
- Measurements
- Estimators
- Ground Stations
- Visibility Gantt
- Estimator Analysis
- Ground Track
- Settings

The app is designed as a local mission-analysis cockpit: it launches existing
experiment scripts, streams logs, loads CSV/PNG outputs, and compares estimator
results without requiring a web server.

---

## Installation

### 1. Create a virtual environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the core dependencies

```bash
pip install -r requirements.txt
```

### 3. Install desktop application dependencies

```bash
pip install -r desktop_app/requirements_desktop.txt
```

### 4. Install development tools

```bash
pip install -r requirements-dev.txt
```

### 5. Optional acceleration backend

Numba is optional. The code falls back to NumPy if Numba is unavailable.

```bash
pip install -r requirements-accelerated.txt
```

---

## SPICE Kernel Setup

Some examples and visibility workflows require NASA SPICE kernels. Place the
required kernels in a local kernel directory and set:

```powershell
$env:LUNAR_OD_KERNEL_DIR = "C:\path\to\kernels"
```

On macOS/Linux:

```bash
export LUNAR_OD_KERNEL_DIR=/path/to/kernels
```

If `LUNAR_OD_KERNEL_DIR` is not set, the loader searches these directories in
order: `~/Documents/mice/kernels`, then `<project>/kernels`.

Required kernels:

| Kernel | Purpose |
| --- | --- |
| `naif0012.tls.txt` | Leap seconds |
| `de421.bsp` | Planetary ephemeris |
| `earth_assoc_itrf93.tf.txt` | Earth frame association (ITRF93) |
| `moon_080317.tf.txt` | Lunar frame association |
| `earth_2025_250826_2125_predict.bpc` | Earth orientation (predicted) |
| `moon_pa_de421_1900-2050.bpc` | Lunar orientation (principal axes) |
| `gm_de431.tpc.txt` | Gravitational constants |
| `pck00010.tpc.txt` | Planetary constants |

The library can still run many synthetic tests without SPICE, but SPICE-backed
campaigns and realistic station visibility analysis require the kernels.

---

## Quick Start

### Run the test suite

From the repository root:

```bash
python -m pytest tests -q
```

If running from the parent project directory:

```bash
python -m pytest python_port/tests -q
```

### Run the desktop application

From the parent project directory:

```bash
python python_port/desktop_app/main.py
```

Or from inside `python_port/`:

```bash
python desktop_app/main.py
```

### Run a baseline estimator comparison

```bash
python examples/baseline_bls_ukf_comparison.py
```

### Run a fragmented-visibility comparison

```bash
python examples/sequential_tracking_comparison.py
```

### Generate the 28-day visibility analysis

```bash
python examples/visibility_28day_dsn_itu_gantt.py
```

### Run the two-way Doppler comparison

```bash
python examples/two_way_doppler_bls_ukf_comparison.py
```

Generated CSV and PNG outputs are written under `results/`.

---

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

Measurements are generated only when the spacecraft satisfies station
visibility constraints and is not occulted by the Moon.

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

## Validation and Testing

The project includes tests for:

- Coordinate geometry
- Orbit conversion
- Dynamics and state propagation
- STM and finite-difference consistency
- Measurement generation and residuals
- Estimator convergence
- SR-UKF sigma points and covariance handling
- Visibility and occultation
- Scenario configuration
- Monte Carlo utilities
- Reporting outputs
- Desktop app smoke tests

Run:

```bash
python -m pytest tests -q
```

For slow or long-running campaigns, use the dedicated scripts in `examples/`
instead of the default fast test suite.

---

## Generated Results

The project writes generated outputs under `results/`. Typical outputs include:

- CSV summary tables
- Per-arc estimator metrics
- Visibility Gantt charts
- Residual and NIS plots
- Runtime comparison plots
- Baseline BLS-LM vs SR-UKF comparison figures

Large generated outputs should normally not be committed to Git unless they are
small, stable, and intentionally used as reference artifacts.

---

## Limitations

This repository is intended for algorithm research, not operational navigation.
Important limitations include:

- Synthetic measurements are used for the main experiments.
- The nominal dynamics model is intentionally simplified relative to
  high-fidelity flight dynamics.
- The two-way Doppler model is a simplified proof-of-capability model.
- Media corrections, transponder delays, clock effects, and full operational DSN
  calibration are outside the current scope.
- SPICE ephemerides are treated as truth in the controlled experiments.

The reported numerical results should be interpreted as controlled estimator
comparison results, not flight-ready navigation performance.

---

## Recommended GitHub Workflow

For a clean public repository, use this directory as the repository root and
avoid committing:

- `__pycache__/`
- `.pytest_cache/`
- virtual environments
- generated result folders unless intentionally curated
- large `.mat` files
- private thesis drafts, presentations, or unrelated PDFs
- local SPICE kernels

Suggested first commit:

```bash
git init
git add lunar_od desktop_app examples tests fixtures docs
git add requirements.txt requirements-dev.txt requirements-accelerated.txt
git add README.md .gitignore
git commit -m "Initial public release of lunar OD framework"
```

---

## License

No license is currently included. Before publishing the repository publicly,
add a license file such as MIT, BSD-3-Clause, or Apache-2.0 depending on how you
want others to use the code.

---

## Citation

If this work is used in academic or research contexts, please cite the related
graduation thesis/project report associated with this repository.

Suggested citation format:

```text
Halici, C. E. Lunar Orbit Determination Framework: Comparative Analysis of
Batch Least-Squares and Square-Root Unscented Kalman Filtering for Lunar Orbit
Determination. Graduation Project, Istanbul Technical University.
```
