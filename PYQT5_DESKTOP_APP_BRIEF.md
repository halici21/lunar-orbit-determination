# PyQt5 Desktop App Brief - Lunar OD

Build a **local PyQt5 desktop application**, not a website. The app should be a
modern, smooth, dark-themed mission-analysis cockpit for this lunar orbit
determination project.

## Project Summary

This project simulates lunar orbit determination and compares:

- **BLS-LM:** offline batch least-squares estimator. Processes a completed arc.
- **SR-UKF:** online sequential square-root UKF. Updates at each measurement.

The backend already includes lunar dynamics, SPICE geometry, visibility,
measurement generation, BLS-LM, SR-UKF, diagnostics, plots, CSV outputs, and
example scripts.

Main package:

```text
python_port/lunar_od/
```

Important result folders:

```text
python_port/results/baseline_bls_ukf/
python_port/results/sequential_tracking/
python_port/results/bls_3day_ablation/
python_port/results/two_way_doppler_bls_ukf/
python_port/results/baseline_bls_ukf_matrix/
```

## Backend Concepts to Expose

Show these concepts in the UI:

- Moon-centered J2000 trajectory propagation
- Earth/Sun third-body perturbations
- SPICE ephemeris/frame usage
- Ground station visibility and lunar occultation
- Single-station vs multi-station tracking
- Gap stitching and tracking arc creation
- Range/azimuth/elevation measurements
- Geometric range-rate and simplified two-way counted Doppler
- BLS-LM arc-end solutions
- SR-UKF measurement-by-measurement updates
- NIS, residuals, covariance condition, runtime, success fraction

## Required UI Structure

Use modular `.ui` files and Python controllers.

Recommended structure:

```text
python_port/desktop_app/
    main.py
    app.py
    ui/
        main_window.ui
        pages/
            dashboard_page.ui
            scenario_builder_page.ui
            dynamics_page.ui
            stations_page.ui
            visibility_page.ui
            measurements_page.ui
            estimators_page.ui
            run_monitor_page.ui
            results_page.ui
            comparison_page.ui
            diagnostics_page.ui
            settings_page.ui
    controllers/
    widgets/
    services/
    workers/
    models/
    styles/
        dark_theme.qss
```

Rules:

- `.ui` files define layout.
- Controllers define behavior.
- Services load CSV/PNG files and launch scripts.
- Workers run long tasks using `QProcess` or `QThread`.
- Never block the GUI thread.

## Main Screens

### Dashboard

Show project status:

- latest result folders
- SPICE/kernel status
- quick metric cards
- latest baseline comparison plot
- latest sequential tracking plot
- quick run buttons

### Scenario Builder

Expose:

- duration
- time step
- arc mode
- visibility on/off
- elevation mask
- gap threshold
- station network
- measurement type
- range-rate physics
- estimator
- start mode: `cold`, `hot`, `formal`, `sqrt_formal`
- noise seed
- output folder

Allow export/import as JSON.

### Dynamics

Show:

- Moon point-mass gravity
- Earth/Sun perturbations
- optional J2/future perturbations
- integrator settings
- orbit preview

### Stations & Visibility

Show:

- station table
- station map
- 28-day Gantt chart
- raw vs stitched network visibility
- arc durations
- measurement counts

### Measurements

Show:

- range, azimuth, elevation
- geometric range-rate
- two-way Doppler settings
- residual plots
- noise/bias settings

### Estimators

Show BLS-LM and SR-UKF settings:

- BLS max iteration, damping, prior, scaling
- UKF alpha/beta/kappa, Q/R, gating, covariance inflation
- algorithm flow diagrams

### Run Monitor

Launch existing scripts with live logs:

```text
examples/baseline_bls_ukf_comparison.py
examples/sequential_tracking_comparison.py
examples/visibility_28day_dsn_itu_gantt.py
examples/two_way_doppler_bls_ukf_comparison.py
examples/bls_7day_ablation_appendix.py
```

Use `QProcess`. Show stdout, stderr, elapsed time, progress, cancel button.

### Results Browser

Browse:

- CSV files as tables
- PNG files as images
- result folders
- aggregate metrics

Allow export selected plots/tables.

### Comparison

Compare:

- BLS-LM vs SR-UKF
- cold vs hot vs formal
- position-only vs range-rate
- geometric RR vs two-way Doppler
- single vs multi-station

Plots:

- grouped bar charts
- time histories
- box plots
- error tables
- runtime comparison

### Diagnostics

Show:

- NIS
- NEES if available
- residual RMS
- covariance condition
- accepted/rejected measurements
- rank/condition number
- worst BLS tail arcs

## Animation Ideas

Keep animations useful, not decorative:

- orbit playback around Moon
- station line-of-sight appears when visible
- visibility Gantt time cursor
- station handover glow
- BLS arc fills with measurements, solves at arc end
- SR-UKF update pulse at every measurement
- sigma-point cloud animation
- Doppler uplink/downlink light-time animation
- covariance ellipse shrinking after update

## Plotting

Use:

- `pyqtgraph` for fast interactive plots
- `matplotlib` Qt canvas for thesis-style figures
- optional 3D later, not required for MVP

## MVP

First version should include:

1. main window + sidebar
2. dark theme
3. dashboard
4. result browser
5. CSV table viewer
6. PNG viewer
7. comparison plots from existing CSVs
8. script launcher with live log
9. scenario preset list

Do not start by building a complicated 3D engine. Build a stable local result
viewer and script launcher first.

## Important Style Requirements

- modern dark UI
- smooth page transitions
- rounded metric cards
- animated sidebar
- clean scientific plots
- tooltips for OD terms
- no web server
- no browser UI
- local desktop only

## Suggested Dependencies

```text
PyQt5
pandas
numpy
matplotlib
pyqtgraph
scipy
spiceypy
```

Optional later:

```text
pyvista
pyvistaqt
vispy
qtawesome
```

## Final Goal

The final app should let a user:

- understand the lunar OD workflow visually,
- configure a scenario,
- run existing experiments,
- watch logs,
- inspect results,
- compare estimators,
- view visibility and tracking arcs,
- animate orbit/measurements,
- export plots and tables.

It should feel like a polished local mission-analysis tool built on the
existing Python backend.

