# Lunar Orbit Determination Desktop App Brief

This document is a complete implementation brief for building a modern local
desktop application for the Lunar Orbit Determination project. The intended UI
framework is **PyQt5**. The application must run locally as a desktop program;
it must not be implemented as a website, browser dashboard, Flask app, or
Electron wrapper.

The goal ios to provide enough project context for another coding assistant or
developer to understand what has been implemented, what can be visualized, what
simulation settings should be exposed, and how the UI should be organized into
modular `.ui` files and Python controllers.

---

## 1. High-Level Project Summary

This project develops a Python-based lunar orbit determination framework. It
simulates a spacecraft orbiting the Moon, generates synthetic ground-station
tracking measurements, applies visibility and lunar occultation constraints, and
compares different orbit-determination algorithms.

The main scientific comparison is between:

1. **BLS-LM**  
   Batch Least-Squares with Levenberg--Marquardt damping. It processes a
   completed tracking arc and returns an end-of-arc orbit estimate.

2. **SR-UKF**  
   Square-Root Unscented Kalman Filter. It processes measurements sequentially
   and updates the spacecraft state and covariance whenever a visible
   measurement becomes available.

The project is a research-grade lunar OD sandbox, not a flight-operational
navigation system. The current physics and measurement models are intentionally
simplified enough to support controlled estimator comparison.

---

## 2. What Has Been Implemented

The active Python implementation lives under:

```text
python_port/
```

The main package is:

```text
python_port/lunar_od/
```

Important modules:

| Module | Role |
|---|---|
| `config.py` | Station database, gravitational constants, nominal measurement noise |
| `orbit.py` | Classical orbital elements, Cartesian state generation, frame setup |
| `geometry.py` | WGS-84, ECEF, SEZ, azimuth/elevation, angle wrapping |
| `spice_loader.py` | SPICE kernel discovery/loading |
| `ephemeris.py` | Earth/Sun ephemeris sampling and PCHIP interpolation |
| `dynamics.py` | Moon-centered dynamics, Earth/Sun third-body perturbations, STM |
| `visibility.py` | Elevation masks, lunar occultation, pass stitching, arc construction |
| `measurements.py` | Range, azimuth, elevation, geometric range-rate measurements |
| `radiometrics.py` | Simplified two-way counted-Doppler and light-time logic |
| `estimators.py` | BLS-LM / SRIF-style estimator logic |
| `filters.py` | Sequential SR-UKF / square-root filtering logic |
| `scenarios.py` | Arc-by-arc scenario runner and handoff workflow |
| `diagnostics.py` | Residuals, chi-square, NIS/NEES helpers, consistency metrics |
| `observability.py` | Fisher information, rank, condition number diagnostics |
| `noise_models.py` | Measurement and process-noise helper contracts |
| `q_tuning.py` | Process-noise Q grid and sweep helpers |
| `adaptive_tuning.py` | Adaptive/robust tuning experiments |
| `monte_carlo.py` | Seeded trial execution and aggregation |
| `scenario_config.py` | JSON scenario validation and normalization |
| `measurement_ingestion.py` | Generic CSV observation ingestion |
| `reporting.py` | CSV/PNG output helpers |
| `thesis_matrix.py` | Frozen thesis comparison cases |

Example scripts live under:

```text
python_port/examples/
```

Important examples:

| Script | Purpose |
|---|---|
| `baseline_bls_ukf_comparison.py` | Main BLS-LM vs SR-UKF comparison over prescribed arcs |
| `baseline_bls_ukf_matrix_campaign.py` | Monte Carlo / initial-error matrix cases |
| `sequential_tracking_comparison.py` | Sequential SR-UKF history vs BLS arc-end estimates |
| `visibility_28day_dsn_itu_gantt.py` | 28-day station visibility Gantt chart |
| `bls_7day_ablation_appendix.py` | BLS range-rate / start-mode ablation |
| `two_way_doppler_bls_ukf_comparison.py` | Simplified two-way Doppler comparison |
| `compare_range_rate_physics.py` | Geometric range-rate vs counted-Doppler comparison |
| `real_visibility_bls_ukf_matrix.py` | SPICE visibility BLS/UKF matrix |
| `run_all_experiments.py` | Run manifest/orchestration helper |
| `scenario_config_cli.py` | Scenario config schema and validation CLI |

Results are written under:

```text
python_port/results/
```

The UI should treat this directory as the local result database.

---

## 3. Core Scientific Pipeline

The local desktop app should expose this pipeline visually and interactively:

```text
Scenario settings
      |
      v
SPICE kernel loading
      |
      v
Truth trajectory propagation
      |
      v
Ground-station visibility analysis
      |
      v
Tracking arc construction
      |
      v
Synthetic measurement generation
      |
      v
BLS-LM / SR-UKF estimation
      |
      v
Diagnostics, plots, CSV summaries
```

Each block should be visible in the UI as a step in a left-to-right or
top-to-bottom workflow. The user should always know:

- what scenario is selected,
- what data has been generated,
- which estimator has been run,
- where the outputs were written,
- and which plots/tables correspond to the current run.

---

## 4. Important Scientific Concepts to Show in the UI

### 4.1 Coordinate Frames

The project uses a Moon-centered inertial frame aligned with J2000 axes. It is
not a separate SPICE-native named frame; it is a translated inertial construction
with the Moon at the origin.

Other frames involved:

- `MOON_PA`: lunar principal-axis body-fixed frame
- `J2000`: inertial frame
- `ITRF93`: Earth-fixed frame used for station coordinates
- local `SEZ`: South-East-Zenith station frame

UI visualization idea:

- A small "Frame Chain" panel:

```text
MOON_PA initial orbit
      -> Moon-centered J2000 propagation
      -> Earth-centered inertial station geometry
      -> ITRF93 station frame
      -> SEZ measurement frame
```

### 4.2 Dynamics

Baseline acceleration:

- Moon point-mass gravity
- differential third-body perturbations from Earth and Sun

Optional / future:

- lunar J2 helper
- high-degree lunar gravity / mascons
- solar-radiation pressure
- maneuvers
- clock/media/model mismatch

UI visualization idea:

- Toggleable force-model cards:
  - `Moon point mass` enabled
  - `Earth third body` enabled
  - `Sun third body` enabled
  - `Lunar J2` optional/experimental
  - `SRP / mascons / maneuvers` future/disabled

### 4.3 Visibility

Ground station visibility uses:

- station elevation mask
- lunar occultation test
- optional gap stitching
- single-station or multi-station network combination

Important visibility products:

- station-level visible windows
- network-level visible windows
- raw vs gap-stitched coverage
- tracking arcs
- measurement counts per arc

UI visualization ideas:

- Gantt chart timeline
- station row heatmap
- network availability row
- gap-stitching before/after overlay
- arc duration histogram
- measurement count histogram
- station handover animation

### 4.4 Measurements

Supported measurement vectors:

1. Position-observation mode:

```text
[range, azimuth, elevation]
```

2. Range-rate mode:

```text
[range, range_rate, azimuth, elevation]
```

Range-rate physics options:

- instantaneous geometric range-rate
- simplified two-way counted Doppler

The two-way Doppler model is simplified. It includes iterative geometric
round-trip light time, fixed uplink frequency, and fixed turnaround ratio. It
does not include full operational DSN effects such as media corrections,
frequency ramps, hardware delays, or relativistic corrections.

UI visualization ideas:

- Measurement vector cards
- residual plots by channel
- observed vs computed measurement tracks
- azimuth/elevation polar plot
- range-rate time history
- Doppler light-time iteration count indicator

### 4.5 Estimators

#### BLS-LM

Batch estimator:

- waits for a complete tracking arc
- uses all measurements in that arc
- estimates the state at a reference epoch
- returns an end-of-arc solution
- uses Levenberg--Marquardt damping

Good for:

- post-processed arc refinement
- offline batch reconstruction

UI representation:

- show BLS markers only at arc ends
- show iterations per arc
- show cost decrease and damping behavior if available
- show condition number/rank warnings

#### SR-UKF

Sequential estimator:

- propagates sigma points
- updates after each visible measurement
- maintains state and square-root covariance
- propagates state/covariance through gaps

Good for:

- real-time or near-real-time navigation
- continuous state availability

UI representation:

- continuous state-error time history
- measurement-by-measurement update markers
- covariance envelope
- accepted/rejected measurement markers
- NIS time history
- update counter animation

---

## 5. Important Existing Result Sets

The UI should be able to browse these result folders and detect CSV/PNG outputs.

### 5.1 Baseline BLS/UKF

```text
python_port/results/baseline_bls_ukf/
```

Typical files:

```text
baseline_bls_ukf_28p0d_step600s_arc2p0h_stride6p0h_noisy_aggregate.csv
baseline_bls_ukf_28p0d_step600s_arc2p0h_stride6p0h_noisy_arc_summary.csv
baseline_bls_ukf_28p0d_step600s_arc2p0h_stride6p0h_noisy_comparison.png
```

Show:

- BLS vs SR-UKF median/p95/max position error
- runtime comparison
- operational success fraction
- arc-by-arc error plot

### 5.2 Sequential Tracking

```text
python_port/results/sequential_tracking/
```

Important files:

```text
sequential_tracking_vis_3p0d_step600s_noisy_cold_bls_arc_markers.csv
sequential_tracking_vis_3p0d_step600s_noisy_cold_ukf_updates.csv
sequential_tracking_vis_3p0d_step600s_noisy_cold.png
```

Show:

- BLS arc-end position error markers
- SR-UKF update-by-update position error
- measurement accepted fraction
- NIS time history
- covariance condition number
- visible vs gap intervals

### 5.3 Visibility Gantt

```text
python_port/results/visibility_28day_dsn_itu_gantt.png
python_port/results/visibility_28day_dsn_itu_summary.csv
```

Show:

- 28-day station visibility
- network coverage
- duty cycle per station
- raw vs gap-stitched network visibility

### 5.4 BLS Ablation

```text
python_port/results/bls_3day_ablation/
python_port/results/bls_7day_ablation/
```

Important files:

```text
bls_3day_ablation_aggregate.csv
bls_3day_ablation_detail.csv
bls_3day_arc_manifest.csv
```

Show:

- position-only vs range-rate
- geometric range-rate vs two-way Doppler
- cold vs hot vs formal handoff
- operation success fraction
- condition number
- runtime

### 5.5 Two-Way Doppler

```text
python_port/results/two_way_doppler_bls_ukf/
```

Important files:

```text
two_way_bls_ukf_24h_step60s_tc60s_taylor3_noisy_cold-formal_4arc_aggregate.csv
two_way_bls_ukf_24h_step60s_tc60s_taylor3_noisy_cold-formal_4arc_arc_summary.csv
```

Show:

- cold vs formal initialization
- BLS-LM vs SR-UKF
- model-evaluation count
- runtime
- light-time computational cost

### 5.6 Monte Carlo / Matrix

```text
python_port/results/baseline_bls_ukf_matrix/
```

Important files:

```text
baseline_matrix_gaussian_mc_20seed_1p0d_cold_summary.csv
baseline_matrix_gaussian_mc_20seed_1p0d_cold_trials.csv
baseline_matrix_initial_sweep_1p0d_cold_0p5-1p0-2p0-4p0_summary.csv
```

Show:

- seed-by-seed distribution
- box plots
- violin plots
- success fraction
- initial-error sensitivity

---

## 6. Desired Desktop Application Experience

The app should feel like a polished mission-analysis desktop tool:

- dark modern theme
- smooth page transitions
- animated side navigation
- responsive progress feedback
- interactive plots
- live job logs
- non-blocking simulation execution
- result browser
- export buttons
- scenario presets
- tooltips explaining OD terms

The app should not feel like a raw script launcher. It should guide the user
through:

1. selecting or creating a scenario,
2. running a simulation/estimation case,
3. viewing results,
4. comparing estimators,
5. exporting plots/tables.

---

## 7. Required UI Architecture

Use **PyQt5** with modular `.ui` files.

Recommended structure:

```text
python_port/desktop_app/
    README.md
    main.py
    app.py
    resources.qrc
    resources_rc.py
    requirements_desktop.txt

    ui/
        main_window.ui
        pages/
            dashboard_page.ui
            scenario_builder_page.ui
            dynamics_page.ui
            station_network_page.ui
            visibility_page.ui
            measurement_page.ui
            estimator_page.ui
            run_monitor_page.ui
            results_browser_page.ui
            comparison_page.ui
            diagnostics_page.ui
            settings_page.ui
        dialogs/
            scenario_preset_dialog.ui
            station_editor_dialog.ui
            export_dialog.ui
            about_dialog.ui

    controllers/
        main_window_controller.py
        dashboard_controller.py
        scenario_builder_controller.py
        dynamics_controller.py
        station_network_controller.py
        visibility_controller.py
        measurement_controller.py
        estimator_controller.py
        run_monitor_controller.py
        results_browser_controller.py
        comparison_controller.py
        diagnostics_controller.py
        settings_controller.py

    widgets/
        metric_card.py
        status_badge.py
        timeline_widget.py
        gantt_widget.py
        orbit_view_widget.py
        station_map_widget.py
        covariance_ellipse_widget.py
        log_console_widget.py
        result_table_widget.py
        plot_toolbar.py

    services/
        project_paths.py
        result_indexer.py
        scenario_service.py
        simulation_runner.py
        csv_loader.py
        plot_loader.py
        export_service.py
        spice_status_service.py
        validation_service.py

    workers/
        process_worker.py
        script_runner_worker.py
        csv_index_worker.py

    models/
        scenario_model.py
        station_model.py
        result_model.py
        estimator_model.py
        plot_model.py

    styles/
        dark_theme.qss
        light_theme.qss
        accent_theme.qss

    assets/
        icons/
        images/
        animations/
```

Important rule:

> `.ui` files define layout only. Python controller files own behavior,
> validation, plotting, subprocess execution, and result binding.

Do not put huge UI construction code directly in Python unless a custom widget
requires it.

---

## 8. Recommended Main Window Layout

### 8.1 Shell

Use a `QMainWindow` with:

- collapsible left sidebar
- top status bar / breadcrumb
- central `QStackedWidget`
- bottom job/status area

Sidebar sections:

```text
Dashboard
Scenario Builder
Dynamics
Stations & Network
Visibility
Measurements
Estimators
Run Monitor
Results
Comparisons
Diagnostics
Settings
```

Top bar:

- current project path
- active scenario name
- SPICE kernel status
- last run status
- theme toggle

Bottom bar:

- current job
- progress bar
- elapsed time
- cancel button
- log toggle

---

## 9. Page Specifications

### 9.1 Dashboard Page

Purpose:

Give a quick status overview.

Widgets:

- metric cards:
  - number of result folders
  - latest run time
  - latest PDF/report generated
  - SPICE status
  - available station count
  - test/result status
- recent result list
- quick actions:
  - run 1-day Gaussian comparison
  - run 3-day visibility comparison
  - open latest thesis PDF
  - open results folder

Visuals:

- latest baseline comparison PNG
- latest sequential tracking PNG
- mini visibility strip

Animation:

- metric cards fade in on app launch
- quick action buttons have hover glow
- result import spinner while indexing CSV files

### 9.2 Scenario Builder Page

Purpose:

Create and validate simulation/estimation cases.

Sections:

1. Scenario identity
   - name
   - description
   - output directory

2. Time settings
   - duration
   - state step
   - ephemeris sampling step
   - start epoch

3. Arc settings
   - prescribed arcs or visibility arcs
   - arc duration
   - stride
   - min visibility samples
   - max arcs

4. Noise and seed settings
   - random seed
   - noise on/off
   - range sigma
   - angle sigma
   - range-rate sigma
   - bias options

5. Execution preset
   - fast preview
   - thesis baseline
   - visibility test
   - Doppler test
   - Monte Carlo

Important behavior:

- Invalid settings should be highlighted immediately.
- Cross-field warnings should appear as yellow info panels.
- The generated scenario should be exportable as JSON.

### 9.3 Dynamics Page

Purpose:

Configure and explain the force model and propagation.

Controls:

- central Moon gravity checkbox
- Earth third-body checkbox
- Sun third-body checkbox
- optional J2 checkbox
- integrator selection display: `DOP853`
- relative tolerance
- absolute tolerance
- STM propagation on/off

Visuals:

- 3D Moon-centered orbit preview
- acceleration contribution chart
- Earth/Sun perturbing body direction indicators
- state norm vs time

Animation:

- play/pause orbit propagation
- moving spacecraft marker around the Moon
- optional Earth/Sun vector arrows

### 9.4 Stations & Network Page

Purpose:

Show and select ground stations.

Widgets:

- station table
- station detail card
- network selector
- station map
- station noise editor

Station table columns:

```text
Enabled
Name
Latitude
Longitude
Altitude
Range sigma
Angle sigma
Range-rate sigma
Family/Network
```

Visuals:

- world map with station pins
- station coverage color coding
- selected network summary

Animation:

- station pins pulse when selected
- network handover preview animates along timeline

### 9.5 Visibility Page

Purpose:

Analyze station visibility and tracking arcs.

Controls:

- elevation mask
- lunar occultation on/off
- gap stitching on/off
- gap threshold
- min samples per arc
- station network selection

Visuals:

- station visibility Gantt chart
- raw vs stitched network row
- arc duration histogram
- measurement count histogram
- visibility duty-cycle table

Important existing result:

```text
python_port/results/visibility_28day_dsn_itu_gantt.png
```

Animation:

- time cursor moving across Gantt chart
- station handover highlight
- arcs fade in/out as gap threshold changes

### 9.6 Measurements Page

Purpose:

Configure synthetic measurements and inspect measurement data.

Controls:

- measurement type:
  - position
  - range-rate
- range-rate physics:
  - geometric instantaneous
  - two-way counted Doppler
- count interval
- uplink frequency
- turnaround ratio
- noise values
- bias mode

Visuals:

- range vs time
- azimuth/elevation vs time
- range-rate vs time
- observed minus computed residuals
- measurement covariance diagonal

Warnings:

- two-way Doppler is simplified
- media/clock/relativity are not operationally modeled

### 9.7 Estimators Page

Purpose:

Configure BLS-LM and SR-UKF.

BLS-LM controls:

- max iterations
- LM damping initial/min/max
- convergence tolerance
- prior covariance
- state scaling
- outlier rejection toggle

SR-UKF controls:

- alpha
- beta
- kappa
- initial covariance
- process noise Q
- measurement covariance R
- gating mode
- NIS threshold
- covariance inflation

Start modes:

```text
cold
hot
formal
sqrt_formal
```

Visuals:

- BLS flow diagram:
  - propagate
  - compute residuals
  - build Jacobian
  - solve LM step
  - accept/reject

- SR-UKF flow diagram:
  - sigma points
  - propagation
  - measurement transform
  - innovation
  - update
  - covariance factor

Animation:

- sigma-point cloud around spacecraft state
- BLS batch arc fills over time, then solves at arc end
- SR-UKF update pulse at each measurement epoch

### 9.8 Run Monitor Page

Purpose:

Launch scripts and monitor execution.

Implementation:

Use `QProcess` or `QThread` workers. Never block the main UI thread.

Features:

- script selection
- command preview
- environment/path display
- start/cancel buttons
- progress indicator
- stdout/stderr console
- elapsed time
- output folder link

Suggested run targets:

```text
python python_port/examples/baseline_bls_ukf_comparison.py
python python_port/examples/sequential_tracking_comparison.py
python python_port/examples/visibility_28day_dsn_itu_gantt.py
python python_port/examples/two_way_doppler_bls_ukf_comparison.py
python python_port/examples/bls_7day_ablation_appendix.py
python python_port/examples/run_all_experiments.py --list
```

### 9.9 Results Browser Page

Purpose:

Browse generated CSV/PNG outputs.

Features:

- result folder tree
- file preview
- CSV table viewer
- PNG viewer
- metadata panel
- open in file explorer
- export selected files

CSV handling:

- load with pandas
- show row/column count
- allow sorting/filtering
- numeric columns should support quick histogram/line plot

PNG handling:

- zoom/pan
- fit to page
- open external

### 9.10 Comparison Page

Purpose:

Compare estimators, measurements, start modes, and networks.

Comparison types:

- BLS-LM vs SR-UKF
- cold vs hot vs formal
- position-only vs range-rate
- geometric range-rate vs two-way Doppler
- single station vs multi station
- raw visibility vs gap-stitched visibility

Visuals:

- grouped bar charts
- line plots
- box plots
- p95/max error cards
- runtime comparison
- success fraction
- condition number

Important metrics:

```text
median position error
p95 position error
max position error
median velocity error
p95 velocity error
runtime
algorithmic success
operational success
NIS
NEES where available
accepted update fraction
condition number
rank
measurement model evaluations
```

### 9.11 Diagnostics Page

Purpose:

Expose technical OD health metrics.

Plots:

- residual RMS
- whitened residual RMS
- reduced chi-square
- NIS time history
- NEES time history where available
- covariance condition number
- innovation lag-one correlation
- rank and singular values
- accepted/rejected measurements

Advanced panels:

- observability spectrum
- Fisher information condition
- station-bias recovery
- arc-boundary jumps
- tail arc analysis

---

## 10. Animation Ideas

The app should be visually modern but not gimmicky. Animations should explain
the OD process.

Recommended animations:

1. **Orbit Playback**
   - spacecraft moves around the Moon
   - trail fades behind it
   - ground station line-of-sight appears when visible

2. **Visibility Handover**
   - time cursor moves over Gantt chart
   - active station row glows
   - network row turns bright when any station is visible

3. **BLS Batch Arc Fill**
   - measurement points accumulate inside an arc
   - BLS solution appears only when arc closes
   - final correction vector pulses

4. **SR-UKF Sequential Update**
   - predicted state propagates
   - measurement arrives
   - innovation arrow appears
   - covariance ellipse shrinks or rotates

5. **Sigma-Point Cloud**
   - sigma points spread around state
   - propagate forward
   - recombine into predicted mean

6. **Doppler Light-Time**
   - station sends uplink
   - signal reaches spacecraft
   - downlink returns
   - round-trip light-time change becomes counted Doppler

7. **Error Timeline**
   - BLS arc-end dots appear discretely
   - SR-UKF line updates continuously
   - gaps shown as shaded intervals

Implementation options:

- `QPropertyAnimation` for page transitions, sidebar, card hover effects
- `QTimer` for orbit/time playback
- `pyqtgraph` for fast live plots
- Matplotlib canvas for thesis-style static plots
- Optional `PyOpenGL`, `PyVista`, or `VisPy` only if needed for high-quality 3D

---

## 11. Plotting Technology Recommendation

Use a hybrid plotting strategy:

### Fast interactive plots

Use:

```text
pyqtgraph
```

Good for:

- time histories
- NIS plots
- covariance condition plots
- real-time run monitor plots
- residual plots

### Publication-style plots

Use:

```text
matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg
```

Good for:

- exact reproduction of thesis figures
- saved PNG previews
- comparison tables/figures

### 3D orbit view

Start simple:

- `matplotlib` 3D orbit preview is acceptable for v1.

Optional later:

- `PyVistaQt`
- `VisPy`
- `QOpenGLWidget`

Do not make the first version depend on complex 3D unless the rest of the app is
stable.

---

## 12. Backend Integration Strategy

The UI should not rewrite all scientific code. It should wrap the existing
Python modules and scripts.

### Phase 1: Result Viewer

No simulation execution yet.

Implement:

- result directory indexing
- CSV loading
- PNG viewing
- dashboard cards
- comparison plots from existing CSV files

This gives immediate value and avoids breaking scientific code.

### Phase 2: Script Launcher

Add:

- run selected example scripts
- show stdout/stderr
- show progress/elapsed time
- refresh results when finished

Use `QProcess` or worker threads.

### Phase 3: Scenario Builder

Add:

- scenario JSON generation
- config validation
- preset management
- eventually connect generated config to a full runner

### Phase 4: Interactive Simulation

Add:

- live trajectory propagation preview
- live visibility recomputation
- live measurement preview
- optional estimator preview for short runs

---

## 13. Scenario Presets

The UI should ship with presets. Each preset should show a short explanation,
estimated runtime, and expected outputs.

### Preset: 1-Day Gaussian Baseline

Purpose:

Compare BLS-LM and SR-UKF under controlled prescribed arcs.

Expected outputs:

- aggregate CSV
- arc summary CSV
- BLS vs SR-UKF comparison PNG

### Preset: 28-Day Prescribed-Arc Stability

Purpose:

Long-duration numerical stability and estimator comparison.

Expected outputs:

- 112 synthetic arcs
- median/p95/max errors
- runtime comparison

### Preset: 28-Day Network Visibility

Purpose:

Show station visibility and network handovers for ITU + DSN stations.

Expected outputs:

- Gantt chart
- duty-cycle table
- raw vs stitched network availability

### Preset: 3-Day Fragmented Visibility

Purpose:

Compare BLS-LM arc-end estimates and SR-UKF sequential history under realistic
SPICE visibility fragmentation.

Expected outputs:

- BLS arc marker CSV
- UKF update CSV
- sequential tracking plot
- arc duration/measurement summary

### Preset: BLS Range-Rate Ablation

Purpose:

Study position-only vs range-rate, geometric RR vs two-way Doppler, and
cold/hot/formal starts.

Expected outputs:

- aggregate table
- detail table
- arc manifest

### Preset: Two-Way Doppler Cost Test

Purpose:

Evaluate simplified two-way counted-Doppler capability and computational cost.

Expected outputs:

- BLS/UKF cold/formal comparison
- process and measurement model evaluation count
- runtime

---

## 14. Data Models for the UI

### ScenarioModel

Fields:

```python
name: str
description: str
duration_days: float
state_step_s: float
ephemeris_step_s: float
arc_mode: str
arc_duration_h: float | None
arc_stride_h: float | None
visibility_enabled: bool
elevation_mask_deg: float
gap_threshold_s: float
min_visibility_samples: int
network_name: str
station_ids: list[int]
measurement_type: str
range_rate_physics: str
count_interval_s: float
noise_enabled: bool
random_seed: int
estimator_type: str
start_mode: str
output_dir: str
```

### StationModel

Fields:

```python
station_id: int
name: str
latitude_deg: float
longitude_deg: float
altitude_m: float
range_sigma_m: float
angle_sigma_rad: float
range_rate_sigma_mps: float
enabled: bool
network_group: str
```

### ResultModel

Fields:

```python
result_id: str
folder: Path
created_time: datetime
scenario_name: str
csv_files: list[Path]
png_files: list[Path]
aggregate_csv: Path | None
arc_summary_csv: Path | None
plot_png: Path | None
status: str
```

### EstimatorSummaryModel

Fields:

```python
estimator_name: str
median_position_error_m: float
p95_position_error_m: float
max_position_error_m: float
median_velocity_error_mps: float | None
runtime_s: float
operational_success_fraction: float
algorithmic_success_fraction: float | None
median_nis: float | None
condition_number: float | None
```

---

## 15. Styling Requirements

Recommended default theme:

- background: deep navy / near black
- panels: slightly lighter charcoal
- accent: cyan or moon-gold
- danger: red/orange
- success: green
- text: soft white
- muted text: grey-blue

Style principles:

- rounded cards
- generous spacing
- subtle shadows
- consistent iconography
- no cluttered scientific control panels
- advanced controls collapsible

Use QSS files:

```text
styles/dark_theme.qss
styles/light_theme.qss
styles/accent_theme.qss
```

Important widgets should have object names in `.ui` files so QSS can style them:

```text
sidebar
topBar
metricCard
primaryButton
secondaryButton
dangerButton
statusBadgeSuccess
statusBadgeWarning
statusBadgeError
plotPanel
```

---

## 16. Required Non-Blocking Execution

Some simulations are slow. The UI must remain responsive.

Use:

- `QProcess` for launching example scripts as subprocesses
- or `QThread` workers for Python API calls

Do not:

- call long-running scripts directly on the GUI thread
- freeze the UI while loading large CSV files
- block the event loop while generating figures

The run monitor should capture:

- command
- start time
- elapsed time
- stdout
- stderr
- return code
- output folder

Add a Cancel button. If cancellation is implemented with `QProcess.kill()`,
clearly warn that partial result files may remain.

---

## 17. File and Path Handling

The app should auto-detect the project root.

Expected root:

```text
C:\Users\erayh\Documents\Python\Grad
```

But do not hard-code this path. Use relative discovery:

```text
project_root/
    python_port/
        lunar_od/
        examples/
        results/
```

If the app starts from inside `python_port/desktop_app`, it should still find:

```text
../lunar_od
../examples
../results
```

Settings should remember:

- project root
- last output directory
- last selected result folder
- theme
- recent scenarios

Use:

```python
QSettings
```

---

## 18. Error Handling and User Messages

The UI should explain scientific and technical failures clearly.

Examples:

| Failure | User-facing message |
|---|---|
| SPICE kernels missing | "SPICE kernel set not found. Please configure the kernel directory." |
| CSV schema unexpected | "This result file does not match a known Lunar OD output format." |
| Simulation process failed | "The run exited with code X. See log panel for stderr." |
| No OD-ready arcs | "Visibility produced no arcs satisfying the minimum sample rule." |
| Light-time non-convergence | "Two-way Doppler light-time iteration failed for one or more measurements." |
| Singular batch solve | "BLS-LM encountered a rank or conditioning failure on this arc." |
| Covariance factor invalid | "SR-UKF covariance factor became non-positive definite." |

Warnings should be visible but not scary when they describe known limitations:

- "Two-way Doppler is simplified."
- "Operational DSN media corrections are not modeled."
- "NEES is only available when truth and covariance histories are present."

---

## 19. Export Features

The app should export:

- selected plots as PNG
- selected tables as CSV
- scenario config as JSON
- run summary as Markdown
- selected report bundle as ZIP

Recommended export bundle:

```text
exported_case/
    scenario.json
    aggregate.csv
    arc_summary.csv
    update_history.csv
    figures/
    notes.md
```

---

## 20. Minimum Viable PyQt5 App

The first useful version should include:

1. main window with sidebar
2. dashboard
3. result browser
4. CSV table viewer
5. PNG viewer
6. baseline comparison plot from CSV
7. sequential tracking viewer
8. script launcher with live log
9. scenario preset list
10. dark theme

Do not start with a full custom 3D engine. Build the stable result viewer and
launcher first.

---

## 21. Advanced Version

After the MVP:

- interactive orbit animation
- visibility timeline with moving cursor
- station map
- sigma-point animation
- covariance ellipse animation
- drag-and-drop result folders
- scenario comparison workspace
- Monte Carlo distribution explorer
- Q/R tuning dashboard
- batch report generator
- LaTeX report figure picker

---

## 22. Suggested Dependencies

Base:

```text
PyQt5
numpy
scipy
pandas
matplotlib
pyqtgraph
spiceypy
```

Optional:

```text
PyOpenGL
pyvista
pyvistaqt
vispy
qtawesome
darkdetect
```

Keep optional 3D dependencies isolated. The app should still launch without
advanced 3D packages.

---

## 23. Implementation Rules for Claude or Any Coding Agent

Follow these rules:

1. Use PyQt5, not a web framework.
2. Keep `.ui` files modular.
3. Keep controllers separate from `.ui` files.
4. Do not rewrite scientific backend logic unless necessary.
5. Wrap existing scripts and result files first.
6. Never block the GUI thread.
7. Use explicit models/services/workers.
8. Save settings with `QSettings`.
9. Prefer pandas for CSV result loading.
10. Prefer pyqtgraph for live/interactive plots.
11. Prefer Matplotlib canvas for thesis-style static figures.
12. Make every long-running action cancellable or visibly monitored.
13. Provide clear error messages.
14. Add tooltips for technical OD terms.
15. Keep future/unsupported physics visibly disabled, not silently hidden.

---

## 24. Glossary for UI Tooltips

| Term | Tooltip |
|---|---|
| BLS-LM | Batch Least-Squares estimator with Levenberg--Marquardt damping |
| SR-UKF | Square-Root Unscented Kalman Filter |
| Arc | A continuous or stitched interval of available tracking measurements |
| Cold start | Each arc starts from an independent initial perturbation |
| Hot start | Previous estimated state is propagated to seed the next arc |
| Formal handoff | Previous covariance/information is also transferred |
| NIS | Innovation consistency statistic for sequential measurements |
| NEES | State-error covariance consistency statistic when truth is known |
| Range-rate | Line-of-sight relative velocity observable |
| Two-way Doppler | Simplified counted-Doppler observable based on round-trip light-time change |
| Occultation | Moon blocks the station-to-spacecraft line of sight |
| Gap stitching | Short visibility outages are bridged to create longer arcs |
| Condition number | Numerical indicator of how ill-conditioned the estimation system is |

---

## 25. Final Product Vision

The final desktop application should let a user open the Lunar OD project and
understand it visually within a few minutes:

- see the lunar orbit,
- see which ground stations can track the spacecraft,
- see when measurements exist,
- see how BLS-LM and SR-UKF behave differently,
- see where the error tail comes from,
- see how range-rate and Doppler change the problem,
- run common experiments without remembering command-line scripts,
- and export clean plots/tables for thesis or presentation use.

The app is not just a skin over scripts. It should become a local mission
analysis cockpit for the project: smooth, modern, scientific, and honest about
the limits of the current models.

