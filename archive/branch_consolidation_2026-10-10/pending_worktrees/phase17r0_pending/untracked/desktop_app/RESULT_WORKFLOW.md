# Native scenario workflow

Launch `python desktop_app/main.py` with the desktop dependencies installed.
Choose Scenario Builder, configure a scenario, and press Run Scenario.
The form is saved in a unique `desktop_runs/<name>_<id>` folder below the chosen
output directory. Relative output paths resolve against this checkout.

Run Monitor launches `desktop_app/scenario_job.py` via QProcess. The child validates
the form-to-backend translation with the canonical ScenarioConfig loader, then calls
`examples/run_scenario_config.py::run_configured_scenario`. The UI stays responsive.
When the child writes a completion record and exits successfully, the console opens
that exact run. Disable “Tamamlanınca konsolda aç” to stay in Run Monitor.

## Settings

Days become duration_h; state_step_s becomes sample_step_s; noise_enabled becomes
noise; network_name becomes network. The form seed is passed explicitly as both
measurement_seed and cold_start_seed. Both seeds are persisted, not inferred.
Numerical and UKF tuning fields survive form roundtrips and are passed to the backend.
The backend validates supported estimator/measurement combinations.

Only the canonical single/multi station networks are executable by this runner.
Custom station networks fail with an explicit message.
Visibility mode uses the configured elevation mask and gap threshold.
Prescribed mode requests planned time windows but retains the **actual** station
visibility mask: invisible stations never generate fake observations. Adjacent
closed windows do not duplicate a boundary sample. In this mode the gap threshold
does not determine the planned window boundaries.

Desktop scheduling options are opt-in keyword arguments. Existing calls that omit
desktop_options keep their existing scheduling and numerical defaults. The fixture
is resolved from the producing checkout instead of the process's current directory.

## Output bundle

- job.json: run ID, original form, mapped inputs, scheduling settings and seeds.
- effective_config.json: backend-validated complete config, including defaults.
- `<name>_force_model_manifest.json`: existing canonical force-model provenance.
- `<name>_summary.csv` and comparison PNG: existing report outputs.
- `<name>_arc_solutions.csv`: one recorded estimate per solved arc.
- `<name>_truth_states.csv`: stored reference trajectory from the actual propagation.
- `<name>_result_manifest.json`: units, frame, epoch, producer, seed, source metadata.
- completed.json: explicit completion protocol for this run.

The manifest records HEAD and whether the checkout has local changes. Selected code
file hashes are captured at artifact serialization; this is not a complete runtime
or environment snapshot. The source tab includes hashes of the persisted job,
effective config and force contract.

## Scientific display contract

BLS/SRIF arc state/error belongs to the start epoch. UKF arc final state/error belongs
to the last observation, which may precede the visibility window endpoint. The
console displays the recorded epoch and never re-propagates a state to another time.
Reference trajectory and arc estimates are labeled separately. An arc-mean NIS is
never labeled as an individual update NIS. Backend operational_success=False stays
false, even if the process exits successfully. Full online UKF update history is
not available from this scenario runner; the existing *_ukf_updates.csv format
continues to support such histories.

Cancellation and failed process starts restore the controls. Failed or cancelled
runs cannot trigger automatic result opening. Partial output folders are retained
for diagnosis. A manually opened arc CSV without completed.json is marked as a
possibly partial result. Source files are read-only in the console.

## Real-run archive

Choose **Koşu arşivi** from the sidebar or Result Console. The default root is
this checkout's `results/`; choose another root for outputs written elsewhere.
The recursive index recognizes only `*_arc_solutions.csv` and
`*_ukf_updates.csv`. It indexes CSV structure and metadata in a background task;
numeric values and source hashes are validated when opening/comparing a run.
The index never loads trajectories eagerly. Refresh runs again on page entry.

Search includes run name, UUID, path, network and seed. Filters cover estimator,
record kind and process status; sorting/filtering retains exact source identity.
Names need not be unique: the UUID and path distinguish repeated runs. The details
pane exposes the stored metadata, effective configuration and scheduling options.

Process status means completion protocol present/missing, legacy provenance or
index error. **Completed does not mean operational_success=True.** Missing
completion and invalid/legacy artifacts remain visible. A moved folder is accepted
only when its completion UUID/result basename match; its old path is explicitly
noted. There is no archive delete/edit operation. Cancelling a scan or encountering
an error preserves the previous index; stale worker responses are ignored.

Select one row to open the source in Result Console; Ctrl-select two rows to open
**Karşılaştırma**. The previous CSV summary utility remains reachable through
**Diğer CSV özetleri**. Archive/comparison pages load only when visited.

## Recorded A/B comparison

Choose two distinct result files or send two archive rows. A/B source loading,
hashing, validation and export use background tasks. Failed/cancelled replacement
loads preserve the last accepted comparison. Edited selectors do not relabel the
currently displayed data. Each source can be reopened in Result Console.

The screen provides charts, exact matched rows, statistics, settings differences,
source snapshots and interpretation limits. The selected configuration variable
is marked in the difference table; other differences remain visible. Configuration
name/output path are ignored as comparison variables; source/job hashes, seeds,
code hashes, measurement/force settings and schedule differences are retained.

Pairing requires known/equal epoch, time scale, frame, origin and record kind, plus
matching station ID conventions. A pair uses **exact t_s, arc_id and station key**;
different epochs are never interpolated or propagated. Duplicate keys are excluded
as ambiguous. Update history and final arc solutions cannot pair. If station ID
conventions are absent on both histories, that uncertainty is explicitly noted.
Batch station keys use the recorded observed_station_ids field.

Compatible sources share chart axes. Incompatible references use separately
labeled axes and disable record deltas. Unknown/different NIS scopes use separate
panels; absent NIS stays absent. Batch solutions are discrete markers, not a fake
continuous trajectory. Plot decimation keeps pixel extrema and source gaps.
Matched row/chart selection uses original record indices. Small windows scroll
the chart area instead of cropping axes. No scientific plotting library is loaded
for these pages.

Median, p95 (linear order-statistic quantile), max and RMS of recorded error norms
are shown with N for **all records** and **exactly paired records** separately.
CSV delta is B minus A of the recorded scalar error norms, not a Cartesian state
difference. NIS/NEES acceptance or algorithm ranking is not computed. Measurement
realizations and initial covariance artifacts are not available to establish full
fairness. A single arc/seed cannot establish statistical significance; small N
is explicitly called out.

Export writes `lunar-od-recorded-comparison-v1` JSON with units, snapshots, metadata,
source SHA-256, notes, differences and statistics, plus a `_paired.csv` only when
N>0. Source overwrite is rejected; zero-pair export rejects an existing stale
paired CSV name. Outputs are staged before replacement; cancelled staging leaves
previous exports intact. Each file is replaced individually (the two files are
not a transactional filesystem bundle). Sources changed during loading, changed
between snapshot/hash reads, or violating a persisted job SHA-256 are rejected.

Targeted verification: `tests/test_recorded_run_workspace.py` plus the console,
desktop workflow and existing config/UI regression modules. These are frontend
and artifact-contract checks; full estimator/scientific acceptance campaigns
remain separate.


## Derive a new scenario from a native run

Select one native archive entry and choose **Koşuyu temel al**. A background
worker verifies result sources, manifest/job hashes, the complete typed native
form and its agreement with the stored job. It also compares the saved effective
configuration against the current canonical configuration. Unsupported fields,
missing values and changed defaults produce an explicit error instead of guessed
settings. Legacy CSVs without a native job cannot supply a complete editable form.

The recovered form opens as a new draft with a copied name and source banner.
**Önceki taslağa dön** restores the draft that was open before recovery. Choosing
a preset or importing another form clears the recovered origin; ordinary edits
retain it. Opening a draft does not run a solver. Starting it creates a new UUID
and output directory, preserving the original source files. The new job records
the parent UUID, verified sources, original form and each changed form field.
The archive detail panel displays the parent and changed fields; searching a
parent UUID also finds its derived runs.

Numeric inputs expand to represent saved ranges and precision. Seeds support
the complete uint32 range. Execution still uses the current code and environment:
recovery is a new run based on verified saved settings, not a historical runtime
snapshot or guarantee of bitwise reproducibility.

## Save and reopen a comparison workspace

The comparison export now includes the active tab, axis mode, paired-only mode,
time range and selected source record IDs. Choose **Kayıtlı karşılaştırma aç** to
reopen its JSON report in a fresh application window. Loading verifies the full
source set, SHA-256 hashes and byte sizes, then recomputes pairs and statistics
from those sources. Persisted metric values are not trusted as calculation input.
A selection absent from the verified pairs is rejected.

Relative source paths accompany original absolute paths. A report and its source
directory tree can move together if their relative layout is retained. Older
reports containing only absolute paths still work. The report is not a
self-contained source archive: its source files and sidecars must remain
available. Missing or changed sources, invalid views and cancellation preserve
the previously accepted comparison. Loading/hashing/export run off the UI thread.

Verification for this stage: `tests/test_run_workspace_resume.py`, the existing
archive/comparison/console/config/UI tests, and a real native run followed by
export and reopening in a fresh MainWindow. These checks establish frontend and
artifact workflow behavior; they do not establish scientific estimator accuracy.


## Unified research workflow

The sidebar presents Senaryo, Çalıştır, Sonuçlar, Arşiv and Karşılaştırma in
workflow order. Alt+1 through Alt+5 select these pages. Advanced tools remain
available in a collapsed section; opening a tool programmatically expands it.
Expanding the section does not construct its pages. A button group keeps one
active route across the two sidebar levels. Settings remains separately available.
Page construction errors have a retry action.

Results includes a readable run inspector beside the existing linked plots:
identity, recorded settings, frame/epoch/units, source lineage and field changes,
source hashes and notes. Technical JSON remains available in a secondary tab.
The archive provides a readable summary and opens the full inspector. Details
are read/verified on the result worker, not during paint or navigation.
Completion status validates the completion UUID/result basename separately; it
does not imply estimator success. Missing provenance is explicitly unknown.
Failed or cancelled replacements retain the accepted charts, inspector and
source path. Compact windows scroll the plot area.

Archive, comparison, scenario, run monitor and result inspector share the
neutral workspace control styles. The log keeps severity text markers. Empty
logs provide a next action; inputs have visible focus and usable text height.

TrackingChart caches one background pixmap per chart. The selected-record
overlay remains live. The cache is invalidated by result/filter tuple identity,
time range, scale, title, size or device pixel ratio. Input references are retained
to prevent identity reuse. Source arrays, extrema selection and numerical values
are unchanged. Cache-vs-fresh pixel parity tests cover selection, filters, scale
and resize. The 20,000-row before/after benchmark also retained an identical chart
pixel SHA-256. Local timing is observational and is not a broad startup or solver
speed claim. See tests/test_workflow_polish.py for focused behavioral regressions.

## Modern / Orchid appearance

The default Modern theme uses the Sunumatik palette gallery's white/plaster
surfaces and signal-red accent. Orchid uses Berry Wine / Orchid Mist.
Appearance is stored under app_settings/appearance in the existing QSettings.
Save the choice on Settings and reopen the app; changing a preference never
reconstructs an active scenario or interrupts a worker. Unknown values use Modern.
Both main.py and the standalone result_console.py resolve the theme before
importing page or plot modules. The legacy MPL_DARK name is retained for callers,
but its colors follow the selected light theme.

The results page separates charts and recorded rows into tabs. Filters and the
inspector are expandable, retaining selection, source rows and filter values.
The filter button shows the visible/total count when rows are excluded.
The scenario form groups arc and measurement fields under Advanced settings.
JSON preview is optional; run/save/import/analysis actions, source provenance
and draft recovery remain independently accessible.

Verification: 117 existing targeted UI/artifact tests passed via an isolated
test directory (identical test sources, this checkout on PYTHONPATH) to avoid
the repository-wide external campaign import in tests/conftest.py. Both themes
were additionally rendered with an existing real arc result and comparison.
These checks do not assess estimator physics or certify all legacy scientific
pages' visual layouts.
