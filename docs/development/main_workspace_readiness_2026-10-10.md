# Main workspace readiness — 2026-10-10

This follow-up starts from consolidated main commit
`4f7ee7d6c0b01562ad440880e8eed0715fb03cc0`. It makes a new checkout's
empty archive usable and records an installable Windows reference environment.

## Changes

- An absent default results directory is an empty archive with next-step guidance.
  Opening the archive does not create directories.
- A different missing archive directory remains an explicit error. Previous rows
  remain available after an unsuccessful replacement scan.
- `requirements-validated-windows.txt` combines the core, acceleration,
  desktop and test dependencies with pinned reference versions.
- Numerical sources under `lunar_od/` and `examples/` have no changes
  from the base commit. Reference tolerances and frozen provenance stay unchanged.

## Recreate the environment

Use Windows x86_64 and CPython 3.13. The recorded run uses Python 3.13.2.
Start in the checkout root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validated-windows.txt
.\.venv\Scripts\python.exe -m pip check
```

This profile pins direct dependencies and the Qt runtime. It is not a universal
cross-platform lockfile: the [validation record](main_workspace_validation_2026-10-10.json)
also lists the complete resolved environment. No global site packages were visible
to the clean venv, and `pip check` reported no broken requirements.

SPICE kernels, optional gravity files and historical campaign data are external
inputs, not installed by pip. This verification reuses existing project data.
A fresh clone without those inputs cannot reproduce all data-dependent campaigns.
See the SPICE setup in [README](../../README.md#spice-kernel-setup) and the earlier
[consolidation validation](branch_consolidation_2026-10-10/validation.json).
Windows `core.autocrlf=true` matches the recorded reference checkout.

## Verification

| Check | Result |
| --- | --- |
| Full suite in isolated venv | 1747 passed, 29 skipped, zero failures/errors |
| Focused font-enabled rendering/layout suite | 15 passed |
| Targeted archive/session/performance regression suite before clean install | 48 passed |
| Real native workflow | Two completed jobs; clone, comparison, export and reopen passed |
| Existing archive | All eight runs opened; 134 source files unchanged |
| Archive session | Chosen root restored after normal shutdown |

The two real workflow jobs deliberately use a short six-hour, one-arc scenario
with three BLS iterations. Both processes completed, and the recorded
`operational_success=False` remained false. The workflow check establishes
artifact and UI behavior, not BLS convergence or estimator accuracy. Each of the
eight existing runs also contains only one recorded arc.

The initial full suite used Qt's offscreen backend. On Windows that backend needs
an explicit font directory for meaningful text rendering; the focused rendering
suite and visual archive review were repeated with Windows fonts. Use this setting
when reproducing visual checks:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = Join-Path $env:WINDIR "Fonts"
.\.venv\Scripts\python.exe -m pytest -p tests.desktop_preferences tests -q
```

The preferences plugin isolates QSettings. The native workflow and archive checks
also used private settings. Normal application launch should use the native Windows
Qt backend, without the offscreen environment setting.

The exact skipped tests and reasons, resolved versions and JUnit hashes are in
the [machine-readable record](main_workspace_validation_2026-10-10.json).
Optional gravity/store inputs and opt-in slow regressions remain outside this run.
A long real-run and manual desktop acceptance pass remains outstanding.
Local timing under concurrent tests does not establish overall solver/startup speed.

## Reuse earlier results

In **Arşiv**, choose the results directory from the earlier checkout. It is indexed
and opened in place, and its path is saved after a normal close. No copying, moving
or rewriting of result files is required. The UI validates each source through the
normal loaders. See [result workflow](../../desktop_app/RESULT_WORKFLOW.md).
