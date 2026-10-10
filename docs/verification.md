# Verification

## Public checks on each push / PR

[Public checks workflow](../.github/workflows/public-checks.yml) runs on Windows
with Python 3.13.2 and the validated pinned dependencies. It checks installation,
then calls [run_public_checks.py](../tools/run_public_checks.py).

The runner selects the documented test modules in its TEST_FILES list. It clears
PYTHONPATH and redirects project, campaign, gravity and kernel overrides to
empty temporary directories. Tests use committed fixtures, not developer-local
campaign files or SPICE kernels.

Scope: core configuration, geometry, orbit and baseline dynamics; typed scenario
configuration; desktop process lifecycle; source integrity; archive/derivation;
comparison/export/reopening; rendering, layouts, preferences and sessions.

    python tools/run_public_checks.py --junitxml outputs/public-checks.xml

JUnit reports are uploaded even when the test step fails. The job has read-only
repository permission. Official GitHub Actions are pinned to exact commits.

This gate is a bounded public regression set. It is not the full scientific
qualification suite, a long real mission run, or proof of estimator convergence.
The checked-in numerical assertions and tolerances are unchanged.

## Full local research regression

[Latest clean-environment local record](main_workspace_readiness_2026-10-10.md):
1,747 passed and 29 skipped. The machine-readable report lists versions, skips
and JUnit hashes. That execution reused existing external project data.

From the repository root with the validated environment and required data:

    python -m pytest -p tests.desktop_preferences tests --durations=15 -q

For headless Windows rendering, set QT_QPA_PLATFORM=offscreen and point
QT_QPA_FONTDIR at the Windows Fonts directory. The preferences plugin isolates
QSettings. Normal desktop launch uses the Windows Qt backend.

| Environment variable | External input |
| --- | --- |
| LUNAR_OD_KERNEL_DIR | SPICE kernels |
| LUNAR_OD_PROJECT_ROOT | Parent project data, including historical ephemeris data |
| LUNAR_OD_CAMPAIGN_ROOT | Read-only od_covariance_campaign tree |
| LUNAR_OD_GRAVITY_ROOT | High-degree gravity coefficient files |

The full suite includes Git-history provenance checks: use a Git clone with
history for those checks. Source ZIP packages do not include Git history.
A clone with no external data cannot reproduce every scientific campaign.
Do not suppress collection errors or widen tolerances to make a data-free run green.

## Long campaigns and release checks

Reduced Monte Carlo, long arcs, high-degree gravity and external comparisons
belong in explicit scientific campaigns with aligned assumptions and saved
configs/seeds. Existing slow regressions remain opt-in through
LUNAR_OD_RUN_SLOW_TESTS where supported. They are not run on every commit.

Before a source prerelease, verify the public gate, setup from an extracted clean
source folder, desktop imports and screenshot. Tag the exact tested commit.
Keep kernels, local runs, venvs and logs outside the source archive.

References:
[GitHub Python workflows](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) ·
[Workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
