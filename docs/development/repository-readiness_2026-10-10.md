# Public repository readiness — 2026-10-10

Starting main commit: 6008ccfc918c9e4ef72367822b78f9a8cbb74651.

## Changes

- Short root README with the current desktop entry point, screenshot, install
  steps, test scope, license and documentation links.
- User/research/verification/history documentation indexed separately.
- Earlier UI briefs, project context and performance plan explicitly labeled
  historical. Existing files and scientific evidence remain in their original paths.
- Portable Windows setup and launch scripts use the extracted checkout root.
  The setup creates a local isolated venv, pins the recorded reference profile,
  checks imports and preserves unusable existing environments for diagnosis.
- Public GitHub Actions gate: Windows x86_64, Python 3.13.2, read-only token,
  exact official Action commits, JUnit upload on test failure.
- GitHub description and eight relevant topic tags configured.

## Local checks

| Check | Result |
| --- | --- |
| Public test runner in validated venv | 219 passed, zero skips/failures/errors |
| Fresh source folder, new isolated venv | Setup, pip check and desktop imports passed |
| Same source folder public runner | 219 passed, zero skips/failures/errors |
| Source folder containing spaces and an ampersand | Setup/launcher argument handling passed |
| Windows launch_lunar_od.cmd --check | Passed |
| PowerShell parser | Passed |
| GitHub workflow actionlint 1.7.12 | Passed |
| New documentation local links | Valid |
| Default scenario screenshot | Reviewed with Windows fonts |

The fresh source folder has no Git history, kernels, historical campaign data,
or developer venv. Public tests redirect every external-data override into
empty temporary directories. Setup uses 64-bit Python 3.13.2.

No files under lunar_od/, examples/, tests/ or desktop_app/ changed from the
starting commit. No numerical tolerance, baseline or qualified provenance hash
was changed. The earlier full combined research result remains 1,747 passed,
29 skipped, with its exact scope recorded
[here](main_workspace_readiness_2026-10-10.md).

## Hosted checks and source distribution

The live [Public checks workflow](https://github.com/halici21/lunar-orbit-determination/actions/workflows/public-checks.yml)
records GitHub-hosted Windows outcomes. Its result is authoritative for hosted
execution; local checks do not stand in for that run.

Source prereleases are linked from [Releases](https://github.com/halici21/lunar-orbit-determination/releases).
The Windows starter package requires Python and separate SPICE data. It is not
a standalone EXE. Binary QProcess/backend packaging and long real-run desktop
acceptance need their own qualification.
