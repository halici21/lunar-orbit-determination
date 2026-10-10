# Contributing

Use main as the current development base. Keep changes focused and describe
their purpose, affected behavior, commands run and required external data.
Use a small pull request for review.

## Set up

Install the validated Windows environment using setup_windows.cmd, or follow
[the quickstart](docs/windows-quickstart.md). The exact dependency profile is
[requirements-validated-windows.txt](requirements-validated-windows.txt).

## Before sharing changes

    python -m pip check
    python tools/run_public_checks.py --junitxml outputs/public-checks.xml
    git diff --check

Public checks deliberately isolate external data. For scientific changes, run
the relevant qualification tests/campaigns as well. Follow the repository's
[project instructions](CLAUDE.md), regression plan and numerical contracts.
Do not loosen an assertion or update a frozen hash simply to make tests pass.

For UI work, keep long calculations off the UI thread, preserve typed settings,
source identity and stored epochs, and exercise cancellation/error paths.
Process completion and operational success must remain distinct.

## Files and evidence

Keep kernels, high-degree gravity data, virtual environments, caches, local runs,
logs and private documents out of Git. Do not delete or relocate committed
scientific evidence or preserved archives as cosmetic cleanup: verify references
and provenance before proposing such a change.

Add a saved config, seed, code identity and input provenance for claimed numerical
results. State missing inputs and qualification limits explicitly.

## Report a problem

Use [GitHub Issues](https://github.com/halici21/lunar-orbit-determination/issues).
Include Python/OS versions, commit or tag, entry point, expected/actual behavior,
the command and relevant error text. Share a minimal config and public/synthetic
fixture where possible; remove credentials and private paths from attachments.

## Releases

Source prereleases require a passing public CI gate and a clean Windows setup
check. They require Python 3.13 and external kernels for SPICE-backed runs.
The source archive does not contain a standalone EXE, local result data or Git
history. A binary desktop distribution needs separate QProcess/backend packaging
qualification before release.
