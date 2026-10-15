# Root layout cleanup — 2026-10-15

The repository root now has **16 tracked files instead of 22**. Optional
dependency profiles are grouped under `requirements/`; the earlier Tkinter and
HTTP entry points and web resources are grouped under `legacy/`.

## Current entry points

- `python -m pip install -r requirements.txt`: unchanged core dependency source.
- `python -m pip install -e .`: unchanged editable installation.
- `setup_windows.cmd` and `launch_lunar_od.cmd`: unchanged Windows commands.
- `python desktop_app/main.py`: unchanged maintained desktop entry point.
- The [dependency profile guide](../../requirements/README.md) and
  [legacy command guide](../../legacy/README.md) list relocated manual paths.

## What moved

| Former location | Current location |
|---|---|
| `requirements-accelerated.txt` | `requirements/accelerated.txt` |
| `requirements-dev.txt` | `requirements/dev.txt` |
| `requirements-docs.txt` | `requirements/docs.txt` |
| `requirements-validated-windows.txt` | `requirements/windows.txt` |
| `lunar_od_app.py` | `legacy/lunar_od_app.py` |
| `ui_server.py` | `legacy/ui_server.py` |
| `ui/` | `legacy/ui/` |

There are no root wrappers for the earlier tools: manual legacy script commands
use their new locations. Their HTTP routes are retained. Their result directory
and test/experiment child processes resolve to this checkout, avoiding the old
hard-coded sibling `python_port` paths. The optional external map location is
unchanged.

## Verification scope

- All dependency specifiers and exact Windows pins match the prior profiles.
  Only relative include paths and the profile command comment changed.
- Pip dry runs resolve all four profiles from outside the repository.
- Actual `setup_windows.cmd --non-interactive` and launcher import checks passed.
- Focused legacy/preflight regressions: 23 passed, including actual HTTP asset
  responses, dry-run/mock-spawn paths, CLI help and the relocated reference profile.
- Public gate: 290 passed, no failures/errors/skips; the new tests join the
  existing bounded public regression set.
- Strict documentation builds and current path/link/byte checks passed.
- Full research regression is also required before merge; its result is recorded
  in the pull request and task verification report.

The [migration manifest](path-migration.json) adds eight aliases with the source
commit `627730af6c3af188995951e04e93abaa2fe1c913`. Existing scientific source,
historical record identities, fixtures, gallery data and numerical tolerances
are unchanged. The `0.1.0-alpha.2` tag and its previously published assets remain
immutable and continue to use their own original paths.
