# Dependency profiles

The standard [requirements.txt](../requirements.txt) at the repository root
remains the core library dependency list and the packaging metadata's source.
Existing `pip install -r requirements.txt` and `pip install -e .` commands work.

| Profile | Purpose | Install from the repository root |
|---|---|---|
| [Core](../requirements.txt) | Scientific library | `python -m pip install -r requirements.txt` |
| [Accelerated](accelerated.txt) | Core plus optional Numba | `python -m pip install -r requirements/accelerated.txt` |
| [Development](dev.txt) | Core, pytest and optional Qt test helpers | `python -m pip install -r requirements/dev.txt` |
| [Documentation](docs.txt) | MkDocs only; no scientific runtime | `python -m pip install -r requirements/docs.txt` |
| [Windows reference](windows.txt) | Exact validated scientific/desktop versions | `python -m pip install -r requirements/windows.txt` |
| [Desktop additions](../desktop_app/requirements_desktop.txt) | Application dependencies above core | `python -m pip install -r desktop_app/requirements_desktop.txt` |

After dependency installation, install the library with `python -m pip install -e .`.
`setup_windows.cmd` uses the Windows reference profile automatically. Included
`-r` paths are relative to their containing file; absolute profile paths also
work when pip is launched from another directory.

The profile move changes no package specifier or pinned version. The former root
files `requirements-accelerated.txt`, `requirements-dev.txt`,
`requirements-docs.txt` and `requirements-validated-windows.txt` now correspond to
the four profiles above. Use the new paths for manual commands. Older release
tags and immutable verification records retain their original locations.
