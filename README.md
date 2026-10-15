# Lunar OD

[![Public checks](https://github.com/halici21/lunar-orbit-determination/actions/workflows/public-checks.yml/badge.svg)](https://github.com/halici21/lunar-orbit-determination/actions/workflows/public-checks.yml)
[MIT](LICENSE) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md)

Lunar OD is research software for lunar orbit determination and tracking-design
studies, with a native Windows desktop workspace for configuring scenarios,
running estimators, and inspecting recorded results.

The Python library includes lunar and third-body dynamics, state-transition and
SRP sensitivity propagation, station visibility, range/angles/range-rate and
qualified two-way observables, BLS-LM, SRIF, and SR-UKF. Optional gravity and
measurement extensions have individual qualification limits.

![Desktop mission view](docs/images/desktop-mission.png)

## Install

The reference environment is Windows x86_64 and Python 3.13.2. Core package
metadata permits Python 3.10+, but other versions/platforms are not covered by
the desktop reference qualification.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-validated-windows.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

For library-only use, install with `python -m pip install -e .` in your virtual
environment; add `-r requirements-dev.txt` for development. The import name is
`lunar_od`. No `PYTHONPATH` override is needed.

On Windows, `setup_windows.cmd` installs the pinned environment and editable
package; `launch_lunar_od.cmd` starts the desktop app. See the
[Windows guide (Türkçe)](docs/windows-quickstart.md).

## Minimal example

```python
from lunar_od import geodetic_to_ecef_wgs84

# Equator, Greenwich, zero ellipsoidal height; coordinates in metres.
station_xyz_m = geodetic_to_ecef_wgs84(0.0, 0.0, 0.0)
print(station_xyz_m)
```

```powershell
python examples/scenario_config_cli.py --schema
python examples/run_scenario_config.py config.json --dry-run
python desktop_app/main.py
```

The desktop can open without SPICE data. Scientific scenarios require the
[kernel setup](docs/windows-quickstart.md#spice-kernels); historical campaigns
may additionally require external data described in [data/README.md](data/README.md).

## Repository layout

```text
src/lunar_od/           importable scientific library
tests/                  unit, integration, regression and science checks
examples/               runnable examples and retained compatibility drivers
studies/phase17/        scripts, configs, reports, results and provenance
docs/                   theory, validation, research and development guidance
desktop_app/            native application (separate from the library)
tools/                  setup, verification and maintenance utilities
fixtures/               small committed reference inputs
data/                   external-data policy (large data are not bundled)
archive/                immutable historical preservation material
```

See the [architecture and artifact policy](docs/development/repository-layout.md)
and [Phase 17 guide](studies/phase17/README.md) for reproduction and historical
path resolution. Local generated runs go to ignored `results/` or `outputs/`.

The station catalogue adapts its actions and map to narrow windows; see the
[layout guide](docs/development/station-layout_2026-10-15.md).

## Validation

```powershell
python -m pip check
python tools/check_repository.py
python tools/run_public_checks.py --junitxml outputs/public-checks.xml
python -m pytest -p tests.desktop_preferences -ra
```

The public CI gate uses committed fixtures and isolates external campaign data.
The full suite includes scientific qualification checks; unavailable external
inputs produce the suite's existing skips. Long campaigns are manual. See
[verification scope](docs/validation/verification.md),
[test categories](tests/README.md), and the
[migration verification record](docs/development/repository-professionalization.md).

## Research status and limitations

Results are primarily synthetic validation and model self-consistency studies.
Passing tests, UI checks, or process completion do not establish operational
navigation accuracy or estimator convergence. This project has no claim of
flight qualification or real-mission validation. External tool comparisons
qualify only their documented assumptions.

For current scientific conclusions, superseded results and known defects, start
with the [authoritative Phase 17 status](studies/phase17/reports/phase17_authoritative_status.md)
and [defect ledger](studies/phase17/reports/phase17_harden_defect_ledger.md).
The [baseline research guide](docs/research/research-guide.md) explains the
models and earlier experiment families.

## Citation and license

Use [CITATION.cff](CITATION.cff) and record the exact Git commit, study
configuration, inputs and seeds used. No DOI is currently declared.
Copyright and reuse terms are in the [MIT license](LICENSE). Third-party data
and dependencies retain their own terms.
