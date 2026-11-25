# External oracle (Orekit) — generator scripts

These scripts produce the pinned Orekit reference files in
[`tests/fixtures/external_oracle/`](../../tests/fixtures/external_oracle/), which
[`tests/test_external_oracle.py`](../../tests/test_external_oracle.py) compares against
lunar_od **without Orekit**. Background, alignment rules and results:
[external cross-validation](../../docs/validation/external_cross_validation.md).
Predeclaration (cases, alignment, thresholds):
[`wp5_1_orekit_oracle_predeclaration.json`](../../studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json).

Orekit is needed only to **regenerate** the fixtures. Running the tests or the gate script needs
only the repository environment, the SPICE kernels and (for the D2 cases) the GL1800F field.

| file | role | imports Orekit |
|---|---|---|
| `scenario.py` | shared scenario, grids, paths, lossless CSV I/O | no |
| `lunar_od_side.py` | lunar_od force model integrated with TT or TDB as the independent variable (A_TIME), production-propagator link, lunar_od light-time solvers on the pinned inputs | no |
| `compare.py` | gate evaluation, shared by the test and the gate script | no |
| `orekit_session.py` | JVM, data, Moon-centred ICRF frame, GM override, MOON_PA_DE440 frame, GL1800F provider | yes |
| `generate.py` | the two generator steps | `orekit` step only |
| `jhelpers/oracle/*.java`, `jhelpers/build.py` | three Java helpers (from the 2026-10-21 trial, unchanged) and their build | — |
| `requirements-oracle.txt` | pinned oracle environment | — |

## Regenerating the fixtures

A fixture change is a new reference. Do it only deliberately, record the reason in the
report, and never to make a failing comparison pass.

1. Repository environment (no Orekit):

   ```powershell
   python tools/oracle/generate.py lunar-od
   ```

   This writes `scenario.json`, runs the lunar_od self-convergence ladder (DOP853 rtol 1e-13,
   atol 1e-9; then step caps 60 → 1 s; a rung is accepted when the next one changes the position by
   ≤ 0.1 mm) into `lunar_od_settings.json`, and writes the shared spacecraft trajectory
   `shared_trajectory_d1a.csv`.

2. Oracle environment: a separate venv with `requirements-oracle.txt` plus numpy, scipy and
   spiceypy, a JDK ≥ 21 in `JAVA_HOME`, an unpacked orekit-data directory, and the JPL binary
   ephemerides `linux_p1550p2650.440` (DE440) and `lnxp1900p2053.421` (DE421) from
   `ssd.jpl.nasa.gov/ftp/eph/planets/Linux/`. lunar_od is imported from this checkout through
   `PYTHONPATH` (kernel loader, GL1800F parser and SPICE preconditions only):

   ```powershell
   $env:JAVA_HOME = "C:\Program Files\Java\jdk-25"
   $env:PYTHONPATH = "$PWD\src"
   <oracle-python> tools/oracle/jhelpers/build.py
   <oracle-python> tools/oracle/generate.py orekit --orekit-data <orekit-data> --jpl-dir <jpl-dir>
   ```

   This runs the preconditions, the Orekit self-convergence ladder (DormandPrince853, 1e-8 m /
   1e-11 m/s / rel 1e-14, maxStep 300 → 1 s), and writes `orekit_dynamics_<case>.csv`,
   `orekit_station_track.csv`, `orekit_measurements.csv` and `provenance.json`. The provenance
   holds the versions (Orekit jars, JDK, Python packages), the SHA-256 of every input
   (orekit-data listing, JPL files, SPICE kernels, GL1800F, Java sources) and of every fixture.
   `--cases` limits the dynamics cases (the provenance keeps the others), and `--no-measurements`
   skips the measurement part.

3. Re-evaluate and record:

   ```powershell
   python studies/phase17/scripts/wp5_1_orekit_oracle_gates.py
   python -m pytest tests/test_external_oracle.py -q
   ```

The first generation (2026-11-25) used the oracle environment of the 2026-10-21 trial: Orekit
13.1.9 (`orekit-jpype` 13.1.9.0), JDK 25, orekit-data and JPL files with the SHA-256 values in
`provenance.json`.

## Why the jar is not committed

`jhelpers/build.py` compiles the three helpers against the Orekit and Hipparchus jars that
orekit-jpype ships. Its output goes to `jhelpers/build/`, which is gitignored. The provenance
records the SHA-256 of the Java sources.

The folder is called `jhelpers` and not `java`: a `java` package on `sys.path` hides JPype's
`java.*` import hook.
