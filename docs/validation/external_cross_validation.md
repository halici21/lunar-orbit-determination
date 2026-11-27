# External cross-validation (Orekit oracle)

This page describes the external oracle that F5 WP5.1 of the [V&V programme](../assurance/vv-program.md)
made permanent. It is the evidence for acceptance criterion AC-3 in
[intended use](../assurance/intended-use.md): against an independent reference, with constants, frames and
time aligned, dynamics agree to < 1 mm/day, two-way range to < 1 mm and counted Doppler to < 1 µm/s.

- **Oracle:** Orekit 13.1.9 (`orekit-jpype` 13.1.9.0, JDK 25).
- **Pinned outputs:** `tests/fixtures/external_oracle/`, with
  [`provenance.json`](../../tests/fixtures/external_oracle/provenance.json) (versions; SHA-256 of every input and output).
- **Test (no Orekit needed):** [`tests/test_external_oracle.py`](../../tests/test_external_oracle.py).
- **Generator scripts (Orekit optional):** [`tools/oracle/`](../../tools/oracle/README.md).
- **Predeclaration:** [`wp5_1_orekit_oracle_predeclaration.json`](../../studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json),
  committed before any comparison number.
- **Results as computed:** [WP5.1 report](../../studies/phase17/reports/wp5_1_orekit_oracle.md) and
  [`wp5_1_orekit_oracle.json`](../../studies/phase17/results/wp5_1_orekit_oracle.json).

The oracle was chosen in a predeclared trial of Orekit and NASA GMAT on 2026-10-21, outside the
repository. Orekit agreed on all eight cases, and GMAT on three. GMAT stays a secondary reference for
measurement physics (see [GMAT](#gmat-secondary-reference)).

## Scenario

| item | value |
|---|---|
| epoch | 2025-03-01T00:00:00 TDB (ET 794 059 200 s); arc 1 day, states every 600 s |
| orbit | 100 km circular polar orbit in MOON_PA_DE440, built by `lunar_initial_state_mci`, Moon-centred ICRF/J2000 Cartesian state |
| GM | gm_de431 (`lunar_od.constants` `MU_*`) for Moon, Earth, Sun, in both tools |
| ephemerides | DE440 (DE421 for D1b): lunar_od through CSPICE `.bsp`, Orekit through its own JPL-binary reader |
| station | Canberra DSN (WGS84 −35.40°, 148.98°, 680 m) |

| case | force model |
|---|---|
| D1a / D1b | Moon point mass + Earth and Sun third body; DE440 / DE421 |
| D2-50 / D2-100 | D1a + GL1800F 50×50 / 100×100 (degree ≥ 2, the file's GM and 1738 km radius) in MOON_PA_DE440 |
| D3 | D1a + cannonball SRP (K = 0.02 m²/kg) with conical penumbra lunar shadow |
| M2 / M3 | two-way range / counted Doppler (Tc = 60 s), 121 receive epochs over 2 h, Moon-centred and barycentric (SSB) Newtonian light time |

## Alignment

The tools are compared only after these differences are removed or pinned. The
[cross-validation contract](../../.claude/skills/_shared/cross-validation-contract.md) requires this.

| item | how it is aligned |
|---|---|
| **time argument** | Orekit integrates in TAI/TT seconds, lunar_od in TDB seconds. Over one day this is a 25 mm along-track term, not an error in either tool. The comparison integrates **lunar_od's own force function**, the 6-state right-hand side that the production `propagate_state` integrates, with TT seconds as the independent variable ([`lunar_od_side.py`](../../tools/oracle/lunar_od_side.py)). Orekit's TDB−TT model drifts 126 ns per day against SPICE's (precondition O_P(d) failed its 50 ns bound), so this adapter uses Orekit's own mapping, pinned at 600 s (`orekit_time_map.csv`). A second gate (O_L) shows that the production `propagate_state` equals that force function integrated in TDB |
| frames | Moon-centred, ICRF axes in both tools (Orekit `BodyCenteredIcrfProvider` on GCRF). The lunar body frame is MOON_PA_DE440 in both, never an IAU frame or the generic `MOON_PA` alias. Orekit evaluates it from the binary PCK's Euler records with its own code |
| constants | GM overridden in Orekit (`GmOverrideBody`). SRP constants (P(1 AU) = 1361 W m⁻²/c, AU, R_sun, R_occ = 1737.4 km) are the scenario's |
| harmonics | the GL1800F coefficients are parsed once by lunar_od's loader and handed to Orekit's Holmes–Featherstone model. This is a shared input: Orekit has no PDS SHADR reader |
| integration | each tool's settings are fixed by self-convergence only, before any cross-tool number: the next rung of its ladder must change the position by ≤ 0.1 mm |
| measurements | Newtonian light time, zero transponder delay, no media, no Shapiro, range = c·round trip/2, interval in TDB. **Both** solvers use the same spacecraft trajectory and Orekit's station track (ITRF, IERS 2010, full EOP), pinned every 10 s. Orekit is given the spacecraft state at the bounce epoch, so its internal Keplerian state shift drops out |

The station is shared on purpose. lunar_od's SPICE ITRF93 Earth-orientation chain and Orekit's modern ITRF
put the station 23–26 cm apart. That is a realization difference handled by WP2.4, not a light-time
property, so it is reported as information (`I_STATION`) and not gated here.

## Results

Computed 2026-11-26 from the pinned fixtures
([`wp5_1_orekit_oracle.json`](../../studies/phase17/results/wp5_1_orekit_oracle.json); report
[`wp5_1_orekit_oracle.md`](../../studies/phase17/reports/wp5_1_orekit_oracle.md)). Scenario: 2025-03-01 TDB, one
day of a low lunar orbit, 145 samples every 600 s. D1a/D1b point mass + Earth and Sun (DE440/DE421), D2 adds
GL1800F to degree 50/100, D3 adds SRP with the lunar shadow (50 shadow-boundary events over the day).

**Dynamics** (maximum position difference over the day):

| case | lunar_od max step | Orekit max step | O_D: lunar_od (TT) − Orekit, < 1 mm | O_L: production − adapter (TDB), ≤ 0.1 mm |
|---|---|---|---|---|
| D1a | none | 300 s | 0.0020 mm — pass | 0.00030 mm — pass |
| D1b | none | 300 s | 0.0013 mm — pass | 0.00041 mm — pass |
| D2-50 | 20 s | 300 s | 0.0017 mm — pass | **0.64 mm — fail** |
| D2-100 | 20 s | 300 s | 0.011 mm — pass | **1.71 mm — fail** |
| D3 | 2 s | 10 s | 0.115 mm — pass | 0.0096 mm — pass |

The force model agrees with Orekit to about 10 µm per day with harmonics and 0.12 mm with SRP and shadow. The
O_L failures are not a force-model difference: they come from the production `propagate_state` integration with
no step cap, which is not converged to 0.1 mm with degree-50/100 harmonics (a 20 s cap is needed). This is
an integrator finding for WP3.3, recorded in the report; the tests mark it as strict xfail. The time-argument
term that the alignment removes is 25.6 mm along-track at the end of the day (`I_TIME`). Using SPICE's TDB−TT
instead of Orekit's own changes the trajectory by 0.18 mm (`I_TIMEMAP`).

**Measurements** (120 epochs every 60 s over two hours, one station, shared inputs):

| quantity | max abs difference | rms | gate |
|---|---|---|---|
| two-way range, Moon-centred light time | 5.8 µm | 3.8 µm | O_M2 < 1 mm — pass |
| two-way range, barycentric light time | 0.13 mm | 0.051 mm | O_M2 < 1 mm — pass |
| counted Doppler (Tc 60 s), Moon-centred | 0.040 µm/s | 0.008 µm/s | O_M3 < 1 µm/s — pass |
| counted Doppler (Tc 60 s), barycentric | 3.8 µm/s | 1.3 µm/s | **O_M3 < 1 µm/s — fail** |
| production counted Doppler − range-difference form | 0.00089 µm/s | | L_M3 ≤ 0.01 µm/s — pass |

The barycentric Doppler failure is attributed to the oracle's numerical noise, not to the light-time physics.
The mean difference is −0.0017 µm/s. Orekit's barycentric two-way range carries twice the sample-to-sample noise
of lunar_od's (38 µm against 19 µm about a local polynomial). Doppler differences that noise over 60 s, which
gives 1.3 µm/s against 0.55 µm/s. In the Moon-centred series both tools have the same noise. Positions of
1.5e11 m have a double-precision spacing of about 30 µm. The test marks this case as a strict xfail; the
threshold is unchanged.

The station is shared. lunar_od's own station chain differs from Orekit's by 0.22–0.26 m (`I_STATION`,
information).

## Independence and limits

- **Independent:** planetary positions (reader), time scales, frames, Earth orientation, station geometry,
  third-body, Holmes–Featherstone and SRP/shadow accelerations, numerical integration, and the light-time
  solution.
- **Shared:**
  - the scenario numbers;
  - the GL1800F coefficients (parsed by lunar_od);
  - the MOON_PA_DE440 Euler records (the same PCK file, read by jplephem rather than CSPICE);
  - the spacecraft trajectory and station track of the measurement cases.
- **Not covered:**
  - Shapiro delay, media, transponder delay, TT interval scaling and topocentric station time;
  - the production default integrator profile (VODE Adams 1e-11; WP3.3);
  - the Earth-orientation realization;
  - other epochs and geometries;
  - estimators.

## Running it

```powershell
python -m pytest tests/test_external_oracle.py -q                       # fast tier
$env:LUNAR_OD_RUN_SLOW_TESTS = "1"; python -m pytest tests/test_external_oracle.py -q   # + slow dynamics cases
python studies/phase17/scripts/wp5_1_orekit_oracle_gates.py              # full gate record
```

The tests need the SPICE kernels (and GL1800F for D2). Without them they skip and name the missing
file. To regenerate the fixtures with Orekit, follow [`tools/oracle/README.md`](../../tools/oracle/README.md).
Regenerated fixtures are a new reference, and the reason for regenerating must be recorded.

## GMAT secondary reference

In the 2026-10-21 trial, GMAT R2026a agreed with lunar_od to:

- 1.5–1.9 mm/day for point mass + third body;
- 1.7 mm for SRP with shadow.

It disagreed by 0.86 m/day with lunar harmonics. GMAT's gravity-field path evaluates the field in a frame
rotated 5.13″ from MOON_PA_DE440, although its own `LunaFixed` coordinates match `pxform`. The cause is not
identified.

Its barycentric, relativistic light time independently confirmed the size of:

- the SSB-vs-Moon-centred two-way range term, 1.83–1.94 m;
- the Shapiro delay, ≈ 7.2 m.

A 13.7 µs time-tag term is unexplained. GMAT is therefore not pinned. It remains the planned secondary
reference for DSN measurement physics: barycentric light time with Shapiro, and DSN data types.
