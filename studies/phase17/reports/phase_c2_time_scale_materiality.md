# Phase C2 — Time scales (UTC/TAI/TT/TDB) and station clocks: materiality survey

- **Date:** 2026-10-22
- **Predeclaration:** `studies/phase17/configs/phase_c2_time_scale_predeclaration.json`.
  - Committed in `294580f` before any number.
  - Amendment A1 was committed in `0606a39`, before any number. It covers delta-DOR per-station propagation,
    counted Doppler per endpoint, sign conventions and the S5 probe offset.
- **Evidence:**
  - Survey: `studies/phase17/results/phase_c2_time_scale_survey.json`, from
    `studies/phase17/scripts/phase_c2_time_scale_survey.py`.
  - Independent reference: `studies/phase17/results/phase_c2_erfa_reference.json`.
    - sha256 `f683654ec3a89666c6538f02ce5155d7cfde2a5e70476f28559d65fac7791e10`.
    - Produced by `studies/phase17/scripts/phase_c2_erfa_reference_sandbox.py` with pyerfa 2.0.1.5 (ERFA 2.0.1,
      SOFA 20231011) in `agent-sandbox/c2-erfa-reference`.
    - Input: `phase_c2_survey_epochs.json`, sha256 `0cf3ab67…f1d139`.
  - T5: `studies/phase17/results/phase_c2_t5_orekit_tdb.json`, from the sandbox harness
    `orekit-gmat-oracle/harness/meas_tdb_c2.py` (copy in `studies/phase17/scripts/phase_c2_t5_orekit_tdb_sandbox.py`).
- **Production code changed:** none.

## 0. Final status

```
C2_OUTCOME (predeclared rule)  IMPLEMENT:            T2 (SPICE TDB-TT model -> full series), T3 (topocentric TDB),
                                                     T4 station clock offset (at the 1 us grid level only)
                               CONVENTION_REAL_DATA: T1 (UTC tags)
                               DOCUMENT_ONLY:        T1C (leap-second chain), T2 rate part, T4 offsets <= 0.1 us,
                                                     T4 frequency (1e-13, 1e-12)
                               OUT_OF_SCOPE:         T6 (spacecraft time)
                               T5:                   UNEXPLAINED as computed
GATES                          S1 S2 S3 S4 S6 PASS as computed; S5 (informational) parity 4.6e-4 relative
OWNER DECISIONS PROPOSED       (a) delta-DOR T3 differential term (~100 urad), s5.2
                               (b) T4 implementation level, s5.3
```

One reference independent of SPICE (ERFA/SOFA) is enough to show three things:

- SPICE's one-term TDB−TT model is 12.4–16.5 µs from the full Fairhead–Bretagnon series at the survey epochs.
- In the RANGE+DOPPLER design at DSN_REALISTIC noise, that epoch error moves the OD solution by **1.77σ**.
- The station-dependent (topocentric) TDB term, absent from production, moves the solution by up to
  **0.64σ**. Its rate part alone gives 4.5 cm in two-way range, 0.47σ in RANGE_ONLY.

## 1. What was computed

The baseline is the production C1-I profile `DSN_LIGHT_TIME_MODEL`:
- light time solved in the SSB frame;
- Sun and Earth Shapiro delay;
- measured interval converted to TT with the analytic rate of SPICE's DELTET model;
- exact-epoch station states.

Each term is the change of an observable when the station-side time relation is made more faithful than the
baseline. The rules (predeclaration §propagation and A1):
- **Epoch part:** a tag error `dt` changes the observable by `dt · dh/dt`. `dh/dt` is a central difference
  with h = 1 s.
- **Rate part:** an error in `d(dt)/dt` scales the station-measured interval.
- **Counted Doppler:** propagated per count-interval endpoint.
- **Delta-DOR:** propagated per station (A1), because each reception is tagged by its own clock.

Population and OD mapping are the same as C1:
- 10 KP2 cells, SIMPLIFIED dynamics: 9,718 range rows and 9,320 Doppler rows in total.
- delta-DOR on the CANON k0 Goldstone–Canberra rows (170 rows, b⊥ ≥ 9,856 km).
- OD impact by linear bias mapping with the square-root posterior.

## 2. Gates

| gate | result | value |
|---|---|---|
| S1 leap seconds | **PASS** | TAI−UTC = 37 s at every epoch in both ERFA and naif0012; no arc contains a leap second |
| S2 linearity (dt = 1e-2 s) | **PASS** | range 0.38 mm (limit 1 mm), Doppler 6.7e-8 m/s (limit 1e-6) |
| S3 SPICE chain sanity | **PASS** | max \|ERFA − SPICE\| TDB−TT = 16.5 µs (screen 100 µs) |
| S4 convergence | **PASS** | every solve converged; 0 ephemeris two-cycles |
| S5 production clock parity (informational) | agrees | the production `station_clock_offset_s` path equals `b·dy/dt` to 4.6e-4 relative (median 2.3e-5), 40 rows, b = 1 ms |
| S6 reference record | **PASS** | sha256 matches the sandbox print; 0 epoch misses |

ERFA itself was checked against the SOFA test suite values before use:
- `dtdb(2448939.5, 0.123, 0.76543, 5.0123, 5525.242, 3190.0)` = −1.2803680059370e-3 s; SOFA gives
  −0.1280368005936998991e-2.
- `utctai(2453750.5, 0.892100694)` matches the SOFA value.

## 3. Results

Maxima over the population. σ is DSN_REALISTIC: range 1 m, Doppler 3e-5 m/s, delta-DOR 5 nrad. "OD" is the
largest |δθ/σ| over the 6 state components and K, over designs, noise levels and cells.

| term | two-way range | counted Doppler | delta-DOR | OD | decision |
|---|---|---|---|---|---|
| T1 size: UTC tags read as TDB (69.1855 s) | 137 km | 102 m/s | 3.8 mrad | 9.8e6 | CONVENTION_REAL_DATA (fixed in advance) |
| T1C leap-second + 32.184 s chain, ERFA vs SPICE | 9e-8 m | 6e-10 m/s | — | 1.5e-5 | DOCUMENT_ONLY |
| T2 SPICE DELTET vs ERFA geocentric, epoch part | 32.0 mm | 24.2 µm/s (0.81σ) | 0.68 nrad (0.14σ) | **1.77** | |
| T2 rate part | 2.2 mm | 0.01 µm/s | — | 0.020 | |
| **T2 total** | 33.0 mm | 24.2 µm/s | 0.68 nrad | **1.77** | **IMPLEMENT** |
| T3 topocentric, epoch part | 4.1 mm | 3.0 µm/s (0.10σ) | 0.11 nrad | 0.59 | |
| T3 rate part | 45.3 mm | 3.3 µm/s (0.11σ) | — | 0.47 | |
| T3 delta-DOR differential (A1) | — | — | **99,958 nrad** | — | |
| **T3 total** | 46.0 mm | 6.3 µm/s (0.21σ) | 99,958 nrad | **0.64** | **IMPLEMENT** |
| T4 clock offset 10 ns | 0.02 mm | 0.015 µm/s | (304 nrad) | 0.0014 | DOCUMENT_ONLY |
| T4 clock offset 100 ns | 0.2 mm | 0.15 µm/s | (3,042 nrad) | 0.014 | DOCUMENT_ONLY |
| T4 clock offset 1 µs | 2.0 mm | 1.5 µm/s | (30,416 nrad) | **0.14** | IMPLEMENT (OD only) |
| T4 frequency 1e-13 | 0.06 mm | 0.015 µm/s | (301 nrad) | 0.0006 | DOCUMENT_ONLY |
| T4 frequency 1e-12 | 0.6 mm | 0.15 µm/s | (3,010 nrad) | 0.0056 | DOCUMENT_ONLY |

The T4 delta-DOR column is in parentheses because it is classified CANCELLED_BY_CALIBRATION in advance.

**0.1σ thresholds for T4**, scaled linearly from the largest grid level:

| | range | Doppler | OD |
|---|---|---|---|
| clock offset | 51 µs | 2.0 µs | 0.70 µs |
| fractional frequency | 1.7e-10 | 2.0e-11 | 1.8e-11 |

**Where the OD impact comes from.** T2 reaches 1.77σ only with Doppler at DSN_REALISTIC weights:
- RANGE_ONLY: 0.11σ at DSN_REALISTIC, 0.02σ at PROJECT.
- RANGE_DOPPLER_FREE_BIAS: 0.53σ at PROJECT.

A 12–16 µs epoch error times the ~1.5 m/s² line-of-sight acceleration gives 2.4e-5 m/s. That is 0.8σ per
Doppler point and systematic over every pass, and neither the free per-pass bias nor the state absorbs it.

T3 is material in both designs:
- 0.46σ RANGE_ONLY and 0.64σ RANGE+DOPPLER at DSN_REALISTIC, worst cell O1 k0.
- Its rate part is a diurnal interval-scale error of ~1.2e-10, the same order as the geocentric rate
  (1.45e-10) that C1 found material.

## 4. Interpretation per term

**T1 (UTC tags).**
- Production has no UTC path, so a real tracking file read as-is would be 69.19 s off.
- The ERFA and SPICE leap-second chains agree exactly (S1, T1C), so the conversion itself is not a risk once
  it exists.
- Required for phase F; nothing for synthetic campaigns.

**T2 (TDB−TT model).**
- SPICE's `K sin(M + EB sin M)` keeps the annual term and its second harmonic. It omits the rest of the
  Fairhead–Bretagnon series (planetary and lunar terms). At the survey epochs that is 12.4 µs (k0) and
  16.5 µs (k12).
- The error enters only where station time (UTC/TT) is converted to TDB:
  - the real-data ingestion path;
  - any generator that tags in UTC.
- It does not enter today's synthetic campaigns, whose tags are TDB by construction.
- Decision by the rule: **IMPLEMENT** — the UTC→TDB conversion for station tags must use a full series
  (ERFA-equivalent), not SPICE's DELTET.
- The rate part (2 mm) does not need the series.

**T3 (topocentric TDB).**
- Two parts:
  - the (v_E · r_station)/c² family of terms, ±2 µs diurnal (epoch part);
  - its rate, up to 1.2e-10 (rate part).
- The rate part acts on the measured interval of every two-way observable. It is a model term, not a tagging
  convention, so it is missing in synthetic and real data alike.
- Decision: **IMPLEMENT**. Both parts go in the station-time → TDB relation of C2-I.

**T4 (station clock).** See §5.3.

**T5 (Orekit vs SPICE TDB).** `out/meas/c2_t5_tdb.json`, copied to `results/phase_c2_t5_orekit_tdb.json`.
- Orekit's TDB−TT differs from SPICE's by 1.18 µs over the C1-I G4 window.
- The predeclared regressor ρ̇·ΔTDB gives slope −0.18 and does not reduce the residual (−534 %).
- Verdict: **UNEXPLAINED** as computed.
- Informational: a constant tag offset of −2.17e-7 s = −1.82 ulp(et0) removes 82 % of the RMS (correlation
  with ρ̇ −0.984). The −0.98 correlation is therefore most likely an epoch-representation offset between
  `AbsoluteDate(2025-03-01 TDB) + t` and SPICE's `et0 + t`, not the TDB model. Orekit hands the same station to
  both sides in that harness, so a TDB-model difference has no path into the comparison. It stays at 0.26 mm
  RMS, inside the G4 limit; no production consequence.

**T6.** Out of scope (coherent transponder; fixed in advance).

## 5. Owner decisions proposed

### 5.1 IMPLEMENT scope for C2-I (from the rule, no adjudication needed)

1. A station-time module: UTC → TAI (leap seconds from the loaded LSK; S1 shows it equals ERFA) → TT → TDB,
   with a full TDB−TT series plus the topocentric terms.
2. The topocentric rate in the measured-interval conversion of `relativistic_light_time` (two-way range,
   counted Doppler). This replaces the geocentric DELTET rate C1-I uses.
3. Ingestion of UTC-tagged observations through (1).

The C2-I predeclaration has to pick the series source:
- vendor the ERFA `dtdb` algorithm (BSD-3, ~800 coefficients);
- or depend on pyerfa;
- or use the IERS 2010 / Fairhead–Bretagnon table directly.

The project's no-new-dependency practice suggests a vendored, test-pinned port qualified against pyerfa in the
sandbox.

### 5.2 Delta-DOR differential topocentric term (needs a decision)

The rule classifies T3's delta-DOR column as material by a factor 2e4:
- −(topo_B − topo_A) = −2.78 to −3.30 µs, about 100 µrad at b⊥ ≈ 9,900 km.
- This is the −(V⊕ · b)/c² term of the VLBI consensus delay model (IERS Conventions 2010 ch. 11): the TDB→TT
  transformation of a baseline delay.
- It is the same for every source direction, so in a spacecraft-minus-quasar delta-DOR it cancels except for
  its change over the spacecraft–quasar separation.
  - Over this 27.5 h arc it spans 2.78–3.30 µs (0.52 µs, ~16 µrad).
  - Its largest rate in the reference is 1.07e-10 s/s. Over a 10 min S–Q gap that is 0.064 µs, ~1.9 µrad
    (~390σ), unless the quasar scans bracket the spacecraft scan and remove the linear part.

Whether it enters depends on the delivered observable:
- the raw spacecraft delay needs it in full;
- a quasar-referenced ΔDOR needs it only for the residual change.

**Proposal:**
- Implement it in C2-I as part of the station-time relation, since phase F needs it.
- Leave the R1O-D/R1O-D2 synthetic delta-DOR science as it is. Generation and model are consistent there, and
  the term is a near-constant delay offset that does not enter the delta-DOR Jacobians the f_perp result
  rests on. C1-I G9 showed f_perp moving by only 5.7e-10 when those Jacobians changed frame.

### 5.3 Station clock level (needs a decision)

By the rule, T4 is IMPLEMENT only at the 1 µs grid level (OD 0.14σ; threshold 0.70 µs).

The documented DSN level found for this report is from a JPL TDA Progress Report, not from DSN 810-005, which
was not accessible here (Clements, *Intercontinental Time and Frequency Transfer Using a GPS Timing Receiver*,
TDA PR 42-75, 1983):
- "maintain knowledge of … time offset [between DSN stations] within 10 microseconds";
- an anticipated requirement of under 10 ns for 1987–1990.

**Proposal:**
- Wire the existing `station_clock_offset_s` into the SSB profile as a **consider** parameter with an a priori
  σ taken from the actual 810-005 / station clock-offset product (to be cited in C2-I).
- Do not add a solve-for.
- Frequency is immaterial at any plausible maser level: threshold 1.8e-11, against ~3e-13 knowledge.

## 6. Limits

- Linear bias mapping at the truth; no nonlinear re-solve (as in C1).
- SIMPLIFIED dynamics: the terms are time-scale physics; the trajectory only sets the geometry.
- One delta-DOR geometry (CANON k0).
- dtdb's UT argument is UTC (UT1 is C3).
- Station coordinates are the production ITRF93 positions (C3).
- T5 used the C1-I G4 harness and its 2025-03 window, not the KP2 population.
