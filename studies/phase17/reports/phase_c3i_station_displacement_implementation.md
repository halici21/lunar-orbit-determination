# Phase C3-I — Station coordinate realization and solid Earth tide: implementation and qualification

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_c3i_station_displacement_predeclaration.json`.
  - Committed in `b021789` before any C3-I code or number.
  - Amendment A1 is in `6a1750d`, after the first G2 run and before any G1/G4/G5/G7 number. It fixes the G2
    reference frame.
- **Basis:** `studies/phase17/reports/phase_c3_station_frame_materiality.md`.
- **Code:**
  - `src/lunar_od/station_frames.py` (new);
  - `src/lunar_od/solid_tide.py` (new);
  - `src/lunar_od/relativistic_light_time.py`: adds `displaced_station_state` and the pass-metadata key
    `station_displacement`;
  - `LICENSES/pyTMD.txt`.
- **Unchanged:** `estimators.py`, `measurement_ingestion.py`, `two_way_range.py`, `radiometrics.py`,
  `delta_dor.py`, `spice_loader.py`.
- **Evidence:**
  - `phase_c3i_qualification.json`, `phase_c3i_estimator_gates.json`.
  - The Orekit reference `phase_c3i_reference.npz` (sha256 `63c273d8…823a`); the first-run G2 record is kept.

## 0. Final status

```
GATES        G1 PASS (bit identity 0; full suite 1,889 passed, 29 skipped, 0 failed)
             G2 PASS (7.3e-6 m; run 1 FAIL from a reference-frame error, amended, s2)
             G3 PASS (8.2e-5 m over 53,640 points)
             G4 PASS (range 5.7e-5 m, Doppler 8.9e-9 m/s, delta-DOR 1.3e-13 s)
             G5 BLS-LM PASS (0.2 mm); SRIF FAIL as computed (6.5 mm, both MaxIter)
             G6 PASS
OUTCOME      C3I_NOT_QUALIFIED as computed; G5 SRIF adjudication ACCEPTED by the owner 2026-10-23 -> C3I_QUALIFIED (s6)
IMPACT (G7)  with the EPSG scale: two-way range up to 0.31 m, Doppler 28 um/s, OD 1.89 sigma (RANGE_ONLY, DSN)
```

## 1. What was built

- **`station_frames.itrf2020_to_itrf93`:** the EPSG 14-parameter transformation (PROJ `data/ITRF2020`;
  Altamimi et al. 2023).
  - Includes the scale: 4.47 ppb + 0.12 ppb/yr, worth up to 4.8 cm at DSN sites over 2000–2040.
  - Orekit's transcription omits the scale; `include_scale=False` reproduces it.
- **`is_predicted_earth_orientation`:** flags epochs after the creation date of the loaded NAIF predict PCK.
- **`solid_tide.solid_tide_displacement_ecef` / `solid_tide_at_et`:** IERS 2010 §7.1.1, conventional tide-free.
  - Step 1 is the eq. 7.5/7.6 in-phase terms with latitude-dependent h2/l2, plus the out-of-phase and L1
    corrections.
  - Step 2 is Tables 7.3a/7.3b, as in DEHANTTIDEINEL.
  - The corrections and tables are transcribed from pyTMD 3.0.9 (MIT).
- **`relativistic_light_time.displaced_station_state`:** wraps a station provider with the ECEF offset
  [Helmert(r) − r + tide(r, t)], rotated by `sxform(ITRF93 → J2000)`, together with its rate.
  - The pass metadata key `station_displacement` makes the generator and the estimator path use it. Absent the
    key, behaviour is bit-identical.

## 2. G2: the first run and amendment A1

The first G2 run compared against Orekit's `getITRF(conventions)` frame and found 7.4 mm.
- That frame's realization follows the EOP file's version label: ITRF 2014 before 2026-06-02, ITRF 2020 only
  where the finals data are labelled so. Its route to ITRF 93 is therefore not the ITRF 2020 → ITRF 93
  transformation.
- Orekit's explicitly versioned `ITRF_2020 → ITRF_1993` equals `Predefined.ITRF_2020_TO_ITRF_1993` to 3 µm.
  Against it, the implementation agrees to **7.3e-6 m**.
- The C3 survey ran at 2027 epochs, where the labelled frame is ITRF 2020. The two give identical offsets at
  2027-01-01, so the survey's T2A is the right transformation.
- Run 1 is kept in `phase_c3i_g2_run1_eop_labelled_frame.json` and `phase_c3i_reference_run1.npz`.

## 3. Gates

| gate | result | value | limit |
|---|---|---|---|
| G1 bit identity, 10 cells, range / Doppler / delta-DOR | **PASS** | 0 / 0 / 0 | 0 |
| G1 full suite | **PASS** | 1,889 passed, 29 skipped, 0 failed | no new failure |
| G2 Helmert (no scale) vs Orekit explicit ITRF 2020 → ITRF 93, 10 sites × 41 years | **PASS** | 7.3e-6 m | 1e-4 m |
| G3 solid tide vs Orekit TidalDisplacement, 3 stations × 10 cells × 60 s grid | **PASS** | 8.2e-5 m | 1e-3 m |
| G4 survey parity: range / Doppler / delta-DOR | **PASS** | 5.7e-5 m / 8.9e-9 m/s / 1.3e-13 s | 2e-3 m / 2e-7 m/s / 1e-11 s |
| G5 BLS-LM from the truth start, with vs without | **PASS** | DampingLimit both, 0.2 mm | 1 mm, same status |
| G5 SRIF from the truth start, with vs without | **FAIL as computed** | MaxIter both, 6.5 mm (0.1015 vs 0.1060 m from truth) | 1 mm, same status |
| G6 predicted-EOP flag (2027 survey epochs True, 2025-03-01 False) | **PASS** | start = 2025-08-26 | |

**G7 (informational):** with the scale, the implemented displacement moves two-way range by up to 0.31 m and
counted Doppler by 28 µm/s. Its OD impact (max |δθ/σ|):

| design | DSN_REALISTIC | PROJECT |
|---|---|---|
| RANGE_ONLY | 1.89 | 0.38 |
| RANGE+DOPPLER | 0.55 | 0.15 |

## 5. G5 SRIF: proposed adjudication (owner decision)

SRIF ends at MaxIter (40 iterations) in both runs.

- Its terminal points are 0.10145 m and 0.10603 m from the truth. These are, to all printed digits, the two
  terminal points of the C2-I truth-start diagnostic D2 (`phase_c2i_estimator_diagnostics.json`: 0.10145 m under
  C2, 0.10603 m under C1-I).
- The iterate wanders on the integrator-limited cost (C1-I G6, Phase D) and stops at one of a small set of
  states; which one depends on the noise path, not on the model.
- BLS-LM on the same data agrees to 0.2 mm.
- The 1 mm SRIF criterion therefore measures the estimator's non-convergent path.

I wrote it without that knowledge, although C2-I G6 had shown the same pattern.

**Proposal:** G5 QUALIFIED on BLS-LM's 0.2 mm agreement and on SRIF's terminal points matching the known
model-independent set. The SRIF MaxIter behaviour goes to Phase D (integrator).

## 4. Open items

- **T2B (from C3)** still awaits the owner. The Earth PCK drift is flagged by `is_predicted_earth_orientation`;
  no EOP ingestion was built.
- **Ocean loading** needs BLQ coefficients.
- **Real DSS coordinates and plate motion** are CONVENTION_REAL_DATA.
- **Opt-in wiring:** the displacement is opt-in per pass. Scenario/desktop plumbing and making it a default are
  later decisions.

## 6. Owner decision (2026-10-23)

The owner accepted the G5 SRIF adjudication of s5.
- **G1–G6 PASS**: G5 by owner-accepted adjudication.
- **Outcome: C3I_QUALIFIED.**
- The SRIF MaxIter behaviour is carried into Phase D2. D1 identified its causes: integrator error, shadow events,
  and the estimator's internal propagation mismatch.
