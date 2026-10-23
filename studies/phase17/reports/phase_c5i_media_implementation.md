# Phase C5-I — Media calibration interface: implementation and qualification

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_c5i_media_predeclaration.json`, committed in `76f4714` before
  any C5-I code or number.
- **Basis:** `studies/phase17/reports/phase_c5_media_materiality.md`.
- **Code:**
  - `src/lunar_od/media.py` (new);
  - `src/lunar_od/relativistic_light_time.py`: media hooks in `solve_two_way`, `counted_doppler`,
    `delta_dor_station_time`, the pass metadata key `media`, and new `TwoWaySolution` fields appended with
    zero defaults.
- **Unchanged:** `estimators.py`, `measurement_ingestion.py`, `two_way_range.py`, `radiometrics.py`,
  `delta_dor.py`, `spice_loader.py`.
- **Evidence:**
  - `phase_c5i_qualification.json`, `phase_c5i_estimator_gates.json`.
  - The Orekit reference `phase_c5i_reference.npz` (sha256 `44c017ea…9a26`).

## 0. Final status

```
GATES     G1 PASS (bit identity 0; full suite 1,895 passed, 29 skipped, 0 failed)   G2 PASS (1.8e-15)   G3 PASS (1.1e-15)
          G4 PASS (range 3.0e-8 m, Doppler 1.1e-13 m/s, delta-DOR 8.5e-19 s)   G6 PASS (1.5e-7)
          G5 FAIL as computed: BLS-LM 1.004 mm vs 1 mm (both DampingLimit)
OUTCOME   C5I_NOT_QUALIFIED as computed; G5 adjudication ACCEPTED by the owner 2026-10-23 -> C5I_QUALIFIED
IMPACT    default model (Niell, IERS ZHD, ZWD 0.15 m, climatological TEC): range up to 12.9 m, OD 70 sigma
```

## 1. What was built

- **`media.py`:**
  - IERS 2010 eq. 9.11 zenith hydrostatic delay at the standard-atmosphere pressure.
  - The Niell (1996) mapping functions. The coefficients and algorithm are those in Orekit 13.1.9, read from
    the jar bytecode: the Niell Table 3 values, latitude interpolation, a DOY offset of 28/211 days and the
    height correction.
  - The Black–Eisner mapping, kept to reproduce the C5 survey.
  - The single-layer ionosphere and the first-order X-band delay.
- **`MediaModel`:** zenith hydrostatic/wet delays and VTEC as inputs (constants, or the default climatology),
  with `leg_delays` giving group and phase delays per leg.
- **Light time:** media delays are added per leg at the ground event.
  - Two-way range uses the group delay; counted Doppler uses the phase delay (ionosphere with the opposite
    sign).
  - Delta-DOR adds the difference of the two downlinks' group delays.
  - Absent the `media` metadata key, behaviour is bit-identical.
- **Robustness fix found by the unit tests:** the elevation `asin` argument is now clamped to [−1, 1]. A line of
  sight exactly along the zenith exceeded 1 by rounding. No KP2 row is near the zenith, and the qualification
  was re-run with the fix.

## 2. Gates

| gate | result | value | limit |
|---|---|---|---|
| G1 bit identity, 10 cells | **PASS** | 0 / 0 / 0 | 0 |
| G1 full suite | **PASS** | 1,895 passed, 29 skipped, 0 failed | no new failure |
| G2 Niell vs Orekit, 8,400 points (el × lat × h × 12 dates) | **PASS** | 1.8e-15 | 1e-9 |
| G3 single layer vs Orekit, 350/450 km | **PASS** | 1.1e-15 relative | 1e-12 |
| G4 survey parity: range / Doppler / delta-DOR | **PASS** | 3.0e-8 m / 1.1e-13 m/s / 8.5e-19 s | 1e-6 m / 3e-9 m/s / 1e-15 s |
| G5 BLS-LM from the truth, media on vs off | **FAIL as computed** | DampingLimit both; terminal points 7.6 mm and 8.6 mm from truth; difference **1.004 mm** | 1 mm, same status |
| G5 SRIF (reported, not gated) | — | MaxIter both; 0.10145 m vs 0.10603 m (the known pair) | — |
| G6 neglected media partial | **PASS** | 1.5e-7 | 1e-6 |

**G7 (informational):** the default model moves two-way range by up to 12.9 m and counted Doppler by 3.7 mm/s.
Its OD impact (max |δθ/σ|):

| design | DSN_REALISTIC | PROJECT |
|---|---|---|
| RANGE_ONLY | 66.2 | 13.3 |
| RANGE+DOPPLER | 70.2 | 22.7 |

## 3. Interpretation

The media physics and its integration are exact to machine precision (G2–G4, G6). G5 misses by 0.4 % of its
threshold:
- BLS-LM stops at its damping limit in both runs, at 7.6 mm and 8.6 mm from the truth.
- With the media delay of up to 12.9 m in the data, the two non-converged stopping points differ by 1.004 mm.

## 4. Proposed adjudication (owner decision)

**G5 → QUALIFIED.**
- Both runs end at the same status, DampingLimit.
- The 1.004 mm difference sits at the estimator's integrator-limited floor, the same mechanism as C2-I G6 and
  C3-I G5.
- The model is exact (G2–G4) and its neglected partial is 1.5e-7 (G6).

**Recommendation:** every C-phase closure gate has now run into the same floor. BLS-LM stops at DampingLimit,
7–12 mm from the truth, and SRIF at MaxIter, about 0.1 m from it. Phase D (integrator tolerance/events and
estimator convergence) should come before further C-phase implementation work, so that closure gates can be
clean.

## 5. Open items

- Reading DSN media calibration files: Phase F.
- GMF/VMF mapping, ramp tables and solar plasma.
- Scenario/desktop plumbing.
- Making media a default: an owner decision, since it changes every synthetic result.

## 6. Owner decision (2026-10-23)

The owner accepted the G5 adjudication of s4. **Outcome: C5I_QUALIFIED.** The owner also approved Phase D2
(re-freeze of dynamics.py, srp.py, estimators.py), the DE440 switch after D2, and LRO data download for Phase F.
