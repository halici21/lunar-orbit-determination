# Phase C5 — Tropospheric and ionospheric delay, ramped uplink: materiality survey

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_c5_media_predeclaration.json`, committed in `e3988ea` with the
  survey script, before any number.
- **Evidence:** `studies/phase17/results/phase_c5_media_survey.json`, from
  `studies/phase17/scripts/phase_c5_media_survey.py`.
- **Production code changed:** none.

## 0. Final status

```
GATES       S1 elevation PASS (10.0-71.4 deg) | S2 ZHD PASS (2.056-2.129 m) | S3 iono PASS (0.169 m at 30 TECU) | S4 PASS
IMPLEMENT   T1 hydrostatic troposphere (OD 66 sigma), T2 wet troposphere (4.8 sigma), T3 X-band ionosphere (3.5 sigma)
FIXED       T4 ramped uplink CONVENTION_REAL_DATA
```

## 1. Models (survey level, predeclared)

- **Hydrostatic troposphere:** Saastamoinen zenith delay at the standard-atmosphere pressure for the station
  height.
- **Wet troposphere:** a nominal 0.15 m zenith delay.
- **Mapping:** Black–Eisner m(E) = 1.001/√(0.002001 + sin²E).
- **Ionosphere:** thin-shell (350 km) VTEC of 5–30 TECU with a diurnal cosine, at 7.2 GHz up and 8.46 GHz down.
- **Propagation:** each delay is added per leg at the ground event. The ionosphere enters counted Doppler (phase)
  with the opposite sign to range and delta-DOR (group).

## 2. Results

Maxima over the 10 KP2 cells. σ is DSN_REALISTIC.

| term | two-way range | counted Doppler | delta-DOR (CANON k0) | OD max |
|---|---|---|---|---|
| T1 hydrostatic | 2.25–11.87 m | 3.51 mm/s (117σ) | 932 nrad | **66.3** |
| T2 wet (0.15 m zenith) | 0.16–0.84 m | 0.26 mm/s (8.5σ) | 65 nrad | **4.8** |
| T3 ionosphere | 0.04–0.36 m | 42 µm/s (1.4σ) | 15 nrad | **3.5** |

OD impact by design (max |δθ/σ|):

| term | RANGE_ONLY PROJECT | RANGE_ONLY DSN | RANGE+DOPPLER PROJECT | RANGE+DOPPLER DSN |
|---|---|---|---|---|
| T1 | 11.7 | 58.6 | 21.4 | 66.3 |
| T2 | 0.84 | 4.19 | 1.55 | 4.80 |
| T3 | 0.69 | 3.46 | 0.20 | 0.75 |

## 3. Interpretation and C5-I scope

- **Expected, and now measured:** the media dominate every other C-phase term by one to two orders of
  magnitude.
  - For synthetic data the omission is self-consistent.
  - For real data no OD solution is meaningful without a media model or the DSN media calibrations.
- **C5-I** (IMPLEMENT):
  - a media module that applies externally supplied zenith delays (hydrostatic and wet) and TEC through a
    mapping function, which is the interface for the DSN media calibrations of Phase F;
  - a default climatological model for synthetic work;
  - opt-in on the SSB light-time profile, and qualified against Orekit's tropospheric (Saastamoinen,
    Niell/Global mapping) and ionospheric (Klobuchar-type) models.
- **Mapping-function coefficients** must come from a verifiable source: the Orekit jar, as for the Helmert
  parameters, cross-checked against the published tables.

## 4. Limits

- The survey models are simple: nominal wet delay, Black–Eisner mapping, a cosine VTEC.
- The magnitudes are representative, not site-calibrated. The materiality conclusion holds by one to two
  orders of magnitude.
- Not covered:
  - solar plasma near conjunction;
  - ramp tables (T4);
  - quasar referencing of delta-DOR, which removes media only to the extent the spacecraft and quasar
    directions share them.
