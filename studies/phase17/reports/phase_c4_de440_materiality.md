# Phase C4 — DE440 vs the production DE421 ephemeris: materiality survey

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_c4_de440_predeclaration.json`, committed in `1af7448` with
  the survey script, before any number.
- **Evidence:** `studies/phase17/results/phase_c4_de440_survey.json`, from
  `studies/phase17/scripts/phase_c4_de440_survey.py`.
- **Production code changed:** none.
- **Baseline:** the C2-I/C3-I measurement stack (`DSN_C2_LIGHT_TIME_MODEL`, exact-epoch stations), KP2 SIMPLIFIED
  dynamics, `de421.bsp`.

## 0. Final status

```
GATES          S1 kernel restore (bit for bit) PASS | S2 sanity PASS (|dE| <= 3.92 m) | S3 interpolation PASS (1.8e-7 m)
               S4 convergence PASS | S5 trajectory reproduced bit for bit PASS
MATERIAL       T1 Earth position (station moves by DE440 - DE421 Moon-centred Earth): OD 4.82 sigma
DOCUMENT_ONLY  T2 SSB light-time ephemeris (<= 1e-4 sigma), T3 SRP Sun direction (0.024 sigma)
DECISION       switching the production ephemeris to DE440 is an OWNER DECISION (fixed in advance): it
               re-baselines every synthetic result and frozen numerical record
```

## 1. Results

Maxima over the 10 KP2 cells (9,718 range rows, 9,320 Doppler rows) and CANON k0 delta-DOR. σ is
DSN_REALISTIC.

| term | two-way range | counted Doppler | delta-DOR | OD |
|---|---|---|---|---|
| T1 Earth position | 400.9 mm (0.40σ) | 18.3 µm/s (0.61σ) | 8.25 nrad (1.65σ) | **4.82** |
| T2 SSB ephemeris | 1.8e-7 m | 4.4e-3 µm/s | 0.1 pnrad | 1e-4 |
| T3 SRP Sun direction (trajectory moves ≤ 2.9 mm) | 2.3 mm | 2.1 µm/s (0.07σ) | 3e-4 nrad | 0.024 |
| TOTAL | 401.2 mm | 18.9 µm/s | 8.25 nrad | 4.82 |

OD impact of T1 by design (max |δθ/σ|):

| design | DSN_REALISTIC | PROJECT |
|---|---|---|
| RANGE_ONLY | 4.82 | 0.97 |
| RANGE+DOPPLER | 4.20 | 1.29 |

## 2. Interpretation

- **The difference that matters** is the Moon-centred position of the Earth, i.e. of every station.
  - DE440 and DE421 put it 3.5–3.9 m apart, mostly transverse to the line of sight, so two-way range moves by
    ≤ 0.40 m.
  - The shift is systematic over the whole arc, which neither the state nor the per-pass Doppler bias
    absorbs: 4.8σ in RANGE_ONLY and 4.2σ with Doppler at DSN_REALISTIC.
  - Every OD solution this project computes against an Earth position from DE421 carries this as a model
    error relative to the current JPL ephemeris, against which real DSN data and NASA's LRO/GRAIL reference
    orbits are produced.
- **The DE440 barycentre redefinition** moves the Moon's SSB position by ~115 km. It does not matter: the
  module's light-time legs are relative vectors, and the shift is nearly a constant translation.
- **The SRP Sun direction** (200 m in 1.5e11 m) is negligible.

## 3. Recommendation (owner decision)

1. **Switch the production planetary ephemeris to DE440** (`spice_loader.REQUIRED_KERNELS`: `de440.bsp` in place
   of `de421.bsp`). It makes the ephemeris consistent with the MOON_PA_DE440 gravity frame and with external
   references.
   - The cost is a re-baselining: every KP2/R1O/GEO synthetic campaign and frozen record (R2/R3 hash gates, KP2
     records, and so on) will change numbers.
   - It should be its own phase (C4-I), with a predeclared list of affected records and a requalification
     plan.
2. Until then, external comparisons (Orekit uses DE440) must state the ephemeris. The Orekit trial already did.
3. Third-body gravity is absent from the frozen KP2 dynamics. Its DE440 sensitivity is a follow-up for the
   high-fidelity configuration.

## 4. Limits

- Linear bias mapping at the truth.
- SIMPLIFIED dynamics (no third-body gravity), so the dynamics part of the ephemeris change is only the SRP Sun
  direction.
- The Earth difference was applied as a station perturbation; Earth orientation and stations otherwise
  unchanged.
