# Phase C3 — Terrestrial reference frame, EOP and station displacements: materiality survey

- **Date:** 2026-10-22
- **Predeclaration:** `studies/phase17/configs/phase_c3_station_frame_predeclaration.json`, committed in
  `944bef0` before any number.
- **Evidence:**
  - Survey: `studies/phase17/results/phase_c3_station_frame_survey.json`, from
    `studies/phase17/scripts/phase_c3_station_frame_survey.py`.
  - Orekit reference: `phase_c3_station_reference.npz`, sha256 `adb268e7…7d684`. Produced in
    `agent-sandbox/orekit-gmat-oracle` by `phase_c3_station_reference_sandbox.py` with Orekit 13.1.9, IERS 2010
    conventions and the `finals2000A` EOP.
  - Post-hoc diagnostic: `phase_c3_eop_divergence.json`.
- **Production code changed:** none.
- **Baseline:** `DSN_C2_LIGHT_TIME_MODEL` with exact-epoch production stations.

## 0. Final status

```
C3_OUTCOME (rule)   IMPLEMENT:            T2A ITRF realization (OD 1.41 sigma), T3 solid Earth tide (OD 1.11 sigma)
                    DOCUMENT_ONLY:        T4 pole tide (1.2 mm)
                    C3_INCONCLUSIVE:      T2B and T2_TOTAL - the S2 screen failed as computed (s2): T2B is 118 m
                    fixed in advance:     T1 station coordinates and T6 plate motion CONVENTION_REAL_DATA;
                                          T5 ocean loading NOT_MEASURED; T7 final EOP NOT_MEASURABLE at 2027
GATES               S1 S4 S5 S6 PASS; S2 FAIL as computed (T2B > 2 m screen); S3 informational
FINDING             the production Earth PCK (earth_2025_250826_2125_predict.bpc) diverges from IERS EOP after its
                    2025-08-26 creation: 6 m by 2025-10, 57 m by 2026-10, 112-116 m (UT1 0.29-0.31 s) at the
                    2027 survey epochs
OWNER DECISION      proposed classification of T2B (s4)
```

## 1. Terms and method

Each term is an Earth-relative station perturbation dr(t) in GCRF/J2000 axes. Every observable is re-solved
exactly with the perturbed station providers: two-way range and counted Doppler on all 10 KP2 cells, and
delta-DOR on CANON k0. OD impact uses the C2 linear bias mapping.

Orekit variants:
- **A:** the production ECEF vector read as ITRF 2020 with IERS EOP.
- **B:** the same vector read as ITRF 93 through Orekit's Helmert chain.
- **C:** A plus Orekit `TidalDisplacement` (IERS 2010, conventional tide-free).
- **POLE:** the IERS 2010 pole tide with the 2018 secular pole.
- **S:** the production SPICE station.

Terms:
- T2A = A − B
- T2B = B − S
- T2_TOTAL = A − S
- T3 = C − A
- T4 = POLE

## 2. Gates

| gate | result | value |
|---|---|---|
| S1 production station = pxform(ITRF93→J2000) r_ecef | **PASS** | 2.3e-5 m (limit 1e-3) |
| S2 sanity screens | **FAIL as computed** | T2B max **118.4 m** (screen 2 m). Tide peaks 0.158–0.231 m (screen [0.05, 0.6]: pass). Pole tide 1.2 mm (screen 0.05 m: pass) |
| S3 Phase B comparison (informational) | — | T2_TOTAL 103–113 m at 2027, against 23–26 cm at the trial's 2025 epoch. The difference is S2's finding |
| S4 spline interpolation vs direct values | **PASS** | 6.7e-6 m (limit 1e-4) |
| S5 reference record | **PASS** | sha256 match |
| S6 convergence | **PASS** | |

**What S2 found** (post-hoc diagnostic `phase_c3_eop_divergence.json`):
- SPICE's ITRF93 orientation from the production predict PCK, compared with Orekit's ITRF 93 chain:
  - 0.16–0.19 m up to 2025-09 (the high-precision PCK gives the same 0.18 m wherever it has coverage);
  - then 6.1 m (2025-10), 17.4 m (2025-12), 42.5 m (2026-06), 56.8 m (2026-10), **111.9 m (2027-03-08)** and
    116.3 m (2027-03-20).
- That is an equivalent UT1 difference of 0.29–0.31 s at the survey epochs.
- The NAIF long-term predict kernel's Earth rotation drifts from the IERS rotation (measured where the IERS
  series is final, and Bulletin A predictions beyond it) right after the kernel's creation date.
- The high-precision PCK has no coverage after 2025-12.

The survey arithmetic is not at fault: S1 and S4 pass, and the drift reproduces with plain `pxform` outside the
survey code.

## 3. Results

Maxima over the population. σ is DSN_REALISTIC: range 1 m, Doppler 3e-5 m/s, delta-DOR 5 nrad.

| term | station \|dr\| | two-way range | counted Doppler | delta-DOR | OD | decision |
|---|---|---|---|---|---|---|
| T2A ITRF 2020 vs ITRF 93 | 0.327 m | 272.6 mm (0.27σ) | 19.6 µm/s (0.65σ) | 23.0 nrad (4.6σ) | **1.41** | **IMPLEMENT** |
| T2B EOP/PN vs SPICE predict PCK | 118.4 m | 112.5 m | 8.47 mm/s | 15,301 nrad | 309 | C3_INCONCLUSIVE (S2) |
| T2_TOTAL | 118 m | 112.4 m | 8.47 mm/s | 15,313 nrad | 309 | C3_INCONCLUSIVE (S2) |
| T3 solid Earth tide | 0.231 m | 227.4 mm (0.23σ) | 19.2 µm/s (0.64σ) | 15.2 nrad (3.0σ) | **1.11** | **IMPLEMENT** |
| T4 pole tide | 1.2 mm | 0.8 mm | 0.06 µm/s | 0.09 nrad | 0.004 | DOCUMENT_ONLY |

OD impact by design (max |δθ/σ|):

| term | RANGE_ONLY PROJECT | RANGE_ONLY DSN | RANGE+DOPPLER PROJECT | RANGE+DOPPLER DSN |
|---|---|---|---|---|
| T2A | 0.28 | 1.41 | 0.09 | 0.38 |
| T3 | 0.22 | 1.11 | 0.09 | 0.35 |

## 4. Interpretation and proposed decision on T2B

- **T2A.** Real DSN coordinates are published in a modern ITRF. Rotating them with SPICE's ITRF93 orientation
  misplaces the station by up to 0.33 m, which is material.
  - C3-I: a Helmert transformation of station coordinates from ITRF 2020 (at an epoch) to ITRF 93 before they
    meet the ITRF93 PCK, qualified against Orekit's chain.
- **T3.** The solid tide is a physical displacement absent from production (0.23 m, 1.1σ OD).
  - C3-I: an independent IERS 2010 step-1/step-2 implementation qualified against Orekit, opt-in in the station
    providers.
- **T2B (owner decision).** At 2027 this term measures how far two *predictions* of Earth rotation are apart.
  It is dominated by the NAIF long-term predict PCK's UT1. Proposal:
  - classify T2B as **CONVENTION_REAL_DATA**: real data requires final EOP / a high-precision Earth PCK covering
    the data epochs. T7 already says this;
  - record the drift as a **production limitation**. Synthetic campaigns at 2027 run on an Earth orientation
    0.29 s of UT1 from IERS. That is harmless for self-consistent synthetic data, but any external comparison at
    those epochs (Orekit, GMAT) carries a ~100 m station offset unless both sides use the same EOP;
  - C3-I adds a fail-loud check that reports when an epoch lies beyond the loaded Earth PCK's measured
    (non-predicted) span.
- **T4.** Immaterial (≤ 0.004σ).

## 5. Limits

- Linear bias mapping at the truth; SIMPLIFIED dynamics.
- The Orekit station variants use the production ECEF vectors; real DSS coordinates are outside the scope.
- T2B cannot be split into its EOP and precession-nutation parts at 2027. Up to 2025-09 the total B − S is
  0.16–0.19 m (diagnostic).
