# Phase 17-R1O-D2 — ΔDOR Literature Contract

Written **before** the D2 robustness grid was frozen
(`artifacts/phase17_r1od2_predeclared_grid.json`). The literature motivates the grid. It does not
predict the repository answer.

Every statement carries one of three tags:
- `LITERATURE_RESULT`: what a source says, quoted or closely paraphrased with its location.
- `REPOSITORY_HYPOTHESIS`: what D2 expects to test.
- `REPOSITORY_RESULT`: what this repository's code does by construction, verified in source.

## Sources inspected

| key | source | version inspected | how |
|---|---|---|---|
| **DSN-210** | *DSN Telecommunications Link Design Handbook 810-005, Module 210: Delta Differential One-way Ranging* | **Rev. E**, released 12/20/2023, change log entry dated 01/11/2024 | full text of `deepspace.jpl.nasa.gov/dsndocs/810-005/210/210E.pdf` (26 pp.) |
| **CCSDS** | *Delta-DOR — Technical Characteristics and Performance*, CCSDS 500.1-G-2, Green Book | **Issue 2, November 2019** (current issue per its change log; Issue 1, May 2013, superseded) | full text of `ccsds.org/Pubs/500x1g2.pdf` (83 pp.) |
| **CB13** | D. W. Curkendall and J. S. Border, *Delta-DOR: The One-Nanoradian Navigation Measurement System of the Deep Space Network*, IPN Progress Report 42-193, May 15 2013 | as published | full text of `ipnpr.jpl.nasa.gov/progress_report/42-193/193D.pdf` (46 pp.) |

Page numbers below are the documents' own (they coincide with the PDF page index for DSN-210 and CB13; CCSDS uses its section-page form, e.g. 4-13). No claim here depends on a source not listed above.

## 1. Measurement geometry

- `LITERATURE_RESULT` (DSN-210 §2.1, p. 7): ΔDOR "requires two stations for each measurement".
  The antennas are "located in different complexes on separate continents". A single measurement
  "only provides information about the location of the target spacecraft in the plane defined by the
  interferometer baseline and the target spacecraft".
- `LITERATURE_RESULT` (DSN-210 §2.1, p. 9): "a VLBI measurement will always use two stations at
  different complexes".
- `LITERATURE_RESULT` (DSN-210 §2.1.4, p. 13): measurements use **either Goldstone–Madrid or
  Goldstone–Canberra**. Goldstone–Madrid is oriented east–west and is most sensitive to right
  ascension near the ecliptic. Goldstone–Canberra is canted, with most sensitivity in the direction
  that splits RA and Dec. "Two baselines with orthogonal components are needed to measure both" RA
  and Dec.
- `LITERATURE_RESULT` (CCSDS §4.1, eq. 11–12, p. 4-1/4-2): the differential delay between two
  sources is approximately `Δτ ≈ (1/c) B sinθ Δθ_B`, where `Δθ_B` is the component of their
  angular separation along the baseline. "Measurements along two independent baselines are needed
  to determine both components of angular position." The accuracy of `Δθ_B` improves with baseline
  length and spanned bandwidth.
- `LITERATURE_RESULT` (CCSDS p. 4-7/4-8): ΔDOR is "a direct geometric measure of plane-of-sky
  spacecraft position". "Range data combined with ΔDOR from two baselines provides full
  three-dimensional observability."
- `REPOSITORY_RESULT` (`lunar_od/delta_dor.py`, verified in source): the observable is
  `DDOR = D_S − D_Q`, with `D_S = τ_{S,B} − τ_{S,A}` from a common spacecraft transmit event, and
  `D_Q = −(r_B − r_A)·ŝ/c`. **`D_Q` does not depend on the spacecraft state or on K**, so the
  quasar direction does not enter the state/K design rows at all. It enters only the error model.
  This is the key fact for D1.
- `REPOSITORY_RESULT`: Madrid–Canberra is a physically possible pair in the code, but it is **not**
  one of the two DSN ΔDOR baselines named in DSN-210 §2.1.4. D2 reports it but excludes it from the
  primary robustness population (predeclared).

## 2. Quasar calibration and separation

- `LITERATURE_RESULT` (CB13, pp. 15–21): the spacecraft is observed in "an interlocking
  back-and-forth pass of both an angularly nearby quasar and the spacecraft itself". Differencing
  "common mode[s] out a large percentage of the stand-alone errors". A second quasar placed on the
  other side (Q1 S Q2 S Q1 …) makes "a virtual source closer to the spacecraft". "Differencing with
  nearby quasars is demonstrably beneficial and the closer the proximity, the better." Interpolating
  quasar delays to the spacecraft scan time "exactly removes the linear portion of the station-clock
  drift".
- `LITERATURE_RESULT` (DSN-210 §2.1.4, p. 13): a typical ΔDOR pass lasts about one hour, with 5–10
  scans of 5–10 minutes each. The reference source "preferably has a small angular separation".
  CB13 (pp. 23–24) reports MRO cruise sessions of about 1 h, nine dwells, giving **three interpolated QSQ
  normal points per session**.
- `LITERATURE_RESULT` (DSN-210 §2.4, p. 18): "Errors in baseline length and orientation and errors
  in media delays scale with the angular separation between spacecraft and quasar. There is a
  natural trade-off between selecting a stronger source that might be at a larger angular
  separation … or a weaker source that is angularly closer."
- `LITERATURE_RESULT` (CCSDS §4.3, eq. 25–26, p. 4-13/4-14):
  - Station location: `δτ = (1/c) ε_BL Δθ`. The error "is sensitive to the component of source
    angular separation in the direction of the projection of the vector representing the baseline
    error".
  - Earth orientation: `δτ = (1/c) ε_UTPM Δθ`.
  - Quasar coordinate (eq. 32): `δτ = ε_θ B_p / c`.
- `REPOSITORY_HYPOTHESIS`: separation and position angle change the repository's **formal** K
  information not at all, because the rows are quasar-independent. They change K's **systematic
  susceptibility** through the terms above, which depend on the separation vector's orientation
  relative to the baseline and to the baseline-error direction.

## 3. Error budget (current DSN)

- `LITERATURE_RESULT` (DSN-210 Table 7, p. 20–21) gives the nominal X-band assumptions:
  - spacecraft–quasar separation 0.1 rad; component along the baseline projection 0.1 rad;
  - spacecraft elevation 20°, quasar elevation 25°;
  - baseline coordinate uncertainty 0.02 m per component;
  - baseline orientation (1-day prediction) 0.02 m per component;
  - zenith wet / dry troposphere 0.005 / 0.002 m;
  - quasar coordinate uncertainty **0.75 × 10⁻⁹ rad**;
  - baseline projection onto the plane of sky 8 × 10⁶ m.
- `LITERATURE_RESULT` (DSN-210 Table 8, p. 22), 1-σ delay error in ns:

  | term | kind | ns |
  |---|---|---:|
  | quasar thermal | random | 0.023 |
  | spacecraft thermal | random | 0.012 |
  | clock instability | random | 0.006 |
  | dispersive phase | random | 0.029 |
  | station location | systematic | 0.007 |
  | Earth orientation | systematic | 0.007 |
  | zenith troposphere | systematic | 0.012 |
  | fluctuating troposphere | random | 0.019 |
  | ionosphere shell | systematic | 0.019 |
  | fluctuating ionosphere | random | 0.019 |
  | solar plasma | random | 0.006 |
  | quasar coordinate | systematic | 0.020 |
  | **RSS total** | | **0.057** |

- `LITERATURE_RESULT` (DSN-210 Table 9 and p. 19): **RSS random = 0.048 ns**, "less than 0.06 ns,
  corresponding to an angular error of 1.9 nrad on an 8000 km baseline".
- `LITERATURE_RESULT` (DSN-210 p. 20): "If ∆DOR data are used for estimation purposes, and these
  systematic effects are explicitly included in the analysis, then only the random error component
  should be used for deriving the ∆DOR data weight." MSL cruise ΔDOR residual RMS was 0.038 ns.
- `LITERATURE_RESULT` (CB13 pp. 23–24): MRO cruise ΔDOR "system noise measures 1.18 nrad (0.031 ns)
  rms".
- `LITERATURE_RESULT` (DSN-210 Table 10, p. 23): common systematic-model parameters:
  - station location 2 cm, with correlations;
  - UT1 2 cm (1-day prediction), 1 cm after the fact;
  - polar motion 1 cm;
  - zenith troposphere wet 0.5 cm, dry 0.2 cm;
  - ionosphere shell 4 cm by day, 1 cm by night;
  - quasar coordinate 0.75 nrad.
- Derived for D2, `REPOSITORY_HYPOTHESIS` (arithmetic on Table 8): RSS of the systematic rows =
  √(0.007² + 0.007² + 0.012² + 0.019² + 0.020²) = **0.033 ns**. D2 uses this as the nominal
  per-session systematic scale. The historical 1 ns bias remains a labelled **stress** value.

## 4. Noise semantics

- `LITERATURE_RESULT` (DSN-210 §2.4, p. 19): "Units of time delay are employed here to
  characterize ∆DOR measurement error." Angular precision is the delay error divided by the
  projected baseline (0.06 ns ↔ 1.9 nrad at 8000 km).
- `REPOSITORY_RESULT` (`examples/phase17_r1od_campaign.py`, verified): the R1O-D/FXR weight takes
  an **angular** σ_θ and converts it per row as `σ_τ = max(B⊥, 1 m)·σ_θ/c`, with `B⊥` the baseline
  component perpendicular to the station-A→spacecraft line of sight. A constant angle σ is
  therefore a *different* delay σ on every pair and every epoch.
- `REPOSITORY_HYPOTHESIS`: for comparisons *between* station pairs and epochs, a constant **delay**
  σ (the DSN random budget, 0.048 ns) is the physically comparable weight. D2 reports both
  semantics and keeps the historical angular convention for continuity with FXR.

## 5. Operational context D2 does not model

- `LITERATURE_RESULT` (DSN-210 §2.5, p. 19): DSN ΔDOR geometry is "generally good" near the
  ecliptic if the Sun–Earth–spacecraft angle exceeds 10° and the spacecraft is observed from
  Goldstone–Canberra, or from Goldstone–Madrid at declination above −15°. The 3-complex DSN "has a
  geometric weakness when spacecraft are at far southern declinations".
- `REPOSITORY_HYPOTHESIS`: not modelled in D2:
  - quasar flux and SNR;
  - source structure;
  - troposphere, ionosphere and solar-plasma estimation;
  - clock networks;
  - scheduling.

  These belong to an operational simulator, not to this robustness study (brief s27).

## 6. What the literature does NOT say

No source above says anything about **K_SRP observability for a low lunar orbit**. The literature
establishes:
- that ΔDOR is an interferometric plane-of-sky observable;
- that two separated stations are needed;
- that baseline geometry matters;
- that quasar proximity controls calibration error;
- that systematic effects are modelled explicitly.

Whether ΔDOR adds K-specific information to range here is a `REPOSITORY_RESULT` for D2 to measure.

`D2_LITERATURE_CONTRACT_GATE = PASS`.
