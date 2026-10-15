# Phase 17-R1O-D3 — literature contract (refreshed from D2)

Labels:
- `LITERATURE_RESULT`: what a cited source states.
- `REPOSITORY_HYPOTHESIS`: a modelling choice or claim made here that the source does not make.
- `REPOSITORY_RESULT`: produced by this repository's computation.

D3's formal K result comes only from the repository's computation. The literature supplies units, noise levels,
session abstractions and operational context.

## 1. Sources

| key | document | revision | how verified |
|---|---|---|---|
| **DSN-210F** | *DSN Telecommunications Link Design Handbook 810-005, Module 210: Delta Differential One-way Ranging* | **Rev. F**, issue date **July 02, 2026** (URS CL#26-2234) | full text of `deepspace.jpl.nasa.gov/dsndocs/810-005/210/210F.pdf` (26 pp.), read 2026-09-26 |
| DSN-107F | *810-005, Module 107: X-Band Radio Source Catalog* | Rev. F, issue date September 15, 2025 (JPL D-19379) | full text of `.../810-005/107/107F.pdf` (107 pp.); Table 3 = ICRF3-SX catalog |
| CCSDS-G2 | *CCSDS 500.1-G-2, Delta-DOR — Technical Characteristics and Performance* | Issue 2, November 2019 | retained; the CCSDS publication search found no later issue (`ccsds.org/Pubs/500x1g2.pdf` is still the current listing) |

**D2 cited DSN-210 Rev. E** (change log 01/11/2024). Rev. E is no longer the current DSN document. D3 cites Rev. F.

## 2. What changed from Rev. E to Rev. F

`LITERATURE_RESULT`. The Rev. F document change log (p. 3) lists only two changes:
- **Figure 2:** added S-band capability at DSS-23/25/35/53/55, K-band at DSS-24/26/34/36/53/54/56, and Ka-band at
  DSS-23/33.
- **Table 1:** added the K-band frequency characteristic (25.5–27.0 GHz), and removed a footnote about an X-band
  bandwidth restriction at DSS-35.

The change log names no other affected sections. Every passage D3 relies on was re-read in Rev. F:
- §2.1.1–2.1.4;
- §2.4–2.5;
- Tables 7–10.

The quoted statements and numbers are **identical to the D2 (Rev. E) contract**. **No D3-relevant guidance changed.**
The band additions concern equipment only; D3 models X-band delay units and never uses band capability.

## 3. Geometry statements (verified in Rev. F)

`LITERATURE_RESULT` (DSN-210F §2.1.4, p. 13; §2, p. 6; §2.1.1, p. 7):
- **Baselines.** "ΔDOR measurements are conducted using either the Goldstone-Madrid baseline or the
  Goldstone-Canberra baseline."
- **Two baselines needed.** "Two baselines with orthogonal components are needed to measure both the right ascension
  and declination coordinates of angular position."
- **Goldstone–Madrid.** "The Goldstone-Madrid baseline is oriented east-west and is most sensitive to right ascension
  for spacecraft near the ecliptic plane."
- **Goldstone–Canberra.** "The Goldstone-Canberra baseline is canted and has most sensitivity in the direction that
  splits the axes of right ascension and declination. Since declination is usually the most difficult component of
  spacecraft state to extract from line-of-sight measurements, the Goldstone-Canberra baseline may provide
  information that cannot otherwise be obtained."
- **GM for force modelling.** "The Goldstone-Madrid baseline is still useful to evaluate small dynamic force modeling
  and improve overall trajectory accuracy."
- **One measurement, one plane.** "A single measurement only provides information about the location of the target
  spacecraft in the plane defined by the interferometer baseline and the target spacecraft."
- **Stations.** ΔDOR "requires two Deep Space Stations located at different complexes for a single measurement."
- **Good geometry (§2.5, p. 19).** "DSN observing geometry is generally good for spacecraft near the ecliptic plane, if
  the Sun-Earth-Spacecraft angle is greater than 10 deg, and either (i) observed from Goldstone-Canberra baseline; or
  (ii) geocentric angular declination above -15 deg and observed from Goldstone-Madrid baseline."

`REPOSITORY_HYPOTHESIS`. The line "still useful to evaluate small dynamic force modeling" motivates D3's
joint-baseline question (H2). It does not establish that GM adds K_SRP-specific direction in this campaign. D3 tests
that.

## 4. Session abstraction (verified in Rev. F)

`LITERATURE_RESULT` (DSN-210F §2.1, p. 7; §2.1.4, p. 13):
- ΔDOR measurements "are of relatively short duration … typically one hour or less".
- "In a typical ΔDOR tracking pass of approximately one hour, the spacecraft and reference are alternately observed
  and a total of 5 to 10 scans, each lasting 5 to 10 minutes, are typically recorded."
- "Only a slight improvement in the overall measurement accuracy can be obtained by increasing the measurement time
  beyond 1 hour."

`REPOSITORY_HYPOTHESIS` (the D3 session abstraction). It continues the D2 operational convention:
- **Session length.** A session is one pass of at most 60 min, placed inside one contiguous dual-visibility window of
  its baseline.
- **Normal points.** It carries three ΔDOR normal points, at 1/6, 1/2 and 5/6 of the session. These are the D2/CB13
  "3 QSQ normal points per session", which represent the spacecraft/quasar interleaving.
- **What is not modelled.** Scan-level interleaving, slewing and data-return latency.
- **No scheduler.** This is a schedule abstraction, not a DSN scheduler.

## 5. Error model and weighting (verified in Rev. F)

`LITERATURE_RESULT` (DSN-210F §2.5, pp. 19–22):
- **Table 9, random only:**
  - quasar thermal 0.023 ns;
  - spacecraft thermal 0.012 ns;
  - clock 0.006 ns;
  - dispersive phase 0.029 ns;
  - fluctuating troposphere 0.019 ns;
  - fluctuating ionosphere 0.019 ns;
  - solar plasma 0.006 ns;
  - **RSS random 0.048 ns**. The text says this is "less than 0.06 ns, corresponding to an angular error of 1.9 nrad on
    an 8000 km baseline".
- **Table 8, random + systematic:** adds the systematic rows:
  - station location 0.007 ns;
  - Earth orientation 0.007 ns;
  - zenith troposphere 0.012 ns;
  - ionosphere shell 0.019 ns;
  - quasar coordinate 0.020 ns;
  - **RSS total 0.057 ns**.
- **Separation of error types.** "For navigational analysis it is common to break out errors into random and
  systematic models." And: "If ΔDOR data are used for estimation purposes, and these systematic effects are explicitly
  included in the analysis, then only the random error component should be used for deriving the ΔDOR data weight."
- **Table 10.** Common systematic-model parameters: station location 2 cm; UT1 1-day prediction 2 cm; polar motion
  1 cm; quasar coordinate 0.75 nrad.
- **Performance anchor.** The MSL ΔDOR residual RMS was 0.038 ns.

**Erratum to the D2 contract (`REPOSITORY_RESULT`, arithmetic only).** D2 gave the RSS of the Table 8 systematic rows
as 0.033 ns. The correct value is:

√(0.007² + 0.007² + 0.012² + 0.019² + 0.020²) = √0.001003 = **0.0317 ns**

This is consistent with √(0.057² − 0.048²) = 0.031 ns. The tables are the same in Rev. E and Rev. F, so this is D2
arithmetic, not a revision change. The difference is 4%, inside D3's ×0.5–×2 prior bracket.
- D3 uses **0.0317 ns** as the nominal per-session bias prior.
- 0.033 ns is used only when reproducing D2 for continuity.

**D3 weighting and prior choices** (`REPOSITORY_HYPOTHESIS`):
- **Data weight.** The primary ΔDOR data weight is the random delay only: σ = **0.048 ns** per normal point.
- **Calibration cases.**
  - C0: no bias nuisance, the ideal case.
  - C1: one delay-bias nuisance per session, with a Gaussian prior of σ_b = k × 0.0317 ns for k ∈ {0.5, 1, 2};
    nominal k = 1.
  - C2: one free bias per session.
- **No double counting.** The systematic terms appear only as the nuisance prior, never inside the data weight.
- **Status of the prior.** Treating each session's systematic sum as one Gaussian bias with this σ is a repository
  modelling choice built from DSN component values. It is **not** a DSN statement that session biases are Gaussian
  with σ = 0.0317 ns.
- **Angular convention.** The D2 angular convention (5 nrad projected by B⊥) is kept only as
  `HISTORICAL_CONTINUITY_DIAGNOSTIC`. A fixed angular σ implies a baseline-dependent delay precision, which is not
  physically comparable across baselines or epochs.

## 6. Reference sources

`LITERATURE_RESULT`:
- **DSN-210F §2.4.** Media and baseline errors "scale with the angular separation between spacecraft and quasar". The
  Table 7 nominal is 0.1 rad separation.
- **DSN-107F.** Table 3 is the ICRF3-SX X-band catalog. 4488 source positions were parsed from the Rev. F text. The
  parse was validated: 3C 273 (1226+023) matched to 0.6 mas.

`REPOSITORY_RESULT` (from D2, verified again in D3's tests): the repository's formal spacecraft state/K design rows do
not depend on the quasar, because D_Q is independent of the spacecraft state. Quasar choice enters only the
systematic/operational layer.

D3's real-source overlay (`REPOSITORY_HYPOTHESIS`) is **coordinate-only**. It checks whether at least one DSN-107F
source lies within 0.1 rad of the geocentric spacecraft direction at the session midpoint. Flux, SNR, source
structure and X/Ka interchangeability are not assessed: the catalog's flux columns cannot be parsed reliably from the
PDF text, and no values are invented.

## 7. What the literature does NOT establish

- K_SRP identifiability for this lunar campaign;
- that any schedule produces K-specific direction;
- actual DSN schedule availability;
- mission covariance.

Those are D3 repository questions, and every D3 number is `SYNTHETIC_CAMPAIGN_TRUTH`.
