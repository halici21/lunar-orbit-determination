# Phase 17-R1O-CD1 — Literature Contract: Two-Way Counted Doppler

Every statement below carries one of three labels:

- **LITERATURE_RESULT:** what a cited primary source states.
- **REPOSITORY_HYPOTHESIS:** what CD1 tests.
- **REPOSITORY_RESULT:** what repository code or evidence shows.

No literature statement is used as evidence that counted Doppler identifies K_SRP in this repository. That is CD1's
question.

Sources were retrieved on 2026-09-26. Hashes are of the downloaded PDFs.

## 1. DSN 810-005 Module 202 (Doppler Tracking): current revision

- **Official index:** `https://deepspace.jpl.nasa.gov/dsndocs/810-005/`. It lists the change letter
  **"CL 22-3475 (for 202E)"**.
- **Document:** `https://deepspace.jpl.nasa.gov/dsndocs/810-005/202/202E.pdf`, 39 pages, sha256
  `29a662594bd0297981cf965de070026b078cabe03bdf4708b5c572b866de4708`.
- **Cover:** "DSN No. 810-005, 202, Rev. E / Issue Date: July 19, 2022 / JPL D-19379; CL# 22-3475".
- **No later revision:** requesting `202F.pdf` or `202G.pdf` returns HTTP 301 to a generic NASA page, not a document.
- **Status:** `CURRENT_DSN_DOPPLER_MODULE = 810-005 Module 202 Rev. E (2022-07-19)`, matching the brief's expectation.

## 2. LITERATURE_RESULT: Doppler physics and measurement (202E)

| # | statement | 202E location |
|---|---|---|
| L1 | "This Doppler shift depends on the range rate — the rate-of-change of the distance separating transmitter and receiver … the calculated Doppler shift is related to the range rate." | §2 |
| L2 | "In the DSN a Doppler measurement consists of a set of carrier phase measurements. From these phase data, frequency may be calculated"; "the accumulating downlink carrier phase is measured and recorded." | §2 |
| L3 | Two-way: the DSS uplink is tracked by the transponder, which produces a downlink "coherently related to the received uplink carrier" (the transponding ratio), received at the same DSS. | §2 |
| L4 | The two-/three-way uplink is synthesized "from a highly stable frequency reference … Since this reference is typically more stable than the spacecraft-borne oscillator, a two-way or three-way Doppler measurement is more accurate than a one-way measurement." | §2 |
| L5 | Error model: σ_V² = σ_VN² (white/thermal) + σ_VF² (phase noise of frequency sources) + σ_VS² (solar phase scintillation). σ_V is the error in the rate of change of the **one-way** range; σ_f = 2 f_C σ_V / c for two-way. | §2.2, Eqs. (12), (13) |
| L6 | The thermal term depends on the integration time T and the carrier-loop SNR: σ_VN ∝ c / (2π f_C T) · ρ_L^(−1/2). | §2.2, Eqs. (15), (25), (26) |
| L7 | "Typically, σ_VF² is negligible for two-way and three-way Doppler measurement, owing to the excellent frequency stability of the frequency sources". | §2.2.2.2 |
| L8 | Solar scintillation: σ_VS² = 0.53 C_band c² / (f_C² T^0.35 [sin θ_SEP]^2.45) for 0° < θ_SEP ≤ 90°, and without the sine factor for 90° < θ_SEP ≤ 180°. For **X-up/X-down, C_band = 2.7 × 10⁻⁶**. | §2.2.2.3, Eqs. (30), (31); Table (31) read from the rendered page |

- **L8 reading note:** the text extraction of Table (31) is misaligned (it suggests 1.3 × 10⁻⁵). The value was read from
  the rendered page.
- **Implied magnitudes:** at T = 60 s with f_C = 7.2 GHz × 880/749 = 8.459 GHz, L8 gives σ_VS = **2.07 × 10⁻⁵ m/s** for
  SEP > 90°, 4.8 × 10⁻⁵ m/s at 30°, and 1.77 × 10⁻⁴ m/s at 10°. These are REPOSITORY arithmetic on the LITERATURE
  formula.

## 3. LITERATURE_RESULT: Moyer's DSN observable formulation

Moyer, T. D., *Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation*, JPL
DESCANSO Monograph 2 (2003; Wiley 2005). PDF: `https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_all.pdf`,
sha256 `23b302fd…87ce`.

- **M1 (§10.2.3.1.2):** "Computed values of two-way doppler (F2) and three-way doppler (F3) observables are calculated
  from the precision round-trip light time … at the end of the doppler count interval Tc and the precision round-trip
  light time … at the start of the count interval". The reception times at the end and start of Tc are the
  independent epochs.
- **M2:** "The count interval Tc is an integer (e.g., 60 s or 600 s)."
- **M3 (§2):** the accuracy of DSN two-way Doppler is "about 0.4 × 10⁻⁵ m/s in the one-way range rate under the very
  best of conditions."
- **M4:** Moyer's F2 includes ramped transmitter frequencies (§13.2), media corrections (§10.2) and relativistic light
  time.

## 4. LITERATURE_RESULT: lunar orbit determination with range and Doppler

- **LRO:** E. Mazarico et al., "Orbit determination of the Lunar Reconnaissance Orbiter", *J. Geodesy* 86(3), 193–207
  (2012), doi:10.1007/s00190-011-0509-4.
  - Precision OD from radiometric tracking.
  - The GEODYN force model includes the full lunar gravity field, third bodies and radiation pressure.
  - Altimetric crossovers improved the reconstructions substantially.
- **NRHO / Gateway:** "Investigating Solar Radiation Pressure Modeling …", AAS 22-728 (NASA NTRS 20220010382).
  - SRP is a significant perturbation there.
  - SRP modeling affects prediction and OD performance.
- **What these do not establish:** that range + Doppler, or Doppler alone, **separates a scalar SRP coefficient from
  the initial state** for a 15-orbit low lunar orbit arc. Flight-OD SRP estimates in the literature are typically
  obtained with long arcs, many passes, empirical accelerations and a priori constraints, and are judged by overlap
  consistency. This is context only.

## 5. REPOSITORY_RESULT: the production observable against the literature

| item | repository | literature |
|---|---|---|
| formulation | y = c/(2Tc) · [τ(t3e) − τ(t3s)], two round-trip light-time solves at the count endpoints (`radiometrics.two_way_counted_doppler_observable`) | M1 (same structure) |
| units | `mps_equivalent`: the rate of change of the one-way range, m/s | σ_V convention of L5 |
| frequency | constant X-band uplink 7.2 GHz; turnaround 880/749; no ramps | M4 (ramps omitted: documented simplification) |
| media / relativity | not modeled (docs/theory/two_way_counted_doppler.md) | M4 |
| count interval | configurable; production default 60 s | M2 |
| events | R3 single bounce, exact event-epoch station states; = the R4 four-event model at zero delay (R4-P08, bitwise) | — |
| RF noise | `src/lunar_od/rf/doppler_noise.py` implements 202E Eqs. (12), (25), (26). With no spacecraft RF inputs the thermal term is **PARAMETRIC**. The H-maser floor at 60 s is 4.5e-7 to 2.1e-6 m/s. | L5–L7 |

## 6. REPOSITORY_HYPOTHESIS for CD1

SRP is an acceleration, so its signature appears first in velocity. Counted Doppler, which measures the line-of-sight
rate of change of the round-trip light time over Tc, may therefore see a K-signature direction that the six initial
state components reproduce less completely than they reproduce the range signature.

**This is a hypothesis.** The six-state includes the initial velocity, so Doppler's K signature may be just as
mimicked. CD1 measures the orthogonal fraction f_perp. It does not assume it.

## 7. Consequences for the CD1 noise semantics (REPOSITORY decision, predeclared)

- **No mission-specific Doppler sigma can be justified from public information** for this project. There is no
  spacecraft RF hardware, so 202E's thermal term cannot be evaluated. Therefore:
- **Primary (structural):** σ_D = 1.0 × 10⁻⁴ m/s at Tc = 60 s.
  - This is the repository's existing DSN station `sigma_range_rate_mps` (`src/lunar_od/config.py`).
  - It sits inside the 202E X/X scintillation span at 60 s (2.1e-5 to 1.8e-4 m/s).
  - It is 25× worse than Moyer's "very best" 4 × 10⁻⁶ m/s.
  - Swept over a ×100 bracket.
- **Operationally anchored lower bound:** per row, σ = 202E Eq. (30) X/X scintillation at the row's Sun–Earth–spacecraft
  angle, RSS the repository H-maser 60 s high floor.
  - Thermal noise is excluded because it is PARAMETRIC. So this is an **optimistic lower bound** on the white noise,
    labelled as such.
- All results are **SYNTHETIC_CAMPAIGN_TRUTH**. None is DSN flight performance.
