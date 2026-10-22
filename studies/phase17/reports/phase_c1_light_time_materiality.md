# Phase C1 — Barycentric light time and Shapiro delay: materiality survey

- **Date:** 2026-10-22
- **Predeclaration:** `studies/phase17/configs/phase_c1_light_time_predeclaration.json`, committed in `1d6394f` before
  any number.
- **Evidence:**
  - `studies/phase17/results/phase_c1_light_time_survey.json`
  - `studies/phase17/scripts/phase_c1_light_time_survey.py`
- **Production code changed:** none. This phase decides what C1's implementation step must contain.

## 0. Final status

```
C1_OUTCOME (predeclared rule)   = IMPLEMENT: FRAME, SHAPIRO_SUN, SHAPIRO_EARTH, TIME_SCALE
                                  DOCUMENT_ONLY: SHAPIRO_MOON, SHAPIRO_JUPITER
GATES                           = S1 S2 S3 PASS; S4 FAIL as computed (1 of 22,876 solves; DE421 Moon-position
                                  discontinuity, s3), adjudicated, owner acceptance pending
AS-COMPUTED WITH GATE VETO      = two-way range: C1_INCONCLUSIVE (S4). Counted Doppler and delta-DOR are unaffected
                                  (every solve converged); their terms decide the same four IMPLEMENT items
POPULATION                      = 10 KP2 cells (5 orbits x 2 epochs), SIMPLIFIED dynamics: 9,718 two-way range rows,
                                  9,310 counted-Doppler rows, 170 delta-DOR rows (CANON k0, Goldstone-Canberra)
```

## 1. What was compared

An independent fixed-point solver re-solves every production observable in two frames:

- **MOON:** the production model, Newtonian light time in the Moon-centred frame.
- **SSB:** Newtonian light time in the solar-system barycentric frame. Each endpoint is its Moon-centred position plus
  the Moon's SPICE SSB position at that endpoint's own epoch.

On top of the SSB solve the survey adds two terms:

- **Moyer's Shapiro delay** (eq. 8-38, γ = 1), per body: Sun, Earth, Moon, Jupiter barycentre.
- **The station interval conversion** `TIME_SCALE`: the measured interval is kept in TT, and the conversion uses
  d(TDB−TT)/dTDB.

OD impact is the linear bias mapping at the truth:

- Formula: δθ = (HᵀWH + P₀⁻¹)⁻¹HᵀWδy, with σ from a QR square root.
- Designs: KP2 range only, and KP2 range + counted Doppler with a free bias per Doppler pass.
- Each design is evaluated at the project noise level (5 m, 1e-4 m/s) and at a DSN-realistic noise level
  (1 m, 3e-5 m/s).

## 2. Gates

| gate | rule | value | result |
|---|---|---|---|
| S1 range parity (MOON vs production) | ≤ 1e-6 m | 1.2e-7 m | PASS |
| S2 Doppler parity | ≤ 1e-7 m/s | 3.3e-9 m/s | PASS |
| S3 delta-DOR parity | ≤ 1e-12 s | 2.2e-16 s | PASS |
| S4 every solve converged | residual ≤ 1e-12 s | 3.9e-12 s in 1 solve | **FAIL as computed** |
| S5 vs Orekit/GMAT trial (informational) | 1.83–1.94 m; ~7.2 m | FRAME 1.89–2.46 m; Sun Shapiro 7.31–7.81 m | consistent |

**Survey-code defect found by S3 and fixed before the population run.** The CANON k0 smoke run gave a delta-DOR
parity of 1.4e-11 s. The survey formed the differential delay as an epoch difference `t_B − t_obs`, which is
quantised at ulp(1e5 s) ≈ 1.5e-11 s. Production uses local delays. With `τ_B − τ_A` the parity is 2.2e-16 s.
This was the S3 gate doing its job on the survey code. Production was not involved.

## 3. The S4 failure: a DE421 Moon-position discontinuity

**Where it happens:** one SSB uplink solve, CANON k0 range row 821 (t₃ = 97 110 s, Goldstone).

**What happens:** the fixed point alternates between two values of τ 3.9e-12 s apart. A map with slope ~v/c ≈ 1e-4 can
only do that across a discontinuity.

**What the diagnostic found** between the two transmit epochs:
- The Moon's SSB position from `spkpos` jumps 3.46 mm beyond its linear motion.
- The station's Moon-centred position steps only 4.5e-5 m.
- The jump is in the SPICE ephemeris; DE421 Chebyshev record boundaries are not exactly continuous.

**Effect:** at most c × 3.9e-12 s / 2 ≈ 0.6 mm on that one range row. That is 70× below the smallest term this survey
classifies IMPLEMENT for range (Earth Shapiro, 3.7–4.1 cm). No Doppler or delta-DOR solve is affected.

**Adjudication** (owner acceptance pending): S4's purpose, that no reported term is a convergence artefact, is met.
The predeclared veto makes the range outcome `C1_INCONCLUSIVE` as computed. Counted Doppler and delta-DOR, which have
no failed solve, decide the same four IMPLEMENT terms on their own: FRAME, Sun Shapiro, Earth Shapiro (OD) and time
scale (OD).

**For C1's implementation:** a production SSB solver must tolerate ephemeris record boundaries. It can bound the
iteration and accept a two-cycle below 1e-11 s, or use DE440. C4 moves production to DE440.

## 4. Size of each term (population maximum)

| term | two-way range | counted Doppler | delta-DOR (GC, CANON k0) | one-way range |
|---|---|---|---|---|
| FRAME (SSB − MOON) | +1.89 … +2.46 m | −86 … +106 µm/s | 3.19 µs, **96.8 µrad** | −19.0 … +13.8 km |
| SHAPIRO_SUN | 7.31 … 7.81 m | ≤ 39 µm/s | 8.1 nrad | — |
| SHAPIRO_EARTH | 3.7 … 4.1 cm | ≤ 0.48 µm/s | 0.39 nrad | — |
| TIME_SCALE | ≤ 0.234 m | ≤ 1.2 µm/s | — | — |
| SHAPIRO_MOON | ≤ 0.75 mm | ≤ 0.14 µm/s | 0.0005 nrad | — |
| SHAPIRO_JUPITER | ≤ 1.7 mm | ≤ 0.01 µm/s | 0.002 nrad | — |

**Ratio to DSN-realistic noise** (range 1 m, Doppler 3e-5 m/s, delta-DOR 5 nrad); the materiality threshold is 0.1:

| term | range | Doppler | delta-DOR |
|---|---:|---:|---:|
| FRAME | 2.46 | 3.52 | 19 360 |
| SHAPIRO_SUN | 7.81 | 1.30 | 1.61 |
| SHAPIRO_EARTH | 0.041 | 0.016 | 0.077 |
| TIME_SCALE | 0.234 | 0.039 | — |
| SHAPIRO_MOON | 0.0007 | 0.0046 | 0.0001 |
| SHAPIRO_JUPITER | 0.0017 | 0.0003 | 0.0003 |

## 5. OD impact: max |δθ/σ| over the 10 cells, state / K

| term | R+D, DSN | R+D, project | R only, DSN | R only, project |
|---|---|---|---|---|
| FRAME | **7.69 / 1.42** | **1.61 / 0.26** | **28.6 / 4.31** | **5.72 / 0.86** |
| SHAPIRO_SUN | **28.3 / 4.49** | **5.81 / 0.65** | **112 / 20.3** | **22.4 / 4.03** |
| SHAPIRO_EARTH | **0.125** / 0.020 | 0.019 / 0.002 | **0.561** / 0.100 | **0.112** / 0.020 |
| TIME_SCALE | **0.211** / 0.033 | 0.043 / 0.007 | **0.795** / **0.100** | **0.159** / 0.020 |
| SHAPIRO_MOON | 0.034 / 0.003 | 0.010 / 0.001 | 0.008 / 0.002 | 0.002 / 0.000 |
| SHAPIRO_JUPITER | 0.006 / 0.001 | 0.001 / 0.000 | 0.024 / 0.004 | 0.005 / 0.001 |

Bold marks entries ≥ 0.1, the predeclared OD-material threshold.

## 6. Reading

- **The Moon-centred frame is the largest light-time error in the project.**
  - Two-way range and counted Doppler: it is second order in the Moon's ~30 km/s barycentric velocity, giving 2 m in
    range and 0.1 mm/s in Doppler.
  - One-way observables: it is first order, giving 14–19 km in one-way range.
  - Delta-DOR: it gives 97 µrad, about 20,000 times the 5 nrad noise.
  - The production one-way range and delta-DOR *values* therefore do not describe DSN physics in absolute terms.
- **The Sun's Shapiro delay is a near-constant 7.3–7.8 m range bias.** Without a range-bias parameter the estimator
  absorbs it into the state and K (up to 112 σ state, 20 σ K for range only).
- **Earth Shapiro and the time-scale term are small on the observable** (4 cm, 23 cm) but material in OD.
  They are coherent over the arc, so with 1 m range noise they move the state by 0.56 σ and 0.80 σ.
- **Moon and Jupiter Shapiro are immaterial everywhere** (≤ 0.034 σ). They are documented only.

## 7. Effect on earlier results

- **No earlier classification is changed by this survey.**
  - Every Phase 17 study generated its truth and its estimator with the same Moon-centred model.
  - A model error common to both cancels in those self-consistent experiments.
  - Their linear information analyses use Jacobians, not absolute values.
- **Not assessed here:** whether the Moon-centred Jacobians differ materially from barycentric ones.
  - This matters most for delta-DOR. The R1O-D/D2 direction results are built on delta-DOR Jacobians, and that
    observable is first order in the frame.
  - It is the first check C1's implementation must run once an SSB delta-DOR exists.
- **Real tracking data cannot be processed correctly by the current models.** One-way range and delta-DOR are off by
  orders of magnitude; two-way range and Doppler carry 2–10 m and 0.1 mm/s errors. This is the gap Phase F depends on.

## 8. What C1's implementation step must contain

1. **Solar-system barycentric light time** for two-way range, counted Doppler, delta-DOR and one-way range, with event
   epochs in TDB. A production frame switch must keep the Moon-centred model selectable, bitwise unchanged, for
   historical reproduction.
2. **Shapiro delay** for the Sun and Earth (Moyer 8-38, γ = 1). Moon and Jupiter terms are optional.
3. **The TT/TDB interval conversion** of the measured range.
4. **Analytic Jacobians** of all of the above, finite-difference qualified, and an Orekit cross-check: the Phase B
   trial already has the barycentric Orekit measurement harness.
5. **Tolerance to ephemeris record boundaries** in the fixed-point solves (§3).
6. **Before closing:** the delta-DOR Jacobian comparison of §7.

## 9. Not claimed

- Any statement about Phase 17 science beyond §7.
- Media delays, station tides, the ITRF realisation, DE440 (C3–C5).
- Angular observables and stellar aberration.
- High-fidelity dynamics. Simplified dynamics only sets the geometry, and the terms are light-time physics.
