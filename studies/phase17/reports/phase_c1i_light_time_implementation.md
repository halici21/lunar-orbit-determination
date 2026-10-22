# Phase C1-I — Barycentric light time, Shapiro delay and TT interval: implementation (interim)

- **Date:** 2026-10-22
- **Predeclaration:** `studies/phase17/configs/phase_c1i_light_time_implementation_predeclaration.json`.
  - Committed in `0622667`.
  - Amendment A1 (Shapiro feedback in G3): `1615f59`.
  - Amendment A2 (G3 reference = erratum survey; G8 replaced): written before the population run.
- **Code:**
  - `src/lunar_od/relativistic_light_time.py` (new)
  - `tests/test_relativistic_light_time.py` (13 tests)
  - `studies/phase17/scripts/phase_c1i_qualification.py`
- **Evidence:**
  - `results/phase_c1i_qualification.json`: the predeclared run, as computed.
  - `results/phase_c1i_qualification_exact_stations.json` and `..._exact_stations_wide_k.json`: adjudication evidence.
- **Production files changed:** none. `estimators.py` is untouched; the dispatch waits for the owner's re-freeze
  authorisation.

## 0. Status

```
GATES (this interim)   G2 PASS | G8 PASS | G3 FAIL as computed | G5 FAIL as computed
                       G1, G4, G6, G7, G9 not yet run (G1/G6/G7 need the estimator dispatch)
ADJUDICATION           G3 and G5 adjudicated QUALIFIED (s3, s4) - owner accepted 2026-10-22, gates PASS
NEW PRODUCTION DEFECT  station-state providers evaluate Earth rotation at the rounded et0 + t
                       (4.5e-5 m steps; 8e-6 m range / 1.5e-7 m/s Doppler noise) - proposed HARDEN-D31
SURVEY ERRATA          1 (epoch quantisation, TT rate) and 2 (SSB arithmetic floor, station rounding):
                       every C1 decision unchanged
```

## 1. The module

`LightTimeModel(frame, shapiro_bodies, interval_time_scale)`:
- **Default:** the historical Moon-centred model.
- **`DSN_LIGHT_TIME_MODEL`:** SSB frame, Sun and Earth Shapiro, TT interval.

Observables, each with an analytic arc-initial-state Jacobian (plus the K_SRP column when the history has 48 columns):
- receive-tagged two-way range;
- counted Doppler;
- delta-DOR (common transmit event);
- one-way range.

Numerical design, three choices each forced by a finding in this phase:

1. **Exact-epoch SPICE reads.** `et0 + t` is quantised at ulp(et0) ≈ 1.19e-7 s, which is 3.46 mm of lunar barycentric
   motion. The ephemeris is read at the rounded epoch and moved by velocity × the exact rounding error (Fast2Sum).
2. **Relative leg vectors.** Each leg is formed as the Moon-centred end-state difference plus the Moon's SSB
   displacement over τ (trapezoid on its velocity). Absolute SSB vectors (~1.5e11 m) would leave a ~3e-5 m floor.
   The light time τ is the iteration variable, so no epoch difference enters.
3. **`exact_epoch_station_state`.** It removes the production providers' Earth-rotation rounding. The SSB/DSN path
   uses it. The MOON path keeps the raw providers, so it reproduces production.

**Measured noise floor** over 40 ms of receive time (CANON k0, Doppler row 861):

| station states | model | Doppler | range |
|---|---|---:|---:|
| production (raw) | MOON / SSB / DSN | 1.5e-7 m/s | 7–8e-6 m |
| exact-epoch | MOON / SSB / DSN | 1.0e-9 m/s | 0.6–1.3e-7 m |

## 2. Gates as computed (predeclared run, raw production stations)

| gate | rule | value | result |
|---|---|---|---|
| G2 range / Doppler / delta-DOR vs production (MOON) | 1e-6 m / 1e-7 m/s / 1e-12 s | 1.2e-7 m / 3.3e-9 m/s / 2.2e-16 s | **PASS** |
| G3 FRAME + TIME_SCALE range vs survey | 1e-6 m | 1.5e-5 m | FAIL |
| G3 Shapiro range (A1) | ≤ 2e-3 m and feedback within 20 % | 1.4e-4 m; feedback ratio error 8.4 | FAIL |
| G3 Doppler | 1e-8 m/s | 2.2e-6 m/s | FAIL |
| G3 delta-DOR | 1e-12 s | 0 | PASS |
| G5 FD (per-component relative) | 1e-6 range / 1e-5 Doppler | max 3e-3 … 0.26 (medians 1.5e-7 … 1.2e-6) | FAIL |
| G8 row 821 converges, no two-cycle | residual ≤ 1e-12 s | 2.6e-13 s | **PASS** |

## 3. G3 adjudication

The failures have three sources. None of them is in the module's physics.

**(a) Production station rounding.** The station-state providers have 4.5e-5 m steps. Two independent fixed-point
solvers converge on different sides of a step, so they disagree by up to the step size. Evidence:
- the step was located by scanning the provider at 2 ns spacing (4.47e-5 m beyond linear motion);
- it matches v_rot × ulp;
- with exact-epoch stations, FRAME and TIME_SCALE range agree to **2.4e-7 m** (PASS), and the Doppler FRAME term to
  **4.4e-9 m/s** (PASS).

**(b) The survey's arithmetic floor.** This is erratum 2 to the C1 survey.
- **Fix:** the survey now uses positions relative to the Moon at an anchor epoch.
- **Result:** Shapiro range terms differ by ≤ 5.1e-5 m, and that difference is the predicted Shapiro feedback to
  **2.1 %**. A1 PASS.

**(c) Two approximations in the survey's Doppler, not errors in the module.**

| survey approximation | module − survey | after removing it |
|---|---:|---:|
| Shapiro superposed at Newtonian geometry (no feedback) | 3.8e-8 m/s | **3.3e-9 m/s** |
| TT rate evaluated at the count midpoint, not at each endpoint | 2.5e-8 m/s | **1.4e-9 m/s** |

Both residuals are below the 1e-8 m/s threshold.

**Decision (owner accepted 2026-10-22):** G3 QUALIFIED. The module reproduces the independent survey to its stated thresholds once
production's station rounding and the survey's own approximations are removed.

## 4. G5 adjudication

The predeclared rule says "max relative error", and the script normalised each parameter column separately.

- **Why that fails:** columns near zero inflate the ratio. The K partial is ~0 at the start of the arc. Every
  worst case sits in the K column (index 6) or at row 0.
- **The predeclared K steps** (1e-5 … 1e-3 m²/kg) sit in the finite-difference noise: the error falls monotonically
  as h grows.
- **Per-column check** (CANON k0, range, exact stations):

  | column | best-over-steps error, relative to the row maximum |
  |---|---:|
  | position | 2.2e-10 |
  | velocity | 1.2e-7 |
  | K, with h = 3e-3 … 1e-2 | 4.2e-8 |

**All three cells, exact stations, K steps extended to 3e-3 and 1e-2, error relative to each row's largest partial:**

| cell | two-way range | counted Doppler | one-way | delta-DOR |
|---|---:|---:|---:|---:|
| CANON k0 | 4.1e-7 | 4.5e-7 | 4.1e-7 | 5.0e-7 |
| O2 k12 | 1.8e-7 | 5.8e-7 | 1.8e-7 | — |
| O3 k0 | 5.5e-7 | 6.5e-7 | 5.5e-7 | — |

Every value is within its threshold: 1e-6 for range, one-way and delta-DOR; 1e-5 for Doppler.

**Decision (owner accepted 2026-10-22):** G5 QUALIFIED, on row-normalised errors with finite-difference steps outside the noise floor.

## 5. New production defect (proposed HARDEN-D31)

`two_way_range.make_exact_sxform_station_state_provider` and `radiometrics` counted-Doppler providers evaluate
`sxform` at `et0 + t`.

- **Effect:** ~4.5e-5 m steps (0.38 km/s × ulp 1.19e-7 s), giving ~8e-6 m of range noise and ~1.5e-7 m/s of
  counted-Doppler noise. That is ≈ 0.005 σ at DSN-realistic noise, so immaterial for OD.
- **Visible in:** every solver-to-solver and finite-difference comparison below 1e-5 m.
- **Repair options:**
  - wrap the providers as `exact_epoch_station_state` does;
  - pass a split epoch to SPICE.
- **Constraint:** both providers are in R2/R3 byte-gated modules, so a repair needs a re-freeze.

## 6. What remains (C1-I)

1. **Estimator dispatch** (owner authorised the `estimators.py` re-freeze on 2026-10-22): a `two_way_light_time_ssb` profile and a generator in
   the module. Then G1, G6, G7.
2. **G4:** Orekit barycentric two-way range and Doppler (sandbox harness).
3. **G9:** delta-DOR Jacobian comparison against the authoritative f_perp.
