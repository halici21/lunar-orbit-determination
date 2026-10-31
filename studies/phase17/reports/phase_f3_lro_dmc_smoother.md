# Phase F3 — Kalman filter / RTS smoother with DMC on the real LRO arc

- **Date:** 2026-10-31
- **Predeclaration:** `studies/phase17/configs/phase_f3_lro_dmc_smoother_predeclaration.json` (`bdfa54e`), written
  before any F3 code or number.
- **Code:** `studies/phase17/scripts/phase_f3_core.py` (filter and smoother, no LRO code),
  `studies/phase17/scripts/phase_f3_lro_dmc_smoother.py` (driver, `--reference` then `--sweep`),
  `tests/test_phase_f3_core.py` (3 closed-form tests) — `40dde5e`.
- **Result:** `studies/phase17/results/phase_f3_lro_dmc_smoother.json`.
- **Data:** F2's real LRO USN range and counted Doppler, days 2009-260..262 fit, 263 predicted, F2o Doppler edits.

## 0. Final status

```
GATES      G0 reference reproduces F2p  PASS (fit range 1.03034 vs 1.03037 m, 263 RIC equal to 1 mm)
           G1 sigma = 0 control         PASS (epoch deviation <= 0.039 sigma; smoother = batch WLS to 0.026 sigma)
SIGMA*     1e-9 m/s^2 (lowest 263 prediction chi2 against the data, 3 643 vs 69 900 at sigma = 0)
H1         NOT MET: fit-arc along / cross-track vs LOLA 10.65 / 14.08 m (limits 6.94 / 8.58; F2p 9.92 / 12.25)
H2         NOT MET: 263 range RMS 1.04 m (limit 5.21, met) but 263 along-track vs LOLA 7.52 m (limit 5.05)
DECISION   F3_NO_DMC_GAIN (as computed)
FINDING    the result is split. Against the 263 tracking data the sigma* smoother predicts far better than F2p
           (range 5.21 -> 1.04 m, Doppler 5.61 -> 1.72 mm/s, Doppler mean -4.76 -> +0.62 mm/s). Against the
           LOLA orbit it is worse along-track on 263 and on the fit arc, better cross-track on 263. Larger
           sigma make everything worse except the fit-day range residual (over-fitting: range NIS 0.19 at 3e-8).
```

## 1. Setup

- **Reference.** One `residual(p, partials=True)` of F2's `Problem` (N = 300, Doppler, n_emp = 4, days 260–263) at F2p's
  estimate, with F2p's historical numerics (legacy solver, 20 s grid, quantised station epochs, rtol / atol
  1e-12 / 1e-13, empirical FD step 1e-9). It gives every residual and partial and Φ(t, t0), dx/dK and the
  empirical sensitivities on the grid. Wall time 5.1 h (6 workers); cached in `lro_data/f3_cache/reference.npz`
  (workspace data, not in git).
- **Filter.** A linearised Kalman filter and RTS smoother on deviations from the reference, 15 states plus w:
  [dr, dv, (w), dK, 3 range biases, Doppler bias, C_T, S_T, C_N, S_N].
  - Local rows H0_x Φ⁻¹ and H0_c − H_loc S, with Φ and S cubic-spline interpolated.
  - Joseph scalar updates with unit-weight rows (range / 1 m, Doppler / 2 mm/s, F2's weights).
  - Priors are F2p's prior pseudo-observations expressed as deviations from F2p. The epoch state gets 10 km and
    10 m/s; the Doppler bias gets 1 m/s, because F2 has none.
- **DMC.** w is first-order Gauss–Markov in J2000 with τ = 1800 s.
  - Transitions are composed over sub-steps of at most 20 s, with the w coupling by the trapezoid rule.
  - Q is the van Loan kinematic Q of `lunar_od.dmc` per sub-step.
- **Epochs.** There are 22 060 epochs: 13 821 fit-day measurement updates plus 60 s comparison epochs on the fit arc
  and on 263. The 263 data are not processed; their predicted residual is r_ref − H dx at the smoothed state.
- **Closed-form qualification** (`tests/test_phase_f3_core.py`, all on a 3-D harmonic oscillator):
  - σ = 0 smoother = batch WLS (1e-6 σ);
  - the composed transition matches the exact augmented `expm` (row-relative < 1e-3);
  - with simulated DMC truth, the matched-σ smoother has NIS 0.8–1.25 and at most half the no-DMC position error.

## 2. Gates as computed

| gate | criterion | value | result |
|---|---|---|---|
| G0 fit range / Doppler | within 1e-3 relative of 1.0304 m / 2.0354 mm/s | 1.03034 m / 2.03547 mm/s | PASS |
| G0 263 range / Doppler | within 0.01 of 5.2134 m / 5.6184 mm/s | 5.2128 m / 5.6177 mm/s | PASS |
| G0 RIC fit / 263 | within 0.01 m of F2p | max diff 0.0004 m / 0.0005 m | PASS |
| G1 epoch deviation | every component ≤ 0.05 of its formal σ | max 0.039 (K_SRP) | PASS |
| G1 RIC fit / 263 | within 0.05 m of F2p | fit 2.00 / 9.92 / 12.25, 263 1.88 / 5.05 / 2.90 m | PASS |
| (reported) σ = 0 smoother vs batch WLS | — | max 0.026 σ | — |

## 3. Results

RMS against the 263 data (predicted, not fitted) and RIC RMS against the LOLA orbit, in metres.

| σ (m/s²) | fit range (m) | fit Doppler (mm/s) | 263 range (m) | 263 Doppler RMS / mean (mm/s) | 263 χ² | fit RIC vs LOLA | 263 RIC vs LOLA | NIS range / Doppler |
|---|---|---|---|---|---|---|---|---|
| 0 (F2p) | 1.030 | 2.036 | 5.21 | 5.61 / −4.76 | 69 900 | 2.00 / 9.92 / 12.25 | 1.88 / 5.05 / 2.90 | 1.01 / 1.21 |
| **1e-9** | 0.959 | 2.033 | **1.04** | **1.72 / +0.62** | **3 643** | 1.99 / 10.65 / 14.08 | 1.89 / **7.52** / **1.06** | 0.92 / 1.17 |
| 3e-9 | 0.753 | 2.038 | 6.13 | 6.97 / +6.15 | 99 225 | 1.99 / 12.89 / 18.64 | 1.74 / 17.35 / 7.90 | 0.68 / 1.12 |
| 1e-8 | 0.493 | 2.052 | 11.04 | 12.23 / +11.02 | 317 626 | 2.00 / 17.95 / 27.35 | 1.69 / 27.30 / 17.93 | 0.31 / 1.09 |
| 3e-8 | 0.438 | 2.036 | 8.84 | 9.74 / +8.69 | 203 134 | 1.90 / 21.50 / 33.01 | 1.73 / 25.03 / 24.05 | 0.19 / 1.07 |

Smoothed constants (deviation from F2p over the smoothed σ):

| σ (m/s²) | K_SRP | Doppler bias (mm/s) | C_T | S_T | epoch r, v (max \|dev\|/σ) | w RMS R / I / C (m/s²) |
|---|---|---|---|---|---|---|
| 0 | 0.0745 (−0.04) | −7.975 | +0.01 | −0.04 | 0.008, 0.034 | — |
| 1e-9 | 0.0513 (−16.8) | −7.956 | +12.5 | −16.6 | 3.6, 3.5 | 1.2e-9 / 1.7e-9 / 1.0e-10 |
| 3e-9 | 0.0233 (−24.9) | −7.917 | +12.7 | −24.8 | 7.5, 6.8 | 7.1e-9 / 8.3e-9 / 6.0e-10 |
| 1e-8 | 0.0145 (−26.5) | −7.854 | +5.2 | −26.8 | 9.9, 9.0 | 1.8e-8 / 2.1e-8 / 3.0e-9 |
| 3e-8 | 0.0141 (−24.0) | −7.834 | +2.3 | −26.9 | 8.0, 7.5 | 2.7e-8 / 3.1e-8 / 1.6e-8 |

Runtime: 3 s at σ = 0 and 11–12 s per DMC σ, on top of the 5.1 h reference.

## 4. Hypotheses as computed

| hypothesis | rule (predeclared) | computed at σ* = 1e-9 | result |
|---|---|---|---|
| H1 fit arc | along ≤ 6.94 m and cross ≤ 8.58 m vs LOLA | 10.65 m, 14.08 m | NOT MET |
| H2 prediction | 263 along ≤ 5.05 m vs LOLA and 263 range ≤ 5.21 m | 7.52 m; 1.04 m | NOT MET (range part met) |

Decision: **F3_NO_DMC_GAIN.**

## 5. Reading

1. **No σ brings the orbit closer to LOLA on the fit arc.** Along-track and cross-track grow monotonically with σ,
   from 9.92 / 12.25 m at σ = 0 to 21.5 / 33.0 m at 3e-8. Radial stays near 2 m. The range residual shrinks with σ
   to 0.44 m and the range NIS to 0.19: the extra freedom fits range noise and moves the orbit in directions the
   Earth-based data do not see. Larger σ over-fit; this is not a model improvement.
2. **At σ* the data and LOLA disagree.**
   - Against the 263 data, which neither solution fitted, the σ* orbit predicts range to 1.04 m and Doppler to
     1.72 mm/s, against 5.21 m and 5.61 mm/s for F2p. It also removes the −4.8 mm/s Doppler mean that F2q traced to
     F2p's prediction error. The 19× drop in χ² is far beyond what choosing the best of four σ can produce.
   - Against LOLA, the same orbit is 2.5 m worse along-track on 263, though 1.8 m better cross-track.
   - Both cannot be fully right. The predeclared scoring puts LOLA first, so the decision stands as computed. F2r
     already showed that the F2p − LOLA along-track growth (3.97 m) is below LOLA's own accuracy. A 7.5 m 263
     along-track difference does not rule out the σ* orbit, but neither does the data prove it.
3. **DMC re-splits the constants.** At σ* K_SRP falls from 0.0745 to 0.0513 (−16.8 σ), and the along-track
   empirical terms C_T / S_T move +12.5 σ / −16.6 σ. The estimated w is a few 1e-9 m/s², mostly radial and
   along-track. The 263 prediction is carried by those constants: w decays with τ = 30 min after the last fit
   epoch. F3 cannot separate whether the data-side gain comes from w during the fit or from the re-split constants.
4. **The Doppler bias does not move** (≤ 0.14 mm/s at every σ). The ~3.3 mm/s of F2p − LOLA difference that F2q
   found inside the bias is not given back by DMC.
5. **The machinery is qualified.**
   - G0 reproduces F2p to the millimetre.
   - The σ = 0 smoother equals the batch solution to 0.026 σ on 13 821 real updates.
   - At σ = 0 the mean NIS is 1.01 (range) and 1.21 (Doppler), the F2 weights' normalised residuals.

## 6. Next steps

1. **Separate the two sources of the σ* prediction gain** (diagnostic, needs its own predeclaration):
   - re-predict 263 from F2p's epoch state with σ*'s constants;
   - re-predict 263 from σ*'s smoothed end-of-fit state with F2p's constants.

   If the constants carry the gain, a batch fit with a different K_SRP / along-track empirical split is worth
   testing directly.
2. **Score against an independent check of LOLA on 263.** For example, the LOLA crossover-based accuracy for this
   period, if it can be found in the published LOLA/LRO orbit product documentation. Data and LOLA currently
   disagree at the level of LOLA's quoted accuracy.
3. The force-model route F2q pointed to (Phase C: box-wing SRP, albedo / IR) is unchanged by F3. DMC at the
   levels tried does not substitute for it in the LOLA comparison.
