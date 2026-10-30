# Phase D4 / D4-R — iterated EKF + RTS smoother

2026-10-30. Branch `feature/ekf-rts-smoother`.

Predeclarations, each written before its run:
- D4: `configs/phase_d4_iterated_smoother_predeclaration.json` (`7fd58cb`);
- D4-R: `configs/phase_d4r_certified_stop_predeclaration.json`.

**Results**
- **D4: `D4_NOT_QUALIFIED` as computed.** G1, G3 and G4 pass; G2 fails on 3 of 300 runs.
- **D4-R: `D4R_QUALIFIED`.** G1–G5 pass.

From any of the three D3 prior sizes (20 m, 2 km, 20 km), the iterated smoother:
- reaches the BLS-LM solution within **0.0005σ**;
- is consistent (mean NEES 5.39–6.91);
- needs at most 4 linearised EKF+RTS passes.

This closes the D3 gap, where the single-pass EKF/RTS had a mean NEES of 10³–10⁹ from a 2 km prior.

## What was built

All in `src/lunar_od/ekf.py`.

- **`run_lunar_ekf(..., linearization=)`** runs a linearised filter about a given trajectory, one state per
  observation:
  - dynamics are expanded as `f(x̄ₖ₋₁) + Φ(x̄ₖ₋₁)(x − x̄ₖ₋₁)`;
  - the measurement as `h(x̄ₖ) + H(x̄ₖ)(x − x̄ₖ)`.

  With `linearization=None` it is the D3 EKF. The only arithmetic change on that path adds `Φ·0` and `H·0`, and G1
  confirms that all 300 D3 runs are reproduced bit for bit.
- **`run_lunar_iterated_smoother`** works in passes:
  - Pass 0 is the D3 EKF+RTS.
  - **Without process noise**, the next linearisation is the trajectory propagated from the smoothed first-epoch
    state. A step is accepted only if it lowers the MAP cost (prior plus whitened measurement residuals); otherwise
    it is halved, up to 10 times. This is damped Gauss–Newton done with sequential passes.
  - **With process noise**, the next linearisation is the smoothed state sequence itself. This is the iterated
    extended Kalman smoother; it has no line search.
- **Stop rule (D4-R).** Converged when either:
  - the step is below 1e-3σ, or
  - the Gauss–Newton decrement `dᵀ P_s0⁻¹ d` is ≤ 0.01 χ². This is the same bound as the F2 TRF certified stop.

## D4 result: G2 fails on 3 of 300 runs

- Range-rate, 20 m prior, seeds 4, 9 and 10 end `NoDescent`. Their states are still within 0.0005σ of BLS-LM.
- **Cause:**
  - the rejected steps are 0.0019–0.0023 smoothed σ, just above the 1e-3σ tolerance;
  - such a step can lower the cost by only about 5e-6 χ²;
  - every halving instead raised the computed MAP cost by 1.2e-5 to 3.1e-5 χ², which is the evaluation's numerical
    floor (integrator at 1e-12).

  The stop rule asked for more precision than the cost has. D4-R replaces it with the certified-stationarity bound.
  D4's record (`results/phase_d4_iterated_smoother.json`) is kept as computed.

## D4-R gates

| gate | result |
|---|---|
| G1 D3 invariance | **PASS**: all 300 D3 runs recomputed on this code equal the committed D3 record (times excluded), in both the D4 and the D4-R run |
| G2 BLS agreement | **PASS**: 300/300 `Converged`; max \|x_IS − x_BLS\| per cell 0.00013–0.0005σ |
| G3 consistency | **PASS**: mean NEES per cell 5.39 / 6.91 / 6.81 (position), 5.43 / 6.45 / 5.62 (range-rate) |
| G4 process-noise path | **PASS**: `tests/test_ekf_smoother.py`, bias fixture with Q, `Converged`, bias within the D3 bounds |
| G5 certified stop | **PASS**: the largest final GN decrement is 0.0072 χ² |

## Results (D4-R, 50 seeds per cell, first epoch)

| cell | position RMS error | mean NEES | passes (mean / max) | halvings |
|---|---|---|---|---|
| position, 20 m | 0.425 m | 5.39 | 1.0 / 1 | 0 |
| position, 2 km | 0.469 m | 6.91 | 2.1 / 3 | 0 |
| position, 20 km | 0.555 m | 6.81 | 3.2 / 4 | 0 |
| range-rate, 20 m | 0.049 m | 5.43 | 1.0 / 1 | 0 |
| range-rate, 2 km | 0.059 m | 6.45 | 2.0 / 2 | 0 |
| range-rate, 20 km | 0.049 m | 5.62 | 2.5 / 3 | 0 |

Compare D3's single-pass EKF+RTS from the 2 km prior: 44 m and NEES 2.9e4 for position, 1.1 m and NEES 922 for
range-rate.

- The full Gauss–Newton step was always accepted: no run needed a halving.
- **Runtime** is 0.5–1.3 s per run against 0.03–0.06 s for BLS-LM.
  - Each pass costs an EKF pass (one STM propagation per epoch plus 2n measurement evaluations for the FD
    Jacobian) plus the cost evaluations.
  - The machine was shared during the runs; D4-R needed fewer passes than D4 and still recorded longer times. Do
    not read the times as more than an order of magnitude.

## Interpretation

- **For a fixed arc without process noise, the iterated smoother is the batch solution computed sequentially.** It
  gives the same answer as BLS-LM and is about 20× slower here. Its value is in what a batch solver cannot do:
  - process noise (the IEKS path; DMC is the next D-phase item);
  - a filter–smoother pair whose covariance matches the batch solution at every epoch, not only at the epoch where
    the batch solution is defined.
- The process-noise path (IEKS) is only unit-tested (G4). It has not been run on a Monte Carlo campaign.

## Not claimed

- Synthetic one-hour arcs only, as in D3.
- No performance work: the FD measurement Jacobian dominates the per-pass cost and could be replaced by analytic
  partials.
- The IEKS path has no line search. On a strongly nonlinear arc with process noise it may need one.
