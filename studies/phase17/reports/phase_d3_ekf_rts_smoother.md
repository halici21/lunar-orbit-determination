# Phase D3 — extended Kalman filter and RTS smoother

2026-10-30. Branch `feature/ekf-rts-smoother`.

The predeclaration (`configs/phase_d3_ekf_rts_predeclaration.json`, commit `2f18d69`) was written before any
comparison number was computed.

**Result: `D3_QUALIFIED`.** G1–G4 pass.
- In the linear regime (20 m prior) the EKF reproduces the SR-UKF, and the RTS smoother reproduces BLS-LM, to a few
  hundredths of σ.
- Beyond the linear regime the single-pass EKF and RTS lose consistency first, as the predeclaration expected.

## What was built

`src/lunar_od/ekf.py`. No existing source file changed.

- **`run_lunar_ekf`** takes the same inputs as `run_lunar_ukf` and uses the SR-UKF's own helpers from `filters.py`:
  - measurement function, noise and residual (angle wrapping);
  - bias-state layout;
  - process-noise coercion and discretisation.

  So the EKF evaluates exactly the physics the UKF evaluates.
- **Time update:**
  - mean and 6×6 STM from `propagate_augmented_state`; bias states are constants;
  - covariance `Φ P Φᵀ + Q`, with Q added after propagation as in the UKF.
- **Measurement update:**
  - the Jacobian is a central difference of the shared measurement function (steps 1 m, 1 mm/s, 1 for biases);
  - Joseph-form covariance update;
  - optional NIS gate.
- **`rts_smooth`** is the fixed-interval Rauch–Tung–Striebel smoother over the stored filtered and predicted
  states, covariances and STMs. Its gain is `C_k = P_k|k Φᵀ (P_k+1|k)⁻¹`, computed with a linear solve.
- **Not supported:**
  - K_SRP solve-for, state freezing, adaptive Q/R, robust updates, iterated EKF;
  - two-way range (the UKF does not support it either);
  - scenario-config and desktop wiring.

## Gates

| gate | result | evidence |
|---|---|---|
| G1 linear oracle | **PASS** | `tests/test_ekf_smoother.py`: RTS equals the stacked batch MAP solution with and without Q (states ≤ 1e-8, covariances rtol 1e-8); the filter's last state equals the MAP last state; Joseph update equals the standard form |
| G2 invariance | **PASS** | No existing source file changed (only new files added). Full suite locally: 1682 passed, 40 skipped, 17 failed, 9 errors and 36 collection errors; the same 17 tests fail on clean `main` `fcaf32f` in the same environment (missing PyQt5/matplotlib, worktree-isolation checks). CI is the reference for the desktop tests |
| G3 linear-regime agreement | **PASS** | S20m, all 50 seeds: max \|EKF − SR-UKF\| = **0.014σ** (position), **0.0026σ** (range-rate); max \|RTS − BLS\| = **0.058σ**, **0.0062σ** (threshold 0.1σ) |
| G4 consistency | **PASS** | S20m mean NEES over 50 seeds, band [4.39, 7.61]: see the table |

## Results

Setup:
- the SR-UKF test fixture extended to one hour: 61 epochs × 3 stations, Moon point mass;
- 50 seeds per cell; the same truth, noise, x_guess and P0 for every estimator;
- EKF and SR-UKF are scored at the final epoch, RTS and BLS-LM at the first epoch;
- source: `results/phase_d3_ekf_rts_comparison.json`, built by `scripts/phase_d3_ekf_rts_comparison.py`.

Each cell shows the position RMS error and, after the slash, the mean NEES (6 is consistent).

| measurement / prior | EKF | SR-UKF | EKF+RTS | BLS-LM |
|---|---|---|---|---|
| position, 20 m | 0.375 m / 5.39 | 0.375 m / 5.39 | 0.425 m / 5.39 | 0.425 m / 5.39 |
| position, 2 km | 31.0 m / 2.9e4 | 0.495 m / 6.92 | 44.1 m / 2.9e4 | 0.469 m / 6.91 |
| position, 20 km | 2150 m / 2.2e9 | 12.7 m / 1.1e4 | 4982 m / 2.2e9 | 0.555 m / 6.81 |
| range-rate, 20 m | 0.055 m / 5.43 | 0.055 m / 5.43 | 0.049 m / 5.43 | 0.049 m / 5.43 |
| range-rate, 2 km | 1.01 m / 921 | 0.063 m / 22.5 | 1.11 m / 922 | 0.059 m / 6.45 |
| range-rate, 20 km | 200 m / 5.5e7 | 0.84 m / 2.5e4 | 237 m / 8.2e7 | 0.049 m / 5.62 |

- No estimator raised an exception or returned a non-finite state in any of the 1 200 runs.
- **Linear regime.**
  - The four estimators are statistically equivalent.
  - The smoother brings the first-epoch estimate to the batch solution: EKF+RTS and BLS-LM have the same error and
    the same NEES at the first epoch.
  - Mean NIS is 3.04 (EKF) and 3.02 (SR-UKF) for the 3-component position measurement, and 4.05 and 4.03 for the
    4-component range-rate measurement.
- **Beyond it.**
  - At 2 km the EKF's first linearisation is already far enough off that its NEES rises to 10³–10⁴. RTS inherits
    this, because it smooths along the same single linearisation.
  - The SR-UKF stays consistent for position at 2 km (6.9) but not for range-rate (22). Both break at 20 km.
  - BLS-LM relinearises every iteration and stays consistent in every cell. Its statuses are mostly `J-Stab`, the
    rest `Converged`.
- **Runtime** (mean per run on this machine, shared CPU): EKF 0.12–0.29 s, SR-UKF 0.76–1.36 s, BLS-LM 0.03–0.06 s.
  - On these short synthetic arcs BLS-LM is the cheapest.
  - The EKF is 3–6× cheaper than the SR-UKF: one STM propagation per step against 13 sigma-point propagations.

## Interpretation

- The EKF and RTS are correct implementations (G1, G3). They give the filter/smoother pair this project lacked, so
  the D-phase items that need a sequential smoother (DMC, consider covariance in a filter) can now build on them.
- A single-pass EKF/RTS is not a substitute for the iterated BLS when the a priori is poor. The natural next step is
  an iterated form:
  - an iterated EKF update;
  - or iterated EKF+RTS passes that relinearise about the smoothed trajectory, the sequential analogue of a
    Gauss–Newton iteration.

  Expectation (not tested here): it closes the 2 km cells while keeping per-step cost.
- Process noise was zero in this comparison, so the RTS–BLS equivalence is the Q = 0 case. With Q > 0 only the
  linear oracle (G1) has been checked.

## Not claimed

- The arcs are synthetic (Moon point mass, identity Earth-fixed frame, Earth at the Moon centre), one hour long.
  Nothing here says how the EKF behaves on the real LRO arcs of Phase F2.
- The 2 km and 20 km cells are reported, not gated.
- The SR-UKF ran with `alpha = 0.35`, the value its own Monte Carlo tests use. No UKF tuning was searched.
