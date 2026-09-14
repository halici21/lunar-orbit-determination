# PHASE 17-R1 — K_SRP OBSERVABILITY, PRIOR SENSITIVITY, AND HOLDOUT QUALIFICATION

## Executive summary

R1 used the deterministic Phase 17-R campaign on the clean R0 feature branch. The result is scientifically informative but does not close K_SRP as predictively useful on this selected short, single-station lunar arc.

The data-only augmented information matrix is rank 6/7. Its weakest mode is almost entirely K (`K` component 0.99999999985), and conditional K information after allowing the orbital state to adjust is only 1.5212 in the raw parameterization. The solved-K posterior is therefore weakly constrained and prior-influenced even though the broad and moderate prior runs show substantial data response.

For the predetermined 1.3-orbit estimation / 0.7-orbit holdout split, final holdout position errors were 4.0166 m for correct-fixed K, 4.0216 m for wrong-fixed K (+50%), and 4.1154 m for solved K. Solving for K did not recover the predictive performance of the correct-K reference and was slightly worse than wrong-fixed K. This is a characterization result, not a pass disguised as one.

## 1. Entering state and campaign definition

The starting feature branch was `feature/phase17-r-k-srp-estimation@b8da5e6`, based on canonical `main@fea476f`. Main and origin/main remained unchanged. The shared synthetic truth used `K_truth=0.01 m²/kg`, explicitly labelled `SYNTHETIC_CAMPAIGN_TRUTH`; the wrong-fixed case used `K_wrong=0.015 m²/kg` (+50%), and solve-K started at `K_initial=0.012 m²/kg`.

The campaign used one Canberra DSN station, two-way range at 60 s cadence, a 1.3-orbit estimation arc (0–9180 s), and a predetermined 0.7-orbit holdout (9240–14160 s). The holdout truth was the same propagated trajectory and was not used in estimation. BLS and SRIF shared this campaign; SR-UKF used the existing three-station range-rate qualification geometry because its production measurement path is different. That architecture difference is documented rather than hidden.

## 2. Data-only observability

Purpose: distinguish information supplied by tracking data from information supplied by a prior. Method: form `AᵀR⁻¹A` for `[r,v,K]` without a prior and evaluate a documented physical scaling of `[10⁶ m,10³ m/s,0.01 m²/kg]`.

Results: rank 6/7; scaled singular values are `[8.3104e14, 1.9706e13, 3.1757e11, 1.1512e11, 8.9252e10, 9.8599e4, 1.2236e-4]`; scaled condition number `6.7917e18`. The weakest right singular vector is `[-5.27e-6, 5.89e-6, -1.43e-5, 3.93e-6, 2.71e-6, -3.51e-6, 0.99999999985]`. Raw `I_KK=1.5500e4`; the Schur-complement conditional K information is only `1.5212` after orbital state adjustment.

Interpretation: this arc contains a very weak independent K direction. The near-pure-K weakest mode and large scaled condition number rule out calling K robustly data-observable from this geometry alone.

## 3. RTN correlations and covariance inflation

The solved-K posterior correlations were `rho(K,R)=+0.0113`, `rho(K,T)=-0.00142`, and `rho(K,N)=+0.000972`. Position 1σ values changed from fixed-K `[2.155377, 3.644526, 8.914358] m` to solve-K `[2.155514, 3.644528, 8.914360] m`; inflation ratios were `[1.0000636, 1.0000005, 1.0000002]`.

The correlations are small in this particular arc, so covariance inflation is negligible. That does not make K strongly observable: the information matrix and holdout remain the decisive diagnostics. A wrong fixed K can also report deceptively small formal covariance while carrying dynamical bias.

## 4. Prior sensitivity

The broad prior used mean `0.012`, sigma `1.0`; posterior K was `0.0109854` with sigma `0.00164481`. The moderate prior used the same mean, sigma `0.01`; posterior K was `0.0114615` with sigma `0.00164481`. Both runs were classified `DATA_DOMINATED` by the campaign's combined information/prior-response evidence, but the data-only rank deficiency means this is not equivalent to strong practical observability. No screening envelope was used as a prior.

## 5. Fixed-K and solve-K holdout

| Case | Final holdout position error | Maximum holdout error | Mean holdout error |
|---|---:|---:|---:|
| Correct fixed K | 4.01657 m | 4.01657 m | 2.91518 m |
| Wrong fixed K | 4.02160 m | 4.30150 m | 3.12683 m |
| Solve K | 4.11538 m | 4.21100 m | 3.06543 m |

The solve-K case did not provide meaningful predictive benefit relative to wrong-fixed K. Its final error was higher than both comparison cases. Therefore `HOLDOUT_PREDICTION_GATE=FAIL` for this selected campaign. The result is not evidence that solve-K is universally harmful; it identifies weak observability and geometry/prior limitations that require a different authorized study.

## 6. Three-estimator interpretation

BLS and SRIF agreed on the nonlinear solve (`K=0.01199596` and `0.01199612` respectively) and retained the R0 common Gaussian posterior gate. SR-UKF recovered `K=0.01197760` with `sigma_K=0.00497182` and passed sigma-point/static-K/negative-K controls. All three support the broad conclusion that the implementation works and K is weakly constrained on the available campaign. SR-UKF did not use the identical measurement geometry, so this is consistency of scientific direction, not a claim of identical posterior statistics.

## 7. Regression protection

P21, Model-S, long-arc, event conditioning, force-K derivative, trajectory-K sensitivity, measurement-K sensitivity, counted-Doppler sensitivity, derivative-chain, and all default six-state estimator gates remained PASS. Main and origin/main remained unchanged. The old event-epoch quantization signature remained absent.

The full suite retained only known pre-existing classes: the Phase16 protected dynamics hash assertion, ten R2 closure setup errors, and the FA-06 workspace pytest-path assertion. No R1-introduced or unknown scientific non-pass occurred.

## 8. What R1 established

K_SRP is technically estimable and responds to data, but on this single-station short arc it is weakly observable in the data-only augmented system. The posterior is sensitive to estimator setup, and solving K did not produce a predictive improvement over wrong-fixed K on the predetermined holdout. Covariance inflation is negligible here because K/state correlations are small, not because K is strongly known. BLS, SRIF, and SR-UKF tell a compatible broad story.

## 9. What R1 did not establish

This phase does not close Phase 17. It does not establish useful K observability for longer arcs, multi-pass/multi-station geometry, range+Doppler combinations, or a statistically calibrated population of runs. It does not authorize estimator redesign, parameter transformations, stochastic K, or a new measurement model.

## Final verdict

`K_OBSERVABILITY_CLASS=K_PRIOR_INFLUENCED_BUT_NOT_PREDICTIVELY_USEFUL_ON_SELECTED_ARC`.

`HOLDOUT_PREDICTION_GATE=FAIL`; `ESTIMATOR_SCIENTIFIC_CONSISTENCY_GATE=PASS`; regression protection remains PASS. `PHASE17_R1_GATE=BLOCKED` for full scientific closure because the required predictive benefit was not demonstrated. The next action is a separately authorized geometry/arc study, not automatic Phase 17-R2 closure.
