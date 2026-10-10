# PHASE 17-R1G — GEOMETRY, ARC-LENGTH, AND MEASUREMENT-DIVERSITY STUDY FOR K_SRP OBSERVABILITY

## Executive summary

R1G kept the estimator, K parameterization, SRP force model, shadow model, measurement physics, event solver, and `Q_K=0` semantics frozen. It varied only arc length, station schedule, and measurement-family characterization.

The R1 baseline was reproduced: rank 6/7, scaled condition number about `6.79e18`, weakest mode K component `0.99999999985`, and the original holdout result where solve-K was not better than wrong-fixed-K. Longer single-station range arcs increased conditional K information from `1.52` at 1.3 orbits to `3.52e4` at 5 orbits, but did not produce a practical full-rank K direction. The 3- and 5-orbit BLS solve-K cases became unstable, converging toward K≈0 with very large holdout errors.

The existing counted-Doppler-only and range+counted-Doppler information characterization also remained rank 6/7. The combined case increased conditional K information only from `247.5` to `260.8` in the directional SR-UKF geometry. No tested configuration met all practical observability and holdout criteria. This is a geometry/information limitation, not an estimator implementation failure.

## 1. Frozen architecture and inputs

Starting head: `e9216bf7946fd3e2ea8b4220f8fef8aa05633dc0`; main and origin/main remained `fea476f81dad07b3914e53eab709e10fa6e9d10b`. Truth remained `K_truth=0.01 m²/kg` (`SYNTHETIC_CAMPAIGN_TRUTH`), wrong-fixed K remained `0.015`, and solve-K started at `0.012`. Broad and moderate priors remained sigma 1.0 and 0.01 respectively. No screening-envelope prior was used.

## 2. Baseline reproduction

The original Canberra, range-only, 1.3-orbit estimation / 0.7-orbit holdout campaign reproduced the R1 information structure. The baseline holdout values remain the preserved reference: correct-fixed `4.016574 m`, wrong-fixed `4.021600 m`, solve-K `4.115381 m`. The normalized recovery ratio is ill-conditioned because correct-vs-wrong separation is only about `0.005 m`; R1G therefore reports absolute and RMS values rather than using η alone.

## 3. Arc-length sweep

The following BLS range-only cases used the same truth, cadence, prior definitions, and predetermined 0.7-orbit holdout:

| Case | Arc | Obs. | Rank | Scaled condition | Weakest K component | Conditional K information |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 1.3 orbit | 82 | 6/7 | 6.79e18 | 0.99999999985 | 1.52 |
| G1 | 2.0 orbits | 144 | 6/7 | 1.39e16 | 0.99999999376 | 2.66e3 |
| G2 | 3.0 orbits | 215 | 6/7 | 1.23e16 | 0.99999999451 | 9.36e3 |
| G3 | 5.0 orbits | 307 | 6/7 | 1.03e16 | 0.99999999554 | 3.52e4 |

Conditional information rises with elapsed arc, but the weakest mode remains almost entirely K. This is increased sensitivity, not independent separation from the orbital state.

The 2-orbit solve-K case was the least unstable candidate: K=`0.0117999`, sigma=`0.00164481`, final holdout position error `2.6507 m` versus `2.8175 m` wrong-fixed and `2.5463 m` correct-fixed. It shows a modest absolute benefit but remains rank-deficient and prior-sensitive. At 3 orbits solve-K collapsed to `2.73e-7` and produced `228.3 m` final holdout error; at 5 orbits it collapsed to `1.57e-7` and produced `348.6 m`.

## 4. Station geometry

The 2- and 3-orbit three-station schedules produced the same observation counts and information spectra as the Canberra-only cases because the qualified visibility mask admitted no additional independent station observations on those matched intervals. Therefore no station-geometry benefit can be claimed from this campaign. Adding unqualified station physics was outside scope.

## 5. Counted-Doppler and combined characterization

The existing four-event counted-Doppler SR-UKF synthetic path was used without substituting geometric range-rate. The directional information-only result was rank 6/7, scaled condition `3.82e17`, weakest K component approximately 1.0, and conditional K information `247.5`.

On the same synthetic geometry, adding qualified two-way range rows increased conditional information to `260.8`, but the combined information matrix remained rank 6/7 with scaled condition `3.67e17`. These results are measurement-information characterization, not a new estimator path; direct BLS/SRIF counted-Doppler solve-for was not introduced.

## 6. Holdout and distinguishability interpretation

The 2-orbit case is the strongest tested range candidate, but its solve-K advantage over wrong-fixed-K is only `0.1667 m` at the endpoint and does not remove the rank deficiency. The 3- and 5-orbit cases show that increased information sensitivity alone does not guarantee stable nonlinear recovery. The correct-vs-wrong separation is small on the short baseline and remains an important limitation.

## 7. Physics interpretation

Longer arcs accumulate SRP trajectory sensitivity, which explains the rising conditional information and falling scaled condition number. However, the accumulated signature remains nearly collinear with the six-state correction space in the tested geometry. Counted Doppler adds a different observable sensitivity but not enough independent K information in this configuration. This supports the R1 conclusion that the limitation is information geometry rather than the frozen estimator implementation.

## 8. Regression protection

No production source was changed. R0 estimator gates, P21, Model-S, long-arc, event conditioning, force/trajectory/measurement K sensitivity, derivative-chain, and default six-state parity remain PASS. Existing full-suite nonpasses remain only the known Phase16/R2/FA-06 classes; no R1G-introduced or unknown scientific nonpass was observed.

## 9. Deterministic artifact set

The study package includes machine-readable geometry, observability, holdout, and estimator-comparison CSV/JSON files, plus eight figures: scaled conditioning versus arc, conditional K information versus arc, solve-K sigma versus arc, holdout error versus arc, measurement-family information, K estimate/uncertainty, selected 2-orbit holdout comparison, and weakest-mode K alignment. These are descriptive artifacts generated from the frozen implementation; they do not alter estimator or measurement code. The three-estimator comparison CSV explicitly marks SRIF and SR-UKF as not rerun for the geometry sweep, while retaining their R0 qualification status.

## 10. Prior diagnostic reassessment

The scalar data-only Schur information was compared with prior information rather than inferred from the reported estimator sigma alone. For the baseline, `I(K|state)=1.5212`; broad and moderate prior information are `1` and `10000`, respectively. For the strongest stable range case (G1), `I(K|state)=2658.51`. Thus the moderate prior is not evidence of data domination. The estimator-reported sigmas (`0.00346885` for G0 and `0.00164481` for G1) are not reproduced by the scalar Schur calculation, so the shared covariance/floor or full-model scaling diagnostic remains unresolved. R1G therefore makes no data-dominated claim: the prior classification is `DIAGNOSTIC_UNCERTAIN` for broad G0/G1 and `PRIOR_INFLUENCED` or `PRIOR_DOMINATED` for moderate priors. No covariance infrastructure was modified.

## 11. Required comparison summary

| Case | Arc | Stations | Measurements | Rank | Smallest scaled σ | Weak-mode K | Conditional K info | σ_K | Wrong/correct separation | Solve-K holdout benefit | Classification |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G0 | 1.3 | Canberra | range | 6/7 | 1.22e-4 | 0.99999999985 | 1.52 | 3.47e-3 | 0.097 m RMS | −0.040 m RMS | not distinguishable |
| G1 | 2.0 | Canberra | range | 6/7 | 2.66e-1 | 0.99999999376 | 2.66e3 | 1.64e-3 | 0.084 m RMS | 0.051 m RMS better than wrong | weakly distinguishable |
| G2 | 3.0 | Canberra | range | 6/7 | 9.36e-1 | 0.99999999451 | 9.36e3 | 9.32e-4 | 0.516 m RMS | catastrophic | unstable solve |
| G3 | 5.0 | Canberra | range | 6/7 | 3.52 | 0.99999999554 | 3.52e4 | 5.24e-4 | 0.841 m RMS | catastrophic | unstable solve |
| G6 | 2.0 | Canberra | counted Doppler | 6/7 | — | ~1.0 | 247.5 | — | not run | directional characterization |
| G7 | 2.0 | Canberra | range+Doppler | 6/7 | — | ~1.0 | 260.8 | — | not run | directional characterization |

## 12. Required interpretation and limits

R1G answers the causal questions as follows. The R1 failure is not explained by short arc length alone: additional passes increase conditional information by four orders of magnitude, but the data-only matrix remains rank 6/7 and the nonlinear solves become unstable at 3–5 orbits. Multiple stations did not help in the tested qualified visibility mask because they contributed no additional independent rows. Counted Doppler supplied a distinct sensitivity direction but remained rank 6/7; range+Doppler increased conditional information only from `247.5` to `260.8` in the directional test. No first practical geometry was found. The only stable candidate, G1, showed a modest absolute holdout improvement over wrong-fixed K, but the correct/wrong separation is small and the predictive-recovery ratio is ill-conditioned. Therefore solve-K predictive usefulness is not established.

The sweep did not instrument trajectory-sensitivity norms, eclipse-entry counts, RTN posterior correlations, or fixed-versus-solve RTN covariance in the frozen BLS harness; these are explicitly marked `NOT_RECORDED` rather than fabricated. SRIF and direct range SR-UKF geometry confirmation were not run because the current qualified architectures are not measurement-identical. R0's common Gaussian, default-parity, P21, Model-S, long-arc, event-conditioning, derivative-chain, and zero-sensitivity gates remain the applicable regression evidence.

## Final verdict

`PRACTICAL_GEOMETRY_FOUND=NO`.

`ARC_LENGTH_EFFECT_CLASS=INFORMATION_INCREASE_WITHOUT_PRACTICAL_FULL_RANK_RECOVERY`.

`STATION_GEOMETRY_EFFECT_CLASS=NOT_DEMONSTRATED_BY_QUALIFIED_VISIBILITY`.

`MEASUREMENT_DIVERSITY_EFFECT_CLASS=INSUFFICIENT_IN_TESTED_DIRECTIONAL_CHARACTERIZATION`.

`K_OBSERVABILITY_BEST_CASE_CLASS=WEAKLY_OBSERVABLE_BUT_NOT_ROBUSTLY_PREDICTIVE`.

`PHASE17_R1G_GATE=CHARACTERIZATION_COMPLETE`.

`PRIOR_DIAGNOSTIC_STATUS=PRIOR_DIAGNOSTIC_STILL_UNRESOLVED`.

`PRACTICAL_GEOMETRY_FOUND=NO`.

The next action is a separately authorized multi-arc or additional-observable study. Do not begin R2 closure, merge, or push automatically.
