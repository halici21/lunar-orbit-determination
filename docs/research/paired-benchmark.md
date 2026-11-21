# Paired BLS-LM / SRIF / SR-UKF comparison

This bounded Level-0 experiment uses **32 paired seeds per noise block**, 64 seeded arcs and 192 estimator fits. The two noise blocks share seeds; they are not 64 independent geometries. [Predeclared design](paired-benchmark-plan.md).

Producing commit: `4d8819ab16fb115caca81c311aa23421c0d41be3`; clean source: `True`. Campaign elapsed time: 39.01 s on the recorded environment.

![Across-seed final position error](../figures/benchmark/position-distribution.png)

| Noise | Estimator | Median / p95 position (m) | Median velocity (m/s) | Numerical criteria | Solver termination | Mean NEES |
|---|---|---|---|---|---|---|
| 1× | BLS-LM | 52.279 / 128.687 | 0.06306 | 32/32 | 100.0% | 4.990 |
| 1× | SRIF | 52.280 / 128.687 | 0.06306 | 32/32 | 100.0% | 4.990 |
| 1× | SR-UKF | 52.277 / 128.684 | 0.06307 | 32/32 | 100.0% | 4.990 |
| 3× | BLS-LM | 96.922 / 176.045 | 0.13206 | 32/32 | 100.0% | 5.175 |
| 3× | SRIF | 96.922 / 176.045 | 0.13206 | 32/32 | 100.0% | 5.175 |
| 3× | SR-UKF | 96.924 / 176.031 | 0.13204 | 32/32 | 100.0% | 5.176 |

## Interpretation and scientific review

The table reports the predeclared numerical criteria separately from solver termination. Every attempted trial is retained. MaxIter counts are: BLS-LM=0, SRIF=0, SR-UKF=0. Such stopped trials are not called converged. A scientific acceptance decision must inspect these outcomes and the complete diagnostics; numerical criteria alone are insufficient.

Final errors are close in this fixed, well-behaved geometry; the data do not establish general estimator superiority. BLS-LM/SRIF estimate an initial state using the complete arc; the UKF filters sequentially. Only their aligned final epoch is compared. Priors, R, zero Q and sigma-point settings are explicit in the design and inputs.

Final NEES has 6 degrees of freedom. For N=32 independent trials **within one block**, the predeclared 95% mean-NEES interval is [4.860, 7.258]. This is a diagnostic under the matched Gaussian prior and same-model assumptions, not a blanket calibration certificate. NIS is recorded for UKF innovations; batch post-fit residuals are stored separately. No independent-sample chi-square claim is made over the 124 correlated within-arc updates. Per-channel means, RMS and acquisition-order lag-1 correlations are in trials.json; no formal whiteness or station-geometry robustness claim is established.

![Across-seed runtime](../figures/benchmark/runtime-distribution.png)

Runtime includes fit, endpoint propagation and diagnostics; batch and sequential paths perform different work. Measurements are serial wall-clock values from the saved environment, without warm-up or hardware normalization. They do not establish portable speed ratios.

## Reproduction and evidence

```text
python examples/benchmark_estimators.py --output outputs/benchmark
```

- [Summary, exact configuration, source/environment identities and all distributions](../figures/benchmark/summary.json)
- [Per-trial metrics, solver stops and failure fields](../figures/benchmark/trials.csv)
- [Residual/innovation channel diagnostics](../figures/benchmark/trials.json)
- [Raw paired observations, truth, initial states, priors, final covariances and UKF histories](../figures/benchmark/raw-trials.zip)
- [Published artifact identities](../figures/benchmark/published.json)

The summary hashes each original NPZ member inside raw-trials.zip; published.json additionally hashes the archive and curated figures. No errors, exceptions, stopped solves or outliers are silently excluded. This same-model, idealized visibility experiment does not qualify SPICE, real station availability, model mismatch or flight performance.
