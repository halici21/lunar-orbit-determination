# Predeclared paired estimator experiment

Question: how do final-epoch error, covariance diagnostics and runtime vary
between BLS-LM, SRIF and a sequential square-root UKF under matched inputs?
This plan is committed before the campaign results are generated.

- Experimental unit: one seeded 1800 s synthetic arc, not an observation row.
- Two blocks: nominal measurement sigmas (5 m range, 10 µrad angles), and 3×
  those sigmas. Each block uses the same 32 declared seeds (41000–41031).
- Geometry and same-model truth reuse the public quickstart: four fixed
  synthetic Earth stations, identity frame transform, lunar point mass only,
  60 s cadence, 124 range/azimuth/elevation observation rows. No visibility
  filtering, light time, aberration, bias or model mismatch. Truth level 0.
- Each seed/block gives all three estimators byte-identical observations, truth,
  initial state and initial covariance. Observation noise uses `default_rng(seed)`;
  initial error uses `SeedSequence([seed, 1])` and the declared Gaussian sigmas.
  Reusing a seed across noise blocks makes the blocks paired, not independent.
- All estimators receive the same explicit Gaussian initial-state prior:
  diagonal covariance from `[200,200,200,0.1,0.1,0.1]` in m and m/s, centred
  on the same perturbed state. Batch priors are supplied explicitly; the UKF
  starts at that same mean and covariance.
- R is the diagonal station noise covariance in m²/rad²; it matches the noise
  generator in each block. UKF Q is zero (`process_noise=None`), square-root
  covariance, α=.35, β=2, κ=0, jitter=1e-12. No adaptation, inflation,
  gating, robust weighting, STM linearization or custom fast propagator.
- Integrator tolerances match the tutorial (rtol=1e-11, atol=1e-12).
  Batch max_iter=40, with the existing default solver stopping criteria.
- Primary endpoint: Euclidean position and velocity error at **1800 s** for all
  three methods. Batch states and posterior covariance are propagated from the
  initial epoch with the STM; UKF returns its last update at the same final epoch.
  Batch smoothing and sequential filtering differ in their intermediate histories;
  no intermediate error ranking is inferred.
- Common numerical success: finite state/covariance, positive-definite final
  covariance, position error <1000 m and velocity error <1 m/s. Solver termination
  is reported separately (batch Converged/J-Stab vs MaxIter/Singular); exceptions
  and incomplete trials remain visible. Outlier: final position error >500 m.
- Report N, mean/std/min/max, p5/median/p95, success/failure/outlier rates,
  solver termination and runtime distributions. Paired differences use identical
  seeds. No post-hoc seed exclusions or estimator tuning.
- Record final NEES (6 dof). The 95% aggregate chi-square interval uses 6N dof
  across independent seeds within each block. Report the actual complete N.
  UKF innovations/NIS (3 dof per observation) and batch post-fit residuals are
  distinct; no chi-square independence claim is made over within-arc NIS samples.
- Save raw observations, sampled initial states, per-trial metrics/failures,
  final covariance, residual/innovation diagnostics, config and source hashes,
  environment and producing commit. Never replace a frozen study baseline.
- Resource bound: 64 paired trials / 192 fits, serial, at most 30 minutes.
  A one-trial pilot checks the runner; a partial timed-out campaign is explicitly
  incomplete and cannot be presented as a complete result.

Reproduce from an installed checkout:

```text
python examples/benchmark_estimators.py --output outputs/benchmark
```

This is a bounded synthetic comparison, not a mission, visibility, independent
propagation or operational accuracy qualification. Limited sample size and fixed
geometry constrain generalization. The results require the scientific review
record supplied with the campaign.
