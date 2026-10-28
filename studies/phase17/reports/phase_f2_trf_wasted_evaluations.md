# F2 TRF path — evaluations spent after convergence (analysis + proposal, not implemented)

2026-10-28. Read-only analysis of the progress logs from the G4e-R refits that were killed on 2026-10-27
(`outputs/numerical-core/trf/run1_killed_2026-10-27/`, gitignored). Each progress-log line is one Jacobian
linearization (accepted point). `cost` is the whitened χ² including priors.

## What the solver is told

`bounded_least_squares.solve_bounded` calls `scipy.optimize.least_squares(method='trf', ftol=None, xtol=None,
gtol=1e-12, max_nfev=30)`. Its docstring says cost and step stopping are disabled on purpose: "small improvements are
not stationarity".

At χ² ≈ 1e4 the scaled gradient never falls below 1e-12. In practice, then, every F2 fit runs until `max_nfev = 30`.
Stationarity is certified separately, at the returned point (`posterior_diagnostics`: GN decrement ≤ 0.1 formal σ,
which is a remaining predicted Δχ² ≤ 0.01).

## What the logs show

| fit | linearization | χ² | Δχ² | wall [min] |
|---|---|---|---|---|
| F2g S1 | 2 | 12 809.01 | −1.8e6 | 436 |
| | 3 | 10 764.871 | −2 044 | 523 |
| | 4 | 10 764.858 | **−0.013** | 625 |
| | 5 | 10 764.837 | **−0.022** | 714 (killed at 776) |
| F2h S1 | 3 | 11 012.570 | −1 200 | 523 |
| | 4 | 11 012.545 | **−0.026** | 727 |
| F2g S2 stage 1 (1 day) | 4 | 3 416.6246 | −3.4e4 | 359 |
| | 5–7 | 3 416.6173 | **−0.005, −0.002, 0** | 395 → 544 |
| F2i S2 | 5 | 11 012.658 | — | 697 |

- After the third linearization each further one gains only |Δχ²| ≈ 0.01–0.03. That is the size of the 0.1σ
  certification threshold itself.
- In the 1-day stage, linearizations 5–7 gain 0.007 in total. The run spent 185 min on them.
- Between accepted points, wall time ranges from 87 to 345 min. A Jacobian costs ~23 propagations (~8 pool rounds); a
  bare residual costs 1 propagation, on one worker while two sit idle. Long gaps therefore mean several rejected
  trial steps. The logs record only linearizations, so `nfev` cannot be reconstructed: this is a logging gap.
- Rough estimate: ~40 % of an F2 TRF fit's wall time is spent after χ² has settled to within 0.03.

The same logs give an early, non-final scientific signal. F2i S2 (shifted start) reached χ² 11 012.66 against F2h S1's
11 012.54 with the same model, and both have K = 0.07493. F2g S2 stage 2 was at 10 816 (still descending) against
S1's 10 764.84. The fits were killed, so these are not records.

## Proposal (owner decision; would change the numerical identity)

1. **Log every residual evaluation** (`evaluation_kind = "residual"`, with x, χ² and elapsed time) beside the
   linearizations, so `nfev` and rejected steps become visible. This is diagnostics only and leaves the numbers unchanged.
2. **Stop on certified stationarity.** Pass a `callback` to `least_squares` (SciPy ≥ 1.16; the venv has 1.17.1).
   After each accepted iteration it evaluates `posterior_diagnostics` on the cached Jacobian at that point (an SVD of
   a 15-column matrix, about a second). If `stationary` holds, it raises `StopIteration`.
   - The final-point diagnostics are still recomputed independently at `fit.x`, so the G1 definition does not change.
   - Only work that cannot change a certified result is removed.
   - `max_nfev = 30` stays as the hard limit for non-stationary runs.
3. A test: a toy problem where `stationary` is reached before `max_nfev` stops early, returns the same `x` (to the
   certification tolerance) as a run to `max_nfev`, and reports the same status.

**Not done now, on purpose.** Any change to `phase_f2_lro_od.py` or `src/lunar_od/` changes `numerical_identity()`.
`_require_current_runs` would then refuse to merge the refits that are running now. The proposal should land only
after these refits finish, or together with a decision to restart them.
