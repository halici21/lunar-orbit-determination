# Phase D6: analytic measurement partials for the EKF

**Outcome as computed: `D6_QUALIFIED`.** All five gates pass.

**What changes for the user.** With `measurement_jacobian="analytic"`, the EKF on two-way counted Doppler runs **4.4× faster** (median 6.34 s → 1.45 s on a 1 h arc). The iterated smoother runs 3.5× faster. The estimates do not change: they agree with the finite-difference EKF to ≤ 1.4e-5σ.

**The default stays `finite_difference`.** D3–D5 numbers therefore cannot change.

- Predeclaration: [`phase_d6_ekf_analytic_partials_predeclaration.json`](../configs/phase_d6_ekf_analytic_partials_predeclaration.json). Committed at `0bca109`, before the implementation, any timing or any study number.
- Driver: [`phase_d6_ekf_analytic_partials.py`](../scripts/phase_d6_ekf_analytic_partials.py).
- Results: [`phase_d6_ekf_analytic_partials.json`](../results/phase_d6_ekf_analytic_partials.json).
- Code: new [`src/lunar_od/ekf_partials.py`](../../../src/lunar_od/ekf_partials.py); the `measurement_jacobian=` option in [`ekf.py`](../../../src/lunar_od/ekf.py); tests in [`tests/test_ekf_partials.py`](../../../tests/test_ekf_partials.py).
- `filters.py`, `measurements.py`, `radiometrics.py`, `estimators.py` and `dynamics.py` are unchanged.

## 1. What was built

The EKF needs dh/dx at the update epoch. D6 reuses the BLS-LM partials, taken locally instead of mapped to the arc's initial state.

| Observable | Source of the partials |
|---|---|
| Position (range/az/el) and instantaneous range rate | The local H̃ blocks of `compute_position_residuals_analytic` / `compute_range_rate_residuals_analytic` |
| Two-way counted Doppler: range, az, el rows | The geometric H̃, as BLS uses |
| Two-way counted Doppler: Doppler row | `two_way_counted_doppler_initial_state_jacobian` on the observable's own local window (see below) |
| Bias states | Differenced, as before. They enter additively and hit the measurement cache, so they cost no physical evaluation. |
| DMC states | Zero columns |

**The Doppler row in detail.** The local window is the one `_two_way_local_histories` builds for the observable itself. On that window, a 6+36 augmented history is propagated both ways from the update-epoch state, with STM = I at the receive mid-time. The "initial state" of the existing Jacobian is therefore the update-epoch state.

**Scope.** The option is accepted only where the partials are the exact derivative of what the EKF evaluates:

- geometric-instantaneous position;
- instantaneous range rate;
- counted Doppler with `companion_geometry="instantaneous"` and `local_state_model="ode"`.

Anything else (for example a Taylor local history or apparent companion geometry) raises `ValueError` rather than silently approximating.

## 2. Experiment

**Arc.** D3's synthetic 1 h arc: Moon point mass, 3 stations, rtol 1e-12 / atol 1e-13. The counted-Doppler setup:

- count interval 20 s;
- legacy interpolated station states;
- observations every 60 s from 60 s to 3540 s, so every count window lies inside the pass grid.

**Runs.** 20 seeds per type with an S20m prior (linear regime), in a single process with one BLAS thread.

**Jacobian oracle.** A Richardson-extrapolated central difference (h = 10 m / 1 cm/s), evaluated at the 3 stations' observations at t = 600, 1200, …, 3000 s. The state is the truth perturbed by seed 0's prior draw.

## 3. Results

**Jacobian error against the oracle.** The figure is the largest relative row error over 15 observations per type.

| Type | Analytic, per row | Central differences used by the D3–D5 EKF, per row |
|---|---|---|
| Position (ρ, az, el) | 2.1e-10, 1.6e-10, 1.1e-8 | 5.1e-10, 5.2e-10, 4.6e-8 |
| Instantaneous RR (ρ, ρ̇, az, el) | 1.4e-10, 7.5e-11, 1.4e-10, 1.2e-8 | 5.6e-10, 1.3e-10, 3.8e-10, 4.4e-8 |
| Counted Doppler (ρ, Doppler, az, el) | 2.0e-10, **3.4e-8**, 1.5e-10, 1.2e-8 | 7.2e-10, **1.7e-7**, 3.3e-10, 4.9e-8 |

**EKF with analytic vs finite-difference partials (20 seeds).**

| Type | Max final-state difference (σ) | Max smoothed first-epoch difference (σ) | Mean final NEES, FD / analytic | Median wall time, FD → analytic |
|---|---|---|---|---|
| Position | 4.4e-8 | 6.5e-8 | 6.612 / 6.612 | 0.169 s → 0.122 s (1.4×) |
| Instantaneous RR | 4.3e-7 | 6.9e-7 | 5.761 / 5.761 | 0.282 s → 0.177 s (1.6×) |
| Counted Doppler | 5.3e-6 | 1.4e-5 | 4.675 / 4.675 | **6.34 s → 1.45 s (4.4×)** |

**Iterated smoother on counted Doppler (reported only, 5 seeds).**

- All 10 runs Converged in 1 iteration.
- Analytic and finite-difference results agree to ≤ 1.7e-6σ.
- Wall time falls from 13.1–14.1 s to 3.4–3.9 s (3.5×).

## 4. Gates

| Gate | Result | Evidence |
|---|---|---|
| G1 Jacobian | **PASS** | Analytic ≤ 1.1e-8 for position and instantaneous RR (bound 1e-6). Counted-Doppler row 3.4e-8 (bound 1e-4); its other rows ≤ 1.2e-8 (bound 1e-6). |
| G2 filter equivalence | **PASS** | ≤ 1.4e-5σ, against a bound of 0.01σ. |
| G3 consistency | **PASS** | Mean NEES 6.61, 5.76 and 4.67, all inside [4.19, 8.18]. |
| G4 cost | **PASS** | Counted Doppler 4.4× faster, against the required ≥ 3×. |
| G5 invariance | **PASS** | `tests/test_ekf_smoother.py` and `tests/test_dmc.py` pass unchanged with the new code (29 tests together with `tests/test_ekf_partials.py`). |

## 5. What the predeclaration got wrong

**The accuracy motivation did not hold.** The predeclaration expected the finite-difference counted-Doppler row to be degraded by the light-time solve tolerance (1e-10 s). It was not: its error is 1.7e-7 relative, because the solve converges far tighter than its tolerance.

**What the analytic Doppler row actually buys.** It is 5× more accurate, but both errors are irrelevant to the estimate. **D6's gain is cost, not accuracy.** For instantaneous observables the gain is only 1.4–1.6×, because each measurement evaluation is already cheap.
