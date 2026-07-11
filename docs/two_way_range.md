# Two-Way Range Observable (M3)

This document defines the converged two-way range measurement model
implemented in `lunar_od/two_way_range.py` and its estimator integration.
It is separate from the two-way counted-Doppler model
(`docs/two_way_counted_doppler.md`); the legacy counted-Doppler solver
`solve_two_way_light_time` is not modified by M3.

## 1. Event contract

```text
station uplink transmit t1
        |  uplink light time
        v
spacecraft uplink receive t2u
        |  fixed transponder delay  (t2d = t2u + delta_tr)
        v
spacecraft downlink transmit t2d
        |  downlink light time
        v
station receive / measurement time tag t3   (fixed)
```

Ordering is validated: `t1 < t2u <= t2d < t3`, both light times positive,
`transponder_delay_s >= 0`, all epochs and states finite.

## 2. Event equations (uniform seconds)

With `rho_u = |r_sc(t2u) - r_st(t1)|` and `rho_d = |r_sc(t2d) - r_st(t3)|`:

```text
G_u  = t2u - t1  - rho_u / c = 0
G_d  = t3  - t2d - rho_d / c = 0
G_tr = t2d - t2u - delta_tr  = 0
```

All three equations are in seconds, so the event matrix `G_y` is O(1) and
its condition number is a meaningful diagnostic (measured ~2 for nominal
lunar geometry).  The production **solver** exploits the causal structure
(nested scalar fixed-point solves: fixed `t3` -> `t2d` -> `t2u = t2d -
delta_tr` -> `t1`); the **sensitivity** is computed on the converged solution
through the uniform 3x3 implicit system — two deliberately different
structures, cross-checked by an independent `scipy.optimize.root`
simultaneous solve in the tests.

Convergence requires both the event-time update tolerance (default 1e-12 s)
and the light-time equation residual tolerance (default 1e-11 s, ~3 mm),
reported in seconds and metres.  Failure raises
`TwoWayEventConvergenceError`; the last iterate is never used as an
observable.  Event epochs outside the propagated spacecraft history raise
`TwoWayEventHistoryError` (reporting the required pre-roll) instead of
extrapolating; measurement generation drops such receive tags and records
`dropped_measurements` in the metadata, while the estimator-side residual
path lets the error propagate (measurements are never silently skipped).

## 3. Observable convention

Both quantities are carried in every `TwoWayRangeEventSolution`:

```text
raw_half_round_trip:               R_raw = (c/2) (t3 - t1)
delay_calibrated_half_round_trip:  R_cal = (c/2) [(t3 - t1) - delta_tr]
```

The production default convention is `delay_calibrated_half_round_trip`.
The residual is observed-minus-computed in metres.  For a fixed transponder
delay `d(delta_tr)/dx0 = 0`, so both conventions share one initial-state
Jacobian (verified by finite difference).

## 4. Sensitivity derivation

With `y = [t1, t2u, t2d]` and fixed `t3`, the implicit-function theorem on
`G(y, x0) = 0` gives

```text
G_y dy/dx0 = -G_x        =>   dy/dx0 = solve(G_y, -G_x)   (linear solve, no inverse)
```

```text
G_y = [ -1 + u_u.v_st(t1)/c    1 - u_u.v_sc(t2u)/c    0                    ]
      [  0                     0                      -1 - u_d.v_sc(t2d)/c ]
      [  0                    -1                       1                   ]

G_x = [ -(1/c) u_u^T Phi_r(t2u, t0) ]
      [ -(1/c) u_d^T Phi_r(t2d, t0) ]
      [  0                          ]
```

where `u_u`, `u_d` are the station-to-spacecraft unit LOS vectors of the two
legs, and `Phi_r` is the cubic-Hermite-interpolated position block of the
propagated STM at the respective spacecraft event epoch (state and STM are
always evaluated at the *same* event epoch; `t2u` and `t2d` are evaluated
separately for nonzero delay).  The two-way range row is

```text
H_R = -(c/2) dt1/dx0        shape (1, 6), units m per state-unit
```

`H_R` is already an arc-initial-state Jacobian: BLS-LM, SRIF, posterior
information and observability consume the identical `(N, 6)` matrix from
`two_way_range_nominal_and_initial_jacobian` and must not apply the STM
again (mutation-tested).

For zero delay this system reduces algebraically to the validated
counted-Doppler nested elimination in
`round_trip_light_time_initial_state_jacobian` (same `dt2` and `dt1`
formulas after eliminating the trivial `G_tr` row).

Verification on an exact linear-trajectory fixture: event-time sensitivities
and the range Jacobian agree with full central finite differences to
relative error < 1e-6 over position steps {0.1, 1, 10} m and velocity steps
{0.01, 0.1, 1} m/s; the static analytic derivative `dR/dr0 = u_hat`,
`dR/dv0 = u_hat (t2 - t0)` is reproduced to 1e-9.

## 5. Frame and station-state policy (Option A)

Production policy: **exact event-epoch `spice.sxform`** at both station
events.

```text
station fixed state  x_F = [r_ecef, 0]  (WGS84 -> rigid ITRF93)
x_st_rel(t) = solve( X_{ITRF93<-J2000}(t), x_F )     exact sxform at t1 and t3
r_st/Moon(t) = earth_state_MCI(t) + x_st_rel(t)
```

The Earth-center Moon-relative translation is a **separate** approximation:
it is cubic-Hermite interpolated from the pass ephemeris grid
(position+velocity Hermite, `interp_state_history`).  Measured against
direct SPICE `spkezr` at 60 s grid midpoints (2027-03-02 epoch):

```text
cubic Hermite: 1.2e-7 m position, 5.0e-9 m/s velocity   (production, M3)
linear:        1.1 m   position                          (legacy paths)
```

Measured effect of replacing exact sxform with linear transform-grid
interpolation for a representative Earth-Moon two-way measurement
(range sigma 5 m):

```text
grid 10 s: station 0.27 m, range 0.086 m  (1.7%  of sigma)
grid 30 s: station 1.03 m, range 0.321 m  (6.4%  of sigma)
grid 60 s: station 2.16 m, range 0.674 m  (13.5% of sigma)
```

A 60 s grid violates the `0.1 sigma` frame-error budget, which is why
interpolation was not adopted for production.  These comparisons are
diagnostics only (`tests/test_two_way_range_integration.py`); no interpolated
transform grid exists in the production two-way range path, and the
counted-Doppler pass-grid interpolation behavior is unchanged.

The uplink epoch `t1` precedes the first receive tag by up to one round-trip
light time; the Earth ephemeris grid extrapolates linearly over that
pre-grid interval with error bounded by `0.5 * a_EM * dt^2 ~ 1 cm` for
`dt ~ 2.7 s` (Earth-Moon relative acceleration ~2.7e-3 m/s^2) — far below
the range noise.  The spacecraft history is never extrapolated.

## 6. Measurement, estimators, configuration

- Observation rows: `[t3, range_m, station_id_1based, time_index_1based
  (, arc_id)]`; noise reuses `station.sigma_range_m`
  (`two_way_range_noise_source: station_sigma_range_m` — not a claim that
  one-way and two-way ranges share physical noise; a dedicated sigma is an
  M4 item).
- `estimate_two_way_range_bls_lm` / `estimate_two_way_range_srif` mirror the
  existing estimator loop structure with the shared helpers
  (`_prior_information_and_scale`, `_lm_step`, robust reweighting, QR SRIF
  update).  Bias solve-for states are rejected in M3.
- Scenario config: `measurement_type: two_way_range`,
  `two_way_range_convention`, and the shared `transponder_delay_s` field
  (also used by counted-Doppler `RangeRatePhysicsConfig`; ownership is
  documented at the field).  UKF + two_way_range is rejected with an
  explicit `ValueError` at the filter, the scenario runner, and config
  validation.

## 7. Relationship to counted Doppler

Zero-delay consistency (measured, 10 s pass grid, real frames):

```text
round-trip light time: new vs legacy solver   4.9e-10 s  (0.15 m equivalent)
counted-Doppler m/s equivalent vs endpoint
half-round-trip range difference / Tc         6.0e-6 m/s
```

Both differences are bounded by the legacy solver's transform-grid linear
interpolation, consistent with the measured frame-audit numbers.

For **nonzero** transponder delay the models deliberately differ: the legacy
counted-Doppler solver keeps a single bounce state (`r_sc(t2)` used for both
legs, delay inserted only in the epoch bookkeeping), while the M3 solver
evaluates `r_sc(t2u)` and `r_sc(t2d)` separately.  The legacy nonzero-delay
behavior is intentionally unchanged in this phase.

## 8. Limitations

```text
same uplink and downlink station only
fixed coordinate-time transponder delay (no drift, no relativistic terms,
  no proper-time conversion, no delay solve-for)
no clock / EOP / station-coordinate sensitivities
no media corrections
no measurement bias solve-for for this observable (M4)
no UKF integration
SPICE validates station states, frames, and Earth/Moon ephemeris;
  spacecraft event equations are validated by an independent numerical
  solver (no spacecraft SPK exists)
Earth-center ephemeris interpolation (cubic Hermite) is measured, not exact
```
