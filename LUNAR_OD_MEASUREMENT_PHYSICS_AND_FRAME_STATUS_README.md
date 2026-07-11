# Lunar OD Measurement Physics and Frame Status

Project-state README for the measurement-physics track (M1-M5 roadmap).
Every statement below is verified against the repository, its committed
documentation (`docs/`), and the isolated M3 regression run of 2026-07-11;
no external or remembered numbers are used.

## M2.1 — Light-time sensitivity contract: COMPLETE

The implicit light-time sensitivity (`d(tau)/dx0` from the implicit
light-time equation) is implemented and documented in
`docs/one_way_light_time_jacobians.md`: receive-time-tagged measurements,
station and J2000->ITRF93 transform at receive time, spacecraft state and
STM at the converged transmit time, STM position block evaluated at the
transmit epoch (never the receive epoch).

## M2.2 — One-way range implicit Jacobian: COMPLETE (opt-in)

`jacobian_model: implicit_light_time` is an explicit, opt-in scenario-config
choice (`ALLOWED_JACOBIAN_MODELS`). The default geometric and
`analytic_first_order_light_time` paths are unchanged (regression-locked).

## M2.3 — Azimuth/elevation Jacobian propagation: COMPLETE

Closed by commit `4f98d83` ("Close M2.3 and frame audit with real Windows
SPICE validation"). The apparent line-of-sight sensitivity chain
(light-time corrected LOS -> optional stellar aberration -> receive-epoch
frame transform -> topocentric az/el) is covered by
`docs/one_way_light_time_jacobians.md` and the stellar-aberration V&V suite
(`docs/spice_cross_validation.md`: `apply_stellar_aberration` vs SPICE
`stelab`, arcsecond-exact in controlled geometries).

## Frame audit: CLOSED — no correctness bug found

`docs/frame_transformations.md` (audit dated 2026-07-11): "no
frame-correctness bug was found in the production one-way measurement
chain." The contract (left-multiplication with target-from-source
operators, SI units, epoch conventions) is locked by
`tests/test_frame_transformations.py` (SPICE-free) and
`tests/test_frame_spice_validation.py` (kernel-gated).

## M3 — Converged two-way range observable: IMPLEMENTED AND VALIDATED

Reference: `docs/two_way_range.md`; implementation
`lunar_od/two_way_range.py`; tests `tests/test_two_way_range.py` (27) and
`tests/test_two_way_range_integration.py` (16).

Event contract (fixed receive tag `t3`):

```text
t1 (station uplink transmit) < t2u (spacecraft uplink receive)
   <= t2d = t2u + delta_tr (downlink transmit) < t3 (station receive)
```

Event equations, all in seconds (`rho_u = |r_sc(t2u) - r_st(t1)|`,
`rho_d = |r_sc(t2d) - r_st(t3)|`):

```text
G_u  = t2u - t1  - rho_u / c = 0
G_d  = t3  - t2d - rho_d / c = 0
G_tr = t2d - t2u - delta_tr  = 0
```

Observable conventions (both carried in every solution; production default
is delay-calibrated):

```text
raw_half_round_trip:               R_raw = (c/2) (t3 - t1)
delay_calibrated_half_round_trip:  R_cal = (c/2) [(t3 - t1) - delta_tr]
```

Initial-state sensitivity via the implicit-function theorem on the 3x3
event system (`G_y dy/dx0 = -G_x`, linear solve), giving the two-way range
row `H_R = -(c/2) dt1/dx0`. It is already an arc-initial-state Jacobian;
estimators and observability consume the identical `(N, 6)` matrix and must
not apply the STM again (mutation-tested). Finite-difference agreement:
relative error < 1e-6 over the documented step sweeps; static analytic
derivative reproduced to 1e-9. Zero-delay reduction to the legacy
counted-Doppler elimination is verified; the legacy solver is not modified.

### Exact-sxform policy (Option A)

Production evaluates **exact event-epoch `spice.sxform`** at both station
events (`t1`, `t3`). Measured cost of a linearly interpolated transform grid
(range sigma 5 m): 10 s grid -> 0.086 m range error (1.7% of sigma), 30 s ->
0.321 m (6.4%), 60 s -> 0.674 m (13.5%). A 60 s grid violates the 0.1-sigma
frame-error budget, so interpolation was rejected for the production path;
those comparisons remain as diagnostics only. The Earth-center Moon-relative
translation is a separate, measured approximation (cubic Hermite on the pass
grid: 1.2e-7 m position, 5.0e-9 m/s velocity vs direct `spkezr` at 60 s
midpoints; the legacy linear path is ~1.1 m).

### Independent solver cross-check (exact maxima, this run)

Production nested fixed-point solver vs a structurally independent
`scipy.optimize.root` (hybr) solution of the simultaneous 3x3 event system
(0.25 s transponder delay, lunar-distance moving geometry):

```text
max |production t1  - reference t1 |    = 0.000e+00 s
max |production t2u - reference t2u|    = 0.000e+00 s
max |production t2d - reference t2d|    = 0.000e+00 s
max |production range - reference range| = 0.000e+00 m
```

Bit-identical agreement at double precision (test gate: 1e-10 s / 0.05 m).

### Estimator integration and residual-consistency result (this run)

Single short pass, perturbed start (|dr0| = 3.419e+01 m,
|dv0| = 1.342e-02 m/s), noise-free observations, max_iter=8:

```text
              BLS-LM          SRIF
stop          MaxIter         MaxIter
iterations    8               8
final cost    1.389e-07       1.231e-09
residual RMS  10.520 m -> 4.167e-04 m
                              10.520 m -> 3.923e-05 m
pos err       3.419e+01 -> 3.022e+01 m
                              3.419e+01 -> 2.277e+01 m
vel err       1.342e-02 -> 1.286e-02 m/s
                              1.342e-02 -> 9.125e-03 m/s
```

(Initial cost is not exposed by `EstimatorStats`; the initial residual RMS
above is the measured pre-fit proxy.) Interpretation: the residual RMS
collapses by 4-5 orders of magnitude at the estimate, but the truth-state
error remains near its initial level (SRIF ~33% reduction, BLS-LM ~12%) —
a single short two-way-range pass is weakly observable transverse to the
line of sight. The test is therefore named and documented as an
**integration and residual-consistency test**, not a truth-recovery test;
no estimator criterion was loosened.

## Authoritative M3-only regression baseline (isolated tree, 2026-07-11)

With unrelated Phase 13G work stashed and only the staged M3 patch present:

```text
tests/test_two_way_range.py                          27 passed
tests/test_two_way_range_integration.py              16 passed
measurements + estimators + observability            64 passed (+2 subtests)
frame transformations + frame SPICE validation       29 passed
FULL tests/ (M3-only tree)     515 passed, 20 skipped, 0 failed,
                               0 deselected, 1 warning, ~22 s
```

The 20 skips are environment-gated (unchanged); the single warning is the
known intentional `RuntimeWarning` from the slow numerical-Jacobian
reference in `test_estimators.py`.

## Known limitations / technical debt

- `lunar_od/two_way_range.py` imports the **private** cubic-Hermite helpers
  `_interp_state` and `_interp_state_transition_position` from
  `lunar_od/radiometrics.py`; promoting them to a shared public
  interpolation module is deferred technical debt (no refactor in M3).
- Same uplink and downlink station only; fixed coordinate-time transponder
  delay (no drift, relativistic terms, proper-time conversion, or delay
  solve-for).
- No clock / EOP / station-coordinate sensitivities; no media corrections.
- No measurement-bias solve-for for this observable and no dedicated
  two-way-range noise sigma (station `sigma_range_m` is reused) — M4 items.
- No UKF integration (explicitly rejected at filter, runner, and config).
- SPICE validates station states, frames, and Earth/Moon ephemeris; the
  spacecraft event equations are validated by an independent numerical
  solver (no spacecraft SPK exists).
- Legacy counted-Doppler nonzero-delay single-bounce behavior is
  intentionally unchanged.

## Next recommended phase: M4

Measurement noise, bias, and data editing: elevation/station-dependent
noise, a dedicated two-way-range sigma, bias and clock solve-for states
(including the two-way-range bias rejected in M3), and synthetic
outlier/dropout with robust editing.
