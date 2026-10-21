# Phase 17-HARDEN-D18 / D19: bounded interpolation (regression summary)

Status: **REPAIRED** (both). Branch `fix/harden-d18-d19-interp-bounds` from `main` 9a65327.
Owner authorisation for the `accelerated.py` re-freeze: 2026-10-21, for exactly the
guard-only change below.

## Defects

| ID | Location | Before | After |
|---|---|---|---|
| D18 | `accelerated.make_lerp_vec3`, `accelerated.rk4_6state` | the binary-search lerp extrapolated the edge interval without a bound on both sides of the ephemeris grid (scipy fallback: `fill_value="extrapolate"`) | `EphemerisCoverageError` beyond `EPHEMERIS_GRID_EXTRAPOLATION_MARGIN_S = 10 s`, the HARDEN-D15 margin of `MoonCenteredEphemeris` |
| D19 | `radiometrics._interp_vector` and the edge branches of `_interp_matrix`, `_interp_state`, `interp_state_history`, `_interp_state_transition_position`; `radiometrics._interp_array_and_slope`; `filters._interp_pass_values` | linear edge-interval extrapolation without a bound; a NaN epoch returned NaN (`_interp_vector`) or an `IndexError` (`_interp_state`) | `HistoryDomainError` via `history_domain.require_supported_epoch`. Generic helpers: at most `GENERIC_HISTORY_MAX_EXTRAPOLATION_S = 10 s` (same HARDEN-D15 margin), more only with an explicit `max_extrapolation_s`. The two production-only helpers: closed support + the 2-ULP allowance. Non-finite epochs raise |

The three guards share one number: the 10 s light-time pre-roll margin that
HARDEN-D15 already qualified for `MoonCenteredEphemeris`.

## Bound selection (measured, not assumed)

A pytest plugin wrapped every interpolator call and recorded the distance outside
the grid on the unmodified code (full suite: 1 826 tests; about 2.07 M generic-leaf
calls, 45 738 UKF pass-history calls, 268 fast-RK4 intervals):

| Population | Out-of-grid requests | Worst |
|---|---|---|
| Measurement-event lookups in the suite (one-way, counted Doppler, M3 range, ΔDOR, UKF local histories) | 0 | 0 s |
| R1O-D2 predeclared grid: 9 geometries × {GC, GM, MC} × {production visibility, elevation-only} ΔDOR rows, incl. station-B receive (separate driver) | 0 | 0 s |
| Fast RK4 `[t0, tf]` (UKF sigma steps, the only D18 caller) | 0 of 268 | 0 s |
| Legacy R1O-D campaign (`phase17_r1od_campaign`, via `test_harden_mutants`): backward light-time probe at the first grid node / station B | 288 / 72 | 1.3 s before / 0.004 s after |
| Test-side Earth callables built on `_interp_state` inside scipy VODE right-hand sides (`dynamics._propagate_vode` steps past the last output epoch); 4 fixtures | 15 000+ | 87.8 s (1.46 grid intervals) |

Choice:

* The ledger's "ΔDOR t_B ≤ ~0.03 s past the grid" was an upper estimate; no current
  geometry reaches it.
* `two_way_range.py` (M3, byte-frozen by the R3-P21-M3 gate) documents a qualified
  linear Earth extrapolation to a pre-grid uplink epoch `t1`, up to one round-trip
  light time (< 3 s). Together with the legacy R1O-D 1.3 s probe, this sets the
  default at the D15 light-time margin of 10 s.
* The VODE overrun is a property of the test fixtures, whose Earth grids end exactly
  at the arc. They opt in explicitly (`max_extrapolation_s=300.0`, about 3.4 times the
  measured worst). No production module needs an opt-in.
* `filters._interp_pass_values` and `radiometrics._interp_array_and_slope` are reached
  only after their callers normalise onto closed support (FA-03B), so they grant no
  margin.

## Bitwise invariance

* Accepted requests run the unchanged arithmetic; the guard only decides whether to
  raise. `_interp_state`'s edge test became `not (t0 < t < tN)`, identical to
  `t <= t0 or t >= tN` for every finite `t`.
* D18 touches Python wrappers only; every `@_optional_njit` kernel is byte-identical.
  `accelerated.py` sha256 e455b36b → ae5dea08 (recorded as
  `PHASE17_HARDEN_D18_REFROZEN_CURRENT_TREE_SHA256`).
* Per-call oracle: during the full post-change suite, every call of
  `radiometrics._interp_vector` (221 208) and `filters._interp_pass_values` (4 560)
  was re-evaluated with the verbatim origin/main implementation and compared byte
  for byte: **0 mismatches**. All 35 refusals come from the new regression tests.
* Full suite, same interpreter and environment:

  | | passed | skipped | failed |
  |---|---|---|---|
  | before (9a65327) | 1 795 | 29 | 2 |
  | after | 1 828 | 29 | 2 |

  The only outcome differences are the 33 new tests. The two failures are
  pre-existing and environmental, identical before and after:
  `test_harden_import_isolation::test_plain_invocation_without_pythonpath_isolates`
  (the shared venv's editable install points at another checkout) and
  `test_tracking_console::test_main_window_creates_only_requested_pages` (a persisted
  desktop page).

## Regression tests

`tests/test_harden_d18_d19_interpolation_bounds.py` (33 tests). Run against the
origin/main sources, it fails 21 (every defect detector) and passes 12 (exactly the
bitwise-equality checks). So it detects both defects and pins the legacy bytes.

## Residual

Study scripts outside the suite that call `_interp_state` directly were covered
through the suite (R1O-D2, R1O-D3, CD1, D23 tests) and the R1O-D2 grid driver. A
script that extrapolates more than 10 s now raises `HistoryDomainError` instead of
continuing silently.
