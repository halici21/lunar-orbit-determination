# Phase 17-HARDEN-D23 — Counted-Doppler Earth Ephemeris Repair and Value-Level Qualification

- **Scope:** HARDEN qualification, not a science phase. No estimator, K parameterization, ΔDOR or force-model change.
- **Code:** `lunar_od/radiometrics.py` (repair); `examples/phase17_d23_qualification.py` (qualification).
- **Tests:** `tests/test_harden_d23_counted_doppler_earth.py`.
- **Acceptance criteria:** `artifacts/phase17_d23_acceptance.json`, canonical sha256 `2d1cb370…`. They were committed in
  `bc640fa` before any repaired result.
- **Evidence:** `artifacts/phase17_d23_*`.

## 1. Outcome

| frozen criterion | result | gate |
|---|---|---|
| production vs direct-SPICE model S (all 239,265 rows) | max **7.64e-8 m/s**, rms **1.58e-8** | ≤ 1e-7 / ≤ 2e-8 — **PASS** |
| production vs direct-SPICE model F (every 10th row) | max 7.04e-8 m/s | ≤ 1e-7 — **PASS** |
| historical linear-Earth mutant reproduced | max **1.83e-3 m/s** | in [1e-3, 3e-3] — **PASS** |
| reduction factor | **2.4e4** | ≥ 1e3 — **PASS** |
| solver floor (production vs model S, identical inputs) | max 9.9e-10 m/s | ≤ 1e-8 — **PASS** |
| CD1 information (240 cells) | f_perp ≤ 4.0e-10; decision identical; 0 class flips | **PASS** |
| KP1 information (240 cells) | metrics ≤ 1.0e-9; classes identical; 0 cell flips | **PASS** |
| CD1 FD oracle rerun on the repaired path | state and K gates | **PASS** |
| Earth epoch contract (Hermite vs SPICE at the event epochs) | 6.39e-5 m | ≤ 1e-6 m — **FAIL as frozen** (see §5) |

- **Strict result.** `PHASE17_HARDEN_D23_REPAIR_GATE` is **FAIL on one criterion**.
- **What the failing number measures.** The failing diagnostic measured the *oracle's* float64 time rounding.
  - It evaluated SPICE at fl(et0 + t) for arbitrary event epochs.
  - Its value equals the bound |v_Earth|·ulp(ET)/2 ≈ 6.4e-5 m.
- **The same criterion measured exactly.** The same frozen 1e-6 m criterion, evaluated at exactly representable
  epochs over every grid interval of all 24 epochs (197,904 points), gives **2.42e-7 m**: PASS.
- **Status.** The exception is adjudicated, not relabelled. It stands until the owner accepts it.
- **Owner decision (2026-10-07): ACCEPTED.** The frozen result above is not rewritten; with the adjudicated
  exception accepted, `PHASE17_HARDEN_D23_REPAIR_GATE = PASS` and `EARTH_EPOCH_CONTRACT = QUALIFIED`.

## 2. Root Cause (confirmed)

- **The design choice.** R3-P22 kept the counted-Doppler Earth on **linear** grid interpolation. Its only stated
  purpose was that the exact path reproduce the byte-frozen R2 model S, so that S−L would isolate the site transform.
- **The mechanism.**
  - The count interval is centred on a grid node at the **receive** epochs t_mid ± Tc/2, where the linear-interpolation
    errors are equal and cancel.
  - The uplink **transmit** epochs t_mid ± Tc/2 − τ (τ ≈ 2.6 s) sit asymmetrically inside their 90 s intervals: median
    interval fraction 0.64 at the start, 0.31 at the end.
  - Their errors do not cancel.
- **Epoch diagnostics** (6,000 rows): the line-of-sight difference of the linear errors is ≤ **0.22 m** at the transmit
  epochs and ≤ 6.9e-5 m at the receive epochs.
- **The implied offset** (asymmetry / 2Tc) correlates with the observed mutant offset at **r = −0.999997**. The sign
  difference is the LOS convention. In the canonical smoke cell the two agree to 5e-8 m/s: 1.66205e-3 vs 1.66210e-3.
- **Why HARDEN missed it.** `phase17_harden_events_and_derivatives.py` differenced only the symmetric receive
  endpoints, and only the x component. That gave the superseded ~1.5e-6 m/s.

## 3. Repair

`make_exact_counted_doppler_station_state_provider` now evaluates the Earth with `_interp_state`. This is the same
cubic-Hermite position/velocity interpolation that M3 two-way range and ΔDOR already use.

- **Unchanged:** the event model, the exact `sxform` site transform, frequencies and turnaround ratio, count-interval
  semantics, units and the R3/R4 model version strings.
- **F06 fail-closed messages** still name the non-finite history (position or velocity) from the bracketing nodes.
- **Provenance:** `earth_ephemeris_method` changes `linear_grid_interpolation` → `cubic_hermite_grid_interpolation`.
- **R4 four-event** inherits the repair through the same provider. It still reproduces R3 bitwise at zero delay.
- **Untouched:**
  - the opt-in legacy interpolated-transform-grid mode;
  - the byte-frozen R2 reference module;
  - the protected `examples/r2_measurement_fidelity_validation.py`.

### Test provenance updates (deliberate, not loosened)

- **R3-P04/P05/P11.** These tests compare event *arithmetic* with the station-state source held fixed (their own
  Addendum-07 note). They now inject model S's station states into production explicitly
  (`_r3_shared_production_provider`). Tolerances are unchanged.
- **R3-P22 label and behaviour tests.** These are re-pointed to the repaired policy. The behaviour test now uses an
  accelerating Earth, where Hermite is exact and linear is off by a·h²/8.
- **The `"linear_grid_interpolation"` label assertions** in four test files now expect the new label, each with a
  supersession comment.

## 4. Oracle

- **Primary (frozen).** The frozen R2 **model S** single-bounce solver, which is an independent implementation, on
  every row. The M3 **four-event model F** runs on every 10th row.
- **Its station input.** Both are fed station states evaluated **directly from SPICE** at every epoch the solver
  requests: `spkezr(EARTH)` plus `sxform⁻¹·[r_ECEF; 0]`.
- **Configuration.** The oracle uses production's light-time tolerances (`CountedDopplerReferenceConfig.from_legacy`),
  the R2 convention.
  - A first run with the reference module's default 1e-12 s update tolerance raised a convergence error at
    t3 = 72690 s.
  - That tolerance is below ulp(t) = 1.46e-11 s for t > 65536 s. This was a harness error.
  - The rerun had **0** convergence failures, which is now a strict gate condition.
- **Population.** CD1/KP1: 5 orbits × 24 D3 epochs × {high fidelity, simplified}, every ALL_VISIBLE Tc 60 s row,
  239,265 rows in total. The worst cell is O5 k = 13, high fidelity.

## 5. Floors Found, and the One Failing Criterion

| source | size | whose |
|---|---|---|
| frozen oracle evaluates SPICE at fl(et0 + t) | Earth ≤ 6.4e-5 m → value rms ~1.6e-8 m/s | **oracle** |
| model S default tolerance below ulp(t) late in the arc | convergence errors; 2.4e-7 m/s on one row | **oracle config** (fixed by `from_legacy`) |
| production `sxform` at fl(et0 + t) (new ledger item **HARDEN-D24**) | value rms ≤ 1.0e-7 m/s, max 3.6e-7 | **production, pre-existing**, shared with range and ΔDOR, outside D23 |
| production cubic-Hermite Earth | ≤ 2.42e-7 m (all epochs, exact ET) | production, the repaired term |

- **Post-hoc ET-exact oracle (labelled, not a frozen gate).** The ET rounding remainder is recovered by an error-free
  two-sum and applied with the SPICE velocity. The site is treated as production treats it, which isolates the Earth
  policy. Over all 239,265 rows production agrees to **max 5.4e-9 m/s, rms 1.2e-9**, which is the a priori floor.
- **The a priori derivation in the acceptance file omitted the oracle's own time-rounding term.** The value gates still
  passed. The epoch-contract diagnostic did not, for that reason alone.

## 6. CD1 and KP1

- **CD1** remains authoritative and unchanged:
  - `PRIMARY_CLASS = COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE`;
  - POP(D0) PASS, POP(D1 FREE) FAIL, ABS FAIL;
  - every f_perp within 4.0e-10, the same as the closure's SPICE-Earth result;
  - 0 class flips;
  - FD gates pass on the repaired path.
- **KP1** remains authoritative and unchanged: M1 PARTIAL, M2 NO_NEW_INFORMATION, M3 DISTINGUISHABLE. Every principal
  sine is within 1.0e-9, with 0 per-cell flips.

## 7. Final Status

```text
PHASE17_HARDEN_D23_REPAIR_GATE = FAIL on one frozen criterion (EARTH_EPOCH_CONTRACT diagnostic measured the oracle's
                                 ET rounding, 6.39e-5 m <= |v_E| ulp/2); PASS on all eight others. The same 1e-6 m
                                 criterion at exactly representable epochs: 2.42e-7 m. Becomes PASS on owner
                                 acceptance of this adjudication (as for CD1's numerical-floor exception).

COUNTED_DOPPLER_VALUE_PATH = QUALIFIED (all frozen value gates PASS)
EARTH_EPOCH_CONTRACT = QUALIFIED by the exact-epoch scan (2.42e-7 m); frozen diagnostic FAIL adjudicated (§5)
ASYMMETRIC_UPLINK_TRANSMIT_EPOCHS = COVERED (mechanism measured and pinned; permanent regression)
DIRECT_SPICE_VALUE_ORACLE = PASS (max 7.64e-8, rms 1.58e-8 m/s; model F 7.04e-8)
HISTORICAL_D23_MUTANT_REGRESSION = PASS (population 1.83e-3 m/s reproduced; unit mutant detected)
CD1_INFORMATION_STATUS = REQUALIFIED / UNCHANGED
KP1_INFORMATION_STATUS = REQUALIFIED / UNCHANGED
D23_DOCUMENTATION_CONSISTENCY = CLOSED (ledger regenerated: D23 FIXED, 1.5e-6 kept as SUPERSEDED; D24 added)

COUNTED_DOPPLER_VALUES_SAFE_FOR_NEXT_ESTIMATION_PHASE = YES, at the 1e-7 m/s level (residual D24 floor
                                                        <= 1e-7 rms, 20x below the optimistic noise floor)

PRODUCTION_CODE_CHANGED = YES (one provider function in lunar_od/radiometrics.py; one docstring)
MEASUREMENT_MODEL_CHANGED = YES (Earth interpolation inside the counted-Doppler station provider only)
PHYSICAL_MODEL_CHANGED = NO
ESTIMATOR_CHANGED = NO
K_PARAMETERIZATION_CHANGED = NO

TOTAL_TESTS_COLLECTED = 1489
TESTS_PASSED = 1460
TESTS_FAILED = 0
TESTS_ERRORS = 0
TESTS_SKIPPED = 29 (the same opt-in/data-dependent set as at entry)

PUSH = NONE
MERGE = NONE
MAIN_CHANGED = NO
ORIGIN_MAIN_CHANGED = NO

OWNER_ADJUDICATION (2026-10-07) = ACCEPTED -> PHASE17_HARDEN_D23_REPAIR_GATE = PASS

NEXT_ACTION = hard stop. Owner: accept or reject the §5 adjudication. A value-based phase (nonlinear / M3 / stochastic
              K) may then proceed; HARDEN-D24 is optional below 1e-7 m/s.
```

## 8. Reproduction

- **Full run:** `python examples/phase17_d23_qualification.py --reuse-fd`, with
  `PYTHONPATH="<worktree>;<campaign>/06_scripts"`. It takes about 75 min on 8 workers.
  - The 18.8 MB per-row raw output is not committed.
  - `artifacts/phase17_d23_cells.json` keeps per-cell counts and sums of squares, which reproduce the pooled RMS
    exactly, plus maxima, quantiles and compact epoch diagnostics.
- **Post-hoc scans:** `--epoch-scan` and `--et-exact-diagnostic`.
