# R1–R4 deterministic long-arc and geometry qualification (System Phase 1)

This campaign answers a different question from the R1–R4 unit and regression
suites. Those established *local* correctness. This one asks whether the
accepted measurement and estimation stack stays physically and numerically
correct when it is exercised repeatedly over hours to days and across materially
different tracking geometries.

It adds **no production physics** and changes **no accepted threshold**. Every
criterion it applies is inherited; the traceability table lives in
`r1_r4_authority_matrix.csv` in the campaign artifact bundle.

## What is deterministic about it

No Monte Carlo, no random measurement noise, no random process noise, no random
initial state, no randomized outliers. Measurement generation runs with
`noise=False`. Statistical testing belongs to a later phase.

## Geometry

Geometry is assembled **only** from accepted repository fixtures:

* the single accepted lunar orbit definition, `r0 = 1737.4e3 + 100e3` m, as used
  throughout `tests/test_estimators.py`;
* the accepted `RANGE_RATE_STATION_DEFS` coordinates in `lunar_od/config.py`;
* different orbital phase along that same accepted orbit.

Eight accepted stations are selected to span the Earth-rotation velocity
projection — from Svalbard KGS at 78.23° (≈ 95 m/s) to Chuuk KGS at 7.25°
(≈ 461 m/s), a 4.86× spread — crossed with three orbital-phase offsets, giving
24 geometries.

**Coverage limitation.** The repository contains exactly one accepted lunar
orbit definition. Orbital-regime diversity (different altitude, eccentricity or
inclination) is therefore **not** available without inventing mission physics,
which this campaign deliberately does not do. Phase-1 geometry diversity comes
from station location and orbital phase only. This is recorded as
`GEOMETRY-COVERAGE-LIMITATION`.

## Horizons

| id | span | purpose |
|---|---|---|
| H1 | 1 h | full local matrix, ~½ orbit |
| H2 | 6 h | dense, ~3 orbits |
| H3 | 24 h | day, ~12 orbits |
| H4 | 72 h | multi-day, ~37 orbits |
| H5 | 7 d | week, ~86 orbits |

The orbital period of the accepted fixture is 7067.5 s (1.96 h).

## Count intervals, cadences and delays

The frozen matrix per horizon is encoded in `FROZEN_MATRIX` in the runner. The
delay sweep is `FROZEN_TRANSPONDER_DELAYS_S = (0, 1e-6, 1e-5, 1e-4, 1e-3) s`,
recovered from `examples/r2_measurement_fidelity_validation.py`. These are a
**hypothetical capability sweep**, not mission hardware values.

A `count_interval <= max(delay)` combination is structurally invalid and is
never padded in to populate a table.

## Epoch density

The full frozen matrix at full epoch density is ~3.5e7 counted-Doppler
observations. On the measured throughput of this environment (R4 ≈ 991 obs/s,
R3 ≈ 1228 obs/s) that is ≈ 9.7 hours for the core generation alone, before the
L/S/F, exact-versus-legacy, model-F, zero-delay-oracle, Jacobian and estimator
campaigns that multiply it.

The runner therefore accepts `--max-epochs-per-cell`, which uniformly subsamples
the measurement epochs **inside** each cell across that cell's horizon. No
geometry, count interval, cadence, delay or horizon is dropped. Any such
reduction must be declared in the campaign definition **before** execution, and
the full-density campaign is tracked separately as a cost decision.

## Running it

```bash
# plan only
python examples/r1_r4_long_arc_qualification.py --horizon H1 --shard 1/8 \
    --plan-only --output-dir results/phase1

# one shard
python examples/r1_r4_long_arc_qualification.py --horizon H1 --shard 3/8 \
    --max-epochs-per-cell 12 --output-dir results/phase1

# aggregate, rejecting missing/duplicate/mismatched shards
python examples/r1_r4_long_arc_qualification.py --aggregate \
    --output-dir results/phase1
```

## Sharding and resumability

The campaign is a deterministic ordered cell list; `--shard k/n` runs indices
`i` with `i % n == k - 1`. Each shard writes its own CSV plus a sidecar JSON
recording the campaign definition hash, the canonical HEAD, the cell and row
counts and a completion flag. An interrupted campaign resumes by re-running only
the missing shards.

Aggregation **refuses** to report a campaign as complete when a shard is
missing, duplicated, incomplete, or carries a different canonical HEAD or
campaign definition hash. That refusal is exercised by
`tests/test_r1_r4_long_arc_qualification.py`.

## Inherited criteria

| quantity | criterion | origin |
|---|---|---|
| per-leg light-time equation residual | ≤ 1e-11 s | M3 / R3 solver default |
| event ordering | `t1 < t2u <= t2d < t3` strict | R4 |
| zero-delay reduction | R4(δ=0) == R3 production, **bitwise** | R4-P08 |
| model-F parity | < 1e-9 relative | R4-P09 |
| migration classification | ≤ 0.1 σ / > 0.1 σ / > 0.5 σ | R2 decision contract, R4 Addendum 03B |
| Earth interpolation | `linear_grid_interpolation` | R3 |
| spacecraft interpolation | `cubic_hermite` | R3 |
| station transform | `exact_event_epoch_sxform` | R3 |

Quantities without an accepted threshold — conditioning trends, error growth,
runtime, information accumulation slope — are recorded and labelled
**DIAGNOSTIC**. A numerical cutoff is never manufactured after seeing a result.

`MIGRATION_BLOCKER_CLASS` remains a *migration* classification. It does not mean
the R4 implementation failed merely because explicitly-selected higher-fidelity
physics differs from R3. R3 remains the default, R4 remains explicit opt-in, and
GOV-01 remains binding.

## What Phase 1 does not establish

Not Monte Carlo validated. Not statistical-consistency validated. Not
fault-tolerance validated. Not soak validated. Those belong to later phases.
