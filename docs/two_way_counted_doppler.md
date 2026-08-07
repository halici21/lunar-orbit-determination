# Two-Way Counted Doppler Model

This note documents the current two-way counted Doppler implementation used by
the Lunar OD Python port. The model is intentionally simplified but has a
consistent analytic Jacobian for BLS/SRIF estimation and observability analysis.

## Observable

The range-rate measurement can use either the legacy instantaneous geometric
model or a simplified two-way counted Doppler model.

For the two-way model, the receive midpoint is split into a count interval:

```text
t_start = t_mid - Tc / 2
t_end   = t_mid + Tc / 2
```

At each endpoint, the code solves a round-trip light-time path:

```text
station transmit time t1 -> spacecraft reflection time t2 -> station receive time t3
tau = t3 - t1
```

The m/s-equivalent counted Doppler observable is:

```text
y = c / (2 Tc) * (tau_end - tau_start)
```

If configured for hertz output, the scale is:

```text
y_hz = turnaround_ratio * uplink_frequency_hz / Tc * (tau_end - tau_start)
```

## Analytic Partial

The BLS/SRIF Jacobian is formed with respect to the arc initial state:

```text
H_y = scale * (d tau_end / dx0 - d tau_start / dx0)
```

The round-trip light-time partial is obtained by differentiating the two
implicit light-time equations for `t1` and `t2`. The implementation follows the
same assumptions as the observable:

```text
no media correction
optional station clock offset/drift (default off)
zero transponder delay (nonzero values are rejected; see below)
no relativistic correction
constant uplink frequency and turnaround ratio
spacecraft state/STM cubic-Hermite interpolation inside closed support
Earth state and 6x6 transform linear interpolation inside closed support
```

Single-bounce note: the legacy solver keeps **one** spacecraft bounce state,
so `r_sc(t2)` is used for both the downlink and uplink legs. P0A rejects every
nonzero `transponder_delay_s` for this profile; representing delay requires
separate `t2u`/`t2d` spacecraft states and remains future four-event work. The
M3 two-way range model (`docs/two_way_range.md`,
`lunar_od/two_way_range.py`) already evaluates those separate events and its
nonzero-delay support is unchanged. At zero delay the two solvers agree to the
measured transform-interpolation bound (4.9e-10 s round-trip light time on a
10 s pass grid).

The state transition matrix stored in the augmented propagation history maps the
spacecraft state at the reflection epoch back to the arc initial state.

## Light-Time Event Derivation

For a receive epoch `t3`, the solved event times satisfy:

```text
F_d(t2, x0) = t3 - t2 - ||r2(t2, x0) - g3(t3)|| / c = 0
F_u(t1, t2, x0) = t2 - t1 - ||r2(t2, x0) - g1(t1)|| / c = 0
```

where:

```text
r2 = spacecraft position at reflection time
g1 = station position at transmit time
g3 = station position at receive time
c  = speed of light
```

The receive time is the independent measurement epoch, so `dt3/dx0 = 0`. Let:

```text
u_d = (r2 - g3) / ||r2 - g3||
u_u = (r2 - g1) / ||r2 - g1||
A2  = dr2/dx0
v2  = dr2/dt2
vg1 = dg1/dt1
```

Differentiating the downlink equation gives:

```text
dt2/dx0 = - u_d^T A2 / (c + u_d^T v2)
```

Differentiating the uplink equation gives:

```text
dt1/dx0 =
  [u_u^T A2 / c - (1 - u_u^T v2 / c) dt2/dx0]
  / [-1 + u_u^T vg1 / c]
```

The round-trip light-time is:

```text
tau = t3 - t1
```

so:

```text
d tau / dx0 = - dt1/dx0
```

For counted Doppler, this partial is evaluated twice, at the start and end of
the count interval:

```text
H_two_way = c / (2 Tc) * [d tau(t_end)/dx0 - d tau(t_start)/dx0]
```

The implementation uses the propagated STM history to construct `A2`.
Spacecraft state and STM use the same cubic-Hermite event interpolation used by
the nominal counted path. Earth position/velocity and J2000-to-ITRF93 state
transforms remain linearly interpolated on the pass grid; the station-state
time slope is differentiated self-consistently from that interpolant. These
in-support interpolation choices are unchanged by P0B-2.

This is the same chain-rule structure used in high-fidelity deep-space OD
software, but with the correction terms listed below intentionally omitted.

## Convergence and History-Domain Safety

P0B-1 requires each count endpoint's round-trip solve to pass both its
fixed-point update tolerance and independently re-evaluated final equation
residual. `count-start` and `count-end` failures are reported separately, and
uplink/downlink diagnostics identify the failing leg in seconds and equivalent
metres. Raw solvers may still return diagnostic nonconverged solution objects;
the observable and Jacobian consumers reject them.

P0B-2 resolves FA-03B for the counted production paths. Every event lookup is
checked against closed history support before interpolation:

- spacecraft state at every downlink solver probe and final event;
- spacecraft STM at the Jacobian reflection event;
- Earth position, Earth velocity, and the 6x6 transform at station receive and
  transmit events, including slope evaluation;
- both `count-start` and `count-end`, with separate `uplink`/`downlink` context.

Exact support endpoints are accepted. A request at most two policy ULP outside
a bound is normalized to the endpoint sample, where one policy ULP is evaluated
at `S=max(1,abs(request),abs(bound))`. Three or more policy ULP outside support,
or any larger physical deficiency, raises `HistoryDomainError` before an
extrapolator is reached. The low-level interpolation helpers retain their
generic behavior; strictness is enforced at every legacy production boundary.

The counted UKF constructs a local physical interval around the measurement.
Because the observable clock-corrects both count endpoints before the
light-time solve, the interval anchors cover the clock-corrected count
endpoints as well (P0B-2E2): the lower anchor is the earlier of the raw and
clock-corrected count-start minus the light-time margin, and the upper anchor
is the later of the raw and clock-corrected count-end. Both clock-corrected
count endpoints are contained in the physical local history interval. A
corrected endpoint that expands the envelope outward is included as the
corresponding exact anchor node. An inward corrected endpoint is contained
but is not guaranteed to be an exact local-grid node. The correction uses the same
`_clock_corrected_receive_time` function the observable applies, so the
envelope demand and the event evaluation cannot diverge. Before local
spacecraft propagation or source resampling, the UKF independently preflights
Earth position, Earth velocity, and transform support over that envelope.
Accepted representation-only boundary offsets use endpoint samples for source
lookup without rewriting the physical local-time grid. Unsupported intervals
raise; the UKF does not convert this error into convergence failure or row
skipping.

Range-rate generation handles an unsupported visible candidate as one ordered
structured drop. With noise enabled, four draws are consumed beforehand in
range/range-rate/azimuth/elevation order so later surviving rows and the RNG
tail remain deterministic; noise-disabled generation consumes no draws.
Partial selected-family arcs continue. If every eligible arc in that family is
empty because of domain drops, scenario assembly raises before estimator entry.
No automatic pre-roll or post-roll is created.

## Verification

The analytic two-way initial-state Jacobian is checked against a central
finite-difference Jacobian that perturbs the initial state, repropagates the
full arc, and recomputes the nonlinear two-way observable.

Covered paths:

```text
two_way_counted_doppler + BLS-LM
two_way_counted_doppler + SRIF
two_way_counted_doppler observability
analytic H vs numerical H
closed-support boundary matrix and first-probe rejection
five guarded history names and count-start/count-end/leg diagnostics
UKF source-interval preflight before propagation/resampling
clock-corrected UKF local envelope (zero/positive/negative offset; source-boundary rejection)
candidate-drop ordering and seeded RNG preservation
family-local partial/all-empty scenario behavior and CSV aggregates
```

This makes the numerical Jacobian a reference test, not the production path.

## Extension Points

To move toward a higher-fidelity DSN model, add these terms as separate
observable corrections and partial blocks:

```text
station clock and frequency bias
troposphere and ionosphere media corrections
four-event counted transponder delay (`t2u`/`t2d`)
relativistic light-time terms
uplink frequency ramping
station location solve-for partials
```

Those corrections should be added only after each correction's computed value
and partial derivative can be tested independently against numerical
finite-difference references.

## R3 — exact event-epoch station transform

R3 replaces the station site-state source of the counted-Doppler path. The
observable equations, the light-time solver, the FA-03B history-domain policy,
the count-interval definition, the clock model and the zero-delay gate are all
unchanged; only the inputs to those equations change.

### The two strategies

`RangeRatePhysicsConfig.station_state_method` (and the matching
`ScenarioConfig.station_state_method`) selects one of:

| Value | Site transform | Status |
|---|---|---|
| `exact_event_epoch_sxform` | `spice.sxform("J2000", "ITRF93", et0_s + t_event)` evaluated at the true event epoch | **production default** |
| `legacy_interpolated_transform_grid` | element-wise linear interpolation of the pre-sampled transform grid | explicit compatibility mode |

Element-wise linear interpolation of a rotation matrix is not a rotation between
grid nodes. The resulting station-position error follows
`(omega*dt)^2 * R_earth / 8` — about 1526 m at a 600 s transform cadence, 15.3 m
at 60 s and 0.42 m at 10 s. Removing that term is the accepted R2 decision
`EXACT_STATION_TRANSFORM_UPGRADE_REQUIRED`.

In exact mode the analytic Jacobian also takes its uplink station velocity
`vg1` from the exact station state (`station_state[3:6]`) instead of the
interpolation slope. Legacy mode keeps the slope, preserving model L bit for
bit.

### The default change

A scenario file written before R3 has no `station_state_method` field. Such a
file now **automatically** receives `exact_event_epoch_sxform`, so its
counted-Doppler observable changes. This is deliberate — it is the intended
consequence of the accepted R2 decision — and it is never silent: every run
records the method it executed in its measurement metadata and in the scenario
CSV. Gate R3-P23 exists specifically to assert that the change is disclosed
rather than prevented.

To reproduce pre-R3 numbers, opt in explicitly:

```json
{
  "range_rate_physics": "two_way_counted_doppler",
  "station_state_method": "legacy_interpolated_transform_grid"
}
```

Selecting the legacy value emits a `DeprecationWarning` naming the accepted R2
decision. The legacy value is only meaningful together with
`range_rate_physics = "two_way_counted_doppler"`; any other combination is
rejected at configuration load.

### No fallback

If the exact evaluation cannot be performed — kernels missing, an unknown frame,
an epoch outside coverage, a non-finite or wrongly shaped transform, a
non-finite Earth history, or a missing/non-finite `et0_s` — the run **fails
closed** with `StationStateEvaluationError` or `ValueError`. There is no
automatic fallback to the interpolated grid under any condition; legacy is a
deliberate opt-in, never a degradation path.

### What R3 deliberately does not change

* the Earth ephemeris stays on **linear** grid interpolation, so the exact
  production path reproduces the accepted R2 model S exactly and the S−L
  difference isolates the site transform alone;
* the spacecraft state and STM stay on **cubic Hermite** interpolation;
* the event model stays **single-bounce**; a nonzero transponder delay is still
  rejected at configuration construction;
* M3 two-way range is untouched, and its provider is mirrored rather than
  imported or shared.

### Provenance

Each run records `station_state_method`, `exact_event_epoch_enabled`,
`legacy_compatibility_mode`, `earth_ephemeris_method`,
`spacecraft_state_interpolation_method`, the `J2000`/`ITRF93` frame pair and
`counted_doppler_model_version = r3.counted-doppler.exact-station.v1`. The
scenario CSV gains `station_state_method` and `counted_doppler_model_version` as
its final two columns, appended after the R1 numerical-provenance segment.

The pre-R3 metadata field `station_velocity_model` reported the literal
`"sxform"` for both paths, which overclaimed the interpolated route. It now
reports `exact_event_epoch_sxform` or `interpolated_sxform_grid` truthfully.
