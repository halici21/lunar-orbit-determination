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
no station clock error
no transponder delay
no relativistic correction
constant uplink frequency and turnaround ratio
linear interpolation over the propagated state history
```

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

The implementation uses the propagated STM history to construct `A2`. Because
the measurement model interpolates the propagated state history linearly, the
analytic partial uses the same piecewise-linear interpolation for the event
state, station state, and STM.

This is the same chain-rule structure used in high-fidelity deep-space OD
software, but with the correction terms listed below intentionally omitted.

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
```

This makes the numerical Jacobian a reference test, not the production path.

## Extension Points

To move toward a higher-fidelity DSN model, add these terms as separate
observable corrections and partial blocks:

```text
station clock and frequency bias
troposphere and ionosphere media corrections
transponder delay
relativistic light-time terms
uplink frequency ramping
station location solve-for partials
```

Those corrections should be added only after each correction's computed value
and partial derivative can be tested independently against numerical
finite-difference references.
