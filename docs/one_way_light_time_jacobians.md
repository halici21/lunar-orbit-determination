# One-Way Light-Time Measurement Jacobians

## Scope

The opt-in `implicit_light_time` Jacobian supports receive-time-tagged one-way
range, azimuth, and elevation measurements. The station and J2000-to-ITRF93
transformation are evaluated at receive time. The spacecraft state and STM are
evaluated at the converged transmit time. Position is in metres, velocity in
metres per second, light time in seconds, and angles in radians.

The default geometric and `analytic_first_order_light_time` paths are unchanged.

## Initial-State Sensitivity

For inertial line of sight `rho = r_sc(t_t) - r_station(t_r)`, with
`t_t = t_r - tau` and `tau = ||rho||/c`, the implementation evaluates

```text
d_tau_dx0 = (rho_hat.T @ Phi_r_tx) / (c + rho_hat.T @ v_sc_tx)
J_los_dx0 = Phi_r_tx - outer(v_sc_tx, d_tau_dx0)
J_unit_los_dx0 = (I - outer(rho_hat, rho_hat)) @ J_los_dx0 / range
```

`one_way_light_time_initial_state_sensitivity` exposes these quantities using
the explicit names `d_tau_dx0`, `phi_r_tx`, `j_los_dx0`, and
`j_unit_los_dx0`. The range row is `c*d_tau_dx0` and reuses the tested M2.1
range-sensitivity kernel.

## SEZ Angle Chain Rule

The repository uses South-East-Zenith coordinates:

```text
u_sez = C_sez_ecef @ C_ecef_mci(t_r) @ rho_hat
u_sez = [S, E, Z]
azimuth = atan2(E, -S)
elevation = atan2(Z, sqrt(S^2 + E^2))
```

The observable computes elevation with the equivalent `asin(Z)` expression for
a unit vector. The Jacobian uses the `atan2` form:

```text
d_az_dsez = [E/h^2, -S/h^2, 0]
d_el_dsez = [-Z*S/(h*u2), -Z*E/(h*u2), h/u2]
h = sqrt(S^2 + E^2), u2 = S^2 + E^2 + Z^2
```

The resulting rows have units of radians per initial-state unit. Angle residuals
remain wrapped with `atan2(sin(delta), cos(delta))`.

## Zenith Policy

Azimuth is physically undefined at zenith. The implicit angle path raises
`MeasurementJacobianError` when the horizontal unit-LOS norm is below `1e-6`.
This threshold is intentionally larger than machine epsilon: it limits the
`1/h` azimuth sensitivity amplification to approximately `1e6`. Tests verify a
finite Jacobian outside the threshold and deterministic rejection inside it.
Legacy geometric and first-order behavior is unchanged.

## Estimator Mapping

`one_way_light_time_position_initial_state_jacobian` returns one ordered `(3,6)`
block: range, azimuth, elevation. The block already contains the transmit-epoch
STM and must not be multiplied by another STM. BLS-LM, SRIF, posterior-information,
and observability paths use the shared augmented-history mapping helper.

## History-Domain Contract (P0B-2)

FA-03B was confirmed at the D1 baseline: production one-way CN/CN+S solver
probes and final state/STM lookups could silently extrapolate beyond the
propagated history. P0B-2 removes that behavior from every production one-way
consumer while leaving the generic raw interpolation helper available for its
separate diagnostic/M3 uses.

For a state/STM history with support `D=[t_start,t_end]`:

```text
S = max(1, abs(q), abs(bound))
policy_ulp = nextafter(S, +infinity) - S
```

Exact endpoints are accepted. A request at most two `policy_ulp` outside a
bound is normalized to that bound and evaluated as an endpoint sample; this is
floating-point representation handling, not extrapolation. A request three or
more `policy_ulp` outside support raises `HistoryDomainError` before model
evaluation. The first unsupported intermediate solver probe fails immediately;
a later hypothetical in-domain root does not validate an iteration that has
already used unsupported history.

The guard covers `_apparent_position_observable`, one-way range sensitivity,
the final transmit-state re-query, and the transmit-epoch STM lookup. Nominal,
sensitivity, and Jacobian paths therefore share both the P0B-1 convergence
policy and the P0B-2 history-domain policy. The receive-epoch Earth state,
station state, and frame transform remain node-indexed inputs as described in
the frame contract.

During position generation, each visible candidate owns one RNG slot. With
noise enabled, its range/azimuth/elevation draws are consumed before model
evaluation; an unsupported candidate is omitted from both clean and noisy
arrays and produces an ordered `HistoryDomainDropRecord`. This keeps later
surviving seeded rows bit-identical to a pre-rolled supported run. With noise
disabled, no RNG draws occur. Metadata records the drop count and ordered
records plus the maximum actual `required_pre_roll_s` and
`required_post_roll_s`; P0B-2 does not extend propagation automatically.

At scenario assembly, partial drops in the selected measurement family leave
surviving arcs estimable. If every eligible arc in that family is empty because
of history-domain drops, `build_measurement_arcs` raises an aggregate
`HistoryDomainError` before any estimator call. This policy is family-local;
the repository has no combined position/range-rate scenario coordinator.

## Stellar Aberration Apparent-LOS Chain

For CN+S, M2.3 preserves the production reception-case Newtonian aberration
mapping and differentiates only that three-dimensional algebraic transform. If
`u_cn` is the M2.2 unit LOS and `v_obs` is the receive-epoch observer velocity
in J2000 axes,

```text
u_app = f_stellar(u_cn, v_obs)
J_app_x0 = J_stellar_tangent @ J_unit_los_dx0
```

The observer velocity is Moon-relative for `local_mci` and SSB-relative for
`spice_ssb`; both include station velocity from Earth rotation. The observer is
not part of the spacecraft solve-for state, so `d(v_obs)/d(x0) = 0`. Station,
clock, or Earth-orientation solve-for extensions must revisit that assumption.

The local derivative uses two deterministic orthonormal tangent directions
`B=[b1,b2]`. For each direction,

```text
u_plus  = normalize(u_cn + h*b_i)
u_minus = normalize(u_cn - h*b_i)
d_i = (f_stellar(u_plus) - f_stellar(u_minus)) / (2*h)
J_stellar_tangent = [d1,d2] @ B.T
```

The internal step is `h=1e-5`. A sweep from `1e-3` to `1e-8` found the stable
region around `3e-5` to `3e-6`: larger steps show truncation error and steps
below about `1e-7` increasingly expose floating-point roundoff. Direct tangent
perturbations of `spice.stelab` give a maximum action difference of approximately
`2.5e-16` in the controlled validation cases, including exact and near
parallel/anti-parallel geometry.

Aberration changes direction only. The first row of the final `(3,6)` block
remains the M2.1 implicit range row; only the azimuth/elevation direction chain
uses the local derivative. The near-zenith policy is evaluated on the apparent
SEZ LOS after aberration.

Metadata identifies the method as
`aberration_jacobian_model=local_central_finite_difference` and
`angle_jacobian_model=hybrid_apparent_chain_rule`. The complete method is not
fully analytic: it combines analytic implicit CN sensitivity, a local numerical
aberration derivative, and the analytic receive-frame/SEZ angle chain.

## M2.3 Numerical Evidence

For the controlled six-state full-chain finite-difference case used by the
regression tests:

| metric | result |
|---|---:|
| apparent-LOS Jacobian absolute Frobenius difference | `1.674e-13` |
| apparent-LOS Jacobian scaled Frobenius difference | `5.413e-8` |
| apparent-LOS maximum absolute component difference | `1.065e-13` |
| azimuth row relative / maximum absolute difference | `8.214e-9` / `4.085e-14` |
| elevation row relative / maximum absolute difference | `6.171e-8` / `1.114e-13` |
| CN vs CN+S implicit range-row difference | exactly `0` |
| apparent unit-LOS norm error | `0` |
| apparent-Jacobian tangency error | `1.311e-17` |

The raw Frobenius value combines the repository's mixed position/velocity state
columns and is therefore accompanied by the scaled norm and maximum component;
it is not used as a standalone physical tolerance. The largest range-row FD
difference is `4.499e-6 s` in initial velocity column `v_z`, whose range
derivative has units `m/(m/s) = s`; that column used a `1e-3 m/s` central
perturbation. Position columns use `0.1 m` perturbations and dimensionless
range derivatives.

Two validation levels are kept separate. The local aberration derivative is
checked against direct tangent perturbations of `spice.stelab`, independent of
the production aberration function. The realistic fixture check applies
`stelab` to the internal CN LOS and receive-epoch observer velocity; because the
synthetic orbiter is not an SPK body, that fixture does not independently solve
the complete light-time problem in SPICE.

The legacy first-order mode remains unchanged and continues to omit the
aberration derivative. Other exclusions are a fully analytic aberration
derivative, media corrections, clock and station-coordinate solve-for states,
Earth-orientation sensitivities, and two-way observables. A profile-aware UKF
CN/CN+S operator is still not implemented; P0A now rejects those UKF
configurations instead of silently using geometric measurement physics.
