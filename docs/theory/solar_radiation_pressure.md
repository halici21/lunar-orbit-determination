# Solar radiation pressure

Cannonball SRP with a conical lunar shadow, in `lunar_od/srp.py`. Opt-in and
default-off: a propagation that does not pass `srp=` runs exactly the force
model it ran before this existed.

## Enabling it

```python
from lunar_od.srp import SRPOptions
from lunar_od.dynamics import propagate_state

states = propagate_state(
    t_eval_s, state0, mu_moon, mu_earth, mu_sun, get_earth_pos, get_sun_pos,
    srp=SRPOptions(
        k_srp_m2_per_kg=0.01,
        k_srp_source="CAMPAIGN_PARAMETRIC",
        provenance="screening value for a materiality sweep; not a spacecraft property",
    ),
)
```

`srp=None` (the default) disables it. `SRPOptions(enabled=False)` also disables
it and needs no coefficient.

## What `K_SRP` is

The cannonball coefficient

```
K_SRP = C_R * A/m        [m^2/kg]
```

Only the **product** is required, and only the product is physically
identifiable from tracking data when the illuminated area is unknown — which is
the usual case. If you happen to know both factors separately:

```python
SRPOptions.from_cr_and_area_to_mass(c_r=1.5, area_to_mass_m2_per_kg=0.02)
```

which multiplies them and records in `provenance` that the coefficient was
composed rather than supplied whole.

## Why the coefficient is always yours to supply

There is no default, and enabling SRP without a coefficient raises
`SRPConfigurationError`.

The project's frozen reference spacecraft, `LTB-IRIS-DSN34X-v1`, reports

```
srp.K_SRP   status = UNKNOWN   value = None
```

because neither the reflectivity nor the illuminated area of that vehicle is
public. `config.require("srp.K_SRP")` raises, and no helper in `lunar_od.srp`
bypasses that — there is deliberately no `SRPOptions.from_reference_configuration`.

So a missing coefficient fails loudly. It does not become 0.01, or a
screening-envelope endpoint, or a midpoint, or a silent disable.

## The screening envelope is not a bound

The Phase 15 configuration carries

```
K_SRP_body_only_screening_envelope = [0.008640, 0.048359] m^2/kg
semantics                          = SCREENING_ENVELOPE
excludes                           = solar array area, attitude-dependent
                                     projected area, deployed appendages
```

Its area factor comes from the stowed body envelope alone. The true coefficient
can lie **above** the upper endpoint, so the interval bounds nothing and is not
a probability distribution.

It is therefore not used as a validator. `SRPOptions(k_srp_m2_per_kg=0.06)` is
accepted even though 0.06 exceeds the upper endpoint; rejecting it would enforce
a limit the evidence does not support. What *is* rejected: negative, NaN,
infinite, and non-numeric coefficients.

Reading an endpoint for a sensitivity sweep is fine, but say so:

```python
SRPOptions(k_srp_m2_per_kg=envelope.upper,
           k_srp_source="SCREENING_ENVELOPE_ENDPOINT")
```

## The force

```
a_SRP = P(d) * K_SRP * nu * u,    u = (r_sc - r_sun) / |r_sc - r_sun|
P(d)  = P_1AU * (AU / d)^2
```

`u` points from the Sun to the spacecraft, so the force is anti-solar. `P_1AU`,
`AU_M` and `R_SUN_M` live in `lunar_od/constants.py` with their sources.

The true instantaneous Sun distance is used. Holding it at one astronomical unit
biases the magnitude by about 2.1% on the project's reference arc, so there is
no fixed-1-AU path.

The Sun position is the same `get_sun_pos` the third-body term already uses — no
second ephemeris, frame or origin convention enters. When SRP is active the two
share one lookup per epoch, so enabling it does not double the ephemeris cost.

## Shadow

`shadow_model="CONICAL_PENUMBRA"` (the default) treats the Moon and the Sun as
disks on the sky and returns the visible solar fraction `nu` in `[0, 1]`,
covering full light, penumbra, umbra and the annular case. It is continuous
through the penumbra: its sampled increment shrinks with the sampling step,
where a binary model steps by a full unit however finely it is sampled. That
continuity, not state-level accuracy, is why production uses it — a binary
illumination discontinuity is what a variational or numerical path must not meet.

A spacecraft on the sunward side of the Moon's centre skips the angular geometry
entirely. That is exact, not an approximation: the Moon's shadow lies wholly
anti-sunward of it.

`shadow_model="NO_SHADOW"` exists for oracle fixtures and sensitivity work.
There is no binary cylindrical option.

**Earth shadow is not modelled.** Phase 13 measured a 118.5 deg margin between
the Earth and Sun disks over the reference arc, so it is not material there.
Supporting a second occultor is a generalisation, not an omission to fix in
passing.

## STM policy

The trajectory gets `a_SRP`. The state transition matrix does **not** get its
position gradient.

That is an explicit approximation, not an exact result. Phase 13 measured

```
|d a_SRP / dr| / |d a_grav / dr| = 4.02e-13
```

on the reference arc — gradient against gradient, not against raw gravitational
acceleration — so the omitted term is negligible there and nowhere else is
claimed.

Two consequences worth stating:

- The SRP gradient is omitted **entirely**, including its shadow term. There is
  no partial gradient that treats the geometry but drops `d nu / dx`, which
  would be inconsistent — Phase 13 showed that dropping only the illumination
  derivative produces 100% relative error in the shadow-gradient contribution.
- An estimator that later solves for `K_SRP` needs its own sensitivity path.
  This one will not supply it.

## What this does not do

No `K_SRP` estimation, no solve-for, no estimator-state augmentation, no
stochastic or time-varying coefficient, no process noise, no prior. `K_SRP` is
constant over a propagation.

Having the capability to propagate SRP is not the same as knowing the real
spacecraft's coefficient. `LTB-IRIS-DSN34X-v1` is a public-sourced reference
configuration, not mission truth, and it still truthfully reports that no point
`K_SRP` is known.
