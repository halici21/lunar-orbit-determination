# Numerical Tolerance Policy

How to select and justify a numerical tolerance. A tolerance is part of the
scientific claim, not an afterthought.

## For each comparison, state
- quantity being compared, and its unit
- absolute tolerance and/or relative tolerance
- expected floating-point scale of the quantity
- integration tolerance (if propagation is involved)
- frame / time-system sensitivity
- stochastic vs deterministic behavior
- whether an observed difference is numerical, physical, or algorithmic

## Quantity categories (pick representative tolerances per category)
position [m] · velocity [m/s] · acceleration [m/s^2] · range [m] ·
range-rate [m/s] · Doppler / frequency-like observables · angles [rad] ·
residuals · covariance entries · NIS / NEES · condition numbers · runtime.

## Rules
- Do not loosen tolerances just to make a test pass.
- Prefer physically meaningful tolerances over arbitrary constants.
- Use **relative** tolerance when scale varies significantly.
- Use **absolute** tolerance when the expected magnitude can approach zero.
- Document why the tolerance is acceptable.
- Distinguish floating-point round-off from model differences.

## Critical rule
A tolerance is part of the scientific claim — choose and justify it accordingly.
