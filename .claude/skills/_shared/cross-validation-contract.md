# Cross-Validation Contract

Rules for validating Lunar OD outputs against external software (GMAT, Orekit,
Tudat, MONTE, Basilisk, SPICE utilities).

> No external cross-validation has been performed in this repo yet. This file
> defines how it must be done if/when it is, and forbids claiming it was done.

## Applicable outputs
propagated states, accelerations, frame transformations, visibility windows,
measurement predictions, residuals, covariance propagation, estimator outputs,
generated figures / tables.

## Alignment (verify BEFORE comparing)
reference frame; epoch; time scale; gravitational constants; body radii / shape;
force-model terms; ephemeris sources; integration method & tolerance; central-body
convention; station coordinates; station frame; light-time convention; aberration
convention; measurement definitions; covariance convention; units & sign
conventions.

## Recommended comparison metrics
state difference norm; radial / along-track / cross-track difference; velocity and
acceleration difference; range / angle / Doppler prediction difference; residual
difference; covariance difference; runtime difference (only after physical
equivalence is established).

## Critical rule
Do not compare outputs from different tools until assumptions are explicitly
aligned. If they cannot be aligned, report the mismatch and treat the comparison
as qualitative, not definitive.
