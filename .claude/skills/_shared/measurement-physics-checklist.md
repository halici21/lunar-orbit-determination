# Measurement Physics Checklist

Use when modifying or reviewing measurement models
(`lunar_od/measurements.py`, `lunar_od/radiometrics.py`).

Verify:
- units
- sign conventions
- frame consistency (MCI / ECEF / SEZ / J2000 / ITRF93)
- station position epoch (receive time)
- spacecraft state epoch (transmit time, for light-time)
- light-time convention (reception vs transmission; one-iteration vs converged)
- station motion during the light time
- stellar aberration (observer-velocity frame: MCI vs SSB)
- lunar occultation logic
- elevation-mask logic
- instantaneous range-rate vs two-way counted Doppler distinction
- residual consistency (generation vs prediction h(x))
- Jacobian consistency (analytic vs the model actually used)
- finite-difference agreement where possible

Critical rule: measurement generation, predicted h(x), residual computation, and
the Jacobian must use the same physical assumptions.
