# Terminology

- **BLS-LM** — Batch Least-Squares with Levenberg-Marquardt damping (completed-arc batch estimator).
- **SRIF** — Square-Root Information Filter (information-form estimation).
- **SR-UKF** — Square-Root Unscented Kalman Filter (sequential estimator).
- **STM** — State Transition Matrix (sensitivity of state to the initial state).
- **NIS** — Normalized Innovation Squared (per-measurement innovation consistency).
- **NEES** — Normalized Estimation Error Squared (state-error consistency; needs truth + covariance).
- **arc** — a continuous or stitched interval of available tracking measurements.
- **visibility window** — interval where a station satisfies elevation / occultation constraints.
- **lunar occultation** — the Moon blocks the station-to-spacecraft line of sight.
- **residual** — observed minus computed (O-C) measurement.
- **covariance** — state-estimate uncertainty matrix.
- **condition number** — numerical indicator of how ill-conditioned the estimation system is.
- **truth trajectory** — reference trajectory used to generate synthetic measurements and to score accuracy.
- **synthetic measurement** — measurement generated from the truth trajectory plus a noise / bias model.
- **two-way counted Doppler** — averaged differenced round-trip range over a count interval (operational-style observable).
- **light-time correction** — accounting for finite signal travel time (transmit vs receive epochs).
- **apparent geometry** — line of sight after light-time (and optional stellar aberration) correction.
