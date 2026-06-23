# Estimator Comparison Contract

Checklist for a fair BLS-LM / SRIF / SR-UKF comparison. Each item must be either
identical across the compared estimators, or be the single explicitly controlled
variable.

## Must be aligned
- truth trajectory
- measurement schedule (epochs, cadence)
- station network
- visibility windows
- measurement noise realization
- random seed
- initial state error
- initial covariance
- estimator start mode (cold / hot / formal / sqrt_formal)
- force model
- measurement model (incl. light-time / Doppler settings)
- arc segmentation
- success / failure criteria

## Critical rule
Do not compare estimators unless these assumptions are explicitly aligned or the
difference is the controlled variable. Always state which variable (if any) differs.
