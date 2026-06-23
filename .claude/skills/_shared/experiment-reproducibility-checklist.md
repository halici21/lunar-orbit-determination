# Experiment Reproducibility Checklist

Each experiment should record:
- scenario config
- random seed
- station network
- force model
- measurement type
- measurement noise
- visibility rules
- estimator settings
- initial state error
- initial covariance
- output artifacts (path under `results/`)
- producing script (path under `examples/`)
- commit hash, if known

Goal: anyone can reproduce the result along
`config -> scenario -> measurements -> estimation -> artifact`.
Mark any unknown field explicitly rather than guessing.
