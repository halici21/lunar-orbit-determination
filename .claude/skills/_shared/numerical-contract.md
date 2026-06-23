# Numerical Contract

Rules that protect numerical integrity and fair comparison. Apply whenever code
or models that affect numbers change, and whenever estimators are compared.

## Fair-comparison invariants
Hold these **identical** across compared runs unless that variable is exactly the
one under study:
- truth trajectory and dynamics / force model
- measurement realization
- visibility schedule and station network
- noise realization (unless explicitly testing stochastic robustness)
- initial state error
- covariance assumptions (where applicable)
- random seed when comparing algorithms

## Tolerances & claims
- Do **not** weaken test tolerances without a documented scientific justification.
- Do **not** claim an improvement if any invariant above changed.
- A passing test suite does **not** prove scientific equivalence — compare against
  a registered baseline when numerical behavior may have changed
  (see `baseline-registry.md`).
- Every accuracy claim must state the truth-model level used
  (see `truth-model-hierarchy.md`).
- Do not present a result as strong if it rests on a single run, unknown config,
  unknown seed, or untraceable artifact.
