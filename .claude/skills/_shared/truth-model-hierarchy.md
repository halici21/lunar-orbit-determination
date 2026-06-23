# Truth-Model Hierarchy

Always state which level a reported accuracy result uses.

## Level 0 - Same-model truth
Truth and estimator use the same propagator and force model. For algorithm
debugging, estimator-consistency, measurement-model, and controlled regression
tests. Risk: may overstate real-world performance.

## Level 1 - Higher-fidelity internal truth
Truth uses a richer internal model than the estimator (extra perturbations,
tighter tolerances, richer force model). For model-mismatch and robustness studies.

## Level 2 - SPICE-driven truth
Truth relies on SPICE ephemerides / kernels / SPICE-derived states. For
lunar / planetary geometry validation, ephemeris-dependent studies, frame and
timing checks.

## Level 3 - External reference truth
Truth or reference results come from external flight-dynamics tools (GMAT,
Orekit, Tudat, MONTE, Basilisk, or other validated software).

## Rules
- Never report "accuracy" without stating what it is accurate against.
- Do not call a result flight-validated unless an appropriate external /
  operational truth source is used. No external (Level 3) validation has been
  performed in this repo unless an artifact proves otherwise.
