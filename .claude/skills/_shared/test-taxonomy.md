# Test Taxonomy

Lunar OD test types and when to use each. Runtime tiers: **fast** (default CI),
**medium** (pre-push / PR), **slow** (nightly / manual / release).

| Test type | Purpose / when | Typical oracle | Tier | Common failure modes |
|---|---|---|---|---|
| unit | isolated deterministic logic | closed-form / known value | fast | over-mocking, trivial assertions |
| regression | protect established numerical behavior | stored baseline / fixture | fast–medium | silent baseline drift |
| finite-difference | Jacobians, STMs, sensitivities | numerical derivative of the model | fast–medium | step too large/small, wrap-around |
| analytical-vs-numerical | closed-form exists | analytic value | fast | algebra error in the oracle |
| fast-path vs slow-path | optimized (Numba) vs reference | reference implementation | fast–medium | hidden divergence at edges |
| fixture comparison | known external / legacy value | MATLAB / SPICE / stored CSV-JSON | fast–medium | stale or unitless fixture |
| scenario smoke | end-to-end workflow health | "runs + schema valid" | medium | mistaken for correctness proof |
| estimator fairness | BLS-LM / SRIF / SR-UKF comparison | aligned-assumption invariants | medium | unaligned truth/seed/noise |
| statistical consistency | NIS / NEES / Monte Carlo behavior | chi-square / distribution expectation | medium–slow | too few trials, no seed |
| artifact reproducibility | output (CSV/PNG) contract | stored schema / prior artifact | medium | environment-dependent output |
| desktop smoke | UI workflow health | "page builds, no exception" | medium | GUI timing, headless issues |
| public-repo install/import | package installs/imports cleanly | import success on clean env | medium | private paths, missing deps |
| external cross-validation | vs GMAT/Orekit/Tudat/MONTE/Basilisk | external tool reference | slow | unaligned assumptions |

## Critical rule
A smoke test proves a workflow runs; it does **not** prove scientific correctness.
