# Test Oracle Patterns

Acceptable oracles (the source of the "expected" value) for Lunar OD tests.

| Oracle | When appropriate | Strengths | Limitations / common mistakes |
|---|---|---|---|
| closed-form analytical | a formula exists | strong, exact | algebra error; only simple cases |
| conservation / invariant | energy, momentum, norm preserved | model-agnostic | weak (passes many wrong models) |
| finite-difference | derivatives (Jacobian/STM) | direct, local | step choice; noise near zero |
| slow-path vs fast-path | optimized vs reference | catches accel bugs | both can share a bug |
| known geometry case | special geometry (e.g. zero range-rate) | intuitive | narrow coverage |
| MATLAB fixture | legacy parity | cross-impl check | drift; units; MATLAB indexing |
| stored CSV/JSON baseline | regression | cheap, repeatable | silent drift if updated blindly |
| SPICE-derived reference | geometry / ephemeris / frames | authoritative for geometry | kernel availability; abcorr mismatch |
| external tool reference | GMAT/Orekit/Tudat/MONTE/Basilisk | independent | assumptions must be aligned first |
| statistical expectation | NIS/NEES/Monte Carlo | proper for stochastic | needs enough trials + seed |
| schema / format expectation | artifact contracts (CSV/PNG/JSON) | guards output shape | does not check values |
| UI smoke expectation | desktop page builds / no exception | catches crashes | not a correctness oracle |

## Critical rule
Every test must have a clear oracle. A test without a meaningful oracle only
proves that the code executed.
