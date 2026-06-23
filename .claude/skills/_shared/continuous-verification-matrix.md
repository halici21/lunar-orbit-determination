# Continuous Verification Matrix

When each test group should run.

| Tier | Purpose | Allowed runtime | Typical tests | Excluded | Artifacts / kernels / GUI |
|---|---|---|---|---|---|
| every commit | catch obvious breakage | seconds | fast unit + deterministic regression | long campaigns, private artifacts | no kernels required; no GUI |
| pre-push | local gate before sharing | ~1-2 min | full fast pytest + targeted-area tests | slow campaigns | kernels optional (skip gracefully) |
| pull request | merge gate | ~few min | fast full regression + artifact schema | private paths, long campaigns | no private local paths |
| nightly | scientific regression | longer | reduced Monte Carlo, reduced long-arc, fragmented-visibility & Doppler smoke, selected baselines | full thesis campaigns | kernels may be required (documented) |
| manual scientific campaign | full studies | hours | full Monte Carlo, full 28-day, thesis figures, external-data workflows | — | kernels/data as needed; documented |
| release | publishable quality | medium | fast regression + registered baselines + public examples + packaging/import + README traceability | — | clean-env install; no private files |
| thesis freeze | final-result integrity | as needed | all thesis numbers traced; final figures regenerated/verified; configs + baseline status saved | — | final scenario configs saved |
| external validation | cross-tool checks | slow | GMAT/Orekit/Tudat/MONTE/Basilisk comparisons | — | only after assumptions aligned |

## Critical rule
Do not put long scientific campaigns in the default fast CI gate.
