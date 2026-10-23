# Phase D1 — Integrator and propagator accuracy, shadow events, and the estimator closure floor

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_d1_integrator_predeclaration.json`, committed in `da3d150`.
  The script followed in `0496853`, before any number.
- **Evidence:** `studies/phase17/results/phase_d1_integrator.json`, from
  `studies/phase17/scripts/phase_d1_integrator.py`.
- **Production code changed:** none. `dynamics.py` and `srp.py` are frozen, and so is `estimators.py`.

## 0. Final status

```
GATES        R1 reference convergence PASS (<= 8.8e-5 m) | R2 events PASS (60 per eclipsing arc, brackets <= 1e-4 s)
             R3 reproduction PASS
CLOSURE      HYPOTHESIS NOT CONFIRMED (predeclared rule): with the truth propagated by the estimator's own nominal
             propagator, BLS-LM still ends at DampingLimit 15.8 m from the truth
FINDINGS     F1 ADAMS (VODE) at rtol 1e-11 is 0.09-0.26 m off over a ~1.2 day arc even without shadow
             F2 no shadow-event handling: up to ~1.7 m more for unsegmented propagation (CANON)
             F3 the estimator is internally inconsistent: Jacobian/nominal with propagate_augmented_state,
                LM candidate cost with propagate_state; the two differ by 0.27-2.0 m
             F4 BLS-LM reports "Converged" 13.0 m from the truth (C2-I G6's run, reproduced): a false convergence
DECISION     production fixes touch frozen modules: OWNER DECISION (s4)
```

## 1. Propagation accuracy (Q1, Q2)

- **Reference:** DOP853 at rtol 1e-13, restarted at every penumbra/umbra boundary. Event epochs come from
  bisection on the illumination fraction to 1e-4 s.
- **Reference self-convergence:** 1e-13 vs 1e-12 agree to ≤ 8.8e-5 m.
- **Arcs:** k0, about 1.2 days.

Max position error against the reference:

| variant | CANON (60 events) | O1 eclipse-rich (60) | O2 full sun (0) |
|---|---|---|---|
| A `propagate_state` ADAMS 1e-11 (closure truth; LM candidates) | **1.93 m** | 0.57 m | **0.26 m** |
| B `propagate_augmented_state` ADAMS 1e-11 (estimator nominal/Jacobian) | 0.105 m | **0.80 m** | 8.4 mm |
| C `propagate_state_with_k_sensitivity` ADAMS 1e-11 (KP2/C-phase histories) | 37 mm | 7.8 mm | 1.3 mm |
| D `propagate_state` DOP853 1e-11, no events | 1.91 m | 0.48 m | **0.18 mm** |
| E ADAMS 1e-11 with NO_SHADOW | 0.17 m | 0.09 m | 0.26 m |

Mutual mismatch: |A − B| is 2.01 / 0.30 / 0.27 m and |A − C| is 1.96 / 0.58 / 0.26 m on the three arcs.

Readings:
- **F1.** On the shadow-free arc DOP853 at 1e-11 is exact to 0.2 mm, while VODE Adams-12 at the same tolerances
  is 0.26 m off. The production default method does not deliver what its tolerance suggests.
- **F2.** With shadow, unsegmented propagation picks up up to ~1.7 m from the acceleration kinks at penumbra
  entry/exit, whatever the method. This is the Orekit trial's finding 2, now quantified per variant.
- **Variant spread.** The augmented propagators (B, C) take different, smaller steps because their error
  control includes the STM/sensitivity components. Their errors vary from 1 mm to 0.8 m with the arc and are not
  monotone.

## 2. The closure floor (Q3)

All runs start from the C2-I G6 start (x0 + N(0, PRIOR_SD), seed [98, 0, 0]): CANON k0, range-only six-state,
DSN_C2 light time, noise-free.

| truth propagated with | BLS-LM | SRIF |
|---|---|---|
| A (closure-gate construction) | **Converged**, 12.98 m from truth (25 it.) | MaxIter, 0.086 m |
| event-segmented reference | MaxIter (80 it.), 5.6 mm | MaxIter, 17.6 mm |
| B (the estimator's own nominal propagator) | **DampingLimit**, 15.81 m | MaxIter, 35 mm |

- **Hypothesis not confirmed.** Feeding the estimator its own nominal propagation does not let it converge.
- **F3.** Inside one solve, the linearisation and nominal come from B, while the LM acceptance cost of a
  candidate comes from A. These are 0.3–2.0 m apart, so a step predicted to reduce χ² is judged on a different
  trajectory, and LM stalls or exits.
- **F4.** BLS-LM's "Converged" at 13.0 m is a false convergence: the termination test passed while the state
  is 13 m off, against an initial offset of 100 m. It reproduces C2-I G6's C2 run.
- **Best case.** With the most accurate truth, BLS reaches 5.6 mm and SRIF 1.8 cm, still without converging.
  The estimator's own 0.1–0.8 m propagation error remains.

## 3. Q4: Hermite interpolator

The interpolated velocity differs from the derivative of the interpolated position by 2.6e-7 m/s in the first
interval, 5.7e-6 m/s in the interior and 6.3e-6 m/s in the last interval. The last interval is **not** special
at this level. The 0.03 m/s last-row effect seen in C2-I G5 therefore has another cause, most likely the event
epoch t1 of the last row falling outside the history grid. It is recorded for D2.

## 4. Recommendations (owner decision; frozen modules)

A phase D2 "propagation consistency" with re-freeze authorisation for `dynamics.py`, `srp.py` and
`estimators.py`:

1. **Event-segmented integration** at shadow boundaries in every production propagator (state, augmented,
   K sensitivity). This removes F2.
2. **Default integrator.** DOP853, or VODE with a tolerance that achieves its nominal accuracy. F1 shows ADAMS
   at 1e-11 does not; the DOP853 cost must be measured.
3. **One propagation for the nominal and the LM candidate cost** (F3), e.g. the candidate state part from the
   same augmented propagator. This removes the internal mismatch.
4. **A termination guard** against F4: require the predicted and actual χ² decrease to agree, or a final
   full-precision check, before reporting Converged.

**Qualification target for D2:** from the C2-I G6 start, BLS-LM and SRIF converge to ≤ 1 mm on noise-free data
from an independent truth. That would turn every C-phase closure gate into a clean pass, and it also matters
for the Phase F real-data fits.

## 5. Limits

- SIMPLIFIED dynamics; three arcs.
- One start point for the closure runs.
- The reference shares the production force functions; it checks integration, not force modelling.
