# Phase D2 — Propagation consistency: shadow-event integration, consistent estimator profile, convergence guard

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_d2_propagation_predeclaration.json`, committed in `8963bf5`
  before any D2 number.
- **Default baseline:** recorded from the unmodified code in `studies/phase17/results/phase_d2_default_baseline.npz`.
- **Implementation:** `a3df595`.
- **Evidence:**
  - `studies/phase17/results/phase_d2_propagation.json` (gates, from `phase_d2_propagation.py`);
  - `studies/phase17/results/phase_d2_diagnostics.json` (diagnostics, not gates).
- **Authorisation:** the owner approved the re-freeze of `dynamics.py`, `srp.py` and `estimators.py` on 2026-10-23.
  `srp.py` was not changed.

## 0. Final status

```
STATUS     D2_NOT_QUALIFIED as computed: G4, G6 and G7 (SRIF status) FAIL; G1, G2, G3, G5, G8 PASS
PROPOSED   owner adjudication for G4, G6 and G7-SRIF (s4); each failure is a floor or a label, not a defect
OUTCOME    G4, G6 and G7-SRIF adjudications ACCEPTED by the owner 2026-10-25 -> D2_QUALIFIED (merged in PR #55)
HEADLINE   with the profile, BLS-LM converges 0.041 mm from the truth (D1: "Converged" 13 m off, best 5.6 mm);
           SRIF ends 0.044 mm from the truth (D1: 17.6 mm)
DEFAULTS   unchanged; flipping them is C4-I (with DE440), per the owner
```

## 1. What changed

### `dynamics.py`

- The three propagators (`propagate_state`, `propagate_augmented_state` including its harmonic branch, and
  `propagate_state_with_k_sensitivity`) gain `shadow_events=False`.
- With `shadow_events=True` and an active `CONICAL_PENUMBRA` SRP, the propagation runs in two passes:
  1. A 6-state DOP853 pre-pass at the caller's tolerances (max step 60 s) finds the penumbra contact epochs as
     roots of smooth contact functions:
     - outer contact: γ − (a☉ + a_occ);
     - inner contact: γ − |a☉ − a_occ|.
  2. The requested propagation then restarts at each contact, carrying the full 6-, 42- or 48-element state
     across.
- The RHS is continuous at a contact, so no saltation term is needed. G6 confirms this for the STM.
- New public helpers: `shadow_contact_times` and `PropagationProfile(method="DOP853", shadow_events=True)`.
- `propagate_state`'s RHS construction was moved, verbatim, into `_build_state_rhs`.

### `estimators.py`

`estimate_two_way_range_bls_lm` and `estimate_two_way_range_srif` gain `propagation_profile=None`. With a
profile:
1. Every propagation in the solve uses the profile: nominal and Jacobian, K sensitivity, candidate, rescore,
   damping-exhaustion relinearisation, and posterior.
2. The early-iteration tolerance loosening is off. This is the **F4 guard**: no Converged or J-Stab can be
   declared on an rtol 1e-8 linearisation.
3. SRIF scores the incumbent with the candidate's 6-state propagation, closing F3 inside SRIF.

### Provenance

- `docs/development/path-migration.json` records the new identities of `dynamics.py` and `estimators.py`; the
  previous ones are kept.
- `examples/r2_measurement_fidelity_validation.py` gains an additive `PHASE_D2_REFROZEN_CURRENT_TREE_SHA256`
  table, newest first in the resolver.
- `tests/test_f1_srp_variational_gradient.py` now asserts that the F1 record is preserved and that D2 is the
  current identity.

## 2. Gates as computed

| gate | result | number |
|---|---|---|
| G1 default invariance | **PASS** | 17/17 outputs bit-identical to the unmodified code: 3 propagators, SRP off, NO_SHADOW, events-on-but-inert, BLS-LM and SRIF state/status/cost |
| G2 suite | **PASS** | final tree: 1 905 passed, 0 failed, 29 skipped (incl. 10 new `test_shadow_events.py` tests). `test_recorded_run_workspace::test_archive_filters_and_sort_keep_actual_source_identity` failed in one run on the unmodified branch and in one on the modified tree, so it is intermittent and unrelated to D2 |
| G3 accuracy (≤ 1 mm) | **PASS** | max 0.43 mm (K-sensitivity, CANON); state 0.19 / 0.20 / 0.18 mm on CANON / O1 / O2 |
| G4 contacts (≤ 1e-3 s, same count) | **FAIL** | counts 60 / 60 / 0 equal; max \|Δt\| 0.64 ms CANON, **1.29 ms O1** |
| G5 consistency (≤ 1 mm) | **PASS** | state−augmented ≤ 0.28 mm, state−K-sensitivity ≤ 0.34 mm (D1: 0.27–2.0 m) |
| G6 variational (≤ 1e-5) | **FAIL** | STM columns ≤ 9.7e-8; **dx/dK 5.8e-5** |
| G7 closure | **FAIL** (SRIF) | BLS-LM **Converged** (STATIONARY_AT_DAMPING_LIMIT), 41 it., **0.041 mm**; SRIF **MaxIter**, 40 it., **0.044 mm** |
| G8 re-freeze | **PASS** | sha256 of `dynamics.py`, `srp.py` and `estimators.py` found in `path-migration.json` |

Runtime (reported, not gated), CANON arc:
- `propagate_state`: 2.8 s vs 0.29 s (~10×).
- augmented: 4.9 s vs 0.77 s (~6×).
- K sensitivity: 9.3 s vs 3.6 s (~2.6×).
- BLS-LM closure: 842 s with the profile, vs 559 s for D1's (false) default convergence. SRIF: 811 s vs 527 s.

## 3. Diagnostics of the three failures (`phase_d2_diagnostics.py`)

### G4: the reference, not production, is off

On the worst O1 contact, the outer-contact function was evaluated on an rtol 1e-13 event-segmented trajectory:

| epoch | value |
|---|---|
| production epoch | 1.45e-7 rad |
| reference bisection epoch | **−1.0e-6 rad** |

- The reference brackets the change of the illumination factor ν from 1. Within about 1 ms of geometric contact,
  the lens overlap is so shallow that ν rounds to exactly 1.0. The reference therefore sees the contact late.
- Production finds the root of the smooth contact function directly.
- Refining the reference grid from 10 s to 2 s does not help: the difference stays 1.36 ms. Its own self-change is
  0.13 ms.
- Production at rtol 1e-11 vs 1e-13 differs by 0.16 ms, consistent with 0.2 mm of trajectory difference.
- Threshold history: the 1e-3 s limit was copied from D1's R2 bracket gate. That gate measured the bracket width,
  not the reference's accuracy.

### G6: finite-difference floor, not a sensitivity error

| K step | rtol 1e-11 | rtol 1e-13 |
|---|---|---|
| 1e-3·K0 (predeclared) | 5.8e-5 | 6.6e-5 |
| 1e-2·K0 | **1.4e-6** | 5.7e-6 |

- A tenfold larger step lowers the error fortyfold, which is the signature of an FD noise floor.
- A K step of 1e-3·K0 moves the final position by only ~0.0024·|dx/dK|·K0. Integration noise in the difference
  quotient dominates at that scale.
- The STM columns, with 1 m and 1 mm/s steps, are at 1e-7.

### G7 SRIF: a termination label

- The SRIF has no damping. A rejected step leaves the state, and therefore the next linearisation and the next
  step, identical. Once one step is rejected at the integration noise floor, every remaining iteration repeats it
  until `max_iter`.
- This follows deterministically from the code: `x_nominal = x_best` on rejection, and nothing else changes.
- The state is 0.044 mm from the truth, at χ² 1.2e-7 on noise-free data.
- BLS-LM, whose damping-exhaustion path tests stationarity, reports it as Converged.

## 4. Proposed adjudications (owner decision)

1. **G4:** accept. The 1.3 ms is the reference's ν = 1 rounding floor. Production's epoch is closer to the
   geometric contact by a factor of 7 in the contact function.
2. **G6:** accept. dx/dK agrees with finite differences to 1.4e-6 at a step above the FD floor. The STM agrees
   to 1e-7.
3. **G7 SRIF:** accept the accuracy (0.044 mm ≤ 1 mm). Record the MaxIter label as a known SRIF limitation:
   there is no stationarity exit after a rejected step. A follow-up could add one (BLS-LM-style
   `_damping_exhaustion_status`), opt-in under the profile.

If all three are accepted: **D2_QUALIFIED**.

## 5. Consequences

- The C-phase closure floors (BLS 1–12 mm DampingLimit, SRIF 0.1 m MaxIter) were propagation artifacts. D1
  showed this; D2 removes them: with the profile, both estimators close to < 0.05 mm on noise-free data.
- Cost: the profile multiplies propagation time by about 3–10× (DOP853 vs VODE Adams, plus the pre-pass).
  Phase E is the place to recover it, e.g. with a VODE tolerance that meets its nominal accuracy, or a cached
  pre-pass.
- **Default flip (C4-I):** make `PropagationProfile()` the estimator default and `shadow_events=True` the
  propagator default together with DE440, in one re-baselining.

## 6. Limits

- SIMPLIFIED dynamics (point mass + J2 + cannonball SRP), three arcs, one closure start.
- The contact pre-pass has a 60 s maximum step, so a grazing penumbra shorter than one step can be missed.
- Earth shadow is not modelled; the SRP shadow is lunar only.
- UKF and the other estimators are untouched.
