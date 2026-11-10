# F3 WP3.2 S1 — The lunar solid tide as an opt-in production force term (REQ-DYN-007, DYN-06)

- **Date:** 2026-11-10
- **Predeclaration:** `studies/phase17/configs/wp3_2_s1_lunar_tide_predeclaration.json`, committed in `bdafb53`
  (2026-11-08) before any WP3.2 code, test or number. No amendment.
- **Code:**
  - S1a (`267854f`): the new module `src/lunar_od/lunar_tide.py`, and the tide in the propagators of
    `src/lunar_od/dynamics.py`;
  - S1b (`237bb93`): its transport through `src/lunar_od/propagation_contract.py` and `src/lunar_od/estimators.py`.
- **Tests:**
  - `tests/test_wp3_2_lunar_tide.py`, 40 tests: T_G1(a) and T_G2–T_G8;
  - `tests/test_wp3_2_lunar_tide_transport.py`, 39 tests: B_G1–B_G3 and the re-freeze record.
- **Results:**
  - `results/wp3_2_s1_default_identity.json` (T_G1(b), `10f2366`);
  - `results/wp3_2_s1_gate_values.json` (`237bb93`; script `scripts/wp3_2_s1_gate_values.py`). It records the
    values the tests compare with their thresholds;
  - `results/wp3_2_s1_tg5_margin_diagnostic.json` (exploratory, section 5). The script was committed in `97dea1a`
    before it ran; the result was committed with this report.
  - The non-timing gate values were reproduced bit for bit on the clean tree at `97dea1a`.
- **Production defaults changed:** none. With `lunar_tide` omitted or `None`, all 25 recorded configurations are
  byte-identical before and after the change (T_G1(b)).

## 0. Final status

```
S1 (as predeclared)   = every gate PASS on the first evaluation, no amendment
  T_G1 identity       = (a) the wrapper returns the RHS itself when off; (b) 25/25 configurations byte-identical
  T_G2-T_G4 physics   = IERS 2010 eq. 6.6 through the production SH engine 8.9e-14 (<= 1e-12);
                        WP3.1 study term 2.1e-15 (<= 1e-13); complex step 1.9e-15 (<= 1e-13)
  T_G5 variational    = STM 7.37e-8 (<= 1e-7; thin margin, section 5); control without the tide gradient
                        2.29e-5 (310x); K_SRP sensitivity 1.5e-7 (<= 1e-6)
                        exploratory diagnostic: the 7.4e-8 is noise in the oracle's differences (~1/h);
                        the best oracle per column agrees with the STM to <= 3.3e-9
  T_G6-T_G8           = path consistency, shadow events and fail-closed validation PASS
  B_G1-B_G3 transport = options, every propagation of the six public estimators, recovery 3.5e-7 m
                        (6e6x below the 2.08 m bias without the tide)
  T_G9 regression     = 2695 tests, 0 failures, 0 errors, 33 skipped (the base count); check_repository passes
REQ-DYN-007           = still partial: S2 LRP, S3 EARTH_SHADOW, S4 SRP_LIGHT_TIME, S5 hold-out replication
DYN-06                = the dynamics half is available opt-in; the tide-system requirement is enforced by a
                        required TIDE_FREE declaration, which the caller must make true (section 2)
```

## 1. What changed

- **`lunar_tide.py` (new).** `LunarTideOptions` is frozen and keyword-only. Its fields are:
  - `static_field_convention`: required, with no default; only `"TIDE_FREE"` is accepted;
  - `love_numbers=((2, 0.024223), (3, 0.0163))`, `radius_m=1738e3`, from the GL1800F label;
  - `raising_bodies=("EARTH", "SUN")`, and `provenance`.

  `tide_acceleration`, `tide_gravity_gradient` and `tide_acceleration_and_gradient` evaluate the elastic tide of
  degrees 2 and 3 raised by the Earth and the Sun in closed form. They are safe under the complex step. Invalid
  options raise (T_G8).
- **`dynamics.py`.** `_with_tide_acceleration` wraps a built RHS and returns it unchanged when the tide is off, the
  same contract as the SRP wrapper. With the tide on it adds:
  - `a_tide` to the acceleration;
  - in the 42-state RHS, `G_tide @ Phi[0:3, :]` to rows 3:6 of `Phi_dot`;
  - in the 48-state RHS, also `G_tide @ s_K[0:3]` to `s_K_dot[3:6]`.

  The 6-state path never evaluates the gradient. The Earth and Sun lookups are cached per epoch only when the tide
  is on.

  The keyword `lunar_tide=None` is added to `propagate_state`, `shadow_contact_times`, both paths of
  `propagate_augmented_state` (plain and harmonic), `propagate_state_with_k_sensitivity` and
  `propagate_truth_with_ephemeris`. The event-segmented pre-pass integrates the same tide-on RHS, so its contact
  cuts belong to the trajectory that is propagated (T_G7).
- **Transport (S1b).** `model_options` and `propagator_options` add the key `lunar_tide` only when a
  `LunarTideOptions` is given, so every tide-free options dict is the historical one. The 15 `estimators.py`
  functions that take `propagation_profile` gain `lunar_tide=None` and forward it to every propagation:
  - nominal, candidate and rescoring propagations;
  - the Jacobian and the K sensitivity;
  - the posterior.
- **Protected files.** `dynamics.py` and `estimators.py` are re-frozen additively in the new table
  `F3_WP3_2_S1_REFROZEN_CURRENT_TREE_SHA256` of `examples/r2_measurement_fidelity_validation.py`:

  | file | previous identity | new identity |
  |---|---|---|
  | `dynamics.py` | `6827b9d9` (F1 WP1.3) | `ae0f3d0d` |
  | `estimators.py` | `ee18f3c9` (F4 WP4.5b) | `0ed77589` |

  The history tables are unchanged, and the history-chain tests are extended. `lunar_od/__init__.py`, also
  protected, is not touched, so the tide is imported from `lunar_od.lunar_tide`.

Usage:

```python
from lunar_od.lunar_tide import LunarTideOptions
tide = LunarTideOptions(static_field_convention="TIDE_FREE")   # the caller asserts the field is tide-free
x = propagate_state(t, x0, mu_moon, mu_earth, mu_sun, earth_pos, sun_pos, lunar_tide=tide)
x_hat, status, stats = estimate_position_srif(..., lunar_tide=tide)
```

## 2. The tide-system requirement

The elastic tide adds the whole tidal potential, including its time average, the permanent tide. That is right
only for a static field that excludes the permanent tide (a tide-free field).

| field | what its label says |
|---|---|
| GL1800F (`jggrx_1800f_sha.lbl`) | "The second and third degree gravity coefficients of this model do not include the permanent tide": tide-free |
| GRGM660PRIM, GRGM1200L | nothing about the tide system |
| production default J2 (2.0346e-4 at 1737.4 km, "IAU/GRAIL") | no stated tide system |

`static_field_convention` therefore has no default. A caller must write `"TIDE_FREE"`, and that is a statement
about the field the caller uses, which the code cannot check. A parsed tide-system field in the loader would not
help either: the repository's fields are PDS SHADR files, whose tide system is free label text (out of scope,
predeclaration). `"ZERO_TIDE"` and `"MEAN_TIDE"` raise, because removing the permanent part requires a conventional
permanent tide for the Moon, which this slice does not define.

**How large is the permanent part.** The predeclaration asked for this number in the report. The Moon rotates
synchronously, so the Earth stays near the +x axis of the body frame (latitude 0, longitude 0). The permanent
Earth-raised tide then follows from IERS 2010 eq. 6.6 with that fixed direction. Inputs:

- the GL1800F k2 and R;
- GM_E / GM_M = 81.30, from `lunar_od.constants`;
- a = 384 400 km, with <d^-3> = a^-3 (1 - e^2)^-3/2 for e = 0.0549 (a factor of 1.0045).

The result:

| quantity | value |
|---|---|
| ΔC̄20 | −4.09e-8 |
| ΔC̄22 | +7.08e-8 (sectoral, because the Earth is fixed in the body frame) |
| ΔJ2 = −√5 ΔC̄20 | 9.14e-8 at 1738 km |
| ΔJ2 R² against the default J2 R² | **4.50e-4** |
| permanent Sun-raised ΔC̄20 | −2.3e-10, 0.56 % of the Earth's |

ΔC̄22 is +7.08e-8 rather than zero because the Earth is fixed in the body frame. These are back-of-envelope values:
they use the elastic k2 for the permanent forcing (the IERS convention) and ignore the small offset between the
mean Earth direction and the principal axes.

- **Pairing the tide with the default J2, or with a GSFC field of unknown tide system, may double count the
  permanent part.** That part is 4.5e-4 of J2 R², comparable to the tide's own signal.
- **Observation, not established.** DYN-04 found the default (J2, R) pair 4.7e-4 above GRGM1200A's in J2 R². That is
  the same sign and size as the permanent tide, but a coincidence is not excluded, and nothing here tests it.
- **The verified pairing is GL1800F with the tide on.** The gate fixtures use a synthetic truth (J2 or a synthetic
  zonal field with the tide), so they show that the term is implemented and transported correctly. They do not show
  which real field it may be paired with.

## 3. Gates

The thresholds are the predeclared ones; the values are as computed.

| gate | threshold | computed | verdict |
|---|---|---|---|
| T_G1(a) identity when off | `rhs` itself for 6, 42, 48 states | object identity | PASS |
| T_G1(b) default paths | 25 configurations byte-identical, pre run twice | 25/25 (pre `bdafb53` ×2; post `237bb93`, omitted and `None`, src clean) | PASS |
| T_G2 IERS eq. 6.6 through the production SH engine | 1e-12 of A r^-(n+2), A r^-(n+3) | acc 1.28e-14, grad 8.88e-14, sums 3.75e-14 | PASS |
| T_G3 = WP3.1 study term | 1e-13 relative | 2.08e-15 | PASS |
| T_G4 gradient | complex step 1e-13; G = Gᵀ exactly; trace 1e-12 | 1.85e-15; exact; 1.81e-15 | PASS |
| T_G5(a) STM against RK4 re-propagation | ≤ 1e-7; control ≥ 10× | 7.37e-8; control 2.29e-5 (310×) | PASS |
| T_G5(b) K_SRP sensitivity | ≤ 1e-6 | 1.52e-7 | PASS |
| T_G5(c) RHS structure | 4 eps, nothing else moves; no gradient on the 6-state path | as stated | PASS |
| T_G6 6/42/48-state paths (J2 and a zonal harmonic model) | positions 1e-5 m; Φ 1e-8 of max\|Φ\| | ≤ 9.8e-7 m; ≤ 2.7e-13 | PASS |
| T_G7 shadow events with the tide | contact checks pass; a cut moves > 1e-3 s; the control raises | 50 cuts, max shift 0.0246 s (46 > 1e-3 s); control raised at 7822.9 s (2.72e-3 s from the contact) | PASS |
| T_G8 fail closed | each predeclared invalid input raises | all raise | PASS |
| B_G1 options dicts | historical without the tide; only `lunar_tide` added with it; TypeError otherwise | as stated | PASS |
| B_G2 transport | every captured propagation carries the same object; none has the key when omitted | position and range rate (BLS-LM, SRIF, with and without a profile); two-way (with and without K solve-for) | PASS |
| B_G3 recovery | \|dr0\| < 1e-3 m, \|dv0\| < 1e-6 m/s; without the tide ≥ 100× | BLS-LM 3.45e-7 m, 1.57e-10 m/s; SRIF 3.42e-7 m, 1.56e-10 m/s; without the tide 2.083 m (contrast 6.0e6) | PASS |
| T_G9 regression | 0 failures, 0 errors; skips unchanged; `check_repository` passes; re-freeze additive | at `10f2366` (src as `237bb93`), 1529 s: 2695 tests (2616 at the WP3.1 base + 79 new), 0 failures, 0 errors, 33 skipped (the base count; the same set as the S1a run, 2656/0/33); `check_repository` passes; one new table, history unchanged, history-chain tests extended | PASS |

T_G6 and T_G7 also record the size of the term on these test orbits: the tide moves the 100 km polar orbit by
10.0 m in 6 h and by 43.6 m in a day. These numbers are information, not gates.

## 4. Expectations against outcomes

The predeclaration stated these before the run. They are not gates.

| expectation | outcome |
|---|---|
| T_G2 and T_G4 near 1e-15 | 1.3e-14 to 8.9e-14 (T_G2), 1.9e-15 (T_G4): met in order |
| T_G5(a) gated error ~1e-9..1e-8 | **7.37e-8: off by about an order of magnitude.** Section 5 traces it to the oracle's noise; the STM itself agrees to ≤ 3.3e-9 |
| T_G5(a) control ~1e-6..1e-5 | 2.29e-5: slightly above the range |
| metres in 6 h, tens of metres in a day | 10.0 m and 43.6 m: met |
| the wrapper costs about as much per RHS call as the SRP wrapper | **wrong**: see below |

Cost per call of the 6-state RHS:

| wrapper state | recorded run | rerun on the clean tree |
|---|---|---|
| tide off | 13 µs | 20 µs |
| SRP on | 25 µs | 44 µs |
| tide on | 100 µs | 156 µs |

The absolute numbers depend on the machine's load. In both runs the tide costs 7–8 times the tide-free call and 3.5–4
times the SRP call. The tide is evaluated in Python (four degree terms and two lookups), whereas the SRP term calls a numba kernel. A
behaviour-preserving fast path (numba, with parity tests against this implementation) is a follow-up. It is not
part of S1, and the tide-off path is not affected.

## 5. The T_G5(a) margin (exploratory, after the result)

T_G5(a) passed at 7.37e-8 against 1e-7, a margin of 1.36×, where 1e-9..1e-8 was expected.

`scripts/wp3_2_s1_tg5_margin_diagnostic.py` varies only the oracle: the RK4 step dt, and a scale on the
central-difference steps h (1 m and 1e-3 m/s). The production STM is computed once, as in the test.

Each entry is max|FD_j − Φ e_j| / max|Φ e_j| for column j, the metric of the gate.

| dt | h (position, velocity) | x | y | z | vx | vy | vz | worst |
|---|---|---|---|---|---|---|---|---|
| 5 s | 1 m, 1e-3 m/s (predeclared) | 6.74e-8 | 3.66e-8 | 1.09e-9 | 2.88e-9 | 2.95e-9 | 7.37e-8 | **7.37e-8** |
| 2.5 s | 1 m, 1e-3 m/s | 5.02e-8 | 8.45e-8 | 1.52e-9 | 1.53e-9 | 5.11e-9 | 1.22e-8 | 8.45e-8 |
| 5 s | 10 m, 1e-2 m/s | 1.50e-9 | 1.76e-9 | 1.48e-8 | 1.31e-8 | 6.16e-9 | 3.27e-9 | 1.48e-8 |
| 2.5 s | 10 m, 1e-2 m/s | 1.57e-8 | 1.16e-8 | 1.44e-8 | 1.28e-8 | 6.90e-9 | 7.77e-9 | 1.57e-8 |
| 5 s | 0.1 m, 1e-4 m/s | 8.44e-7 | 2.33e-7 | 1.69e-8 | 3.51e-9 | 1.55e-8 | 2.44e-7 | 8.44e-7 |

The first row reproduces the gate value exactly. The file also compares the oracles with each other.

Reading:

- **At the predeclared steps, the error is floating-point noise in the oracle's central differences.** It grows
  about tenfold when h shrinks tenfold (8.44e-7 at h × 0.1). Halving dt does not reduce it (8.45e-8), so RK4
  truncation does not dominate.
- **At h × 10 the noise falls, and a part that scales with h² takes over.**
  - In the z and vx columns the error is 1.4e-8 at both steps, and the two oracles agree there to 4e-10. A part
    that does not depend on dt and was 10–100 times smaller at h × 1 is the central-difference truncation of the
    nonlinear 6 h flow.
  - In the x and y columns the two h × 10 oracles differ by 1.4e-8. That part is noise: twice the steps,
    differently rounded.
- **What this says about the production STM.** For every column, at least one of the five oracles agrees with the
  production STM to ≤ 3.3e-9. Unless the oracle errors happen to cancel, the production STM is therefore accurate to
  a few 1e-9. The predeclared expectation of 1e-9..1e-8 holds for the variational equations. What it missed is the
  oracle's own noise at the predeclared steps.

**What this means.**

- No gate, test or threshold changed. T_G5(a) stays as predeclared and passed.
- The margin is a portability risk: on another platform, or with another numpy or numba build, the floating-point
  noise in the oracle may cross 1e-7 without any defect in the variational equations. If that happens, the remedy is
  a larger difference step recorded in the amendment log, and it is exploratory there. It is not a looser
  threshold.
- About 3 times the predeclared steps would balance the two parts: noise near 2.5e-8, truncation near 1.3e-9.
  That estimate is scaled from the table and was not run.
- The control is 310× the gated error, so the gate still separates a missing tide gradient from noise by more than
  two orders of magnitude.

## 6. Test-setup fixes before evaluation

Both were made while the tests were written, before any gate value was evaluated. Neither changes a gate.

- **B_G2, two-way spy.** The first fixture was 120 s with 3 epochs. It left the two-way posterior rank-deficient
  (`RankDeficientCovarianceError`, rank 2 of 6) with or without the tide, so the spy captured nothing. The fixture
  is now 1800 s.
- **B_G2, SRIF with K solve-for.** SRIF requires an initial K_SRP uncertainty, so the test passes
  `k_srp_prior_sigma=0.01`.
- **T_G5(c), "the 6-state path never calls the gradient".** The test patches the gradient functions to raise.
  Its positive control, which shows that the patch is live, runs the 42-state propagation with DOP853, not the
  default VODE: an exception raised inside VODE's callback does not propagate reliably.

## 7. Not done and limits

- **Deferred (predeclared).** The following do not get the parameter in S1: UKF, EKF and the fast sigma propagator;
  scenarios, `scenario_config`, the CLI and the desktop app; observability, `numerical_metadata` and the force
  contract.
  - None of them can enable the tide. They fail closed by absence: the parameter does not exist there.
  - They join with the scenario-level transport after S4.
  - Until then a scenario run cannot use the tide, and its metadata does not record it.
- **Out of scope.**
  - The frequency-dependent and dissipative tide (lagged, Q ~ 40), and order-dependent k2m: GRGM660PRIM's k20, k21
    and k22 differ by up to 4 %. Both are bounded in S5's error budget.
  - Zero-tide and mean-tide fields.
  - The loader half of DYN-06.
- **Validated geometry.** Synthetic Earth and Sun on circular orbits, a 100 km polar orbit, 6 h and 1 day arcs,
  noiseless position observations. The tide's OD materiality on the WP3.1 population is S5's hold-out replication.
- **REQ-DYN-007 stays partial** until S5.

## 8. Agent decisions

All were made under the owner's 2026-11-08 delegation ("en doğru kararı ver ve uygula"). They are flagged for the
owner's review, and merging stays with the owner.

1. A wrapper (additive, identity when off) instead of threading the term through `f3body_moon` and
   `dynamics_jacobian_a_matrix`. It leaves the numba kernels and every frozen path untouched.
2. A required `static_field_convention` with only `TIDE_FREE` accepted.
3. GL1800F's elastic k2 and k3 as the only parameter set.
4. The deferral list (section 7).
5. The additive re-freeze of `dynamics.py` and `estimators.py`. `propagation_contract.py` is not protected.
6. The branch `f3/wp3-2-force-terms`, stacked on PR #115.
7. Section 5's diagnostic, which was run after the result and is exploratory.

## 9. Reproduce

From the worktree root, with `NUMBA_CACHE_DIR` set:

```
.venv/Scripts/python -m pytest tests/test_wp3_2_lunar_tide.py tests/test_wp3_2_lunar_tide_transport.py -q -p no:cacheprovider
.venv/Scripts/python studies/phase17/scripts/wp3_2_s1_gate_values.py OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s1_tg5_margin_diagnostic.py OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s1_default_identity.py run omit OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s1_default_identity.py compare A.json B.json --out OUT.json
.venv/Scripts/python tools/check_repository.py
```

The permanent-tide numbers in section 2:

```python
import numpy as np
from lunar_od.constants import MU_EARTH_M3S2 as GE, MU_MOON_M3S2 as GM, J2_MOON_UNNORMALIZED as J2, R_MOON_M as RM
k2, R, a, e = 0.024223, 1738e3, 384400e3, 0.0549
f = k2 / 5 * GE / GM * (R / a) ** 3 * (1 - e * e) ** -1.5
dc20, dc22 = f * np.sqrt(5) * -0.5, f * np.sqrt(10 / 24) * 3      # Pbar20(0), Pbar22(0)
print(dc20, dc22, -np.sqrt(5) * dc20 * R**2 / (J2 * RM**2))      # -4.09e-8, 7.08e-8, 4.50e-4
```
