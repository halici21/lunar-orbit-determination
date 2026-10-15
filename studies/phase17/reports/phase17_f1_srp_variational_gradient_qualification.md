# PHASE 17-F1 — SRP POSITION-GRADIENT VARIATIONAL CLOSURE

> **The question, answered first.** Can the variational equations carry the exact position gradient of the
> production SRP force, without changing that force? And does the penumbra-driven STM/Jacobian discrepancy of
> HARDEN-D05 then collapse to the oracle floor?
>
> **Yes.**
> - **Force unchanged.** The SRP acceleration and the state-only truth are **bitwise unchanged**.
> - **STM.** Against independent re-propagation the corrected STM's worst best-resolved error over all six columns
>   is **9.9e-8**; the oracle floor is 3.8e-8. The pre-F1 model fails the same oracle at 1.2e-6. A fixed-step oracle
>   separates them further: 2.2e-8 versus 3.4e-6.
> - **dx/dK** is qualified at **8.7e-7**.
> - **Science.** Every Phase-17 class is preserved. The canonical numbers move in the 5th–6th significant digit:
>   ΔDOR `f_perp` 0.172351 → **0.172343**, range-only 0.091508 → **0.091502**.
>
> `PHASE17_F1_GATE` is in §46. All numbers are `SYNTHETIC_CAMPAIGN_TRUTH`.

---

## 1. Executive Summary

F1 adds `lunar_od.srp.srp_acceleration_position_gradient`, the complete ∂a_SRP/∂r of the current cannonball force
with its conical lunar shadow, to the variational A matrix of every SRP-active propagation path.

| check | result |
|---|---|
| local gradient vs independent oracles (every regime) | ≤ 1.5e-7 |
| STM oracle (HARDEN-D05 setup) | 9.9e-8, versus 1.2e-6 for the old model |
| S_K oracle | 8.7e-7 |
| range, counted-Doppler, ΔDOR and optical derivative chains | pass |
| physical-force parity | bitwise |
| D2 primary population (18 cells) | classes preserved |
| GEO spot check (5 frozen cases) | classes preserved; no escalation |
| D2 high-fidelity spot check | classes preserved |
| materiality label | `NUMERICAL_BASELINES_UPDATED_CLASSES_PRESERVED` |

The protected `dynamics.py` was changed under the brief's explicit authorisation. It was re-frozen additively,
only after qualification.

## 2. Entering Authoritative State

Verified, not assumed:

| field | value |
|---|---|
| branch | `feature/phase17-r-k-srp-estimation` |
| HEAD | `1025f4abf8ecc726a09c20ab09d103875751d01f` |
| tree | `e63be848a0579290979eb678eb954148ddf69298` |
| main = origin/main | `fea476f81dad07b3914e53eab709e10fa6e9d10b` |
| entering gates | PHASE17_HARDEN_GATE = PASS; INFRASTRUCTURE = QUALIFIED_FOR_NEW_SCIENCE |
| entering suite | 1391 collected / 1362 passed / 0 failed / 0 errors / 29 skipped |
| canonical baseline | range 0.091508, combined 0.172351, gain +0.0808, σ_K/K 7.46% (`studies/phase17/reports/phase17_authoritative_status.md`) |
| v2 instructions | `CLAUDE_OPUS_5_5_REPO_INSTRUCTIONS_v2.md` absent (not blocking, per the brief) |

The pre-F1 materiality run at this HEAD reproduced the stored D2 and GEO values exactly.

## 3. Why F1 Was Required

HARDEN-D05: the pre-F1 A matrix held gravity gradients only. The SRP force's position dependence was omitted.
- **Average size.** Phase 13 measured the omitted term at 4.02e-13 of the gravity gradient on average.
- **In penumbra.** The term reaches 1.09e-11 s⁻² there, while its median is 7.7e-19.
- **Effect on the STM.** Its effect accumulates secularly: 2e-6 to 1e-5 relative STM error over 4 orbits.

HARDEN proved this was the cause, but `dynamics.py` was protected. F1 is the authorised repair.

## 4. Repository / Protection State

| file | gate | before | after |
|---|---|---|---|
| `src/lunar_od/dynamics.py` | R2 current-tree gate (`expected_current_tree_sha256`) | b369899a (HARDEN re-freeze, content of 0e785f6, blob c973822f) | c6e7975c (9a0e786, blob 44cf92ca), via additive `PHASE17_F1_REFROZEN_CURRENT_TREE_SHA256` |
| `src/lunar_od/accelerated.py` | R2 closure re-freeze e455b36b | unchanged | unchanged |
| `src/lunar_od/srp.py` | none (value parity with Phase 13 only) | d9ba4c22 | extended (the new gradient functions only) |
| `src/lunar_od/two_way_range.py` | R3-P21-M3 byte gate | unchanged | unchanged |

## 5. Task-State Completion

`studies/phase17/results/phase17_f1_task_state.json` tracks the checklist; every item is closed. Predeclarations were committed
before implementation in bfc98e4 (`studies/phase17/configs/phase17_f1_predeclaration.json`). They cover:
- the GEO spot cases;
- the escalation rule;
- all numerical gates.

## 6. Parallel Tools / Subagents Used

- Parallel tool calls were used for independent inspections, and process pools for the oracles.
- **No subagents were used.** The independent verification is carried by oracles that never touch the code under
  test (§15–§23). A second agent would not have added independence.

## 7. Existing SRP Force Model

`srp_acceleration_with_lunar_shadow(r, r_sun, opts)` = K · `srp_acceleration_kernel`, with

  g = P(d) ν u,  q = r − r_sun,  d = |q|,  u = q/d,  P(d) = P₁AU (AU/d)².

- The frame is Moon-centred J2000.
- There is no velocity dependence.
- K ≥ 0.
- K = 0 or umbra returns exact zeros.

## 8. Existing Shadow Model

ν comes from `_illumination_for`:
- NO_SHADOW gives 1.
- The sunward half-space (r·r_sun > 0) gives 1.
- Otherwise `illumination_fraction` compares the apparent radii α_s = asin(R_sun/d_s) and α_o = asin(R_moon/d_o)
  with their separation γ:
  - FULL_LIGHT (γ ≥ α_s + α_o): 1;
  - UMBRA: 0;
  - ANNULAR: 1 − (α_o/α_s)²;
  - PENUMBRA: 1 − L(α_s, α_o, γ)/(π α_s²), with the planar lens area L, and max(0, ·).

The model was not modified.

## 9. Exact Position-Gradient Derivation

a = K P₁AU AU² ν q/d³, so

  **∂a/∂r = K [ (P ν / d)(I − 3uuᵀ) + P u (∇ν)ᵀ ]**

using ∂(q/d³)/∂r = I/d³ − 3qqᵀ/d⁵.

**∇ν in the constant regimes.** ∇ν = 0 in NO_SHADOW, on the sunward exit, in FULL_LIGHT and in UMBRA.

**∇ν in the variable regimes.** Let e_s and e_o be the unit vectors from the spacecraft to the Sun and to the Moon,
and c = e_s·e_o. Then:
- ∇α = R/(d√(d²−R²)) e, for each body;
- ∇γ = [(e_o − c e_s)/d_s + (e_s − c e_o)/d_o]/sin γ;
- ANNULAR: ∇ν = −2(α_o/α_s)(α_s∇α_o − α_o∇α_s)/α_s²;
- PENUMBRA: ∇ν = −∇L/(πα_s²) + 2L∇α_s/(πα_s³).

The lens-area partials are exact: ∂L/∂rᵢ = 2 rᵢ θᵢ (the arc of disk i inside the other) and ∂L/∂γ = −2h (the
common chord).

**Behaviour at the contacts.** Both partials vanish at external and internal tangency, so **∇ν is continuous across
both contacts**. ν itself is only C¹ there (ν ∼ ε^{3/2}).

## 10. Units / Signs / Frame Contract

- **Units.** G is in s⁻² (checked against dimensional closed forms, §15).
- **Sign.** u points from the Sun to the spacecraft, matching the production convention; the force is anti-solar.
- **Frame and origin.** Moon-centred J2000, the same as the acceleration and A.
- **Placement.** G enters only the acceleration-vs-position block of A.

## 11. Variational-Equation Integration

The same A drives Φ̇ = AΦ and Ṡ_K = A S_K + ∂f/∂K. ∂f/∂K = [0; g] is unchanged, but S_K changes through A.

| path | how G enters |
|---|---|
| 48-state `propagate_state_with_k_sensitivity` (Phase-17 science) | `dynamics_jacobian_a_matrix(..., srp=srp)` |
| 42-state harmonic path | same |
| 42-state numba / reference path | the `_with_srp_acceleration` wrapper adds G·Φ[0:3,:] to rows 3:6 of Φ̇, i.e. exactly (A_grav + [[0,0],[G,0]])Φ |

## 12. Implementation

The smallest shape that works:
- one helper `_illumination_and_gradient`, which mirrors `_illumination_for` branch for branch on the same
  `apparent_radii_and_separation` quantities, and returns the ν that production uses (tested);
- one public `srp_acceleration_position_gradient`;
- an `srp=` argument on `dynamics_jacobian_a_matrix`, plus one wrapper term.

There is no legacy switch and no production FD. The gradient is analytic.

## 13. What Changed in Production

- **`src/lunar_od/srp.py`:** +`_illumination_and_gradient` and +`srp_acceleration_position_gradient`.
- **`src/lunar_od/dynamics.py`:**
  - `dynamics_jacobian_a_matrix(srp=None)`;
  - the harmonic and K-sensitivity paths pass `srp`;
  - the 42-state wrapper adds the Φ term;
  - four docstrings updated.

## 14. What Did NOT Change in Production

- the SRP acceleration and kernel (bitwise, 2000 random states);
- the shadow law;
- K semantics;
- ∂a/∂K;
- the 6-state truth (bitwise);
- state ordering;
- F-order STM packing;
- `accelerated.py`;
- measurement models;
- `two_way_range.py`;
- estimators, visibility and tracking geometry.

## 15. Full-Sun Gradient Qualification

`studies/phase17/results/phase17_f1_local_gradient_oracle.json`. Two Sun directions × two altitudes, sunward and anti-sunward:
- the best-resolved error is 3.6e-10 to 1.8e-7;
- the analytic value equals the closed form K(P/d)(I−3uuᵀ) to 1e-12;
- `FULL_SUN_GRADIENT_GATE = PASS`.

## 16. Umbra Qualification

Umbra-interior points give an exact zero gradient, and the production FD is exactly zero there as well.
`UMBRA_GRADIENT_GATE = PASS`.

## 17. Penumbra Gradient Qualification

The penumbra points are ν ≈ 0.1 / 0.25 / 0.5 / 0.75 / 0.9, plus the edge neighbourhoods ν = 0.005 / 0.995.

**Oracles:**
- central FD of the production acceleration at 2nd and 4th order, over the predeclared 1e-2 … 1e4 m sweep;
- a difference-free integral oracle: ∫_{−L}^{L} G e ds (32-point Gauss–Legendre) = a(r+Le) − a(r−Le), for
  L = 50 / 200 m, on segments that stay inside one regime.

**Results:**
- interior points: 3e-9 to 6e-8;
- edge points: 2.4e-8 to 1.5e-7;
- ∇ν against production ν: ≤ 1.5e-7;
- `PENUMBRA_GRADIENT_GATE = PASS`.

**The harness was corrected twice after its first run.** Both corrections improve the oracle only; the 1e-6 gate did
not change.
1. The first "plateau" test compared error magnitudes, which is ill-posed in the roundoff regime.
2. At ν = 0.005 / 0.995 the FD oracle is roundoff-limited: the error falls ∝ 1/h all the way to ~1e-6, because
   production ν carries acos-cancellation roundoff. The integral oracle removes that floor.

## 18. Shadow Boundary Behaviour

`boundary_characterisation` in the local-oracle artifact:
- **Finiteness.** At the exact external and internal contacts the analytic gradient is finite.
- **Continuity.** It is about 100× smaller than in the interior: ∇ν → 0 at tangency, so it is continuous.
- **FD at the contacts.** FD straddling a contact disagrees by 20–80%, which is expected at a non-C² point and is
  not a gate.
- **ν consistency.** The gradient's ν equals production ν at every point.
- `SHADOW_BOUNDARY_GATE = PASS`.

Observation, OPTIONAL_CLEANUP and out of scope: at exact external tangency, production ν can come out as 1 + 1.8e-8
because of lens-area cancellation. It has no effect.

## 19. K=0 / No-Shadow Controls

- K = 0 gives exactly zero, and so does an inactive configuration.
- NO_SHADOW gives exactly the full-sun tensor, with no illumination term.
- `K_ZERO_CONTROL = NO_SHADOW_CONTROL = PASS`.

## 20. Old-Approximation Negative Control

The pre-F1 variational equations are replicated for analysis only: the same trajectory RHS, with A lacking the SRP
gradient. On the HARDEN-D05 penumbra-rich case:

| oracle | new model | old model |
|---|---:|---:|
| adaptive (best resolved) | 9.9e-8 | 1.2e-6 (12×) |
| fixed-step RK4 | 2.2e-8 | 3.4e-6 (150×) |
| S_K, canonical 15 orbits | 8.7e-7 | 3.6e-5 |

`OLD_VARIATIONAL_APPROXIMATION_FAILURE_REPRODUCED = YES`. The permanent test does the same over one eclipse orbit.

## 21. STM Independent Oracle

`studies/phase17/results/phase17_f1_stm_oracle.json`. Setup: canonical x0 and epoch, 4 orbits, 90 s grid, as in HARDEN-D05.

**Oracle:** 6-state production re-propagation, all six columns, h_pos ∈ {1, 10, 100} m and h_vel ∈ {1e-3, 1e-2,
1e-1} m/s. Position and velocity rows are both scored.

**Results (worst best-resolved error over the six columns):**

| variant | error |
|---|---:|
| penumbra (science) | **9.9e-8** |
| NO_SHADOW | 3.8e-8 |
| K = 0 | 1.9e-8 |

`F1_STM_GATE = PASS`.

## 22. STM Oracle Floor / Tolerance Study

Worst best-step error by rtol, for the penumbra variant:

| rtol | new model | old model |
|---:|---:|---:|
| 1e-13 | 1.56e-6 | 2.9e-6 |
| 1e-14 | 3.2e-7 | 3.0e-6 |
| 3e-15 | 9.9e-8 | 1.2e-6 |

The new model converges towards the controls' floor, 3.8e-8. The old model stalls at its model error.

The fixed-step RK4 cross-check at dt = 2 s and 1 s gives 1.5e-8 and 2.2e-8.

## 23. K-Sensitivity Independent Oracle

`studies/phase17/results/phase17_f1_k_sensitivity_oracle.json`. Cases: CANON 4 orbits, FULL_SUN 4 orbits (D2
O2_FULL_SUN_HIGH_BETA) and CANON 15 orbits.

**The adaptive K-FD is noise-limited.**
- The error goes as ∝ 1/ΔK and ∝ rtol.
- A K perturbation moves the orbit by only a few metres, because ΔK ≤ 0.01 when K − ΔK ≥ 0.
- The adaptive step pattern of the ± runs leaves about 5e-5 m of noise.

**Fixed-step RK4 oracle.** It uses the production force, with identical steps for every run, at dt ∈ {2, 1} s. Best
resolved:

| case | new model | old model |
|---|---:|---:|
| CANON 4 orbits | 7.1e-7 | 1.8e-6 |
| FULL_SUN 4 orbits | 1.3e-7 | 1.3e-7 (no penumbra: identical) |
| CANON 15 orbits | 8.7e-7 | 3.6e-5 |

The remaining ~1e-6 scatter is the oracle's noise/(S_K ΔK) floor. `F1_K_SENSITIVITY_GATE = PASS` (8.7e-7).

## 24. Range Derivative Requalification

`studies/phase17/results/phase17_f1_measurement_derivatives.json`. Four runs: new and old models (the old one via an in-process
analysis-only zeroed gradient), each at the science configuration (analytic rtol 1e-12, oracle 1e-13) and at
rtol 3e-15. The old model at the science configuration reproduces the HARDEN pre-F1 values exactly.

Range, worst state column:

| configuration | new | old |
|---|---:|---:|
| science | 1.2e-6 | 1.9e-6 |
| resolving (rtol 3e-15) | 8.9e-8 | 2.0e-6 |

K: 1.8e-4, which is the K-FD floor. `RANGE_DERIVATIVE_GATE = PASS`.

## 25. Counted-Doppler Derivative Requalification

| configuration | new | old |
|---|---:|---:|
| science | 9.4e-7 | 1.3e-6 |
| resolving (rtol 3e-15) | 8.5e-7 | 1.3e-6 |

K: 2.2e-4 → 8.7e-6 at the resolving tolerance. `DOPPLER_DERIVATIVE_GATE = PASS`.

## 26. ΔDOR Derivative Requalification

| configuration | new | old |
|---|---:|---:|
| science | 9.6e-7 | 1.2e-6 |
| resolving (rtol 3e-15) | 8.7e-8 | 1.3e-6 |

**The predeclared rule tripped once.** The rule was "no column worse than 2× pre-F1", and ΔDOR vy0 went 2.8e-7 →
9.6e-7 at the science configuration. Both values sit at that configuration's oracle floor. At the resolving tolerance
every new column is better than every old one (3.7e-8 to 8.7e-8, versus 1.9e-7 to 1.3e-6).

The trigger is reported, not hidden. It is not a regression. `DDOR_DERIVATIVE_GATE = PASS`.

## 27. Optical Derivative Requalification

| configuration | new | old |
|---|---:|---:|
| science | 3.5e-6 | 4.2e-6 |
| resolving (rtol 3e-15) | 1.1e-6 | 3.0e-6 |

K: 9.1e-6 versus 6.2e-6, both at the K-FD noise floor. `OPTICAL_DERIVATIVE_GATE = PASS`.

## 28. State-Truth / Force Parity

`studies/phase17/results/phase17_f1_state_parity.json`. Captured before any edit and compared after.

**Physical force: bitwise equal.**
- the 6-state truth (4 orbits);
- the SRP acceleration and kernel at 2000 random states.

`PHYSICAL_FORCE_CHANGED = NO`.

**Augmented-integrator numerical parity.** The state part of the coupled 42/48-state integrations moves because the
STM shares the adaptive step control:

| path | shift at rtol 1e-12 |
|---|---:|
| 48-state | 2.3e-5 m |
| 42-state | 1–2 cm |

This converges with rtol. At 1e-13 the 42- and 48-state errors against a tight reference are 3.9e-6 m, and Φ₄₂
agrees with Φ₄₈ to 2e-10. It is the same size class as the truth propagator's own tolerance error (9.7 cm at 1e-11,
5.5 mm at 1e-12).

## 29. Accelerated / Reference Parity

`ACCELERATED_VARIATIONAL_PATH_STATUS = NOT_APPLICABLE`. `accelerated.py` contains no SRP code; the SRP variational
term is added in `dynamics.py` for both the numba and the reference 42-state RHS. A permanent test requires both to
match the 48-state Φ to 1e-8.

## 30. Canonical Phase-17 Materiality

`studies/phase17/results/phase17_f1_science_materiality.json`. The same script ran pre (at the entering HEAD) and post.

| quantity | pre | post |
|---|---:|---:|
| range f_perp / θ_K | 0.0915076 / 5.25034° | 0.0915024 / 5.25005° |
| range I(K\|x) / σ_K/K | 154974 / 25.4021% | 154972 / 25.4023% |
| GC 5 nrad f_perp / θ_K | 0.172351 / 9.92453° | 0.172343 / 9.92407° |
| GC σ_K/K | 7.45652% | 7.45653% |
| gain | +0.080843 | +0.080841 |
| b_K norm (range / combined) | 4302.02 / 7781.26 | 4302.24 / 7781.62 |
| weakest design mode K component | 1.000 | 1.000 |

The design singular values are stored in the artifact. The normal-matrix rank is never used as a gate.

## 31. D2 Primary-Grid Requalification

All 18 cells (9 geometries × {GC, GM}, 5 nrad) were requalified.
- **Class changes: none.** Both the relative and the absolute class are preserved in every cell.
- **Largest movement:** O5 GM, Δ −0.081364 → −0.081375.
- E3 GM has no dual-visible rows, before and after.

## 32. GEO Materiality Spot Check / Escalation

The five cases were frozen before implementation (bfc98e4):
- G0_consistent (canonical);
- G4_net_Madrid (minimum f_perp);
- G3_i26.7_bcan (maximum f_perp);
- G1_b40 (closest to a threshold, margin 5.1e-4);
- G1_b71 (penumbra-rich).

| result | value |
|---|---|
| class changes | none |
| max \|Δd\| | 1.8e-5 |
| tightest escalation trigger | 2.5e-4 (G1_b40) |
| max \|Δf\| | 1.3e-5 (trigger 0.01) |

`GEO_ESCALATED_TO_FULL_GRID = NO`.

## 33. High-Fidelity Dynamics Requalification

The D2 fidelity propagation (GL1800F 50×50, Earth and Sun third body, SRP, harmonic STM) uses
`propagate_state_with_k_sensitivity`, so it shares the corrected A. The same declared cases were re-run: CANON GC,
O2 GC, and O5 GM (the rule-selected case).

- **Classes:** all preserved.
- **Largest change:** fidelity Δ, O5 GM −0.111037 → −0.111050.

`F1_HIGH_FIDELITY_REQUALIFICATION = PRESERVES_CLASS` (`studies/phase17/results/phase17_f1_high_fidelity.json`).

## 34. Protection / Provenance Update

`PHASE17_F1_REFROZEN_CURRENT_TREE_SHA256`, in `examples/r2_measurement_fidelity_validation.py`, commit 434b9c3,
made after qualification:
- **old:** b369899a;
- **new:** c6e7975c, which equals `git cat-file --filters 9a0e786:lunar_od/dynamics.py`;
- **authorisation:** the F1 brief;
- **documented:** the exact reason and the function scope.

The HARDEN, closure and R2-historical tables are untouched. `expected_current_tree_sha256` resolves newest-first.

The R2 suite passes with 81 tests. A permanent test checks that the re-freeze is additive.
`F1_PROTECTED_CHANGE_AUTHORIZATION = USED`.

## 35. Permanent Regression Tests

`tests/test_f1_srp_variational_gradient.py` has 15 tests:
- penumbra and edge gradients via the integral oracle;
- full-sun closed form and FD;
- no illumination term in NO_SHADOW;
- K = 0, umbra and inactive give exact zero;
- linearity in K and finiteness at the contacts;
- the exact A block with no velocity terms;
- the state-only path never evaluates the gradient;
- STM against fixed-step RK4, with the old model failing by ≥ 10×;
- S_K against RK4;
- fast/reference 42-state parity;
- an additive re-freeze.

The HARDEN root-cause test now pins the closure.

## 36. Relevant Slow Tests

None of the 29 entering skips use SRP. The opt-in slow regressions exercise STM/dynamics paths, so all of them were
run with `LUNAR_OD_RUN_SLOW_TESTS=1`:
- `test_filters.py` (11 slow tests);
- `test_scenarios.py` (9);
- `test_r1_r4_long_arc_qualification.py` (1).

Result: **91 passed, 0 skipped**. The other 8 skips need data that is not present (GRGM660PRIM, the Phase 13G store).

## 37. Full Suite

Normal run: `python -m pytest -o addopts="" -q -rs`, from this worktree.

| collected | passed | failed | errors | skipped | subtests |
|---:|---:|---:|---:|---:|---:|
| 1406 | **1377** | **0** | **0** | 29 | 146 passed |

**Skips.** They are the same 29 as at entry:
- 20 opt-in slow regressions and 1 slow slice, all run separately (§36) and passing;
- 8 skipped for data that is not present.

**Count.** The collected count rose by 15, all from the new F1 tests.

**Intermediate failures.** An intermediate full run found two FXR tests pinning historical values at 1e-9. Those values
were computed with the pre-F1 STM. Each test now reproduces the historical value bitwise under an analysis-only pre-F1
model and pins the post-F1 authoritative value separately. They were fixed and re-run.

`NEW_SCIENTIFIC_REGRESSIONS = 0`, `UNKNOWN_NONPASSES = 0`.

## 38. Final Diff Review

`git diff 1025f4a..HEAD`: 28 files.

**Production (2 files, in scope):**
- `srp.py`: the gradient functions only;
- `dynamics.py`: the A-matrix `srp=` argument, the 42-state wrapper term, two `srp=` pass-throughs, docstrings.

**Protection:** an additive re-freeze block in `examples/r2_measurement_fidelity_validation.py`.

**Analysis:**
- 5 new F1 harnesses;
- `--f1` and `--out` modes on two existing harnesses, which never overwrite HARDEN or D2 evidence (checked by
  sha256).

**Tests:** 1 new file, 2 targeted updates.

**Documentation and artifacts:**
- the authoritative status page, the HARDEN ledger (D05) and the contract inventory;
- 10 `phase17_f1_*` artifacts.

No unrelated change. Scratch files were removed.

`FINAL_DIFF_REVIEW = PASS`.

## 39. Academic Interpretation

The STM is the linearisation of the actual force model along the nominal trajectory. If a force's position dependence
is omitted, Φ and S_K evolve under a slightly different local dynamical system from the trajectory they accompany.

In full light the SRP gradient is tiny: ~K P/d ≈ 3e-19 s⁻². In penumbra ν changes by O(1) across a band a few
kilometres wide, so the gradient is about 10⁷ times larger. Each crossing kicks the linearisation, and the kicks
accumulate over repeated eclipses.

A correct ∂f/∂x matters for measurement partials, covariance, observability metrics and solve-for sensitivities even
though the nominal truth trajectory does not change.

This is a numerical and model-consistency qualification of the repository's model (REPOSITORY_RESULT). It is not a
new result about lunar physics. No external literature was needed: the derivation follows directly from the
implemented force.

## 40. Operational / OD Interpretation

A trajectory can be right while its STM is slightly wrong. That error never shows in truth-state residuals, but it
enters partials, formal covariance, weak-mode orientation and parameter correlations. It is most relevant for arcs with
many penumbra passages.

Here the effect on Phase-17 information metrics is at the 1e-5 relative level and changes no conclusion. That is a
statement about these synthetic campaigns, not a mission-performance claim.

## 41. Remaining Limitations

- The simplified campaign dynamics stay as they are.
- The shadow law is C¹ at the contacts, so any variational integrator loses order there. Integrations resolve to
  ~1e-7 only at tight tolerance (§22).
- The K-FD oracle floor is ~1e-6 (§23).

## 42. Remaining Technical Debt

- **OPTIONAL_CLEANUP:** production ν slightly above 1 at exact external tangency (§18).
- **RECOMMENDED_FOLLOWUP:** a K-derivative oracle with larger signal, if a future phase needs S_K below ~1e-6.
- Unchanged HARDEN follow-ups D14 and D18–D23.

## 43. Remaining Unknowns

**None material.** Every discrepancy measured in F1 has a demonstrated cause:
- the adaptive-oracle noise floors, from ΔK and rtol studies and the fixed-step oracle;
- the augmented-integrator state shift, from rtol convergence;
- the ΔDOR 2× trigger, resolved at a tight tolerance.

## 44. What This Phase Actually Established

- The SRP position gradient is derived and implemented exactly for the current model.
- It is independently qualified in every regime.
- The STM and S_K match re-propagation to the oracle floor.
- The pre-F1 model's failure is reproduced.
- The force and the truth are bitwise unchanged.
- The Phase-17 classes are all preserved, with authoritative digits updated.

## 45. What This Phase Did NOT Establish

- No new science.
- No change of force fidelity.
- No mission-level covariance claim.
- No ΔDOR tracking design.
- No K parameterisation.
- No qualification of variational accuracy below ~1e-7 (the oracle floor).

## 46. Final Gate

Every strict PASS criterion is met:
- the entering state was verified;
- the model was inspected and the complete gradient derived;
- the local gradient passes in every regime and at the boundaries;
- the K = 0 and NO_SHADOW controls pass;
- the old approximation's failure is reproduced;
- the STM and S_K oracles pass;
- all four derivative chains pass;
- the force and the truth are bitwise unchanged;
- the accelerated path is resolved (not applicable);
- provenance is additive;
- canonical, D2 and GEO materiality were measured;
- high fidelity is resolved;
- targeted and slow tests pass;
- the full suite has no unknown non-passes;
- the diff is clean, scratch files are removed and the documentation is current.

**`PHASE17_F1_GATE = PASS`**, **`SRP_VARIATIONAL_MODEL_STATUS = QUALIFIED`**.

Decision-tree outcome: Case A, `SRP_VARIATIONAL_POSITION_GRADIENT_QUALIFIED`.
Materiality: `NUMERICAL_BASELINES_UPDATED_CLASSES_PRESERVED`.

## 47. Whether ΔDOR Tracking Design Is Safe To Start

**Yes. `TRACKING_DESIGN_SAFE_TO_START = YES`.**
- F1 passed.
- No Phase-17 classification changed.
- The authoritative status page carries the corrected digits.

A tracking-design phase should cite `studies/phase17/reports/phase17_authoritative_status.md`, which gives range 0.091502 and combined
0.172343.

## 48. Exact Next Action

**Stop (F1 hard stop).** The recommended next scientific task is the already-planned, predeclared ΔDOR
tracking-design phase, under its own brief. It is not started here.
