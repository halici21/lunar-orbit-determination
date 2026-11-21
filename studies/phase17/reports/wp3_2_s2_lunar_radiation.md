# F3 WP3.2 S2 — Lunar albedo and thermal-IR radiation pressure as an opt-in production force term (REQ-DYN-007, DYN-07)

- **Date:** 2026-11-21 (code and gate runs 2026-11-11)
- **Predeclaration:** `studies/phase17/configs/wp3_2_s2_lunar_radiation_predeclaration.json`, committed in `da5a4a0`
  (2026-11-10) before any S2 code, test or number.
- **Amendment:** one, after the L_G6 result: the exploratory gate set L_G6-A, recorded in
  `docs/assurance/amendment-log.md` (2026-11-11) and committed with its script in `0d554c4` before it ran.
- **Code:**
  - S2a (`cdb9802`): the new module `src/lunar_od/lunar_radiation.py`, and the term in the propagators of
    `src/lunar_od/dynamics.py`;
  - S2b (`f539572`): its transport through `src/lunar_od/propagation_contract.py` and `src/lunar_od/estimators.py`.
- **Tests:**
  - `tests/test_wp3_2_lunar_radiation.py`: L_G1(a) and L_G2–L_G8;
  - `tests/test_wp3_2_lunar_radiation_transport.py`: LB_G1–LB_G3 and the re-freeze record.
- **Results:**
  - `results/wp3_2_s2_gate_values.json`: L_G2–L_G7 at `4d48178` (src as `cdb9802`, clean tree; identical to the
    uncommitted-tree run), LB_G3 added at `f539572` (clean). Script `scripts/wp3_2_s2_gate_values.py`;
  - `results/wp3_2_s2_default_identity.json` (L_G1(b)) and its four inputs in `results/wp3_2_s2_default_identity/`;
    the two intermediate post-S2a runs are kept there as well;
  - `results/wp3_2_s2_lg6_floor_diagnostic.json` (exploratory L_G6-A, section 5; run at `3fdb892`, clean).
  - LB_G3 was re-run on 2026-11-21 on the clean tree at `f539572`: identical to the recorded values.
- **Production defaults changed:** none. With `lunar_radiation` omitted or `None`, all 50 recorded configurations
  (the S1 set, plain and with the S1 tide on) are byte-identical before and after the change (L_G1(b)).

## 0. Final status

```
S2 (as predeclared)   = NOT QUALIFIED: L_G6 FAIL (3 of 4 cases), and L_G9 through it; every other gate PASS
  L_G1 identity       = (a) the wrapper returns the RHS itself when off; (b) 50/50 configurations byte-identical
  L_G2 C6 parity      = 1.4e-15 (numba), 4.6e-16 (Python fallback) (<= 1e-12)
  L_G3 accuracy       = closed forms <= 1.8e-13 (<= 1e-10); adaptive reference worst 1.97e-4 at 1000 km,
                        Sun 90 deg from nadir (<= 3e-4, the materiality requirement)
  L_G4 gradient       = complex step 9.3e-15 (<= 1e-10)
  L_G5 variational    = STM 3.6e-8 (<= 1e-7); control without the LRP gradient 2.6e-6 (73x);
                        K_SRP sensitivity 3.4e-8 (<= 1e-6)
  L_G6 path agreement = FAIL: 6- vs 42/48-state positions 1.56e-5, 1.03e-5, 1.06e-5 m (> 1e-5 m) in three cases,
                        7.2e-6 m in the fourth; Phi <= 1.9e-9 (<= 1e-8) in all four
  L_G7, L_G8          = shadow events and fail-closed validation PASS
  LB_G1-LB_G3         = options, transport through every propagation of the six public estimators, recovery
                        7.7e-6 m (3.4e4x below the 0.262 m bias without LRP)
  L_G9 regression     = FAIL only through L_G6: 2642 tests, 3 failed (the L_G6 cases), 0 errors, 29 skipped;
                        check_repository passes
L_G6-A (exploratory)  = A1, A2, A3 PASS: the excess is the C6 rule's own integration floor, not a path defect
Decision              = the owner's (amendment log 2026-11-11: "A1-A3 pass -> floor documented as a limit of
                        the C6 rule and the S2 decision is left to the owner")
REQ-DYN-007           = still partial: S3 EARTH_SHADOW, S4 SRP_LIGHT_TIME, S5 hold-out replication
```

## 1. What changed

- **`lunar_radiation.py` (new).** `LunarRadiationOptions` is frozen and keyword-only, with the C6 defaults:
  albedo 0.12, emissivity 0.95, night temperature 100 K, 60 × 72 cap nodes, radius `R_MOON_M`, and a provenance
  string. `lunar_radiation_acceleration_per_k`, `lunar_radiation_gradient_per_k` and
  `lunar_radiation_acceleration_and_gradient_per_k` evaluate the C6 model unchanged (a uniform Lambertian sphere
  with an equilibrium dayside and a 100 K night floor) as K-independent kernels. The kernels are numba loops
  (`_optional_njit`, no fastmath) with the forward-mode derivative of the discrete quadrature sum; without numba the
  same code runs as Python. Invalid options raise (L_G8).
- **`dynamics.py`.** `_with_lunar_radiation_acceleration` wraps a built RHS after the tide wrapper and returns it
  unchanged when LRP is off. With LRP on, K_LRP = K_SRP, so active SRP is required (ValueError otherwise). It adds:
  - `K g` to the acceleration;
  - in the 42-state RHS, `K G @ Phi[0:3, :]` to rows 3:6 of `Phi_dot`;
  - in the 48-state RHS, also `K G @ s_K[0:3] + g` to `s_K_dot[3:6]` (`g` is LRP's part of df/dK).

  The 6-state path never evaluates the gradient. The keyword `lunar_radiation=None` is appended wherever S1 added
  `lunar_tide`: `propagate_state`, `shadow_contact_times`, both paths of `propagate_augmented_state`,
  `propagate_state_with_k_sensitivity`, `propagate_truth_with_ephemeris`, `_build_state_rhs` and the harmonic path.
  The event-segmented pre-pass integrates the same LRP-on RHS (L_G7).
- **Transport (S2b).** `model_options` and `propagator_options` add the key `lunar_radiation` only when a
  `LunarRadiationOptions` is given (TypeError for anything else that is not `None`). The 15 `estimators.py`
  functions that carry `lunar_tide` gain `lunar_radiation=None` and forward it to every propagation: nominal,
  candidate, rescoring, Jacobian, K sensitivity and posterior.
- **Protected files.** `dynamics.py` and `estimators.py` are re-frozen additively in the new table
  `F3_WP3_2_S2_REFROZEN_CURRENT_TREE_SHA256` of `examples/r2_measurement_fidelity_validation.py`:

  | file | previous identity | new identity |
  |---|---|---|
  | `dynamics.py` | `ae0f3d0d` (WP3.2 S1) | `eefbb63b` |
  | `estimators.py` | `0ed77589` (WP3.2 S1) | `d0e69634` |

  The history tables are unchanged; the history-chain tests (`test_f1_srp_variational_gradient.py`, the S1
  transport test) keep S1's identity as history.

Usage:

```python
from lunar_od.lunar_radiation import LunarRadiationOptions
from lunar_od.srp import SRPOptions
srp = SRPOptions(...)                                   # must be enabled: K_LRP = K_SRP
x = propagate_state(t, x0, mu_moon, mu_earth, mu_sun, earth_pos, sun_pos, srp=srp,
                    lunar_radiation=LunarRadiationOptions())
x_hat, status, stats = estimate_position_srif(..., srp=srp, lunar_radiation=LunarRadiationOptions())
```

## 2. Size and cost of the term

- On the L_G5/L_G6 test orbit (100 km polar, K = 0.05 m²/kg, 6 h) LRP moves the spacecraft by 1.33 m.
- L_G7 (1 day, Sun in the orbit plane, K = 0.05): 46 of 50 shadow-contact cuts move by more than 1 ms, at most by
  16.4 ms.
- Cost per kernel call (numba, information): 228 µs for the value, 522 µs with the gradient. The predeclaration
  expected 100–300 µs and 3–5× that with the gradient; the value is in range and the gradient costs 2.3×.

## 3. Gates

The thresholds are the predeclared ones; the values are as computed.

| gate | threshold | computed | verdict |
|---|---|---|---|
| L_G1(a) identity when off | `rhs` itself for 6, 42, 48 states | object identity | PASS |
| L_G1(b) default paths | 50 configurations byte-identical, pre run twice | 50/50 (pre `0d537f2` ×2; post `f539572`, omitted and `None`, src clean). The four tide-on range-rate solves hash S1's fail-closed GM error, as expected | PASS |
| L_G2 C6 parity | 1e-12 of \|a_C6\| | numba 1.41e-15; Python fallback 4.60e-16 | PASS |
| L_G3(a)(b) closed forms | 1e-10 relative; transverse 1e-12 | ≤ 1.77e-13 (subsolar radial, 20 km); transverse ≤ 2.5e-15; the adaptive reference reproduces both to ≤ 1.8e-13 | PASS |
| L_G3(c) adaptive reference, 21 geometries | ≤ 3e-4 | worst 1.97e-4 (1000 km, 90°); then 1.52e-4 (100 km, 100°); 12 of 21 ≤ 1e-14 | PASS |
| L_G4 gradient | complex step 1e-10; reference vs C6 1e-12; py_func 1e-12 | 9.26e-15; 1.39e-15; 2.89e-16 | PASS |
| L_G5(a) STM against RK4 re-propagation | ≤ 1e-7; control ≥ 10× | 3.61e-8; control 2.63e-6 (73×) | PASS |
| L_G5(b) K_SRP sensitivity | ≤ 1e-6 | 3.36e-8 | PASS |
| L_G5(c) RHS structure | 4 eps, nothing else moves; no gradient on the 6-state path | as stated | PASS |
| L_G6 6/42/48-state paths | positions 1e-5 m; Φ 1e-8 of max\|Φ\| | see the table below | **FAIL** |
| L_G7 shadow events with LRP | contact checks pass; a cut moves > 1e-3 s; the control raises | 50 cuts, max shift 0.0164 s (46 > 1e-3 s); control raised at 7822.9 s (1.39e-3 s from the contact) | PASS |
| L_G8 fail closed | each predeclared invalid input raises, including LRP with `srp=None` and with disabled SRP | all raise | PASS |
| LB_G1 options dicts | S1 dicts without LRP; only `lunar_radiation` added with it; TypeError otherwise | as stated, every tide/SRP/profile combination | PASS |
| LB_G2 transport | every captured propagation carries the same object; none has the key when omitted | six public estimators (two-way with and without K solve-for) | PASS |
| LB_G3 recovery | \|dr0\| < 1e-3 m, \|dv0\| < 1e-6 m/s; without LRP ≥ 100× | BLS-LM 7.68e-6 m, 7.72e-9 m/s; SRIF 7.53e-6 m, 7.70e-9 m/s; without LRP 0.262 m (ratio 3.4e4) | PASS |
| L_G9 regression | 0 failures, 0 errors; skips unchanged apart from the new tests' own; `check_repository` passes; re-freeze additive | at `5614dec` (src as `f539572`, before the xfail marking), 2026-11-21: 2642 tests, 2606 passed, **3 failed (the three L_G6 cases, nothing else)**, 0 errors, 29 skipped, 4 xfailed (existing); `check_repository` passes; one new table, history unchanged, history-chain tests extended. The collection at S1's `10f2366` gives 2547 tests today, file for file the same as here apart from the two new S2 files (95 tests); S1's recorded 2695/33 skipped was counted in a different environment and its skip set was not re-run | **FAIL** (only through L_G6) |

L_G6 as computed (maximum position difference from the 6-state propagation over the 6 h arc, metres; Φ relative to
max\|Φ\| against the 48-state run):

| case | 42-state | 48-state | 42-state reference | Φ (worst) | verdict |
|---|---|---|---|---|---|
| J2, no tide | 1.52e-5 | **1.56e-5** | 2.25e-5 | 5.0e-10 | FAIL |
| J2, tide | 4.77e-6 | **1.03e-5** | 5.80e-6 | 1.3e-9 | FAIL |
| zonal harmonics, no tide | 6.19e-6 | 7.19e-6 | — | 1.9e-9 | PASS |
| zonal harmonics, tide | 8.47e-6 | **1.06e-5** | — | 1.1e-9 | FAIL |

The Φ half passes everywhere, by a factor ≥ 5. The position half fails by up to 1.56× in three cases.

## 4. Expectations against outcomes

| expectation (predeclared, not gates) | outcome |
|---|---|
| L_G2 ~1e-15 | 1.4e-15: met |
| L_G3(a)(b) ~1e-13 | ≤ 1.8e-13: met |
| L_G3(c) ~1e-6..1e-4, worst where a radiance kink crosses the cap at 20 km, real chance of exceeding 3e-4 | worst 1.97e-4 at 1000 km, 90°: within the gate; the worst case is at the terminator, as expected, but at the largest altitude, not at 20 km (at 20 km the worst is 3.2e-5) |
| L_G4 ~1e-13 | 9.3e-15: better |
| L_G5(a) ~2e-8..5e-8, control ~1e-6..4e-6 | 3.6e-8, 2.6e-6: met |
| L_G7 several ms | max 16.4 ms: met |
| kernel 100–300 µs, gradient 3–5× | 228 µs, 2.3×: value met, gradient cheaper |
| L_G6 (no stated expectation; the S1 analogue passed at ≤ 9.8e-7 m) | **FAIL** at up to 1.56e-5 m |

## 5. L_G6 and the exploratory L_G6-A

**Hypothesis, recorded with the failure.** The C6 discrete rule is not smooth in position: a quadrature node crossing
the terminator or the thermal floor puts a kink into the RHS, and the nadir frame's seed axis switches at
\|z_x\| = 0.9. The 6-, 42- and 48-state integrations take different adaptive step sequences, so they accumulate
different integration error across these kinks. The wiring itself would be correct. The information measurement in
the gate values supports this: the 6-state path's self floor (rtol 1e-13 against 3e-14) is 7.8e-7 m without LRP and
8.5e-6 m with it, that is, already close to the 1e-5 m threshold.

**L_G6-A**, written in the amendment log and committed with its script (`0d554c4`) before it ran, on the four L_G6
cases (`scripts/wp3_2_s2_lg6_floor_diagnostic.py`, run at `3fdb892`):

- **A1.** The module's two functions are replaced by a smooth surrogate,
  g = C0 (R/\|r\|)² r̂ (1 + 0.5 ŝ·r̂) with C0 = 1e-6 N/m², and its exact gradient, and pushed through the same
  production wrapper and the three propagators: positions agree to the predeclared 1e-5 m, Φ to 1e-8, and the
  effect exceeds 0.1 m.
- **A2.** In every case the C6 6-state self floor is at least 3× the surrogate's.
- **A3.** With C6, the 42/48 − 6 position difference stays within 4× the same case's C6 self floor, Φ within 1e-8.

| case | surrogate: 42 / 48 vs 6 (m) | surrogate floor (m) | C6 floor (m) | C6: 48 vs 6 (m) | A1 | A2 | A3 |
|---|---|---|---|---|---|---|---|
| J2, no tide | 7.9e-7 / 7.7e-7 | 6.5e-7 | 7.1e-6 | 1.56e-5 (2.2× floor) | PASS | PASS (11×) | PASS |
| J2, tide | 8.8e-7 / 9.3e-7 | 8.0e-7 | 1.6e-5 | 1.03e-5 (0.6×) | PASS | PASS (20×) | PASS |
| harmonics, no tide | 8.0e-7 / 9.4e-7 | 8.0e-7 | 8.9e-6 | 7.2e-6 (0.8×) | PASS | PASS (11×) | PASS |
| harmonics, tide | 6.9e-7 / 7.2e-7 | 6.2e-7 | 9.4e-6 | 1.06e-5 (1.1×) | PASS | PASS (15×) | PASS |

The surrogate's effect is 2.06 m and its Φ agreement ~1e-13 in every case. Information only: with 120 × 144 nodes
the two J2 cases give 2.9e-6 and 5.5e-6 m between the paths, and floors of 2.5e-6 and 6.5e-6 m.

**Reading.**

- The wrapper and the three propagators carry a smooth LRP-like term consistently, to the S1 level (< 1e-6 m).
- The C6 rule raises the integration floor of each path tenfold or more, and the path differences are of the same
  size as that floor. This is a property of the C6 quadrature (the term WP3.1 measured), not of the S2 wiring.
- 1.6e-5 m in 6 h is about 1e-5 of the LRP effect itself (1.33 m), and far below the OD threshold of WP3.1's
  materiality analysis. It matters for the 1e-5 m path-consistency gate, not for an OD solution.
- A kink-aware or finer quadrature would lower the floor, but it would change the term WP3.1 measured, which the
  predeclaration put out of scope.

**What this means.** The predeclared L_G6 stays FAIL and S2 is NOT QUALIFIED as computed. L_G6-A is exploratory
(it was written after the L_G6 result). By the amendment log's rule, because A1–A3 pass, the floor is documented as a
limit of the C6 rule (assumptions and limits, item 15) and whether S2 is merged is the owner's decision.

## 6. Test-setup fixes before evaluation

None besides those S1 already made, which S2's tests reuse.

## 7. Not done and limits

- **Deferred (predeclared, as S1).** UKF, EKF and the fast sigma propagator; scenarios, `scenario_config`, the CLI
  and the desktop app; observability, `numerical_metadata` and the force contract. None of them can enable LRP
  (the parameter does not exist there), and they join with the scenario-level transport after S4.
- **Out of scope (predeclared).** The Earth's shadow on the lunar surface during eclipses and thermal lag (both lower
  the LRP in the umbral phases; WP3.1 notes the overestimate); albedo and temperature maps; a separate K for the IR;
  a faster or kink-aware quadrature.
- **Integration floor (L_G6).** With LRP on, path-to-path and tolerance-to-tolerance position differences of
  ~1e-5 m in 6 h are expected at rtol 1e-13 (section 5).
- **Validated geometry.** Synthetic Earth and Sun, a 100 km polar orbit, 6 h and 1 day arcs, noiseless position
  observations. OD materiality on the WP3.1 population is S5's hold-out replication.
- **REQ-DYN-007 stays partial** until S5.
- **After the merge with main (2026-11-21).** Main's F1 X-02 made the GRAIL/DE440 lunar J2 the default; it refuses
  the legacy `J2_MOON_UNNORMALIZED` that the S1 and S2 gates use. The four WP3.2 test modules therefore run on the
  legacy mean-pole model (`use_lunar_j2_model("iau_mean_pole_legacy")`, module-scoped autouse fixture, main's own
  pattern), which is the pre-X-02 physics bit for bit, so the gates test what was evaluated. The tide and LRP with
  the GRAIL/DE440 J2 default are not gated here. The recorded identity hashes and gate values belong to their
  commits on the WP3.2 branch, before the merge; the scripts are not re-run on the merged tree.

## 8. Agent decisions

Made under the owner's 2026-11-08 delegation and flagged for review (predeclaration `agent_decisions`):

1. The C6 model and quadrature unchanged, with its accuracy gated against the materiality requirement.
2. A separate `LunarRadiationOptions` and keyword instead of an `SRPOptions` field, so `asdict(SRPOptions)` and every
   SRP-enabled record stay unchanged.
3. K_LRP = K_SRP; LRP requires active SRP.
4. A numba loop kernel with a forward-mode gradient (the numpy kernel costs 3.7 ms per call).
5. K = 0.05 and 3× the S1 difference steps in the variational gates, chosen before any S2 number.
6. The additive re-freeze of `dynamics.py` and `estimators.py`.
7. The exploratory gate set L_G6-A after the L_G6 failure (section 5).
8. The three failing L_G6 cases are marked `xfail(strict=True)` in the test file after the result
   (amendment log, 2026-11-21): the predeclared assertion still runs unchanged and the test reports an unexpected
   pass if the paths ever agree within 1e-5 m. This keeps the full suite usable as a regression signal; it does not
   change the L_G6 verdict.

## 9. Reproduce

From the worktree root, with `NUMBA_CACHE_DIR` set:

```
.venv/Scripts/python -m pytest tests/test_wp3_2_lunar_radiation.py tests/test_wp3_2_lunar_radiation_transport.py -q -p no:cacheprovider
.venv/Scripts/python studies/phase17/scripts/wp3_2_s2_gate_values.py LABEL OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s2_gate_values.py --add-lb-g3 OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s2_lg6_floor_diagnostic.py OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s2_default_identity.py run {omit|none} OUT.json
.venv/Scripts/python studies/phase17/scripts/wp3_2_s2_default_identity.py compare A.json B.json [...] --out OUT.json
.venv/Scripts/python tools/check_repository.py
```
