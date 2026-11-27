# F3 WP3.2 S4 — SRP light-time package as an opt-in production force term (REQ-DYN-007, DYN-19)

- **Date:** 2026-11-27 (gate runs the same day)
- **Predeclaration:** `studies/phase17/configs/wp3_2_s4_srp_light_time_predeclaration.json`, committed in `cc7a454`
  before any S4 code, test or number. Two errata before any result (`e7e261d`: the LT_G2 band sampler is
  rho = R_MOON + q a_sun with q in [-3, 3]; the LT_G4 finite-difference stencil must not cross a contact), recorded
  in the predeclaration's `errata_before_any_result`.
- **Amendments** (`docs/assurance/amendment-log.md`):
  - before any value: the SPICE time string of the LT_G3(b) test (`2026-01-01 00:00:00 TDB`; the first form raised
    SpiceUNPARSEDTIME before computing anything);
  - after the results, exploratory: LT_G3(a), LT_G4(a) and LT_G4(d) FAIL, the FD-step diagnostic and the strict
    xfails (`1d2671a`).
- **Code:**
  - S4a/S4b (`f39844b`): the new module `src/lunar_od/srp_light_time.py`; the wrapper and the light-time lunar
    contacts in `src/lunar_od/dynamics.py`; the transport through `src/lunar_od/propagation_contract.py` and
    `src/lunar_od/estimators.py`;
  - re-freeze (`b320f6d`): new table `F3_WP3_2_S4_REFROZEN_CURRENT_TREE_SHA256` first in
    `expected_current_tree_sha256`, `docs/development/path-migration.json` history extended, the history-chain tests
    of F1, S1, S2 and S3 extended, a new S4 re-freeze test.
- **Tests:** `tests/test_wp3_2_srp_light_time.py` (LT_G1(a), LT_G2–LT_G8; failing first in `cbc553e`),
  `tests/test_wp3_2_srp_light_time_transport.py` (LTB_G1–LTB_G3 and the re-freeze record).
- **Results:**
  - `results/wp3_2_s4_gate_values.json`: LT_G2–LT_G7, LTB_G3 and cost, at `b320f6d`, clean `src`
    (script `scripts/wp3_2_s4_gate_values.py`);
  - `results/wp3_2_s4_default_identity.json` (LT_G1(b)) and its four inputs in `results/wp3_2_s4_default_identity/`:
    two runs before the change (`e7e261d`, `cbc553e`; clean `src`), the omit and `None` runs after it at `b320f6d`
    (clean `src`);
  - `results/wp3_2_s4_lt_g4_step_diagnostic.json` (exploratory, after the LT_G4 result; script
    `scripts/wp3_2_s4_lt_g4_step_diagnostic.py`, at `0afcdf1` with the S4 src committed).
- **Production defaults changed:** none. With `srp_light_time` omitted or `None`, all 83 recorded configurations
  are byte-identical before and after the change (LT_G1(b)).

## 0. Final status

```
S4 (as predeclared)    = NOT QUALIFIED: LT_G3(a) and LT_G4(a)/(d) FAIL; every other gate PASS
  LT_G1 identity       = (a) the wrapper returns the RHS itself when off; (b) 83/83 configurations byte-identical
  LT_G2 WP3.1 parity   = bit for bit, 0 mismatches in 2000 samples (1201 lit, 348 penumbral, 451 umbral)
  LT_G3 accuracy       = (a) FAIL: Lorentz force 2.30e-8 (<= 1e-7), precondition 1.7e-16 (<= 8.9e-16), but the
                             geometric-SRP negative control is at least 4.75e-7, only 21x (>= 100x required)
                         (b) PASS: Sun direction vs SPICE CN+S 4.07e-8 rad (<= 1.5e-7), control 8.9e-5 rad (2176x)
                         (c) PASS: occulter vs exact closest approach 0.034 m (<= 0.1 m)
  LT_G4 gradients      = (a) FAIL: h = 10 m FD 5.66e-4 (penumbra), 5.96e-6 (NO_SHADOW) (<= 1e-6)
                         (b) PASS: increment in full light, h = 1e6 m, 6.9e-7 (<= 1e-5)
                         (c) PASS: velocity gradient 3.4e-10 (<= 1e-6), exactly zero in umbra
                         (d) FAIL: composed with the Earth shadow, h = 100 m, 1.86e-5 (<= 1e-6); velocity 3.4e-10
  LT_G5 variational    = (a) PASS: STM 2.3e-8 (<= 1e-7); control without the package gradients 2.3e-8 (information)
                         (b) PASS: K sensitivity 4.1e-7 (<= 1e-6), control without dF/dK 4.0e-3 (9810x >= 10x)
                         (c) PASS: RHS structure
  LT_G6 path agreement = PASS in all six cases (table in section 3)
  LT_G7 shadow events  = PASS: 13 light-time cuts, 13 geometric; every cut within 3.0e-6 s of a light-time contact
                         and >= 45 ms from the geometric cuts; shifts -45.2 to -46.8 ms; the control raises;
                         with the Earth shadow 17 cuts (4 Earth)
  LT_G8 fail-closed    = PASS
  LTB_G1-LTB_G3        = options; transport through every propagation of the six public estimators; recovery
                         2.5e-7 m and 1.0e-10 m/s (BLS-LM), 2.5e-7 m (SRIF), against 7.8e-5 m without the package
                         (307x >= 100x)
  LT_G9 regression     = section 6
LT_G4 step diagnostic  = exploratory: the untouched production geometric gradient fails the same oracle at h = 10 m
(exploratory)            (8.4e-4); at the best step per sample the total error is <= 1e-6 in 326/333 (a) and
                         633/714 (d) samples, worst 6.1e-6 and 3.4e-6
Decision               = the owner's (reject_if: not offered for merge as qualified until a fix passes a recorded
                         gate set)
REQ-DYN-007            = still partial: S5 hold-out replication
```

## 1. What changed

- `SRPLightTimeOptions(*, ssb_velocities, provenance=...)` (keyword-only, validated) selects the package. Its
  `SSBVelocityTable` carries the Sun and Moon velocities relative to the solar-system barycentre on a time grid
  (PCHIP per component, `EphemerisCoverageError` outside the grid plus a 10 s margin);
  `sample_ssb_velocities(et0, t_s)` fills it from SPICE. The package needs an active `SRPOptions` (ValueError
  otherwise).
- The model is WP3.1's ray-crossing term, unchanged: the Sun at emission `r_emit = r_sun - v_sun |r_sc - r_sun|/c`,
  the Moon at the ray crossing `r_occ = -v_moon (r_sc . u)/c`, and the Poynting-Robertson form
  `a_LT = K P(d) nu_LT [(1 - v.u/c) u - v/c]` with `v = v_sc + v_moon - v_sun`. The increment added to the force
  model is `a_LT - a_SRP`; LT_G2 shows it equals `wp3_1_materiality_diagnostics.full_ray_crossing` bit for bit.
- Variational terms: the force depends on the velocity, so the 42-state block gains `G_r Phi[0:3] + G_v Phi[3:6]` in
  rows 3:6, with `G_v = -K P nu_LT (u u^T + I)/c`; the 48-state block gains `G_r s_K[0:3] + G_v s_K[3:6] + da/K`.
  The 6-state path never evaluates a gradient.
- With the Earth shadow (S3) on, the increment is multiplied by `nu_E` (total `nu_E a_LT`), and
  `G_r -> nu_E G_r + da (x) grad nu_E`. The wrapper is applied last, after the Earth-shadow wrapper, on every path.
- D2 shadow events: with `shadow_events` and the package on, the light-time lunar contact functions (Sun at
  `r_emit`, Moon at `r_occ`) replace the geometric pair in the pre-pass and in the contact check. The tolerances are
  unchanged.
- Every propagation entry point (`propagate_state`, `propagate_augmented_state`, the harmonics path,
  `propagate_state_with_k_sensitivity`, `propagate_truth_with_ephemeris`, `shadow_contact_times`) and the six public
  batch estimators take `srp_light_time`; `model_options`/`propagator_options` carry it only when given.
- Not transported (predeclared `out_of_scope`): UKF, EKF, scenario config, CLI, desktop, observability.
- Known model limit, kept from WP3.1 (agent decision 1): `v` is taken relative to the Sun's SSB motion, which counts
  the Sun's velocity twice in the aberration direction (at most 5.2e-8 rad). The LT_G3(b) information value
  `corrected_rad` (2.1e-9 rad after the predicted v_sun correction) confirms the size.

## 2. Size and cost of the term

- 6 h effect on the 6-state positions at K = 0.2: 9.1 mm on the TERM orbit (cases i-iii), 7.1 mm on the PLANE orbit
  with lunar eclipses (iv), 6.4 mm on the real E2 eclipse (v), 4.7 mm with the Earth shadow and events (vi). The
  predeclared expectation was 1-5 mm.
- Shadow contacts move by 45-47 ms against the geometric ones (the predeclared expectation was ~0.1 s; WP3.1's
  `ray_crossing_check` gives 0.112 s for a 100 km orbit, the LT_G7 orbit is different).
- OD impact (LTB_G3): solving without the package on light-time truth biases the epoch position by 7.8e-5 m and the
  velocity by 8.1e-8 m/s over 6 h.
- Cost per wrapper evaluation: 100 us (value, 6-state), 187 us (value and gradients, 48-state).

## 3. Gates that failed

**LT_G3(a).** The force law against the Lorentz-transformed radiation force passes (2.30e-8 <= 1e-7; precondition
1.7e-16). The negative control, production's geometric SRP against the same oracle, has to differ by at least 100x
the gated error at every sample. Its largest value is 2.1e-4, but its smallest is 4.75e-7, 21x. The predeclared test
reads the minimum. No diagnostic; hypothesis: at some samples the light-time and geometric forces nearly coincide
(a small `v.u` and a small aberration), so the minimum control is a poor discriminator.

**LT_G4(a) and (d).** Analytic position gradients against a fourth-order central difference:

| check | step | computed | threshold |
|---|---|---|---|
| (a) total gradient, lunar penumbra (333 samples) | 10 m | 5.66e-4 | 1e-6 |
| (a) total gradient, NO_SHADOW (2000 samples) | 10 m | 5.96e-6 | 1e-6 |
| (d) composed with the Earth shadow (714 samples) | 100 m | 1.86e-5 | 1e-6 |

The exploratory diagnostic (written after the result, the same samples, ten steps from 0.1 m to 3000 m) splits the
error into the production geometric part, which S4 did not change, and the S4 increment part:

| check | 0.1 m | 1 m | 10 m | 100 m | 3000 m | best step per sample |
|---|---|---|---|---|---|---|
| (a) penumbra, total | 3.7e-4 | 3.0e-5 | 5.7e-4 | 0.37 | 14 | 326/333 <= 1e-6, worst 6.1e-6 |
| (a) penumbra, production only | 1.7e-4 | 3.3e-5 | 8.4e-4 | 0.37 | 14 | 327/333 <= 1e-6, worst 1.1e-5 |
| (a) NO_SHADOW, total | 5.9e-4 | 6.4e-5 | 6.0e-6 | 6.1e-7 | 2.2e-8 | 2000/2000 <= 1e-6 |
| (a) NO_SHADOW, production only | 5.9e-4 | 5.7e-5 | 6.2e-6 | 5.8e-7 | 2.0e-8 | 2000/2000 <= 1e-6 |
| (d) composed, total | 9.1e-3 | 6.9e-4 | 1.9e-4 | 1.9e-5 | 1.0 (stencil crosses a contact) | 633/714 <= 1e-6, worst 3.4e-6 |

Reading (a hypothesis, not a proof): the untouched production gradient fails the same oracle in the same way, so the
predeclared FD steps, not the analytic gradients, carry the error. In full light the error falls as 1/h (FD
round-off: the gradient is ~|a|/d with d = 1.5e11 m); in the thin lunar penumbra it grows steeply above ~3 m (FD
truncation across the curvature of nu). The increment alone is ~1e-4 of |a_SRP|, so its FD is round-off limited at
every step, and (d) is an increment-only quantity. LT_G4(b), the increment at h = 1e6 m in full light, passes
(6.9e-7). Seven penumbral samples and 81 composed samples reach no step below 1e-6; the diagnostic cannot decide
whether those carry a small gradient error.

LT_G4(d) prints a RuntimeWarning (0/0) for samples in the lunar umbra, where both the composed gradient and its FD are
exactly zero, as S3's E_G4.

The three failing tests are strict xfails (as S2's L_G6 and S3's E_G3(a)/E_G5(a)/E_G6): the predeclared assertions
run unchanged, and a pass fails the suite.

## 4. Path agreement (LT_G6)

6- vs 42/48-state positions (max over the pairs), Phi relative to max|Phi|, and the package effect on the 6-state
positions; the gate is `on <= 2 off + 1e-6 m`, Phi <= 1e-8, effect >= 1e-4 m:

| case | on (m) | off (m) | on/off | Phi | effect (m) |
|---|---|---|---|---|---|
| (i) TERM, J2, NO_SHADOW | 8.6e-7 | 9.3e-7 | 0.93 | 1.2e-13 | 9.1e-3 |
| (ii) + S1 tide | 9.2e-7 | 9.0e-7 | 1.02 | 1.3e-13 | 9.1e-3 |
| (iii) zonal harmonics | 9.0e-7 | 9.3e-7 | 0.96 | 1.7e-13 | 9.1e-3 |
| (iv) PLANE, CONICAL, events | 9.3e-6 | 1.07e-5 | 0.87 | 3.0e-10 | 7.1e-3 |
| (v) real E2, GRAIL J2, events | 8.7e-6 | 1.00e-5 | 0.87 | 2.3e-11 | 6.4e-3 |
| (vi) CONICAL, Earth shadow, events | 4.1e-6 | 4.6e-6 | 0.89 | 3.4e-11 | 4.7e-3 |

The package adds nothing above the integration floor of each case.

## 5. Expectations against outcomes

| gate | expected | computed |
|---|---|---|
| LT_G2 | exact | exact |
| LT_G3(a) | ~1e-8, control ~1e-4 | 2.3e-8, control 4.75e-7..2.1e-4 |
| LT_G3(b) | 3e-8..5e-8 rad, control ~1e-4 rad | 4.1e-8 rad, 8.9e-5 rad |
| LT_G3(c) | ~1e-2 m | 0.034 m |
| LT_G4 | 1e-9..1e-7, (b) 1e-6..1e-5 possible | (a) 5.7e-4 / 6.0e-6, (b) 6.9e-7, (c) 3.4e-10, (d) 1.9e-5 |
| LT_G5(a) | ~6e-8 | 2.3e-8 |
| LT_G5(b) | ~2e-7, control ~1e-4 | 4.1e-7, 4.0e-3 |
| LT_G6 | on/off ~1, effect 1-5 mm | 0.87-1.02, 4.7-9.1 mm |
| LT_G7 | shifts ~0.1 s | 45-47 ms |
| LTB_G3 | ~1e-7 m, 1e-5..1e-3 m without | 2.5e-7 m, 7.8e-5 m |

## 6. Regression (LT_G9)

- Full suite at `b320f6d` (the re-freeze commit; the later commits change only results and documents), run with
  `QT_QPA_PLATFORM=offscreen python -m pytest -p no:cacheprovider tests --junitxml=...`: JUnit header
  `tests="3406" failures="0" errors="0" skipped="71"`, pytest exit 0, 4899 s. The 3258 `testcase` elements split into
  3187 passed, 49 skipped and 22 strict xfails. The header counts 148 more tests than the elements; the reason is
  BİLİNMİYOR (not investigated).
- The 22 xfails are the expected ones: S4 three (LT_G3(a), LT_G4(a), LT_G4(d)), S3 seven, S2 L_G6 three, EST-01 two,
  the external-oracle `cd_ssb` case one, and WP3.3 K_G1 six. No strict xfail passed.
- `tools/check_repository.py` passes (597 migrated paths, 409 preserved byte identities).
- The re-freeze is additive: `F3_WP3_2_S4_REFROZEN_CURRENT_TREE_SHA256` is resolved first, the S3 identities stay in
  the S3 table and at the end of the path-migration history; the history-chain tests of F1, S1, S2, S3 and the new
  S4 test pass.
- Not checked: whether the 49 skips equal main's skips apart from new tests; no main baseline run was made in this
  slice.

## 7. Agent decisions

As predeclared (`agent_decisions` 1-9), plus, after the results: the strict xfails and the exploratory FD-step
diagnostic. All flagged for the owner's review.

## 8. Reproduce

```text
python -m pip install -e .
python -m pytest tests/test_wp3_2_srp_light_time.py tests/test_wp3_2_srp_light_time_transport.py -q
python studies/phase17/scripts/wp3_2_s4_gate_values.py LABEL OUT.json
python studies/phase17/scripts/wp3_2_s4_default_identity.py run omit OUT.json
python studies/phase17/scripts/wp3_2_s4_default_identity.py compare A.json B.json [...] --out RESULT.json
python studies/phase17/scripts/wp3_2_s4_lt_g4_step_diagnostic.py OUT.json
```
