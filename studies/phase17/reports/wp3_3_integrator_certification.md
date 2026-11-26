# F3 WP3.3 — Long-arc integrator certification of every propagation profile (REQ-DYN-006, DYN-14)

- **Date:** 2026-11-26 (campaign run 2026-11-25)
- **Predeclaration:** `studies/phase17/configs/wp3_3_integrator_certification_predeclaration.json`, committed in
  `63a82eb` before any code, test or number. Amendment A1, made before any number, is recorded in the same file:
  the predeclared legacy J2 coefficient is refused by the production default `grail_pa_de440` model, so the
  production default (`J2_MOON_GRAIL_UNNORMALIZED`, arc epoch) is used.
- **Tests:** `tests/test_wp3_3_integrator_certification.py`. It was committed failing first (`1591b15`: the driver
  did not exist yet). It holds K_G1 `test_two_body_10rev_closure` and the C_G1 rule tests.
- **Driver:** `scripts/wp3_3_integrator_certification.py` (`run`, `gates`, `kg1`, `smoke`).
- **Results:** `results/wp3_3_integrator_certification.json`. The campaign ran at `ce1f1bb` with a clean `src`,
  took 3011 s on 14 workers, and used Python 3.13.2, numpy 2.4.3, scipy 1.17.1 and numba 0.65.1.
- **Exploratory, after the result:** `results/wp3_3_rg2_diagnostic.json` from `scripts/wp3_3_rg2_diagnostic.py`,
  and the per-orbit validity table (section 5). Both are recorded in `docs/assurance/amendment-log.md`.
- **Production code changed:** none. This is a characterization. No defect was found that needs a fix here.
  Section 6 lists two findings for the owner.

## 0. Final status

```
WP3.3 (as predeclared)  = per-profile validity range DOCUMENTED; certification of the long arcs mostly
                          INCONCLUSIVE: the reference failed its own gates on 2 of 3 orbits
  R_G1 reference        = O1 PASS at 1 d (6.5e-5 m), fails from 3 d; O2 PASS to 3 d, fails from 7 d;
                          O3 FAIL at 1 d (1.55e-3 m) -> O3 INCONCLUSIVE
  R_G2 contacts         = O1 PASS (1370 cuts), O3 PASS (446); O2 FAIL (16 of 432 probes) -> O2 INCONCLUSIVE
  K_G1 two-body closure = FAIL for 7 of 8 profiles (both cases); only Radau passes (1.7e-5 / 3.6e-5 m)
  C_G1 validity         = no profile is valid on all three orbits for any tier: every range is 'none'
                          or 'inconclusive' (table 4.1)
  Orekit                = OUT OF SCOPE (not installed locally); deferred to WP5.1
FINDINGS   F1 P_D2 (PropagationProfile()) refuses long arcs: the contact-consistency guard raises at 28 d (T_LIB,
              O1/O3) and at 7 d (T_THESIS, O1/O3); fail-closed, not silent
           F2 the default ADAMS and the truth DOP853 (no events) are not valid even at 1 m on the shadowed orbit
              over 1 day (1.07 m, 1.47 m)
           F3 the SRP illumination fraction is not resolved near a contact to better than ~1e-7 (it reaches
              1 + 6.7e-8 inside the outer contact), so contact epochs at grazing geometry are fuzzy by ~10 ms
REQ-DYN-006 stays partial (Orekit point open; long-arc reference inconclusive on O2/O3)
```

## 1. Setup

- **Profiles:** every combination that `propagation_contract.resolve_profile` accepts:
  - ADAMS (estimator default), DOP853 (truth-wrapper default), RK45, RK23, Radau, BDF and LSODA, each with
    `shadow_events=False`;
  - `PropagationProfile()`, which is DOP853 with `shadow_events=True` (P_D2).
- **Not a library profile:** the UKF RK4 fast-sigma propagator is supplied by the caller, so it is out of scope.
- **Tolerances:**
  - T_LIB: rtol 1e-11, atol 1e-12 (the `propagate_state` default);
  - T_THESIS: rtol 1e-10, atol 1e-11 (`THESIS_RTOL`/`ATOL`).
- **Arcs:** 1, 3, 7 and 28 days.
  - Each arc is a separate `propagate_state` call from t0, with output every 60 s.
  - The error is the maximum position error over that output.
  - Each propagation has a wall-clock budget of 1800 s. No run hit it; the slowest was RK23 at 28 d, 1431 s.
- **Orbits:** epoch 2027-03-02T00:00 UTC (E1 of WP3.1), with the WP3.1 geometry:
  - O1: circular 100 km, β ≈ 0, in eclipse on every revolution;
  - O2: circular 100 km, high β; its eclipse season starts at day 19;
  - O3: 100 × 3000 km, in the plane of O1.
- **Forces:** point Moon, Earth and Sun (DE440 GMs and ephemerides), Moon J2 (the GRAIL PA model), and SRP with
  K = 0.010 m²/kg and conical penumbra.
- **Reference:** DOP853 at rtol 1e-13, restarted at every penumbra/umbra contact.
  - The contacts are found independently of production. They come from illumination-class changes on a 10 s
    grid, bisected to 1e-6 s, with one refinement iteration.
  - The self-convergence twin uses the same cuts at rtol 1e-12.
  - The reference shares the production force functions. It checks integration, not force modelling.

## 2. K_G1: two-body 10-revolution closure

Pure Kepler (no Earth, Sun, J2 or SRP), T_LIB tolerances. Gate: closure ≤ 1 mm and |ΔE/E| ≤ 1e-11.

| profile | K_CIRC (100 km) closure / energy | K_ECC (100 × 3000 km) closure / energy | verdict |
|---|---|---|---|
| ADAMS | 6.8e-3 m / 1.2e-9 | 0.75 m / 3.6e-9 | FAIL |
| DOP853, D2 | 1.8e-4 m / **2.5e-11** | **5.6e-2 m** / 5.0e-10 | FAIL |
| RK45 | 1.6e-2 m / 1.9e-10 | 4.3e-2 m / 2.0e-10 | FAIL |
| RK23 | 2.2e-2 m / 2.6e-10 | 3.2e-2 m / 1.4e-10 | FAIL |
| Radau | 1.7e-5 m / 4.2e-12 | 3.6e-5 m / 7.0e-12 | **PASS** |
| BDF | 3.2 m / 3.7e-8 | 2.4 m / 9.9e-9 | FAIL |
| LSODA | 3.7e-2 m / 4.4e-10 | 7.2e-2 m / 6.3e-10 | FAIL |

- **Independent check.** A plain `scipy.integrate.solve_ivp` DOP853 run on a pure Kepler right-hand side, at the
  same tolerances, gives 1.8e-4 m (circular) and 6.3e-2 m (eccentric). At rtol 1e-13 it gives 1.9e-6 m and
  3.6e-4 m. The oracle and the setup are sound. The failures are what these methods deliver at rtol 1e-11. They are
  not production defects.
- **In the test.** The 14 failing cases are strict `xfail`s. The thresholds are unchanged, so a case that starts
  passing fails as an unexpected pass.

## 3. Reference adequacy (R_G1, R_G2)

| orbit | dR 1 d | dR 3 d | dR 7 d | dR 28 d | contacts (28 d) | R_G2 | orbit status |
|---|---|---|---|---|---|---|---|
| O1 shadowed 100 km | **6.5e-5** | 3.8e-4 | 2.0e-3 | 3.6e-2 | 1370 | PASS | CONCLUSIVE |
| O2 sunlit 100 km | 2.1e-5 | 7.5e-5 | 8.1e-4 | 1.4e-2 | 432 (from day 19) | **FAIL** (16 probes) | INCONCLUSIVE |
| O3 100 × 3000 km | **1.55e-3** | 1.6e-2 | 9.2e-2 | 1.57 | 446 | PASS | INCONCLUSIVE |

dR is the maximum of |r(1e-13) − r(1e-12)| in metres. R_G1 requires dR ≤ 1e-4 m.

- **What dR measures.** dR bounds the reference error from above. It is dominated by the rtol 1e-12 twin's own
  error, so the 1e-13 reference is likely much better than dR. The predeclared rule still stands: a tier verdict
  at arc L counts only if dR(L) ≤ 0.1 × the tier threshold, and an orbit failing R_G1 at 1 d or R_G2 is
  INCONCLUSIVE.
- **Contact refinement.** The iteration moved the cuts by up to 0.46 s on O1, 3.6 ms on O2 and 6.1 ms on O3.
  Bisection brackets are ≤ 7.7e-7 s.
- **C_G2 (information).** On day 1, production `shadow_contact_times` finds the same contacts as the reference:
  48/48 on O1 and 20/20 on O3, with |Δt| ≤ 1.3e-4 s.

## 4. Results

### 4.1 Validity range as predeclared (C_G1, with the orbit-level reject_if)

A range is the longest arc, in days, for which every orbit is within the tier and the verdict is conclusive.

| profile / tolerance | MM 1 mm | CM 1 cm | DM 1 dm | M 1 m |
|---|---|---|---|---|
| ADAMS, DOP853, BDF (both tolerances), LSODA/T_THESIS | none | none | none | none |
| D2 / T_LIB | inconclusive | inconclusive | inconclusive | inconclusive |
| D2 / T_THESIS | none | inconclusive | inconclusive | inconclusive |
| RK45 / T_LIB, LSODA / T_LIB | none | none | none | inconclusive |
| RK45 / T_THESIS, RK23 / T_LIB | none | none | inconclusive | inconclusive |
| RK23 / T_THESIS | none | none | none | inconclusive |
| Radau (both tolerances) | none | inconclusive | inconclusive | inconclusive |

- **'none'.** A conclusive failure on O1 (or a raised run) already at 1 day.
- **'inconclusive'.** O1 passes at 1 day, but O2 and O3 cannot confirm it.
- **The rule without the orbit-level reject_if.** This is what the driver first computed. It is kept in the JSON
  as `validity_without_orbit_reject_if`. Under it D2/T_LIB is MM/CM inconclusive, DM 1 d and M 7 d.

### 4.2 Maximum position error, T_LIB (m)

| profile | O1 1 / 3 / 7 / 28 d | O2 1 / 3 / 7 / 28 d | O3 1 / 3 / 7 / 28 d | runtime 28 d |
|---|---|---|---|---|
| ADAMS | 1.07 / 10.1 / 55 / 968 | 0.105 / 0.51 / 5.9 / 115 | 0.024 / 0.22 / 3.5 / 54 | 17–22 s |
| DOP853 | 1.47 / 1.9 / 30 / 179 | 1.8e-4 / 1.5e-3 / 1.3e-2 / 0.65 | 0.79 / 11 / 49 / 434 | 24–39 s |
| **D2** | **2.0e-4 / 1.1e-3 / 1.05e-2 / ERROR** | 1.8e-4 / 1.5e-3 / 1.3e-2 / 0.24 | 1.4e-2 / 0.14 / 0.84 / ERROR | 489–664 s |
| RK45 | 0.11 / 1.1 / 4.5 / 33 | 1.9e-2 / 0.17 / 0.94 / 15 | 1.9e-3 / 7.7e-2 / 0.69 / 7.9 | 56–155 s |
| RK23 | 2.0e-2 / 0.18 / 0.97 / 16 | 2.5e-2 / 0.22 / 1.2 / 19 | 1.1e-2 / 9.4e-2 / 0.51 / 8.5 | 959–1431 s |
| **Radau** | 1.3e-3 / 4.7e-3 / 8.5e-3 / 6.4e-2 | 2.4e-5 / 1.6e-4 / 8.0e-4 / 1.4e-2 | 8.7e-4 / 3.8e-3 / 9.8e-3 / 8.1e-2 | 453–694 s |
| BDF | 4.2 / 38 / 208 / 3360 | 4.6 / 41 / 221 / ERROR | 0.34 / 2.7 / 14 / ERROR | 47–118 s |
| LSODA | 0.69 / 9.8 / 58 / 801 | 0.23 / 2.0 / 12 / 186 | 0.13 / 1.1 / 3.4 / 59 | 19–32 s |

T_THESIS values are in the JSON. They are mostly 1–10× larger. The exceptions are DOP853 on O1 (section 6, F2)
and LSODA on O3 at 3 d. Velocity errors are recorded per cell as `e_v_mps`.

## 5. Exploratory (after the result; amendment log 2026-11-26)

### 5.1 Per orbit

This applies the same C_G1 tier rule to each orbit alone. It keeps the per-arc conclusiveness rule but drops the
orbit-level reject_if.

| profile / tol. | O1 shadowed: MM / CM / DM / M | O2 sunlit | O3 elliptic |
|---|---|---|---|
| D2 / T_LIB | **1 / 3 / 7 / 7** (28 d raises) | 1 / 3 / 7 / 28 | inc. / inc. / 1 / 7 |
| D2 / T_THESIS | none / 1 / 3 / 3 | none / 1 / 3 / 7 | inc. / inc. / none / 1 |
| Radau / T_LIB | none / 3 / 7 / 28 | 3 / 7 / 7 / 28 | inc. / inc. / 1 / 7 |
| DOP853 / T_LIB | none / none / none / none | 1 / 3 / 7 / 28 | inc. / inc. / none / 1 |
| ADAMS / T_LIB | none everywhere | none / none / none / 3 | inc. / inc. / 1 / 3 |
| RK45 / T_LIB | none / none / none / 1 | none / none / 1 / 7 | inc. / inc. / 1 / 7 |

The other profiles are in `validity_per_orbit_exploratory`.

### 5.2 R_G2 on O2

All 16 failing probes are at the start of the eclipse season (days 19.0–20.1). The rerun reproduces them.

- **Hypothesis 1, not supported.** The cut set might not have converged against the final trajectory. Only 3 of
  the 16 failing probes have a cut shift above the 1 ms probe.
- **Explanation found.** At these slow, grazing contacts, the illumination fraction changes by only ~1e-7 within
  ±1 ms. At that level `srp._illumination_for` is not resolved:
  - outer contact: 1 − ν = −6.7e-8 at a point geometrically inside the penumbra;
  - inner contact: the class at ±1 ms is the same, and it differs at ±10 ms.
- **Reading.** The R_G2(a) probe is narrower than the class resolution there. The effect on the reference
  trajectory is a cut misplaced by ≲ 10 ms at an acceleration kink, which is negligible. The predeclared FAIL
  stands.

## 6. Findings (owner)

- **F1. P_D2 refuses long arcs.** `_shadow_contact_check` raises "Shadow contact inconsistent with final
  trajectory ... (> 1e-03 s); tighten propagation tolerances". It happens at 28 d on O1 and O3 (T_LIB), and at 7 d
  on O1 and O3 and 28 d on O2 (T_THESIS).
  - **Cause.** The contact pre-pass is a single unsegmented run. Its error grows to metres over days, so its
    contacts drift more than 1 ms from the final segmented trajectory.
  - **Behaviour.** This is fail-closed by design: no wrong answer is returned. In practice, though,
    `PropagationProfile()` is limited to about ≤ 7 d at T_LIB and ≤ 3 d at T_THESIS on shadowed orbits.
  - **Possible remedy.** An iterated (segmented) pre-pass, or per-segment contact refinement. That is a
    `dynamics.py` change, so it is not done here. It is a candidate WP3.x item.
- **F2. The default profiles are far from tolerance-level accuracy on shadowed orbits.**
  - The estimator default ADAMS: 1.07 m on O1 in 1 day, 968 m at 28 d.
  - The truth default DOP853 without events: 1.47 m on O1 in 1 day.
  - This confirms D1 F1/F2 and assumptions item 17 at arc lengths up to 28 d. ADAMS is also poor on the sunlit
    arc (0.1 m in 1 day, 115 m at 28 d).
  - Without events, the tolerance does not control the error on shadowed orbits. At T_THESIS, DOP853 on O1 at 1 d
    is 1.18 m, against 1.47 m at T_LIB.
- **F3. Resolution of the illumination fraction near a contact.** `srp._illumination_for` has numerical noise of
  ~1e-7 near both contacts and exceeds 1 by up to 6.7e-8 (section 5.2).
  - **Physical effect.** ~1e-16 m/s² (6.7e-8 × the SRP acceleration). That is negligible.
  - **What it limits.** Contact epochs at grazing geometry are fuzzy by ~10 ms. Production's 1 ms
    contact-consistency tolerance is therefore close to the function's own resolution there. Information for
    the owner; not a defect.
- **F4. Integrator choice.** Radau is the only method that passes K_G1, and the most accurate unsegmented method on
  every orbit: CM to 3 d on O1 and 7 d on O2, without event segmentation. It costs 450–700 s per 28-day arc,
  against ~30 s for DOP853. BDF fails outright (step-size underflow on O2/O3 at 28 d). RK23 and LSODA are not
  competitive.

## 7. Limits and open items

- **Orekit.** Not installed locally, so the independent-tool point is deferred to WP5.1, which builds the
  permanent Orekit oracle. REQ-DYN-006 therefore stays partial.
- **Reference self-convergence.** The 1e-13 reference cannot be certified beyond 1 d (O1, O3) or 3 d (O2) at
  1e-4 m with a 1e-12 twin. A 28-day millimetre certification needs a higher-order or extended-precision reference
  (for example an Orekit or quad-precision run), or a twin closer to the reference.
- **Scope.** The 6-state `propagate_state` only. The 42- and 48-state propagators, harmonics, tide, LRP and Earth
  shadow are not covered.
- **Coverage.** Three orbits at one epoch. The force model is the production one, so force fidelity is not
  assessed here.
