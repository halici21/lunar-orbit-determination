# Phase C6: lunar albedo + thermal-IR radiation pressure on the LRO arc (linear survey)

**Decision as computed: C6_NOT_QUALIFIED** (three runs, gate G_QUAD failed in each; amendments A1-A4 are in the
predeclaration). The experiment numbers are far from every hypothesis threshold and are identical between runs 2
and 3, so they are reported below as information. They do not support lunar radiation pressure as the cause of
F2p's K_SRP excess.

- Predeclaration: `studies/phase17/configs/phase_c6_lunar_radiation_survey_predeclaration.json` (`66ba3e2`;
  A1/A2 `1624776`, A3 `0d60fb0`, A4 `860716d`).
- Model `scripts/phase_c6_lunar_radiation.py`; columns `scripts/phase_c6_forced_columns.py`; driver
  `scripts/phase_c6_lunar_radiation_survey.py`; tests `tests/test_phase_c6_lunar_radiation.py` (5 pass).
- Results `results/phase_c6_lunar_radiation_survey.json` (run 3), `..._run2_interp_forcing.json`,
  `..._run1_quadrature.json`, with logs.

## 1. Why

F2p estimates K_SRP = 0.0745 m²/kg. The physical prior is 0.010 ± 0.005, consistent with FDF operations (cannonball
14 m², Cr 1.0–1.67; Slojkowski et al., ISSFD 2014). F2's force model has no lunar radiation pressure (LRP). At
50 km, dayside lunar thermal emission is of the order of the solar flux and points radially outward, so a
cannonball K could be absorbing it.

## 2. Model and method

- **Moon:** uniform Lambertian sphere.
  - Albedo 0.12.
  - Thermal emission in radiative equilibrium: T = max(T_ss cos^¼ψ, 100 K), ε = 0.95.
  - Cap visible from the spacecraft: 60 Gauss–Legendre rings (A1) × 72 sectors.
- **Spacecraft:** cannonball, K_LRP = 0.010 m²/kg.
- **Linear method:** sensitivity column of the LRP forcing along F3's cached F2p trajectory. It is mapped to every
  measurement row through F3's local rows.
- **Batch:** linearised about F2p with F3's priors. Fit days 260–262 are fitted; 263 is predicted.

Acceleration along the arc (K_LRP = 0.010):

| quantity | value |
|---|---|
| RMS / max | 2.06e-8 / 4.16e-8 m/s² |
| RIC RMS | 2.06e-8 / 7.0e-10 / 2.4e-10 m/s² (radial) |
| radial mean | +1.32e-8 m/s² (outward) |
| LRP / SRP RMS, sunlit (same K) | 0.60 |
| column max position, 260–263 | 22.1 m (G_SIZE PASS, ≤ 200 m) |

## 3. Gates

| gate | run 1 (quadrature) | run 2 (A3) | run 3 (A4) |
|---|---|---|---|
| G_UNIT (5 closed-form tests) | PASS | PASS | PASS |
| G_QUAD K column (≤ 1e-3) | **FAIL** 1.4e-2 | **FAIL** 2.4e-3 | **FAIL** 1.07e-3 |
| G_QUAD C_T column (≤ 1e-3) | **FAIL** 7.2e-2 | PASS 8.5e-4 | PASS 8.3e-4 |
| G_E0 (F3 σ = 0 chi² to 1e-6) | PASS | PASS | PASS |
| G_SIZE | PASS | PASS | PASS |

What each run fixed:

- **Run 1** formed the columns as Φ(t)∫Φ(τ)⁻¹. F3's finite-difference Φ has cond ≈ 8.5e11 at the arc end, and the
  product loses all accuracy after about a day: the C_T column error is 1e-5 at 6 h and 0.36 at 2.3 d.
- **A3** integrates the forced variational equations directly. The gradient is the full one: GL1800F N = 300 in
  MOON_PA_DE440, plus the Moon point mass and the Earth and Sun. The integrator is RK4 with 2 s sub-steps.
- **A4** evaluates the forcings exactly at the sub-step nodes, because the SRP forcing switches at shadow
  transitions.

The remaining K-column difference is 9.2 m on an 8.6 km column. A3 omits the SRP position gradient, including the
penumbra illumination term, while the cached column is a full nonlinear finite difference at K = 0.0745. That
omission is the likely cause; it was not tested.

## 4. Experiments (information; run 3, identical to run 2)

| exp | what | 263 chi² | share | K_SRP | S_T (m/s²) | 263 range / Doppler |
|---|---|---|---|---|---|---|
| E0 | F2p (no LRP) | 69 899.8 | 0 | 0.0745 ± 0.0001 | 2.55e-7 | 5.21 m / 5.61 mm/s |
| E1 | LRP fixed, K_LRP 0.010 | 71 473.2 | −0.024 | 0.0688 | 2.23e-7 | 5.27 m / 5.68 mm/s |
| E1b | as E1, albedo 0.19 | 71 473.2 | −0.024 | 0.0688 | 2.23e-7 | 5.27 m / 5.68 mm/s |
| E2 | LRP scale free | 81 902.5 | −0.181 | 0.0327 ± 0.0017 | 2.0e-8 | 5.63 m / 6.11 mm/s |

Hypotheses as computed (not read, because the decision is NOT_QUALIFIED):

- **H_K false:** K drops 0.0057; the threshold is ≥ 0.03.
- **H_PRED false:** share −0.024; the threshold is ≥ 0.3.
- **H_SCALE false:** E2 scale 7.31 ± 0.29, which is not within 3σ of 1.

The fit-day chi² barely moves: 14 496 → 14 517 (E1) → 14 656 (E2).

## 5. Reading (information only)

1. **Lunar radiation pressure at its physical size does not explain the K excess.** It removes 0.0057 of the
   0.0645 excess and makes the 263 prediction slightly worse.
2. **E2 is a degeneracy, not a detection.** The data accept an LRP 7.3× the physical size if K drops to 0.033.
   Scale and K are correlated −0.998, and the 263 prediction gets worse (share −0.18). The radial,
   once-per-revolution-like signature can stand in for part of SRP + S_T, but it does not predict better.
3. **E1b = E1 is a property of the model.** With radiative equilibrium and Lambertian reflection, the dayside
   radiance is aS cosψ/π + (1−a)S cosψ/π = S cosψ/π for any albedo. Only the 100 K night floor depends on a.
   An albedo sensitivity therefore needs an albedo map with non-equilibrium (Diviner) temperatures, not a
   different constant.
4. **Next candidate, per the predeclared reading rule** (applied informally, since NOT_QUALIFIED): box-wing / panel
   SRP. LRO's solar array tracks the Sun, so the true response differs from a cannonball (Stiller and Dirkx 2023
   found opposite along/cross signs). Both C6 and that test depend on LRO attitude data that are not yet
   available.

## 6. Finding outside C6: F3's local transitions

F3's `transition()` builds each 20 s transition as Φ(t+20)·Φ(t)⁻¹ from the same finite-difference Φ:

- The symplectic error of those transitions is 6.5e-8 at the start, 1.8e-4 at 5.6 h, 0.07 at 1.2 d and 0.6–0.8
  after 2.3 d.
- They differ from the symplectic-inverse form by up to 10.

In a σ = 0 run the algebra stays self-consistent: F3's G0 reproduces the batch exactly. With σ > 0, however, the DMC
process noise and the w→x coupling are mapped through wrong local dynamics after day 1. The local measurement
rows hx = h0 Φ⁻¹ are mostly intact: |hx_r| = 1 and |hx_v| ≈ 1.2 s, the light-time partial. The exception is a few
263 rows (max |hx_v| 25 s).

So the σ > 0 results of F3 and F3b (σ*'s 263 gain, the share split) need re-qualification with local transitions
from `phase_c6_forced_columns.forced_columns(..., local_transitions=True)`. F3c is a plain batch and is not
affected. That re-qualification is the next step (F3d).

## 7. Limitations

- Linear, first order about F2p.
- Uniform albedo; analytic temperature model.
- Cannonball target.
- No spacecraft thermal re-radiation.
- G_QUAD's K residual is unexplained beyond the stated likely cause.
