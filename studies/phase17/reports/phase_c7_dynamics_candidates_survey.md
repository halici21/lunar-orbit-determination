# Phase C7: lunar solid tide, Moon 1PN, Venus and Jupiter on the LRO arc (linear survey)

**Decisions as computed, all gates PASS:**
- **C7_TIDE_MATERIAL**
- C7_PN_IMMATERIAL
- C7_PLANETS_IMMATERIAL

Files:
- predeclaration `configs/phase_c7_dynamics_candidates_survey_predeclaration.json` (`f64adab`)
- code `02cb385`
- model `scripts/phase_c7_dynamics_candidates.py`
- driver `scripts/phase_c7_dynamics_candidates_survey.py`
- tests `tests/test_phase_c7_dynamics_candidates.py` (4 pass)
- results `results/phase_c7_dynamics_candidates_survey.json`

## 1. Why

F2's force model has no lunar solid tide, no relativistic term and no planets beyond the Earth and Sun.

The GL1800F label says that the degree-2 and degree-3 coefficients **do not include the permanent tide**: `jggrx_1800f_sha.lbl`, lines 31–34, give k2 = 0.024223 and k3 = 0.0163. A user of the field therefore has to add the full tide, and F2 does not.

## 2. Models and method

| candidate | model |
|---|---|
| Tide | Elastic induced potential of the Moon raised by the Earth and Sun, degrees 2 and 3, R = 1738 km |
| 1PN | Moon Schwarzschild term (IERS 2010 eq. 10.12) |
| Planets | Venus and Jupiter barycentre, indirect third body |

The method is C6 run 3's:
- forced variational columns along F3's cached F2p trajectory;
- exact forcing at each sub-step node;
- C6's cached GL1800F N = 300 gradients.

The fit is a linearised batch about F2p with F3's priors: days 260–262 are fitted and 263 is predicted. Each candidate is held fixed at its physical size.

## 3. Gates

| gate | result |
|---|---|
| G_UNIT | PASS. Tide = finite difference of its potential (1e-6, n = 2, 3); 1PN radial on a circular orbit; third body closed form |
| G_QUAD (C_T column) | PASS, 8.3e-4 |
| G_E0 | PASS, F3 σ = 0 chi² to 1e-6 |
| G_SIZE (≤ 200 m) | PASS. Tide 47.5 m, 1PN 0.10 m, planets 0.0003 m |

## 4. Size of the accelerations along the arc

| candidate | RMS (m/s²) | RIC RMS (m/s²) | radial mean (m/s²) |
|---|---|---|---|
| Tide, total | 5.7e-7 | 4.6e-7 / 2.8e-7 / 1.9e-7 | −1.5e-7 |
| · Earth k2 | 5.7e-7 | | |
| · Earth k3 | 2.0e-9 | | |
| · Sun k2 | 2.9e-9 | | |
| 1PN | 1.4e-10 | radial | +1.4e-10 |
| Venus + Jupiter | 1.2e-12 | | |

For scale, SRP at K = 0.010 on the sunlit part is 4.5e-8 m/s². The tide is about **13× SRP**.

## 5. Experiments

| exp | 263 chi² | share | K_SRP | C_T | S_T | 263 range / Doppler | fit range / Doppler |
|---|---|---|---|---|---|---|---|
| E0 (F2p) | 69 900 | 0 | 0.0745 | −1.9e-8 | 2.55e-7 | 5.21 m / 5.61 mm/s | 1.030 m / 2.036 mm/s |
| **E_TIDE** | **27 460** | **0.641** | **0.0395** | −1.2e-8 | **1.17e-7** | **3.22 m / 3.68 mm/s** | 1.077 m / 2.057 mm/s |
| E_PN | 69 902 | −0.000 | 0.0745 | | | 5.21 / 5.61 | 1.030 / 2.036 |
| E_PLANETS | 69 900 | −0.000 | 0.0745 | | | 5.21 / 5.61 | 1.030 / 2.036 |
| E_ALL | 27 459 | 0.641 | 0.0395 | | | 3.22 / 3.68 | 1.077 / 2.057 |

Hypotheses as computed:
- H_TIDE is true on both criteria: share 0.64 ≥ 0.3, and K drops 0.035 ≥ 0.03.
- H_PN and H_PLANETS are false: |share| < 0.001.

## 6. Reading

1. **The missing lunar tide is the largest force-model defect found so far.**
   - It explains 64 % of the gap between F2p's 263 prediction and the σ* DMC smoother's.
   - It removes more than half of the K_SRP excess: 0.0745 → 0.0395, against a physical 0.010.
   - It halves S_T, the once-per-revolution along-track empirical: 2.55e-7 → 1.17e-7.

   Before the tide, K and S_T were absorbing it. This also explains F3b: the DMC smoother's gain was a "K / S_T re-split" because it was partly recovering the tide.
2. **The fit-day misfit grows slightly in the linear survey** (range 1.030 → 1.077 m). This is expected when one parameter set is held at F2p's linearisation point while a 47 m force column is added; a nonlinear refit should recover it. The predeclared reading rule sends a MATERIAL candidate to a nonlinear refit, **C7-I**: the tide as a study-level force term in F2's propagation, then F2p's recipe.
3. **1PN and Venus/Jupiter are closed for this arc.** 1PN shifts the trajectory by 0.1 m and the planets by 0.3 mm; neither changes any score.

## 7. Limitations

- Linear about F2p.
- Elastic tide: no lag, no frequency dependence. GL1800F gives only real k2 and k3.
- The 1 % gradient-model gap from C6/F3d applies to the columns; G_QUAD bounds it at 8e-4 on the C_T column.
- The Moon's own tidal response to its orbital eccentricity and libration is inside this model, because the Earth's position is taken from DE440 at every epoch.
