# PHASE 17-R1O-OPT — PRODUCTION LUNAR LANDMARK OPTICAL MEASUREMENT MODEL AND K_SRP IDENTIFIABILITY QUALIFICATION

**Status: `CHARACTERIZATION_COMPLETE`** — every physics/derivative gate PASSED; the identifiability
question resolved with a **decisive negative result about Phase 17-R1O's published claim**, and a
**defect found in R1O's own committed analysis code** that invalidates that claim's headline numbers.

---

## §1. The one-paragraph answer

Phase 17-R1O reported that a lunar-landmark line-of-sight observable rotates K_SRP's information
direction dramatically — `f_perp` 0.25 → **0.8757**, `sigma_K/K` 9.3% → **0.19%**. This phase built the
production measurement model to test that hypothesis and found that **it does not survive**. The
production image-plane observable delivers `f_perp` 0.295 → 0.325 and `sigma_K/K` 3.50% → 2.97%, an
operational improvement of **1.175×**, not the ~18× R1O implied. The cause was isolated by direct
construction, not inference: **R1O's analysis helper `chain_to_augmented_columns` unflattens the 6×6
state-transition matrix with NumPy's default C order, while `lunar_od.dynamics` packs and propagates
it column-major (`order="F"`)**. Φ is not symmetric, so every surrogate state-design row R1O built
used Φᵀ. Re-running R1O's own surrogate with the single reshape corrected drops `f_perp` from 0.8766
to **0.2531** — i.e. **the entire claimed gain was a linear-algebra defect**, and the production
model then agrees with the corrected surrogate to within 0.007 in `f_perp`.

This is the second consecutive R1O follow-up phase to return a negative result, and the two failed
for completely different reasons: R1O-D's ΔDOR surrogate failed because it discarded differential
light-time; R1O-OPT's landmark surrogate failed because of a transposed Φ.

---

## §2. Entering state and input gate

| Item | Value |
|---|---|
| Worktree | `python_port_phase17r0` |
| HEAD at start | `1045898b779bce62e51a5cf01053d4bff5e50ea4` (Phase 17-R1O-D) |
| `main` | `fea476f81dad07b3914e53eab709e10fa6e9d10b` (unchanged) |
| `origin/main` | `fea476f81dad07b3914e53eab709e10fa6e9d10b` (unchanged) |
| Untracked at start | the two pre-existing orphaned R1O-D scripts only |

`R1O_OPT_INPUT_GATE = PASS`.

---

## §3. Literature contract

`docs/phase17_r1o_opt_literature_contract.md`, written **before any code**, gives the term-by-term
contract (15 terms: literature source, exact value, repository representation, status).
`OPT_LITERATURE_CONTRACT_GATE = PASS`.

Primary anchors: Federici et al., *Aerospace* **12**(5) 374 (2025) — 2.5 px noise ↔ ~160 m ground
resolution for feature-based lunar-orbit optical navigation; Apollo CM sextant — 28× magnification,
1.8° FOV, 10 arcsec RMS sighting accuracy; CubeSat star-tracker surveys — 8–10 arcsec (1σ) for good
compact trackers, up to ~120 arcsec (3σ) for coarse ones; R1O's own carried-forward LONEStar figures
(0.25–1 px LOS error) and its conservative 0.5 mrad attitude floor.

**Camera choice was reasoned, not copied.** Federici et al.'s exact intrinsics were not recoverable
(access-restricted). The default `CameraIntrinsics` (35 mm focal length, 5.5 μm pixel pitch,
2048×2048) gives 157.1 μrad/px IFOV, an 18.3° full FOV, and **16.6 m/px ground-sample distance at
this campaign's measured 105.47 km altitude** — the same order of magnitude as Federici's implied
~64 m/px, finer by ~4× consistent with choosing a dedicated navigation camera over a wide-FOV
context camera. The reasoning is recorded in the contract so the choice is auditable rather than
asserted.

---

## §4. Reference-spacecraft optical-navigation capability audit

Mirroring R1O-D's RF-capability audit, this repository was searched for any frozen reference
spacecraft camera definition:

- **No camera model anywhere** in `lunar_od/*.py` prior to this phase.
- **No spacecraft attitude representation.** `orbit.py`'s `rot_x`/`rot_z` are generic
  orbital-element DCM helpers (MATLAB-matching); `force_contract.py`'s `body_frame` field and
  `scenario_config.py`'s `body_frame="not_applicable"` refer to the **gravity** body frame, not a
  spacecraft body or camera frame.
- **`reference_config` has zero camera-related fields** (verified by grep over
  `lunar_od/reference_config/*.py` and `tests/test_reference_config.py`).

`REFERENCE_SPACECRAFT_OPTICAL_NAV_CAPABILITY = UNKNOWN` — exactly as R1O-D concluded for ΔDOR RF
capability. Any statement that this mission "has" or "lacks" a navigation camera is unsupported by
the repository. The camera modelled here is a **declared, literature-anchored representative
design**, not a mission asset.

---

## §5. What was built

**`lunar_od/lunar_landmark_optical.py`** (new production module, ~330 lines):

| Component | Purpose |
|---|---|
| `CameraIntrinsics` | focal length (metric and pixel), pixel pitch, sensor format; FOV and GSD derived from the same numbers used in the projection |
| `nadir_pointing_camera_frame` | default error-free camera-from-inertial DCM, boresight = nadir, built with the SAME tangent-plane construction R1O's surrogate used |
| `small_angle_rotation`, `perturbed_camera_frame` | attitude error injection, shared mechanism for random noise and deterministic bias |
| `pinhole_project` | `rho_cam = C_ci (r_lm − r_sc)`, `u = f_px x/z + u0`, `v = f_px y/z + v0` |
| `pinhole_position_jacobian` | **analytic** `d(u,v)/d(r_sc)`, chained through `d(rho_cam)/d(r_sc) = −C_ci` |
| `landmark_optical_state_and_k_sensitivity` | `dg/dx0 = dg/dr · Φ[:3,:]`, `dg/dK = dg/dr · S_K[:3]` |
| `landmark_inertial_position_m`, `landmark_surface_outward_normal_j2000` | spherical-Moon landmark through the already-qualified `moon_pa_de440_rotation_at_et` |
| `landmark_geometrically_visible` | three-condition gate: far-side occultation, front-of-camera, field of view |
| `optical_one_way_light_time_s` | the light-time quantity §8 tests for significance |

**`tests/test_lunar_landmark_optical.py`** — 27 permanent tests.

**Analysis scripts** (analysis space only, no production code modified):
`examples/phase17_r1o_opt_oracles.py`, `_core.py`, `_jacobian_and_k.py`, `_bridge.py`, `_campaign.py`.

---

## §6. Oracles — hand-verifiable geometry

| Gate | Result | Evidence |
|---|---|---|
| `BORESIGHT_LANDS_AT_PRINCIPAL_POINT` | **PASS** | a landmark exactly on the boresight projects to (1024.000000000, 1024.000000000) |
| `OFFAXIS_PROJECTION_MATCHES_EXACT_TAN` | **PASS** | 2° off-axis: expected `f_px·tan θ` = 222.223079 px, actual 222.223079 px, rel_err **0.000e+00** |
| `CAMERA_ATTITUDE_FRAME_GATE` | **PASS** | orthonormality 1.110e-16, det−1 = 0.000e+00, boresight−nadir 0.000e+00 |
| `SMALL_ANGLE_ROTATION_SELF_CONSISTENT` | **PASS** | first-order boresight shift exact to 0.000e+00; orthonormality residual 8.900e-09 against the expected \|a\|² = 9.800e-09 |
| `VISIBILITY_GATE_THREE_CASES` | **PASS** | near-side VISIBLE, antipodal OCCULTED_FAR_SIDE, 20°-off-nadir OUTSIDE_FOV |

`CAMERA_PROJECTION_ORACLE_GATE = PASS`. The small-angle orthonormality tolerance is set **above**
the construction's own known O(\|a\|²) truncation — the Phase 17A-R floor-consistency discipline
applied to an attitude model rather than an FD oracle.

---

## §7. Analytic Jacobian vs finite difference (hard block §73)

§73 forbids resting on one arbitrary FD step. A six-step sweep was run:

| FD step | analytic-vs-FD rel_err |
|---|---|
| 1000 m | 1.690e-04 |
| 100 m | 1.690e-06 |
| 10 m | 1.689e-08 |
| **1 m** | **1.697e-10** |
| 0.1 m | 9.534e-10 |
| 0.01 m | 1.338e-09 |

Clean quadratic convergence (ratio 100.0 per decade, as central differencing requires) down to a
rounding-noise floor near 1 m. `CAMERA_PROJECTION_ORACLE_GATE = PASS` at best rel_err **1.697e-10**.

**Documented mistake, kept as evidence.** An earlier version of this sweep scaled the FD step to
`|r_sc|` (~1.8e6 m), giving ~1840 m steps and an apparent 1.4e-4 "disagreement" that was entirely
the FD side's own truncation error. The step must scale to the **observation geometry** (landmark
range ~1e5 m), not to the spacecraft's inertial position magnitude.

---

## §8. Light time, time tag, and aberration — three distinct effects, deliberately separated

An earlier version of this section reported only "spacecraft motion during the light time" and
labelled it `OPTICAL_LIGHT_TIME_SIGNIFICANCE`. That is **not** the light-time retardation; it is the
cost of getting the time-tag convention wrong. Conflating them is the same class of error R1O-D
caught in its own session-bias study, so each is computed on its own terms. At `v_sc` = 1631.078
m/s, τ(300 km slant) = 1000.692 μs, GSD 16.574 m/px, IFOV 157.143 μrad/px:

| Effect | Magnitude | As a fraction of the finest centroiding precision tested (0.1 px) |
|---|---|---|
| (1) Landmark retardation — the true light-time effect; in MCI the landmark moves only by lunar rotation | 4.63 mm → **2.79e-04 px** | 2.79e-03 → **NEGLIGIBLE** |
| (2) Time-tag convention — cost of using the emission rather than shutter epoch for the spacecraft | 1.632 m → **9.85e-02 px** | 0.98 → the wrong convention costs ~0.1 px |
| (3) Stellar aberration — the observer's velocity in the frame the landmark is expressed in; not captured by (1) | 5.441 μrad → **3.46e-02 px** | 0.35 → **SIGNIFICANT** |

- `OPTICAL_LIGHT_TIME_SIGNIFICANCE = NEGLIGIBLE`
- `OPTICAL_TIME_TAG_CONTRACT = SHUTTER_RECEPTION_EPOCH` (the module takes `r_sc` at shutter time by
  construction; the contract is stated because the wrong choice costs ~0.1 px)
- `OPTICAL_ABERRATION_SIGNIFICANCE = SIGNIFICANT`, `CHARACTERIZED_NOT_IMPLEMENTED`

**An honest caveat on aberration.** It is not a pure nuisance. Because it is proportional to
spacecraft velocity, it carries state information of its own, so omitting it is not equivalent to
omitting a constant boresight bias. Quantifying that coupling was not attempted here and is recorded
as a limitation, not resolved.

---

## §9. State Jacobian and K chain, on the real campaign arc

| Gate | Result | Evidence |
|---|---|---|
| `DIRECT_MEASUREMENT_K_DEPENDENCE_IS_NO` | **PASS** | no SRP/K token appears anywhere in `pinhole_project` or `pinhole_position_jacobian` (checked structurally via `inspect.getsource`, not asserted) |
| `OPTICAL_STATE_JACOBIAN_GATE` | **PASS** | max rel_err **1.207e-05** across all six components against independently re-propagated trajectories |
| `OPTICAL_K_COMPOSITION_SENSITIVITY_GATE` | **PASS** | composition vs module rel_err **0.000e+00** |
| `OPTICAL_K_E2E_SENSITIVITY_GATE` | **PASS** | best rel_err **1.586e-05** at dK = 1e-3, with the residual demonstrated to track the oracle's own tolerance |

At the arc midpoint the analytic sensitivity is `d(u)/dK = −0.1912685115`, `d(v)/dK = +0.1675474133`
px per (m²/kg) — four to five orders of magnitude larger, relative to its own noise, than ΔDOR's
3.5e-9 s/(m²/kg), so unlike R1O-D this phase's end-to-end FD is comfortably resolvable.

**Floor consistency applied to the K chain.** Rather than loosen a threshold when the first
end-to-end sweep landed at 1.29e-4, two diagnostics established what limits the oracle:

1. Two near-identical propagations differ by **exactly 0.000e+00 px** — the integrator is
   deterministic, so the residual is systematic truncation, not random noise.
2. Repeating the whole sweep at a tighter tolerance moved the best agreement **1.290e-04 →
   1.586e-05**. The residual tracks the oracle's own accuracy, which is what an oracle-limited
   comparison must do and what a genuine model discrepancy would not do.

---

## §10. THE CENTRAL FINDING — a transposed state-transition matrix in R1O's analysis code

### §10.1 How it surfaced

The production analytic state Jacobian disagreed with the end-to-end finite difference by factors of
**1e2 to 1e8**, with the velocity components worst — the classic signature of position and velocity
blocks being swapped.

### §10.2 The defect

`lunar_od/dynamics.py` packs the state-transition matrix column-major
(`np.eye(6).reshape(-1, order="F")`, line 1249) and propagates it that way, and **every production
consumer** unflattens it with `order="F"`: `delta_dor.py:411`, `dynamics.py:394/847/1255`,
`accelerated.py:448/464`, `estimators.py`. `delta_dor.py` even carries an explicit warning that
"reusing a different order here would silently transpose every Phi lookup."

`examples/phase17_r1o_core.py:130` (`chain_to_augmented_columns`, Phase 17-R1O) uses
`nom48_row[6:42].reshape(6, 6)` — NumPy's **default C order** — while its docstring claims to match
`_two_way_range_k_srp_column`. Φ is not symmetric, so this yields **Φᵀ**.

### §10.3 Direct evidence

At one campaign epoch, `d(u,v)/dx0` computed three ways:

```
C order (R1O convention): [[-7.556e-04 -2.848e-03 -4.183e-03  3.127e-06  4.563e-07 -9.268e-07] ...]
F order (production):     [[-1.715e-01 -2.915e-02  4.738e-02 -2.277e+01 -1.515e+02 -1.338e+02] ...]
end-to-end FD truth:      [[-1.715e-01 -2.915e-02  4.738e-02 -2.277e+01 -1.515e+02 -1.338e+02] ...]
```

The F-order convention matches re-propagated finite differences to ~1e-5; the C-order convention is
wrong by orders of magnitude.

### §10.4 Blast radius

| Site | Affected? |
|---|---|
| `phase17_r1o_core.py:240` — `build_ddor_arc` (R1O ΔDOR surrogate) | **YES** |
| `phase17_r1o_core.py:370` — `build_landmark_arc` (R1O landmark surrogate) | **YES** |
| `phase17_r1o_celestial.py:88` — R1O's Earth-LOS negative control | **YES** |
| `phase17_r1od_surrogate_bridge.py:101` — R1O-D's bridge, surrogate side | **YES** |
| `build_range_arc` (the W15 range baseline) | **NO** — built by production `two_way_range_nominal_and_initial_jacobian`, which handles Φ internally |
| `lunar_od/delta_dor.py` — R1O-D's production ΔDOR Jacobian | **NO** — uses `order="F"` explicitly |
| K columns everywhere | **NO** — `S_K` is a plain 6-vector needing no reshape (verified: the K column is bitwise identical between the two runs) |

**So R1O's range-only baseline and R1O-D's production ΔDOR result are unaffected; every R1O
*surrogate* state-design matrix is wrong.**

### §10.5 What it did to R1O's published numbers

Reproducing R1O's own code path first, as a control:

| Configuration | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---|---|---|
| Range-only baseline (90 s cadence) | 0.295486 | 17.1867° | 10.4083% |
| **+ R1O landmark surrogate, exactly as published (Φᵀ)** | **0.876626** | 61.2380° | **0.1939%** |
| + the same surrogate, Φ unflattened correctly | **0.253154** | 14.6642° | 0.6716% |
| + production optical model, ideal limit | 0.260151 | 15.0790° | 1.3263% |

Step 1 reproduces R1O's published headline (0.8757 / 0.19%) to within rounding — confirming this is
faithfully R1O's own path, not a different calculation. Correcting the single reshape collapses it.

**Attribution:**
- Φ transposition alone: **+0.623473** in `f_perp`
- Measurement model + visibility gate: **−0.006998** in `f_perp`

The defect accounts for essentially the entire claimed gain. The production measurement model —
pinhole camera, explicit attitude chain, real FOV gate — makes almost no difference by comparison.

---

## §11. Production information geometry

The 90 s cadence inherited from radiometric tracking is a poor grid for a narrow-field camera: the
~34 km FOV footprint at 105 km altitude is crossed in ~21 s, so at 90 s sampling the only in-FOV
frames are the eight epochs that *define* the pre-declared landmarks (mean off-boresight exactly
0.000° — a pure sampling artifact). The full family is reported so the choice is visible:

| Cadence | Trajectory samples | Visible obs | Mean off-boresight | `f_perp` range-only → combined | `sigma_K/K` |
|---|---|---|---|---|---|
| 90 s | 1179 | 8 | 0.000° | 0.2955 → 0.2602 | 1.3263% |
| 30 s | 3535 | 22 | 4.896° | 0.2952 → 0.2638 | 0.8587% |
| **10 s** | 10602 | **50** | 6.817° | 0.2950 → 0.2636 | 0.5825% |

**A 6.25× increase in observation count leaves `f_perp` flat (0.2602 → 0.2636) while `sigma_K/K`
improves by 2.3× ≈ √6.25.** That is precisely magnitude accumulation without direction rotation —
and it resolves the observation-count confound: the production model's low `f_perp` is not an
artifact of having few rows.

Headline runs use the 10 s grid. This changes the **sampling of the trajectory**, not the landmark
selection rule and not the physics.

### §11.1 Literature-anchored noise sweep (10 s grid)

Range-only baseline: `f_perp` = 0.294990, `theta_K` = 17.1569°, `sigma_K/K` = 3.4954%.

| Centroiding σ | Source | `f_perp` | `theta_K` | `sigma_K/K` |
|---|---|---|---|---|
| 0.10 px | dedicated-camera best case | 0.319308 | 18.6211° | **2.9746%** |
| 0.25 px | LONEStar best in-flight | **0.325272** | 18.9821° | 3.1257% |
| 1.00 px | LONEStar worst in-flight | 0.324580 | 18.9401° | 3.1740% |
| 2.50 px | Federici et al. 2025 nominal | 0.317862 | 18.5337° | 3.2435% |
| 0.001 px | unreachable, shows the asymptote | 0.261783 | — | 0.0932% |

Optical block **alone** (0.25 px, 100 rows): `f_perp` = 0.261735, `theta_K` = 15.1730°.

Note `f_perp` is **not monotone** in measurement precision: it is a direction metric, not a quality
metric. At very high optical weight the combined system tends toward the optical block's own value
(≈0.262); the peak at 0.325 is a genuine mixture effect from combining two blocks with different
Jacobian directions. Reporting the optical-only value alongside the combined value is what makes the
combined number interpretable.

### §11.2 Classification

| Quantity | Value |
|---|---|
| `f_perp` range-only | 0.294990 |
| `f_perp` optical alone | 0.261735 |
| `f_perp` best combined | 0.325272 |
| `sigma_K/K` range-only | 3.4954% |
| `sigma_K/K` best combined | 2.9746% |
| **Operational improvement factor** | **1.175×** |
| R1O's published claim (for comparison, not as a target) | `f_perp` 0.8757, `sigma_K/K` 0.19% |

`OPTICAL_INFORMATION_CLASS = MARGINAL_DIRECTION_GAIN`

The threshold is stated rather than applied silently. R1O's decision case for this observable was a
**step change** in K identifiability — an ~18× `sigma_K` improvement with K nearly orthogonal to the
state subspace. The criterion used here is therefore: a step change requires ≥3× `sigma_K`
improvement **and** `f_perp` ≥ 0.60; a marginal gain is any measurable improvement below that. The
production model returns a real but small directional gain that does not approach the decision case
R1O's claim rested on.

---

## §12. Error sweeps

### §12.1 Random attitude knowledge error (enters the weight in quadrature — legitimately, because independent per-observation error *is* measurement noise)

| Attitude σ | px equivalent | Source | `f_perp` | `sigma_K/K` |
|---|---|---|---|---|
| 8 arcsec | 0.351 | BCT XACT in-orbit (MinXSS) | 0.325690 | 3.1433% |
| 10 arcsec | 0.397 | Apollo sextant RMS sighting | 0.325739 | 3.1477% |
| 40 arcsec | 1.259 | compact tracker pixel scale | 0.323654 | 3.1841% |
| 120 arcsec | 3.711 | coarse tracker, 3σ roll | 0.312234 | 3.3022% |
| 103 arcsec (0.5 mrad) | 3.192 | R1O's conservative floor | 0.314524 | 3.2781% |

Erosion across the entire literature-supported attitude range is mild — because the gain being
eroded is itself small. R1O reported "severe erosion (70–92%)" of a gain that §10 shows did not
exist.

### §12.2 Attitude / boresight bias — the two distinct formulas

R1O-D established that an **unmodeled** coherent error and a **solved-for** nuisance parameter are
different questions that must not share a formula. Both are computed for the same physical bias:

| Bias | Case B: K point-estimate shift (omitted-variable) | Case C: `sigma_K/K` when solved for |
|---|---|---|
| 1 arcsec | −5.641e-06 (**−0.0564%** of K) | 3.1261% |
| 8 arcsec | −4.513e-05 (−0.4513%) | 3.1267% |
| 40 arcsec | −2.257e-04 (−2.2565%) | 3.1268% |
| 120 arcsec | −6.770e-04 (**−6.7696%**) | 3.1268% |

The R1O-D pattern repeats exactly: a coherent bias shifts the K point estimate by percent-level
amounts while the formal covariance **barely moves** (3.1261% → 3.1268% across a 120× bias range).
Formal covariance alone would not reveal this risk.

### §12.3 Landmark catalog error — random and coherent

Random per-observation (identification jitter):

| Map σ | px equivalent | `f_perp` | `sigma_K/K` |
|---|---|---|---|
| 50 m | 3.129 | 0.314817 | 3.2750% |
| 200 m | 12.477 | 0.298047 | 3.4596% |

Coherent per-landmark — the physically correct model, since a catalog entry is wrong the *same way*
in every image of it:

| Map bias | Case B: K point shift | Case C: `sigma_K/K` when solved |
|---|---|---|
| 50 m | −3.709e-04 (**−3.7094%** of K) | 3.4094% |
| 200 m | −1.484e-03 (**−14.8375%** of K) | 3.4792% |

**A 200 m coherent catalog error shifts K by ~15% of truth while the formal covariance moves by
under 0.04 percentage points.** Treating a per-landmark catalog error as white noise — which
inflating σ in quadrature implicitly does — understates the real risk by more than two orders of
magnitude. This is the single most operationally important number in the phase after §10.

### §12.4 Camera calibration (focal length)

| df/f | `f_perp` | `sigma_K/K` |
|---|---|---|
| 1e-4 | 0.325272 | 3.1257% |
| 1e-3 | 0.325270 | 3.1256% |
| 1e-2 | 0.325252 | 3.1250% |

Negligible, and for a structural reason worth stating: a focal-length error rescales the image plane
and the Jacobian *together*, so it largely cancels in the information geometry. It would matter for
absolute residuals, which this phase does not evaluate.

### §12.5 Combined realistic case (components pre-declared from the literature contract)

Centroiding 0.25 px (LONEStar best in-flight) + attitude 10 arcsec (Apollo sextant class) + catalog
50 m random → effective σ = 3.1437 px.

`f_perp` = 0.314746, `theta_K` = 18.3455°, **`sigma_K/K` = 3.2757%** versus range-only 3.4954% —
an operational improvement of **1.07×**.

---

## §13. Classifications

| Field | Value |
|---|---|
| `OPT_LITERATURE_CONTRACT_GATE` | PASS |
| `REFERENCE_SPACECRAFT_OPTICAL_NAV_CAPABILITY` | UNKNOWN |
| `CAMERA_PROJECTION_ORACLE_GATE` | PASS |
| `CAMERA_ATTITUDE_FRAME_GATE` | PASS |
| `LANDMARK_POSITION_ORACLE_GATE` | PASS |
| `OPTICAL_GEOMETRIC_VISIBILITY_GATE` | PASS |
| `OPTICAL_STATE_JACOBIAN_GATE` | PASS |
| `OPTICAL_K_COMPOSITION_SENSITIVITY_GATE` | PASS |
| `OPTICAL_K_E2E_SENSITIVITY_GATE` | PASS |
| `R1COV_OPTICAL_COVARIANCE_PATH_GATE` | PASS (reuses the R1COV-qualified QR square-root path exclusively; the normal matrix is never formed) |
| `DIRECT_MEASUREMENT_K_DEPENDENCE` | NO |
| `LANDMARK_SELECTION_NOT_K_TUNED` | YES |
| `OPTICAL_LIGHT_TIME_SIGNIFICANCE` | NEGLIGIBLE |
| `OPTICAL_ABERRATION_SIGNIFICANCE` | SIGNIFICANT (`CHARACTERIZED_NOT_IMPLEMENTED`) |
| `OPTICAL_TIME_TAG_CONTRACT` | SHUTTER_RECEPTION_EPOCH |
| `LANDMARK_POSITION_MODEL` | SPHERICAL_REFERENCE |
| `LANDMARK_CATALOG_FRAME` | MOON_PA_DE440 (versioned, never the generic alias) |
| `DISTORTION_MODEL` | DEFERRED_CHARACTERIZED |
| `FEATURE_DETECTABILITY_MODEL` | CHARACTERIZED_NOT_FULLY_IMPLEMENTED |
| `OPTICAL_INFORMATION_CLASS` | **MARGINAL_DIRECTION_GAIN** |
| `R1O_OPTICAL_SURROGATE_BRIDGE` | **SURROGATE_INVALIDATED_BY_TRANSPOSED_PHI** |
| `OPTICAL_OPERATIONAL_REALISM` | LIMITED — representative camera, no mission asset, no distortion or photometric model |
| `PROVENANCE` | SYNTHETIC_CAMPAIGN_TRUTH |

### Pre-declaration (hard block §75)

The landmark set is imported **verbatim** from Phase 17-R1O's committed
`_landmark_latlon_from_nadir`: nadir-ground-track points at epoch fractions (0.0, 0.15, 0.3, 0.45,
0.6, 0.75, 0.9, 1.0). That rule was fixed in a prior phase, before this phase's production model
existed, so no production K result could have influenced it. Landmark locations and schedules were
never adjusted after inspecting K observability. `LANDMARK_SELECTION_NOT_K_TUNED = YES`.

### Hard blocks §70–§76 — status

| Block | Status |
|---|---|
| §70 — no idealized LOS angle without camera/image-plane geometry | Satisfied: the observable is a pinhole image-plane (u, v) pair |
| §71 — camera attitude must be in the production chain | Satisfied: explicit camera-from-inertial DCM with three separable error sources |
| §72 — landmark body-fixed coordinates and inertial transform must be qualified | Satisfied: MOON_PA_DE440, oracle-tested |
| §73 — Jacobian must not rest on one arbitrary FD step | Satisfied: analytic Jacobian, six-step convergence sweep, quadratic region confirmed |
| §74 — no headline K sigma from the floored normal-matrix covariance | Satisfied: R1COV square-root path exclusively |
| §75 — landmark selection must be pre-declared | Satisfied: rule inherited from committed prior-phase code |
| §76 — no geometry scope creep | Satisfied: single trajectory, single Sun geometry, no inclination/altitude/beta sweeps |

---

## §14. What was NOT run, and why

Per the phase's own sequencing (§50/§51) and R1O-D's precedent: **no nonlinear BLS/SRIF/SR-UKF
integration and no holdout prediction study were run.** The information gate did not deliver the
step change that would justify estimator integration, so running it would spend significant effort
characterizing a 1.07–1.175× effect. `NONLINEAR_ESTIMATOR_INTEGRATION = NOT_RUN_BY_GATE`.

Also explicitly **not** done, per §10 and hard block §76: no geometry generalization (inclination,
altitude, Sun geometry, beta angle, eclipse richness, Earth-view geometry, DSN scheduling, A/m,
K_truth, spacecraft configuration).

**Not corrected in place:** R1O's committed `chain_to_augmented_columns` was deliberately left
unmodified. This phase reports what R1O published *and* what it should have been; silently rewriting
the published script would destroy the first of those. The correction is implemented in this phase's
own `corrected_landmark_arc` for the bridge comparison. **Repairing R1O's published results is
recommended as a follow-up phase with owner authorization**, since it changes conclusions in an
already-closed phase.

---

## §15. Regression

| Metric | Without new tests | With new tests |
|---|---|---|
| Tests | 1433 | 1460 |
| Passed | 1392 | 1419 |
| Failures | 2 | 2 |
| Errors | 10 | 10 |
| Skipped | 29 | 29 |

The entire delta is exactly the 27 new tests. The non-passing set is **identical** to the R1O-D
baseline and consists of the two known pre-existing environment artifacts (`FA-06` isolated-import
guard, tripped by the workspace-root `pytest.ini` injecting a sibling worktree onto `pythonpath`;
and the R2 protected-tree byte gate) plus the 10 dependent R2 errors. **Zero new failures, zero new
errors, zero new skips.**

---

## §16. Limitations — stated, not buried

1. **The camera is a declared representative design**, not a mission asset
   (`REFERENCE_SPACECRAFT_OPTICAL_NAV_CAPABILITY = UNKNOWN`). Results are conditional on it.
2. **Stellar aberration is significant (0.035 px) and not implemented**, and because it is
   velocity-proportional it is not a pure nuisance — the coupling to the state was not quantified.
3. **Spherical Moon, no topography.** Real landmark elevations of a few km would enter as
   landmark-position error; §12.3's coherent-bias result bounds its severity and it is severe.
4. **No lens distortion, no photometric detectability model.** Illumination was not used to gate
   observations, so the visible-observation counts here are optimistic.
5. **One trajectory, one Sun geometry.** No claim of generality across orbits — deliberately, per
   §76, and reinforced by this phase's own finding: sweeping geometry with an unqualified model
   would only have generalized a modeling error.
6. **The `f_perp` non-monotonicity in §11.1** means single-number comparisons of `f_perp` across
   different noise levels are not meaningful without the optical-only value alongside.

---

## §17. Verdict fields

```
PHASE                                       = 17-R1O-OPT
R1O_OPT_INPUT_GATE                          = PASS
OPT_LITERATURE_CONTRACT_GATE                = PASS
REFERENCE_SPACECRAFT_OPTICAL_NAV_CAPABILITY = UNKNOWN
CAMERA_PROJECTION_ORACLE_GATE               = PASS
CAMERA_ATTITUDE_FRAME_GATE                  = PASS
LANDMARK_POSITION_ORACLE_GATE               = PASS
OPTICAL_GEOMETRIC_VISIBILITY_GATE           = PASS
OPTICAL_STATE_JACOBIAN_GATE                 = PASS
OPTICAL_K_COMPOSITION_SENSITIVITY_GATE      = PASS
OPTICAL_K_E2E_SENSITIVITY_GATE              = PASS
R1COV_OPTICAL_COVARIANCE_PATH_GATE          = PASS
DIRECT_MEASUREMENT_K_DEPENDENCE             = NO
LANDMARK_SELECTION_NOT_K_TUNED              = YES
OPTICAL_LIGHT_TIME_SIGNIFICANCE             = NEGLIGIBLE
OPTICAL_ABERRATION_SIGNIFICANCE             = SIGNIFICANT_NOT_IMPLEMENTED
OPTICAL_TIME_TAG_CONTRACT                   = SHUTTER_RECEPTION_EPOCH
LANDMARK_POSITION_MODEL                     = SPHERICAL_REFERENCE
LANDMARK_CATALOG_FRAME                      = MOON_PA_DE440
DISTORTION_MODEL                            = DEFERRED_CHARACTERIZED
FEATURE_DETECTABILITY_MODEL                 = CHARACTERIZED_NOT_FULLY_IMPLEMENTED
OPTICAL_INFORMATION_CLASS                   = MARGINAL_DIRECTION_GAIN
F_PERP_RANGE_ONLY                           = 0.294990
F_PERP_OPTICAL_ALONE                        = 0.261735
F_PERP_BEST_COMBINED                        = 0.325272
SIGMA_K_FRAC_RANGE_ONLY                     = 3.4954%
SIGMA_K_FRAC_BEST_COMBINED                  = 2.9746%
SIGMA_K_FRAC_COMBINED_REALISTIC             = 3.2757%
OPERATIONAL_IMPROVEMENT_FACTOR              = 1.175x
R1O_OPTICAL_SURROGATE_BRIDGE                = SURROGATE_INVALIDATED_BY_TRANSPOSED_PHI
R1O_PUBLISHED_F_PERP_REPRODUCED             = 0.876626 (published 0.8757)
R1O_F_PERP_WITH_PHI_CORRECTED               = 0.253154
PHI_TRANSPOSITION_DELTA_F_PERP              = +0.623473
MEASUREMENT_MODEL_DELTA_F_PERP              = -0.006998
PRIOR_PHASE_DEFECT_FOUND                    = YES (phase17_r1o_core.chain_to_augmented_columns)
PRIOR_PHASE_DEFECT_REPAIRED_IN_PLACE        = NO (reported; repair needs owner authorization)
AFFECTED_PRIOR_RESULTS                      = R1O DDOR surrogate, R1O landmark surrogate,
                                              R1O Earth-LOS control, R1O-D bridge surrogate side
UNAFFECTED_PRIOR_RESULTS                    = R1O range-only baseline, R1O-D production DDOR,
                                              all K columns (S_K needs no reshape)
ATTITUDE_BIAS_K_SHIFT_AT_120_ARCSEC         = -6.7696% of K_truth
MAP_BIAS_K_SHIFT_AT_200_M                   = -14.8375% of K_truth
FORMAL_COVARIANCE_DETECTS_BIAS_RISK         = NO (3.1261% -> 3.1268% across a 120x bias range)
NONLINEAR_ESTIMATOR_INTEGRATION             = NOT_RUN_BY_GATE
GEOMETRY_GENERALIZATION                     = DEFERRED_BY_SCOPE
OPTICAL_OPERATIONAL_REALISM                 = LIMITED
PROVENANCE                                  = SYNTHETIC_CAMPAIGN_TRUTH
REGRESSION_TOTAL                            = 1460
REGRESSION_PASSED                           = 1419
REGRESSION_NEW_FAILURES                     = 0
NEW_PERMANENT_TESTS                         = 27
PRODUCTION_FILES_ADDED                      = 1 (lunar_od/lunar_landmark_optical.py)
PRODUCTION_FILES_MODIFIED                   = 0
PUSH                                        = NONE
MERGE                                       = NONE
MAIN_CHANGED                                = NO
ORIGIN_MAIN_CHANGED                         = NO
PHASE17_R1O_OPT_GATE                        = CHARACTERIZATION_COMPLETE
```

---

## §18. Recommendation

1. **Do not pursue lunar landmark optical navigation as a K_SRP identifiability solution** on the
   evidence available. The production model delivers a 1.07–1.175× improvement, not a step change.
2. **Authorize a repair phase for Phase 17-R1O.** Its published surrogate conclusions — including
   the ranking that sent R1O-D to ΔDOR first — rest on transposed state-design matrices. The range
   baseline and R1O-D's production ΔDOR result survive; the surrogate comparisons do not.
3. **Carry the coherent-catalog-error result forward.** A 200 m per-landmark bias shifts K by ~15%
   of truth while formal covariance moves by <0.04 points. Any future optical work must treat
   catalog error as coherent per landmark, never as white noise.
4. **Both R1O candidate observables have now failed production qualification** — ΔDOR on physics,
   landmark optical on a defect in the analysis that motivated it. The open question is no longer
   "which new observable?" but whether K_SRP is identifiable from this trajectory class at all,
   which R1M's Schur sum-rule result already pointed toward.
