# Lunar Orbit Determination Project Context

> Canonical onboarding document for future Claude sessions. Read this before
> changing anything in the repository.
> Last verified against the repository: **2026-07-11**.
> Branch `feature/lunar-j2-force-models`, HEAD `5401db8`, 13 local commits, never pushed.
> This version supersedes and merges the earlier Turkish-language edition of the
> same date; all statements below were re-verified against Git history, source
> files, and generated reports (not copied blindly).
> Repository structure, commit history, tests, and recorded campaign values were
> re-verified. Physical interpretations and nmax recommendations remain
> Level-1 internal model-vs-model conclusions unless an external validation
> source is explicitly cited.

## 1. Executive Summary

Comparative study of orbit-determination (OD) estimators — **BLS-LM, SRIF, SR-UKF**
— for low lunar orbits (LLO). TÜBİTAK UZAY **AYAP-1** scholarship project and
İTÜ Space Engineering graduation thesis of Cemil Eray Halıcı (2026). The project
started in MATLAB (GMAT cross-validated) and was ported to Python; the Python
layer (`lunar_od/`) is now the primary development environment.

Current state of work:

- The **lunar gravity campaign** (J2 framework → spherical-harmonics engine →
  real GRAIL validation → orbit-effect and sensitivity campaigns) is complete
  through Phase 13G-c1 and fully committed (13 commits, `cd6b4a8`…`5401db8`).
- Headline scientific result: **classical J2-only lunar dynamics is inadequate
  in every tested LLO regime** (6–53 km/day trajectory error vs. a full GRAIL
  field at nmax 64–128); the recommended truncation degree is orbit-regime
  dependent: nmax=64 at 200 km and nmax=32 at 500 km are
  **threshold-demonstrated** by their N→2N ladder steps, while nmax=128 at
  100 km / polar is a **recommendation from available evidence** (the matching
  128→256 closure was not run for those exact cases; see Section 14).
- **Harmonics are acceleration-only**: no gravity gradient / STM support yet, so
  BLS-LM/SRIF cannot use them; scenario-runner threading (Phase 13B2b) is queued.
- The working tree contains **parallel, uncommitted measurement-model work**
  (light-time, aberration, measurement-model profiles) that is *not* part of the
  gravity phases and must never be mixed into them (Section 20).

## 2. Current Repository and Git State

- Repo root: `C:\Users\erayh\Documents\Python\Grad\python_port` (the parent
  `Grad/` folder holds the MATLAB reference code and `pytest.ini`).
- Remote: `https://github.com/halici21/lunar-orbit-determination.git`.
  `origin/main` = `3867105`. The working branch has **never been pushed**.
- Branch: `feature/lunar-j2-force-models`, HEAD `5401db8`
  ("Add lunar gravity sensitivity campaign"), 13 commits ahead of `main`.
- Untracked (intentional): `PROJECT_CONTEXT.md` (this file),
  `.claude/settings.json`, `exports/` (visible artifact copy of the phase 6 J2
  campaign).
- Modified working tree (as of 2026-07-11): 42 files. **Two distinct groups:**
  1. **Whitespace/line-ending churn only** — all 21 `.claude/skills/*/SKILL.md`,
     `SPICE_CN_CNPLUS_VALIDATION.md`, `UI_README.md`, `docs/*.md`,
     `desktop_app/ui/pages/*.ui`, `desktop_app/models/scenario_model.py`,
     `lunar_od/thesis_matrix.py`, `lunar_od/visibility.py`. Verified:
     `git diff --stat --ignore-all-space` collapses these to zero.
  2. **Substantive parallel measurement-model work** (~1,063 insertions):
     `lunar_od/measurements.py`, `estimators.py`, `filters.py`,
     `scenario_config.py`, `scenarios.py`, `__init__.py`, `reporting.py`,
     `desktop_app/controllers/analysis_controller.py`,
     `examples/run_scenario_config.py`, `tests/test_measurements.py`,
     `tests/test_reporting.py`, `tests/test_scenario_config.py`. See Section 20.
- Hygiene verified: `git ls-files data/gravity results` → empty;
  `data/gravity/`, `results/`, `*.mat` are gitignored.

## 3. Project Goals and Scientific Scope

- Compare BLS-LM, SRIF, and SR-UKF estimation performance for LLO OD with
  ground-station radiometric tracking (range, az/el, range-rate, two-way
  Doppler), under realistic visibility (elevation masks + lunar occultation,
  İTÜ Ayazağa + DSN stations).
- Establish a defensible **truth-model hierarchy** (point mass → J2 →
  low-degree harmonics → high-degree GRAIL field) with quantified inter-layer
  effects, feeding thesis conclusions and estimator-model selection.
- Working discipline is **phase-gated**: every phase gets an
  implementation-before plan, user ACCEPT, a test gate, a scientific-validation
  gate, and an explicit user-requested commit checkpoint (file-level `git add`,
  never `git add .`, never push).

## 4. Repository Architecture

Main package `lunar_od/` (module → responsibility):

| Module | Responsibility |
|---|---|
| `constants.py` | Central physical constants (GM from gm_de431, radii, J2) with the **pairing rule**: a J2 value is meaningful only with its own model's reference radius (Moon: J2=2.0346e-4 ↔ R=1,737,400 m; Earth: EGM96 J2=1.08262668e-3 ↔ R=6,378,136.3 m) |
| `dynamics.py` | Moon-centered dynamics: `f3body_moon`, `propagate_state` (6-state), `propagate_augmented_state` (42-state, column-major [x(6); Φ(36)]), J2 terms, harmonics splice + guards |
| `accelerated.py` | Numba kernels (RHS, J2, Pines harmonics) with NumPy fallback; scalar-tuple `inline="always"` design, no heap allocation on the hot path |
| `force_models.py` | Generic body-J2 helper: `body_j2_acceleration`, `body_j2_gravity_gradient` (single code path for Moon + Earth J2) |
| `gravity_harmonics.py` | Pines spherical-harmonic engine (pure-Python reference): `SphericalHarmonicGravityModel`, `spherical_harmonic_acceleration` |
| `gravity_model_loader.py` | SHADR `.tab` / ICGEM `.gfc` loader: `load_lunar_gravity_model`, `resolve_gravity_dir`, `describe_model`; metadata validation, SHA-256 provenance |
| `lunar_frames.py` | MOON_PA rotation provider: `sample_moon_pa_rotations` (SPICE pre-sampling), `nearest_rotation_at_time` (pure-NumPy nearest lookup); no import-time `furnsh` |
| `ephemeris.py` | Moon-centered Earth/Sun ephemeris from MATLAB `.mat` export (PCHIP interpolation) |
| `spice_loader.py` | SPICE kernel resolution/loading — DE421 default set; unchanged by the gravity phases |
| `scenario_config.py` | JSON scenario configuration incl. the 13B2a lunar-harmonics fields (with temporary not-yet-consumed guard, Section 15) |
| `scenarios.py` | Scenario execution / runner |
| `measurements.py`, `radiometrics.py`, `measurement_ingestion.py` | Measurement models (⚠ changing in the parallel uncommitted work) |
| `estimators.py` | BLS-LM, SRIF |
| `filters.py` | SR-UKF |
| `observability.py`, `monte_carlo.py`, `diagnostics.py`, `q_tuning.py`, `adaptive_tuning.py`, `noise_models.py` | OD analysis support |
| `visibility.py`, `geometry.py`, `orbit.py` | Station visibility (elevation mask + occultation), geometry, orbital elements |
| `reporting.py`, `thesis_matrix.py`, `config.py` | Reporting and thesis experiment matrix |

Other top-level areas: `examples/` (all phase/campaign scripts, phase0 →
phase13g), `tests/` (46 test modules; `tests/fixtures/gravity/` holds tiny
committed synthetic SHADR/.gfc fixtures), `desktop_app/` (PyQt5 MVP),
`results/` (generated, gitignored), `data/gravity/` (real GRAIL files, local
only, gitignored), `exports/` (intentionally untracked artifact copies),
`ephemeris_data.mat` (169 MB DE421-derived Moon-centered ephemeris, 10 s grid,
gitignored).

## 5. Runtime and Data Flow

Gravity-enabled propagation flow (direct API — the only supported path today):

1. **Scenario config** (JSON) → `ScenarioConfig` validation. Harmonics fields
   are validated but *not yet consumed* by the runner (Section 15).
2. **Ephemeris**: `ephemeris.py` interpolates Earth/Sun positions from
   `ephemeris_data.mat`. The `.mat` `first_jd` is **TDB**
   (`et0 = (first_jd − 2451545.0)·86400`; campaign epoch 2027-03-02,
   first_jd = 2461467.480984).
3. **Force models**: point mass + Earth/Sun third body (indirect), optional
   Moon J2 / Earth J2 via `force_models.py`.
4. **Harmonics setup**: `load_lunar_gravity_model(path, nmax=…)` → canonical
   SI, fully-normalized model carrying its own GM/R_ref.
5. **Rotation grid**: `sample_moon_pa_rotations(et0, t_grid, frame="MOON_PA_DE421",
   load_kernels=False)` before propagation; margin `max(2·cadence, 120 s)`.
6. **Propagation**: `propagate_state(teval, s0, MU_M, MU_E, MU_S, get_earth,
   get_sun, harmonic_model=model, harmonic_rotation=(t_grid, rots))`. Inside the
   RHS: pure-NumPy nearest-neighbor rotation lookup (tie → lower index;
   out-of-range/monotonicity errors raise `ValueError`, no clamping). Python and
   Numba paths share the same lookup → parity is structural. Zonal-only models
   may pass a constant (3,3) matrix instead.
7. **Measurements → residuals (observed−computed) → estimator/filter →
   reporting** — the established OD chain (currently J2-level dynamics only).
8. **Desktop app** (`desktop_app/`, `lunar_od_app.py`) wraps scenario building
   and analysis; not part of the gravity phases.

Complete today: flows 2–6 via the direct API; the OD chain with J2-level
dynamics. Partial: scenario-runner consumption of harmonics config (13B2b
queued); DE440 as a production `spice_loader` profile (13B2c queued).

## 6. Claude Skills and Working Method

`.claude/skills/` contains **21 project skills** (the folder name matches each
YAML `name`; the YAML `description` is the trigger) plus `_shared/` with 31
reference documents. Skills relevant to selection:

| Skill | Use for |
|---|---|
| `lunar-od-repo-navigator` | Locating code, tracing data flow, mapping results to producing scripts — use before proposing edits |
| `lunar-od-dynamics-spice` | Force models, propagation, STM, frames, SPICE kernels/ephemerides |
| `lunar-od-measurement-physics` | Range/az-el/range-rate/Doppler, light time, aberration, residuals, Jacobians |
| `lunar-od-estimator-engineering` | BLS-LM/SRIF/SR-UKF internals, covariance, comparison fairness |
| `lunar-od-scenario-config` | Scenario JSON workflow, `scenario_config.py`/`scenarios.py` |
| `lunar-od-campaign-design` | Designing fair, reproducible experiments before running |
| `lunar-od-validation-gates` | Scientific/numerical validation planning for any code or model change |
| `lunar-od-software-quality-gates` | Lint/type/CI/dependency hygiene (software, not science) |
| `lunar-od-result-reproducer` / `lunar-od-result-validator` | Tracing a result to script→config→artifact→commit / judging plausibility, fairness, truth-model level |
| `lunar-od-test-strategist`, `lunar-od-continuous-verification` | Test design; recurring test/CI scheduling |
| `lunar-od-model-review`, `lunar-od-statistical-diagnostics`, `lunar-od-figures`, `lunar-od-literature-review`, `lunar-od-thesis-writing` | Model critique; statistics; plots; literature; thesis text |
| `lunar-od-desktop-app` / `lunar-od-interface-designer` | PyQt5 implementation / UI-UX design (design skill never edits implementation) |
| `lunar-od-performance-optimizer`, `lunar-od-public-repo-curator` | Behavior-preserving speedups; public-release hygiene |

**Routing policy** (`_shared/skill-routing-policy.md`): use the *smallest
sufficient* skill set — one primary skill by default; read at most 1–2 shared
files unless the task demands more; full multi-skill governance only for code
changes, numerical-output changes, baseline design, result acceptance, or
release/thesis-freeze work. Key shared references:
`numerical-contract.md` and `numerical-tolerance-policy.md` (tolerance and
parity rules), `truth-model-hierarchy.md` (every accuracy claim states its
truth level — all gravity-campaign numbers are **Level 1**, internal
model-vs-model, not external-tool validation), `software-quality-checklist.md`,
`ci-quality-gates.md`, `test-design-guidelines.md`,
`dependency-hygiene-checklist.md`, `repo-map.md`, `result-artifact-map.md`,
`terminology.md`. Do not edit skills unless that is the explicit task.

## 7. Phase and Commit Timeline

All 13 branch commits, oldest first (verified against `git log`):

| Commit | Phase | Content |
|---|---|---|
| `cd6b4a8` | 0–5 | J2 force-model framework: `constants.py`, `force_models.py`, dynamics/accelerated updates, `test_force_models.py`, `test_earth_j2.py` |
| `948054d` | 0–9 | Verification + scenario-comparison campaigns: `phase0_*`, `phase5*`, `phase6_*`, `phase7c`, `phase8` example scripts |
| `3e45954` | 11A | Low-degree harmonics prototype: Pines vs. classical spherical-gradient (Cunningham-style) cross-check |
| `a7ba558` | 11B | Production pure-Python Pines engine (`gravity_harmonics.py`) |
| `3458460` | 11C | Numba twin + Python↔Numba parity gate |
| `9bbc75e` | 12 | GRAIL loader (`gravity_model_loader.py`): SHADR/.gfc, metadata validation, fixtures |
| `010dd3a` | 13A | MOON_PA rotation provider (`lunar_frames.py`) |
| `54806d0` | 13B | 6-state dynamics splice + guards |
| `d95eef9` | 13C | Low-degree harmonics validation campaign + cadence preflight |
| `461869d` | 12B/12BX | Real GRAIL validation campaign (`phase12b_real_grail_validation.py`) + `tests/test_real_grail_optional.py` |
| `2a331dc` | 13B2a | Scenario-config lunar-harmonics validation (+ `test_scenario_config_harmonics.py`) |
| `a081dcb` | 13G-b | Gravity orbit-effect validation campaign (`phase13g_gravity_orbit_effects.py`; initial 18-test Phase 13G suite) |
| `5401db8` | 13G-c1 | Gravity sensitivity campaign; expanded the Phase 13G test file to 25 tests (HEAD) |

Pre-branch history (on `main`): the full OD infrastructure, Jan–Jun 2026 —
GMAT/MATLAB cross-validation; SPICE/MICE integration; ITRF93 station chain;
visibility; measurement models with analytic Jacobians + finite-difference
verification; BLS-LM; SRIF/QR with cold/hot start and arc-to-arc handoff;
SR-UKF; Monte Carlo; adaptive Q-tuning; two-way Doppler; light-time and stellar
aberration; a 28-day İTÜ visibility campaign (114 arcs, 105 usable; noise-free
geometric range-rate SRIF median final position error ~1.3 cm); plus the Claude
skills package (`3867105`).

## 8. Force-Model Architecture

- **Constants pairing rule** (`constants.py`): GM values from gm_de431
  (MU_MOON = 4902.800066163796e9 m³/s²); each J2 is paired with its model's
  reference radius (Section 4 table). Never mix a J2 with a foreign radius.
- **Generic body-J2 helper** (`force_models.py`): Moon J2 was refactored onto
  it bit-identically; Earth J2 added with `earth_j2_mode="indirect"` default
  (physically correct relative form for a Moon-centered frame; `"direct"` is
  debug-only). Earth orientation uses a fixed J2000 mean-pole approximation —
  documented as not high-accuracy.
- **Campaign results (1/7/30/120/360 days, 5 scenarios)**: Moon J2 dominates
  Earth J2 by ~3–4 orders (at 120 d: Earth-J2 separation 9.6 m vs. Moon-J2
  3.27e6 m; at 360 d Earth J2 ≈ 403 m). Separations are oscillatory (final ≪
  max). The J2×J2 interaction term is small vs. Moon J2 but comparable to Earth
  J2 — never call it negligible without that reference. 360-day runs require
  the **native 10 s ephemeris grid** (60 s decimation corrupts small signals).
- `.mat` ↔ SPICE DE421: Earth position agreement ≤ 1.5 cm, Sun sub-metre,
  velocities tiny; **TDB** is the correct `first_jd` interpretation (UTC shown
  wrong). These are ephemeris-*source* differences, not force-model errors.
- Consequence: wiring Earth J2 into the OD estimator (old "Phase 6B") is low
  priority; the scientific path went to GRAIL harmonics.

## 9. Lunar Harmonics Architecture

Decisions fixed in Phase 10–13 (all enforced in code, not convention):

- **Pines formulation** (Cartesian direction cosines) — no polar singularity;
  polar LLO is a primary scenario. The classical formulation remains only as a
  prototype cross-validation oracle (proven to lose the horizontal m=1
  acceleration exactly at the pole).
- **Internal representation**: fully-normalized C̄nm/S̄nm in a square `[n, m]`
  array. Zonal bridge: **C̄n0 = −Jn/√(2n+1)**, so `C̄20 = −J2/√5` and
  `J2 = −√5·C̄20` (sign and direction critical). **J3 is not hard-coded** — it
  is the engine's (n=3, m=0) term.
- **Model carries its own GM and R_ref** (GRAIL R_ref = 1,738,000 m ≠
  R_MOON_M = 1,737,400 m used for altitude — never conflate; the loader returns
  model values, `constants` is only a sanity reference).
- Only **n ≥ 2** perturbation (n=0 point mass is the caller's job; n=1 ignored).
- **Acceleration-only**: no harmonic gradient/STM. STM+harmonics requests raise
  *"lunar harmonics gradient not implemented; use 6-state propagation or
  disable lunar harmonics for STM"*. Silent acceleration/gradient inconsistency
  is forbidden.
- **Double-count guard**: a model containing C̄20 cannot combine with
  `j2_moon != 0` → `ValueError` at the propagate entry point. Earth J2 is a
  different body and composes freely.
- **Frame rule**: any m>0 (tesseral/sectoral) coefficients require an
  **epoch-dependent MOON_PA rotation grid** (the Moon rotates synchronously; a
  fixed frame freezes C22 in inertial space). Constant matrix allowed only for
  zonal-only models; a fixed matrix with tesseral terms is rejected.
- **No SPICE on the hot path**: rotations are pre-sampled; harmonics-off
  behavior is bit-identical (exact zero difference).

API sketch:

```python
from lunar_od.gravity_model_loader import load_lunar_gravity_model
from lunar_od.lunar_frames import sample_moon_pa_rotations
from lunar_od.dynamics import propagate_state

model = load_lunar_gravity_model(path, nmax=64)
t_grid = np.arange(-120.0, T + 120.0, 60.0)
rots = sample_moon_pa_rotations(et0, t_grid, frame="MOON_PA_DE421", load_kernels=False)
traj = propagate_state(teval, s0, MU_M, MU_E, MU_S, get_earth, get_sun,
                       harmonic_model=model, harmonic_rotation=(t_grid, rots))
```

## 10. Gravity Data and SPICE Kernel Profiles

Real GRAIL models under `data/gravity/` (present locally, gitignored,
untracked; each directory has the PDS-named coefficient `.tab`, its `.lbl`,
`SOURCE.txt` provenance, and `SHA256SUMS.txt`):

| Model | Files | Degree | Frame | Role |
|---|---|---|---|---|
| GRGM660PRIM | `grgm660prim/gggrx_0660pm_sha.tab` (26.7 MB) + `.lbl` | 660×660 | **DE421 PA** (frame-exact with the existing kernel chain) | Primary validation model |
| GRGM1200L | `grgm1200l/gggrx_1200l_sha.tab` (87.9 MB) + `.lbl` | 1199×1199 | DE430 heritage (via GRGM1200A; label silent) | **Loader sanity only** — excluded from propagation (no validated DE430 lunar-PA kernel chain) |
| GL1800F | `gl1800f/jggrx_1800f_sha.tab` (198 MB) + `.lbl` | 1800×1800 | **DE440 PA** | High-resolution independent cross-check |

All three carry **R_ref = 1,738,000 m** (model radius — not the 1,737,400 m
physical radius used for altitude). Real loaded values: C̄20 ≈ −9.088e-5 (all
three; derived J2 ~0.12% from the constants value), C̄22 ≈ 3.467e-5,
S̄22 ≈ 1e-10 ≈ 0 (PA-frame signature).

SPICE kernels live outside the repo at `C:\Users\erayh\Documents\mice\kernels`:

- **DE421 profile** (default): `naif0012.tls.txt`, `de421.bsp`,
  `earth_assoc_itrf93.tf.txt`, `moon_080317.tf.txt`,
  `earth_2025_250826_2125_predict.bpc`, `moon_pa_de421_1900-2050.bpc`,
  `gm_de431.tpc.txt`, `pck00010.tpc.txt`.
- **DE440 minimal rotation profile** (campaign-script level only):
  `naif0012.tls.txt`, `moon_de440_220930.txt`, `moon_pa_de440_200625.bpc`
  (`de440.bsp`/`pck00011` deliberately excluded — `pxform` only needs
  orientation). `moon_de440_250416.tf` was not required for the validated
  Phase 12B/13G campaign profile: the locally available
  `moon_de440_220930.txt` supplied the required DE440 lunar-frame
  definitions. Adoption of the newer FK remains a separate maintenance
  decision.
- **Alias trap (Rule)**: both lunar frame kernels can redefine the bare
  `MOON_PA` alias, so alias resolution depends on kernel-pool load order.
  Therefore: **never use bare `MOON_PA`** — always `MOON_PA_DE421` /
  `MOON_PA_DE440`, and `spice.kclear()` before every profile switch. Bare
  `MOON_PA` is also rejected at config level.
- Translational ephemeris note: even GL1800F+DE440-rotation runs use the
  DE421-derived `.mat` for Earth/Sun third-body positions (declared). Its
  contribution is expected to be small for the tested third-body tidal terms,
  but Phase 12B/13G did not independently isolate it from coefficient-model
  and PA-frame differences.

## 11. Verification Results (implementation correctness)

- Python↔Numba Moon-J2 parity; STM/gradient finite-difference checks; J2-off
  and third-body regressions (Phase 0, `948054d` scripts).
- Prototype (11A): Pines vs. classical cross-check; C20/J2 equivalence; C22
  longitude behavior; C30/J3 asymmetry; exact-pole behavior (classical loses
  m=1 horizontal acceleration at the pole; Pines does not).
- Engine vs. `body_j2_acceleration` for C20-only: worst ‖Δa‖ ≈ 2–5e-19 m/s².
- Python↔Numba Pines parity (nmax=8, fastmath): ~5e-15 relative; exact-pole and
  near-pole tests pass.
- MOON_PA sampling: orthonormality 3.3e-16; rotation rate 13.1775°/day
  (sidereal ✓).
- Splice: harmonics-off is bit-identical.
- Loader: all three real models parse (660PM 0.6 s, 1200L 2.3 s, 1800F 4.6 s
  full-file); script-local `truncate_model` is bit-identical to direct
  truncated load. Guards re-proven with real models: double-count PASS,
  STM-refusal PASS, Earth-J2 composability PASS.

## 12. Physical Validation Results (Phase 13C, synthetic low degree)

- **60 s rotation cadence ACCEPTED**: cadence-error/C22-signal ratio
  E60/S22 = 2.0e-5 (orbit), 5.2e-4 (day) vs. 0.01 threshold — rotation stepping
  error ≪ harmonic effects.
- C20-only vs. Moon-J2 bridge established; C22, C30, truncation, zonal/full and
  frame effects quantified.
- MOON_PA vs. fixed mean-pole (physical frame difference): 26 m/orbit,
  **1,008 m/day** — the mean-pole/fixed-frame approximation causes meaningful
  trajectory error; m>0 terms matter.
- Performance observation: m>0 + nearest-rotation increases VODE RHS
  evaluations ~10× (3.3k → 40k/day).

## 13. Gravity Orbit-Effect Findings (Phase 13G-b, `a081dcb`)

Baseline: phase6 LLO (a = R+100 km, e = 0.01, i = 45°, perilune ~81.6 km),
1 orbit + 1 day; GRGM660PRIM primary, GL1800F comparison points, frame
diagnostics. Key day-1 numbers (Level 1, model-vs-model):

| Comparison | Result (day 1) |
|---|---|
| **J2-only vs. full@64 (headline)** | **13.87 km** (orbit: 647 m) |
| C20 bridge (C20-only vs. J2-only) | ≈1,018 m/day separation — primarily reflects differences in the C20/J2 value, model reference radius, and model-GM pairing. Its numerical proximity to the independent 1,008 m/day mean-pole frame diagnostic is coincidental; the two experiments isolate different effects and must not be treated as mutual validation |
| C22 effect | 3.95 km |
| Tesseral (full−zonal ↔ direct diagnostic) | 13.1 ↔ 12.6 km — coupling ~4% (decomposition ≈ linear) |
| Cross-model 660@64 ↔ 1800F@64 | 3.5 m (insignificant) |
| 1800F 128→256 | 2.2 m (converged) |
| DE421↔DE440 deliberate wrong pairing (diagnostic) | 0.9 m — negligible |
| Frozen frame ("Moon not rotating") @64 (diagnostic) | 1.7 km (grows with nmax) |
| Fixed mean-pole @64 (diagnostic) | 28 km — the most expensive neglect |

Element physics (day 1): **the sign of argument-of-perilune drift reverses**
(J2-only +0.022 vs. full −0.162 rad/day); **eccentricity drift grows ~80×**
(−1.6e-5 vs. −1.32e-3 /day — the LLO lifetime driver). Inclination drift was
nearly common between the J2-only and full-harmonics cases, making it
consistent with a shared non-harmonic contribution such as third-body
dynamics; Phase 13G-b did not isolate this attribution with a
third-body-on/off experiment. Kepler-only energy sanity 1.5e-10/orbit. Methodology:
Cartesian differences, RTN decomposition, orbital-element drift via
**orbit-averaged estimation** (a plain linear fit leaks short-period signal
into the slope — found by a failing test, replaced with first/last-period-mean
differencing), finite/surface-crossing checks, no dense trajectory storage.

## 14. Gravity Sensitivity Findings (Phase 13G-c1, `5401db8`)

8 single-factor cases (no cross product): altitude 100/200/500 km (circular,
i=45°); inclination 0/30/60/90° (100 km circular); eccentric 80×500 km (i=45°,
e≈0.1036). Per case ≤9 runs: J2-only / C20 / C20+C22 / zonal@64 / full ladder
nmax 8..128. Each case's "1 orbit" window uses its own period. **GL1800F ran
only on S1/S7/S8** (64/128; 256 only S8, runtime-gated) — it was *not omitted*:
it is a high-resolution independent cross-check, the full matrix would
duplicate computation, and observed cross-model differences (0.15–2.2 m/day)
are orders below the km/day effects studied. Key day-1 results
(campaign-specific observations, not universal laws):

| Case | J2-only error | 64→128 | Recommended nmax from available campaign evidence |
|---|---|---|---|
| 100 km, i=45° | 13.5 km | 86 m | **128** — recommended; matched 128→256 closure not run for this exact circular case |
| 200 km, i=45° | **15.1 km (non-monotonic in altitude!)** | 2.1 m | **64** — threshold demonstrated (64→128 ≈ 2.1 m/day) |
| 500 km, i=45° | 6.4 km | 0.15 m | **32** — threshold demonstrated (32→64 ≈ 0.5 m/day) |
| 100 km, i=0° | 11.3 km | 24.6 m | 128 — recommended; threshold not directly closed |
| 100 km, i=30° | 6.1 km (zonal n≥3 jump 8.2 km) | 82 m | 128 — recommended; threshold not directly closed |
| 100 km, i=60° | 34.6 km | 3.8 m | 64 — threshold demonstrated (64→128 ≈ 3.8 m/day) |
| **100 km, polar** | **52.9 km (hardest; tesseral 50.1 km)** | 39 m | 128 — recommended; threshold not directly closed |
| 80×500 km eccentric | 12.8 km | **2.4 m (~1/35 of circular-100)** | 64 — threshold demonstrated (64→128 ≈ 2.4 m/day) |

Evidence labels: *threshold demonstrated* = the N→2N ladder step for the
recommended N was measured below 10 m/day in that exact case.
*recommended; threshold not directly closed / matched 128→256 closure not
run* = the 64→128 step exceeded 10 m/day, so nmax=128 is the recommendation,
but the 128→256 step that would formally close the `<10 m/day` rule was not
run for that case. The only measured 128→256 closures are the 13G-b baseline
compare (≈2.2 m/day) and the S8 GL1800F **selected cross-check only**
(≈0.8 m/day) — supportive, but not substitutes for the missing per-case runs.

Interpretation: the polar result is consistent with broad longitude sampling
and strong tesseral exposure (a fully isolated causal proof was not
performed); in the eccentric case apolune dwell filters high degrees
(perilune-window |ν|<30° ratio ~1.17 on high-degree ladder steps vs. 0.84 for
J2-only, whose error is along-track-drift dominated). No surface crossing in
any case. For the 200-vs-100 km non-monotonicity, a working hypothesis is
accumulated along-track phase beating and sampling of the rotating body-fixed
gravity field; the mechanism was not independently isolated in this campaign.
Baseline-regression rerun was **bit-identical** (13.87 km / 64.6 m / 1,018 m).
The 11-section "Calculation and Rationale" report is
`results/phase13g/phase13g_sensitivity_report.md`, reproducible from the
cumulative JSON store without rerunning the campaign.

## 15. Scenario Configuration Status (Phase 13B2a, `2a331dc`)

Eight additive fields on `ScenarioConfig` (end of dataclass, defaulted →
backward compatible; also in the JSON schema so the UI can see them):
`enable_lunar_harmonics=False`, `lunar_gravity_model_path=None`,
`lunar_gravity_nmax=None`, `lunar_gravity_mmax=None`,
`lunar_gravity_frame="MOON_PA_DE421"`,
`lunar_gravity_rotation_cadence_s=60.0`,
`lunar_gravity_rotation_margin_s=None`, `lunar_gravity_kernel_profile=None`.

Rules (active only when `enable=True`; off = legacy parse bit-identical with
zero filesystem access, proven by mock):

- Explicit model path and explicit nmax required (no auto-pick).
- Bare `MOON_PA` rejected; frame↔kernel-profile mismatch rejected
  (profile `None` derives from frame).
- `enable` + `j2_moon≠0` → double-count `ValueError`.
- `enable` + `bls_lm`/`srif` → STM `ValueError` (they use
  `propagate_augmented_state`; UKF is 6-state and not blocked by this rule).
- Earth J2 + lunar harmonics allowed.
- **Temporary not-yet-consumed guard** (remove in 13B2b only when the runner
  truly consumes the config): an otherwise-valid `enable=True` config is
  rejected with *"enable_lunar_harmonics is not yet consumed by the scenario
  runner (Phase 13B2b); use the direct propagate_state API"* — preserving the
  invariant "if a config is accepted, it is applied."

Helper `scenario_lunar_gravity_model(config)`: off→None; absolute path as-is;
relative → `resolve_gravity_dir()` base; single load at setup. Tests:
`tests/test_scenario_config_harmonics.py` (26, SPICE-free, fixture-based;
constructor tests build the dataclass directly — the guard's planned escape
hatch).

## 16. Measurement and Estimator Status

Committed capability (on `main` + branch): range, azimuth/elevation,
range-rate, two-way counted Doppler; light-time and stellar-aberration layers;
analytic Jacobians with finite-difference verification; BLS-LM, SRIF (QR,
cold/hot start, arc handoff, bias modes), SR-UKF; Monte Carlo, adaptive
Q-tuning, NIS/NEES diagnostics.

Estimator implications of the gravity campaign — keep two things separate:

- **Physics recommendation**: estimator dynamics need at least low-degree
  harmonics; likely floor nmax ≈ 32–64, regime-dependent (fixed in 13G-d).
- **Current software capability**: UKF could use 6-state harmonics *after*
  runner threading (13B2b); BLS-LM/SRIF depend on STM propagation and the
  harmonic gradient **is not implemented**; J2-only is operationally available
  but physically weak in all tested LLO regimes; a truth-harmonics/filter-J2
  mismatch strategy would need explicit validation and process-noise analysis.

**No filter/OD performance with harmonics has been validated.** Do not claim
otherwise.

The uncommitted parallel work (Section 20) is extending measurements with
one-way light-time solvers/Jacobians, stellar aberration application, and
config-level `measurement_model_profile` / `companion_geometry` /
`jacobian_model` enums — in progress, not part of any committed phase.

## 17. Tests and Reproduction Commands

Run from `python_port/`: `python -m pytest` (**rootdir is the parent `Grad/`,
`pytest.ini` lives there**). Numba, SPICE kernels, and the `.mat` ephemeris are
present on the user's machine (not in this sandbox). Last observed full-suite
baseline (2026-07-11): **399 passed / 20 skipped** — includes parallel-work
tests; the gravity-only baseline at 13G-cX was 395. Re-verify the live count
before recording it anywhere; skips are environment-dependent, non-harmonics.

Gravity-chain test files (test counts as of the last run):
`test_gravity_harmonics_prototype` (21), `test_gravity_harmonics` (15),
`test_gravity_harmonics_numba` (10), `test_gravity_model_loader` (14),
`test_lunar_frames` (21), `test_harmonics_dynamics_splice` (17),
`test_phase13c_harmonics_validation` (5), `test_real_grail_optional` (7;
skipUnless when data absent — proven with an empty-dir test; 660PM-only,
nmax=8), `test_scenario_config_harmonics` (26),
`test_phase13g_gravity_orbit_effects` (25). J2 side: `test_force_models` (7),
`test_earth_j2` (12), `test_scenario_comparison` (9).

Campaign commands (deterministic; 12B and 13G scripts use **cumulative JSON
stores** — stage runs don't overwrite each other; CSV/MD regenerate from the
store):

```text
python examples/phase12b_real_grail_validation.py --inventory
python examples/phase12b_real_grail_validation.py --loader
python examples/phase12b_real_grail_validation.py --sweep
python examples/phase12b_real_grail_validation.py --guards

python examples/phase13g_gravity_orbit_effects.py --baseline
python examples/phase13g_gravity_orbit_effects.py --compare
python examples/phase13g_gravity_orbit_effects.py --frames
python examples/phase13g_gravity_orbit_effects.py --sensitivity
```

Deliberately gated / not implemented: `--sevenday` (refuses until Phase 13G-c2
approval); plotting beyond summary outputs. **Do not run campaigns during
onboarding or without approval.** Dependencies (`requirements*.txt`):
numpy≥1.23.5, scipy≥1.14, spiceypy≥8.1.0, matplotlib≥3.8; optional
numba≥0.61 (`requirements-accelerated.txt`; NumPy fallback exists); PyQt5 for
the desktop app.

## 18. Data, Results, and Git Hygiene

- `data/gravity/` — real GRAIL files: local only, **gitignored, untracked,
  never committed**. Each model dir: coefficient `.tab`, `.lbl`, `SOURCE.txt`
  (PDS URL, original name, frame-declaration quote), `SHA256SUMS.txt`. New
  downloads are a separately approved step.
- `results/` — generated outputs: gitignored, reproducible from scripts.
- `exports/` — visible artifact copies (`phase6_j2_campaign/`): intentionally
  untracked.
- Kernels — outside the repo (`C:\Users\erayh\Documents\mice\kernels`), never
  committed. `ephemeris_data.mat` gitignored via `*.mat`.
- Dense/full trajectory files are never written by default — summary
  CSV/JSON/MD only.

Verification commands for any future checkpoint (all must show
tracked-file-free data/results):

```text
git ls-files data/gravity
git ls-files results
git check-ignore -v data/gravity/grgm660prim/gggrx_0660pm_sha.tab
git check-ignore -v results/phase13g/phase13g_sensitivity_report.md
```

(Verified 2026-07-11: `git ls-files` empty for both; ignores hit
`.gitignore:46 data/gravity/`, `:30 results/`, `:35 *.mat`.)

## 19. Known Limitations

- Lunar harmonics **acceleration** implemented; **gradient/STM not
  implemented** → 6-state propagation only; BLS-LM/SRIF cannot use harmonics.
- Scenario-runner threading of harmonics config not done (13B2b); the
  temporary not-yet-consumed guard is active.
- DE440 kernel profile validated in campaign scripts but **not** a production
  `spice_loader` profile (13B2c).
- GRGM1200L has no validated DE430 lunar-PA runtime chain → loader sanity only.
- GL1800F uses DE440 PA rotations while translational ephemeris remains
  DE421-derived `.mat` (declared; expected small for the tested third-body
  terms but not independently isolated).
- Filter/OD impact of truth-vs-estimator model mismatch not validated.
- Seven-day gravity confirmation (13G-c2) **not run**; 50 km perilune case
  **not approved/run**.
- Earth orientation for Earth-J2 is a fixed J2000 mean-pole approximation.
- Worst campaign single run ~36 s — performance phase (11D: njit uniform-grid
  lookup, zero-alloc Pines) deferred until 7-day/Monte-Carlo needs it.
- Real data, kernels, results are local; a fresh clone cannot run real-GRAIL
  campaigns without re-downloading (optional tests skip cleanly).

## 20. Uncommitted Work in Progress

The working tree carries a parallel **measurement-model/profile** effort
(M1–M5 roadmap: measurement architecture/traceability, light-time Jacobian,
two-way range, noise/bias models, external validation), independent of the
gravity phases and **not part of the gravity-phase commits**. Substantive
files as of 2026-07-11 (authoritative set = `git status` / `git diff`, not
this list):

```text
lunar_od/measurements.py        lunar_od/estimators.py
lunar_od/filters.py             lunar_od/scenario_config.py (fields on top of 13B2a)
lunar_od/scenarios.py           lunar_od/__init__.py
lunar_od/reporting.py           desktop_app/controllers/analysis_controller.py
examples/run_scenario_config.py tests/test_measurements.py
tests/test_reporting.py         tests/test_scenario_config.py
```

Observed content (read-only diff inspection): one-way light-time solver and
range sensitivities (`solve_one_way_light_time`,
`one_way_light_time_range_sensitivity`, initial-state Jacobians),
`apply_stellar_aberration`, range-rate companion observables, measurement
sigma/covariance helpers, and new config enums `measurement_model_profile`
(e.g. `one_way_light_time`, `one_way_light_time_aberrated_spice_ssb`,
`analytic_first_order_light_time`, `implicit_light_time`),
`companion_geometry`, `jacobian_model`.

Additionally, ~30 more files (all `SKILL.md`s, several docs, `.ui` files,
`visibility.py`, `thesis_matrix.py`, `scenario_model.py`) show
**whitespace/line-ending-only** modifications (verified: they vanish under
`git diff --ignore-all-space`). Do not "clean these up".

Rules: **never edit, stage, revert, or commit these files from a gravity or
documentation task**; never assume they belong to 13G; they get their own
checkpoints.

## 21. Recommended Next Steps (each separately approval-gated)

1. **Phase 13G-d — Gravity Model Synthesis and Truth/Estimator
   Recommendation**: combine 13G-b + 13G-c1; truth nmax per orbit regime;
   estimator-physics floor; thesis model hierarchy with inter-layer numbers;
   decide whether selected 7-day confirmation is needed; state limitations.
   **No new propagation runs automatically.** Preliminary (to be finalized):
   **candidate truth nmax=128** for 100 km/polar/low LLO (recommended from
   available evidence; matched 128→256 closure not run for those exact cases),
   **threshold-demonstrated nmax=64** at 200 km / i=60° / eccentric 80×500 km,
   **threshold-demonstrated nmax=32** at 500 km; truth candidates
   GRGM660PRIM@128 (consistent DE421 pipeline) and GL1800F@128
   (independent modern JPL, DE440 PA); GL1800F@256 selected confirmation only.
   Truth-model selection is **not** finalized in the repository yet.
2. **Phase 13G-c2 (optional) — selected seven-day confirmation**: candidates
   S1 (100 km, i=45° anchor), S7 (polar), S8 (eccentric), S3 (500 km control);
   `--sevenday` refuses until approval. 50 km perilune is a separate decision.
3. **Phase 13B2b — scenario-runner threading**: config → model load → rotation
   grid → `propagate_state`; UKF sigma-propagator decision;
   parallel-pickling check; remove the temporary guard only when the config is
   truly consumed.
4. **Phase 13B2c (optional) — production kernel profiles**: additive
   `KERNEL_PROFILES` in `spice_loader` (DE421 default unchanged; DE440
   explicit only; kernel-pool clearing; explicit frame names).
5. **Only afterward — filter/OD validation**: harmonics in UKF; truth/filter
   mismatch; residual sensitivity; process noise; BLS/SRIF gradient strategy
   (harmonic STM or alternative Jacobian). Do not start filters before the
   gravity synthesis is complete.
6. **Phase 11D — performance** (if triggered by 7-day/Monte-Carlo runtimes).

## 22. Instructions for Future Claude Sessions

1. Read `PROJECT_CONTEXT.md` before changing the project.
2. Check `git status` before every task.
3. Never overwrite or stage unrelated uncommitted work (Section 20).
4. Never use `git add .` — commits are explicit file lists at user-requested
   checkpoints only.
5. Never push unless the user explicitly asks (this branch has never been
   pushed).
6. Separate scientific campaigns, production changes, and measurement work
   into distinct commits.
7. Use explicit `MOON_PA_DE421` / `MOON_PA_DE440` frame names; bare `MOON_PA`
   is forbidden; `spice.kclear()` between kernel profiles.
8. Never silently mix lunar J2 and a C20-containing harmonics model.
9. Never silently fall back from harmonics to J2 (no silent disable, no silent
   correction — invalid combinations raise `ValueError`).
10. Never accept harmonics-enabled STM propagation while the gradient is
    unavailable.
11. Do not commit real gravity data, kernels, or generated results
    (Section 18 verification commands).
12. Small, reversible phases: implementation-before plan → user ACCEPT → test
    gate → scientific-validation gate → explicit checkpoint.
13. Preserve bit-identical default behavior when new physics is disabled (new
    parameters keyword-only, default off).
14. Distinguish **verification** (implementation correctness), **validation**
    (physical adequacy), and **estimation performance** (filter/OD behavior).
15. Do not call a model "validated" merely because it runs; state the
    truth-model level of every accuracy claim (campaign numbers here are
    Level 1, internal model-vs-model).
16. Report: what was calculated, why, equations/conventions, quantitative
    results, physical interpretation, limitations.
17. Use project skills per their trigger descriptions and the routing policy
    (smallest sufficient set; Section 6).
18. No long campaigns without explicit approval (`--sevenday` and any
    multi-day/Monte-Carlo runs are gated).
19. Do not add new tool/config files unless explicitly requested; do not run
    repository-wide formatters (much of the tree shows line-ending churn —
    leave it alone).
20. At the end of every phase report: changed files; tests; numerical
    findings; Git status; data/results hygiene; commit/push status; next
    decision gate.

Practical session notes: run tests from `python_port/` with `python -m pytest`
(rootdir is parent `Grad/`); Python↔Numba parity (better than ~1e-12) is a
mandatory gate for any dual-path physics; code/comments/commits in English
with ASCII-safe console output (Windows cp1254 trap); user communication in
Turkish; commit messages end with
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; default integrator
VODE Adams (rtol 1e-11, atol 1e-12); STM state is 42-state
[x(6); Φ(36)] column-major. Glossary: MCI = Moon-Centered Inertial (J2000
orientation); SHADR = PDS Spherical Harmonic ASCII Data Record; PA/ME =
Principal Axis / Mean Earth lunar frames; nmax ladder = 8/16/32/64/128(/256)
truncation ladder, N→2N difference = truncation-error proxy of full@N; RTN =
Radial/Transverse/Normal decomposition; TDB = Barycentric Dynamical Time.
