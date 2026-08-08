# Frame Architecture: Contracts and Inconsistencies

> **Audit type:** read-only frame/origin/epoch/state-transform architecture audit
> **Audited commit:** `eb92461f781c3fccd39012e3a02cd6ace64d893b` ("Add converged two-way range observable (M3)")
> **Worktree:** `python_port_measurement_fix`, branch `fix/measurement-model-safety`
> **Date:** 2026-07-11
> **Scope:** no production code, tests, or configuration were modified; no commit or push was made.
> **Consumers:** the F2 (Codex) executable-diagnostics phase and later remediation phases.
> **Revision (2026-07-13):** reconciled section numbering/titles against the
> completion spec; renumbered the pytest-provenance item from FA-07 to
> **FA-06** (and the M3-metadata-CSV item from FA-06 to FA-07) to match the
> spec's issue IDs; split the confirmed-correct table into 16 explicit
> per-contract rows; added an equation-to-code appendix; expanded the
> station-state comparison with per-path field lists; re-verified FA-06 with
> new empirical evidence (below) — the existing 507/28/0 baseline
> (Section 2.2) was **not** re-run for this revision, since no production,
> test, or configuration file changed.
> **P0A status update:** the P0A measurement-safety patch on this branch
> resolves **FA-01** (confirmed at eb92461/D1; resolved by hard rejection of
> UKF + CN/CN+S position profiles at both the scenario-config loader and the
> `run_lunar_ukf` runtime boundary via the shared
> `filters.validate_ukf_measurement_support` helper) and **FA-02** (confirmed
> at eb92461/D1; resolved by the one-way solver constants
> `ONE_WAY_LIGHT_TIME_TOLERANCE_S` / `ONE_WAY_LIGHT_TIME_MAX_ITERATIONS` and
> the measurement-type metadata branch). P0A also implements the short-term
> **legacy counted-Doppler nonzero-transponder-delay rejection**
> (`RangeRatePhysicsConfig.__post_init__`); the four-event counted-Doppler
> model remains future work (CD-4). The fixed scalar delay term cancels
> directly in the endpoint RTLT difference, but nonzero delay can still
> affect counted Doppler through the t2u/t2d separation, spacecraft motion
> during the delay, and the resulting uplink/downlink event geometry.
> **P0B-1 status update (2026-07-14):** the uncommitted P0B-1 patch (branch
> `fix/light-time-nonconvergence`, based at P0A HEAD `3265dbe9`)
> resolves **FA-03A** (confirmed at eb92461/D1 baseline; resolved by a dual
> convergence criterion — fixed-point update tolerance AND an independently
> re-evaluated light-time equation residual, both required — enforced at the
> nominal observable and Jacobian/sensitivity call sites for both the
> one-way (`_apparent_position_observable`,
> `one_way_light_time_range_sensitivity`) and legacy counted-Doppler
> (`two_way_counted_doppler_observable`,
> `round_trip_light_time_initial_state_jacobian`, validated separately per
> count-start/count-end endpoint) paths). Raw solvers
> (`solve_one_way_light_time`, `solve_two_way_light_time`) still return a
> diagnostic non-raising `converged=False` result — only the
> observable/sensitivity/Jacobian consumers now reject it, via the new
> `LightTimeConvergenceError` / `RoundTripLightTimeConvergenceError`
> exceptions, whose messages report both criteria's residuals in seconds and
> metres. Nominal converged behavior is unchanged; M3 (`two_way_range.py`)
> was not touched.
> **P0B-2 status update (2026-07-15):** the uncommitted P0B-2 patch on branch
> `fix/history-domain-enforcement` resolves **FA-03B**. Legacy production
> one-way CN/CN+S, counted-Doppler nominal/Jacobian, and counted UKF source
> histories now use one closed-support policy: exact endpoints are accepted,
> at most two policy ULP of representation-only overshoot is normalized to an
> endpoint sample, and anything farther outside raises `HistoryDomainError`
> before interpolation/extrapolation. Generation records deterministic
> candidate drops while preserving seeded RNG slots; scenario assembly carries
> aggregate counts and pre/post-roll deficiencies, continues surviving arcs,
> and raises before estimation when every eligible arc in the selected family
> is domain-empty. No history is automatically extended. Low-level generic
> interpolators and M3's separate four-event history contract are unchanged.

---

## 1. Executive summary

This audit reconstructed the frame, origin, epoch, and state-transform
architecture of the Lunar OD repository from the committed implementation at
`eb92461` and classified every finding as confirmed-correct, confirmed
defect, numerical approximation, suspected risk, validation gap, or
documentation mismatch.

Headline results:

- The repository-wide **passive column-vector transform convention is
  uniformly implemented and confirmed correct**: `pxform`/`sxform` direction,
  full-state inverses via `np.linalg.solve` (no `np.linalg.inv` exists in the
  package), station inertial velocity through the 6x6 `sxform` including the
  `Cdot @ r` coupling, Earth/Moon origin-translation signs, SEZ handedness
  and azimuth convention, observable/Jacobian parity for the geometric,
  one-way CN/CN+S, and M3 two-way range models, single STM application, and
  the lunar-gravity rotate-back direction (Section 12).
- **One CRITICAL confirmed defect**: the UKF position path silently evaluates
  the receive-epoch geometric observable for configurations whose
  measurement generation used a one-way CN or CN+S profile (FA-01).
- **Two HIGH confirmed defects** in solver safety policy, split per this
  audit's acceptance instructions: a nonconverged light-time solution could
  reach observable paths (FA-03A — **resolved by P0B-1**, see the header note
  above and Section 13) and event-state histories can be silently
  extrapolated without a domain guard (FA-03B — **resolved by P0B-2**). The M3 two-way
  range model was the reference strict policy both defects were measured
  against; the legacy production paths now reject unsupported history too,
  while retaining their own event and interpolation models.
- Five further confirmed items are metadata/documentation/API-hygiene level
  (FA-02, FA-04, FA-05, FA-06, FA-07).
- Measured interpolation evidence exists for 10/30/60 s transform grids and
  the Earth-ephemeris methods; the 240 s default sample grid, the UKF local
  re-sampling, the PCHIP position/velocity consistency, uplink visibility,
  and the MOON_PA lookup cadence at production settings remain **suspected
  risks / validation gaps requiring measurement (F2)** — they are explicitly
  *not* recorded as confirmed defects.

No frame-direction, origin-translation, or transform-inversion bug was found
in any production measurement path. The confirmed defects are epoch/physics
*parity* and solver *policy* issues, not transform-math issues.

---

## 2. Audit baseline and source provenance

### 2.1 Repository state

| Item | Value |
|---|---|
| Worktree | `C:/Users/erayh/Documents/Python/Grad/python_port_measurement_fix` |
| Branch | `fix/measurement-model-safety` |
| HEAD | `eb92461f781c3fccd39012e3a02cd6ace64d893b` |
| Working tree | clean before and after the baseline test run |
| Main worktree | untouched; its dirty state (2 modified phase13g files, 1 untracked doc) preserved |

### 2.2 Baseline test execution

```text
python -m pytest tests/            (isolated worktree)
507 passed, 28 skipped, 0 failed, 1 warning, 8 subtests passed, 56.8 s
no deselection
```

Skip accounting: 20 slow-regression gates (`LUNAR_OD_RUN_SLOW_TESTS=1`,
`tests/test_scenarios.py` x9 + `tests/test_filters.py` x11), 7 GRAIL-data
optional (`tests/test_real_grail_optional.py`), 1 artifact-dependent
(`tests/test_phase13g_synthesis_report.py`; git-ignored `results/` content
absent in a fresh worktree). All kernel-gated SPICE frame and two-way tests
executed and passed.

Import-provenance caveat: see FA-06. The baseline imported `lunar_od` from
the **main** worktree via the out-of-repo `pytest.ini` `pythonpath`; the two
`lunar_od/` trees are bit-identical at `eb92461` (verified through
`git status` on the main worktree), so the baseline result stands.

### 2.3 Sources

| Source | Status |
|---|---|
| Repository code and docs at `eb92461` | authoritative; all file:line citations refer to it |
| Existing test suite | executed (Section 2.2); coverage mapped in Appendix D |
| `measurement_models_physics_derivatives_and_gaps_AUDIT_REFERENCE.md` (repo-external copy at `C:/Users/erayh/Documents/Python/Grad/`) | **uncommitted documentation reference copied from the main worktree** — not part of the production repository, not an authoritative committed document; every frame-relevant claim adopted from it was independently re-verified against the committed code |
| `LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md` | tracked at `eb92461` (176 lines); committed doc source |
| NAIF / IERS contracts | Appendix E, with per-source contract mapping |

---

## 3. Frame glossary (as implemented)

| Code name | Physical origin | Axes / orientation | Rotating | Units | Primary production users | Kernel dependency | Ambiguity notes |
|---|---|---|---|---|---|---|---|
| `MCI` | Moon center | J2000-aligned | no | m, m/s | spacecraft states, dynamics, all measurement inputs | DE421 SPK (ephemeris sampling) | never means body-fixed; consistent |
| Earth-centered J2000 intermediate | Earth center | J2000 | no | m | transient `r_sat_mci - r_earth_mci` before Earth-fixed rotation (`measurements.py:1247`) | — | exists only inline; never stored as a named frame |
| `J2000` (SPICE) | per-call observer | J2000/ICRF | no | SPICE km -> m | `sxform`/`pxform`/`spkezr`/`spkpos` calls | `naif0012.tls`, planetary SPK/PCK | SSB origin appears only in the `spice_ssb` observer-velocity sample (`measurements.py:1239`) |
| `ITRF93` / production "ECEF" | Earth center | Earth-fixed (ITRF93) | yes | m | station coordinates, topocentric input | `earth_*.bpc` | naming split: identifiers say `ecef`, frame is ITRF93 everywhere in production (FA-05) |
| `SEZ` | station | South-East-Zenith | station-fixed | m | az/el observables (`geometry.py:42-96`) | — | none |
| `MOON_PA` (bare) | Moon center | lunar principal axes (kernel-resolved alias) | yes | m | **default parameter** of `sample_moon_pa_rotations` (`lunar_frames.py:92`); informational labels in `gravity_harmonics`/`gravity_model_loader`; legacy fixture `tests/test_spice_snapshots.py:34` | `moon_080317.tf` + PA `.bpc` | alias binds to the last-furnished kernel; config layer bans it, sampler default does not (FA-04) |
| `MOON_PA_DE421` / `MOON_PA_DE440` | Moon center | versioned principal axes | yes | m | `ScenarioConfig.lunar_gravity_frame` (validated enum + kernel-profile pairing, `scenario_config.py:49-54`) | versioned PA `.bpc` | correct and explicit |
| Moon mean-pole BF (`_MCI_TO_MOON_BF`) | Moon center | constant IAU 2006 mean pole | frozen | m | Moon-J2 term only (`dynamics.py:47-60`) | none | documented approximation (<0.1 % of J2 accel) |
| Earth BF for Earth-J2 (`_J2000_TO_EARTH_BF`) | Earth center | identity (J2000 mean pole) | frozen | m | Earth-J2 term (`dynamics.py:71`) | none | documented low-fidelity approximation; Earth J2 measured negligible for LLO |
| Legacy GST "ECEF" | Earth center | simple sidereal-z rotation | yes | m | `analyze_visibility_gap` comparison path only (`visibility.py:96-124`) | none | UTC treated as UT1; explicitly non-production |
| `SSB` | solar-system barycenter | J2000 | no | m/s | observer velocity for `spice_ssb` aberration only | SPK | velocity-only; never mixed into position origins |
| `IAU_MOON` | — | — | — | — | **not used anywhere** in the package (grep: zero production hits) | — | no confusion risk present |

Key clarifications demanded by the task:

- **MCI is not a Moon body-fixed frame** — confirmed: it is Moon-centered
  with J2000 axes everywhere.
- **Production "ECEF" is ITRF93** — confirmed by every production `sxform`
  call (`spice.sxform("J2000", "ITRF93", et)`); the name mismatch is FA-05.
- **Bare `MOON_PA`** — production *campaign* configuration enforces
  versioned names; bare usage survives in the sampler default, one legacy
  fixture, and Phase 13A tests/examples (FA-04, an API/default bypass risk,
  **not** a claim that the production gravity path uses a wrong frame).
- **DE421/DE440 pairing** — `_LUNAR_FRAME_TO_PROFILE` derives the kernel
  profile from the versioned frame name; mismatched combinations are rejected
  by config validation (`test_scenario_config_harmonics`).

---

## 4. Transform convention

Authoritative repository convention (confirmed uniform):

```text
r_B = C_{B<-A}(t) @ r_A          C_{B<-A} = spice.pxform(A, B, et)
x_B = X_{B<-A}(t) @ x_A          X_{B<-A} = spice.sxform(A, B, et)

vectors: (3,) and (6,) with column semantics, matrix @ vector everywhere
state layout: [x, y, z, vx, vy, vz]
```

Task Section 6 questions, answered with evidence:

1. **Column semantics** — yes; every transform application is
   `matrix @ vector` (e.g. `measurements.py:1248`, `visibility.py:229`,
   `filters.py:1486`, `radiometrics.py:377`).
2. **Direction uniformity** — yes; no counter-convention site found.
3. **Reverse position transform** — orthonormal-transpose is used only for
   3x3 rotations with an explicit orthonormality contract
   (`lunar_frames.py:43-48`, `force_models.py:55`,
   `gravity_harmonics.py:288`); production Earth-fixed reverses use `solve`.
4. **Reverse full-state transform** — `np.linalg.solve(X, state)` at every
   site (`measurements.py:391,451,1051`, `radiometrics.py:377,401-402`,
   `two_way_range.py` provider). Never a transpose.
5. **`X.T @ state` misuse** — none found (full-package grep of `.T @` and
   `transpose`; all hits are 3x3 gravity rotations or estimator quadratic
   forms).
6. **`pxform`/`sxform` swapped roles** — none found; `pxform`/rotation-block
   for directions, positions, and accelerations; `sxform` + solve wherever a
   station **velocity** is consumed.
7. **Docs correctness** — `docs/frame_transformations.md` §1–§2 states the
   position-rotation vs state-transform split correctly and matches the code.

`np.linalg.inv` does not appear anywhere in `lunar_od/` (grep: zero hits).

---

## 5. Origin translation graph

Forward (spacecraft -> topocentric), as implemented in every generator and
residual path:

```text
Spacecraft MCI (Moon-centered J2000)
    |  subtract Earth Moon-relative position/state        [translation, epoch t_r]
    |    measurements.py:1247 (position), :1479 (RR), filters.py:1485 (UKF),
    |    visibility.py:226
    v
Spacecraft Earth-centered J2000
    |  J2000 -> ITRF93                                    [rotation, epoch t_r]
    |    3x3 block for positions (measurements.py:1248, visibility.py:229),
    |    6x6 for full states (measurements.py:1480)
    v
Spacecraft ITRF93
    |  subtract station ITRF93 position                    [translation, station-fixed]
    |    measurements.py:1268, filters.py:1487
    v
Station-centered Earth-fixed LOS
    |  ITRF93 -> SEZ                                       [rotation, station constants]
    |    geometry.py:79
    v
Local SEZ  ->  range / azimuth / elevation
```

Reverse (station -> MCI), used by all light-time paths:

```text
Station fixed ITRF93  [r_F, 0]
    |  inverse J2000<->ITRF93 state transform              [rotation, np.linalg.solve]
    |    exact sxform at event epoch: measurements.py:391/1051 (t_r),
    |      two_way_range provider (t1, t3)
    |    linearly interpolated sxform: radiometrics.py:375 (counted-Doppler t1/t3)
    v
Station Earth-relative J2000 [r, v]
    |  add Earth Moon-relative state                       [translation, same epoch]
    |    measurements.py:392, radiometrics.py:378, two_way_range provider
    v
Station Moon-centered J2000 (MCI)
```

Findings on the graph edges: **no wrong sign, no double subtraction/addition,
no missing Earth translation, no rotation-before-required-translation, and no
Moon/Earth/SSB origin mixing** was found. Rotation and translation are
evaluated at the same event epoch on every edge (exact at grid nodes for
geometric/one-way; both interpolated consistently for legacy counted-Doppler;
exact rotation + Hermite translation for M3, an intentional, documented,
separately measured split). The single deliberate asymmetry is the
`local_mci` vs `spice_ssb` observer-velocity reference center — an
intentional, profile-labeled velocity-only approximation.

---

## 6. Measurement-model epoch matrix

Reconstructed from code; agrees with `docs/frame_transformations.md` §5 and
with the external reference §26 (independently re-verified):

| Model | Time tag | Spacecraft epoch | Station epoch | Transform epoch | Observer-vel epoch | STM epoch | Origin-translation epoch | Exact / interpolated |
|---|---|---|---|---|---|---|---|---|
| Geometric range | t_r | t_r | t_r | t_r | — | t_r | t_r | exact node `sxform` |
| Geometric az/el | t_r | t_r | t_r | t_r | — | t_r | t_r | exact node |
| One-way CN range | t_r | t_t = t_r − τ | t_r | n/a (norm) | — | t_t | t_r | exact node; Hermite SC/STM |
| One-way CN az/el | t_r | t_t | t_r | t_r | — | t_t | t_r | exact node |
| CN+S local_mci | t_r | t_t | t_r | t_r | t_r (Moon-relative) | t_t | t_r | exact node |
| CN+S spice_ssb | t_r | t_t | t_r | t_r | t_r (SSB, J2000 axes) | t_t | t_r | exact node + `spkezr` at t_r |
| Geometric range-rate | t_r | t_r | t_r | t_r | — | t_r | t_r | exact node (6x6) |
| Two-way counted Doppler | mid; endpoints mid ± Tc/2 | single t2 per endpoint | t1, t3 | interpolated at event epochs | — | t2 (Hermite) | same interpolated epochs | linear transform/ephemeris interpolation |
| M3 raw two-way range | t3 | t2u and t2d (separate) | t1, t3 | **exact `sxform` at t1, t3** | — | t2u and t2d (Hermite) | Hermite Earth at same epochs | exact + Hermite |
| M3 calibrated two-way range | t3 | t2u, t2d | t1, t3 | exact t1, t3 | — | t2u, t2d | same | same; identical Jacobian to raw |
| UKF geometric position | t_r | t_r | t_r | t_r node | — | — | t_r | exact node |
| UKF counted Doppler | mid | endpoint t2 per leg | t1, t3 | pass-grid values **linearly re-sampled** onto local 1–5 s grid (`filters.py:1743-1746`) | — | — | same | re-sampled |
| Visibility (production) | sample t | t | t | t node | — | — | t | exact node rotation block |

Epoch risks evaluated against the expected contracts:

- *receive transform used at transmit event / vice versa*: not found in any
  production path; proven measurable by the epoch-mutation test (17.9 arcsec
  at 1.35 s lunar light time, `test_frame_spice_validation`).
- *sample-grid transform used for event epoch*: present by design in the
  legacy counted-Doppler path (linear interpolation; measured budget,
  Section 11) — an approximation, not an epoch-direction bug.
- *observer velocity wrong epoch*: not found; receive-epoch everywhere.
- *STM wrong event epoch*: not found; state and STM always share the epoch
  (M3 evaluates t2u and t2d pairs separately).
- *Earth translation at a different epoch from rotation*: not found.
- **UKF position ignores the configured profile entirely** — the CRITICAL
  parity defect FA-01 (its frame math is internally correct; the defect is
  epoch/physics parity with generation).

---

## 7. Station-state architecture

Five coexisting station-state paths:

| # | Path | Symbol | Transform method | Earth translation | Velocity source | Epoch | Extrapolation policy | Consumers |
|---|---|---|---|---|---|---|---|---|
| 1 | Geometric / one-way CN(+S) | `_station_position_mci_at_receive_epoch`, `_station_relative_state_j2000_at_receive_epoch` (`measurements.py:442-462`) | exact node `sxform` (6x6 solve) | receive-node ephemeris value (generator getters; PCHIP-backed in campaigns) | 6x6 solve (observer velocity) | t_r | n/a (node-indexed) | position generators/residuals/Jacobians |
| 2 | Counted Doppler | `_station_state_mci`, `_station_state_mci_with_time_slope` | **linear `_interp_matrix`** over pass grid | **linear `_interp_vector`** | grid slope (`d(X^-1 x_F)/dt = -X^-1 Xdot X^-1 x_F`) | event t1, t3 | P0B-2 closed support; <=2 policy ULP uses endpoint sample, farther outside raises | legacy two-way solver + counted partials |
| 3 | M3 two-way range | `make_exact_sxform_station_state_provider` (`two_way_range.py`) | **exact event-epoch `sxform`** | **cubic Hermite** (pos+vel) | exact from 6x6 solve | t1, t3 | Earth pre-grid linear only, documented ~1 cm bound; spacecraft never | M3 solver/sensitivity/residuals |
| 4 | UKF two-way local | `_two_way_local_histories` | `_interp_pass_values` — linear **re-sampling of the coarse pass grid** | linear from pass grid | implicit in re-sampled 6x6 | local interval | P0B-2 source-interval preflight before propagation/resampling; interval anchors cover the clock-corrected count endpoints (P0B-2E2) | UKF counted-Doppler updates |
| 5 | Visibility (production) | `_station_arrays` + `visibility_mask_ecef` (`visibility.py:232-241`) | exact node rotation blocks | PCHIP getters at nodes | not needed | sample t | n/a | arc selection |

### 7.1 Per-path field comparison

The summary table above is expanded here per path across all fields
required by the task (station representation, origin, axes, epoch,
transform method, Earth translation method, velocity method, interpolation
method, extrapolation policy, consumers, metadata).

**Path 1 — Geometric / one-way CN(+S):**
station representation: WGS84-derived ITRF93 constants (`Station.r_ecef_m`);
origin: Earth center; axes: ITRF93, rotated to J2000 per call; epoch: t_r
(receive); transform method: exact node `sxform`; Earth translation method:
receive-node ephemeris value (PCHIP-backed in campaigns); velocity method:
full 6x6 solve (includes `Cdot @ r`); interpolation method: none (node-exact);
extrapolation policy: n/a; consumers: position/RR generators, residuals,
Jacobians; metadata: position-pass metadata carries light-time solver
parameters from the one-way policy constants (FA-02 was resolved by P0A).

**Path 2 — Legacy counted-Doppler:** station representation: same ITRF93
constants; origin: Earth center; axes: ITRF93 -> J2000; epoch: event t1, t3
per leg; transform method: linear `_interp_matrix` over the pass grid (not
exact at the event epoch); Earth translation method: linear `_interp_vector`;
velocity method: grid slope `d(X^-1 x_F)/dt = -X^-1 Xdot X^-1 x_F`;
interpolation method: linear (both transform and Earth translation);
extrapolation policy: P0B-2 closed support with endpoint normalization through
two policy ULP and `HistoryDomainError` beyond it (FA-03B resolved);
consumers: legacy two-way solver, counted partials, and (transitively) UKF
local histories; diagnostics name Earth position, Earth velocity, transform,
spacecraft state/STM, count endpoint, and leg. The interpolation error budget
itself is still not carried in metadata.

**Path 3 — M3 exact event provider:** station representation: same ITRF93
constants; origin: Earth center; axes: ITRF93 -> J2000, exact at t1/t3;
epoch: t1, t3 (separate uplink/downlink station events); transform method:
**exact event-epoch `sxform`** (Option A; station transform is never
interpolated); Earth translation method: cubic Hermite (position+velocity),
pre-grid linear only with a documented ~1 cm bound; velocity method: exact,
from the 6x6 solve; interpolation method: Hermite (Earth center only);
extrapolation policy: bounded pre-grid Earth approximation only; the
spacecraft state is never extrapolated (`TwoWayEventHistoryError` on
out-of-history epochs); consumers: M3 solver, sensitivity, residuals;
metadata: event/convention fields (`two_way_range_convention`,
`transponder_delay_s`) exist internally but are not yet carried into the
result CSV (FA-07).

**Path 4 — UKF counted-Doppler local path:** station representation: same
ITRF93 constants, re-sampled; origin: Earth center; axes: ITRF93 -> J2000,
linearly re-sampled from the pass grid; epoch: local 1-5 s grid endpoints
spanning the clock-corrected count interval plus the light-time margin
(P0B-2E2); transform method: `_interp_pass_values` — a
linear re-sampling of the already-interpolated pass-grid transform (i.e., an
interpolation of an interpolation); Earth translation method: linear,
inherited from the pass grid; velocity method: implicit in the re-sampled
6x6; interpolation method: linear re-sampling, fidelity capped by the
coarser source (pass) grid, not by the local grid density (SR-02);
extrapolation policy: Earth position, Earth velocity, and transform source
intervals are independently preflighted before local propagation or
re-sampling, over an envelope anchored at the clock-corrected count endpoints
computed with the observable's own clock function (P0B-2E2);
accepted <=2-policy-ULP offsets use endpoint source samples while
the physical local grid remains unchanged; larger offsets raise
`HistoryDomainError` (FA-03B resolved); consumers: UKF counted-Doppler
measurement updates; interpolation-fidelity metadata remains absent (VG-04).

**Path 5 — Visibility (production):** station representation: same ITRF93
constants; origin: Earth center; axes: ITRF93, exact node rotation blocks;
epoch: sample time t (simulation grid node); transform method: exact node
rotation block (3x3; no velocity needed); Earth translation method: PCHIP
getters at nodes; velocity method: not needed (geometric range/elevation
only, no Doppler); interpolation method: none (exact node); extrapolation
policy: n/a; consumers: arc/pass selection; metadata: none; this path is
light-time-unaware by design (SR-04/VG-06).

**Preserved distinction:** M3 uses exact event-epoch `sxform`; legacy
counted-Doppler uses an interpolated transform/pass-grid station state. This
split is intentional (documented in `docs/two_way_range.md` §5 and the
counted-Doppler doc note) but creates a real fidelity difference between the
two models, quantified in Section 11.

Answers to the task's Section 10 questions:

1. Yes — the same station is served by different physics per model path
   (the table above); differences are intentional and documented for
   M3-vs-counted, under-documented for the UKF local path (VG-04).
2. The M3-exact vs counted-interpolated split is **intentional** and
   documented (`docs/two_way_range.md` §5, counted-Doppler doc note); the
   external reference records it as "intentional split, not a bug".
3. The UKF local 1–5 s grid is **not** freshly sampled: transforms and Earth
   states are linearly re-sampled from the coarse pass grid, so local density
   does not add transform fidelity (suspected risk SR-02 / gap VG-04).
4. Earth-center translation methods in production: PCHIP (ephemeris layer),
   linear (counted-Doppler), cubic Hermite (M3) — three methods coexist;
   measured comparison in Section 11.
5. Rotation and translation share the event epoch on every path — confirmed.
6. Station-history extrapolation: P0B-2 guards paths 2 and 4 before their
   unchanged in-support interpolators. Paths 1/3/5 retain their separate
   node/exact-event contracts; no legacy production path silently extrapolates.
7. Metadata: M2/M3 metadata reports methods correctly; the generic position
   solver metadata was corrected by P0A; UKF local interpolation fidelity still
   has no quantitative metadata (VG-04), while domain errors carry structured
   support and pre/post-roll diagnostics.

---

## 8. Observable/Jacobian/STM frame parity

| Model | Observable path | Jacobian path | Same LOS/origin/epoch/station/SEZ/observer-vel | STM ownership |
|---|---|---|---|---|
| Geometric position / RR | `compute_*_residuals` | `compute_*_residuals_analytic` + `apply_stm_to_jacobian` | yes (identical formulas) | local rows, STM applied once (`accelerated.py:436-466`) |
| One-way CN / CN+S | `_apparent_position_observable` | `one_way_light_time_position_initial_state_jacobian` | yes — shared light-time solve contract, receive transform, SEZ basis, observer velocity (M2.1–M2.3, FD-verified) | initial-state block; consumers must not re-apply STM (mutation-tested) |
| Counted Doppler | `two_way_counted_doppler_observable` | `two_way_counted_doppler_initial_state_jacobian` | yes — partial deliberately differentiates the observable's own interpolated physics | STM embedded via interpolated `Phi_r(t2)`; applied once |
| M3 two-way range | `solve_two_way_range_events` + `two_way_range_from_solution` | `two_way_range_event_sensitivity` on the **same converged solution** | yes — one linearization point per observation | initial-state row; no-double-STM mutation test |
| UKF position | `_position_measurement_from_state` | (sigma-point; no analytic Jacobian) | **no** — geometric regardless of profile (FA-01) | n/a |

Estimator/observability consumers: BLS-LM, SRIF, posterior information, and
observability consume identical numeric blocks for position, RR, and M3
(shared-block and equality tests in `test_estimators`, `test_observability`,
`test_two_way_range_integration`). Rotation matrices are applied to Jacobians
from the left (frame chain), consistent with the column-vector convention.

Ownership contract, as implemented:

```text
local/epoch-state Jacobian  -> frame chain -> STM applied exactly once
initial-state Jacobian      -> caller must not apply the STM again
```

Both directions are guarded by mutation tests (position, M3).

---

## 9. Visibility frame contract

Actual production contract (`analyze_visibility_gap_with_transforms`,
`visibility.py:184-245`):

```text
spacecraft frame:     MCI -> Earth-centered J2000 -> ITRF93 (node-exact rotation blocks)
station frame:        ITRF93 constants (WGS84-derived)
horizon frame:        SEZ elevation vs min_elevation_deg
Moon occultation:     evaluated in ITRF93 (r_moon_ecef, spacecraft ECEF)
sample epoch:         simulation grid node t (receive-epoch geometric)
light-time awareness: none (instantaneous geometric selection)
```

- Visibility is geometric receive-epoch: **yes** (by design).
- CN/CN+S and M3 measurement selection depends on this geometric visibility:
  **yes** — a light-time-aware pass would differ by O(ω_E·τ) in station
  geometry, negligible for gating but nowhere stated (VG-06).
- Uplink visibility (station elevation at t1) is **not** separately checked
  for two-way observables: suspected-risk/documentation item SR-04/VG-06,
  not a confirmed defect (t1 precedes t3 by ≤ ~2.6 s of Earth rotation).
- Moon-intersection frame: ITRF93, consistent with the LOS frame used.
- Visibility and measurement paths share station arrays but duplicate the
  frame chain in separate code (technical debt, no divergence found).
- The GST path (`analyze_visibility_gap`, UTC≈UT1, sidereal-z) is a legacy
  comparison path, documented as non-production.

---

## 10. Lunar body-fixed frame contract (truth-trajectory frame risk)

This section is deliberately separated from the measurement audit: errors
here corrupt the *truth trajectory*, not the measurement frame chain.

```text
gravity coefficients frame:   lunar principal axes (GRGM/GL sets), label "MOON_PA"
MCI -> lunar fixed:           C = pxform("J2000", <frame>, et), sampled on a grid
                              (lunar_frames.sample_moon_pa_rotations), nearest-
                              neighbour lookup in the RHS (exactly orthonormal)
lunar fixed -> MCI accel:     a_i = C.T @ a_bf  (force_models.py:55,
                              gravity_harmonics.py:288)  — confirmed correct
variational equations:        G_i = C.T @ G_bf @ C (force_models.py:74) — correct
Moon J2 only:                 constant mean-pole _MCI_TO_MOON_BF (documented
                              <0.1 % J2 approximation)
Earth J2:                     identity Earth BF (documented low-fidelity
                              approximation; effect measured negligible)
```

Hard safety gates confirmed in `dynamics._prepare_harmonic_context`
(`dynamics.py:429-501`): J2 double-count ban; constant-matrix ban for m > 0
coefficients; rotation-grid coverage requirement; no silent kernel fallback
(`lunar_frames.py:32-35`, explicit `FileNotFoundError`).

Findings:

1. Harmonics frame handling is architecturally sound; **no rotate-back
   direction bug**; Jacobian and acceleration share the convention.
2. DE421/DE440 compatibility is enforced at the **configuration layer**
   (`ALLOWED_LUNAR_GRAVITY_FRAMES`, frame-to-profile pairing, bare name
   rejected: `test_scenario_config_harmonics`).
3. **FA-04 (API/default bypass risk):** `sample_moon_pa_rotations` defaults
   to bare `frame="MOON_PA"`, and its Phase 13A test/example callers plus the
   legacy fixture `tests/test_spice_snapshots.py:34` use the bare alias.
   Under the project's DE421-only `REQUIRED_KERNELS` set this resolves
   deterministically today; it bypasses the configuration-layer rule if a
   caller reaches the sampler directly or a second PA kernel is furnished.
   This is a default/hygiene risk — **not** evidence that the production
   gravity path uses a wrong frame.
4. Nearest-neighbour lookup quantizes lunar orientation (≈8.0e-5 rad per
   30 s of lunar rotation; bound asserted in `test_lunar_frames:118`). The
   *production-cadence* error budget is example-level (Phase 13C preflight),
   not config-enforced: suspected risk SR-05 / gap VG-07. No measured
   production-magnitude claim is made here.

---

## 11. Transform and state interpolation architecture

| Path | Source grid | Method | Consumers | Extrapolation | Measured error (repository evidence) | Metadata |
|---|---|---|---|---|---|---|
| Transform grid, linear | pass grid (nodes exact) | `_interp_matrix` on 6x6 | counted-Doppler station states | P0B-2 closed-support guard; no production extrapolation | 10/30/60 s midpoint: station 0.32/2.88/11.5 m; two-way range effect 0.036/0.34/1.40 m (`frame_transformations.md` §11/§14); event-offset form: 0.086/0.32/0.67 m range at t1 ≈ node−2.7 s (M3 integration diagnostics) | domain policy/drop diagnostics; no interpolation-error budget |
| Earth ephemeris, linear | pass grid | `_interp_vector` | counted-Doppler | P0B-2 closed-support guard; no production extrapolation | 1.1 m @ 60 s midpoint | domain policy/drop diagnostics |
| Earth ephemeris, Hermite | pass grid | `interp_state_history` (pos+vel) | M3 provider | pre-grid linear, documented ~1 cm bound | 1.2e-7 m / 5.0e-9 m/s @ 60 s midpoint | `earth_center_ephemeris_interpolation` |
| Ephemeris PCHIP | ephemeris grid | independent `PchipInterpolator` per pos/vel (`ephemeris.py:32-34`) | dynamics RHS, generators | PCHIP end behavior | **not separately measured**; velocity interpolant is not the derivative of the position interpolant (VG-03) | none |
| Spacecraft state/STM Hermite | propagation grid | `_interp_state`, `_interp_state_transition_position` | all light-time paths | P0B-2 guards CN/counted production boundaries; M3 retains its own guard | FD-validated ≲1e-6 relative (M2/M3 suites) | `spacecraft_state_interpolation`, domain policy/drop diagnostics |
| MOON_PA nearest-neighbour | rotation grid (60 s default suggestion) | exact sampled matrix, time-quantized | harmonics RHS | hard error, no extrapolation | ≤1e-3 rad bound asserted at 180 s grid midpoint | config cadence field |
| UKF local re-sampling | pass grid -> 1–5 s local grid | linear `_interp_pass_values` | UKF counted-Doppler | P0B-2 preflights all three source histories; no production extrapolation | **unmeasured** (VG-04) | controlled domain diagnostics; no fidelity metric |

Grid-policy assessment:

- **10 s:** measured 0.086 m two-way range effect (1.7 % of a 5 m sigma) —
  acceptable under a 0.1 sigma budget.
- **30 s:** 0.32 m (6.4 %) — near the budget boundary.
- **60 s:** 0.67–1.40 m depending on evaluation point (13.5 %+) — exceeds
  the 0.1 sigma budget; this measurement is why M3 adopted exact `sxform`.
- **120 s / 240 s:** **no measurement exists.** Scaling the measured
  near-node error term (∝ 0.5·|f''|·d·(h−d)) gives an *extrapolation
  estimate only* of roughly 3–10 m station-position class at 240 s
  (`THESIS_SAMPLE_STEP_S = 240`); this must be measured in F2 before any
  conclusion is drawn (SR-01/VG-02). It is explicitly **not** a measured
  result and **not** a confirmed defect.
- A dense local grid produced by re-sampling a coarse source grid is **not**
  exact — its fidelity is capped by the source grid (applies to path
  "UKF local re-sampling").

Time-scale audit (task §16): `et0` is SPICE ET (TDB s past J2000) from
`str2et(UTC)` or `(first_jd_TDB − 2451545)·86400`; propagation uses relative
seconds and every SPICE call adds `et0` exactly once (no double-add found);
count-interval endpoints are formed in relative seconds before clock
correction; the UTC≈UT1 approximation exists only in the legacy GST path;
large-ET finite-difference cancellation is a known measured effect (frame
audit FD sweep) and is a design constraint for F2 diagnostics (SR-06).

---

## 12. Confirmed-correct frame contracts

The following were checked and **no defect was found**. This statement is
limited to what static inspection of `eb92461` plus the existing test suite
(Appendix D) can establish; it is not a proof of correctness beyond that
coverage, and it is not a claim of universal correctness for paths or
cadences outside what was inspected.

| # | Contract | Evidence |
|---|---|---|
| 1 | Passive column-vector transform convention (`matrix @ vector` uniformly, never `vector @ matrix`) | Section 4; full-package grep of `.T @`/`transpose`/`@` sites |
| 2 | `pxform` direction (`C_{B<-A} = pxform(A, B, et)`, `r_B = C @ r_A`) | Section 4, item 6; `lunar_frames.py:43-48`, visibility rotation blocks |
| 3 | `sxform` direction (`X_{B<-A} = sxform(A, B, et)`, applied the same way to 6-vectors) | Section 4, item 6; every station-velocity site (Section 7) |
| 4 | Full-state reverse transform via `np.linalg.solve(X, state)` — never a transpose, never `np.linalg.inv` (zero package hits) | Section 4, items 4–5; `measurements.py:391,451,1051`, `radiometrics.py:377,401-402`, `two_way_range.py` provider |
| 5 | Station inertial velocity correctly obtained through the full 6x6 `sxform` (not a 3x3 rotation of a body-fixed zero vector) | `measurements.py:1050-1052`, `radiometrics.py:389-406`, M3 provider |
| 6 | `Cdot @ r` coupling term present and populated (zero body-fixed station velocity still maps to nonzero inertial velocity) | same sites; `test_frame_spice_validation` station-velocity FD sweep + range-rate contribution test |
| 7 | Moon/Earth origin-translation signs and ordering (translate before rotate on the forward path; add back on the reverse path; no double add/subtract) | Section 5 |
| 8 | SEZ handedness (S x E = Z, right-handed) | `geometry.py:42-96`; 17 SPICE-free frame tests incl. cardinal-direction and mutation guards |
| 9 | North-clockwise azimuth convention (`atan2(E, -S)` wrapped to [0, 2*pi), 0 = north, increasing toward east) | `geometry.py:42-96`; cardinal-direction tests assert north/east/south/west azimuth values |
| 10 | Geometric observable/Jacobian frame parity | Section 8 row 1; `test_estimators`/`test_observability` shared-block equality tests |
| 11 | One-way CN(+S) observable/Jacobian frame parity | Section 8 row 2; M2.1-M2.3 finite-difference suites |
| 12 | M3 (two-way range) observable/Jacobian frame parity | Section 8 row 4; `test_two_way_range` FD suite, same-linearization-point check |
| 13 | STM applied exactly once (initial-state Jacobian blocks are never re-mapped by a caller) | mutation tests in `test_estimators`, `test_two_way_range` |
| 14 | M3 exact event-epoch station-state provider (Option A), with its Earth-ephemeris approximation separately measured and reported | `two_way_range.py`; M3 integration diagnostics (Section 11) |
| 15 | Lunar gravity acceleration rotate-back direction (`a_MCI = C.T @ a_bf`) | Section 10; `force_models.py:55`, `gravity_harmonics.py:288`; harmonics validation suites |
| 16 | Lunar gravity-gradient similarity transform (`G_MCI = C.T @ G_bf @ C`) | Section 10; `force_models.py:74`; harmonics validation suites |

Also confirmed, supporting the above: harmonics safety gates (J2
double-count ban, m>0 constant-matrix ban, rotation-grid coverage
requirement, no silent kernel fallback) at `dynamics.py:429-501` and
`lunar_frames.py:32-35`.

---

## 13. Confirmed defects

### FA-01 — UKF position path ignores the configured measurement profile

| Field | Record |
|---|---|
| Category | LOCAL-FRAME / FRAME-EPOCH (physics parity) |
| Severity | **CRITICAL** |
| Difficulty | EASY (hard rejection) / HARD (profile-aware operator) |
| Status | confirmed |
| Accepted configuration path | `scenario_config_from_mapping` -> `_validate_cross_field_rules`: no rejection of `estimator_type="ukf"` + `measurement_type="position"` + non-geometric `measurement_model_profile` |
| Generation observable path | `generate_position_measurements` -> `_apparent_position_observable` (`measurements.py:1252-1266`): converged light time, transmit-epoch spacecraft state, optional CN+S aberration |
| UKF sigma-point observable path | `run_lunar_ukf` -> `_measurement_context` -> `_position_measurement_from_state` (`filters.py:1480-1489`): receive-epoch instantaneous geometric range/az/el only; `pass_geo.measurement_model_profile`, `apply_light_time`, `apply_stellar_aberration` never consulted |
| Actual behavior | measurements generated with CN or CN+S physics are filtered against a geometric predicted observable |
| Expected behavior | sigma-point prediction uses the same measurement physics as generation, or the configuration is rejected |
| Minimum reproduction | build a UKF `position` scenario with `measurement_model_profile="one_way_light_time_aberrated_spice_ssb"`; evaluate `compute_position_residuals` and `_position_measurement_from_state` on one truth-state row at the same epoch; the difference is nonzero although the state is the truth (range bias ~ c·τ sensitivity ≈ km-scale; angle bias up to ~20 arcsec from aberration + light-time) |
| Affected profiles | `one_way_light_time`, `one_way_light_time_aberrated_local_mci`, `one_way_light_time_aberrated_spice_ssb` (position type) |
| Affected estimators | UKF only |
| Validity matrix | geometric UKF **valid**; CN/CN+S BLS-LM and SRIF **valid** (M2 implicit chain); CN/CN+S UKF configuration **inconsistent (this defect)**; M3 + UKF **safely rejected** at config, runner, and filter layers |
| Scientific impact | systematic innovations (light-time + aberration signatures), biased state and covariance, misleading NIS/NEES |
| Estimator impact | invalidates UKF-vs-batch comparisons for non-geometric position scenarios |
| User-visible impact | silently accepted configuration produces scientifically wrong filter output |
| Current mitigation | none at runtime; only the batch estimators implement the profiles |
| Recommended fix | **hard rejection** of non-geometric position profiles for UKF at config validation and in `run_lunar_ukf` (smallest backward-compatibility surface) |
| Alternative | profile-aware sigma-point operator: bounded local history + `_apparent_position_observable` per sigma point (HARD — history domain, caching, runtime) |
| Required tests | config rejection tests; truth-state closure per profile per estimator; geometric-UKF bitwise regression; if implemented, CN/CN+S UKF seeded consistency (NIS/NEES) |
| Backward-compatibility risk | hard rejection breaks previously accepted (but inconsistent) configs — intended |
| Recommended phase | F1 |

### FA-03A — Light-time nonconvergence result can reach observable paths

| Field | Record |
|---|---|
| Category | NUMERICAL-SOLVER / VALIDATION |
| Severity | HIGH |
| Difficulty | MEDIUM |
| Status | confirmed at `eb92461` D1/P0A baseline; **resolved by P0B-1** |
| Baseline behavior | Raw one-way and round-trip solvers could return `converged=False`; one-way nominal and both counted nominal/Jacobian paths consumed the final iterate. One-way sensitivity alone rejected nonconvergence, creating a nominal/Jacobian policy split. |
| Implemented validity rule | A result is valid only when both the fixed-point iteration-update tolerance and an independently re-evaluated final light-time equation-residual tolerance pass. The final equation residual is not inferred from loop bookkeeping. |
| Raw solver contract | `solve_one_way_light_time` and `solve_two_way_light_time` may still return diagnostic nonconverged solution objects. They expose update convergence/residual data and final equation residuals without publishing an observable. |
| One-way consumer contract | `_apparent_position_observable`, one-way sensitivities, and the local/initial-state Jacobian chain share one strict policy. Invalid results raise `LightTimeConvergenceError`; nominal and Jacobian paths no longer diverge. |
| Counted consumer contract | `two_way_counted_doppler_observable` validates `count-start` and `count-end` independently. `two_way_counted_doppler_initial_state_jacobian` and `round_trip_light_time_initial_state_jacobian` enforce the same endpoint policy and raise `RoundTripLightTimeConvergenceError`. |
| Diagnostics | One-way errors report update and equation residuals. Counted errors distinguish uplink/downlink update and equation failures. Equation residuals are reported in seconds and equivalent metres. |
| Affected symbols | `LightTimeSolution`, `RoundTripLightTimeSolution`, `solve_one_way_light_time`, `_apparent_position_observable`, one-way sensitivity/Jacobian helpers, `solve_two_way_light_time`, `two_way_counted_doppler_observable`, `two_way_counted_doppler_initial_state_jacobian`, `round_trip_light_time_initial_state_jacobian` |
| Numerical compatibility | Normal converged numerical behavior is intended to remain unchanged; the full P0B-1 regression passed. Only invalid/pathological last-iterate consumption is rejected. |
| Required tests implemented | forced `max_iter=1` update failures; update-passes/equation-fails cases; independent equation-residual recomputation; counted start/end and uplink/downlink diagnostics; positive finite tolerance validation; normal cross-layer regression |
| Not affected | P0B-1 did not address FA-03B; P0B-2 resolves it separately. The legacy counted model remains single-bounce, the four-event counted model remains future work, and M3 (`two_way_range.py`) behavior is unchanged. |
| Backward-compatibility risk | low and intentional: callers that consumed invalid last iterates now receive a controlled exception; converged cases retain their numerical path |
| Completed phase | P0B-1 / F1 |

### FA-03B — Event-state history was silently extrapolated (resolved)

| Field | Record |
|---|---|
| Category | HISTORY-DOMAIN / INTERPOLATION |
| Severity | HIGH |
| Difficulty | MEDIUM |
| Status | confirmed at `eb92461`/D1; **resolved by P0B-2** |
| Baseline behavior | `_interp_state`, `_interp_vector`, `_interp_matrix`, `_interp_array_and_slope`, and `_interp_pass_values` could evaluate arbitrarily outside source support. One-way CN/CN+S, counted nominal/Jacobian, and counted UKF production consumers reached those helpers without a guard. |
| Implemented boundary rule | Every guarded source has closed support `D=[t_start,t_end]`. Exact endpoints pass. With `S=max(1,abs(q),abs(bound))`, at most `2*(nextafter(S,+inf)-S)` outside a bound is normalized to the endpoint sample. Three policy ULP and larger excursions raise `HistoryDomainError` before evaluation. Solver event variables and physical UKF local times are not clipped. |
| One-way coverage | Every production transmit-state solver probe, final state re-query, and state/STM Jacobian lookup uses the shared guard. Nominal, sensitivity, and Jacobian paths have one domain policy. |
| Counted coverage | `count-start`/`count-end`; uplink/downlink station events; spacecraft state and STM; Earth position/velocity; and 6x6 transform lookups are guarded independently. The first unsupported intermediate probe stops the solve. |
| Counted UKF coverage | `_two_way_local_histories` preflights Earth position, Earth velocity, and transform source intervals before local propagation or `_interp_pass_values`. Representation-only accepted offsets alter source lookup only, not the physical local grid. The preflighted local envelope is anchored at the clock-corrected count endpoints — min/max of the raw versus corrected count-start/count-end, computed with the observable's own `_clock_corrected_receive_time` — so nonzero station clock offset/drift of either sign cannot push a corrected receive epoch outside the locally built grid (P0B-2E2). |
| Generation and RNG | Unsupported visible position/range-rate candidates are omitted with ordered `HistoryDomainDropRecord` entries. Candidate noise slots are consumed before physics evaluation (3 position or 4 range-rate draws), preserving later seeded rows; noise-disabled generation draws nothing. |
| Scenario/reporting | Partial drops in the selected family retain surviving arcs. An all-domain-empty selected family raises an aggregate `HistoryDomainError` before estimator entry. `ScenarioResult` carries drop counts, family counts, max required pre/post-roll, the all-empty flag, and detailed records; summary CSV appends the six scalar fields. |
| Ownership | Propagation/history construction remains caller-owned. P0B-2 reports actual deficiencies and never creates automatic pre-roll/post-roll. The policy is family-local because one build/result owns one measurement family. |
| Low-level compatibility | Generic interpolation helpers retain their prior behavior for non-production/diagnostic compatibility; strict guards wrap every affected legacy production boundary. |
| Not affected | M3 retains `TwoWayEventHistoryError`, its separate four-event contract, exact event transforms, and existing generation policy. The legacy counted model remains single-bounce; nonzero delay is rejected and a four-event counted model remains future work. |
| Numerical compatibility | Supported-interior P0B-1/P0B-2 comparisons were bit-identical in the validated fixtures. Only unsupported history use changes behavior. |
| Required tests implemented | exact/1/2-policy-ULP acceptance; symmetric 3-policy-ULP rejection and mutation guard; first unsupported one-way/counted probe rejection; five counted history names; endpoint/leg diagnostics; UKF preflight ordering; clock-corrected UKF local envelope (zero/positive/negative offsets, corrected-endpoint containment with outward corrected endpoints as exact anchor nodes, source-boundary rejection with clock-sized post-roll); candidate-drop/RNG determinism; estimator/observability propagation; family-local partial/all-empty scenario behavior; CSV transport; M3 regression |
| Completed phase | P0B-2A/B/C/D1/E2 / F1 |

### Other confirmed items (register in Section 18)

- **FA-02** — position-pass metadata reports light-time tolerance/max-iter
  from `RangeRatePhysicsConfig` defaults (1e-10 s / 20) although
  `solve_one_way_light_time` runs with 1e-12 s / 10 (`measurement_model_metadata`
  builds from `range_rate_physics_config(pass_geo.range_rate_physics)`;
  MEDIUM, metadata-only).
- **FA-04** — bare `MOON_PA` **API/default bypass risk** (Section 10, item 3):
  sampler default + Phase 13A tests/examples + legacy fixture bypass the
  config-layer versioned-frame rule (LOW-MEDIUM).
- **FA-05** — production identifiers named `ecef` denote ITRF93; this is a
  naming/documentation mismatch only and does **not** imply that all IERS
  station-correction models (plate motion, tides, polar motion) are
  implemented — see Section 14's WGS84-as-rigid-ITRF93 approximation row
  (LOW, naming/documentation).
- **FA-06** — isolated-worktree test import provenance (Section 7/register
  below); re-verified with new empirical evidence 2026-07-13.
- **FA-07** — M3 event/convention metadata (`two_way_range_convention`,
  `transponder_delay_s`, event model) is not carried into the result CSV
  (`reporting.py` has M2 columns only; MEDIUM, traceability).

---

## 14. Numerical approximations (intentional; error budget required or held)

| Approximation | Where | Budget status |
|---|---|---|
| Linear 6x6 transform interpolation for counted-Doppler events | `radiometrics._interp_matrix` | measured at 10/30/60 s (Section 11); 120/240 s unmeasured (VG-02) |
| Linear Earth-ephemeris interpolation (counted path) | `radiometrics._interp_vector` | measured: 1.1 m @ 60 s midpoint |
| Hermite Earth-ephemeris interpolation (M3) | `interp_state_history` | measured: 1.2e-7 m @ 60 s; pre-grid ~1 cm bound documented |
| Constant mean-pole rotation for Moon J2 | `dynamics._MCI_TO_MOON_BF` | documented <0.1 % of J2 acceleration |
| Identity Earth BF for Earth J2 | `dynamics._J2000_TO_EARTH_BF` | documented; Earth-J2 effect measured negligible for LLO |
| WGS84 station coordinates as rigid ITRF93 (no plate motion/tides) | `config.Station.r_ecef_m` | documented (frame docs §9); acceptable at current noise |
| Geometric (non-light-time) visibility selection | `visibility.py` | assumption implicit — must be documented (VG-06) |
| `local_mci` observer velocity omits SSB motion | aberration profile | intentional, profile-labeled |
| Nearest-neighbour MOON_PA lookup, 60 s cadence suggestion | `lunar_frames` | bound asserted in tests; production cadence budget not config-enforced (VG-07) |
| Single-bounce spacecraft state for nonzero-delay counted Doppler | `solve_two_way_light_time` | documented (M3 doc updates); zero-delay unaffected |

A fixed scalar transponder-delay term cancels directly in the endpoint
round-trip-light-time difference. That cancellation does not make the legacy
single-bounce and four-event models equivalent for nonzero delay: separating
`t2u` and `t2d` changes the spacecraft state and the uplink/downlink geometry.
The short-term safety policy should therefore reject legacy counted Doppler
when `transponder_delay_s != 0`; a four-event counted implementation belongs to
a later architecture phase.

---

## 15. Suspected numerical risks

(Measurement required — these are NOT confirmed production defects.)

| ID | Risk | Basis | Required measurement |
|---|---|---|---|
| SR-01 | 240 s (`THESIS_SAMPLE_STEP_S`) counted-Doppler transform-grid error | extrapolation estimate only (~3–10 m station-position class from the measured 10/30/60 s series; **no measured value exists**) | F2: exact-vs-interpolated sweep incl. 120/240 s |
| SR-02 | UKF 1–5 s local-grid re-sampling fidelity | local grid is a linear re-sample of the coarse pass grid (`filters.py:1743-1746`) | F2: local-vs-exact station/transform comparison over a count interval |
| SR-03 | PCHIP position/velocity interpolant inconsistency | independent interpolants (`ephemeris.py:32-34`); velocity is not the derivative of the position interpolant | F2: consistency diagnostic vs direct `spkezr` |
| SR-04 | Uplink-visibility impact for two-way observables | station geometry at t1 vs t3 differs by ≤ ~2.6 s of Earth rotation; unquantified | F2: elevation-at-t1 vs elevation-at-t3 sweep near masks |
| SR-05 | MOON_PA nearest-neighbour error magnitude at production cadence | test bound (≤1e-3 rad on a 180 s grid) is not a production-cadence budget | F2/F6: cadence sweep vs trajectory effect |
| SR-06 | Large-ET finite-difference cancellation in future diagnostics | measured once in the frame-audit FD sweep (round-off below ~1 s steps) | F2 design constraint: difference in relative seconds, plateau-based tolerances |

---

## 16. Validation gaps

| ID | Gap | Consequence |
|---|---|---|
| VG-01 | No test compares the UKF measurement operator against `compute_position_residuals` under CN/CN+S profiles | FA-01 survived every suite |
| VG-02 | No measured interpolation budget for 120/240 s transform grids | SR-01 undecidable |
| VG-03 | No PCHIP position/velocity consistency diagnostic | SR-03 undecidable |
| VG-04 | No fidelity measurement or metadata for the UKF local re-sampling | SR-02 undecidable |
| VG-05 | No import-provenance assertion in the test environment | FA-06 undetectable by the suite |
| VG-06 | Geometric/no-uplink visibility assumption undocumented and untested | SR-04 invisible |
| VG-07 | MOON_PA cadence budget not enforced/recorded at config level | SR-05 unbounded in config space |
| VG-08 | One-way/counted nonconvergence was not surfaced by consumers | **Closed by P0B-1:** controlled exceptions carry update/equation diagnostics; raw diagnostic solution objects remain inspectable |

---

## 17. Documentation and metadata mismatches

| Claim | Location | Actual behavior | Type | Severity | Correction |
|---|---|---|---|---|---|
| "linear interpolation over the propagated state history" | `docs/two_way_counted_doppler.md` derivation section | spacecraft state/STM interpolation is cubic Hermite (`_interp_state`); transforms/Earth are linear | docs drift | MEDIUM | **corrected by P0B-2D2** with per-quantity methods and closed-support policy |
| Position-pass light-time tolerance/max-iter metadata | `measurement_model_metadata` | baseline mismatch resolved by P0A; position reports one-way constants (1e-12/10), range-rate reports its own config | metadata (FA-02) | MEDIUM | **corrected by P0A** |
| M3 metadata completeness in results | `reporting.py` CSV columns | M2 fields only; no two-way convention/delay/event columns | traceability (FA-07) | MEDIUM | add M3 columns/manifest |
| "ECEF" naming | production identifiers/docstrings | frame is ITRF93 | naming (FA-05) | LOW | docstring/comment clarification |
| Nonzero-delay counted Doppler | legacy counted profile configuration | P0A rejects nonzero delay because separate `t2u`/`t2d` event geometry is not represented by the single-bounce model | physics/API safety | HIGH | short-term rejection complete; implement four-event counted Doppler separately |
| "Frame audit: CLOSED — no correctness bug found" | `LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md` §Frame audit | true for the M2-era frame-direction scope; later parity/policy findings FA-01, FA-03A, and FA-03B are now resolved by P0A/P0B-1/P0B-2 | scope drift | LOW | retain the narrower frame-direction claim and the dated resolution record |
| CN pre-grid extrapolation "just before the grid start" | `_apparent_position_observable` docstring | **resolved by P0B-2:** strict closed support raises or drops instead of extrapolating; caller-owned pre-roll is required | docs vs behavior | MEDIUM | corrected in code/docstring and P0B-2D2 model documentation |

---

## 18. Prioritized issue register

Ordered by severity (CRITICAL > HIGH > MEDIUM > LOW-MEDIUM > LOW), then by
remediation phase.

| ID | Title | Category | Severity | Difficulty | Status | Phase |
|---|---|---|---|---|---|---|
| FA-01 | UKF position path ignores configured measurement profile | LOCAL-FRAME / FRAME-EPOCH | CRITICAL | EASY (reject) / HARD (implement) | confirmed at eb92461/D1; **resolved by P0A hard rejection** | F1 (done) |
| FA-03A | Nonconverged light-time solution can reach observable paths | NUMERICAL-SOLVER / VALIDATION | HIGH | MEDIUM | confirmed at eb92461/D1; **resolved by P0B-1 strict dual-criterion enforcement** | F1 (done) |
| FA-03B | Event-state history silently extrapolated without bound | HISTORY-DOMAIN / INTERPOLATION | HIGH | MEDIUM | confirmed at eb92461/D1; **resolved by P0B-2 closed-support enforcement** | F1 (done) |
| FA-02 | Position metadata reports wrong light-time solver parameters | DOCUMENTATION / metadata | MEDIUM | EASY | confirmed at eb92461/D1; **resolved by P0A metadata correction** | F0 (done) |
| P0A-CD0 | Legacy counted-Doppler nonzero transponder delay | NUMERICAL-SOLVER (safety gate) | MEDIUM | EASY | **rejection retained for the single-bounce model (P0A, narrowed by model selection)**; four-event model **implemented in R4** as `counted_doppler_model='four_event_delay'` (CD-4 closed) | CD-0 (done), CD-4 (R4) |
| FA-06 | Isolated worktree tests can import code from the main worktree | TEST-PROVENANCE / TECHNICAL-DEBT | MEDIUM | EASY-MEDIUM | confirmed | F0 |
| FA-07 | M3 event/convention metadata absent from result CSV | DOCUMENTATION / traceability | MEDIUM | EASY-MEDIUM | confirmed | F0 |
| FA-04 | Bare `MOON_PA` sampler default / fixtures bypass versioned-frame rule | LUNAR-FIXED-FRAME (API/default bypass) | LOW-MEDIUM | EASY-MEDIUM (fixture decision) | confirmed | F6 |
| FA-05 | `ecef` identifiers denote ITRF93 (naming only; does not imply full IERS station-correction coverage) | DOCUMENTATION | LOW | EASY | confirmed | F0 |
| SR-01…SR-06 | Section 15 | INTERPOLATION / VISIBILITY / LUNAR-FIXED-FRAME / TIME-SCALE | — | — | suspected (measurement required) | F2 |
| VG-01…VG-08 | Section 16 | VALIDATION-GAP | — | — | VG-01…VG-07 open; VG-08 closed by P0B-1 | F2 / P0B-1 |

### FA-06 — Isolated worktree tests can import code from the main worktree

| Field | Record |
|---|---|
| Category | TEST-PROVENANCE / TECHNICAL-DEBT |
| Severity | MEDIUM |
| Difficulty | EASY-MEDIUM |
| Status | confirmed; re-verified with new empirical evidence on 2026-07-13 |
| Actual behavior | `pytest.ini` lives at `C:\Users\erayh\Documents\Python\Grad\` (outside both repositories) with `testpaths = python_port/tests` and `pythonpath = python_port, python_port/desktop_app`; this file is picked up as `rootdir`/`configfile` from *either* worktree, so `import lunar_od` inside a pytest session can bind to the **main** worktree package even when pytest is invoked from `python_port_measurement_fix` |
| Root cause (2026-07-13 finding) | `_pytest/config/__init__.py: Config._configure_python_path` runs `for path in reversed(getini("pythonpath")): sys.path.insert(0, str(path))` during pytest startup, unconditionally, for every pytest invocation that has this `pytest.ini` as its config file. It is not a togglable plugin (`-p no:pythonpath` has no effect — verified, see below); it is core `Config` behavior triggered directly by the presence of the `pythonpath` ini key. |
| Import-provenance commands (as specified) | `python -c "import lunar_od; print(lunar_od.__file__)"` and `python -c "import sys; print('\n'.join(sys.path))"` |
| Expected path | `C:\Users\erayh\Documents\Python\Grad\python_port_measurement_fix\lunar_od\__init__.py` |
| Plain-`python -c` result (outside pytest) | **correct** — resolves to the isolated worktree, because `sys.path[0] == ''` (CWD) for `-c`/interactive invocations and pytest's `pythonpath` insertion never runs |
| Actual pytest-session result (verified empirically) | **wrong** — `lunar_od.__file__` resolves to `C:\Users\erayh\Documents\Python\Grad\python_port\lunar_od\__init__.py` (the **main** worktree); `sys.path[:3]` inside the pytest session is `['...\\python_port', '...\\python_port\\desktop_app', '']` — the ini-configured main-worktree path sits ahead of the CWD entry |
| Tested candidate fix: bare `PYTHONPATH` env var pointing at the isolated worktree | **does not fix it** — empirically verified; the env var's entries land *after* the ini-inserted `python_port` paths in `sys.path` (interpreter-startup insertion happens before pytest's own `Config._configure_python_path` runs, and the latter still inserts at index 0), so `import lunar_od` still resolves to the main worktree. This corrects the audit's original recommendation, which assumed a plain `PYTHONPATH` would suffice. |
| Tested candidate fix: `-p no:pythonpath` | **does not fix it** — empirically verified; there is no plugin named `pythonpath` to disable (see root cause above), so this flag is a silent no-op |
| Working fix (verified 2026-07-13) | `python -m pytest ... --override-ini=pythonpath=` — a per-invocation command-line override of the ini key to an empty value, with **no file edit and no environment variable**. Verified: with this flag, `sys.path[0] == ''` (CWD) and `lunar_od.__file__` correctly resolves inside `python_port_measurement_fix`. This is the "equivalent repository-local test invocation" referred to in the audit's forward-looking guidance. |
| Audit-result validity | during this audit the main and isolated `lunar_od/` trees are bit-identical at `eb92461` (main worktree dirt is limited to Phase13G example/test files and an untracked doc), so the Section 2.2 baseline is **not invalidated** by this defect |
| Forward risk | any Codex/remediation change made to `lunar_od/` in the isolated worktree will **not** be what a plain `pytest` invocation imports — producing false-positive "regressions" (or false passes, if the main worktree happens to already contain a compatible fix) against unmodified main-worktree code. Every F2 diagnostic and every future test run in this worktree must use the verified `--override-ini=pythonpath=` invocation (or an equivalent) until `pytest.ini` itself is fixed, which requires explicit approval (out of scope here) |
| Recommended fix | either (a) always invoke pytest in this worktree with `--override-ini=pythonpath=`, standardized into the F2 runner, or (b) a repo-local pytest configuration for this worktree (requires explicit approval; not applied by this audit); the external `pytest.ini` is deliberately **not** modified |
| Required tests | a provenance assertion at the start of every F2 diagnostic harness invocation (fail fast if `lunar_od.__file__` is outside the executing worktree) |
| Phase | F0 (first item) |

---

## 19. Remediation roadmap

Order fixed by the acceptance instructions; **F2 is the Codex phase and must
produce executable evidence before any production fix.**

| Phase | Goal | Files / symbols | Why | Risk | Tests required | Numerical acceptance | Backward compat | Depends on |
|---|---|---|---|---|---|---|---|---|
| **F0 — Test provenance and documentation/metadata** | FA-06 verification + provenance assertion; FA-02 metadata fields; FA-05 naming notes; FA-07 M3 CSV columns; counted-doc wording | `pytest` env / F2 harness; `measurements.measurement_model_metadata`; `reporting.py`; docs | trustworthy evidence before anything else | minimal | provenance check; metadata unit tests | n/a | none | — |
| **F1 — UKF hard rejection and solver/domain safety** | FA-01 hard rejection (**done P0A**); FA-03A dual convergence criterion + failure policy (**done P0B-1**); FA-03B closed-support/drop/reporting policy (**done P0B-2**) | `scenario_config._validate_cross_field_rules`, `filters.run_lunar_ukf`; shared `history_domain`; one-way/counted/UKF consumers; scenario/reporting transport | the three physics-safety defects | invalid configurations, nonconverged solves, and unsupported histories now fail intentionally; domain drops may alter early arc composition | completed FA-01/03A contracts plus P0B-2 boundary, RNG, integration, empty-family, and reporting tests | supported-interior outputs unchanged in validated comparisons; exact/<=2 policy ULP accepted, 3+ rejected | explicit safety breaks only | F0, completed evidence |
| **F2 — Executable frame diagnostics (Codex)** | Section 20 | new `examples/`/`tests/` diagnostics only | convert SR/VG items into measurements | none (read-only w.r.t. production) | self-validating diagnostics | per Section 20 | none | F0 |
| **F3 — Counted-Doppler exact event station provider** | reuse M3 provider in `_station_state_mci*` | `radiometrics.py` | removes measured interpolation error class | changes counted outputs at interp-error level | zero-delay bitwise fixture first, then budgeted diffs | ≤0.1σ observable shift justified by F2 numbers | counted baselines shift | F2 |
| **F4 — Shared station/frame provider (HARD — Opus-level architecture decision)** | one provider, method labels, all five paths | `measurements`/`radiometrics`/`two_way_range`/`filters`/`visibility` | eliminate 5-way duplication | wide blast radius | behavior-freeze fixtures per path | bit-identical where policy unchanged | high if rushed | F3 |
| **F5 — Visibility parity** | document geometric/no-uplink assumption; optional LT-aware diagnostic | `visibility.py`, docs | close VG-06 | minimal | SR-04 sweep | mask changes bounded | none | F2 |
| **F6 — Lunar fixed-frame/default consolidation** | FA-04: versioned default threading, fixture regeneration decision, config-enforced cadence budget | `lunar_frames.py`, `tests/test_spice_snapshots.py`, `scenario_config` | close bypass + VG-07 | fixture regeneration | frame-name propagation tests; cadence sweep | SR-05 budget met | fixture churn | F2 |
| **F7 — External cross-validation** | SPK-target SPICE light-time; GMAT/Orekit range & Doppler comparison | new examples | independent evidence beyond internal FD | HARD | frozen-contract comparisons | documented per-observable agreement | none | F1–F6 |

---

## 20. Codex diagnostic requirements (F2 inputs)

Each diagnostic must be executable, print measured values, assert only
self-consistency (not yet fix acceptance), and start with the FA-06
provenance assertion.

1. **Provenance guard** — assert `lunar_od.__file__` is inside the executing
   worktree; print `sys.path` head.
2. **Transform-grid sweep (SR-01/VG-02)** — exact `sxform` vs linear
   interpolation at 10/30/60/**120/240** s for counted-Doppler event epochs;
   report station pos/vel, event-time, observable, and Jacobian deltas, and
   the fraction of each observable's sigma.
3. **UKF parity reproduction (FA-01/VG-01)** — truth-state closure per
   profile: `compute_position_residuals` vs `_position_measurement_from_state`;
   print range/az/el deltas for geometric (expected 0) and CN/CN+S (expected
   nonzero).
4. **Nonconvergence surfacing (FA-03A/VG-08, completed by P0B-1)** -
   `max_iter=1` and update-passes/equation-fails tests now assert controlled
   rejection in one-way nominal/sensitivity/Jacobian and counted start/end
   endpoint paths; raw diagnostic solutions retain measured residuals.
5. **History-boundary sweep (FA-03B, completed by P0B-2)** — shared tests now
   enforce exact/1/2-policy-ULP endpoint sampling and symmetric 3+-policy-ULP
   rejection; model tests cover first unsupported probes, all five counted
   histories, UKF preflight ordering, deterministic generation drops/RNG, and
   family-local scenario/reporting behavior. External evidence retains the
   pre-rolled comparison without permitting production extrapolation.
6. **PCHIP consistency (SR-03/VG-03)** — position-derivative vs velocity
   interpolant vs direct `spkezr` on representative grids.
7. **UKF local-grid fidelity (SR-02/VG-04)** — local re-sampled station
   states vs exact `sxform` over a count interval.
8. **Uplink-visibility sweep (SR-04/VG-06)** — elevation at t1 vs t3 near
   mask boundaries.
9. **MOON_PA cadence sweep (SR-05/VG-07)** — lookup-angle and trajectory
   effect vs cadence at production settings.
10. **FD design rule (SR-06)** — all diagnostics difference in relative
    seconds and select steps on measured plateaus (frame-audit precedent).

---

## Appendix A — Symbol glossary

| Symbol | Meaning |
|---|---|
| `C_{B<-A}(t)` | 3x3 passive rotation, frame A to frame B |
| `X_{B<-A}(t)` | 6x6 state transformation `[[C,0],[Cdot,C]]` |
| `t_r`, `t_t` | one-way receive / transmit epochs, `t_t = t_r − τ` |
| `t1, t2, t3` | legacy two-way: station transmit, single bounce, station receive |
| `t1, t2u, t2d, t3` | M3 events: uplink transmit, spacecraft receive, spacecraft transmit (`t2d = t2u + δ_tr`), station receive |
| `Φ_r(t, t0)` | position rows of the STM w.r.t. the arc initial state |
| `et0` | SPICE ET (TDB s past J2000) of scenario t = 0 |
| `τ`, `δ_tr` | light time; fixed transponder delay |

## Appendix B — Frame graph

```text
                 (orientation)                    (orientation)
   MOON_PA_DE421/440  <--- pxform ---  J2000  --- sxform --->  ITRF93
        |                                |                        |
   gravity a_bf,G_bf              MCI (Moon origin)          station r_F
   C.T rotate-back                Earth origin via            |
                                  +/- r_E/M translation       v
                                        |                    SEZ (station origin)
                                        v                     az/el
                             Earth-centered J2000
```

Origins: Moon (MCI, spacecraft, gravity), Earth (J2000 intermediate, ITRF93),
station (SEZ), SSB (observer velocity only). Orientation transforms never
change origins; translations are explicit vector additions (Section 5).

## Appendix C — Equation-to-code map

| Equation | Meaning | Code |
|---|---|---|
| `r_B = C_{B<-A}(t) @ r_A`, `C_{B<-A} = pxform(A, B, et)` | passive 3x3 position/direction rotation | `lunar_frames.py:43-48`; visibility rotation blocks (`visibility.py:229`) |
| `x_B = X_{B<-A}(t) @ x_A`, `X_{B<-A} = sxform(A, B, et) = [[C, 0], [Cdot, C]]` | passive 6x6 state (position+velocity) transform | every station-velocity site (Section 7); `filters.py:1486`; `radiometrics.py:377`; `two_way_range.py` provider |
| `x_A = solve(X_{B<-A}, x_B)` | reverse full-state transform (never `X.T`) | `measurements.py:391,451,1051`; `radiometrics.py:377,401-402`; `two_way_range.py` provider |
| `v_station_inertial = C @ v_bf + Cdot @ r_bf` (`Cdot @ r` term from the `sxform` block) | station inertial velocity including body-fixed-rotation coupling | `measurements.py:1050-1052`; `radiometrics.py:389-406`; M3 provider |
| `r_sc_topocentric = r_sc_ITRF93 - r_station_ITRF93` | Earth-fixed line-of-sight vector | `measurements.py:1268`; `filters.py:1487` |
| `[S, E, Z] = R_SEZ<-ITRF93(lat, lon) @ r_topocentric`, `S x E = Z` | SEZ rotation and handedness | `geometry.py:42-96` |
| `az = atan2(E, -S) mod 2*pi`, `el = asin(Z / |r|)` | north-clockwise azimuth, elevation | `geometry.py:42-96` |
| `a_MCI = C.T @ a_bf` | lunar gravity acceleration rotate-back (body-fixed to inertial) | `force_models.py:55`; `gravity_harmonics.py:288` |
| `G_MCI = C.T @ G_bf @ C` | lunar gravity-gradient similarity transform | `force_models.py:74` |
| `G_u = t2u - t1 - \|r_sc(t2u) - r_st(t1)\| / c = 0` | M3 uplink light-time event equation | `two_way_range.py: solve_two_way_range_events` |
| `G_d = t3 - t2d - \|r_sc(t2d) - r_st(t3)\| / c = 0` | M3 downlink light-time event equation | `two_way_range.py: solve_two_way_range_events` |
| `G_tr = t2d - t2u - delta_tr = 0` | M3 fixed transponder-delay constraint | `two_way_range.py: solve_two_way_range_events` |
| `H_R = -(c/2) dt1/dx0`, from `G_y dy/dx0 = -G_x` (implicit-function theorem on the 3x3 event system) | M3 initial-state range Jacobian | `two_way_range.py: two_way_range_event_sensitivity` |
| `Phi_local -> STM applied exactly once` (initial-state Jacobian never re-mapped by a consumer) | STM single-application ownership contract | `accelerated.py:436-466`; mutation tests in `test_estimators`, `test_two_way_range` |
| `d(tau)/dx0` from the implicit one-way light-time equation `t_t = t_r - tau(x0)` | one-way light-time sensitivity | `measurements.py: one_way_light_time_range_sensitivity` |

## Appendix C.1 — Code-symbol map (frame-relevant)

| Symbol | File | Role |
|---|---|---|
| `ecef2sez_dcm`, `ecef2razel_sez`, `geodetic_to_ecef_wgs84`, `wrap_to_pi` | `geometry.py` | station geodesy, SEZ, angles |
| `_station_relative_state_j2000_at_receive_epoch`, `_station_position_mci_at_receive_epoch`, `_observer_velocity_j2000_at_receive_epoch` | `measurements.py` | one-way station/observer states |
| `_apparent_position_observable`, `one_way_light_time_*` | `measurements.py` | CN/CN+S observables and Jacobians |
| `_station_state_mci`, `_station_state_mci_with_time_slope`, `_interp_*` | `radiometrics.py` | counted-Doppler station states, interpolators |
| `make_exact_sxform_station_state_provider`, `solve_two_way_range_events`, `two_way_range_event_sensitivity` | `two_way_range.py` | M3 event chain |
| `_position_measurement_from_state`, `_two_way_local_histories`, `_interp_pass_values` | `filters.py` | UKF operators (FA-01, SR-02) |
| `normalize_supported_epoch`, `HistoryDomainError`, `HistoryDomainDropRecord`, `summarize_history_domain_drops` | `history_domain.py` | shared FA-03B closed-support, diagnostics, and drop aggregation contract |
| `analyze_visibility_gap_with_transforms`, `sample_j2000_to_itrf93_transforms`, `calc_gst_curtis` | `visibility.py` | visibility frames |
| `sample_moon_pa_rotations`, `nearest_rotation_at_time`, `validate_rotation_matrix` | `lunar_frames.py` | lunar PA provider |
| `_build_moon_j2_rotation`, `_MCI_TO_MOON_BF`, `_J2000_TO_EARTH_BF`, `_prepare_harmonic_context` | `dynamics.py` | body-fixed gravity frames |
| `body_j2_acceleration` rotate-back, gradient similarity | `force_models.py`, `gravity_harmonics.py` | gravity frame math |
| `lunar_initial_state_mci`, `coe2rv`, `rot_x/rot_z` | `orbit.py` | initial state (MOON_PA -> J2000 via 6x6) |
| `MoonCenteredEphemeris`, `sample_moon_centered_ephemeris` | `ephemeris.py` | MCI ephemeris (PCHIP) |
| `apply_stm_to_jacobian` | `accelerated.py` | single STM application |

## Appendix D — Existing test matrix (frame-relevant)

| Suite | Type | Frames/epochs covered | Mutations caught | Blind spots |
|---|---|---|---|---|
| `test_frame_transformations` (17) | SPICE-free | SEZ orthonormality/handedness/cardinals/wrap/zenith, frame-Jacobian FD | transpose, reversed, double-SEZ, deg/rad | — |
| `test_frame_spice_validation` (12, kernel-gated) | SPICE | sxform properties, station-velocity FD sweep, receive/transmit epoch mutation, round trips, interpolation diagnostics | FROM/TO reversal, epoch swap | 120/240 s grids |
| `test_two_way_range` (27) + `test_two_way_range_integration` (16) | mixed | M3 events, FD, exact-vs-interp, reversed-transform, frozen-epoch, pxform-style station-velocity mutation, scipy root | factor-2, delay-twice, double-STM | — |
| `test_measurements`, `test_stellar_aberration_vv` | mixed | M2 chain, aberration operator vs `spice.stelab` | STM mutation, wrap | UKF parity |
| `test_estimators`, `test_observability` | SPICE-free | shared blocks, no-double-STM | second STM | UKF profile parity (VG-01) |
| `test_history_domain`, measurement safety/owner tests, `test_filters`, `test_scenarios`, `test_reporting` | SPICE-free | closed support, two-policy-ULP normalization, one-way/counted/UKF guards, deterministic drops/RNG, family-local empty arcs, CSV transport | 3-ULP allowance mutation, unsupported-probe leakage, estimator-entry spy | cross-family coordinator (not present by design) |
| `test_lunar_frames` | kernel-gated | MOON_PA sampling/lookup/no-fallback | bad grids, missing kernels | production cadence budget |
| `test_visibility` | SPICE-free/comparison | GST vs transform path | — | uplink epoch, LT-awareness |
| `test_ephemeris`, `test_spice_snapshots` | mixed | ephemeris sampling; MATLAB parity (bare `MOON_PA` fixture) | — | PCHIP pos/vel consistency |

## Appendix E — Source bibliography

| Source | Supported contract | Relation | Repository symbols |
|---|---|---|---|
| NAIF Frames Required Reading — https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html | frame naming/aliasing semantics (incl. `MOON_PA` alias binding to the furnished kernel), inertial vs fixed frame classes | direct | `lunar_frames.py`, `scenario_config.ALLOWED_LUNAR_GRAVITY_FRAMES` (FA-04 rationale) |
| NAIF `pxform_c` — https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pxform_c.html | position/direction rotation only; `r_B = C r_A` direction of `pxform(A,B,et)` | direct | `sample_moon_pa_rotations`, visibility rotation blocks |
| NAIF `sxform_c` — https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sxform_c.html | 6x6 `[[C,0],[Cdot,C]]` state transform; required for rotating-frame velocities | direct | every station-velocity path; `orbit.lunar_initial_state_mci`; Section 12 `Cdot @ r` confirmation |
| NAIF SPK Required Reading — https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html | observer/target/center semantics, geometric (`'NONE'`) states, km units | direct | `sample_moon_centered_ephemeris`, `spice_ssb` observer velocity |
| NAIF PCK Required Reading — https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html | binary PCK lunar principal-axis frames, DE421/DE440 PA kernels | direct | `MOON_PA_DE421/DE440` profile pairing, `moon_pa_de421_1900-2050.bpc` |
| IERS Conventions 2010 — https://iers-conventions.obspm.fr/conventions_material.php | ITRF realization, EOP, station displacement models (documented omissions) | conceptual | ITRF93 usage, WGS84-as-rigid-ITRF93 approximation (Section 14), FA-05 naming |

---

*End of audit document. Produced read-only at `eb92461`; no production code,
tests, or configuration were modified. Before D1 baseline finalization, the
audit and its 2026-07-13 FA-06 revision had not been committed or pushed.
P0B-2D2 later updated documentation only to reconcile the validated P0A,
P0B-1, and P0B-2 safety status; historical baseline evidence remains labeled.*
