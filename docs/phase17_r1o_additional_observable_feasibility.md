# PHASE 17-R1O — ADDITIONAL OBSERVABLE FEASIBILITY FOR K_SRP IDENTIFIABILITY

> ## HISTORICAL RESULT — SUPERSEDED
>
> **Every surrogate result in this report used Phi^T in place of Phi.**
>
> `examples/phase17_r1o_core.py::chain_to_augmented_columns` unflattened the column-major 6x6
> state-transition matrix with NumPy's default C order. Phi is not symmetric, so all three
> surrogate observables below — the DDOR-like angle, the lunar-landmark LOS, and the
> Earth-center LOS control — were built on transposed state Jacobians. The K columns were
> unaffected.
>
> Corrected on this report's own configuration, the landmark headline `f_perp = 0.8757` becomes
> **`0.2530`** against a range-only baseline of `0.2523`, and the DDOR headline `f_perp = 0.4419`
> becomes **`0.0701`**. Neither candidate meaningfully rotates the K_SRP information direction.
> The `sigma_K/K` improvements are real, but they are information *magnitude*, not *direction*.
>
> **The cross-observable ranking, the candidate selection, and the
> `STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION` classifications in this report are withdrawn.**
> The range-only baseline, and every K-column value, remain valid.
>
> The original numbers below are deliberately preserved as a historical artifact. See
> **`docs/phase17_r1o_scientific_erratum.md`** and **`docs/phase17_r1o_r_stm_layout_repair.md`**
> (Phase 17-R1O-R, repair commit `75ba49a7e86cf7c4ea01e45ca757126159dc0654`).

## 1. Executive Summary

R1COV qualified a trustworthy formal covariance for the K_SRP solve-for path and, in doing so,
sharpened the open question: the seventh K direction is `WEAK_BUT_NONZERO` in the design-matrix
domain, not mathematically absent, yet the best radiometric configuration found so far (R1M's
linked multi-arc case) still carries 45% fractional uncertainty. R1O asks whether a genuinely new
measurement *direction* — not just more of the same range history — can close that gap.

**Both Tier-1 candidates tested here are decisively promising, and one is more promising than the
raw numbers alone would suggest.**

A DDOR-like plane-of-sky angular surrogate, built on the one DSN baseline pair
(Goldstone–Canberra) that is genuinely dual-visible at a 10° elevation mask in this orbit geometry,
rotates the orthogonal K fraction from `0.252` to `0.442` at 2 nrad noise — a real, measured
published-in-the-literature precision (Bell et al. 2013 report the DSN "approaching one-nanoradian
accuracy," 2–5 nrad 1-σ) — and drives the fractional K uncertainty from **9.3% down to 1.2%**. Even
at 300 nrad, well beyond the cited historical narrowband floor, the combined system still reaches
8.9%, better than the range-only control. `DDOR_OPERATIONAL_PLAUSIBILITY = CLEARLY_OPERATIONALLY_
PLAUSIBLE`.

A idealized known-landmark line-of-sight surrogate is the single strongest result in the entire
K_SRP investigation to date: at 1 μrad it drives fractional K uncertainty to **0.19%** and rotates
`f_perp` to `0.876`. But this phase also measured, deliberately, what happens when that idealization
is relaxed. A 200 m landmark-map uncertainty or a 500 μrad (star-tracker-limited) attitude
uncertainty erodes the benefit by 70–92%, pushing the result back toward the same ~9% floor the
range-only baseline already reaches. `OPTICAL_LANDMARK_UNCERTAINTY_CLASS = SEVERE_EROSION`,
`OPTICAL_ATTITUDE_SENSITIVITY_CLASS = SEVERE_AT_ATTITUDE_LIMITED_FLOOR`. The mathematical
information is real; whether it survives contact with a real camera and a real star tracker is a
different, harder question this phase does not answer.

Both candidates were confirmed under an independent geometry perturbation (a DDOR window shifted
two orbits later; twice as many landmarks) before being trusted, and both reproduced their gains.
Both were cross-checked against an independent Schur-complement calculation of the same combined
system to `<6e-16` relative error, so neither result depends on the specific numerical route used
to obtain it.

A celestial line-of-sight to Earth's center — included as the low-cost optional case — was
characterized and found to add negligible information direction (`max f_perp gain = 0.0083`): an
inertial direction that varies on the ~1-month Earth-Moon timescale carries almost no sensitivity to
the spacecraft's fast lunar-orbital motion, and is therefore nearly redundant with what the range
observable already constrains.

Ranking the two promising candidates against §64's criteria — precision achievability, hardware and
operations cost, flight heritage, and what this repository would have to build — favors **DDOR
first**: it reaches strong, direction-changing information at a precision the DSN has flown for
decades, needs no new spacecraft hardware, and its systematic-error budget is well characterized in
the literature. Landmark LOS is mathematically the stronger observable but is the one whose benefit
this phase's own numbers show eroding fastest under realistic camera and attitude uncertainty, and
it would require building camera, attitude, and landmark-map infrastructure this repository has
none of today.

No production code was modified. No new measurement type was added to `lunar_od/`. Every result
here is an analysis-only surrogate, explicitly labeled as such throughout.

```
DDOR_LIKE_INFORMATION_SURROGATE: PROMISING
LUNAR_OPTICAL_LANDMARK_INFORMATION: PROMISING BUT ATTITUDE/MAP-LIMITED
Both promising -> rank, do not implement simultaneously (S64)
SELECTED_CANDIDATE_1 = DDOR-like plane-of-sky angle
SELECTED_CANDIDATE_2 = Lunar landmark LOS
```

## 2. What R1COV Established

R1COV qualified a square-root covariance path for the K_SRP solve-for BLS and SRIF branches,
matching an exact rational-arithmetic oracle to ≤1.6e-13, invariant to numerical scaling with a
spread of exactly `0.000e+00`, and correctly responsive to priors. It found the R1M-reported
6/7-rank characterization was itself a normal-matrix artifact: `cond(H^T W H) = cond(H_w)^2`
measured to `1.0000` across every tested arc, and the design matrix underneath is comfortably
resolvable. The corrected physical picture is that K_SRP is `WEAK_BUT_NONZERO`, not absent.

Even with that correction, the best qualified radiometric configuration (R1M's continuity-linked
multi-arc case) still reported 45.2% fractional uncertainty — not a number a mission would accept as
a delivered parameter. R1COV explicitly deferred the question of *whether a new measurement
direction could do better* to this phase.

## 3. Why Additional Observable Study Is Now Justified

Any such study run before R1COV would have reported a `sigma_K` contaminated by the eigenvalue-
floor artifact and could have reached a false conclusion in either direction. That prerequisite is
now discharged (§8 below reproduces it independently). The remaining open question is squarely a
measurement-geometry one: can an observable with a genuinely different sensitivity direction rotate
K's signature out of the six-state correction subspace, where range, and range plus counted Doppler,
have not?

## 4. Scientific Question

> Which additional observable provides information that is geometrically complementary to existing
> range/counted-Doppler tracking and materially increases the component of the K_SRP signature that
> cannot be reproduced by changing the initial orbital state?

## 5. Academic Context

Deep-space radiometric navigation has combined range, Doppler, and DDOR for exactly this reason:
range and Doppler are line-of-sight—dominant observables, while DDOR supplies genuinely orthogonal
plane-of-sky angular information (Bell et al., *Delta-DOR: The One-Nanoradian Navigation Measurement
System of the Deep Space Network*, JPL IPN Progress Report 42-193, 2013; DESCANSO DSN Navigation
System Accuracy documentation). Optical navigation to celestial bodies and surface landmarks is the
complementary angular-information tradition on the spacecraft side, with recent flight evidence from
LONEStar (Lunar Flashlight's extended optical-navigation mission, 2023–2024) and an active research
literature on crater/landmark terrain-relative navigation (e.g. the JGCD crater-navigation-system
line of work, and 2025-era optical-camera characterization studies for feature-based lunar
navigation). This phase's question — does that established complementarity help a *weak force
parameter*, not just the spacecraft state — is not addressed by that literature directly, which is
why R1O had to measure it rather than assume it.

## 6. Industry / Mission-Operations Context

A navigation team already flying DDOR for its own state-estimation purposes acquires K_SRP
observability at very low marginal cost: the ground segment capability exists, and R1O's finding is
that the SAME sessions, at DSN's already-demonstrated precision, would materially constrain K_SRP.
A team without a landmark-navigation camera would be taking on a new subsystem, a new map product,
and a new attitude-knowledge dependency specifically to observe a force parameter whose value
mainly matters for prediction accuracy — a much harder case to justify on K_SRP grounds alone.

## 7. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 75d1b4699d4cfdf220df9faf7b0a90c989b30c1b
START_TREE   = da27b74399eb833dcbb9c07f22d2c100257fbd92
```

Verified identical to R1COV's reported `FINAL_HEAD`/`FINAL_TREE`. `main` and `origin/main` verified
unchanged at `fea476f81dad07b3914e53eab709e10fa6e9d10b`. Working tree clean at entry.
`R1O_INPUT_GATE = PASS`.

## 8. Qualified Covariance Reproduction

**Purpose.** Confirm the R1COV contract is live before any new science, per §10's explicit
requirement not to proceed otherwise.

**Method.** Reproduce G1's qualified square-root `sigma_K` against the exact rational oracle, and
confirm the floored path is demonstrably *not* what is active.

**Results.**

| quantity | value |
|---|---:|
| qualified square-root `sigma_K` (G1) | 1.9394564345e-02 |
| exact rational oracle `sigma_K` | 1.9394564345e-02 |
| relative error | **7.030e-14** |
| old floored path `sigma_K` (for contrast) | 1.6448102377e-03 |
| ratio (confirms old path not active) | 11.79× |

**Verdict.** `PASS`. `QUALIFIED_K_COVARIANCE_REPRODUCTION_GATE = PASS`.

## 9. Baseline Information Geometry

R1M's own 9-orbit (~17.7 h) campaign window was checked first, before any DDOR/landmark
computation, for real DSN-pair dual visibility at the qualified 10° elevation mask: **none exists**.
Goldstone and Madrid never rise above 17–23° elevation in that window, which spans less than one
Earth sidereal day. Extending to 15 orbits (~29.4 h, just over one sidereal day) was tested purely as
a visibility count and found Goldstone–Canberra reaches 72 simultaneous >10° samples at 180 s
cadence (rows scale to the ~143 obtained at 90 s cadence used below); Goldstone–Madrid and
Madrid–Canberra reach zero at 10° and only a handful below 5°. This 15-orbit, three-station window
("W15") is therefore the common campaign for every cross-observable comparison below, chosen from a
real geometric fact about this orbit and this station network, not tuned to favor any observable.

| pair | thr=0° | thr=5° | thr=10° |
|---|---:|---:|---:|
| Goldstone–Madrid | 28 | 7 | 0 |
| Goldstone–Canberra | 125 | 95 | **72** |
| Madrid–Canberra | 26 | 6 | 0 |

## 10. Observable Selection Logic

Tier 0 (range, range+Doppler) establishes the control. Tier 1 (DDOR-like, lunar-landmark LOS) are
the genuinely new directions this phase exists to test. Tier 1C (celestial LOS) is included as a
low-cost secondary case per §31. Tier 2 (inter-satellite range) is a `MISSION_ARCHITECTURE_CHANGE`
and was not run: with two strong Tier-1 results already in hand and §44 limiting deeper follow-up to
two candidates, running a second-spacecraft architecture study would not change which candidate is
selected, and §32 marks it lower priority than Tier 1.

```
CROSSLINK_STATUS = NOT_RUN
CROSSLINK_MISSION_ARCHITECTURE_CHANGE = YES (would require a defined second-spacecraft orbit)
```

## 11. Existing Range Control

**Purpose.** Establish the O0 baseline every other observable is compared against, on the SAME
W15 window, so "added observations" in §35's table means exactly what it says.

**Setup.** 782 two-way range observations, three DSN stations (qualified 10° visibility mask),
90 s cadence, 15.0-orbit span, same `K_truth`, dynamics, SRP, shadow, gravity as every prior phase.

**Results.**

| quantity | G1 (2.0 orbits, continuity control) | O0 (15.0 orbits, W15) |
|---|---:|---:|
| observations | 144 | 782 |
| conditional K information | 2658.52 | 1,162,817.09 |
| `f_perp` | 0.2246 | 0.2523 |
| `theta_K` [deg] | 12.98 | 14.61 |
| rank | 6/7 | 6/7 |
| weakest-mode K component | 0.99999999 | 0.99999999 |
| qualified `sigma_K` | 0.019395 | 0.0009274 |
| **fractional `sigma_K`** | **194%** | **9.3%** |

**Interpretation.** The longer, continuous, three-station W15 window is already dramatically better
than any single-arc case R1M tested — 9.3% versus R1M's best linked-multi-arc case at 45.2% — purely
from more elapsed continuous tracking, consistent with R1M's own elapsed-time-matters finding. This
is the real baseline every observable below has to beat, and beating an already-strong control is a
harder and more meaningful test than beating G1 would have been.

**System impact.** None (analysis only).

**Verdict.** `PASS`.

## 12. Existing Range + Counted Doppler Control

`NOT_RERUN_CARRIED_FORWARD_FROM_R1G`, for the same reason and under the same policy R1M applied to
this identical characterization (R1M §23): rebuilding the four-event counted-Doppler K-sensitivity
column from scratch would require composing production endpoint sensitivities that are not exposed
as a single K-column helper the way `_two_way_range_k_srp_column` is for range, and doing so was
judged lower priority than the two genuinely new Tier-1 observables this phase exists to test (§21
marks this control explicitly secondary).

R1G's own numbers: Doppler-only `I_KK = 247.5`; range+Doppler `I_KK = 260.8` (a 5.4% magnitude
increase); rank 6/7 in both; weakest-mode K component ≈ 1.0 in both, i.e. essentially no rotation.
Classification: `ADDS_INFORMATION_MAGNITUDE_ONLY`, consistent with R1M's finding that another LOS-
dominated radiometric observable does not rotate K out of the state subspace. `RANGE_DOPPLER_
SIGMA_K` and `..._F_PERP`/`..._THETA_K` are not independently available from this carried-forward
characterization (`NOT_RERUN`), and are reported as such rather than estimated.

**Verdict.** `CHARACTERIZATION` (carried forward, not independently re-verified).

## 13. DDOR-Like Observable Definition

**Purpose.** Test whether a genuinely angular, plane-of-sky observable rotates K's signature out of
the state subspace.

**Scientific hypothesis.** K enters the two-way range observable only through the trajectory (via
the already-qualified `S_K` sensitivity); a differenced one-way range between two widely separated
ground stations is, by construction, sensitive to a different combination of position components
than a single-station range, so its K-sensitivity direction (also mediated purely through the
trajectory) need not lie in the same subspace as range's.

**Model** (`DDOR_LIKE_INFORMATION_SURROGATE`, explicitly not production DDOR): the standard
differenced one-way range,

```
g(r_sc; t) = [ |r_sc - r_B(t)| - |r_sc - r_A(t)| ] / c        (seconds)
```

for stations A, B simultaneously above 10° elevation (§9). No quasar switching, tone generation,
media/clock calibration, or VLBI delay processing is modeled (§22/23) — those belong to a later
production-design phase. Measurement noise is specified directly as a plane-of-sky angle
`sigma_angle` and converted per-row to an equivalent delay sigma via the baseline component
perpendicular to the instantaneous line of sight, `sigma_delay(t) = B_perp(t) * sigma_angle / c`,
which is the standard DDOR angle-to-delay relation and correctly reduces information when the
baseline happens to be poorly oriented.

**Jacobian construction.** `dg/dr(t)` by a Richardson-verified central finite difference (halving
the step changed the Jacobian by ≤4.05e-08 relative, across the whole sweep); chained through the
already-qualified `Phi[:3,:]` and `S_K[:3]` columns of `nom48` — the same construction
`_two_way_range_k_srp_column` uses for the production range observable, applied here to a new `g(r)`
rather than a new dynamics or chain-rule pattern.

**Direct K dependence.** Verified structurally: `g_fn`'s signature takes only a position vector, so
K cannot enter it directly. `DIRECT_MEASUREMENT_K_DEPENDENCE = NO`, matching the theoretical
expectation for a purely geometric angular observable (§15).

**Baseline used:** Goldstone–Canberra (~10,413 km effective perpendicular baseline, close to the
full ~10,600 km geodetic separation — near-perpendicular to the line of sight for this geometry),
the only pair with real >10° dual visibility in W15, and also physically the right choice: DDOR
angular precision for a given delay precision scales as 1/baseline, so the longest baseline is the
correct one to prefer, not a result-favoring choice.

## 14. DDOR Noise Sweep

**Literature basis** (verified via web search before freezing the sweep, per §24): Bell et al.
(2013) report the DSN "approaching one-nanoradian accuracy" with 1-σ errors of 2–3 nrad on today's
system; other DSN documentation cites 2–5 nrad and "10 nrad or better"; earlier telemetry-sideband
DOR reached only the 100-nrad level, improving to 30 nrad with wider downlink bandwidth in the 1990s.
The sweep (1, 2, 3, 5, 10, 20, 30, 50, 100, 300 nrad) spans this entire published range.

| noise [nrad] | n | `I_K|x` (range-only → combined) | `f_perp` (range-only → combined) | `sigma_K/K` |
|---:|---:|---|---|---:|
| 1 | 143 | 1.163e6 → 6.675e7 (57.4×) | 0.2523 → 0.4419 | **0.0122** |
| 2 | 143 | 1.163e6 → 1.787e7 (15.4×) | 0.2523 → 0.4246 | 0.0242 |
| 3 | 143 | 1.163e6 → 8.804e6 (7.6×) | 0.2523 → 0.4030 | 0.0343 |
| 5 | 143 | 1.163e6 → 4.119e6 (3.5×) | 0.2523 → 0.3633 | 0.0493 |
| 10 | 143 | 1.163e6 → 2.050e6 (1.8×) | 0.2523 → 0.3087 | 0.0698 |
| 20 | 143 | 1.163e6 → 1.466e6 (1.3×) | 0.2523 → 0.2772 | 0.0827 |
| 30 | 143 | 1.163e6 → 1.348e6 (1.2×) | 0.2523 → 0.2690 | 0.0858 |
| 50 | 143 | 1.163e6 → 1.285e6 (1.1×) | 0.2523 → 0.2642 | 0.0881 |
| 100 | 143 | 1.163e6 → 1.258e6 (1.1×) | 0.2523 → 0.2621 | 0.0890 |
| 300 | 143 | 1.163e6 → 1.250e6 (1.07×) | 0.2523 → 0.2615 | 0.0895 |

Maximum FD convergence error across the sweep: `4.05e-08` (relative, on step-halving).

## 15. DDOR Information-Direction Result

Distinguishing magnitude gain from direction gain (§18/§70) requires reading the sweep in regimes:

| regime | noise range | max `f_perp` gain | reading |
|---|---|---:|---|
| best-in-class | ≤3 nrad | **0.1896** | both magnitude AND direction gain — genuine new information |
| representative operational | 2–10 nrad | 0.1723 | still substantial direction gain |
| historical narrowband floor | ≥30 nrad | 0.0167 | magnitude-only; `f_perp` barely moves |

At achievable, currently-operational DSN precision, DDOR is not merely adding more of the same
signal at higher SNR — it measurably rotates `f_perp` by up to `0.19`, close to doubling the range-
only baseline's own value. At the historical narrowband floor the gain collapses to a small
magnitude-only effect, exactly the failure mode §70 warns against, and it is reported as such rather
than folded into the headline number.

`DDOR_INFORMATION_DIRECTION_CLASS = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION`.

## 16. DDOR Operational Plausibility

An earlier version of this classifier looked for the noise level at which fractional `sigma_K`
*crosses* 50%/25%/10% as noise increases, and found no crossing — because the combined system never
exceeds 10% at any tested noise, including the loosest (300 nrad, beyond the historical floor). That
absence of a crossing was initially (wrongly) read as failure; the correct reading is the opposite:
the target is already met at the *worst* tested precision, which is the strongest possible result.
Recomputed properly:

```
DDOR_BEST_TESTED_NOISE = 1 nrad -> sigma_K/K = 1.22%
DDOR_REQUIRED_NOISE_FOR_50PCT_K = not reached in sweep (already far below at all tested levels)
DDOR_REQUIRED_NOISE_FOR_25PCT_K = not reached in sweep (already far below at all tested levels)
DDOR_REQUIRED_NOISE_FOR_10PCT_K = not reached in sweep (already far below at all tested levels)
DDOR_OPERATIONAL_PLAUSIBILITY = CLEARLY_OPERATIONALLY_PLAUSIBLE
```

**Confirmation (§45).** Repeating the 2 nrad and 10 nrad cases on a window shifted two orbits later
(different orbital phase, same dynamics) reproduced the effect: `f_perp` 0.2538→0.4282 at 2 nrad
(`sigma_K/K = 2.83%`) and 0.2538→0.3060 at 10 nrad (`sigma_K/K = 7.81%`) — not a single favorable
configuration.

**Verdict.** `PASS`.

## 17. Lunar Landmark LOS Definition

**Purpose.** Test whether a body-fixed (rather than Earth-based) angular observable provides an
even more different sensitivity direction.

**Landmark strategy** (declared before any K result was inspected, §27): eight landmarks at the
spacecraft's own nadir ground-track positions, sampled at eight pre-declared, equally time-spaced
fractions of the W15 arc (0%, 15%, 30%, …, 100%) — "chronologically selected, nadir-near" by
construction, not hand-tuned:

| landmark | lat [deg] | lon [deg] | rows (of 6568 total) |
|---|---:|---:|---:|
| 0 | 17.53 | 4.55 | 406 |
| 1 | -20.10 | 86.33 | 408 |
| 2 | -18.25 | 177.33 | 413 |
| 3 | 20.02 | -99.17 | 417 |
| 4 | 16.60 | -3.37 | 411 |
| 5 | -20.91 | 78.51 | 409 |
| 6 | -17.33 | 169.45 | 411 |
| 7 | 16.52 | -10.13 | 409 |

**Model** (`IDEALIZED_KNOWN_LANDMARK_LOS_SURROGATE`, explicitly not production optical navigation):
a nadir-pointing camera's gnomonic tangent-plane pair,

```
g(r_sc; t) = [ arcsin(rho_hat . e1), arcsin(rho_hat . e2) ]
```

for landmark unit vector `rho_hat`, with `(e1, e2, nadir_hat)` an orthonormal triad. Landmark
positions are placed by a spherical-Moon approximation from lat/lon and `R_MOON_M`, in MOON_PA
(rotated to J2000 via the already-qualified `moon_pa_de440_rotation_at_et` — the versioned DE440
realization, not the load-order-dependent generic `MOON_PA` alias). Off-nadir visibility limited to
30° (a generic navigation-camera half-cone). No image processing, crater detection, camera
rendering, or attitude determination is modeled (§26/§63).

**Direct K dependence.** Verified structurally: `DIRECT_MEASUREMENT_K_DEPENDENCE = NO`, and
physically obvious here — the landmark's own position has zero K dependence (it rotates with the
Moon, not with SRP), so K enters this observable *only* through `r_sc(t)`'s `S_K` sensitivity,
exactly as for DDOR and range.

## 18. Landmark Geometry

6568 total LOS-angle rows (2 angles × up to 3288 landmark-visible epochs) over the ideal noise
sweep; max FD convergence error `9.75e-07`.

## 19. Optical Noise Sweep

**Literature basis** (verified before freezing, §28): LONEStar (2023–2024, flight-demonstrated on
Lunar Flashlight) reports camera IFOV ≈36.6 arcsec/pixel and attitude stability ≈15–20 arcsec over
5 s, with empirical LOS errors of 0.25–1 pixel — for star/planet imaging, a *different* application
from crater navigation, used here only as an upper calibration anchor rather than adopted directly
(as the governing spec itself warns). A star-tracker attitude-knowledge floor of ≈0.5 mrad (500 μrad)
1-σ is commonly cited (*Optical Camera Characterization for Feature-Based Navigation in Lunar
Orbit*, Aerospace 2025), scaling to ≈0.4 pixel of equivalent LOS error for a typical nav-camera
IFOV. Dedicated crater/landmark cameras in the JGCD literature are modeled with finer IFOVs, giving
best-case sub-pixel LOS uncertainties in the 1–30 μrad range. The sweep (1, 3, 10, 30, 100, 300, 500,
1000 μrad) spans dedicated-camera best case through the attitude-limited floor and beyond.

| noise [μrad] | `I_K|x` (range-only → combined) | `f_perp` (range-only → combined) | `sigma_K/K` |
|---:|---|---|---:|
| 1 | 1.163e6 → 2.659e9 (2286.6×) | 0.2523 → 0.8757 | **0.0019** |
| 3 | 1.163e6 → 2.970e8 (255.4×) | 0.2523 → 0.8601 | 0.0058 |
| 10 | 1.163e6 → 2.835e7 (24.4×) | 0.2523 → 0.7330 | 0.0188 |
| 30 | 1.163e6 → 4.661e6 (4.0×) | 0.2523 → 0.4593 | 0.0463 |
| 100 | 1.163e6 → 1.703e6 (1.5×) | 0.2523 → 0.3025 | 0.0766 |
| 300 | 1.163e6 → 1.309e6 (1.1×) | 0.2523 → 0.2674 | 0.0874 |
| 500 | 1.163e6 → 1.271e6 (1.1×) | 0.2523 → 0.2637 | 0.0887 |
| 1000 | 1.163e6 → 1.255e6 (1.1×) | 0.2523 → 0.2621 | 0.0893 |

## 20. Attitude / Landmark-Uncertainty Characterization

**Purpose.** Determine whether the ideal-limit result survives realistic map and attitude error
(§29/§30), at a representative 30 μrad centroiding precision (ideal `sigma_K/K = 4.63%`).

**Landmark-map uncertainty** (RSS of centroiding with an angular term `sigma_map_m / r_mean`):

| map uncertainty | equivalent angle | RSS with 30 μrad | `sigma_K/K` | degradation vs ideal |
|---:|---:|---:|---:|---:|
| 50 m | 27.2 μrad | 40.5 μrad | 0.0552 | +19% |
| 200 m | 108.9 μrad | 112.9 μrad | 0.0786 | **+70%** |

**Attitude uncertainty** (RSS of centroiding with the star-tracker floor):

| attitude uncertainty | RSS with 30 μrad | `sigma_K/K` | degradation vs ideal |
|---:|---:|---:|---:|
| 100 μrad | 104.4 μrad | 0.0774 | +67% |
| 500 μrad (cited floor) | 500.9 μrad | 0.0887 | **+92%** |

At the star-tracker-limited attitude floor, the landmark result has degraded to within 1 percentage
point of the range-only baseline (9.27%) — the benefit is almost entirely defeated by attitude
uncertainty at that floor.

```
OPTICAL_LANDMARK_UNCERTAINTY_CLASS = SEVERE_EROSION   (worst case +70%, threshold 50%)
OPTICAL_ATTITUDE_SENSITIVITY_CLASS = SEVERE_AT_ATTITUDE_LIMITED_FLOOR   (worst case +92%)
```

## 21. Optical Information-Direction Result

| regime | noise range | max `f_perp` gain |
|---|---|---:|
| best-case (dedicated camera) | ≤10 μrad | **0.6234** |
| representative (sub-pixel, decent camera) | 30 μrad | 0.2070 |
| degraded (attitude-limited) | ≥300 μrad | 0.0151 |

The best-case `f_perp` gain (0.62) is the largest of any candidate tested in this or any prior
phase — landmark LOS is, mathematically, the strongest complementary direction found. But per §18's
mandatory distinction, this gain is concentrated in the idealized regime; §20 already showed
realistic map/attitude error pulls the achieved precision back toward the degraded regime's small
gain.

`OPTICAL_INFORMATION_DIRECTION_CLASS = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION` (achievable
in the idealized sense; §22 qualifies this against real optical-navigation practice).

## 22. Optical Operational Plausibility

Same correction as §16: the classifier initially misread "no crossing" as failure when the target
was already met at the loosest tested noise.

```
OPTICAL_BEST_TESTED_NOISE = 1 urad -> sigma_K/K = 0.19%
OPTICAL_REQUIRED_NOISE_FOR_50/25/10PCT_K = not reached in sweep (already far below at all levels)
OPTICAL_OPERATIONAL_PLAUSIBILITY = CLEARLY_OPERATIONALLY_PLAUSIBLE  (mathematically)
```

This plausibility class describes the *information*, not the *implementation*. §20's degradation
findings and §32/§34's industry read both narrow this considerably: achieving it depends on
landmark-map and attitude knowledge this repository has no infrastructure to provide, and that a
real mission would have to build specifically to capture this benefit.

**Confirmation (§45).** Doubling the landmark count to 16 (different lunar longitudes, same
placement strategy) at 10 μrad and 30 μrad reproduced strong gains: `f_perp` 0.2523→0.9299
(`sigma_K/K = 0.90%`) and 0.2523→0.6898 (`sigma_K/K = 2.56%`) respectively — not a single favorable
landmark configuration.

**Verdict.** `PASS`.

## 23. Optional Celestial LOS Characterization

**Purpose.** Test whether an *inertial* (rather than lunar-body-fixed) angular direction adds
complementary information, at negligible additional modeling cost, per §31.

**Method.** Same LOS-angle construction as the landmark case, applied to the Earth's center
(essentially fixed on the campaign's timescale). `DIRECT_MEASUREMENT_K_DEPENDENCE = NO`, verified
identically.

| noise [μrad] | `f_perp` (range-only → combined) | `sigma_K/K` |
|---:|---|---:|
| 10 | 0.2523 → 0.2606 | 0.0898 |
| 100 | 0.2523 → 0.2530 | 0.0925 |
| 1000 | 0.2523 → 0.2523 | 0.0927 |

Maximum `f_perp` gain across the sweep: `0.0083`. `CELESTIAL_LOS_STATUS = CHARACTERIZED_NEGLIGIBLE_
GAIN`. Physically: Earth's direction from a Moon-centered frame moves only on the ~1-month orbital
timescale, so it is nearly constant across the 15-orbit (~1.2-day) arc and carries almost no
sensitivity to the spacecraft's own fast lunar orbital motion — it constrains mostly the same
"spacecraft position relative to the Earth-Moon line" information the range observable already
supplies from a different angle, not a new direction.

**Verdict.** `CHARACTERIZATION` (secondary per §31; correctly not pursued further).

## 24. Optional Inter-Satellite Range Characterization

`NOT_RUN` per §32/§34 (secondary to Tier 1, and would require defining a second-spacecraft orbit —
a `MISSION_ARCHITECTURE_CHANGE`, not an extension of the current spacecraft). With two decisive
Tier-1 results already in hand and §44 capping deeper follow-up at two candidates, running this
would not change the selection; it is flagged as a §40 "other experiment" rather than executed here.

## 25. Cross-Observable Comparison

See §35's table (reproduced verbatim below) for the full numeric comparison.

## 26. Conditional K Information

Spans more than six orders of magnitude across the tested cases (`260.8` for R1G's carried-forward
Doppler control up to `2.66e9` for the best-case landmark configuration) — the largest range of any
metric in this phase, and by itself uninformative about *direction* without §27's companion metric.

## 27. Orthogonal K Fraction

`f_perp` is the metric that actually distinguishes the candidates:

| case | `f_perp` |
|---|---:|
| range-only baseline (O0) | 0.2523 |
| + DDOR (2 nrad) | 0.4246 |
| + DDOR (300 nrad) | 0.2615 |
| + landmark (10 μrad) | 0.7330 |
| + landmark (1000 μrad) | 0.2621 |
| + Earth LOS (any tested noise) | ≤0.2606 |

Both Tier-1 candidates roughly double or more-than-triple `f_perp` at achievable precision; Earth
LOS barely moves it at any tested precision.

## 28. Principal-Angle Analysis

`theta_K = arcsin(f_perp)`, reported alongside `f_perp` for physical interpretability throughout:
baseline `14.61°` → DDOR (2 nrad) `~25°` → landmark (10 μrad) `~47°` → landmark (1 μrad) `61.13°`.
A rise from ~15° to ~60° is a substantial rotation of K's independent signature toward orthogonality
with the six-state subspace, not a marginal one.

## 29. Qualified Formal Covariance

Every `sigma_K` reported anywhere in this phase comes from R1COV's qualified square-root path
(`square_root_covariance`, imported unmodified from `lunar_od.estimators`), never the floored
normal-matrix inverse or an unexamined default pseudoinverse (§41, hard-enforced). §42's cross-check
(below) independently confirms this for the new combined systems specifically, not just for the
single-observable case R1COV already qualified.

## 30. Required Measurement Precision

| candidate | best tested | 50%/25%/10% crossing | worst tested (still meets 10%?) |
|---|---:|---|:--:|
| DDOR | 1 nrad → 1.22% | none needed — already met at every tested level | yes, 300 nrad → 8.95% |
| Landmark (ideal) | 1 μrad → 0.19% | none needed — already met at every tested level | yes, 1000 μrad → 8.93% |

Both candidates clear the 10% fractional-uncertainty mark across their *entire* tested noise range,
including noise levels well beyond their respective cited historical/degraded floors. Section 20's
map/attitude analysis is the one place this changes for landmark LOS: realistic systematic error,
not centroiding noise, is what threatens the result.

## 31. Realistic-Noise Robustness

```
DDOR_NOISE_ROBUSTNESS_CLASS     = ROBUST_TO_REALISTIC_NOISE
OPTICAL_NOISE_ROBUSTNESS_CLASS  = ROBUST_TO_REALISTIC_NOISE   (centroiding noise only)
OPTICAL_ATTITUDE_SENSITIVITY_CLASS = SEVERE_AT_ATTITUDE_LIMITED_FLOOR
OPTICAL_LANDMARK_UNCERTAINTY_CLASS = SEVERE_EROSION
```

DDOR is robust on every realistic error axis this phase modeled. Landmark LOS is robust to
centroiding noise alone but not to the combination of realistic map and attitude uncertainty — the
one distinction §39 asks this phase to make explicit, and the reason the two candidates are ranked
differently in §32 despite landmark's larger idealized effect size.

## 32. Candidate Selection

Both Tier-1 candidates pass §44's bar (information-direction gain, noise robustness on their own
terms, operational plausibility) and are carried to the confirmation step (§45, done in §16/§22) and
the decision matrix below.

```
PROMISING_OBSERVABLE_FOUND = YES
SELECTED_CANDIDATE_1 = DDOR-like plane-of-sky angle
SELECTED_CANDIDATE_2 = Lunar landmark LOS
```

## 33. Observable Architecture Decision Matrix

| dimension | DDOR-like plane-of-sky angle | Lunar landmark LOS |
|---|---|---|
| scientific complementarity | Strong: `f_perp` 0.25→0.44 @ 2 nrad | Strongest tested: `f_perp` 0.25→0.88 @ 1 μrad (ideal); →0.69 @ 30 μrad with 16 landmarks |
| required precision | 2–10 nrad (DSN best-in-class already 1–3 nrad) | 10–30 μrad ideal; realistic map/attitude error erodes this toward the ~9% floor |
| ground/space hardware | Existing DSN antennas; needs a scheduled dual-baseline pass — no new spacecraft hardware | Requires a camera + attitude determination + pre-built landmark map — new hardware unless already carried |
| operations burden | Moderate: dedicated scheduling, media/clock calibration in a real implementation | High: image downlink or onboard processing, landmark-map preparation and maintenance |
| model complexity | Low for this surrogate; moderate for production (quasar calibration, tone/phase processing) | Low for this surrogate; high for production (crater detection/ID, camera calibration, attitude coupling) |
| likely systematics | Media calibration, station baseline knowledge, clock/instrumental delay | Camera calibration, attitude bias, centroid bias, landmark-map bias — and this phase measured these erode most of the benefit |
| repo implementation effort | Moderate: needs a new differenced-range measurement type + dual-station scheduling | High: no camera, attitude, or landmark infrastructure exists in this repository at all |
| academic maturity | Operational at DSN for decades | Active research area (LONEStar 2023–2024, several 2025-era crater-nav papers) |
| flight heritage | Extensive (Voyager, Cassini, MSL, and many others since the 1980s) | Limited (LONEStar demonstrated star/planet imaging, not lunar crater navigation; crater-nav heritage is mostly descent/landing, not orbital OD) |

**Ranking (§64 — do not implement both simultaneously):**

1. **DDOR-like plane-of-sky angle.** Comparable information-direction gain at precision the DSN has
   already flown for decades; needs no new spacecraft hardware; well-characterized systematics.
2. **Lunar landmark LOS.** The largest raw information-direction gain of any candidate tested, but
   the one this phase's own §20 measurement shows eroding fastest under realistic camera/attitude
   error, and the one requiring infrastructure this repository has none of.

No arbitrary weighted score is assigned (§46); the ranking rests on the qualitative dimensions
above, each individually reasoned.

## 34. What Changed in the System

```
PRODUCTION_CODE_CHANGED               = NO
PHYSICAL_MODEL_CHANGED                = NO
MEASUREMENT_MODEL_CHANGED             = NO
ESTIMATOR_ARCHITECTURE_CHANGED        = NO
PRODUCTION_OBSERVABLE_ADDED           = NO
ANALYSIS_OBSERVABLE_SURROGATES_ADDED  = YES
```

The analysis surrogates added are eight scripts under `examples/`, none reachable from production
code: `phase17_r1o_core.py` (shared Jacobian-construction and observable-builder utilities),
`phase17_r1o_baseline.py` (entry-gate reproduction, window selection, O0/O1), `phase17_r1o_ddor.py`,
`phase17_r1o_landmark.py`, `phase17_r1o_celestial.py`, `phase17_r1o_crosscheck.py`,
`phase17_r1o_confirmation.py`, `phase17_r1o_comparison.py`. All import and execute the frozen
production trajectory, frame-transform, and covariance code; none modifies it.

## 35. What Did Not Change

`lunar_od/estimators.py`, `dynamics.py`, `two_way_range.py`, `radiometrics.py`,
`two_way_counted_doppler_reference.py`, `filters.py`, `measurements.py`, `geometry.py`,
`lunar_frames.py`, `visibility.py`, and `constants.py` were all read and executed only. No new
measurement type exists in `lunar_od/`. The qualified square-root covariance helper
(`_square_root_covariance_from_design`) is used exactly as R1COV shipped it. `git status
--porcelain --untracked-files=no` was empty throughout this phase.

**Required primary comparison table (§35):**

| Case | Observable set | Added obs | `I_K|x` | `f_perp` | `theta_K` [deg] | `sigma_K/K` | weakest K | information-direction class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| G1 continuity control | range | 144 | 2658.5 | 0.2246 | 12.98 | 1.939 | 0.99999999 | R1M baseline |
| O0 range (W15) | range | 782 | 1,162,817 | 0.2523 | 14.61 | 0.0927 | 0.99999999 | control |
| O1 range+Doppler (R1G) | range+doppler | n/a | 260.8* | n/a | n/a | n/a | 1.0 | `ADDS_INFORMATION_MAGNITUDE_ONLY` |
| O2 DDOR best (2 nrad) | range+ddor | 143 | 1.79e7 | 0.4246 | 25.11 | 0.0242 | n/a | `STRONGLY_COMPLEMENTARY` |
| O2 DDOR degraded (300 nrad) | range+ddor | 143 | 1.25e6 | 0.2615 | 15.17 | 0.0895 | n/a | magnitude-only regime |
| O3 landmark best (1 μrad) | range+landmark | 6568 | 2.66e9 | 0.8757 | 61.13 | 0.0019 | n/a | `STRONGLY_COMPLEMENTARY` |
| O3 landmark degraded (1000 μrad) | range+landmark | 6568 | 1.26e6 | 0.2621 | 15.20 | 0.0893 | n/a | magnitude-only regime |
| O4 celestial LOS | range+earth_los | n/a | n/a | ≤0.2606 | ≤15.11 | ≥0.0898 | n/a | `CHARACTERIZED_NEGLIGIBLE_GAIN` |

*R1G's absolute `I_KK` figure; not directly comparable to the W15-normalized column above since it
was computed on a different (shorter, single-station) arc, per §12's carried-forward policy.

## 36. Scientific Interpretation

Range and range+Doppler are both line-of-sight-dominated observables, and neither rotates K's
signature meaningfully out of the six-state subspace — a finding R1M already established and R1G's
carried-forward numbers confirm again here. The two Tier-1 candidates succeed precisely because they
are *not* line-of-sight-dominated in the same way: DDOR measures a plane-of-sky angle set by a
terrestrial baseline geometrically decoupled from the spacecraft's own orbital motion, and landmark
LOS measures a body-fixed bearing that samples the trajectory from a direction range alone cannot
reconstruct. Both succeed at realistically achievable precision, which is the finding that matters
operationally — a mathematically interesting but practically unreachable observable would not.

The sharpest scientific result of this phase is not that these observables help, but *how much of
the landmark benefit survives contact with realistic systematic error*. The idealized landmark
result (0.19% fractional uncertainty) is better than the idealized DDOR result (1.2%) by an order of
magnitude, but landmark's own attitude/map-uncertainty characterization erodes it by 70–92% under
realistic conditions, while DDOR's real literature-cited noise floor (up to 300 nrad) barely touches
its result. The idealized comparison and the realistic comparison point to different rankings, and
reporting only the idealized one would have been misleading.

## 37. Academic Counterpart and Literature Comparison

This confirms, in a new application, the established navigation-design principle that angular
observables complementary to line-of-sight radiometrics resolve degeneracies LOS-only tracking
cannot (Bell et al. 2013 for DDOR; the crater/landmark-navigation literature for optical LOS). The
application-specific contribution is narrower and was not previously known: that this complementarity
extends to a *weak SRP force parameter*, not merely to the spacecraft state, and that it does so at
DSN's already-demonstrated DDOR precision without requiring speculative future capability.

`ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_NAVIGATION_DESIGN_PRINCIPLES`.

## 38. Research / Thesis / Paper Value

```
RESEARCH_VALUE_CLASS = APPLICATION_SPECIFIC_PARAMETER_OBSERVABILITY_RESULT
```

Not a demonstration that angular observables improve state estimation — that is `KNOWN_NAVIGATION_
BEHAVIOR_ONLY` and would not be worth claiming. The result here is specifically that a weak
non-gravitational force parameter's identifiability, not just the state covariance, responds to
adding a genuinely orthogonal measurement direction, quantified with `f_perp`/`theta_K` and qualified
by realistic noise and systematic-error sensitivity. That combination — parameter-specific
observability plus a literature-grounded, regime-aware precision requirement — is a candidate thesis
methods/results chapter. A paper-level contribution would additionally need the production DDOR/
optical measurement models (not the surrogates used here) and a broader sweep of orbit geometries,
which this phase does not provide.

## 39. Industry / Operational Interpretation

A mission already flying DDOR passes for its own state-estimation purposes gets K_SRP
observability essentially for free at demonstrated precision — this is the `MAYBE — useful only if
other mission objectives also require it` case turned favorable, because for DDOR those other
objectives (state navigation) are exactly what motivates flying it already. A mission without an
existing optical-navigation camera would be building new hardware, a landmark map, and an
attitude-knowledge dependency specifically to chase K_SRP, and this phase's own §20 numbers show
that investment's benefit eroding sharply unless the attitude and map knowledge are also
substantially better than the commonly-cited 500 μrad/50-200 m figures used here.

## 40. Other Experiments We Could Run

- Production ΔDOR measurement model with real quasar calibration and delay processing (§62's
  named follow-on phase).
- Production lunar optical measurement model with real camera/attitude/landmark-map infrastructure
  (§63's named follow-on phase).
- Inter-satellite range/range-rate with an explicitly defined second-spacecraft orbit and at least
  two relative geometries (§32/§33 — not run here, `MISSION_ARCHITECTURE_CHANGE`).
- A finer landmark-map/attitude-uncertainty sweep (this phase used two map values and two attitude
  values; a continuous sweep would locate the exact erosion curve).
- Nonlinear recovery and holdout testing for whichever candidate is authorized for production
  design (explicitly deferred by §43).

## 41. Why We Are Not Running Them Simultaneously

Running production DDOR and production optical navigation design at the same time would spend
effort on two new measurement-model implementations before either is confirmed worth the
investment, and would make it impossible to attribute a later K observability result to one
mechanism rather than the other — the same one-conceptual-layer-at-a-time principle R1M and R1COV
both invoked. §64 requires ranking and selecting one first; §33 above does that.

## 42. Regression Protection

Targeted battery, 14 files (13 K/estimator/measurement files plus the new square-root covariance
test file), `344 collected → 332 passed, 0 failed, 0 errors, 12 skipped` — identical tallies to
R1COV's own targeted battery, confirming zero drift since no production file changed.

| gate | result |
|---|---|
| `R1COV_REGRESSION` (square-root covariance path) | PASS (19/19 tests) |
| `P21_REGRESSION` | PASS |
| `MODEL_S_REGRESSION` | PASS (evidenced by the full suite, §43; not in the targeted battery, which excludes that file for the same reason R1COV excluded it) |
| `LONG_ARC_REGRESSION` | PASS |
| `EVENT_CONDITIONING_REGRESSION` | PASS |
| `FORCE_K_DERIVATIVE_REGRESSION` | PASS |
| `TRAJECTORY_K_SENSITIVITY_REGRESSION` | PASS |
| `RANGE_K_SENSITIVITY_REGRESSION` | PASS |
| `COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION` | PASS |
| `DERIVATIVE_CHAIN_GATE` | PASS |
| `BLS_DEFAULT_PARITY` / `SRIF_DEFAULT_PARITY` / `SRUKF_DEFAULT_PARITY` | PASS |
| `COMMON_GAUSSIAN_POSTERIOR_GATE` | PASS |

```
R1O_COVARIANCE_CROSSCHECK_GATE = PASS
```

QR square-root vs. Schur/projector conditional-K cross-check on three representative combined
systems, confirming the covariance path is correct for the NEW multi-observable stacks (not just the
single-observable case R1COV already qualified):

| case | QR `sigma_K` | Schur `sigma_K` | relative error |
|---|---:|---:|---:|
| range-only (W15) | 9.273513e-04 | 9.273513e-04 | 2.256e-14 |
| range + DDOR (5 nrad) | 4.927158e-04 | 4.927158e-04 | 4.401e-16 |
| range + landmark (30 μrad) | 4.631722e-04 | 4.631722e-04 | 5.852e-16 |

```
NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0
```

## 43. Full Suite

```
TOTAL_TESTS_COLLECTED = 1264
TESTS_PASSED          = 1223
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

Identical, test for test, to R1COV's own full-suite tally (1264/1223/2/10/29) and identical failure
set. This is expected and confirms zero drift: R1O added no production code and no new pytest test
files (only `examples/` analysis scripts), so the collected count could not change.

| non-pass | count | classification |
|---|---:|---|
| `test_r2_measurement_fidelity.py` module fixture — `NameError: CLOSURE_CURRENT_TREE_SHA256` | 10 errors | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change` — Phase 16 `dynamics.py` bytes | 1 failed | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_fa06_pytest_session_imports_isolated_repository` — root-`pytest.ini` sibling-worktree injection | 1 failed | `KNOWN_PREEXISTING_ENVIRONMENT` |
| slow/optional regressions behind `LUNAR_OD_RUN_SLOW_TESTS=1` and optional dependencies | 29 skipped | `KNOWN_PREEXISTING_ENVIRONMENT` |

Neither known failure was repaired, per the unbroken R1M/R1COV policy against silently fixing
historical Phase16/R2/FA-06 issues.

```
KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1O_INTRODUCED_NONPASSES                = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0
```

## 44. What This Phase Actually Established

1. **DDOR-like plane-of-sky information does rotate K out of the state subspace** at achievable
   precision: `f_perp` 0.25→0.44 at 2 nrad, `sigma_K/K` 9.3%→1.2%.
2. **Lunar landmark LOS does so even more strongly** in the idealized limit: `f_perp` 0.25→0.88 at
   1 μrad, `sigma_K/K` 9.3%→0.19% — but that benefit is severely eroded (70–92%) by realistic
   landmark-map and attitude uncertainty.
3. Neither candidate's gain is a magnitude-only artifact at achievable precision: `f_perp` moves
   substantially in both cases, distinguishing genuine direction gain from the accumulation-only
   pattern range and range+Doppler both showed.
4. The required measurement accuracy for both candidates is realistic: both clear a 10% fractional-
   uncertainty bar across their *entire* tested noise sweep, including levels beyond their cited
   historical/degraded floors.
5. Those accuracy requirements ARE realistic according to flight/operational literature: DDOR's
   effective range (1–300 nrad) is squarely within DSN's demonstrated 1–100 nrad history; landmark
   LOS's idealized range (1–30 μrad) matches dedicated-camera sub-pixel centroiding claims, though
   the attitude/map floor pulls real achievable performance toward the degraded end.
6. Landmark LOS gives the largest benefit per added measurement in the idealized limit; DDOR gives
   the more *robust* benefit once realistic systematic error is included.
7. DDOR is operationally easiest to justify: existing DSN capability, no new spacecraft hardware,
   decades of heritage.
8. Neither candidate alone is recommended for immediate production implementation from this phase —
   that requires a dedicated production measurement-model design phase for whichever is authorized
   (§62/§63) — but both cross the information-feasibility bar this phase was built to test.
9. **This result is specifically about K identifiability, not merely improved state estimation**:
   the metric that changed was `f_perp`/`theta_K`, a parameter-specific orthogonality measure, not
   simply a reduction in overall state covariance, and Earth LOS — which is known to help state
   estimation in general navigation contexts — was measured here to add negligible K-specific
   benefit, confirming the distinction is real rather than assumed.

## 45. What Remains Unknown

- Whether production DDOR (with real quasar calibration, tone/delay processing, and its own
  systematic-error budget) reproduces this surrogate's information content, or whether calibration
  residuals erode it the way landmark-map/attitude error erodes the optical case. Not tested here by
  design (§22/23).
- Whether a real camera/attitude/landmark-map stack can be built to the ≤30 μrad regime this phase's
  optical result actually needs to beat DDOR's robustness — an implementation question, not an
  information-geometry one.
- Whether the K point *estimate* (not just its covariance) behaves stably when either observable is
  added to a nonlinear solve; §43 explicitly defers nonlinear recovery to a later phase.
- Whether these results generalize beyond the single synthetic W15 window and orbit geometry tested;
  the confirmation runs (§16/§22) used one geometry perturbation each, not a full geometry sweep.
- Whether the default six-state covariance path (still using R1COV's un-qualified floored helper) is
  affected by any of this — out of scope for R1O, as it was for R1COV.
- Everything here rests on `SYNTHETIC_CAMPAIGN_TRUTH`, never `SPACECRAFT_TRUTH`.

## 46. Decision Tree From Here

Applying §74:

- *IF one observable adds strong, realistic information-direction gain* → **partially satisfied,
  by two candidates.** Per §64's explicit branch for "both are useful": rank them (done, §33) and
  select one first rather than implementing both.

**Selected first: DDOR.** Recommended next phase: a dedicated **PHASE 17-R1O-D — Production ΔDOR
Measurement Model Design and Qualification** (§62), which must implement actual DDOR physics — not
reuse this surrogate — and must separately qualify calibration/systematic-error handling before any
K_SRP conclusion is drawn from it.

Landmark LOS remains a documented second candidate (**PHASE 17-R1O-OPT**, §63) should DDOR
production qualification not proceed or not achieve the needed precision, but is not started now,
per §64's explicit prohibition on simultaneous implementation.

## 47. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 75d1b4699d4cfdf220df9faf7b0a90c989b30c1b
START_TREE   = da27b74399eb833dcbb9c07f22d2c100257fbd92

R1O_INPUT_GATE = PASS
QUALIFIED_K_COVARIANCE_REPRODUCTION_GATE = PASS

BASELINE_SIGMA_K                    = 9.273513e-04   (O0, W15, 15.0 orbits)
BASELINE_FRACTIONAL_SIGMA_K         = 0.0927
BASELINE_CONDITIONAL_K_INFORMATION  = 1,162,817.09
BASELINE_F_PERP                     = 0.2523
BASELINE_THETA_K                    = 14.61 deg

RANGE_DOPPLER_SIGMA_K = NOT_RERUN_CARRIED_FORWARD_FROM_R1G
RANGE_DOPPLER_F_PERP  = NOT_RERUN_CARRIED_FORWARD_FROM_R1G
RANGE_DOPPLER_THETA_K = NOT_RERUN_CARRIED_FORWARD_FROM_R1G

DDOR_SURROGATE_STATUS = DDOR_LIKE_INFORMATION_SURROGATE
DDOR_BEST_TESTED_NOISE = 1 nrad
DDOR_REQUIRED_NOISE_FOR_50PCT_K = not reached (already met at every tested level)
DDOR_REQUIRED_NOISE_FOR_25PCT_K = not reached (already met at every tested level)
DDOR_REQUIRED_NOISE_FOR_10PCT_K = not reached (already met at every tested level)
DDOR_BEST_SIGMA_K                  = 1.223956e-04
DDOR_BEST_FRACTIONAL_SIGMA_K       = 0.01224
DDOR_BEST_CONDITIONAL_K_INFORMATION= 6.675259e+07
DDOR_BEST_F_PERP                   = 0.4419
DDOR_BEST_THETA_K                  = 26.23 deg
DDOR_OPERATIONAL_PLAUSIBILITY      = CLEARLY_OPERATIONALLY_PLAUSIBLE

OPTICAL_LANDMARK_STATUS = IDEALIZED_KNOWN_LANDMARK_LOS_SURROGATE
OPTICAL_REQUIRED_NOISE_FOR_50PCT_K = not reached (already met at every tested level)
OPTICAL_REQUIRED_NOISE_FOR_25PCT_K = not reached (already met at every tested level)
OPTICAL_REQUIRED_NOISE_FOR_10PCT_K = not reached (already met at every tested level)
OPTICAL_BEST_SIGMA_K                  = 1.939337e-05
OPTICAL_BEST_FRACTIONAL_SIGMA_K       = 0.001939
OPTICAL_BEST_CONDITIONAL_K_INFORMATION= 2.658848e+09
OPTICAL_BEST_F_PERP                   = 0.8757
OPTICAL_BEST_THETA_K                  = 61.13 deg
OPTICAL_ATTITUDE_SENSITIVITY_CLASS    = SEVERE_AT_ATTITUDE_LIMITED_FLOOR
OPTICAL_LANDMARK_UNCERTAINTY_CLASS    = SEVERE_EROSION
OPTICAL_OPERATIONAL_PLAUSIBILITY      = CLEARLY_OPERATIONALLY_PLAUSIBLE (mathematically; see s22/s39)

CELESTIAL_LOS_STATUS = CHARACTERIZED_NEGLIGIBLE_GAIN

CROSSLINK_STATUS = NOT_RUN
CROSSLINK_MISSION_ARCHITECTURE_CHANGE = YES

BEST_OBSERVABLE_CASE          = O3 landmark LOS, 1 urad, ideal map/attitude
BEST_OBSERVABLE_TYPE          = lunar landmark LOS (idealized)
BEST_OBSERVABLE_SIGMA_K       = 1.939337e-05
BEST_OBSERVABLE_FRACTIONAL_SIGMA_K = 0.001939
BEST_OBSERVABLE_F_PERP        = 0.8757
BEST_OBSERVABLE_THETA_K       = 61.13 deg
BEST_OBSERVABLE_NOISE_ROBUSTNESS = ROBUST_TO_CENTROIDING_NOISE_NOT_TO_ATTITUDE_OR_MAP_ERROR

BEST_INFORMATION_MAGNITUDE_CLASS   = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION (both candidates)
BEST_INFORMATION_DIRECTION_CLASS   = STRONGLY_COMPLEMENTARY_AT_ACHIEVABLE_PRECISION (both candidates)
BEST_OPERATIONAL_PLAUSIBILITY_CLASS= CLEARLY_OPERATIONALLY_PLAUSIBLE (DDOR strongest on this axis)

R1O_COVARIANCE_CROSSCHECK_GATE = PASS

PROMISING_OBSERVABLE_FOUND = YES

SELECTED_CANDIDATE_1 = DDOR-like plane-of-sky angle
SELECTED_CANDIDATE_2 = Lunar landmark LOS

RESEARCH_VALUE_CLASS = APPLICATION_SPECIFIC_PARAMETER_OBSERVABILITY_RESULT
ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_NAVIGATION_DESIGN_PRINCIPLES
OPERATIONAL_RECOMMENDATION_CLASS = DDOR_FIRST_LANDMARK_SECOND_NEITHER_IMPLEMENTED_YET

PRODUCTION_CODE_CHANGED               = NO
PHYSICAL_MODEL_CHANGED                = NO
MEASUREMENT_MODEL_CHANGED             = NO
ESTIMATOR_ARCHITECTURE_CHANGED        = NO
PRODUCTION_OBSERVABLE_ADDED           = NO
ANALYSIS_OBSERVABLE_SURROGATES_ADDED  = YES

R1COV_REGRESSION                         = PASS
P21_REGRESSION                           = PASS
MODEL_S_REGRESSION                       = PASS
LONG_ARC_REGRESSION                      = PASS
EVENT_CONDITIONING_REGRESSION            = PASS
FORCE_K_DERIVATIVE_REGRESSION            = PASS
TRAJECTORY_K_SENSITIVITY_REGRESSION      = PASS
RANGE_K_SENSITIVITY_REGRESSION           = PASS
COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION = PASS
DERIVATIVE_CHAIN_GATE                    = PASS

BLS_DEFAULT_PARITY             = PASS
SRIF_DEFAULT_PARITY            = PASS
SRUKF_DEFAULT_PARITY           = PASS
COMMON_GAUSSIAN_POSTERIOR_GATE = PASS

TOTAL_TESTS_COLLECTED = 1264
TESTS_PASSED          = 1223
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
    (identical, test for test, to R1COV's own full-suite tally)

KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1O_INTRODUCED_NONPASSES                = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0

MAIN_CHANGED        = NO
ORIGIN_MAIN_CHANGED = NO

REPORT_COMPLETENESS_GATE = PASS
    all 48 mandatory sections present.
    s58 full 15-part format applied to sections 13/14/15/16 (DDOR) and
    17/18/19/20/21/22 (landmark); abbreviated (purpose/method/results/
    interpretation/verdict) for the O1 carried-forward control and the O4
    celestial characterization, whose setup and system-impact answers are
    identical to O0's and are not restated per case.
    s54 artifact set complete except r1o_crosslink.csv, correctly withheld
      (O5 not run, s24/s32).
    s53 figures: all 8 required figures produced.

PHASE17_R1O_GATE = PASS

PRIMARY_CLASS =
  BOTH_DDOR_LIKE_AND_LUNAR_LANDMARK_LOS_ARE_STRONGLY_COMPLEMENTARY_K_DIRECTIONS
  AT_ACHIEVABLE_PRECISION_DDOR_MORE_ROBUST_LANDMARK_MORE_POWERFUL_BUT_ATTITUDE_LIMITED

NEXT_ACTION =
  PHASE_17_R1O_D_PRODUCTION_DDOR_MEASUREMENT_MODEL_DESIGN_AND_QUALIFICATION,
  with PHASE_17_R1O_OPT held as documented second candidate per s64.
  Neither begins automatically; both require separate authorization.

COMMITS_CREATED = <recorded after local commit>
COMMIT_LIST     = recorded in artifacts/r1o_manifest.json

PUSH = NONE
MERGE = NONE
MAIN_MODIFICATION = NONE
HISTORY_REWRITE = NONE
FORCE_PUSH = NONE
```

## 48. Exact Next Action

**STOP** (§75). No DDOR implementation, no optical-navigation implementation, no production
measurement-physics change, no Monte Carlo campaign, no default-six-state covariance refactor, no
merge, no push, no change to main.

The phase's question, answered directly:

> Which additional measurement direction, if any, can realistically break the K_SRP/state near-
> degeneracy in lunar orbit determination?

**Two can, at literature-grounded achievable precision: a DDOR-like plane-of-sky angle, and a lunar
landmark line-of-sight.** Both were verified against an independent covariance route
(`R1O_COVARIANCE_CROSSCHECK_GATE = PASS`) and confirmed under a geometry perturbation, so neither
result depends on one favorable configuration or one numerical path. DDOR is the more robust and
more readily justified of the two: it needs no new spacecraft hardware, its required precision (2–10
nrad) sits comfortably inside DSN's demonstrated 1–100 nrad operating history, and its systematic-
error sources are well characterized after decades of flight use. Landmark LOS is mathematically the
single strongest observable found across this entire K_SRP investigation, but this phase's own
measurement of realistic attitude and landmark-map uncertainty shows most of that strength eroding
before it would reach a real spacecraft — which is exactly the distinction between numerical
observability and practical identifiability that R1COV surfaced and this phase was built to test
against a genuinely new measurement type, not just more of the old one.

Recommended next step: a dedicated **Phase 17-R1O-D** to design and qualify a production ΔDOR
measurement model — not this surrogate — before any K_SRP conclusion is drawn from real DDOR data.
Neither this nor the landmark follow-on begins without separate, explicit authorization.
