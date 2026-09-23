# PHASE 17-GEO — GEOMETRY-DEPENDENT K_SRP IDENTIFIABILITY IN LUNAR ORBIT DETERMINATION

> **The question, answered in plain language first (§73).** When the lunar orbit's altitude,
> inclination, Sun orientation, eclipse history and tracking geometry are changed, does K_SRP start
> to leave a measurement pattern that small initial position/velocity errors can no longer imitate?
>
> **No.** Across 48 predeclared primary geometries plus 24 post-hoc phase variants, the K signature
> never becomes more than ~28% independent of the six-state signature (`f_perp ≤ 0.28`,
> `theta_K ≤ 16.3°`). The predeclared "moderate direction gain" line is `f_perp ≥ 0.35`, and nothing
> comes close. What geometry *does* change, by a factor of ~170, is how much K information is
> collected (`sigma_K/K` from 4.4% to 761%). Geometry changes the *size* of the K signature. It does
> not change its *shape* relative to the state errors.
>
> Formal class: `GEOMETRY_CHANGES_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY`.
> `PHASE17_GEO_GATE = CHARACTERIZATION_COMPLETE`, not PASS, because the predeclared cross-geometry
> Jacobian gate failed (§39). The failure is measured to be ~5,000× too small to affect any result.

---

## 1. Executive Summary

1. **A second fixture defect, larger in effect than R1O-R's.** The canonical R1M/R1O range fixture
   (`phase17_r1m_core.build_range_arc`) builds the measurement geometry from the Sun and Earth
   rotation at one epoch and the **Earth's position at an epoch 5.83 days earlier**, clamped after 4
   orbits. Reproduced exactly (`f_perp = 0.2523`, `sigma_K/K = 9.27%`, 782 obs), then fixed on the
   same orbit, the canonical baseline becomes **`f_perp = 0.0915`, `sigma_K/K = 25.4%`**. The epoch
   offset, not the clamp, carries most of the effect (§19). Every absolute range-only baseline
   number in R1M, R1COV, R1O, R1O-D, R1O-OPT and R1O-R came from the offset fixture.
2. **No geometry breaks the K/state degeneracy.** On the consistent fixture, `f_perp` stays within
   0.03–0.28 across beta (0°–88°), altitude (30–1000 km), inclination (26.7°–150°), Earth-view
   geometry, lunar phase, and station network. No case reaches `MODERATE_DIRECTION_GAIN`, and
   `STRONG_DIRECTION_GAIN_FOUND = NO`.
3. **Geometry controls magnitude enormously.** `||b_K||` varies 94 → 18,750 and `sigma_K/K` varies
   4.4% → 761%. Full-Sun geometry *destroys* K information magnitude (β 88°: `sigma_K/K` 761%) while
   leaving direction unchanged.
4. **Eclipses do not create a distinct K signature.** H1 (from LRO's operational experience) is
   rejected. Eclipse-rich orbits have slightly *lower* `f_perp` than full-Sun ones, and removing the
   shadow from the same orbit (the production `NO_SHADOW` option) *raises* `f_perp` slightly.
5. **The direction metric is set more by tracking phasing than by coarse geometry.** Changing only
   the orbital phase at epoch moves `f_perp` over 0.10–0.28 on one plane, while `sigma_K/K` stays
   flat (§33). Coarse geometry sets *how much* K information there is; phasing against the DSN passes
   perturbs its *direction* by ±0.09, and never upward past 0.28.
6. **K magnitude control confirms the physics.** As predicted *before running* from the exact
   linearity `a = K·g`, `f_perp` is invariant to four decimals from K = 0.0025 to 0.04, and
   `sigma_K/K ∝ 1/K` exactly.
7. **Rigour.** Matched observation counts change no class (max `|Δf| = 0.0035`). Every plotted value
   recomputes from its cached design (max difference 0.0). The cross-geometry Jacobian gate FAILED its
   predeclared ratio criterion on a tolerance-independent ~10⁻⁶–3×10⁻⁵ model-level difference, which
   was measured to move `f_perp` by ≤ 3.8×10⁻⁶.

## 2. What R1O-R Corrected

R1O-R repaired `chain_to_augmented_columns`, which had unflattened the column-major STM in C order
and so used Φᵀ. That withdrew R1O's observable ranking (landmark `f_perp` 0.8757 → 0.2530, DDOR
0.4419 → 0.0701). GEO inherits the repair: every GEO state column comes from the production two-way
range Jacobian, which uses `order="F"`. The permanent regression `tests/test_r1o_stm_layout.py` was
re-run first (5/5 PASS, §17).

## 3. What Remains Valid from R1O-D

R1O-D's production ΔDOR result (`MAGNITUDE_ONLY`, below the range baseline at every noise level)
stands. Its original explanation, that the surrogate/production gap came from differential
light-time, was superseded by R1O-R: the STM defect explained 97–99% of it. GEO adds a further
caveat. R1O-D's comparisons were made on the offset fixture (§19), so its *absolute* `f_perp` values
belong to that geometry. Its *relative* conclusion (production ΔDOR does not beat range) was not
re-tested here.

## 4. Why the Research Question Changed

Both R1O candidates have now been examined in production form, and neither rotates K's information
direction. The remaining hypothesis was geometric: perhaps the canonical orbit was simply an
unfavourable geometry. GEO tests that with the physics and estimation mathematics frozen.

## 5. Plain-Language Definition of K Identifiability

Changing K_SRP changes the trajectory, and so the measurements. Changing the initial position and
velocity also changes the trajectory. If some combination of initial-state changes reproduces the
K-induced measurement change, an estimator cannot tell a K error from a state error. That is the
degeneracy. **K is identifiable to the extent its measurement pattern has a part no state change
can reproduce.**

## 6. f_perp Explained

Whiten the design with the measurement weights. The six state columns span a subspace; project the
K column `b_K` onto it (QR, never the normal matrix):

```
b_K = b_K,par + b_K,perp,     f_perp = ||b_K,perp|| / ||b_K||,     theta_K = arcsin(f_perp)
```

- `f_perp` near 0: almost all of K's effect can be imitated by adjusting the initial state.
- `f_perp` near 1: K produces a pattern no state change can mimic.

`f_perp` is a statement about *direction*. It is not a data-quality score.

## 7. Information Direction vs Information Magnitude

`sigma_K` (and `I_K|x = ||b_K,perp||²`, `||b_K||`) measure *how much* K information was collected.
More data or a stronger signal shrinks `sigma_K` even when `f_perp` is unchanged. This phase keeps the
two apart in every table and figure. It turns out to be the whole story: magnitude moves ~170×,
direction does not break out of the weak regime.

## 8. Literature Context

`docs/phase17_geo_literature_contract.md` (written before the grid was frozen). The operational
anchor is Slojkowski (NTRS 20140008968 §6.2): LRO could estimate C_R "for non-full-Sun orbit periods"
but "consistent estimation of C_R … is still not possible during full-Sun periods" with a cannonball
model. The same source attributes the full-Sun problem partly to the area model. The EKF follow-up
(ISSFD 2015) models C_R as a stochastic Vasicek process rather than a constant. Mazarico et al. 2018
was accessible only as a summary (publisher 403), and only its confirmed statements are used.

## 9. Solar Beta Geometry

`beta = arcsin(n̂ · ŝ)`, with `n̂` the osculating orbit normal and `ŝ` the Moon→Sun unit vector at
the same epoch. It is signed; `|beta| → 90°` means the orbit is face-on to the Sun. It is computed
at arc start, middle and end (J2 precession and the Sun's motion move it by ~0.3° over the arc).
The full-Sun boundary at the canonical radius is `|beta| ≈ arcsin(R_Moon/a) = 71.0°`.

## 10. Eclipse Physics

Eclipses come from the **production** conical-penumbra model with a spherical Moon
(`lunar_od/srp.py`), never from beta. An eclipse event is counted when `nu < 0.5`; the force always
uses the continuous `nu`. Earth shadow is not modelled. At the GEO epoch the Sun–Earth separation
seen from the Moon is 174.4° (new moon), so the Earth cannot occult the Sun for any GEO orbit.

## 11. Altitude and Inclination Motivation

Altitudes {30, 50, ~100 (canonical), 200, 500, 1000} km span LRO's operational band and higher
orbits, where eclipses shorten and the full-Sun boundary moves to lower |beta|. Inclinations
{26.7 (canonical), 45, 60, 90, 120, 150}° are measured from the **lunar mean equator**, the pole the
production J2 term uses. Beta was matched across inclinations at 0° and at the canonical 21.8°.

## 12. Tracking-Geometry Motivation

Slojkowski (§3, §6.4) reports that LRO's radial and cross-track accuracy follows a two-week
face-on/edge-on cycle. A distinctive K trajectory signature could still be poorly *projected* into
range. That separates dynamical observability from measurement projection.

## 13. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = fac1564e57c83b962cfa9e5332d371999accb876
START_TREE   = 9c34f227784cfe47c036d9c52635347de504f01f
MAIN_HEAD = ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b
```

The working tree held only the two known untracked R1O-D orphans, left untouched.
`GEO_INPUT_GATE = PASS`.

## 14. Graphify Discovery

`artifacts/phase17_geo_graphify_discovery.md`. The graph was current (no `.py` change since its
build) and was not refreshed. `graphify explain "K_SRP"` resolved only to docstrings; real symbols
(`SRPOptions`, `srp_acceleration_kernel`, `illumination_fraction`,
`propagate_state_with_k_sensitivity`, `_two_way_range_k_srp_column`, `build_range_arc`) gave useful
`affected`/`path` results. `graphify path "SRPOptions" "propagate_state_with_k_sensitivity"` found
**no path**, although the dependency is direct. An options object passed as an argument is not an
edge. `GRAPHIFY_DISCOVERY_GATE = PASS` (discovery performed, then verified in source).

## 15. Manual Dependency / Configuration Audit

Traced by hand (`GEO_RUNTIME_CONFIG_AUDIT = PASS`):

| item | where | finding |
|---|---|---|
| K_SRP | `srp.py` L9–29, L171–195 | `K = C_R·A/m`, m²/kg; only the product enters |
| SRP force | `srp.py` L229–252 | `a = P(d)·K·nu·û`, anti-solar; `P ∝ 1/d²` at the true distance |
| shadow | `srp.py` L307–355 | conical penumbra, spherical Moon; `NO_SHADOW` is a supported production mode |
| J2 pole | `dynamics.py` L69–81 | IAU mean lunar pole |
| third bodies | `phase17_r1m_core.py` L146–158 | **`mu_earth = mu_sun = 0`** in the canonical fixture, a literal argument |
| STM / S_K | `dynamics.py` L1249–1292 | Φ column-major `[6:42]`, S_K `[42:48]` |
| tolerances | `phase17_r1m_core.py` | rtol 1e-12, atol 1e-13 |
| stations, mask, cadence, noise | campaign module + fixture | 3 DSN, 10°, 90 s, 5 m |
| **epoch** | fixture + campaign module | **Sun & Earth rotation at 857806357; Earth position at 857302357 (−5.83 d), table clamped after 4 orbits** |

## 16. Scientific Provenance Audit

`GEO_PROVENANCE_GATE = PASS`.

| phase | conclusion | status after GEO |
|---|---|---|
| R1M | multi-arc cannot rotate K; normal-matrix floor artifact | **qualitative conclusion stands**; its absolute baseline (`f_perp` ~0.25) came from the offset fixture |
| R1COV | QR square-root covariance qualified | **valid**, and used by GEO for every `sigma_K` |
| R1O | two strongly complementary observables | **superseded** (R1O-R) |
| R1O-D | production ΔDOR `MAGNITUDE_ONLY` | result valid on its fixture; its absolute baseline came from the offset fixture |
| R1O-OPT | production optical `MARGINAL_DIRECTION_GAIN` | same as R1O-D |
| R1O-R | STM repair, R1O requalified | **valid**; its range baseline (0.2523) is the offset-fixture value |
| **GEO** | geometry changes magnitude, not direction; consistent canonical `f_perp = 0.0915` | new |

The prior reports were **not** edited. Whether to issue a fixture erratum is an owner decision (§59).

## 17. STM Repair Gate

`tests/test_r1o_stm_layout.py` 5/5 PASS. No ambiguous `reshape(6, 6)` exists in `lunar_od/` or any
analysis path (grep, excluding R1O-R's deliberate negative controls and an already-2-D argument).
GEO code never reshapes Φ. `STM_PACK_ORDER = STM_UNPACK_ORDER = F`. `R1O_R_STM_LAYOUT_GATE = PASS`.

## 18. K_SRP Parameterization Audit

`K_SRP_PARAMETERIZATION = K_SRP = C_R · A/m (m²/kg), only the product enters the runtime`.
An independent A/m sweep would re-sweep the same scaling:
`A_OVER_M_INDEPENDENT_SWEEP_JUSTIFIED = NO`. G5 sweeps K itself as the magnitude control.

## 19. Canonical Baseline Reproduction

The canonical orbit, measured: altitude 99.9 km mean (89.9–109.8), e = 0.0053, i = 26.71° (lunar
mean equator), period 7068 s, beta 21.8°, Earth-view 113° (edge-on side), eclipse fraction 38.8%,
16 eclipses averaging 43 min, 15 revolutions (29.4 h), 3 DSN stations.

| Earth ephemeris seen by the measurement model | obs | `f_perp` | `theta_K` | `I_K|x` | `sigma_K/K` |
|---|---:|---:|---:|---:|---:|
| **published** fixture (5.83 d early + clamped) | 782 | **0.2523** | 14.61° | 1.1628e+06 | **9.27%** |
| GEO builder fed the campaign's own table | 782 | 0.2523 | 14.61° | 1.1628e+06 | 9.27% (bitwise identical) |
| 5.83 d early, not clamped | 768 | 0.2067 | 11.93° | 7.290e+05 | 11.71% |
| correct epoch, clamped after 4 orbits | 968 | 0.0857 | 4.92° | 1.234e+05 | 28.47% |
| **correct epoch, whole arc (GEO fixture)** | **960** | **0.0915** | **5.25°** | **1.550e+05** | **25.40%** |

The published baseline reproduces **exactly**, and the GEO builder reproduces it **bitwise** when
given the same Earth table. So the Earth ephemeris is the only difference.
`GEO_BASELINE_REPRODUCTION_GATE = PASS`.

**Why the epoch matters physically.** At the true epoch the Moon is at new moon. Seen from the Moon,
the Earth is almost exactly anti-sunward, and SRP pushes the spacecraft anti-sunward, so the range
line of sight nearly coincides with the SRP direction. K's effect on range is then close to an
along-line-of-sight push, which an orbit adjustment imitates well. With the Earth displaced by
5.83 days of lunar motion (~77°), the fixture presented a different projection. It was more
favourable to K, and it was not the scenario's geometry.

`f_ref = 0.0915` (G0 consistent) is the reference for every class below.

## 20. Predeclared Geometry Grid

`artifacts/phase17_geo_predeclared_grid.json`, v2 (SHA-256
`2ba87a471c6b303e7d39a0607daed60bca693b21cb6075642ed7b64e4c7e754d`). The campaign refuses to run
against any other hash. The grid was written before any K metric existed: it holds 57 cases, the
thresholds, the stage rules, and the G5 prediction. `GEO_GRID_PREDECLARED = YES`.

**Amendment A1**, made after v1 (hash `40965dca…`) and still before any K metric. v1's geometry
showed that at the new-moon epoch the Earth-view angle ≈ 90° + beta for every polar orbit, so beta
and Earth-view could not be separated by any RAAN choice. A1 added three polar cases at the nearest
quarter phase (+6.85 d, Sun–Earth separation 90.13°), where the beta→Earth-view mapping flips. No
case was moved or removed.

Thresholds (absolute `Δ = f_perp − f_ref`): MAGNITUDE_ONLY `|Δ| < 0.02`; MARGINAL `0.02 ≤ |Δ| < 0.10`;
MODERATE `Δ ≥ 0.10` **and** `f_perp ≥ 0.35`; STRONG `Δ ≥ 0.25` and `f_perp ≥ 0.50`; DIRECTION_LOSS
`Δ ≤ −0.10`. A gain counts only if it survives matched count and validation neighbours.

## 21. Physical Validity Rules

Minimum altitude > 5 km, finite states and illumination, successful propagation, and an observation
count > 0. All 57 cases passed. The lowest altitude reached anywhere was 25.9 km (the 30 km case).
`INVALID_GEOMETRY_CASES = NONE`.

## 22. Time-Normalization Rules

Fixed duration D = 15 canonical revolutions (106,012 s = 29.4 h), inside LRO's operational 36–60 h
band. The G2 eclipse-rich series was also run at a fixed 15 revolutions of each orbit's own period:
27.8 h at 30 km, 53.5 h at 1000 km. `ALTITUDE_TIME_NORMALIZATION = FIXED_DURATION_AND_FIXED_REVOLUTION_COUNT`.

## 23. Matched-Observation-Count Rules

Per stage, `N_match` = the stage minimum. Each case keeps rows at `round(linspace(0, N−1, N_match))`
of its time-ordered rows (deterministic, time-stratified, predeclared). Result: the largest
full-vs-matched change in `f_perp` over all 54 matched cases is **0.0035**, even for G4's
948 → 252 thinning. **No case changes class.** `MATCHED_COUNT_CONTROL_GATE = PASS`.

## 24. Beta / Eclipse Results (G1, polar, 100 km, new-moon epoch)

| beta (mid) | eclipse | n_ecl | Earth-view | obs | `f_perp` | `theta_K` | `sigma_K/K` | `||b_K||` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6° | 39.5% | 16 | 103° | 955 | 0.0784 | 4.50° | 36.2% | 3520 |
| 19.4° | 38.8% | 15 | 97° | 956 | 0.1056 | 6.06° | 37.7% | 2514 |
| 39.4° | 36.1% | 15 | 117° | 976 | 0.1120 | 6.43° | 46.2% | 1933 |
| 54.4° | 31.0% | 15 | 132° | 1016 | 0.1259 | 7.23° | 61.7% | 1288 |
| 64.4° | 22.8% | 15 | 142° | 1068 | 0.1391 | 8.00° | 81.3% | 884 |
| 68.4° | 15.4% | 15 | 146° | 1106 | 0.1368 | 7.86° | 92.6% | 789 |
| 70.4° (grazing) | 7.5% | 15 | 148° | 1126 | 0.1227 | 7.05° | 98.4% | 828 |
| 72.4° | 0.0% | 0 | 150° | 1154 | 0.1149 | 6.60° | 105% | 828 |
| 79.4° | 0.0% | 0 | 157° | 1339 | 0.1142 | 6.56° | 201% | 437 |
| 87.4° | 0.0% | 0 | 165° | 1511 | 0.1394 | 8.01° | 761% | 94 |

`f_perp` moves 0.078 → 0.139 (the best is marginal, Δ = +0.048), while `sigma_K/K` moves 21×.
**H2 is not supported.** There is a small dip across the full-Sun boundary (0.139 → 0.115), but
`f_perp` recovers to 0.139 at 88°. **H1 is rejected.** Full-Sun `f_perp` (0.114–0.139) is *not* below
eclipse-rich (0.078–0.106). The grazing cases are spherical-Moon statements only (literature
contract C6). `BETA_EFFECT_CLASS = MARGINAL_DIRECTION_CHANGE`.

## 25. Altitude Results (G2, polar)

| alt | beta | normalization | rev | obs | `f_perp` | `sigma_K/K` |
|---:|---:|---|---:|---:|---:|---:|
| 30 km | 20° | fixed D | 15.9 | 866 | 0.1019 | 52.6% |
| 30 km | 20° | 15 rev | 15 | 784 | 0.1033 | 59.6% |
| 50 km | 20° | fixed D | 15.6 | 901 | 0.1055 | 47.3% |
| 100 km | 20° | fixed D | 15.0 | 956 | 0.1056 | 37.7% |
| 200 km | 20° | fixed D | 13.8 | 1037 | 0.1082 | 28.3% |
| 200 km | 20° | 15 rev | 15 | 1126 | 0.1084 | 25.6% |
| 500 km | 20° | fixed D | 11.1 | 1110 | 0.1165 | 17.5% |
| 500 km | 20° | 15 rev | 15 | 1472 | 0.1149 | 12.0% |
| 1000 km | 20° | fixed D | 8.2 | 1219 | 0.1155 | 10.7% |
| 1000 km | 20° | 15 rev | 15 | 2218 | 0.1206 | **4.42%** |
| 30–1000 km | 85° (full Sun) | fixed D | — | 1134–1574 | 0.104–0.114 | 207–593% |

Altitude changes `f_perp` by only 0.019 within the stage (below the 0.02 marginal line), under both
normalizations. It changes `sigma_K/K` 13×. Higher orbits see more of the arc, have longer periods,
and suffer SRP that is relatively stronger against gravity, so they collect more K information of the
same shape. `ALTITUDE_EFFECT_CLASS = MARGINAL_DIRECTION_CHANGE` under the predeclared rule (best
case vs `f_ref`, Δ = 0.029, which mostly reflects polar-vs-canonical). The altitude effect proper is
`MAGNITUDE_ONLY`.

## 26. Inclination / Plane Results (G3, 100 km, matched beta)

| i | `f_perp` beta 0 | `f_perp` beta 21.8° | `sigma_K/K` beta 0 / 21.8° |
|---:|---:|---:|---:|
| 26.7° | 0.0918 | **0.2814** | 28.7% / 28.3% |
| 45° | 0.0833 | 0.1556 | 32.6% / 28.0% |
| 60° | 0.0766 | 0.1348 | 36.2% / 29.9% |
| 90° | 0.0784 | 0.1057 | 36.2% / 38.1% |
| 120° | 0.0980 | 0.1136 | 28.6% / 38.3% |
| 150° | 0.1095 | 0.2502 | 25.1% / 30.6% |

At beta 21.8° the grid shows an apparent inclination trend: 0.28 at 26.7° and 150°, 0.11 at 90°. The
phase diagnostic (§33) shows this is **not** robustly an inclination effect. On the 26.7° plane,
orbital phase alone spans 0.099–0.281; on the 90° plane it spans 0.075–0.213. The two ranges
overlap almost entirely. At beta 0 the inclination effect is flat (0.077–0.110).
`INCLINATION_EFFECT_CLASS = MARGINAL_DIRECTION_CHANGE`, not attributable to inclination beyond the
phase-induced spread.

## 27. Earth / DSN Viewing Results (G4)

| case | beta | Earth-view | obs | `f_perp` | `sigma_K/K` |
|---|---:|---:|---:|---:|---:|
| G1 b00 (new moon) | 0.6° | 103° (edge-on) | 955 | 0.0784 | 36% |
| **quarter b00** | 0.6° | 171° (face-on) | 1573 | 0.1387 | 79% |
| G1 b20 (new moon) | 19.4° | 97° (edge-on) | 956 | 0.1056 | 38% |
| **quarter b20** | 19.4° | 12° (face-on) | 1562 | 0.1821 | 83% |
| G1 b80 (new moon) | 79.4° | 157° (face-on) | 1339 | 0.1142 | 201% |
| **quarter b80** | 79.4° | 72° (edge-on) | 948 | 0.1124 | 144% |
| RAAN root 2, b20 | 20.6° | 123° | 988 | 0.0821 | 41% |
| RAAN root 2, b80 | 80.6° | 176° | 1571 | 0.2199 | 271% |
| canonical, Goldstone only | 21.5° | 111° | 377 | 0.0867 | 38% |
| canonical, Madrid only | 21.5° | 111° | 252 | **0.0326** | 144% |
| canonical, Canberra only | 21.5° | 111° | 331 | 0.1054 | 43% |

At matched beta, face-on Earth views give higher `f_perp` in 3 of 4 pairs (Δ up to +0.076). The
decoupling is imperfect. The quarter-phase epoch also rotates the SRP direction relative to the line
of sight (§19), so Earth-view and SRP–LOS angle move together here, and the effect cannot be
assigned to one of them. With identical dynamics, the station set alone moves `f_perp` from 0.033
to 0.105: measurement projection is real and of the same size as every geometry effect.
`TRACKING_GEOMETRY_EFFECT_CLASS = MARGINAL_DIRECTION_CHANGE`.

## 28. K Magnitude Control (G5)

| K (m²/kg) | `f_perp` | `||b_K||` | `sigma_K` (m²/kg) | `sigma_K/K` |
|---:|---:|---:|---:|---:|
| 0.0025 | 0.091508 | 4302 | 2.540e-03 | 101.6% |
| 0.005 | 0.091508 | 4302 | 2.540e-03 | 50.8% |
| 0.01 | 0.091508 | 4302 | 2.540e-03 | 25.4% |
| 0.02 | 0.091508 | 4302 | 2.540e-03 | 12.7% |
| 0.04 | 0.091508 | 4302 | 2.540e-03 | 6.35% |

This matches the prediction recorded in the grid *before any run*. The force is exactly linear in
K, so the design column `∂z/∂K` does not depend on K, and absolute `sigma_K` is fixed by geometry
alone. A "stronger" SRP spacecraft is estimated to better *relative* precision, with exactly the same
direction. `K_MAGNITUDE_EFFECT_CLASS = MAGNITUDE_ONLY`.

## 29. Targeted Interaction Results (G6)

By the predeclared rule, an interaction surface runs only if some single-factor stage reaches
MODERATE. None did, so G6 is empty: `TOTAL_INTERACTION_CASES = 0`. The predeclared validation
neighbours (triggered only by MODERATE/STRONG cases) were likewise not required.

## 30. SRP RTN Interpretation

`docs/figures/phase17_geo/f6_srp_rtn_and_eclipse_timelines.png`. At beta 0°, SRP is entirely
in-plane (R/T share 0.44/0.55, N 0.01), square-wave-modulated by eclipses. At the canonical 21.5°
it is in-plane plus a steady cross-track part (0.33/0.43/0.24). At beta 88°, SRP is a **constant
cross-track push** (N share 0.95). A constant cross-track acceleration is almost exactly what a small
change in orbit-plane orientation produces, and range sees cross-track motion weakly. So `||b_K||`
collapses 37× and `sigma_K` explodes, while `f_perp` is unchanged. This is a physical explanation of
"C_R is not estimable in full Sun" through *magnitude*, with no need for an area-model error.

## 31. Eclipse-Modulation Interpretation

The counterfactual (`artifacts/phase17_geo_results.json`, stage CF) propagates the identical orbit
with the production `shadow_model="NO_SHADOW"`. It is analysis-only, never a default, never mission
truth.

| orbit | eclipse | `f_perp` with shadow | without shadow | Δ | `sigma_K/K` with / without |
|---|---:|---:|---:|---:|---:|
| canonical (consistent) | 38.8% | 0.0915 | 0.1130 | +0.0215 | 25.4% / 17.6% |
| polar beta 0 | 39.5% | 0.0784 | 0.0847 | +0.0063 | 36.2% / 23.2% |
| polar beta 20 | 38.8% | 0.1056 | 0.1032 | −0.0024 | 37.7% / 24.4% |

Switching SRP off in shadow does **not** give K a signature the state cannot imitate. If anything the
on/off structure slightly *reduces* `f_perp`, while it plainly costs magnitude (less sunlit time).
Result Path B is rejected. `ECLIPSE_EFFECT_CLASS = NO_DIRECTION_GAIN (MAGNITUDE_ONLY)`.

## 32. Measurement-Projection Interpretation

Three measurement-side levers each move `f_perp` by up to ~0.07–0.18: the station set (§27), lunar
phase (the angle between line of sight and SRP direction, §19/§27), and tracking phasing (§33). No
dynamical lever (beta, altitude, inclination, eclipses) moves it further. Where direction moves at
all in this problem, it is decided by how range *projects* a fixed dynamical K signature. It is not
decided by the signature's shape.

## 33. f_perp Results

All 57 cases are in `artifacts/phase17_geo_information_metrics.csv`. Over the 48 primary geometry
cases (G1–G4), `f_perp` = 0.0326–0.2814 (the minimum is single-station Madrid; the minimum for a
three-station geometry is G3 i60 beta 0 at 0.0766).

**POST-HOC phase diagnostic.** `artifacts/phase17_geo_phase_robustness.json` is not predeclared and
is used for no classification. Only the argument of latitude at epoch is varied (8 values):

| geometry | `f_perp` range over phase | spread | `sigma_K/K` range |
|---|---:|---:|---:|
| i 26.7°, beta 21.8° | 0.099–0.281 | 0.183 | 27.4–28.3% |
| i 90°, beta 21.8° | 0.075–0.213 | 0.139 | 37.8–38.9% |
| polar beta 88° | 0.139–0.188 | 0.048 | 756–767% |
| canonical plane: (e, u₀) canonical / e = 0 at canonical u₀ / e = 0, u₀ = 0 | 0.0915 / 0.1004 / 0.2815 | — | 25.4 / 27.6 / 28.3% |

Eccentricity contributes 0.009; orbital phase contributes 0.18. The grid's best value is a phasing
peak, not a geometric regime. The maximum over every phase tried is still 0.2815, below the 0.35
MODERATE line.

## 34. theta_K Results

`theta_K` = 1.9°–16.3° everywhere. On the consistent fixture the canonical value is 5.25°: K's
independent part is a 5° tilt out of the state subspace.

## 35. Conditional K Information

`I_K|x` follows magnitude, not direction. It is largest for high orbits and long arcs, and smallest
at full Sun (e.g. beta 88°: `||b_K||` 94, `I_K|x` ≈ 170). It is reported for every case in the
metrics CSV.

## 36. Qualified sigma_K

Every `sigma_K` comes from the R1COV QR square-root route (`square_root_covariance`). The normal
matrix is never formed; singular values come from an SVD of the scaled whitened design itself.
`GEO_INFORMATION_NUMERICS_GATE = PASS`. The range over the primary cases is 4.4% (1000 km,
15 revolutions) to 761% (beta 88°).

## 37. Singular-Mode / Correlation Diagnostics

In **every** case the weakest right-singular vector of the scaled design is essentially pure K (K
component 1.000). No geometry moves K out of the weakest mode. The K/state correlations are moderate
(|ρ| 0.3–0.73). Which components mimic K depends on geometry: x/vx on the canonical orbit, z at
polar beta 0, x/vx/z at beta 88°, y/vx at altitude. Design condition numbers are 1.2×10⁸–3.6×10⁹.

## 38. Full-Data vs Matched-Count Comparison

`docs/figures/phase17_geo/f5_full_vs_matched_count.png`: all 54 points sit on the identity line, with
a maximum deviation of 0.0035. Direction differences are not produced by one geometry simply
yielding more data.

## 39. Cross-Geometry Jacobian Qualification

`artifacts/phase17_geo_jacobian_checks.{csv,json}` covers six representative geometries (canonical,
polar beta 0 and 88°, 30 km, 1000 km, i 26.7° beta 0) on 4-revolution arcs. Seven columns each
(six state + K) are swept over five steps (×4 … ×0.25) against central differences of the
**production** nominal range on re-propagated trajectories.

**Predeclared criterion:** error at the best-resolved step ≤ 10× the measured step-halving floor.
**Result: FAIL.** The K column passes in all six geometries. State columns fail in five of six
(G1 beta 88° passes), with ratios up to 35.

The failure is characterized, not loosened (`artifacts/phase17_geo_jacobian_diagnosis.json`):

- **Shape.** The error is **flat across the 16× step range** (~1–3×10⁻⁶ relative for most columns,
  up to 3×10⁻⁵). It follows neither truncation (∝h²) nor roundoff (∝1/h). It is a systematic
  model-to-oracle difference.
- **Not integration tolerance.** Repeating everything at 10× and 100× tighter rtol/atol leaves it
  essentially unchanged (3.3e-6 → 2.8e-6 → 3.1e-6 canonical; 3.3e-5 → 2.5e-5 at 30 km). It is
  model-level, in the production two-way range Jacobian chain that GEO calls unmodified. It is
  consistent in size with R1O-D's 6.8×10⁻⁶.
- **Immaterial.** Substituting the full FD design for the analytic one moves `f_perp` by
  **≤ 3.8×10⁻⁶** absolute, about 5,000× below the smallest class threshold, and `sigma_K/K` by
  ~10⁻⁵ relative.

`GEO_CROSS_GEOMETRY_JACOBIAN_GATE = FAIL` (predeclared criterion, characterized, measured immaterial).
Under §71 this is a hard block, so the phase is **not** closed PASS. The geometry conclusions are
reported as characterization, conditional on the measured materiality.

## 40. Best Geometry Family

By `f_perp`: canonical-plane circular orbits at a favourable tracking phase (0.281). Robustly, no
geometry *family* stands out once phase is varied (§33). By magnitude: high altitude, eclipse-rich,
long arc (1000 km, beta 20°, 15 revolutions: `sigma_K/K` 4.4%).

## 41. Weakest Geometry Family

By magnitude: full Sun (beta ≥ 80°, `sigma_K/K` 200–761%). By direction: single-station tracking
(Madrid 0.033) and low-beta mid-inclination orbits (0.077).

## 42. Physical Explanation of the Difference

Magnitude is set by how much of the SRP acceleration is sunlit, how long it acts, and how strongly
range projects it. Full Sun puts SRP cross-track, where range is nearly blind, and so collapses
magnitude. Direction is set by whether the time-integrated K displacement has a pattern outside the
six-dimensional space of initial-state perturbations. For a force that is smooth, or smoothly
switched, at orbital frequency, the state perturbations (which also live at orbital frequency) span
nearly all of it in every geometry tested. Eclipses do not supply the missing independence, because
their on/off timing is itself phase-locked to the orbit.

## 43. What Changed in the Repository

New files only: `examples/phase17_geo_{core,grid,campaign,fixture_attribution,jacobian_checks,
jacobian_diagnosis,phase_robustness,figures}.py`, `docs/phase17_geo_literature_contract.md`, this
report, `docs/figures/phase17_geo/` (9 PNG figures + 9 data CSVs), and `artifacts/phase17_geo_*`.

## 44. What Did Not Change

No file under `lunar_od/`, no test, no historical report, no configuration. The canonical fixture
`phase17_r1m_core.py` was deliberately left as is: the GEO builder fixes the epoch in new code and
reproduces the old one bitwise on request.

## 45. Academic Interpretation

Did geometry change the *shape* of the K signature relative to state errors, or only make the same
signature larger? **Only larger (or smaller).** This extends R1M's result from "more data does not
rotate K" to "different orbits do not rotate K either", across the synthetic lunar-orbit class
tested. The static K_SRP degeneracy with the initial state is structural in this problem.

## 46. Literature Comparison

- **Literature established:** LRO could not estimate C_R consistently in full Sun (S1), and C_R is
  handled stochastically in operations (S2).
- **This repository reproduced:** full Sun is where K becomes unestimable, via a **21× loss of
  magnitude**, not a loss of direction.
- **Newly observed here:** eclipses are *not* what makes K estimable in direction (the counterfactual
  shows no gain), and lunar phase / tracking phasing matter as much as orbit geometry.
- **Remains synthetic:** truth = model (no area error), cannonball spacecraft, spherical-Moon shadow,
  no third-body gravity, one epoch pair, range only.

## 47. Operational Interpretation

For a navigation team this argues against relying on geometry to make a *static* K_SRP solve-for
well-conditioned. Geometry can make K's *precision* acceptable in some windows (high altitude,
eclipse season, long arcs), but never makes K cleanly separable from the orbit. A K error will
always be partly absorbed into the state. That supports treating K as a consider parameter or a
stochastic/empirical force term, as LRO operations do, over a static solve-for. This is an
interpretation; no estimator was changed.

## 48. Thesis / Paper Significance

A clean negative result with a mechanism: geometry modulates K information magnitude by ~170× while
the direction metric stays bounded. The predeclared prediction (G5) and the counterfactual (CF) make
it falsifiable, not merely descriptive. A second contribution is methodological: an analysis fixture
with a 5.83-day Earth-epoch offset shifted the canonical `f_perp` 2.8×, and it was found only by
tracing configuration by hand.

## 49. Limitations

Truth = model, so no mismodelling. Cannonball SRP only. Spherical-Moon conical shadow (grazing cases
are model statements). No Earth/Sun third-body gravity and no high-order lunar gravity (frozen from
the canonical fixture). Range only: the counted-Doppler secondary check (§36 of the spec) was **not
run**, because no production Doppler K-column builder exists and building and qualifying one is its
own step. One circular-orbit family per stage. Tracking-phase sensitivity is measured for 3
geometries only. Linearized information geometry, with no nonlinear estimation. The Jacobian gate
failed (§39).

## 50. Synthetic-Truth Boundary

K_SRP = 0.01 m²/kg is `SYNTHETIC_CAMPAIGN_TRUTH`. No statement here is a statement about a real
spacecraft's K observability.

## 51. Higher-Fidelity SRP Deferral

`SPACECRAFT_CONFIGURATION_GENERALIZATION = DEFERRED_PENDING_HIGHER_FIDELITY_SRP`. Attitude, panel
self-shadowing and multi-plate effects cannot be represented by K alone.

## 52. Monte Carlo Deferral

No Monte Carlo and no nonlinear estimator were run. Because no geometry breaks the degeneracy, a
geometry-targeted nonlinear campaign is not motivated by this phase.

## 53. Regression Protection

Re-run: the STM-layout regression (5/5); the full suite (§54), which contains the R1COV,
K-derivative, trajectory/range/Doppler K-sensitivity, production range, counted Doppler, ΔDOR,
optical, BLS/SRIF/SR-UKF default-parity and linear-Gaussian oracle tests. GEO changed no file those
tests cover.

## 54. Full Suite

See §58. The non-pass set is identical to the R1O-R baseline (the FA-06 isolated-import guard, the R2
protected-tree byte gate, and 10 dependent R2 errors, all pre-existing). `lunar_od/` is
byte-identical to START_TREE.

## 55. What This Phase Actually Established

1. The canonical fixture's measurement geometry used an Earth 5.83 days out of epoch (and clamped).
   Corrected, the canonical `f_perp` is 0.0915, not 0.2523.
2. Across 48 predeclared geometries, beta, eclipses, altitude, inclination, Earth view and lunar
   phase change K information magnitude by ~170× but never lift `f_perp` above 0.28.
3. Eclipse on/off modulation does not create a distinct K direction (counterfactual).
4. Full Sun makes K unestimable through loss of magnitude (cross-track SRP), consistent with LRO
   operations.
5. K magnitude is a pure relative-precision effect, exactly as predicted from linearity.
6. The direction metric is sensitive to tracking phasing (±0.09) and to the station set, more than to
   coarse geometry.

## 56. What This Phase Did NOT Establish

Real-spacecraft observability. Behaviour under force-model error (the likely LRO mechanism).
Behaviour with Earth/Sun third-body gravity or high-order gravity. Range + Doppler behaviour.
Topographic eclipse timing. Nonlinear estimator performance. Statistical calibration. That the
production range Jacobian's ~10⁻⁵ discrepancy is understood (it is only shown to be immaterial here).

## 57. Decision Tree From Here

- The fixture epoch defect affects the absolute baselines of six prior reports. Options: issue a
  fixture erratum phase (R1O-R style), or accept GEO's consistent baseline going forward and annotate.
- The Jacobian gate failed. Options: owner accepts the measured-immateriality characterization; or a
  focused investigation of the production range Jacobian's ~10⁻⁵ model-level difference.
- The science question now becomes Result Path D's: should K_SRP be a static solve-for at all?
  Compare static solve-for, consider parameter, stochastic K, and empirical acceleration, on the
  consistent fixture.

## 58. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = fac1564e57c83b962cfa9e5332d371999accb876
START_TREE   = 9c34f227784cfe47c036d9c52635347de504f01f
FINAL_HEAD   = (the GEO commit; see git log)
FINAL_TREE   = (the GEO commit's tree)
MAIN_HEAD        = fea476f81dad07b3914e53eab709e10fa6e9d10b
ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b

GEO_INPUT_GATE           = PASS
GRAPHIFY_AVAILABLE       = YES (graphifyy 0.9.65)
GRAPHIFY_REFRESHED       = NO (graph current: no .py change since its build)
GRAPHIFY_DISCOVERY_GATE  = PASS
GEO_RUNTIME_CONFIG_AUDIT = PASS (found: mu_earth = mu_sun = 0; 5.83 d Earth-epoch offset + clamp)
GEO_PROVENANCE_GATE      = PASS

R1O_R_STM_LAYOUT_GATE = PASS
STM_PACK_ORDER   = F
STM_UNPACK_ORDER = F

K_SRP_PARAMETERIZATION = K_SRP = C_R * A/m [m^2/kg]; only the product enters the runtime
A_OVER_M_INDEPENDENT_SWEEP_JUSTIFIED = NO

GEO_LITERATURE_CONTRACT_GATE = PASS
GEO_GRID_PREDECLARED = YES (v2 sha256 2ba87a47..., amendment A1 geometry-only, pre-K)

GEO_BASELINE_REPRODUCTION_GATE = PASS (published 0.2523 exact; builder bitwise)

BASELINE_ALTITUDE          = 99.9 km mean (89.9-109.8)
BASELINE_INCLINATION       = 26.71 deg (lunar mean equator)
BASELINE_BETA              = 21.8 deg (epoch), 21.5 deg (mid-arc)
BASELINE_ECLIPSE_FRACTION  = 0.388 (16 eclipses, mean 43 min)
BASELINE_ORBIT_PERIOD      = 7068 s

BASELINE_F_PERP                    = 0.0915 (consistent)   [published fixture: 0.2523]
BASELINE_THETA_K                   = 5.25 deg              [14.61 deg]
BASELINE_CONDITIONAL_K_INFORMATION = 1.550e+05             [1.163e+06]
BASELINE_SIGMA_K_FRAC              = 0.254                 [0.0927]

BETA_CASE_COUNT              = 10
ALTITUDE_CASE_COUNT          = 18
INCLINATION_CASE_COUNT       = 12
TRACKING_GEOMETRY_CASE_COUNT = 8
K_MAGNITUDE_CASE_COUNT       = 5 (4 + reference)
TOTAL_PRIMARY_GEOMETRY_CASES = 48 (+2 G0, +4 G5, +3 eclipse counterfactual = 57 run; +27 post-hoc diagnostics)
TOTAL_INTERACTION_CASES      = 0 (predeclared rule not triggered)

ALTITUDE_TIME_NORMALIZATION = FIXED_DURATION_AND_FIXED_REVOLUTION_COUNT

MATCHED_COUNT_CONTROL_GATE       = PASS (max |df| 0.0035, no class changes)
GEO_INFORMATION_NUMERICS_GATE    = PASS (QR projector, SVD of design, R1COV sqrt covariance)
GEO_CROSS_GEOMETRY_JACOBIAN_GATE = FAIL (predeclared ratio; tolerance-independent ~1e-6..3e-5
                                          model-level difference; f_perp impact <= 3.8e-6)

BEST_F_PERP                 = 0.2814
BEST_F_PERP_GEOMETRY        = G3_i26.7_bcan (i 26.7, beta 21.8, 100 km, circular, u0 = 0) -- a phasing peak
BEST_THETA_K                = 16.34 deg
BEST_SIGMA_K_FRAC           = 0.0442 (G2_alt1000_b20_fixN)
BEST_MATCHED_COUNT_F_PERP   = 0.2818
BEST_MATCHED_COUNT_GEOMETRY = G3_i26.7_bcan
WORST_F_PERP                = 0.0326
WORST_F_PERP_GEOMETRY       = G4_net_Madrid (single station); 3-station worst G3_i60.0_b00 = 0.0766

ECLIPSE_RICH_F_PERP = 0.0784 (polar beta 0, 39.5%) .. 0.1056 (beta 20, 38.8%)
ECLIPSE_POOR_F_PERP = 0.1227 (beta 70.4, 7.5%)
FULL_SUN_F_PERP     = 0.1142 .. 0.1394 (beta 72-87)

BETA_EFFECT_CLASS              = MARGINAL_DIRECTION_CHANGE
ECLIPSE_EFFECT_CLASS           = MAGNITUDE_ONLY (no direction gain; counterfactual)
ALTITUDE_EFFECT_CLASS          = MARGINAL_DIRECTION_CHANGE by rule; MAGNITUDE_ONLY within stage
INCLINATION_EFFECT_CLASS       = MARGINAL_DIRECTION_CHANGE (within phase-induced spread)
TRACKING_GEOMETRY_EFFECT_CLASS = MARGINAL_DIRECTION_CHANGE
K_MAGNITUDE_EFFECT_CLASS       = MAGNITUDE_ONLY (as predicted)

STRONG_DIRECTION_GAIN_FOUND = NO

SPACECRAFT_CONFIGURATION_GENERALIZATION = DEFERRED_PENDING_HIGHER_FIDELITY_SRP

PRODUCTION_CODE_CHANGED   = NO
ANALYSIS_CODE_CHANGED     = YES (new examples/phase17_geo_*.py only)
CONFIGURATION_CHANGED     = NO
PHYSICAL_MODEL_CHANGED    = NO
MEASUREMENT_MODEL_CHANGED = NO
ESTIMATOR_CHANGED         = NO
K_SOLVE_FOR_MATH_CHANGED  = NO
COVARIANCE_METHOD_CHANGED = NO

TOTAL_TESTS_COLLECTED = 1465
TESTS_PASSED          = 1424 (1278 + 146 subtests)
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
  identical to the R1O-R baseline; non-passes = FA-06 isolated-import guard
  (KNOWN_PREEXISTING_ENVIRONMENT) + R2 protected-tree byte gate and its 10 dependent
  errors (KNOWN_PREEXISTING_PROVENANCE); lunar_od/ byte-identical to START_TREE

GEO_INTRODUCED_NONPASSES   = 0
NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES          = 0

MAIN_CHANGED        = NO
ORIGIN_MAIN_CHANGED = NO

PHASE17_GEO_GATE = CHARACTERIZATION_COMPLETE
PRIMARY_CLASS    = GEOMETRY_CHANGES_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY
                   (Result Path D also holds: STATIC_K_SRP_REMAINS_WEAKLY_IDENTIFIABLE_ACROSS_TESTED_LUNAR_ORBIT_GEOMETRIES)
NEXT_ACTION      = see s59

PUSH  = NONE
MERGE = NONE
```

## 59. Exact Next Action

1. **Owner decision on the fixture epoch defect.** It shifts the absolute range-only baseline of R1M,
   R1COV, R1O, R1O-D, R1O-OPT and R1O-R (0.2523 → 0.0915). Recommended: an R1O-R-style fixture
   erratum phase before any further K science. The logic is the one R1O-R used: repair first,
   generalize second.
2. **Owner decision on the Jacobian gate FAIL.** Accept the measured-immateriality characterization,
   or authorize a focused look at the production two-way range Jacobian's tolerance-independent
   ~10⁻⁵ difference.
3. Then the Result-Path-D question on the consistent fixture: static K solve-for vs consider
   parameter vs stochastic K vs empirical acceleration. Not started here.
