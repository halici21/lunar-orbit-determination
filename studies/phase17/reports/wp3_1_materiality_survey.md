# F3 WP3.1-R — Materiality of the force terms production lacks (REQ-DYN-007)

- **Date:** 2026-11-07
- **Predeclaration:** `studies/phase17/configs/wp3_1_materiality_predeclaration.json`, committed in `a93dbc7` before any
  number. Two amendments:
  - **A1** (`399fdc8`, before any result) corrects one unit-test statement;
  - **A2** (`b79e461`, after a gate result, so **exploratory**) adds a second full run with event-segmented
    integration.
- **Code:** terms `scripts/wp3_1_force_terms.py`, tests `tests/test_wp3_1_force_terms.py` (`3499929`), driver
  `scripts/wp3_1_materiality_survey.py` (`b79e461`), diagnostics `scripts/wp3_1_materiality_diagnostics.py`
  (`fda2749`, ray crossing `0a6262a`; exploratory).
- **Results:** `results/wp3_1_materiality_survey.json` (P0, as predeclared) and
  `results/wp3_1_materiality_survey_a2.json` (D2, amendment A2), both `aafba32`;
  `results/wp3_1_materiality_diagnostics.json` (`3b996c6`).
- **Production code changed:** none. This phase decides what WP3.2 must implement.

## 0. Final status

```
AS PREDECLARED (P0)    = every term WP3_1_INCONCLUSIVE (gate veto)
                         G1 G2 G3 G6 PASS; G4 FAIL in the 6 shadowed cells (floor 0.20-1.43 m > 1e-3 m);
                         G5 FAIL (worst column 2.30 > 1e-4)
AMENDMENT A2 (D2)      = exploratory (changed after the G4 result); every gate PASS
                         (floor <= 7.5e-5 m, G5 worst 9.5e-5)
  IMPLEMENT            = TIDE (230 sigma), LRP (165), EARTH_SHADOW on eclipse arcs (200),
                         SRP_LIGHT_TIME (1.41), EARTH_J2 (0.108, marginal)
  DOCUMENT_ONLY        = PN (<= 0.029), PLANETS (<= 0.0062, below the floor)
  REPORTED             = EARTH_J2_POLE (production's identity Earth-J2 frame, DYN-19): <= 2.2e-4
DIAGNOSTICS            = exploratory, no decision changed:
                         EARTH_J2 holds at the floor tolerance (0.1075);
                         SRP_LIGHT_TIME: the predeclared occulter delta = |r|/c is 3.07x too long at the
                         100 km contacts; with the ray-crossing delta = (r.u)/c the term is marginal
                         (0.109 in O2xE1, 0.101 in O3xE3)
DECISIONS (2026-11-08) = delegated by the owner to the agent (section 9): A2's classification sets the WP3.2
                         scope; REQ-DYN-007 closes after WP3.2's predeclared hold-out replication
```

"sigma" is the predeclared response: the largest |delta_theta_i / sigma_i| over the initial state and K_SRP, in some
cell, design and noise level (OD_MATERIAL at >= 0.1).

## 1. Question and method

Which force terms, missing from production, bias this project's OD by at least 0.1 sigma on a production
synthetic arc?

- **Baseline:** the production 6-state RHS (`dynamics._build_state_rhs`):
  - point-mass Moon;
  - Earth and Sun third bodies from the production Moon-centred providers (60 s grid);
  - cannonball SRP with K_SRP = 0.010 m²/kg and the conical lunar shadow.
  - Gate G3 confirms it equals `propagate_state` bit for bit in every cell.
- **One term at a time:** each term is added alone, after the production closure (`with_term`). The nonlinear
  trajectory difference gives the signature δy on the baseline's visible rows. δy uses production instantaneous
  range and range-rate *values*, because the production residual function zeroes residuals below 1e-7 m and
  1e-10 m/s.
- **OD impact:** δθ = least-squares fit of δy by θ = (x0, K_SRP).
  - Design: H = H_tilde [Φ | dx/dK] (production analytic H_tilde, production K-sensitivity propagation).
  - Prior: diag(100 m, 0.1 m/s)² on x0, none on K.
  - Solved by QR of the stacked whitened design.
  - Two designs: range only; range + range-rate.
  - Two noise levels: PROJECT (5 m, 1e-4 m/s); DSN_REALISTIC (1 m, 3e-5 m/s).
- **Population:** 3 orbits × 3 epochs, 100 km circular orbits, 1-day arcs at 60 s, Goldstone, Madrid and Canberra
  above 10°.

| orbit | definition | beta | lunar shadow |
|---|---|---|---|
| O1 | polar, Sun in the plane | 0° | 38.5 % of the day in umbra |
| O2 | polar terminator | 89.4–89.9° | none |
| O3 | lunar equator | −0.1 to 0.6° | 39–41 % |

| epoch | arc start (UTC) | Earth shadow |
|---|---|---|
| E1 | 2027-03-02T00:00:00 | none (Moon centre always lit, G6) |
| E2 | 2026-03-02T23:34:14 | total lunar eclipse; t_ge 2026-03-03T11:34:14 (geometric rule, 0.360° from the axis) |
| E3 | 2026-08-27T16:13:28 | partial eclipse; t_ge 2026-08-28T04:13:28 (0.465°) |

The geometric t_ge rule lands ~30 s from the published greatest eclipses, which use apparent positions (the ~20″
annual aberration of the Sun). Each arc is centred on its eclipse either way.

## 2. Terms

| term | model | finding |
|---|---|---|
| TIDE | elastic tide of the Moon raised by the Earth and Sun, degrees 2 and 3, GL1800F k2 = 0.024223, k3 = 0.0163, R = 1738 km (C7 model) | DYN-06 |
| LRP | lunar albedo (0.12) + thermal IR (emissivity 0.95, night 100 K) on the cannonball, K_LRP = K_SRP (C6 model) | DYN-07 |
| EARTH_SHADOW | a_SRP × (ν_E − 1), conical Earth shadow, R_E = 6378.137 km, occulters multiplied | DYN-08 |
| PN | Moon Schwarzschild term (IERS 2010 eq. 10.12 without Lense–Thirring and de Sitter) | DYN-09 |
| PLANETS | Venus and the Jupiter barycentre, indirect third body, DE440 GMs | C7 item |
| SRP_LIGHT_TIME | a_LT − a_SRP, a_LT = K P ν_LT [(1 − v·u/c) u − v/c] (Poynting–Robertson), Sun at emission, Moon at −v_moon\|r\|/c (meant as the ray crossing; 3.07× too far at the 100 km contacts, section 6.2), v relative to the Sun | DYN-17 |
| EARTH_J2 | indirect Earth J2 with the ITRF93 pole at the epoch | DYN-19 |
| EARTH_J2_POLE (reported) | EARTH_J2 minus the same with production's identity frame | DYN-19 |

## 3. Gates

| gate | P0 (as predeclared) | D2 (A2) |
|---|---|---|
| G1_UNIT | PASS: 18 new tests + C6/C7, 28 passed, none skipped | PASS (same) |
| G2_NULL (scale 0 bit for bit) | PASS, all terms and cells | PASS |
| G3_PARITY (= production `propagate_state` bit for bit) | PASS | PASS (`shadow_events=True`) |
| G4_FLOOR (≤ 1e-3 m) | **FAIL** in O1 and O3 (1.43, 1.25, 1.42 m; 0.20, 0.49, 0.79 m); O2 1.9–2.1e-5 m | PASS: 5.5–7.5e-5 m (O1), 1.9–2.1e-5 (O2), 3.0–7.2e-5 (O3) |
| G5_DESIGN (≤ 1e-4, O1×E1) | **FAIL**: K_SRP column 2.3, state columns 2.3e-4 to 1.8e-2 | PASS: K_SRP 9.5e-5 (range rate), 8.6e-5 (range); state ≤ 1.3e-6 |
| G6_ECLIPSE | PASS: E1 lit; E2 umbra; E3 below 0.5 | PASS |

Under P0 the integrator steps over the kink of the conical lunar-shadow acceleration at every penumbra contact,
about 48 times a day in O1 and O3. Phase D1 measured the same mechanism at 0.1–1.9 m on ~1.2-day arcs; D2 added
event segmentation (`PropagationProfile()`) for it. Before the survey a diagnostic with no force term isolated the
cause in O1×E1:

| O1×E1, 1 day | floor (1e-12 vs 1e-13) |
|---|---|
| conical shadow, no events (P0) | 1.434 m |
| shadow off (NO_SHADOW) | 2.1e-5 m |
| conical shadow, events on (D2) | 6.0e-5 m |

That G4 result is what A2 responds to, which is why A2 is exploratory.

## 4. Results under D2 (amendment A2)

| term | decision | max response | where | max \|δr\| (m) | max \|δy\| range (m) / range-rate (m/s) | acceleration RMS (m/s²) |
|---|---|---|---|---|---|---|
| TIDE | IMPLEMENT | 230 | O2×E2, R+RR, DSN | 86.6 | 85.3 / 7.6e-2 | 5.0e-7 |
| LRP | IMPLEMENT | 165 | O3×E2, R+RR, DSN | 4.5 | 3.75 / 2.7e-3 | 2.1e-8 |
| EARTH_SHADOW | IMPLEMENT (eclipse arcs) | 200 | O1×E2, R+RR, DSN | 6.39 | 5.80 / 4.6e-3 | 1.5e-8 |
| SRP_LIGHT_TIME | IMPLEMENT | 1.41 | O3×E3, R+RR, DSN | 0.0425 | 4.2e-2 / 3.5e-5 | 4.2–4.8e-12 (O1, O2), 7.5–9.0e-11 (O3) |
| EARTH_J2 | IMPLEMENT (marginal) | 0.108 | O1×E1, R+RR, DSN | 2.9e-3 | 2.6e-3 / 2.3e-6 | 1.1–2.0e-11 |
| PN | DOCUMENT_ONLY | 0.029 | O3×E1, R+RR, DSN | 0.0248 | 2.5e-2 / 2.2e-5 | 1.3e-10 |
| PLANETS | DOCUMENT_ONLY | 0.0062 | O2×E2, R+RR, DSN | 2.5e-4 (below the floor) | 2.5e-4 / 2.2e-7 | 1.5e-12 |
| EARTH_J2_POLE | reported | 2.2e-4 | O1×E2 | 1.2e-5 (below the floor) | 1.2e-5 / 1.1e-8 | 1.1e-13 |

Response per cell (largest over designs and noise levels), with max |δr| in metres; * = below 10× the cell's floor:

| term | O1×E1 | O1×E2 | O1×E3 | O2×E1 | O2×E2 | O2×E3 | O3×E1 | O3×E2 | O3×E3 |
|---|---|---|---|---|---|---|---|---|---|
| TIDE | 131 [38.1] | 46.7 [86.6] | 36 [80.2] | 220 [51.9] | 230 [68.4] | 181 [64.5] | 24.8 [72.3] | 80.6 [0.54] | 43.5 [11.3] |
| LRP | 69.7 [4.5] | 164 [4.46] | 155 [4.3] | 1.5 [0.21] | 0.33 [0.2] | 0.24 [0.19] | 146 [4.1] | 165 [4.06] | 158 [4.15] |
| EARTH_SHADOW | 0 | 200 [6.39] | 132 [4.28] | 0 | 12.2 [0.11] | 6.45 [0.12] | 0 | 195 [2.89] | 132 [3.41] |
| SRP_LIGHT_TIME | 0.018 [1.9e-3] | 0.028 [1.8e-3] | 0.021 [3.5e-3] | 0.109 [2.5e-3] | 0.017 [2.5e-3] | 0.017 [2.4e-3] | 0.993 [0.038] | 1.07 [0.043] | 1.41 [0.036] |
| EARTH_J2 | 0.108 [9.7e-4] | 5.6e-3 [2.0e-3] | 2.4e-3 [1.7e-3] | 0.019 [6.5e-4] | 0.015 [2.9e-3] | 0.017 [2.5e-3] | 2.1e-3 [7.4e-4] | 8.0e-3 [2.6e-3] | 3.7e-3 [1.8e-3] |
| PN | 0.019 [0.025] | 0.013 [0.025] | 7.3e-3 [0.025] | 0.011 [0.025] | 1.9e-3 [0.025] | 2.3e-3 [0.025] | 0.029 [0.025] | 0.019 [0.025] | 0.010 [0.025] |
| PLANETS | 1.1e-3* | 5.9e-4* | 3.3e-4* | 2.6e-4* | 6.2e-3* | 5.5e-3* | 2.5e-4* | 3.7e-4* | 2.9e-4* |

Largest response over the nine cells, by design and noise level:

| term | range, PROJECT | range, DSN | range + RR, PROJECT | range + RR, DSN |
|---|---|---|---|---|
| TIDE | 1.59 | 7.85 | 68.1 | 230 |
| LRP | 2.12 | 10.6 | 49.5 | 165 |
| EARTH_SHADOW (E2, E3) | 1.53 | 7.77 | 60 | 200 |
| SRP_LIGHT_TIME | 0.0092 | 0.047 | 0.42 | 1.41 |
| EARTH_J2 | 6.0e-4 | 3.0e-3 | 0.032 | 0.108 |
| PN | 2.1e-4 | 1.0e-3 | 8.7e-3 | 0.029 |
| PLANETS | 3.5e-5 | 1.8e-4 | 1.8e-3 | 6.2e-3 |

## 5. What the numbers mean

- **TIDE.** The largest term, as on LRO in C7: 0.5 m to 87 m in a day. The fit absorbs most of it into x0, but
  only by moving x0 by up to 230 sigma: 7.7 m in position in O2×E2.
- **LRP.** About 4 m in a day where the orbit is shadowed. It aliases into K_SRP: δK = −2.08e-3 m²/kg, **−21 % of
  K**, in O1 and O3 (165 sigma), and −11 % in O2. The lunar radiation pushes away from the sunlit surface, against
  SRP on the day side, so the fitted cannonball looks smaller. A K_SRP estimated without LRP is therefore biased low
  by about a fifth for this orbit class.
- **EARTH_SHADOW.** On the eclipse arcs SRP is off for 17–22 % of the day. A constant K absorbs that as
  δK = −1.9e-3 to −2.5e-3 m²/kg (−19 to −25 %), and x0 still moves by 2.4–9.1 m (200 sigma). The term is zero
  outside eclipses by construction; its decision covers the eclipse arcs only.
- **SRP_LIGHT_TIME.** As computed:
  - 1.0–1.4 sigma in O3, about 4 cm in a day;
  - 0.109 in O2×E1 and ~0.02 in O1.

  Section 6 shows that the O3 values come mostly from the predeclared occulter position. That position puts the
  shadow 3.07× too far at the contacts of a 100 km orbit. The term has two first-order parts.
  - *Aberration.* A body moving at ~30 km/s around the Sun sees the Sun's direction tilted by v/c ≈ 1e-4, the ~20″
    annual aberration.
    - The Poynting–Robertson force tilts with it: a transverse 4.6e-12 m/s² at K = 0.010.
    - The spacecraft's own orbital velocity adds a drag of (v_sc/c)·a_SRP ≈ 2.5e-13 m/s².
  - *Shadow light time.* The ray that grazes the limb passed the Moon (r·u)/c ≈ 2 ms before it arrives.
    - The shadow therefore lags the Moon's ~30 km/s motion by ~60 m.
    - In O3 that lag lies in the orbit plane and moves both contacts by 0.11 s.
    - The predeclared term uses δ = |r|/c ≈ 6 ms instead, which gives ~184 m and 0.34 s.
  - *Why the parts cancel.* In the Moon's frame the light arrives from the aberrated direction, and the shadow is cast
    along that same direction, so the aberrated force and the displaced shadow tilt together.
    - Either tilt alone makes the shadow asymmetric about the force. The sunlit arc then does net work, the
      semi-major axis drifts and the along-track error grows quadratically.
    - Together the two tilts largely cancel. In O3×E3, with the ray-crossing occulter, the parts give 1.35 cm and
      1.48 cm in a day and the full term 0.19 cm.
    - What remains is expected from the drag of the orbital velocity and a consistent 20″ rotation of the Sun
      direction.
  - *Clean case.* O2 has no shadow, so its 0.109 comes from the velocity terms of the force alone, at 129× its floor.
    It does not depend on the occulter.
- **EARTH_J2.** The tidal (indirect) Earth J2 at the Moon is 1–2e-11 m/s². It reaches 0.108 sigma in one cell
  (O1×E1, DSN-realistic, range-rate), just above the threshold, with a signal 16× the floor. It is material by the
  rule, marginally; section 6 tests it at the floor tolerance. Production already has Earth J2 as an option, off by
  default. The identity-frame question of DYN-19 does not matter: the pole part is ≤ 2.2e-4 sigma.
- **PN.** On a circular orbit the Schwarzschild term is radial: 3(GM)²/(c²r³) = 1.29e-10 m/s² at r = 1837 km.
  - It acts as a GM change δ = 3GM/(c²r) = 8.9e-11.
  - With the initial state fixed, the mean motion changes by 2δ. The along-track error therefore grows as 2δvt, which
    is 2.5 cm in a day; the survey computes 2.48 cm in every cell.
  - x0 absorbs it, leaving ≤ 0.029 sigma.
  - The assessment's estimate for DYN-09 (1e-9 to 1e-8 m/s²) is 8–80× the term the survey adds (IERS 2010 eq. 10.12,
    Schwarzschild part).
- **PLANETS.** ≤ 2.5e-4 m, below 10× the floor in every cell; ≤ 0.0062 sigma.

### What P0 would have said

Under P0 the shadowed cells give SRP_LIGHT_TIME 23 sigma, PN 13, EARTH_J2_POLE 13, PLANETS 12 and EARTH_J2 9.4.
These responses are the integration floor (0.2–1.4 m), not physics; every one of them is flagged below the floor.
The P0 file's summary leaves these cells out under the G4-fail cell rule, and the G5 veto stopped them from becoming
decisions.

The finding carries over to production use. `propagate_state` defaults to `shadow_events=False` (with the Adams
method), so it also steps over the penumbra kinks. In the shadowed 100 km orbits here, a single DOP853 run without
events has a floor of 0.2–1.4 m in a day at rtol 1e-12; the Adams default was not measured. On shadowed orbits,
small-force and cm-level comparisons need `PropagationProfile()` (assumptions-and-limits item 17).

## 6. Diagnostics after the result (exploratory)

These ran after the A2 numbers were known. They change no decision and no results file of the survey. Each uses
profile D2's cells, rows, design and fits.

### 6.1 SRP_LIGHT_TIME without uncut kinks

A2 cuts the integration at production's shadow contacts, but the term moves those contacts, so the term run still
steps over its own kinks.
- **Setup:** the baseline and each variant are integrated with DOP853 at a 1 s maximum step, which puts every kink
  inside a 1 s step.
- **Check:** the 1 s baselines agree with the event-segmented D2 baselines to 2.4e-5 to 1.0e-4 m.
- **Split:** the term is also split into its two parts.
  - *Aberration only*: the force K P ν [(1 − v·u/c) u − v/c] with production's geometric shadow.
  - *Shadow light time only*: the force along u, with the Sun at emission and the Moon displaced as predeclared, and no
    velocity in the force.

Response, with max |δr| in metres:

| cell | survey (D2) | full, 1 s steps | aberration only | shadow light time only |
|---|---|---|---|---|
| O3×E3 | 1.41 [0.036] | 1.29 [0.032] | 0.471 [0.013] | 1.76 [0.045] |
| O3×E1 | 0.993 [0.038] | 0.898 [0.035] | 0.344 [0.012] | 1.24 [0.047] |
| O1×E1 | 0.0178 [1.9e-3] | 0.0172 [1.9e-3] | 0.0291 [3.0e-3] | 0.0120 [1.1e-3] |
| O2×E1 | 0.109 [2.5e-3] | 0.109 [2.5e-3] | 0.109 [2.5e-3] | 0.0000 [1.5e-6] |

- **Kinks:** the uncut kinks inflate the O3 responses by about 10 %. O1 agrees to 4 % and O2 exactly.
- **Additivity:** the two parts have opposite signs and add nearly linearly. In O3×E3, 4.53 cm − 1.35 cm = 3.18 cm,
  against 3.22 cm for the full term.

### 6.2 Where the ray passes the Moon

The predeclared term puts the Moon at −v_moon δ with δ = |r|/c, the light time over the spacecraft's distance from
the Moon's centre.
- The ray that grazes the limb passed the Moon only (r·u)/c before it arrives, with u the light's direction.
- At the shadow boundary of a 100 km orbit, r·u = √(r² − R²) = 598 km, against r = 1837 km. The predeclared δ is
  therefore 3.07× too long.

`ray_crossing_check` tests this without a first-order expansion. It finds the shadow boundary from the exact closest
approach of the ray to the uniformly moving Moon. Setup: 100 km orbit, point Sun, Moon velocity 29.8 km/s in the
orbit plane and normal to the Sun line.

| occulter model | contact shift (s) |
|---|---|
| exact closest approach | −0.1118 |
| δ = (r·u)/c (ray crossing) | −0.1118 |
| δ = \|r\|/c (predeclared) | −0.344 |

The diagnostic of section 6.1 was rerun with δ = (r·u)/c. O2 has no shadow and is unchanged.

| cell | shadow light time, predeclared δ | shadow light time, ray crossing | ratio of max \|δr\| | full term, ray crossing |
|---|---|---|---|---|
| O3×E3 | 1.76 [0.0453] | 0.572 [0.0148] | 3.07 | 0.101 [1.9e-3] |
| O3×E1 | 1.24 [0.0472] | 0.404 [0.0154] | 3.07 | 0.060 [3.8e-3] |
| O1×E1 | 0.0120 [1.1e-3] | 0.0039 [3.7e-4] | 3.14 | 0.025 [2.6e-3] |

- **Scaling:** the shadow part scales by exactly the predicted factor.
- **Full term:** with the ray-crossing occulter it is 0.06–0.10 in O3, against 0.90–1.29. The aberrated force and the
  displaced shadow, 0.47 and 0.57 in O3×E3, nearly cancel.
- **Decision as computed:** it stands, since the survey's term is the predeclared one.
- **Re-evaluated (exploratory):** with the ray-crossing occulter, SRP_LIGHT_TIME is marginal: 0.109 in O2×E1, where
  the occulter does not enter, and 0.101 in O3×E3. By the predeclared rule it would still be IMPLEMENT, now at the
  threshold like EARTH_J2.
- **Implementation:** in O3, either part alone leaves a larger error than leaving both out (0.57 or 0.47, against
  0.10).

### 6.3 EARTH_J2 at the floor tolerance

O1×E1, baseline and term at (1e-13, 1e-14), profile D2:

| run | response | max \|δr\| (m) |
|---|---|---|
| survey | 0.1076 | 9.65e-4 |
| floor tolerance | 0.1075 | 9.75e-4 |

The marginal response is not an integration artefact.

## 7. For WP3.2

WP3.2 needs its own predeclaration. Section 9 records the scope decision. A2's decisions support:

1. **TIDE**: Earth and Sun, degrees 2 and 3, GL1800F Love numbers.
2. **LRP**: albedo and thermal IR. Implement it together with the K_SRP estimate, which LRP otherwise biases by about
   −21 %. The Earth's shadow on the lunar surface (eclipse arcs) is not in the C6 model.
3. **EARTH_SHADOW**: multiply SRP by ν_E. Refinements left out here: the ~2 % atmospheric enlargement of the
   umbra; the light-time displacement of the Earth's shadow (~38 km at the Moon).
4. **SRP_LIGHT_TIME**: marginal (0.109 in O2×E1; 0.101 in O3×E3 with the ray-crossing occulter, exploratory).
   - If implemented, it needs the P–R force with the Sun at emission and the shadow with the Moon at the ray
     crossing, δ = (r·u)/c, together and never one without the other.
   - Both cost one SSB state per evaluation.
   - The predeclared δ = |r|/c must not be used for low orbits.
5. **EARTH_J2**: marginal (0.108 in one cell). Production can already switch it on (identity frame, which does not
   matter at this level); whether to make it a default is the owner's choice.
6. **Integration profile**: every comparison at cm level or below needs event segmentation (`PropagationProfile()`).
   Making it the default for SRP-on propagation belongs with REQ-DYN-006 (WP3.3).

## 8. Not done and limits

- Synthetic production arcs only: 100 km circular orbits, three DSN stations, 1-day arcs. The LRO arc (C6/C7) is not
  revisited. Responses scale with the noise level, arc length and estimated parameters, so a different solve-for set
  (e.g. empirical accelerations) can absorb more.
- Box-wing SRP (WP3.4); the Earth's albedo and IR on the spacecraft (~1e-12 m/s²); lunar de Sitter and Lense–Thirring
  (~1e-11 m/s²); a lagged or frequency-dependent tide; lunar gravity beyond the point mass in the baseline.
- LRP is overestimated during the umbral phases of E2/E3 (the Earth's shadow on the lunar surface is not modelled).
- SRP_LIGHT_TIME as predeclared puts the occulter 3.07× too far at the 100 km contacts (section 6.2). Its
  survey values in O3 measure that model, not the physics.
- The A2 integration of SRP_LIGHT_TIME was not cut at the shifted contacts (A2 `not_cut`), which adds about 10 % in O3
  (section 6.1). A2's `not_cut` note gives ~0.1 s for the shift; the predeclared term moves the contacts by 0.34 s.
- The predeclaration's agent decisions were approved on 2026-11-08 (section 9): synthetic arcs, not LRO; box-wing
  deferred to WP3.4; base origin/main, independent of the F2 stack.

## 9. Decisions (2026-11-08)

The owner delegated the open decisions to the agent ("en doğru kararı ver ve uygula", 2026-11-08). The agent
decided as follows.

1. **REQ-DYN-007: A2's classification sets the WP3.2 scope; the requirement closes after WP3.2's hold-out
   replication.**
   - *P0 agrees where it is valid.* In O2, P0 is valid: no shadow kink, and G4 passes with a floor of ~2e-5 m. There
     P0 and D2 give the same response for every term to four digits, for example TIDE 230, EARTH_SHADOW 12.2,
     LRP 1.5, SRP_LIGHT_TIME 0.109, PN 0.011 and PLANETS 0.0062. G5's veto came from the shadowed O1×E1 design.
   - *Which decisions depend on A2.* The predeclared profile, in its valid cells, already finds TIDE, LRP,
     EARTH_SHADOW and SRP_LIGHT_TIME material and PN and PLANETS not. Only EARTH_J2's 0.108 comes from a shadowed
     cell, and so from A2 alone.
   - *Remaining status.* The classification stays exploratory under the amendment rule. REQ-DYN-007 therefore stays
     `partial`. WP3.2's predeclaration repeats the classification on a hold-out population (new epochs) with the
     production implementation, and the requirement closes on that confirmatory result.
2. **WP3.2 scope.**
   - **TIDE, LRP and EARTH_SHADOW go into production as opt-in terms.**
     - Defaults and frozen results stay unchanged. This follows the owner's earlier decisions: the BLS-LM damping
       schedule (2026-11-02) and the NAIF predict kernel, which was not adopted because it would change every 2027
       synthetic result.
     - EARTH_SHADOW needs its own contacts in the event-segmented profile; otherwise its penumbra kinks bring the
       integration floor back on eclipse arcs.
     - LRP must enter the K_SRP sensitivity, since it scales with the same area-to-mass ratio.
   - **SRP_LIGHT_TIME goes in opt-in as one package** (the P–R force with the Sun at emission, and the shadow with the
     Moon at the ray crossing), after the three terms above. Its O2 response of 0.109 holds under P0, D2, 1 s steps
     and either occulter.
   - **EARTH_J2 is already a production option.** Its identity frame is immaterial (≤ 2.2e-4). It joins the
     recommended configuration; the default stays off.
   - **Event segmentation stays opt-in** (`PropagationProfile()`). WP3.3 certifies the profiles, including the
     Adams default's floor on shadowed orbits.
   - **Recommended configuration for real data (F5/F6):** every material term on, Earth J2 on, event-segmented
     integration. The marginal terms are implemented because long arcs make them grow: the P–R drag's along-track
     error is quadratic in time.
3. **The predeclaration's agent decisions are approved.**
   - Synthetic arcs: the LRO days 262/263 are already used for model selection (RD-01), and synthetic arcs give a
     known truth.
   - Box-wing SRP in WP3.4: it needs the LRO macro-model and attitude.
   - Base `main`: WP3.1-R touches no F2 code.
