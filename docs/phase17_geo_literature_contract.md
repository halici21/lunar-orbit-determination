# PHASE 17-GEO — LITERATURE CONTRACT

Written **before** the geometry grid was frozen and before any K metric was computed. Each claim is
tagged:

- **LITERATURE RESULT** — what a cited source states, quoted or closely paraphrased from the pages
  actually read.
- **REPOSITORY HYPOTHESIS** — what this phase tests because of it.
- **REPOSITORY RESULT** — filled in from this phase's own numbers in the main report
  (`docs/phase17_geo_geometry_identifiability.md`, §24–§33).

The literature motivates the grid. It does not answer the question for this repository.

`GEO_LITERATURE_CONTRACT_GATE = PASS` — every claim below was read from its source this session, or
is explicitly marked as not verified in full text.

## Sources actually read

| ID | Source | Access |
|---|---|---|
| S1 | S. Slojkowski, *Lunar Reconnaissance Orbiter Orbit Determination Accuracy Analysis*, NASA NTRS 20140008968 | full text, pp. 1–16 read |
| S2 | S. Slojkowski, J. Lowe, J. Woodburn, *Orbit Determination for the Lunar Reconnaissance Orbiter Using an Extended Kalman Filter*, ISSFD 2015 (NTRS 20150019754) | full text, pp. 1–6 read |
| S3 | E. Mazarico et al., *Orbit determination of the Lunar Reconnaissance Orbiter: Status after seven years*, Planet. Space Sci. 162, 2–19 (2018) | **abstract/summary only**: publisher returned 403, PubMed a cookie wall. Only statements confirmed by accessible summaries (search excerpts, NASA PGDA product page 65) are used. |

## C1 — Full-Sun periods defeat static C_R estimation (the central motivation)

- **LITERATURE RESULT (S1 §6.2, p.11).** LRO has "twice-yearly full-Sun exposure periods, where it does
  not experience any umbra or penumbra due to lunar eclipse … each last about 36 days". "Estimation of
  C_R using a cannonball area model is possible for non-full-Sun orbit periods with the GSFC-GRAIL-270
  model. … Consistent estimation of C_R, even with the GRAIL gravity model, is still not possible during
  full-Sun periods when using the cannonball area model, so FDF LRO operations currently only applies
  the value of 1.67 and does not attempt to estimate C_R."
- **Confounder the source itself names.** S1 attributes the full-Sun problem to "the coarse SRP and
  spacecraft modeling". A multi-plate model with definitive attitude fixed full-Sun *prediction*
  (S1 Table 4: 325 m → 128 m). So the literature does not separate "no eclipses" from "area-model error
  that eclipses happen to average out".
- **REPOSITORY HYPOTHESIS H1.** With a *correct* cannonball model (truth = model, so no area error),
  eclipse-rich geometries give K_SRP a signature less imitable by initial-state errors than full-Sun
  geometries: `f_perp(eclipse-rich) > f_perp(full-Sun)`.
- **REPOSITORY RESULT.** Main report §24, §31.

## C2 — High beta degrades prediction

- **LITERATURE RESULT (S2 §1).** Predictive accuracy "has been poorer in the elliptical commissioning
  orbit … particularly during high beta-angle periods". S1 Fig. 4 plots |β| up to ~85° over the
  nominal mission.
- **REPOSITORY HYPOTHESIS H2.** f_perp decreases monotonically as |β| rises through the full-Sun
  boundary (|β| ≈ 71° at the canonical radius).
- **REPOSITORY RESULT.** Main report §24.

## C3 — Face-on vs edge-on Earth viewing geometry

- **LITERATURE RESULT (S1 §3, p.5; §6.4, p.14).** "In 'face-on' orientation, when the LRO orbit normal
  vector is aligned with the Earth-Moon line and the entire orbit is visible from Earth, tracking
  provides poor observability of the radial component." "Cross-track OD accuracy is best in face-on
  geometry, and poorest in edge-on geometry." The effect follows a two-week cycle. Orbit geometry drove
  radial error 10–70 m and cross-track error 100–800 m above baseline noise.
- **REPOSITORY HYPOTHESIS H3.** At matched Sun geometry, the Earth-view angle changes the *measurement
  projection* of the K signature, so f_perp differs between face-on and edge-on members of a matched
  |β| pair.
- **REPOSITORY RESULT.** Main report §27, §32.

## C4 — SRP scale factors are modelled stochastically in operations

- **LITERATURE RESULT (S2 abstract, §2).** "A Vasicek stochastic model produces better estimates of the
  coefficient of solar radiation pressure than a Gauss-Markov model." SRP uncertainty enters as
  "along-axis" process noise on a stochastic scale factor, plus optional "off-axis" white noise. "No
  process noise is added when the satellite is in eclipse."
- **Relevance.** Operations does not treat C_R as a well-observed static constant. That is the
  operational counterpart of this project's Result Path D question (§59): should K be static at all?
- **REPOSITORY HYPOTHESIS.** None tested here. Stochastic K is explicitly out of scope (§74). Carried
  as interpretation only.

## C5 — Arc length and time-dependent SRP error

- **LITERATURE RESULT (S1 §6.3, p.14).** A 36-hour arc beat a 60-hour arc. "Dynamical modeling errors,
  particularly time-dependent ones like solar radiation pressure variations due to variable spacecraft
  area, do not accumulate as much as for longer arcs." Arc lengths: LRO FDF 60 h → 36 h (S1 Table 1);
  LOLA precision OD "typically 2.5 days" (S3 summary, PGDA product 65).
- **Use here.** Justifies the fixed-duration reference D = 15 canonical revolutions ≈ 29.4 h, which sits
  inside the operational 36–60 h band. It is not a hypothesis.

## C6 — Eclipse timing and lunar shape

- **LITERATURE RESULT (S3 summary; PGDA 65).** Precision LRO OD uses "knowledge of the lunar shape to
  compute the incident solar flux on the spacecraft" and models "self-shadowing of spacecraft panels".
  **Not verified in full text:** any quantitative statement about how scale factors vary with β, or
  about grazing-eclipse timing errors.
- **Limitation carried.** Production here uses a **spherical-Moon conical penumbra**
  (`lunar_od/srp.py`). Near the full-Sun boundary a real, topographic Moon changes eclipse timing. The
  grazing cases (|β| ≈ 69–73°) are therefore statements about the spherical model, not mission-truth
  eclipse timing (§28).

## C7 — Orbit regimes for the altitude grid

- **LITERATURE RESULT (S1 §2).** LRO flew a 50 km mean-altitude circular polar orbit (nominal), then a
  30 × 180 km / 40 × 180 km frozen orbit (commissioning/extended). S1 Table 1 SRP model: spherical 14 m²,
  C_R 1.0 → 1.67.
- **Use here.** The altitude grid {30, 50, ~100 (canonical), 200, 500, 1000} km spans the LRO operational
  band (30–50 km), the canonical ~100 km regime, and higher orbits where eclipses shorten and the
  full-Sun boundary moves to lower |β|. Chosen before any K metric was computed.

## What the literature does NOT establish for this repository

- Nothing in S1–S3 measures K/state **information direction** (`f_perp`). Their results are
  estimation-consistency and prediction-error outcomes under real modelling error. The repository
  question is structural (linearized, truth = model). The two can disagree.
- S1's full-Sun C_R failure is confounded with area-model error. A repository result where eclipses do
  *not* rotate K's direction would not contradict S1, because S1's mechanism may be the area model.
