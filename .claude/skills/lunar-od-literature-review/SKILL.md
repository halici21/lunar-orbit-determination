---
name: lunar-od-literature-review
description: >-
  Literature-backed review for lunar orbit determination topics: light-time and
  stellar-aberration corrections, two-way / counted Doppler and DSN radiometrics,
  SPICE/NAIF conventions, batch (BLS) and unscented (UKF) estimation theory,
  covariance / NIS consistency, and thesis references. Use when the user asks
  what the literature says, wants primary sources, wants to ground a modeling
  choice, or to compare this implementation against Moyer, Thornton & Border,
  Tapley-Schutz-Born, Montenbruck & Gill, Vallado, DSN 810-005, or SPICE docs.
---

# Lunar OD Literature Review

Produce concise, primary-source-grounded reviews for this lunar orbit
determination project, and connect each finding to a concrete implementation
implication under `python_port/`.

## Source priority
1. Books & peer-reviewed papers — Moyer (JPL DESCANSO Mon. 2); Thornton & Border
   (DESCANSO Mon. 1); Tapley/Schutz/Born *Statistical Orbit Determination*;
   Montenbruck & Gill *Satellite Orbits*; Vallado *Fundamentals of Astrodynamics*.
2. Agency / standards docs — NASA/JPL/ESA, DSN 810-005, NAIF/SPICE Required
   Reading (e.g. the aberration-corrections module).
3. Tool docs — SPICE Toolkit; GMAT / MONTE measurement-type definitions.

Rely on training-data knowledge of these only when confident; otherwise say so
and ask the user to confirm the exact page/section/equation before it is quoted.

## How to operate
- Restate the question precisely (e.g. "is one-iteration light time adequate for
  Earth–Moon?").
- Give the source-based answer first, then a clearly separate **Engineering
  inference** line for anything not directly from a source.
- Tie it to the code that implements or omits the concept, e.g.
  `lunar_od/radiometrics.py`, `lunar_od/measurements.py`, and the validation
  notes (`docs/spice_cross_validation.md`, `SPICE_CN_CNPLUS_VALIDATION.md`,
  `DOPPLER_RANGE_RATE_MODEL_REVIEW.md`).
- For the canonical reference list and what each source is authoritative for, see
  `references/key_sources.md`.

## Output
- Short prose answer (no filler), optionally followed by:
  - a comparison table — concept / literature source / status in this repo, and
  - an "Implementation implications" bullet list.
- Always keep **confirmed (source)** separate from **inferred (engineering)**.

## Do not
- Invent citations, page numbers, or equation numbers you are unsure of.
- Pad with generic academic background unrelated to the question.
- Claim operational / flight validity from this synthetic framework.
