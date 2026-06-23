---
name: lunar-od-literature-review
description: >-
  Literature-backed review for lunar orbit determination: tracking measurement
  models, SPICE/NAIF conventions, DSN-style radiometrics (range, two-way /
  counted Doppler), light-time and apparent geometry, batch (BLS/SRIF) and
  unscented (UKF) estimation theory, covariance / NIS-NEES consistency, and
  thesis references. Use when the user asks what the literature says, wants
  primary sources, wants to justify a measurement-realism upgrade, or to compare
  this implementation against Moyer, Thornton & Border, Tapley-Schutz-Born,
  Montenbruck & Gill, Vallado, DSN 810-005, or SPICE docs.
metadata:
  version: "2.0"
  adapted-from: K-Dense literature-review, paper-lookup, citation-management
---

# Lunar OD Literature Review

Produce concise, primary-source-grounded reviews and tie each finding to a
concrete implementation implication under `python_port/`.

## Workflow (phased)
1. **Define** the precise question (e.g. "is one-iteration light time adequate
   for Earth-Moon?") and its scope.
2. **Search / recall** primary sources in the priority order below.
3. **Screen** for relevance and authority; use the most authoritative source for
   each claim.
4. **Extract** what the source actually says — the equation, bound, or definition.
5. **Synthesize thematically** (by concept, not source-by-source).
6. **Verify** every citation you assert; if unsure of a page / equation / DOI,
   say so and ask the user to confirm before it enters the thesis.

## Source priority
1. Books & peer-reviewed papers — Moyer (JPL DESCANSO Mon. 2); Thornton & Border
   (DESCANSO Mon. 1); Tapley/Schutz/Born; Montenbruck & Gill; Vallado.
2. Agency / standards — NASA/JPL/ESA, DSN 810-005, NAIF/SPICE Required Reading.
3. Tool docs — SPICE Toolkit; GMAT / MONTE measurement-type definitions.

See `references/key_sources.md` for what each source is authoritative for.

## Output
- Short prose answer (no filler), optionally a concept / source / repo-status
  table and an "Implementation implications" list.
- Always keep **confirmed (source)** separate from **engineering inference**, on
  their own lines.
- Tie to code / validation docs: `lunar_od/radiometrics.py`,
  `lunar_od/measurements.py`, `docs/spice_cross_validation.md`,
  `SPICE_CN_CNPLUS_VALIDATION.md`, `DOPPLER_RANGE_RATE_MODEL_REVIEW.md`.

## Removed from the K-Dense base
PubMed / MeSH / bioRxiv strategies, Cochrane / AMSTAR / PRISMA tooling, clinical
journal prioritization, and the mandatory AI-schematic / PRISMA-diagram step.

## Do not
- Invent citations, page numbers, or equations you are unsure of, or pad with
  generic academic background.
- Claim operational / flight validity from this synthetic framework.

## Scope Boundary
Primary for literature-backed answers and primary sources.

## Do Not Use For
- code changes (domain skills)
- interpreting this project's results (lunar-od-result-validator / statistical-diagnostics)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
