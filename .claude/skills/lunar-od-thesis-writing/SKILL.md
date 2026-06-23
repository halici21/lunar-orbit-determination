---
name: lunar-od-thesis-writing
description: >-
  Write thesis, report, and presentation text for the lunar OD project: problem
  definition, methodology, measurement model, BLS-LM / SRIF / SR-UKF formulation,
  experiment setup, results interpretation, limitations, and conclusion — as
  thesis paragraphs and as slide speaker notes. Use when the user asks to draft
  or revise academic text, figure captions, or talking points for this project.
metadata:
  version: "2.0"
  adapted-from: K-Dense scientific-writing, scientific-slides
---

# Lunar OD Thesis Writing

Produce clear, presentable academic English grounded in this project's results.

## Two-stage process
1. Draft a **bullet outline** of the section's key points (each claim plus the
   result / figure it rests on).
2. Expand into **flowing prose**; one idea per sentence, easy to read aloud.

## Voice
Plain, precise, short-to-medium sentences; define each term once; no buzzwords
("novel", "cutting-edge"); active voice where natural; quantify and hedge
honestly ("under the simulated conditions", "for this arc geometry").

## Structure & sections
Thesis / IMRAD-style: **problem & motivation -> methodology -> measurement model
-> estimators -> experiment design -> results -> limitations -> conclusion**.
- Methodology: the controlled, matched-condition comparison (shared truth /
  visibility / seed).
- Estimators: BLS-LM (completed-arc batch, LM damping), SRIF, SR-UKF (sequential,
  square-root covariance, NIS monitoring) — say what each is good for.
- Results: median / p95 / max, success fraction, runtime, NIS/NEES with careful
  interpretation; separate typical from tail behavior.
- Citations: IEEE or Chicago, applied consistently (hand off to
  lunar-od-literature-review for the sources).
- Revision pass: check logical flow, consistent terminology, and that every
  figure/table claim matches its artifact.

## Slides & speaker notes
Visuals primary, text supporting; narrative arc (hook -> context -> problem ->
approach -> results -> implications -> closure), with results ~40-50% of the
talk; high contrast, large fonts (titles 36-44 pt, body 24-28 pt); one to three
spoken-register sentences of speaker notes per slide.

## Hard rules
- **Never invent numbers** — take results from CSVs / figures / validation docs,
  or ask the user to point to them; match every claim to its artifact.
- Keep **controlled synthetic results** vs **operational flight performance**
  explicit; do not imply flight readiness.

## Removed from the K-Dense base
Mandatory graphical abstract, CONSORT / STROBE / PRISMA / ARRIVE reporting
guidelines, IUPAC / HGVS / clinical terminology, and bio / ecology figure
conventions.

## Scope Boundary
Primary for thesis/report/presentation text and speaker notes.

## Do Not Use For
- inventing numbers (never)
- producing figures (lunar-od-figures)

## Context control
Default to this one skill; add another only if the task clearly needs it. Load at
most 1-2 shared/reference files by default (only those the task needs); read the
rest only when required. For narrow questions or lookups, inspect the relevant
file and answer concisely. See `../_shared/skill-routing-policy.md`.
