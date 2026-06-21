---
name: lunar-od-thesis-writing
description: >-
  Write thesis, report, and presentation text for the lunar OD project: problem
  definition, methodology, measurement model, BLS-LM and SR-UKF formulation,
  experiment design, results interpretation, limitations, and conclusion — as
  thesis paragraphs and as slide speaker notes. Use when the user asks to draft
  or revise academic text, figure captions, or talking points for this project.
---

# Lunar OD Thesis Writing

Produce clear, presentable academic English grounded in this project's results.

## Voice
- Plain, precise, easy to read aloud; short-to-medium sentences.
- Define each term once; avoid buzzwords ("cutting-edge", "novel",
  "revolutionary").
- Prefer the active voice where natural; quantify claims; hedge honestly
  ("under the simulated conditions", "for this arc geometry").

## Section guidance
- **Problem & motivation** — the lunar OD need; the batch vs sequential
  trade-off.
- **Methodology** — shared synthetic truth, visibility and arc construction,
  measurement realization; emphasize the controlled, matched-condition
  comparison.
- **Measurement model** — range / az / el, range-rate, two-way Doppler; note the
  optional light-time / stellar aberration and that the model is simplified.
- **Estimators** — BLS-LM (completed-arc batch, LM damping) vs SR-UKF
  (sequential, square-root covariance, NIS monitoring); describe what each is
  *good for*.
- **Experiments** — state the controlled variables and the seeds.
- **Results** — report median / p95 / max, success fraction, runtime, NIS/NEES
  with careful interpretation; separate typical from tail behavior.
- **Limitations** — synthetic self-consistency; simplified Doppler / media /
  relativity; frame approximations — state them plainly.
- **Conclusion** — what the comparison shows, scoped to the assumptions.

## Hard rules
- **Never invent numbers** — take results from CSVs, figures, or the validation
  docs, or ask the user to point to them.
- Keep the line between **controlled synthetic results** and **operational flight
  performance** explicit; do not imply flight readiness.
- Match every figure/table claim to the actual generated artifact.

## Output
- Thesis paragraphs, OR
- Slide speaker notes (one to three sentences per slide, spoken register), as
  requested.
