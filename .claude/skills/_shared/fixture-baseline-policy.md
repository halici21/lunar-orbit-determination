# Fixture & Baseline Policy

How fixtures and scientific baselines are created, stored, updated, and reviewed.

## Fixture types
small deterministic arrays; MATLAB comparison fixtures; SPICE sample outputs;
stored trajectory snippets; stored measurement predictions; stored estimator
summary metrics; CSV/JSON schema fixtures; protected baseline outputs.

## Rules
- Fixtures must be small enough for repository use unless intentionally external.
- Fixtures must include units and source information.
- Baseline updates require an explanation.
- Baseline updates must distinguish intentional model changes from accidental drift.
- Do not overwrite baselines silently.
- Do not invent expected values; if unknown, mark as `to be registered`
  (see `baseline-registry.md`).

## Critical rule
Changing a baseline is a scientific decision, not just a file update.
