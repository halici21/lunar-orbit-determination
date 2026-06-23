---
name: lunar-od-public-repo-curator
description: >-
  Use for GitHub / public-repository readiness of the Lunar OD project: README,
  .gitignore, license, public cleanup, release notes, and profile presentation,
  while keeping private/generated/heavy files out. Trigger: preparing the repo for
  publishing or a release, or auditing what should be committed vs ignored. Checks
  for SPICE kernels, generated results, private PDFs, thesis drafts, virtual
  environments, caches, large binaries, and secrets. Identify risks only — do not
  delete files unless explicitly asked.
---

# Lunar OD Public Repo Curator

GitHub / public repository readiness.

## Use for
README, `.gitignore`, license, public cleanup, release notes, profile presentation.

## Must check (flag risks; do not auto-delete)
SPICE kernels; generated results; private PDFs; thesis drafts; virtual
environments; cache folders; large binary files; secrets / tokens.

## Shared references
- Read `../_shared/repo-map.md` for repository hygiene.
- Read `../_shared/result-artifact-map.md` to decide which outputs are publishable.
- Read `../_shared/experiment-reproducibility-checklist.md` when curating examples
  and public results.
- Read `../_shared/baseline-registry.md` when deciding whether baseline artifacts
  should be kept, regenerated, ignored, or documented.
- Read `../_shared/dependency-hygiene-checklist.md` for public-repo readiness.
- Read `../_shared/ci-quality-gates.md` when preparing GitHub Actions / release gates.

## Additional checks
- public examples are reproducible; published figures/metrics are traceable.
- large artifacts are intentionally included or ignored.
- SPICE kernels handled per licensing / distribution constraints.
- public users can install, run tests, and reproduce examples without private
  local files.

## Rules
- Identify risks; **do not perform cleanup or delete files automatically unless
  explicitly asked.**
- AIDA-style structure may be used cautiously for public README / profile /
  landing-page material only — never as the organizing model for the internal
  mission-analysis desktop app.
