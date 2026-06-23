# Slow Test Policy

How slow tests and long scientific campaigns are handled.

## Slow tests include
long Monte Carlo campaigns; 28-day full campaigns; full thesis result
regeneration; external-tool cross-validation; large plotting batches; runtime
benchmarks; GUI tests requiring manual interaction.

## Rules
- Slow tests must **not** run in the default CI gate.
- Mark them as manual / nightly / benchmark-only / release-only / thesis-freeze.
- Document required data, kernels, environment, and expected runtime.
- Do not depend on private local paths.
- Save enough metadata (config, seed, commit) to reproduce the run.

## Critical rule
A slow test is useful only if its purpose, inputs, outputs, and acceptance
criteria are clear.
