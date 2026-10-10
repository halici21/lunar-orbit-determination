# Lunar OD — Project Instructions

## Mandatory Skill Discovery and Routing

Before analyzing, editing, or validating project code, evaluate all available
project skills and invoke every skill that is **materially relevant** to the
task. Use the smallest sufficient skill set, but invoke every materially
relevant specialist skill. There is no fixed maximum number of skills. Do not
invoke skills that are merely adjacent, speculative, or unrelated — and never
all skills indiscriminately. Token-efficiency concerns must not suppress a
materially relevant specialist skill.

Routing map (invoke every row that matches; scoped worktree variants take
precedence over unscoped ones when editing files inside that worktree):

| Task domain | Skill(s) |
|---|---|
| Measurement equations, light-time, Doppler, two-way range events (t1/t2u/t2d/t3), aberration, station states at event epochs, observables/residuals/Jacobians | `lunar-od-measurement-physics` |
| Frames, SPICE, sxform/pxform, MCI/J2000/ITRF93/SEZ, station velocity, dynamics, propagation, STM, gravity | `lunar-od-dynamics-spice` |
| BLS-LM, SRIF, SR-UKF, observability, covariance, convergence, estimator fairness | `lunar-od-estimator-engineering` |
| Validation plan for a single change, regression selection, tolerance updates, FD-comparison strategy | `lunar-od-validation-gates` |
| Recurring/scheduled test execution (CI cadences, nightly, release) | `lunar-od-continuous-verification` |
| Test design/review, tolerances, oracles, fixtures | `lunar-od-test-strategist` |
| Result CSV statistics and estimator comparison metrics | `lunar-od-statistical-diagnostics` (+ `lunar-od-result-validator` for the acceptance decision) |
| Read-only critical model review | `lunar-od-model-review` |
| Scenario JSON/config work | `lunar-od-scenario-config` (+ `lunar-od-campaign-design` when designing the experiment itself) |
| Locating code / tracing results | `lunar-od-repo-navigator` / `lunar-od-result-reproducer` |
| Desktop app implementation / UI design | `lunar-od-desktop-app` / `lunar-od-interface-designer` |
| Lint/types/CI/coverage/software hygiene | `lunar-od-software-quality-gates` |
| Figures, thesis text, literature, performance, public-repo readiness | matching `lunar-od-*` specialist skill |

Rules:

1. **Evaluate all, invoke the materially relevant.** Cross-domain tasks
   require every matching specialist (e.g., a two-way range Jacobian change
   spans measurement-physics + estimator-engineering + validation-gates);
   skills whose domain the task does not touch stay unloaded.
2. **Re-evaluate at phase transitions.** Skill selection is not one-shot:
   re-check the map when moving between analysis, implementation, testing,
   validation, documentation, and final review (e.g., validation-gates
   typically joins at the testing phase even if it was not needed during
   analysis).
3. **Report selections briefly.** Before substantive work, state in one line
   which skills were selected and why (e.g., "Skills:
   lunar-od-measurement-physics + lunar-od-validation-gates — Jacobian change
   with regression planning").
4. **Never claim unused skills.** Only report a skill as used if it was
   actually invoked via the Skill tool in this session.
5. **Specialist over general.** Do not substitute a generic skill (e.g.,
   repo-navigator) when a materially relevant specialist skill exists
   (e.g., measurement-physics for a Jacobian); general skills complement,
   never replace.

All project skills are auto-invocable; none are manual-only.

## Skill roots and synchronization

The **repository-tracked `.claude/skills/` is canonical**; the user-level
`~/.claude/skills/` copies are synchronized snapshots kept only so the skills
resolve outside repository working directories. Edit the repository copy
first, then mirror to the user level. Never let the copies silently diverge:
after any skill edit, run

```text
python .claude/skills/_shared/validate_skill_sync.py
```

which compares name, description, remaining frontmatter, and body hash for
every matching skill name across all roots (repo worktrees + user level) and
reports every mismatch.

## Graphify dependency graph (selective use)

`graphify-out/graph.json` is a deterministic graph (AST + Markdown, no LLM) of
`lunar_od/`, `examples/`, `tests/`, `desktop_app/` and the `docs/` reports. It is a **discovery and impact-analysis aid. The source
code is authoritative.** Never implement a change from graph edges alone: open the
real files and verify every relationship you act on. Skill routing (above) still comes
first; Graphify complements `lunar-od-repo-navigator`, it does not replace it.

**Use it** when the change surface is not already known: the impact of changing a
shared symbol, callers and tests of a function, refactor scope, what consumes a config
or interface, or orienting in an unfamiliar subsystem. **Skip it** when the file and
symbol are already known: single-file edits, doc or text changes, locations the user
named, small scoped fixes.

Commands, most reliable first (validated against source 2026-09-23):
- `graphify affected "<symbol>" --depth 2`: reverse dependencies, including tests.
  Reproduced the hand-built Phase 17-R1O-R blast radius of
  `chain_to_augmented_columns` exactly.
- `graphify explain "<symbol>"`: file:line, callers, callees, tests.
- `graphify path "<A>" "<B>" --undirected`: where two symbols are *wired together*
  (e.g. `build_range_arc` joins `propagate_state_with_k_sensitivity` to
  `_two_way_range_k_srp_column`), not how data flows between them.
- `graphify explain "<symbol>"` also lists the doc sections that name the symbol,
  with file:line (e.g. `_two_way_range_k_srp_column` -> sections of the R1O, R1O-D
  and R1O-OPT reports and the erratum). These are `[INFERRED]` name matches: open
  the section to confirm it really describes that code.
- `graphify god-nodes`: architectural hubs (`load_spice_kernels`,
  `RangeRatePhysicsConfig`, `propagate_augmented_state`, `PassGeometry`, ...).
- `graphify query "<question>"`: free-text search, noisy in code-only mode (matches
  docstrings; missed `_square_root_covariance_from_design`). Name symbols instead.

**`affected` is a lower bound.** Calls made through a module alias
(`import lunar_od.estimators as estimator_helpers; estimator_helpers.f()`) are not
resolved; 39 such aliases exist, mostly in `tests/`. Before declaring a symbol unused
or a change safe, confirm with a Grep for the bare symbol name.

Also invisible to the graph, so verify in source:
- **Data carried inside arrays.** The K_SRP chain (`srp_acceleration_kernel` ->
  `propagate_state_with_k_sensitivity` -> `nom48[:, 42:48]` ->
  `_two_way_range_k_srp_column`) and the column-major STM block `[6:42]` have no
  edge. The Phase 17-R1O storage-order defect lived exactly there.
- `lunar_od/__init__.py` re-exports everything, so any two exported functions look
  two hops apart through it. Discard paths routed through `__init__.py`.
- Physical and numerical meaning: units, frames, epochs, `order="F"` layouts,
  tolerances. Communities are unlabeled ("Community N"; no LLM backend configured).
- Doc links are name matches, so very short symbol names false-positive (a theme
  class `C` in `desktop_app/styles/theme.py` "appears" in two reports).

High-value queries here:
- `graphify affected "propagate_state_with_k_sensitivity"`: everything built on the
  K_SRP variational propagator.
- `graphify affected "RangeRatePhysicsConfig"` or `"PassGeometry"`: measurement-layer
  hubs; changing either implies broad regression.
- `graphify explain "_interp_state"`: the Hermite interpolator every event solver
  shares (the R1O-D FD-fidelity bug).
- `graphify affected "_square_root_covariance_from_design"`: estimator branches and
  tests on the R1COV covariance path.
- `graphify explain "scenario_config_from_mapping"`: config entry to its runners.
- `graphify affected "load_spice_kernels"`: SPICE-dependent test scope.

Non-trivial change protocol (internal discipline, do not narrate): identify the
subsystem; if the impact is not obvious, run `affected` / `explain` before broad
Grep; read the real implementation; implement; run focused tests; run broader
validation when a god node or shared module changed.

**Refresh.** The graph is per-worktree and gitignored. The installed git hook
rebuilds only in the primary checkout `python_port/`; in linked worktrees such as
this one it is a deliberate no-op. After architecture-relevant code changes (new,
moved or deleted modules; changed call structure) run `graphify update .` (no LLM,
~25 s, also refreshes doc links; add `--force` after deleting code or adding an
exclusion). Build and refresh with `update`, not `extract --code-only`, which drops
the doc links. Not needed for edits inside a
function body. If a shell cannot resolve `graphify`, it is at
`%USERPROFILE%\.local\bin\graphify.exe`. `/graphify` in chat runs the full skill
pipeline; prefer the CLI commands above.
