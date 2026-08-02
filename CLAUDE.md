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
