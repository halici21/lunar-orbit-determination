# Skill Routing Policy

Prevent overuse of skills and excessive shared-reference loading. Default to the
**smallest sufficient** skill set.

## Routing rules
- Use the smallest sufficient skill set; use **one primary skill** by default.
- Add secondary skills only when the task clearly requires them.
- Do not load broad governance files for small explanation or lookup tasks.
- Do not use `lunar-od-result-validator` unless interpreting numerical results,
  plots, tables, or claims.
- Do not use `lunar-od-validation-gates` unless accepting a code/science change
  or selecting required validation.
- Do not use `lunar-od-test-strategist` unless designing, reviewing, or repairing
  tests.
- Do not use `lunar-od-continuous-verification` unless scheduling recurring tests
  or CI / nightly / release workflows.
- Do not use `lunar-od-software-quality-gates` unless checking lint, format, type
  checking, dependency hygiene, CI, maintainability, or packaging.

## Small Task Mode
For narrow questions, explanation requests, or file lookups:
- inspect the directly relevant file or section,
- avoid broad validation workflows,
- answer concisely,
- mention optional validation only if relevant.

## Escalation Mode
Use full governance (multiple skills + their shared checklists) only when:
- code changes are requested,
- scientific behavior may change,
- numerical outputs may change,
- tests / baselines / fixtures are being designed,
- result claims are being accepted or published,
- public release or thesis-freeze work is being prepared.

## Shared-reference loading
Read at most **1-2** shared files by default — only those the current task needs.
Load additional shared files only when the task requires them.
