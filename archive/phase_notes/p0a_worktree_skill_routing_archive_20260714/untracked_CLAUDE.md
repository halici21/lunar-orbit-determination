# CLAUDE.md

## Mandatory Skill Discovery and Routing

Skills in `.claude/skills/` are project capabilities, not optional
suggestions. Before substantive analysis, editing, testing, or validation:

1. Check every available skill's description against the requested outcome,
   the files/modules involved, the domain, and the current phase of work.
2. Invoke **all** skills that are materially relevant — not just the first
   match, and not every skill indiscriminately. A specialist skill must not
   be skipped just because a broader one was already invoked.
3. Re-evaluate relevance when the task moves between phases (understanding ->
   implementation -> testing -> validation -> documentation), when new
   files/subsystems become relevant, or when scope changes.
4. Before substantive work, state: `Skill routing: invoking [names] because
   [brief scope mapping].` When new skills become relevant later: `Skill
   routing update: invoking [names] because [newly discovered need].`
5. A skill counts as used only when actually invoked through the Skill tool —
   not because its description was read, or another skill covered similar
   ground, or its name appeared in the response.

This is "invoke every *relevant* skill," not "invoke every skill." Several of
the skills below apply only within a specific trigger; for a narrow lookup
with no matching trigger, naming one skill (or none) is correct routing, not
under-routing. Full detail and the per-skill triggers live in
`.claude/skills/_shared/skill-routing-policy.md` — read it when routing is
ambiguous or when a task spans more than one domain.

## Skills in this project

See `.claude/skills/*/SKILL.md` for the full list (measurement physics,
dynamics/SPICE/frames, estimators, campaign design, scenario config, result
validation, statistical diagnostics, test strategy, continuous verification,
software quality, performance, figures, desktop app/UI, thesis writing,
literature review, repo navigation, result reproduction, public-repo
curation). Cross-domain lunar-OD tasks routinely need more than one — e.g. a
measurement-Jacobian change touching frame transforms needs both
`lunar-od-measurement-physics` and `lunar-od-dynamics-spice`.
