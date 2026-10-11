# Mission status iteration — 2026-10-11

Base: b568667ccf1968abeaa51a3e69a973fccb3931bc. A focused desktop display
iteration following the saved usability backlog. No scientific module,
measurement equation, solver, tolerance, observation file or acceptance rule
was changed.

## Observed problem

Mission view exposed truth states, station measurements and a working animation,
but the recorded process and arc-success status was only available on another
page. The verified 24-hour workflow example has four unsuccessful arc records;
its truth orbit alone does not reveal this.

## Change

- The loaded run's process status and recorded arc-success counts are now
  visible directly below its name. Unknown success fields stay unknown and
  missing completion records retain their existing explanation.
- A separate line explicitly says that the source has no convergence-validation
  field. Neither truth states nor process completion infer a scientific verdict.
- **Koşu ayrıntıları** opens the same result's **Koşu** inspector on the results
  page. It requires no new file load and pauses mission playback. The action is
  disabled before a result is loaded.
- A failed replacement retains the prior result and its explicitly labelled
  loaded-run status, alongside the load error.
- The existing headline formatter was extracted from result_summary so the
  console and mission use the same semantics. Mission does not calculate error
  medians, reread sources or rebuild the status on each playback frame.

The layout remains a flat caption and secondary action. Modern/Orchid themes,
station selection, the Gantt, stored-time playback and all tools remain available.

## Verification

Targeted command:

    python -m pytest -p tests.desktop_preferences tests/test_mission_overview.py tests/test_result_summary.py -q

34 passed. New cases cover all-false, partially unknown and wholly unknown
arc-success fields despite completed process/truth states; stable status after
time selection; disabled first-run action; shared-result inspector navigation;
playback pause on leaving the page; retained status after failed replacement;
and status formatting from sequential-run arc flags without diagnostic fields.
These protect display/data interpretation, not solver convergence.

Native Qt rendering reused the verified 24-hour synthetic-observation solver
output: 145 reference states, four stations, 12 measurement windows, four
operational_success=False arc solutions. The screen says **İşlem tamamlandı ·
0/4 yay kaydı başarılı** and explicitly leaves convergence unknown. Source
hashes remain unchanged after load, selection, playback and rendering. Wide
1480×1000, narrow 910×465 and minimum 760×420 controls are checked; the mission
plot layout and no-horizontal-scroll behavior remain intact.

Eight existing single-arc runs also reopened. Their recorded success flags
remain distinct from completion, including successful, unsuccessful and
sequential-run records. All 48 verified source files retained their hashes;
legacy station contacts were not invented.

Public command:

    python tools/run_public_checks.py --junitxml outputs/public-checks.xml

264 passed, zero failures/errors/skips, 28.218 seconds, with external-data paths
isolated. All 30 mission cases and four summary cases are included.

The [sanitized render/archive receipt](mission-status-receipt_2026-10-11.json)
records the displayed status and source-preservation checks.

Full local command:

    python -m pytest -p tests.desktop_preferences tests -q -ra

1,792 passed checks, 29 skipped, zero failures/errors, 237.103 seconds. JUnit
counts 1,821 checks: 1,675 named cases plus 146 subtest checks. Existing optional
scientific tests remain skipped under their original policy; no numerical
tolerances were relaxed. Passing display tests does not qualify convergence.

Local JUnit SHA-256:

- full.xml: dec300ffbdc4d17d021e4a6edd764c2e655310641c897edc43b39ca60915ad0c
- public.xml: 94e6722da9578002f37d007317e33f40486c4f13fcff9f3eaeb1fcd3faed75bb

Graphify was refreshed, changed documentation links resolve, and git diff
--check passes.

## Remaining design work

- Review first-run empty screens for instructions that clearly distinguish an
  unopened run from a loaded result missing optional data.
- Review remaining specialist pages for readable units and keyboard access.
- Profile slow operations against representative large saved runs when available;
  synthetic stress checks do not establish long-run acceptance.
