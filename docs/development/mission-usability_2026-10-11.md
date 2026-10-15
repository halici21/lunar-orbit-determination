# Mission usability iteration — 2026-10-11

Base: a0885cb10ec75776ade6d3055677a6a9f89e05d9. Native Qt interactions,
layout and user documentation. No solver, scientific module, source artifact,
numerical tolerance or scientific skip policy was changed.

## Problems and fixes

- Station selection depended on small mouse targets. A native station dropdown
  now provides keyboard access. Dropdown, map marker and Gantt row label share
  selection; row-label clicks retain the current time.
- Marker fill was the only current-epoch indicator. The station caption now
  reports whether a measurement is recorded at the selected epoch, or lists
  the stations with recorded measurements when no station is selected.
  Missing geometry/history is explicitly distinct from absence of a measurement.
- Narrow windows retained a 960-pixel page width and needed sideways scrolling.
  Mission plots now stack below 920 logical pixels of page width. The mission
  page can shrink with the viewport; other scientific forms retain their
  existing width. Both plots stay full-size and reachable by vertical scrolling.
- Gantt names are elided when too long; hover exposes the full name and action.
  Loading a replacement run clears the prior station selection.

## Verification

Targeted regression:

    python -m pytest -p tests.desktop_preferences tests/test_mission_overview.py -q

26 passed. New behavior-based cases check keyboard selection, clickable Gantt
labels without a time seek, recorded observation gaps, all-station reset,
replacement with legacy missing geometry, and narrow/wide layout transitions
that preserve selection. The existing compact-window test includes the station
dropdown. These assertions protect UI/data interpretation, not solver convergence.

The existing 24-hour native solver output was reopened and rendered at
1480×1000, 910×465 and 760×420: 145 reference states, four stations and 12 measurement
windows. Playback moved from T+21,600 to T+25,800 seconds. All original source
hashes remained unchanged. Dropdown, archive, help and playback controls were
reachable in the small window; the plots switched to vertical orientation.
All four bounded-run arc solutions still report operational_success=False;
the observation data is synthetic, not spacecraft telemetry.

The selected light palette and every existing scientific tool remain available.
Wide and narrow screenshots were inspected after native Qt rendering.
The small viewport has no horizontal scroll range. See the
[sanitized render receipt](mission-usability-receipt_2026-10-11.json) and
[complete narrow page capture](../images/desktop-mission-narrow.png).

Public regression:

    python tools/run_public_checks.py --junitxml outputs/public-checks.xml

259 passed, zero failures/errors/skips, 26.392 seconds. External-data paths were
isolated. All 26 mission cases are part of this gate.

Full local regression:

    python -m pytest -p tests.desktop_preferences tests -q -ra

1,787 passed checks, 29 skipped, zero failures/errors, 247.042 seconds. JUnit
counts 1,816 checks: 1,670 named cases plus 146 subtest checks. The existing
scientific skip policy and tolerances were retained. The final two-pixel
minimum-window adjustment was then verified in the final public gate, including
the strengthened compact-window regression at 760×420.

Local JUnit SHA-256:

- full-checks.xml: 07d529ff9e62c51693bd2190173fb6e732df75f8ec3fc2a9089f70325dd67990
- public-checks-final.xml: 8b2808fdbaf4e68cf8387ab921bb743211e0639e88ae9567a7da337bbe5d3ee2

Graphify was refreshed and changed documentation links resolve.

## Next design checks

- Completed: expose recorded process/arc-success status without inferring
  convergence; see the [status iteration](mission-status_2026-10-11.md).
- Review first-run guidance and remaining specialist pages for short explanations,
  consistent units and keyboard access.
- Profile slow operations against large recorded runs when representative data
  becomes available; existing synthetic stress checks are not long-run acceptance.
