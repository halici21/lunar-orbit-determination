# Mission overview — 2026-10-11

Base: c7e18a3d9c736d9113b574d19e740469ee77b7fc. Native desktop/UI and
artifact display changes. Scientific modules, measurement equations, solvers,
fixtures, numerical tolerances and scientific skip policy are unchanged.

![Recorded orbit, Earth stations and measurement Gantt](../images/desktop-mission.png)

## What changed

- A new **Görev görünümü** page places the recorded orbit and Earth station map
  above a measurement Gantt. All three views share the same stored sample time.
- Station clicks expose names/coordinates and highlight their Gantt row. Gantt
  clicks and the slider select recorded epochs. Playback has pause, speed and
  reset controls; it stops when the page is hidden or the window closes.
- Height, stored-frame inclination, semi-major axis, eccentricity and period
  appear with expandable definitions. Unknown GM/state history stays unknown.
- Station catalogue, visibility planning and lunar surface-track tools are
  primary navigation items. Visible page descriptions explain their purpose.
- The existing station map and orbit designer now use the selected light palette
  for land and Moon surfaces. Modern/Orchid theme selection is retained.
- New native runs record station identities and actual measurement epochs in a
  hashed geometry sidecar. Legacy results show their missing-data scope clearly.
- Cached map/Gantt/orbit drawing avoids rebuilding the background on each frame.
  Geometry is not duplicated into comparison cells or exported report metadata;
  its file/hash remains a verified source and is reread when reopening a report.
- A parent-owned startup timer is cancelled on close. A deterministic paired
  test reproduced an access violation from a queued startup callback after closed
  windows; cancelling the timer fixed it. The regression is included in CI.

First-run and archived-result instructions are in the
[Turkish Windows guide](../windows-quickstart.md#yörünge-harita-ve-gantt).
The [artifact contract](../../desktop_app/RESULT_WORKFLOW.md#mission-overview)
describes units, grouping, hashes, size limits and legacy behavior.

## Display scope

The Gantt shows **recorded measurement samples**, grouped only within one arc
and one configured sampling cadence. It is not complete station visibility,
uninterrupted radio contact or estimator success. Singleton epochs remain
zero-duration events. No contact is inferred from arc start/end alone.

The separate **Temas planı** computes independent geometric visibility. It does
not retrospectively populate an old run with supposedly recorded station contacts.
Catalogue positions are labelled as catalogue data when run geometry is missing.
Custom station networks remain catalogue tools; the native runner supports its
canonical single/multi networks.

Playback uses stored truth states, not new propagation or telemetry. It draws a
connected sample path and selects stored points without interpolation. Display
inclination is relative to the stored frame's XY plane; J2000 inclination is not
lunar-equatorial inclination. Two-body descriptors use the run's recorded lunar
GM. The fixed reference Moon radius for altitude is 1737.4 km.

## Verification

Validated Windows x64 / Python 3.13.2 environment with the pinned dependencies.

| Gate | Result | Time |
| --- | --- | --- |
| Full local regression | 1,783 passed checks, 29 skipped, zero failures/errors | 303.288 s |
| Final public gate, external inputs isolated | 256 passed, zero skips/failures/errors | 27.640 s |

The full JUnit suite reports 1,812 checks: 1,666 named test cases plus 146
subtest checks. The full run includes 22 initial mission cases. A final
comparison report-compaction/reopen regression was then added and passed in the
final public gate, which includes all 23 mission cases. The numeric comparison
metrics and pairing rules were unchanged. Existing slow/optional scientific
skips remain explicit; no tolerances or skips were relaxed.

New contracts cover the three actual observable station-ID columns; local station
identity; observation gaps and arc boundaries; invalid coordinates/windows;
geometry hash tampering and folder escape; a circular polar analytical orbit;
linked map/Gantt/orbit selection; elapsed-time playback/end/hide behavior;
20,000-window cached drawing; preserved original source bytes; legacy unknown
states; lazy startup; visible help; safe session reopening; compact-window reach;
cancelled startup callbacks; and compact comparison export with source revalidation.

### Actual runner and native rendering

A newly generated 24-hour SPICE scenario used multi-station geometric range-rate
measurements, seed 42, hot start, 600-second state samples, four planned arcs,
and BLS-LM limited to three iterations. Its job, effective config, force manifest,
result manifest and geometry were preserved in local outputs, outside Git.

- Four stations: Goldstone, Madrid, Canberra, ITU Ayazaga.
- 145 reference state samples; four arc solutions; 12 recorded measurement windows.
- The native playback advanced from T+21,600 s to T+25,800 s in the render check.
- Source hashes were unchanged after load, station selection, playback and rendering.
- Screens were rendered and inspected at 1480×1000 and 910×465. Playback,
  archive and help controls remain reachable by scrolling at the smaller size.
- Eight legacy runs reopened; all 134 files in the historical result tree retained
  their original hashes. A native/legacy comparison exported and reopened with
  identical pair and descriptive metric values.

All four short-run arc solutions retain **operational_success=False**. This is
an actual solver workflow demonstration with synthetic observations, not a
convergence qualification or real mission observation claim. See the
[render/source receipt](mission-ui-receipt_2026-10-11.json).

Local JUnit SHA-256 receipts:

- mission-full-checks.xml: 0a0fa44e1664f5d28744ad0fa35f4189ca26a506f2f46a5c91a31580bb547d89
- mission-public-checks-final.xml: 6da6ad610560231ab1671ab5ddc301ab18493b25300edee4430f8d6553be2272

Graphify was refreshed and git diff --check passed. This feature is on main;
the immutable v0.1.0-alpha.1 source ZIP predates it.
