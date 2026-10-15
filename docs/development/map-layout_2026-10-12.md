# Mission map proportions and labels — 2026-10-12

Base: a44f44e0bf217c139e6f22b4c748ca50a5ee4466. This iteration addresses the
mission map's shape and station-name usability. Scientific code and saved
station coordinates are unchanged.

## Observed problems and changes

The map stretched its longitude and latitude independently to fill any widget.
A 1200×300 widget therefore drew a much wider world than a 320×230 widget.
Station labels also used a fixed right/above offset, clipping Canberra at the
right edge and allowing nearby names to collide.

- The 360° by 180° map now fits a centered **2:1** rectangle. Land, graticule,
  markers and click geometry use that same rectangle. Equal angular spans have
  equal pixel scales; this equirectangular projection does not preserve distances
  or areas over the globe.
- Labels try four positions within the map bounds, avoid markers and other
  labels, and have connecting lines. The selected station gets priority and its
  full name; optional names may hide in crowded regions. Every marker remains.
- Visible names are clickable and expose the station's full name/coordinates
  in a tooltip. The station dropdown remains available for every station.
- Native review found Madrid and Cebreros less than a pointer pixel apart at
  world scale. An explicit selection is now retained when rounding cannot
  distinguish their centers. This is a pixel hit-test rule, not a change to
  scientific coordinates or numerical tolerances.

![Small Modern map with Canberra selected](images/desktop-map-minimum.png)

![Orchid map with ITU selected at display scale 2](images/desktop-map-orchid.png)

## Verification

    python -m pytest -p tests.desktop_preferences tests/test_mission_overview.py tests/test_tracking_render_semantics.py tests/test_large_workspace.py -q

The final targeted set passed **44 checks**. Four new cases cover equal angular
pixel scales at 320×230, 600×260 and 1200×300; bounds; station identity after
resizing; label priority/containment; exact label clicks; and Madrid/Cebreros
selection at a shared pointer pixel. Existing linked map/Gantt/orbit tests still
check exact saved epochs and missing observation gaps.

The final isolated public gate passed **271 cases**, with zero failures, errors
or skips, in 28.959 seconds. The full local regression passed **1,799 checks**,
with 29 existing skips and zero failures/errors, in 249.447 seconds. JUnit records
1,828 checks: 1,682 named cases and 146 subtest checks. Optional slow/data-dependent
skips and scientific tolerances were retained.

    python tools/run_public_checks.py --junitxml outputs/public-checks.xml
    python -m pytest -p tests.desktop_preferences tests -q -ra --junitxml outputs/full.xml

Local JUnit receipt SHA-256 hashes:

| Receipt | SHA-256 |
| --- | --- |
| Public | `B78FD0CBE89FC06C9720A43154BD9364DFE208E071B259EB7612901B04E99097` |
| Full | `2FB80663F1317DFDC938F2DB65268514C037ABD265DD61B0449F314105B5F787` |

Native Qt review covered initial/loaded 1480×1000 views and loaded 910×465 /
760×420 windows, retaining reachable controls and zero horizontal scrolling.
Separate maps were inspected at 320×230, 500×260 and 1200×300, with the full
15-station catalogue: Modern at display scale 1 and Orchid at display scale 2.
Selected names fit; selections were correct; changing active markers reused
the cached land drawing.

The existing verified 24-hour solver output retained 145 saved truth states,
four stations and 12 measurement-sample windows. Its observations are synthetic;
all four recorded arc-success flags remain false. Source hashes remained unchanged
after selection/playback/rendering. Eight legacy runs reopened and all 48 verified
source files retained their hashes. See the [receipt](map-layout-receipt_2026-10-12.json).

No new scientific campaign was run. These checks do not establish estimator
convergence or separate stations that occupy the same world-map pixel.

## Remaining design work

- Review specialist pages for clear units, short help and keyboard access.
- Evaluate whether dense station clusters need an explicit regional zoom;
  precise station selection already remains available through the dropdown.
- Profile workflows with representative large saved runs when available.
