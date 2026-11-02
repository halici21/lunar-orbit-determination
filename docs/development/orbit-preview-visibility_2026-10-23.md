# Orbit preview: skip camera frames outside the viewport

The analytical designer stays usable while editing in a small window. Its
camera animation resumes when any part of the preview returns into view.

## Demonstrated issue and change

At 910 × 465 and 760 × 420, the form places the entire preview below the
page viewport. The tab still counts as visible, so its 66 ms timer repeatedly
rotated and redrew the clipped 3-D canvas. Before the fix, four paired
Modern/Orchid samples performed 27–28 unnecessary draws each in about 2.2 s.
The draw calls occupied 85–91% of the sampled event-loop time.

`_animate_preview` now returns when the canvas's Qt visible region is empty.
In all four clipped samples there are zero subsequent camera draws and no
phase advance. The 20 ms responsiveness probe's median interval changes from
67–71 ms to about 16 ms. These short local samples describe this environment,
not a general application speedup or an OS CPU-usage measurement.

Both partly visible narrow previews and the restored wide view still rotate.
The camera pauses at its current angle while clipped; it is decorative, not
the recorded-run time cursor. The timer interval and existing hidden-tab
lifecycle remain unchanged. Input edits can still update the analytical
preview, and generating a surface track remains an explicit action.

Production scope is three added lines in one method. Removing them restores
the entire controller source (normalized line endings) against base
`02fdc366d69fd327f58e03e0147d35347e168b4b`.
All 72 other source segments, including the SPICE worker, Kepler
calculations, guards, numerical settings, plot geometry and exports, match.

## Verification

Local result: **pass with caveats**. The completed rerun supplies the counts
below; an earlier session-interrupted attempt is excluded.

- Two new regression cases fail on the baseline, then pass with the guard.
  They exercise real Qt clipping at both narrow sizes, 30 skipped camera
  ticks, partial reveal/resume, scroll-up/pause, preserved inputs and objects.
  Frames are driven explicitly; no machine-speed timing threshold is used.
- 67 targeted UI tests and 389 isolated public checks pass.
- Full regression: 2026 passed checks and 29 existing skips,
  1909 named cases; no failures or errors. JUnit includes unittest subchecks.
- Twelve before/after native animation states cover both themes and three
  window sizes, with deferred Qt deletions delivered before capture.
  Wide and small captures were visually reviewed; no restyle or pixel-perfect
  identity claim is made. Eighteen additional layout/state captures preserve
  labels, keyboard reach, rejected-orbit state and pending-SPICE behavior.
- Stored circular/eccentric analytical CSV references and the prior 361-row
  SPICE example replay retain their hashes. No new SPICE propagation is run.
- The existing solver-output run retains 145 truth states, four stations,
  12 **measurement-sample** windows, Madrid selection and linked playback
  21600 → 25800 s. Observations are synthetic; operational success remains
  0/4, and convergence is unknown. Eight legacy runs preserve all 48 source
  hashes and their missing-data guidance. This protects UI/source contracts,
  not scientific accuracy.
- Dependency, repository and whitespace gates pass. No tolerance, physical
  model, dependency, workflow, portable asset or release tag changes.

```powershell
python -m pytest -p tests.desktop_preferences tests/test_ground_track_lifecycle.py -q
python tools/run_public_checks.py --junitxml=outputs/public.xml
python -m pytest -p tests.desktop_preferences tests -q -ra --junitxml=outputs/full.xml
python -m pip check
python tools/check_repository.py
git diff --check
```

See the [sanitized receipt](orbit-preview-visibility_2026-10-23-receipt.json)
for local test counts, hashes, scope and paired measurements. Required GitHub checks and Release
qualification must pass before merge; exact merged-main checks and downloaded
JUnit have a separate local receipt outside Git.

## Next work and limits

- Visible full 3-D redraw medians remain roughly 70–126 ms in these samples.
  Profile and compare a rendering-only improvement before changing meshes,
  plot layout, frame cadence or interaction. Do not claim it is fixed here.
- The timer still polls every 66 ms while clipped; the costly camera draw is
  skipped. Fully hidden tabs/windows retain their existing timer lifecycle.
- Keep deferred-delete delivery in manual QA loops. Offscreen Qt/Fusion does
  not qualify every Windows/high-DPI interaction. The existing modal-painting
  limitation remains; original dialogs are unchanged.
- Revisit archive performance when a measurable bottleneck is available.
  No user app restart or new scientific campaign in this change.
