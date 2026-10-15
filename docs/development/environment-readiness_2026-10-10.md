# Desktop environment readiness — 2026-10-10

## Change and scope

Base: ff9def8b3c4e16a25bbb881aa133e567458fde48. Added startup and Scenario Builder
preflight, a Settings report, persistent kernel selection, and a fresh inherited
environment for each reused QProcess launch. Scientific modules, examples,
fixtures, numeric tolerances and pinned dependency requirements were unchanged.

The service reads the canonical spice_loader.py definitions and invokes its
unchanged loader in a separate Python process. All ten files, including the
DE440 pair, must be readable and nonempty. Actual SPICE loading and sample epoch
queries must pass; file presence alone is insufficient. A saved explicit missing
folder cannot silently fall back. The GUI's SPICE pool is untouched.

Packages are actually imported and compared to the validated profile. Version
mismatch is a warning, not scientific qualification. Scenario probes use the
fixture UTC and the maximum padded endpoint of the state/ephemeris grids.
Earth/Sun and ITRF93/DE421/DE440 frames are queried at start, midpoint and end.
Output permission checks are read-only; actual writes may still fail later.

A failed preflight creates no job folder and opens the actionable report. Source
archives remain usable without kernels. High-degree gravity/campaign assets,
continuous kernel coverage, convergence and long real-run acceptance are outside
this feature. Historical Run Monitor scripts retain their existing validation.

## Verification

Validated Windows x64 / Python 3.13.2 environment from the pinned profile.

| Gate | Result | Time |
| --- | --- | --- |
| Targeted readiness / loader / desktop workflow | 35 passed, zero skips/errors | 6.132 s |
| Full local regression | 1,759 passed, 29 skipped, zero failures/errors | 288.338 s |
| Final public gate, external inputs isolated | 233 passed, zero skips/failures/errors | 25.632 s |

The full regression collected 12 initial readiness cases. Two final presentation
and inconsistent-report cases were added during review; both are included in the
233-case final public gate (14 readiness cases total). The 29 full-suite skips
remain explicit long-regression or optional external gravity/store conditions.
No tolerances or skip policy were relaxed.

Meaningful contracts exercised: missing explicit path despite a valid fallback;
empty file and directory masquerading as a kernel; missing dependency with retained
inventory; corrupt nonempty kernels with real SPICE rejection; padded endpoints and
pool cleanup; saved selection and inherited restoration; reused child environment;
responsive Qt event loop during a real probe; deferred launch with no failed-job
creation; timeout/shutdown; inconsistent ready status; failures visible before
healthy packages.

Additional desktop checks:

- Real local kernels: all ten files loaded and sampled 24-hour queries passed.
- Actual QProcess launched pythonw.exe (the desktop shortcut interpreter) and
  received the machine-readable ready report.
- Healthy and missing-folder screens were rendered at 1200×900 and 1000×720,
  visually reviewed, and corrected to show actionable errors first.
- Screenshots/settings/reports were kept in the local task outputs, outside Git.
  Synthetic corrupt kernel bytes were verified unchanged by the probe.
- Graphify dependency graph refreshed; git diff --check passed.

Local JUnit report SHA-256 receipts:

- environment-readiness-targeted.xml: 04c8580ef48ce4e1e389cfdf9053c172807ea8989445c556fa991b427c1f40fc
- environment-readiness-full.xml: b4a551a4fc8f4e0a9823b47fde967ea34c2e070bdfd28043dc76ba8e73bcafe4
- environment-readiness-public.xml: 71da5752edcf0511e5817abe18033875768ef27a53601e5e10d841be155aaae9

## Use

Reopen the desktop app. Click **Ortam** in the bottom status bar or navigate to
**Ayarlar → Hesaplama ortamı**. Select a kernel directory or restore automatic
lookup. The selection is saved immediately and applies to subsequent child jobs.
See [Windows quickstart](windows-quickstart.md#ortam-kontrolü).

The previously published v0.1.0-alpha.1 ZIP predates this change. Use the current
main checkout for this feature; no existing release asset was replaced.
