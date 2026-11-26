# Test levels and CI cadence

V&V programme work packages WP7.1 and WP7.2 (requirements REQ-SW-002 and REQ-SW-003,
finding TST-01). Every test carries exactly one **verification level**, chosen by the
oracle it relies on, plus **resource markers** for what it needs. CI selects tests by
marker, so each cadence runs a stated set rather than a hand-kept file list.

## Levels

| Level | Marker | Oracle | Tests (2026-11-26) |
|---|---|---|---|
| L0 | `analytic` | closed form, convex reference problem, invariant, same-model recovery | 777 |
| L1 | `derivative` | finite-difference or Richardson check of a Jacobian, STM or partial | 376 |
| L2 | `external_oracle` | independent tool or record: SPICE, ERFA/SOFA, Orekit, MATLAB fixtures, published standards | 298 |
| L3 | `model_mismatch` | truth and estimator deliberately use different models | 11 |
| L4 | `statistical` | Monte Carlo, NIS/NEES, chi-square, whiteness | 56 |
| L5 | `real_data` | real tracking or real gravity-field data | 7 |
| L6 | `golden` | frozen record of the project's own outputs (snapshots, approved bytes) | 298 |
| L7 | `contract_negative` | interface, schema, provenance, UI-workflow and fail-closed contracts | 1012 |

Resource markers: `spice` (916 tests), `campaign` (313), `qt` (322), `numba` (181) and
`slow` (21). The derived marker `requires_kernels` (431) marks every `spice` test at L2
or above.

The counts show where the suite is thin: L3 and L5 are small, as the independent
assessment reported (TST-02). The levels make that gap visible; they do not close it.

## Where the levels come from

The levels are kept in one table, [`tests/levels.py`](../../tests/levels.py), and
applied by [`tests/conftest.py`](../../tests/conftest.py) at collection. Test files
carry no level markers, so parallel work on test files does not collide. A test is
classified by, in order:

1. a level marker written on the test itself (for a single exception);
2. the longest `NODE_LEVELS` entry that prefixes its node id;
3. the `FILE_LEVELS` row of its file.

A file with no row is **unclassified**. An ordinary run lists it under
"unclassified tests (WP7.1)" with the row to add. Every CI tier runs with
`--strict-levels`, which stops before running anything and prints that row. The
check in [`tests/test_levels.py`](../../tests/test_levels.py) fails until the row is
added. To fix a new file, add one row to `FILE_LEVELS`:

```python
    "test_new_feature.py": (_X, "spice"),   # level, resource markers
```

All markers are registered in `pyproject.toml` and pytest runs with `--strict-markers`.
The level is a file-wide default: a file of mostly finite-difference checks is L1 even
if it also holds a few analytic tests. Refine single tests with `NODE_LEVELS` where
the difference matters for selection.

## Missing kernels fail (REQ-SW-003)

- A `spice` test at L2 or above (`requires_kernels`) **fails** at setup when the
  DE440 or DE421 kernel set does not resolve. It never skips.
- `--require-kernels` extends this to every `spice` test. The kernel-backed tiers
  pass it, so no kernel test in those tiers can skip either.
- A run without kernels must leave these tests out explicitly with `-m`; the
  deselection then shows in the pytest summary. The commit tier uses `not spice`.

`campaign` and gravity-field data are a separate case: hosted runners never have
them, so those tests still skip there. They run in the local kernel-backed full
regression.

## Cadence

[`tools/run_tier.py`](../../tools/run_tier.py) defines each tier; run
`python tools/run_tier.py --list` to print them.

| Tier | When | Selection (`-m`) | Kernels | Tests | Workflow |
|---|---|---|---|---|---|
| commit | every push and PR | `(analytic or derivative or contract_negative) and not spice and not campaign and not qt and not slow` | none | 1247 | [Scientific tiers](../../.github/workflows/scientific-tiers.yml) |
| pr | every PR | `(external_oracle or ((analytic or derivative or contract_negative) and spice)) and not campaign and not qt and not slow` | required | 719 | Scientific tiers |
| nightly | 01:17 UTC daily, manual | all tests, slow tests on (L3 + L4 + full regression) | required | 2835 | [Scientific nightly](../../.github/workflows/scientific-nightly.yml) |
| release | `v*` tags, manual | `real_data` | required | 7 | Scientific nightly |

Desktop (`qt`) tests stay in the required **Public checks** workflow. The four
required checks (Windows public checks, Documentation build, Library qualification,
Dependencies review) keep their names and content. The scientific tiers are added
alongside them and are not required checks.

```powershell
python tools/run_tier.py commit
python tools/run_tier.py pr
python tools/run_tier.py nightly
```

## Kernels in CI

Kernels are never committed. [`tools/fetch_kernels.py`](../../tools/fetch_kernels.py)
assembles the CI set: both ephemerides and both Earth-orientation profiles, plus the
station SPK and the high-precision Earth PCK the tests furnish directly. It checks
every file against the SHA-256 in `lunar_od.kernel_manifest` and takes each file
from, in order:

1. the verified copy in the Actions cache (`outputs/kernels`, keyed on the manifest);
2. `--source-dir` (a local validated kernel directory);
3. the zip at the repository secret `LUNAR_OD_KERNEL_ARCHIVE_URL`, with
   `LUNAR_OD_KERNEL_ARCHIVE_TOKEN` as an optional bearer token;
4. the NAIF URL recorded in the manifest.

NAIF rotates several of the validated files (the Earth predict and high-precision
PCKs, the versioned lunar frame kernel), so step 4 alone cannot rebuild the set.
Build the archive once from the validated local set and host it:

```powershell
python tools/fetch_kernels.py --make-archive outputs/lunar-od-kernels.zip --source-dir $env:USERPROFILE\Documents\mice\kernels
```

Until the secret is set, the kernel-backed jobs fail at "Verify or fetch SPICE
kernels" and list each missing kernel. That failure is intended: a kernel tier that
skips would hide exactly what REQ-SW-003 forbids.
