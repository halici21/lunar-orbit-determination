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

## Reference platform for SIMD-sensitive gates

Some predeclared gates sit within floating-point or integration-floor distance of
their thresholds. Their measured value depends on the CPU instruction set: whether
numpy dispatches AVX-512 or only AVX2, and which CPU Numba compiles for. GitHub
assigns hosted runners at random, so these gates gave a different verdict from
run to run. The measurements are in
`studies/phase17/results/wp7_1_platform_gate_values/` (script: [`wp7_1_platform_gate_values.py`](../../studies/phase17/scripts/wp7_1_platform_gate_values.py)):

| Gate (test) | Threshold | numpy AVX-512 (recorded) | numpy AVX2 |
|---|---|---|---|
| S1 T_G5(a) (`test_tide_stm_matches_repropagation_and_omitting_its_gradient_does_not`) | ≤ 1e-7 | 7.37e-8 PASS | 1.43e-7 FAIL |
| D5 G1 (`test_dmc_stm_matches_central_finite_differences`) | ≤ 1e-6 per column | 7.90e-7 PASS | 1.18e-6 FAIL |
| S2 L_G6 (`test_six_42_and_48_state_paths_carry_the_same_lrp`, 4 cases) | ≤ 1e-5 m | 3 FAIL, 1 PASS | 4 PASS |

**Rule.** The verdict of such a gate is the verdict on the **reference platform**:
numpy dispatches `AVX512F` and `AVX512_SKX`, and Numba compiles for a host CPU with
`avx512f` and no `NUMBA_CPU_NAME`, `NUMBA_CPU_FEATURES` or `NUMBA_DISABLE_JIT`
override ([`tests/reference_platform.py`](../../tests/reference_platform.py)). All
recorded S1, S2 and D5 verdicts were produced there. No threshold changed. A FAIL
on the reference platform stays a FAIL, and an off-platform PASS is not evidence.

- The tests carry the `reference_platform` resource marker (rows in `tests/levels.py`).
- The commit tier excludes them (`not reference_platform`).
- Off the reference platform they **skip** with the reason, so a nightly run on an
  AVX2 runner shows them as skipped, not as passed or failed.
- `python tools/run_tier.py reference` runs them with `--reference-platform`: off
  the reference platform every one of them **fails** at setup, so the tier cannot
  report a verdict from the wrong machine.

`campaign` and gravity-field data are a separate case: hosted runners never have
them, so those tests still skip there. They run in the local kernel-backed full
regression.

## Cadence

[`tools/run_tier.py`](../../tools/run_tier.py) defines each tier; run
`python tools/run_tier.py --list` to print them.

| Tier | When | Selection (`-m`) | Kernels | Tests | Workflow |
|---|---|---|---|---|---|
| commit | every push and PR | `(analytic or derivative or contract_negative) and not spice and not campaign and not qt and not slow and not reference_platform` | none | 1351 | [Scientific tiers](../../.github/workflows/scientific-tiers.yml) |
| pr | every PR | `(external_oracle or ((analytic or derivative or contract_negative) and spice)) and not campaign and not qt and not slow` | required | 719 | Scientific tiers |
| reference | manual, on the reference platform (recording machine) | `reference_platform` | none | 6 | none (hosted runners are not guaranteed to be the reference platform) |
| nightly | 01:17 UTC daily, manual | all tests, slow tests on (L3 + L4 + full regression) | required | 2835 | [Scientific nightly](../../.github/workflows/scientific-nightly.yml) |
| release | `v*` tags, manual | `real_data` | required | 7 | Scientific nightly |

Desktop (`qt`) tests stay in the required **Public checks** workflow. The four
required checks (Windows public checks, Documentation build, Library qualification,
Dependencies review) keep their names and content. The scientific tiers are added
alongside them and are not required checks.

```powershell
python tools/run_tier.py commit
python tools/run_tier.py pr
python tools/run_tier.py reference
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
