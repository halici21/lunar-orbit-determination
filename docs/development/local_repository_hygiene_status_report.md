# Local Repository Hygiene — Status Report (2026-09-24)

Repository hygiene audit only. No science phase, no physics change, no historical number rewritten.
Machine-readable evidence: `artifacts/repository_hygiene/` (pre/post inventories, deletion manifest,
reference checks, and the three scripts that produced them).

## 1. Executive Summary

The target worktree (`python_port_phase17r0`, branch `feature/phase17-r-k-srp-estimation`) was
already in good order: 550 tracked files, **0 tracked modifications**, 2 untracked files, and
535 ignored files. The ignored files are 179 bytecode/Numba caches, the Graphify graph and the
Phase 17-GEO design-matrix cache.

The only genuine clutter was **179 cache files (5,752,114 bytes)** in five `__pycache__/`
directories. They were deleted one path at a time, each recorded in
`docs/development/repository_hygiene/deletion_manifest.csv`. **Nothing else was deleted and nothing was
moved.** Every other file is either active, provenance, or evidence.

The five exact-duplicate groups are all intentional. The two untracked R1O-D orphan scripts are
preserved for owner review. No reference was broken by the cleanup.

The sibling worktrees and the workspace-level folders were audited **read-only**, per the workspace
`CLAUDE.md` rule 4 ("do not edit, stash, reset, stage, clean, or restore a non-target worktree").

## 2. Repository Identity

| item | value |
|---|---|
| remote | `origin https://github.com/halici21/lunar-orbit-determination.git` |
| workspace | `Documents/Python/lunar-orbit-determination/` (renamed from `Grad/` on 2026-09-24; hidden junction `Grad` keeps old absolute paths working) |
| target worktree | `python_port_phase17r0` |
| shared `.git` | `python_port/.git` |

## 3. Current Branch / HEAD / Tree

```
CURRENT_WORKTREE = python_port_phase17r0
CURRENT_BRANCH   = feature/phase17-r-k-srp-estimation   (tracks origin, in sync)
START_HEAD       = 595f5b9853048bcb74ee622688821d237095a24b
START_TREE       = 411a8fe2a83a09ef83df3987e938f372e0a17fa1
GIT_STATUS_PORCELAIN (start) =
  ?? examples/phase17_r1od_fd_convergence.py
  ?? examples/phase17_r1od_qualification.py
```

The branch is 16 commits ahead of `main` and 0 behind. Its last commit is Phase 17-GEO (2026-09-23).

## 4. Main / Origin Main Status

```
MAIN_HEAD        = fea476f81dad07b3914e53eab709e10fa6e9d10b
ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b
```

This matches the expected canonical main. `integration/phase17-canonical` points at the same commit.

## 5. Local Worktrees

| path | branch | HEAD | state | class |
|---|---|---|---|---|
| `python_port` | `feature/lunar-j2-force-models` (upstream gone, fully merged into main) | 632560d | 1 untracked file: `docs/measurement_models_physics_derivatives_and_gaps.md` (protected) | ACTIVE (primary checkout, holds shared `.git`) |
| `python_port_main_canonical` | `main` | fea476f | clean | ACTIVE |
| `python_port_phase17r0` | `feature/phase17-r-k-srp-estimation` | 595f5b9 | 2 untracked (R1O-D orphans) | ACTIVE (target) |

`git worktree list --porcelain` reports no prunable or broken entries. Worktree links were repaired
after the folder rename (`git worktree repair`), and `git status` succeeds in all three worktrees.
`STALE_WORKTREE_CANDIDATES = 0`.

## 6. Local Branches

"merged" means it is an ancestor of `main`; every branch below is contained in some `origin/*` ref.

| branch | HEAD | upstream | behind/ahead main | merged | last commit | class |
|---|---|---|---|---|---|---|
| `main` | fea476f | in sync | 0/0 | — | 2026-09-14 | ACTIVE |
| `feature/phase17-r-k-srp-estimation` | 595f5b9 | in sync | 0/16 | no | 2026-09-23 | ACTIVE |
| `feature/lunar-j2-force-models` | 632560d | gone | 20/0 | yes | 2026-08-02 | POSSIBLY_STALE (checked out in `python_port`; cannot be deleted while checked out) |
| `archive/worktree-snapshots-2026-09-24` | a25d6fa | pushed | 0/3 | no | 2026-09-24 | HISTORICAL_KEEP (only copy of former worktree-local content) |
| `wip/phase17-r-estimator-integration` | 13e75f1 | in sync | 12/1 | no | 2026-09-14 | HISTORICAL_KEEP (cited in `studies/phase17/reports/phase17_r1m_multi_arc_identifiability.md`) |
| `integration/phase17-canonical` | fea476f | none | 0/0 | yes | 2026-09-14 | HISTORICAL_KEEP (cited in `studies/phase17/provenance/phase17_canonical_integration_manifest.txt`) |
| `feature/phase14-17ar-qualified` | 0e785f6 | none | 11/0 | yes | 2026-09-14 | HISTORICAL_KEEP (cited in `studies/phase17/provenance/phase14_17ar_reconstruction_manifest.txt`) |
| `feature/lunar-high-order-gravity-closure` | fc13cc4 | gone | 12/0 | yes | 2026-08-13 | SAFE_DELETE_CANDIDATE |
| `feature/r0b-force-contract-parity` | 9dd19c3 | gone | 35/0 | yes | 2026-07-30 | SAFE_DELETE_CANDIDATE |
| `feature/r1-covariance-observability-parity` | 6e88015 | gone | 31/0 | yes | 2026-07-30 | SAFE_DELETE_CANDIDATE |
| `feature/r2-measurement-fidelity-parity` | 5ca652a | gone | 24/0 | yes | 2026-08-02 | SAFE_DELETE_CANDIDATE |
| `feature/r3-exact-station-transform` | d03a913 | gone | 18/0 | yes | 2026-08-07 | SAFE_DELETE_CANDIDATE |
| `feature/r4-four-event-counted-doppler` | 5bf0c54 | gone | 15/0 | yes | 2026-08-09 | SAFE_DELETE_CANDIDATE |
| `fix/history-domain-enforcement` | da28e5a | gone | 41/0 | yes | 2026-07-15 | SAFE_DELETE_CANDIDATE |
| `fix/light-time-nonconvergence` | e572a64 | none | 42/0 | yes | 2026-07-14 | SAFE_DELETE_CANDIDATE |
| `fix/measurement-model-safety` | 3265dbe | none | 43/0 | yes | 2026-07-14 | SAFE_DELETE_CANDIDATE |
| `fix/r0a-force-config-ukf-parity` | 543b6b9 | gone | 40/0 | yes | 2026-07-23 | SAFE_DELETE_CANDIDATE |
| `fix/r1-r4-long-arc-time-conditioning` | a347397 | gone | 10/0 | yes | 2026-08-10 | SAFE_DELETE_CANDIDATE |
| `validation/r1-r4-long-arc-qualification` | 3c6f3b0 | gone | 13/0 | yes | 2026-08-09 | SAFE_DELETE_CANDIDATE |

The 12 SAFE_DELETE_CANDIDATE branches are fully merged into `main` (0 unique commits) and are not
cited by name in any tracked file. **None were deleted**; branch deletion is left to the owner.

## 7. Stashes

`git stash list` is empty. `STASH_COUNT = 0`.

## 8. Tracked Modified Files

None at start. The only tracked change made by this task is the addition of this report and of
`artifacts/repository_hygiene/`.

## 9. Untracked Files

| path | class | action |
|---|---|---|
| `examples/phase17_r1od_fd_convergence.py` | REVIEW_REQUIRED | kept (see §18, §24) |
| `examples/phase17_r1od_qualification.py` | REVIEW_REQUIRED | kept (see §18, §24) |

## 10. Ignored Files

| ignored path | files | size | class | action |
|---|---|---|---|---|
| `*/__pycache__/` (5 dirs) | 179 | 5.75 MB | DELETE_CACHE_TEMP | deleted |
| `graphify-out/` | 294 | 36 MB | KEEP_ACTIVE (regenerable tool output used by the `CLAUDE.md` discovery workflow) | kept |
| `results/phase17_geo_cache/` | 62 | 9.3 MB | KEEP_GENERATED_EVIDENCE | kept |

`.gitignore` already covers the local junk classes: `__pycache__/`, `*.py[cod]`, `.pytest_cache/`,
`.coverage`, `*.log`, `*.tmp`, editor/OS files and `results/`. **No `.gitignore` change was
needed.** No broad `*.csv`/`*.json`/`*.png` rule was added.

## 11. Source-Code Inventory

- `lunar_od/`: 53 files, production package, KEEP_ACTIVE. All 52 modules import cleanly.
- `desktop_app/`, `ui/`, `lunar_od_app.py`, `ui_server.py`: KEEP_ACTIVE.
- `.claude/skills/`: 66 files, the canonical project skills, KEEP_ACTIVE.

No source file was touched.

## 12. Analysis-Script Inventory

`examples/`: 92 tracked scripts, plus the 2 untracked orphans.

| group | scripts | class |
|---|---|---|
| Phase 17-GEO (current) | 8 | KEEP_ACTIVE |
| Phase 17-R1O-OPT | 5 | KEEP_PROVENANCE (`phase17_r1o_opt_core.py` KEEP_ACTIVE) |
| Phase 17-R1O-D | 5 tracked + 2 untracked orphans | KEEP_PROVENANCE / REVIEW_REQUIRED |
| Phase 17-R1O (incl. R1O-R repair path) | 10 | KEEP_PROVENANCE (`phase17_r1o_core.py` KEEP_ACTIVE) |
| Phase 17-R1COV | 7 | KEEP_PROVENANCE |
| Phase 17-R1M | 6 | KEEP_PROVENANCE (`phase17_r1m_core.py` KEEP_ACTIVE; imported by later phases) |
| Phase 0–13, R1/R2/R4 | 17 | KEEP_PROVENANCE (`r2_measurement_fidelity_validation.py` protected) |
| general runners / demos | 34 | KEEP_ACTIVE |

No duplicate implementations were collapsed. Historically distinct scripts, including superseded or
erroneous paths, remain byte-identical so that each phase's executed code stays reconstructible.

## 13. Test Inventory

- 75 `tests/test_*.py` files, **1319 tests collected, 0 collection errors**.
- No test file is uncollected, and none refers to a missing file. `tests/README.md`, `tests/slow.py`
  and `tests/fixtures/gravity/*` are support files.
- Protected regressions are present, including `tests/test_r1o_stm_layout.py` (R1O STM-layout
  regression) and `tests/test_delta_dor.py`.
- No test was deleted, moved or edited.

## 14. Documentation Inventory

`docs/` holds 24 Markdown documents plus 21 figure files:
- **Phase reports, contracts and errata (KEEP_PROVENANCE):** R1, R1G, R1M, R1COV, R1O, R1O erratum, R1O-D (report + literature contract), R1O-OPT (report + contract), R1O-R, GEO (report + contract), R1–R4 long-arc qualification, and the P2 canonical integration report.
- **Current technical docs (KEEP_ACTIVE):** frames, light-time Jacobians, SRP, two-way range/Doppler, SPICE cross-validation and stellar aberration.

Root-level `.md` files: `README.md`, `CLAUDE.md`, `docs/development/PROJECT_CONTEXT.md`, `docs/development/plan.md`, `docs/development/UI_README.md`,
`docs/development/PYQT5_DESKTOP_APP_README.md` (protected) and `docs/development/PYQT5_DESKTOP_APP_BRIEF.md` are KEEP_ACTIVE.
`docs/theory/DOPPLER_RANGE_RATE_MODEL_REVIEW.md`, `docs/validation/SPICE_CN_CNPLUS_VALIDATION.md` and
`docs/research/LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md` are KEEP_PROVENANCE.

Those last three were considered for a move to `docs/` and **not moved**, for three reasons:
- They are cited by path from skills (`lunar-od-literature-review`, `lunar-od-model-review`), `docs/development/PROJECT_CONTEXT.md` and `docs/theory/frame_architecture_contracts_and_inconsistencies.md`.
- They are identical on `main`, so moving them only here would create branch divergence purely for cosmetics.
- Rule 24: do not over-organize.

`docs/measurement_models_physics_derivatives_and_gaps.md` (protected) is **not part of this
worktree**. It exists untracked in the sibling `python_port` worktree, and a byte-identical copy
(SHA-256 `8f01b2be…468e`) sits at the workspace root as
`measurement_models_physics_derivatives_and_gaps_AUDIT_REFERENCE.md`. It is also preserved on
`archive/worktree-snapshots-2026-09-24`. All copies were left untouched.

## 15. Scientific Artifact Inventory

`artifacts/`: 141 tracked files, 4.2 MB, class KEEP_GENERATED_EVIDENCE. They are phase evidence (CSV/JSON/PNG) for R1, R1G, R1M, R1COV, R1O, R1O-D, R1O-OPT, R1O-R and GEO. Several are pinned by SHA-256 in phase manifests (`r1od_manifest.json`, the GEO predeclared-grid hash, and others).

`results/phase17_geo_cache/`: 62 `.npz` design-matrix caches, 9.3 MB, gitignored, class KEEP_GENERATED_EVIDENCE.
- They are read by `studies/phase17/scripts/phase17_geo_figures.py` and `studies/phase17/scripts/phase17_geo_fixture_attribution.py`.
- Regenerating them requires re-propagating the full predeclared GEO grid (`phase17_geo_campaign.py`), so they are cached evidence, not junk.

`08_manifests/`: 4 phase manifests, KEEP_PROVENANCE.

## 16. Figure / CSV Inventory

48 tracked scientific figures (plus the UI icon `ui/assets/lunar-icon.svg`) and 70 tracked CSVs.
**Every tracked CSV** is referenced by a report, a script or an artifact manifest.

| finding | files | status |
|---|---|---|
| figure with producing script in `examples/` | 32 (R1M, R1COV, R1O, R1O-D, GEO, aberration) | OK |
| GEO figure CSVs co-located with figures (`docs/figures/phase17_geo/*.csv`) | 9 (f1–f9) | OK; `f3`/`f4` are also written to `artifacts/` (see §17) |
| **figure with no producer script anywhere** (repo HEAD, all git history, workspace) | 14: `artifacts/phase17r1_*.png` (6) and `artifacts/phase17r1g_*.png` (8) | kept; provenance gap, see §24 |
| figure not cited by any report | 2 older demo figures (`docs/figures/lt_demo.png`, `lt_jacobian_check.png`, added with the one-way light-time feature in 267ce96) | kept; historical |
| missing figure referenced by a report | 0 | — |

The R1/R1G PNGs were committed together with their phase report and manifest (e9216bf, 3abc883).
Their data CSVs (`phase17r1_holdout_metrics.csv`, `phase17r1g_*.csv`) are tracked, but the plotting
code was never committed.

## 17. Duplicate Files

Byte-identical groups (SHA-256). **None deleted**, because each duplicate is intentional:

| duplicate | authoritative / partner | SHA-256 (prefix) | why both stay |
|---|---|---|---|
| `studies/phase17/results/figures/phase17_geo/f3_altitude.csv` | `studies/phase17/results/phase17_geo_altitude.csv` | 89e444f4 | `phase17_geo_figures.py` deliberately writes the figure data next to the figure and into the artifact set |
| `studies/phase17/results/figures/phase17_geo/f4_inclination.csv` | `studies/phase17/results/phase17_geo_inclination.csv` | 8382865f | same |
| `results/phase17_geo_cache/FIXTURE_PUBLISHED.npz` | `FIXTURE_CAMPAIGN_TABLE.npz` | 315cb9d8 | their identity **is the result** `phase17_geo_fixture_attribution.py` checks (lines 64–65) |
| `results/phase17_geo_cache/G3_i90.0_b00.npz` | `G1_b00.npz` | 387edcc8 | physically identical grid cases; cache is looked up by case ID |
| `results/phase17_geo_cache/G2_alt100_b20_fix{D,N}.npz` | `G1_b20.npz` | 19c1c238 | same |

`EXACT_DUPLICATES_DELETED = 0`

## 18. Orphaned Files

**R1O-D orphan scripts** (untracked, dated 2026-09-18):

| path | SHA-256 | size |
|---|---|---|
| `examples/phase17_r1od_qualification.py` | ec36d2327ba7…04f13f6 | 14,168 B |
| `examples/phase17_r1od_fd_convergence.py` | f48fd6de0254…21bbb6 | 6,408 B |

- **Why they are untracked:** it is deliberate and documented. `studies/phase17/results/r1od_manifest.json` lists both under `orphaned_scripts_not_committed` ("reference a superseded pre-rewrite delta_dor.py API"), and `studies/phase17/reports/phase17_r1o_r_stm_layout_repair.md` §5 records them as left untouched. `.graphifyignore` excludes both by name.
- **Are they runnable?** No. They import `spacecraft_differential_delay`, `delta_dor_observable`, `delta_dor_k_column` and `transmit_time_consistency_check`, none of which exist in the current `src/lunar_od/delta_dor.py` (only `QuasarDirection` survives).
- **Tracked equivalent:** `studies/phase17/scripts/phase17_r1od_jacobian_and_k.py` now writes the same artifacts that `qualification.py` used to write (`r1od_jacobian_fd_sweep.csv`, `r1od_k_sensitivity_fd.csv`, `r1od_qualification_summary.json`). The manifest-pinned hashes belong to the tracked producer.
- **Outputs of `fd_convergence.py`:** `r1od_fd_convergence_{state,k}.csv` exist nowhere.
- **Did later phases copy their logic?** Their helper `phi_6x6_history` and the case labels `W15_ddor`/`W15_fd` appear in no other file.
- **Unique content:** `fd_convergence.py`'s docstring records the first-pass FD finding (relative error rising monotonically from 9.97e-4 to 1.25 as the step shrank, i.e. the plateau was never reached). That finding is **not** stated in the R1O-D report.
- **Class:** REVIEW_REQUIRED, kept in place (see §24).

Other orphans: the 14 R1/R1G figures without a producer (§16). No orphaned source or test file was found.

## 19. Cache / Temporary Files

Deleted: `examples/__pycache__/` (18), `lunar_od/__pycache__/` (67, of which 30 are Numba
`.nbi`/`.nbc` JIT caches), `lunar_od/reference_config/__pycache__/` (10),
`lunar_od/rf/__pycache__/` (6), `tests/__pycache__/` (78). Total: 179 files, 5,752,114 bytes.
The next Numba-using run recompiles once.

No `.pytest_cache`, `*.tmp`, `*.log`, `*.orig`, `*.swp`, `.ipynb_checkpoints` or `.DS_Store`
existed in the target worktree.

## 20. Large Files and Directories

| path | size | tracked | generated | referenced | regenerable | authoritative copy elsewhere |
|---|---|---|---|---|---|---|
| `graphify-out/` (graph.json ×2 at 10.5 MB) | 36 MB | no (ignored) | yes | by `CLAUDE.md` workflow | yes (`graphify update .`) | no; keep |
| `results/phase17_geo_cache/` | 9.3 MB | no (ignored) | yes | by GEO figure/attribution scripts | only by full re-propagation | no; keep |
| `artifacts/` | 4.2 MB | yes | yes | by reports/manifests | partly | no; keep |
| `docs/figures/` | 1.7 MB | yes | yes | by reports | yes (GEO), no (lt_*) | no; keep |

Largest tracked file: `studies/phase17/results/figures/phase17_geo/f6_srp_rtn_and_eclipse_timelines.png` (0.32 MB).
The worktree has no large binaries.

Workspace level (read-only, outside this worktree):
- `ephemeris_data.mat` (516 MB; read by absolute path, do not move)
- `python_port/data/` (299 MB, ignored) and `python_port/ephemeris_data.mat` (169 MB, ignored) inside the primary checkout
- `_archive/` (432 MB), `scratchpad/` (127 MB), `thesis_presentation/` (63 MB), `od_covariance_campaign/` (37 MB)

## 21. Files Deleted

179 files, all DELETE_CACHE_TEMP. Each row of `docs/development/repository_hygiene/deletion_manifest.csv`
records path, size, SHA-256, classification, reason, the regenerating source module, the
reference-search result and tracked status. The deletion script
(`tools/repository_hygiene/hygiene_delete_cache.py`) had three safeguards:
- It processed only five named directories.
- It refused any file that was tracked or not git-ignored.
- It removed a directory only with `os.rmdir` after the directory was empty.

The post-cleanup import check recreated one empty `lunar_od/__pycache__/` directory, which was
removed with `rmdir`.

## 22. Files Moved

None. `FILES_MOVED = 0`.

## 23. Files Preserved for Provenance

These were all kept unchanged:
- All phase reports, errata and literature contracts (R1 through GEO).
- `08_manifests/`, all of `artifacts/`, `docs/figures/` and `results/phase17_geo_cache/`.
- Every phase script, including historically superseded ones.
- The protected files: `docs/development/PYQT5_DESKTOP_APP_README.md`, `examples/r2_measurement_fidelity_validation.py`, and the GEO contract, report, figures and artifacts.
- The two R1O-D orphans.

## 24. Files Requiring Owner Review

See the **OWNER REVIEW REQUIRED** section at the end.

## 25. Graphify Discovery Findings

- The graph (`graphify-out/graph.json`) was built at `4c2fad2` (Phase 17-R1O-R), **before** the GEO commits. As a result:
  - 243 of 252 tracked `.py` files are visible.
  - The 8 `examples/phase17_geo_*.py` scripts and `.claude/skills/_shared/validate_skill_sync.py` (excluded by design) are invisible.
- `graphify explain "spacecraft_differential_delay"` matched only a test-function name (`test_zero_baseline_gives_zero_spacecraft_differential_delay`); `delta_dor_k_column` and `phi_6x6_history` have no node. This is consistent with the orphans' API having been removed and with the orphans being excluded by `.graphifyignore`.
- Graphify was used for discovery only. No deletion or keep decision rests on a missing graph edge.

## 26. Manual Dependency Findings

- Reference searches covered every file name (and module stem for `.py`) across all tracked and untracked text files; per-file counts are in `post_cleanup_files.csv`.
- `results/phase17_geo_cache` is consumed through a path constant (`CACHE = ROOT / "results" / "phase17_geo_cache"`), which is invisible to an import graph.
- The orphans' producer role for tracked artifacts was found only by string search (`ARTIFACTS / "r1od_…"`).
- No code, test or doc references `__pycache__` contents (only `README.md:449` lists it as ignored).

## 27. Cases Graphify Could Not Resolve

- GEO scripts (stale graph).
- Data dependencies through file paths: the GEO cache, artifact writers, and the absolute `…/Python/Grad/…` paths.
- Producers of the R1/R1G figures (no code exists).
- Branch/manifest provenance links (`08_manifests/*.txt` cite branch names).

## 28. Broken Reference Check

Checked Markdown links and repo-relative path mentions in all tracked `.md/.py/.json/.txt` files
(`reference_check_pre.json` vs `reference_check_post.json`).

- **New broken references introduced by cleanup: 0** (the pre and post sets are identical).
- Pre-existing findings, all documented and none touched:
  - 2 "links" in `studies/phase17/reports/phase17_r1o_r_stm_layout_repair.md` L80–81 are false positives (`[:3](t)` math inside a code block).
  - 17 mentions of `results/phase6|12b|13g/...` are output paths written by scripts into gitignored `results/`, so they are absent locally by design.
  - `artifacts/r1m_holdout.csv` is intentionally withheld (the R1M report §53 says so).
  - 2 illustrative paths appear in the Graphify skill's extraction spec.
  - **One genuine stale reference:** the `src/lunar_od/lunar_landmark_optical.py:371` docstring cites `examples/phase17_r1o_opt_light_time.py`, which exists in no commit. The test it describes lives in `studies/phase17/scripts/phase17_r1o_opt_oracles.py` (§8 of the R1O-OPT report). It was not fixed, because that is production code (see Owner Review).
- **Absolute paths:** 42 lines in 31 tracked files hard-code `…/Documents/Python/Grad/…`. They resolve through the hidden `Grad` junction created in the 2026-09-24 folder rename, so they are working, not broken. Rewriting them would change historical scripts and reports, so they were left as they are.

## 29. Test / Import Validation

| check | result |
|---|---|
| import every `lunar_od` module (`pkgutil.walk_packages`) | 52 / 52 OK |
| `pytest tests --collect-only` (before and after) | 75 files, 1319 tests, 0 errors (unchanged) |
| `pytest tests/test_r1o_stm_layout.py tests/test_delta_dor.py` | 28 passed |

All runs used `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider` and `NUMBA_CACHE_DIR` in a
scratch directory, so validation did not recreate the deleted caches. The full suite was not run:
no tracked source, test or configuration file was touched.

## 30. Git Diff Validation

- `git diff` and `git diff --cached` are empty for all pre-existing tracked files.
- The only staged changes are additions under `artifacts/repository_hygiene/` and this report.
- `UNRELATED_TRACKED_CHANGES = 0`.

## 31. Repository State After Cleanup

- Branch `feature/phase17-r-k-srp-estimation`: one new local commit on top of `595f5b9`, containing only this report and `artifacts/repository_hygiene/`.
- Untracked: the 2 R1O-D orphans. Ignored: `graphify-out/`, `results/`. No `__pycache__`.

## 32. Remaining Technical Debt

1. Graphify graph is stale (built at `4c2fad2`); run `graphify update .` in this worktree.
2. 31 tracked files hard-code the old `…/Python/Grad/…` absolute path and depend on the hidden junction. A future change could derive paths from `__file__`, but historical scripts should stay as executed.
3. `lunar_landmark_optical.py:371` docstring points to a nonexistent script.
4. Sibling-worktree clutter, not touched:
   - `python_port/` holds 11 `__pycache__` dirs, a `.pytest_cache`, `frame_audit_run.log`, and an ignored nested `python_port/python_port/` directory.
   - `python_port_main_canonical/` holds 5 `__pycache__` dirs.
   - These should be cleaned from their own sessions.
5. `od_covariance_campaign/` (separate repo, no remote) has 84 untracked entries, e.g. `03_data/phase14_*`.

## 33. Remaining Scientific Debt

This audit did not touch any of these:
- The 5.83-day Earth-epoch fixture defect.
- The production two-way range Jacobian discrepancy (GEO cross-geometry Jacobian gate).
- GEO conclusions, K parameterization, SRP model fidelity and third-body configuration.

Identified here: the R1/R1G figures lack committed plotting code, so those figures cannot be
regenerated from the repository.

## 34. What Was Intentionally Not Touched

- All tracked files.
- The sibling worktrees `python_port` and `python_port_main_canonical`.
- `od_covariance_campaign/`, `scratchpad/`, `_archive/`, `thesis_presentation/` and the workspace-root data files.
- All branches, the (empty) stash list, `graphify-out/`, `results/`, the R1O-D orphans and `.gitignore`.

## 35. Recommended Next Repository Actions

1. Decide on the R1O-D orphans (Owner Review item 1).
2. Optionally delete the 12 fully-merged SAFE_DELETE_CANDIDATE branches (`git branch -d`, which refuses unmerged branches). Their commits are on `origin/main`.
3. Refresh Graphify in this worktree (`graphify update .`).
4. Clean caches in `python_port` and `python_port_main_canonical` from sessions that own them.
5. Fix the one stale docstring reference in a small, separately reviewed commit.

## 36. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 595f5b9853048bcb74ee622688821d237095a24b
START_TREE   = 411a8fe2a83a09ef83df3987e938f372e0a17fa1
FINAL_BRANCH = feature/phase17-r-k-srp-estimation
FINAL_HEAD   = the local commit "chore(repo): audit and clean local repository artifacts" (child of 595f5b9)
FINAL_TREE   = tree of that commit (START_TREE + this report + artifacts/repository_hygiene/)
MAIN_HEAD        = fea476f81dad07b3914e53eab709e10fa6e9d10b
ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b

TRACKED_FILES_TOTAL    = 550 (before this commit)
UNTRACKED_FILES_BEFORE = 2
IGNORED_FILES_BEFORE   = 535

KEEP_ACTIVE_COUNT             = 614
KEEP_PROVENANCE_COUNT         = 69
KEEP_GENERATED_EVIDENCE_COUNT = 223 (excluding this audit's own outputs)

FILES_MOVED   = 0
FILES_DELETED = 179
BYTES_DELETED = 5752114

CACHE_TEMP_DELETED        = 179
EXACT_DUPLICATES_DELETED  = 0
REGENERABLE_JUNK_DELETED  = 0
STALE_SCRATCH_DELETED     = 0

REVIEW_REQUIRED_COUNT = 2
UNKNOWN_COUNT         = 0

BROKEN_REFERENCES_AFTER_CLEANUP = 0 (new); 1 genuine pre-existing stale docstring reference documented

LOCAL_BRANCH_COUNT       = 19
BRANCH_DELETE_CANDIDATES = 12 (reported, none deleted)
WORKTREE_COUNT             = 3
STALE_WORKTREE_CANDIDATES  = 0
STASH_COUNT = 0

GRAPHIFY_DISCOVERY_USED    = YES (graph stale; discovery only)
MANUAL_REFERENCE_AUDIT_USED = YES

UNRELATED_TRACKED_CHANGES = 0
TESTS_RUN   = import sanity (52 modules); collect-only (1319 tests); test_r1o_stm_layout.py + test_delta_dor.py
TEST_RESULT = PASS (52/52 imports, 0 collection errors, 28 passed)

PRODUCTION_CODE_CHANGED          = NO
SCIENCE_RESULTS_CHANGED          = NO
HISTORICAL_REPORT_NUMBERS_CHANGED = NO

COMMITS_CREATED = 1
PUSH  = NONE
MERGE = NONE

LOCAL_REPOSITORY_HYGIENE_GATE = PASS
```

## File-by-File Action Table (notable candidates)

| PATH | STATUS | CLASS | ACTION | WHY | REFERENCES FOUND | REPLACEMENT / AUTHORITATIVE COPY |
|---|---|---|---|---|---|---|
| `*/__pycache__/*` (179 files) | ignored | DELETE_CACHE_TEMP | deleted | bytecode / Numba JIT cache | none (README lists dir as ignored) | regenerated from source modules |
| `examples/phase17_r1od_qualification.py` | untracked | REVIEW_REQUIRED | kept | superseded API, but cited by manifest/report | `r1od_manifest.json`, R1O-R report §5, `.graphifyignore` | `phase17_r1od_jacobian_and_k.py` (for its outputs) |
| `examples/phase17_r1od_fd_convergence.py` | untracked | REVIEW_REQUIRED | kept | holds first-pass FD history not in report | same | none |
| `results/phase17_geo_cache/*.npz` (62) | ignored | KEEP_GENERATED_EVIDENCE | kept | GEO cached designs; costly regeneration | `phase17_geo_{campaign,figures,fixture_attribution}.py` | none |
| `graphify-out/` | ignored | KEEP_ACTIVE | kept | discovery tool output | `CLAUDE.md` | `graphify update .` |
| `studies/phase17/results/figures/phase17_geo/f3_altitude.csv`, `f4_inclination.csv` | tracked | KEEP_GENERATED_EVIDENCE | kept | intentional duplicate of `artifacts/phase17_geo_{altitude,inclination}.csv` | `phase17_geo_figures.py` | both authoritative by design |
| `artifacts/phase17r1_*.png`, `artifacts/phase17r1g_*.png` (14) | tracked | KEEP_GENERATED_EVIDENCE | kept | phase evidence; producer missing | committed with R1/R1G report + manifest | none |
| `docs/figures/lt_demo.png`, `lt_jacobian_check.png` | tracked | KEEP_PROVENANCE | kept | historical feature figures, uncited | none | none |
| `docs/theory/DOPPLER_RANGE_RATE_MODEL_REVIEW.md`, `docs/validation/SPICE_CN_CNPLUS_VALIDATION.md`, `docs/research/LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md` | tracked | KEEP_PROVENANCE | not moved | cited by path; identical on `main` | skills, `docs/development/PROJECT_CONTEXT.md`, frame docs | — |
| `docs/development/PYQT5_DESKTOP_APP_README.md`, `examples/r2_measurement_fidelity_validation.py` | tracked | KEEP_ACTIVE / KEEP_PROVENANCE | kept | protected | — | — |
| `docs/phase17_geo_*`, `docs/figures/phase17_geo/`, `artifacts/phase17_geo_*` | tracked | KEEP_PROVENANCE / KEEP_GENERATED_EVIDENCE | kept | latest scientific evidence | GEO scripts, report | — |

# OWNER REVIEW REQUIRED

1. **`examples/phase17_r1od_fd_convergence.py`, `examples/phase17_r1od_qualification.py`** (untracked, this worktree)
   - *What they are:* first-draft R1O-D qualification and FD-convergence scripts written against the pre-rewrite `delta_dor` API. They no longer run.
   - *Why uncertain:* the manifest calls them deliberately uncommitted, but `fd_convergence.py` documents a first-pass FD non-convergence finding that appears nowhere else.
   - *Recommended:* commit both unchanged under an archival location, e.g. to the `archive/worktree-snapshots-2026-09-24` branch, then delete locally. Alternatively, keep them as they are.
   - **Resolved (Phase 17-HARDEN, HARDEN-D11):** both are archived byte-for-byte in `archive/phase17_orphans/`,
     with a README and a SHA-256 manifest that matches the hashes above.
   - *Risk if removed:* loss of the only record of the first FD sweep's failure mode.
2. **R1/R1G figures without a producer** (`artifacts/phase17r1_*.png` ×6, `artifacts/phase17r1g_*.png` ×8)
   - *What they are:* committed phase evidence whose plotting code was never committed.
   - *Why uncertain:* the figures cannot be regenerated from the repository.
   - *Recommended:* keep them; if the plotting code is found (e.g. in a thesis-side notebook), commit it.
   - *Risk if removed:* irrecoverable figures.
3. **`src/lunar_od/lunar_landmark_optical.py:371`** docstring cites nonexistent `examples/phase17_r1o_opt_light_time.py`
   - *What it is:* a stale reference in production code.
   - *Why not fixed:* this task must not change production code.
   - *Recommended:* change the reference to `studies/phase17/scripts/phase17_r1o_opt_oracles.py` in a separate docs-only commit.
   - *Risk:* none to behavior; misleading documentation.
4. **12 fully-merged local branches** (§6)
   - *What they are:* completed R0–R4 and fix branches, all contained in `origin/main`.
   - *Recommended:* delete them with `git branch -d` if no longer wanted locally.
   - *Risk if removed:* none to history, since their commits remain reachable from `main`.

## Final Question

**What currently exists in the local repository, what was genuine clutter, what was removed or
reorganized, what was deliberately preserved, and what still requires owner attention before the
next scientific phase?**

**What exists:**
- The active worktree holds the production package, 1319 collected tests, and 92 analysis scripts covering Phase 0 through Phase 17-GEO.
- It also holds 24 technical documents and phase reports, 141 evidence artifacts, 4 phase manifests, and a cached GEO design-matrix set.
- There are three worktrees (main, the Phase 17 science branch, and the primary checkout) and 19 local branches, and every commit is on GitHub.

**Genuine clutter:** only Python and Numba caches.

**Removed:** 179 cache files, 5.75 MB. **Reorganized:** nothing, because nothing was
misplaced enough to justify diverging from `main` or breaking cited paths.

**Deliberately preserved:**
- Every report, erratum, contract, manifest, figure, CSV and phase script, including superseded ones.
- The GEO evidence and the GEO cache.
- The protected files and the two R1O-D orphans.

**Needs owner attention before the next scientific phase:**
- Decide the fate of the two R1O-D orphans.
- Note that 14 R1/R1G figures have no reproducible producer.
- Refresh the stale Graphify graph so the GEO scripts are visible to impact analysis.
- Optionally prune the 12 merged branches and fix one stale docstring reference.

The known scientific defects (Earth-epoch fixture, production range Jacobian) remain open and
belong to their own dedicated phases.
