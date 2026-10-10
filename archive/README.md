# Worktree snapshots — 2026-09-24

A preservation archive, **not code to merge**. Every lunar-OD worktree on the development machine
was scanned for uncommitted content whose exact bytes exist in no commit on any branch. Those
worktrees are snapshotted here so nothing exists only on one disk. The worktrees themselves were
not modified.

Each `worktree_snapshots/<name>/` holds:

- `changes.patch` — `git diff HEAD --binary` of the worktree (tracked changes). Verified to apply
  cleanly onto `base_commit` with `git apply --check`.
- `untracked/` — byte copies of untracked, non-ignored files.
- `meta.json` — base commit, branch, and for every file whether its exact content already exists in
  some commit (`content_status`, `unique_files`).

Restore: `git worktree add <dir> <base_commit>`, then `git apply changes.patch` and copy
`untracked/` back.

All files here are stored byte-exact (`.gitattributes: * -text`), so patches survive
`core.autocrlf`.

## What each snapshot is

| worktree | base | what it is | status |
|---|---|---|---|
| `python_port_r1_r4_phase1_complete_20260810_090000` | `a347397` (branch `fix/r1-r4-long-arc-time-conditioning`) | First draft of the Q1-F07/F08 light-time update-criterion repair: `two_way_range.py`, `radiometrics.py`, `test_od_contracts.py`, and a new `tests/test_q1_f08_update_conditioning.py` (191 lines). | The repair itself **is already on `main`** in its final form (`two_way_range.py` Q1-F07/F08 blocks). The 191-line test was **never committed anywhere**. This snapshot is its only copy. |
| `python_port` | `632560d` (`feature/lunar-j2-force-models`) | `docs/measurement_models_physics_derivatives_and_gaps.md` (2,061 lines, 2026-07-12, never committed; byte-identical to `Grad/measurement_models_physics_derivatives_and_gaps_AUDIT_REFERENCE.md`), plus a one-line change to `PYQT5_DESKTOP_APP_README.md`. | The README change is **accidental corruption**: a sentence is spliced mid-word ("…what has s to provide…"). It is preserved here only so nothing is lost; it was reverted in the worktree. |
| `python_port_phase17r0` | `595f5b9` | `examples/phase17_r1od_fd_convergence.py`, `examples/phase17_r1od_qualification.py`. | Deliberately never committed by Phase 17-R1O-D: written against a superseded `delta_dor` API. |
| `python_port_p0b2_final_closure_20260716_181920` | `da28e5a` | `_closure_full.xml`, `_closure_m3.xml`, `_closure_slow.xml`: pytest JUnit reports of the P0B-2 final closure run (full run: 619 tests, 0 failures, 0 errors, 28 skipped, 2026-07-16). | Closure **test evidence** for `da28e5a`, never committed. Added in a second archive commit, after a first scan wrongly took these single-line files for empty. |
| `python_port_p0b2*_validate_*` (6) | `e572a64` | Intermediate P0B-2 / history-domain validation states, 2026-07-14/15. | **Superseded.** The final P0B-2 work was committed and pushed as `da28e5a` (`lunar_od/history_domain.py`, `measurements.py` etc. are byte-identical to it). The "unique" files are earlier iterations of the same work. |

Worktrees whose uncommitted content is *entirely* already in commits (r0a/r0b/r1 validation copies
and others) are not snapshotted: nothing of theirs is unique.
