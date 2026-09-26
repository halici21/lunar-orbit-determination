# Phase 17-HARDEN — persistent task checklist

Legend: `[x]` done with evidence · `[~]` in progress · `[ ]` open.
Class: **R** = REQUIRED_FOR_CORRECTNESS (blocks closure) · **F** = RECOMMENDED_FOLLOWUP ·
**O** = OPTIONAL_CLEANUP · **X** = OUT_OF_SCOPE.

| # | item | class | status | evidence |
|---|---|---|---|---|
| 1 | verify repository state | R | [x] | start HEAD 57ca034, main/origin fea476f, 2 untracked orphans |
| 2 | inspect project instructions | R | [x] | worktree + workspace CLAUDE.md; CLAUDE_OPUS_5_5_REPO_INSTRUCTIONS.md NOT FOUND |
| 3 | build defect ledger | R | [x] | 23 defects: `docs/phase17_harden_defect_ledger.md`, `artifacts/phase17_harden_defects.json` (single source `examples/phase17_harden_ledger.py`) |
| 4 | architecture / contract reconnaissance | R | [x] | 17 contracts, `artifacts/phase17_harden_physical_contracts.csv` |
| 5 | subagent decision | R | [x] | 2 read-only agents (physics-contract audit sweep; FA-06/R2 non-pass investigation), within the limit of 3; all shared edits by the lead |
| 6 | time-contract audit | R | [x] | D02, D16, D20, D21; event ledgers carry local t and absolute ET |
| 7 | ephemeris-contract audit | R | [x] | D02, D03, D14, D15; `lunar_od/ephemeris.py` contract, `lunar_od/measurement_contracts.py` |
| 8 | frames / origins / units audit | R | [x] | mutants E/F; station-velocity FD test; third-body epoch test |
| 9 | physical-visibility audit | R | [x] | D04: single contract `station_visibility_mask`; all builders default to it |
| 10 | measurement event-timing audit + ledgers | R | [x] | `artifacts/phase17_harden_{range,doppler,ddor,optical}_event_ledger.json` |
| 11 | quarantine unsafe legacy paths | R | [x] | 11 scripts gated by `PHASE17_LEGACY_REPRODUCTION`; legacy row rule warns; leak tests |
| 12 | close range-Jacobian discrepancy (root cause) | R | [x] | D05: omitted SRP position gradient; prediction explains all 6 columns (corr ≥ 0.986) with the oracle at rtol 3e-15 |
| 13 | requalify measurement derivatives (ΔDOR, Doppler, optical) | R | [x] | `artifacts/phase17_harden_derivatives.json` |
| 14 | historical-defect mutant tests A–F | R | [x] | `tests/test_harden_mutants.py` (31 tests) |
| 15 | boundary / long-arc / windowed-arc / cross-measurement / stress tests | R | [x] | `tests/test_harden_boundaries.py` (17); windowed arc in `test_fxr_earth_epoch_fixture.py` |
| 16 | investigate existing full-suite nonpasses (FA-06, R2, 10 errors) | R | [x] | D07, D09; cbc3af6 |
| 17 | orphan-script disposition | R | [x] | D11: `archive/phase17_orphans/` + README + manifest |
| 18 | external-input inventory + path resolver | R | [x] | D10, D12: `lunar_od/project_paths.py`, guarded campaign import |
| 19 | silent-fallback / frozen-state search | R | [x] | D15, D16 fixed; D18, D19 documented follow-ups (bounded in Phase-17 use) |
| 20 | SRP/shadow + third-body contract tests | R | [x] | `tests/test_harden_boundaries.py` |
| 21 | STM reshape audit | R | [x] | 25 sites order='F'; 3 labelled C-order defect oracles |
| 22 | targeted tests | R | [x] | per-batch runs, see report §Testing |
| 23 | broader integration tests | R | [x] | 374-test affected batch |
| 24 | full suite | R | [x] | see closure report |
| 25 | final diff inspection | R | [x] | see closure report |
| 26 | authoritative documentation (errata, status doc, stale docstring) | R | [x] | `docs/phase17_authoritative_status.md`; 6 errata blocks; D06 |
| 27 | closure report | R | [x] | `docs/phase17_harden_closure_report.md` |
| F1 | add SRP position gradient to the variational equations | F | [x] | done in Phase 17-F1 (`docs/phase17_f1_srp_variational_gradient_qualification.md`) |
| F2 | bound the extrapolation helpers (`accelerated`, `radiometrics`, `filters`) | F | [ ] | D18, D19 |
| F3 | reject untagged Earth providers once all callers are tagged | F | [ ] | D14 |
| F4 | Hermite Earth interpolation in counted Doppler | F | [ ] | D23 (documented design R3-P22) |
| O1 | retire `VisibilityConfig.epoch_utc` on the transform path; explicit origin for `earth_position_mci`; drop residual zeroing | O | [ ] | D20–D22 |
