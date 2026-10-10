# Phase 17 orphan scripts (archived by Phase 17-HARDEN)

These two scripts were untracked working-tree files left over from Phase 17-R1O-D.
`docs/local_repository_hygiene_status_report.md` (§18, §24) classified them as
REVIEW_REQUIRED. Phase 17-HARDEN archives them here and does not delete them, so
their provenance survives (defect HARDEN-D11).

| file | why it is dead |
|---|---|
| `phase17_r1od_qualification.py` | Imports `delta_dor_k_column` and `spacecraft_differential_delay` from `lunar_od.delta_dor`. Neither exists after commit 1045898 (the production common-transmit ΔDOR model), so the import fails. Its artifacts (`r1od_jacobian_fd_sweep.csv`, `r1od_k_sensitivity_fd.csv`, `r1od_qualification_summary.json`) are now written by the tracked `examples/phase17_r1od_jacobian_and_k.py`. |
| `phase17_r1od_fd_convergence.py` | Has the same dead imports. Its outputs (`r1od_fd_convergence_{state,k}.csv`) exist nowhere. |

- **Not runnable.** They are kept only as a record, outside `examples/` so that neither `pytest` nor any `sys.path` setup can import them.
- **Superseded evidence.** The current ΔDOR derivative evidence is in `examples/phase17_harden_events_and_derivatives.py` and `tests/test_harden_*.py`.
- **Machine-specific path.** Each file still contains its original machine-specific `ARTIFACTS` literal. It is preserved deliberately, byte for byte.

`MANIFEST.sha256` records SHA-256 hashes of the archived bytes in two forms: as they are on disk (CRLF), and LF-normalized. The LF-normalized hashes match the hygiene report.
