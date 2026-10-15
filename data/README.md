# Reference and external data

Small committed reference inputs live in `fixtures/` and
`tests/fixtures/` (including synthetic gravity models). Study configurations
and authoritative outputs live with their studies, not in this directory.

SPICE kernels, full GRAIL gravity fields, MATLAB ephemerides, and the historical
`od_covariance_campaign` data tree are external and are not bundled or newly
downloaded by repository checks. See the
[kernel setup](../docs/windows-quickstart.md#spice-kernels) and the selected
study report for required model identities and source provenance.

| Variable | Meaning |
| --- | --- |
| `LUNAR_OD_PROJECT_ROOT` | Parent containing the campaign tree and `ephemeris_data.mat` |
| `LUNAR_OD_CAMPAIGN_ROOT` | Explicit historical campaign directory |
| `LUNAR_OD_GRAVITY_ROOT` | Historical study gravity-data root |
| `LUNAR_OD_GRAVITY_DIR` | Production gravity-loader search override |
| `LUNAR_OD_KERNEL_DIR` | SPICE kernel directory |

Absent overrides, historical project paths default relative to the repository's
parent. A dedicated worktree in another directory should set the overrides
explicitly. Kernel/gravity loaders retain their existing search order.
Large local gravity files under `data/gravity/` are ignored. Generated runs,
plots and caches belong in ignored output directories. Do not commit data
without checking its provenance, redistribution terms and reproducibility role.
