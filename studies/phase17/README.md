# Phase 17 research campaign

Phase 17 investigates SRP-parameter observability, covariance qualification,
tracking geometry and additional observables, followed by production estimator
and measurement hardening. Its reports distinguish surrogate studies from
qualified production paths and retain scientific errata and failed hypotheses.

| Directory | Purpose |
| --- | --- |
| [scripts](scripts/) | Campaign drivers, numerical oracles, analysis and plotting |
| [configs](configs/) | Predeclarations, grids, campaign configurations and saved job specifications |
| [results](results/) | Committed scientific evidence: summaries, supporting arrays/tables and cited figures |
| [reports](reports/) | Qualification reports, literature contracts, status and defect ledger |
| [provenance](provenance/) | Original integration/reconstruction manifests |

Begin with [authoritative status](reports/phase17_authoritative_status.md),
[closure report](reports/phase17_harden_closure_report.md), and
[defect ledger](reports/phase17_harden_defect_ledger.md). Later qualification
reports and errata determine which historical conclusions remain current.

The FE-L field-error study includes its committed driver, ten cell records,
validation gates and report. FE-P currently has a committed predeclaration
only; no FE-P driver or results are introduced by the branch integration.

## Reproduce a study

Install the repository with `python -m pip install -e .` and use the validated
dependency profile described in the root README. Read the chosen script and
matching report/predeclaration before running it; drivers can start expensive
campaigns or deliberately rewrite study evidence.

```powershell
python studies/phase17/scripts/phase17_kp2_campaign.py --help
```

Set `LUNAR_OD_PROJECT_ROOT` to the directory containing the external
`od_covariance_campaign` tree and `ephemeris_data.mat`. Individual overrides are
`LUNAR_OD_CAMPAIGN_ROOT`, `LUNAR_OD_GRAVITY_ROOT`, and `LUNAR_OD_KERNEL_DIR`.
The external campaign imports the baseline `phase6_scenario_comparison` helper
from `examples/`; the test harness and study loader preserve that relationship.
See [data policy](../../data/README.md). Missing historical external data must
be reported explicitly; committed summaries alone do not regenerate a run.

Historical drivers write to the committed evidence directory. Run them in a
disposable worktree, retain the original evidence, and review output diffs.
Local exploratory output belongs in ignored `generated/`, root `results/`, or
`outputs/`; promote only reviewed evidence with its config and provenance.

## Historical paths and identity

All configurations, predeclarations, result records, figures and original
manifests were moved without changing their bytes. Embedded paths, timestamps,
commits, hashes and numeric values describe the historical run. They are not
silently rewritten to suggest that a historical run used this layout.

Use the [path migration manifest](../../docs/development/path-migration.json)
to resolve old `artifacts/`, `08_manifests/`, `docs/phase17*` and
`examples/phase17*` paths to current locations. Reports' navigable links and
live code lookups use the current paths. Earlier snapshots in `archive/` remain
unchanged and are not current entry points. Git history is preserved through
renames; use `git log --follow -- <current-path>`.
