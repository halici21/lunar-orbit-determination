# Runnable examples and compatibility drivers

Install the library first with `python -m pip install -e .`.

| Start here | Purpose |
| --- | --- |
| `scenario_config_cli.py --schema` | Inspect the JSON scenario schema without a scientific run |
| `run_scenario_config.py config.json --dry-run` | Validate and summarize a scenario |
| `compare_range_rate_physics.py` | Compare explicitly selected measurement-physics assumptions |
| `compare_visibility_models.py` | Inspect visibility-model differences |
| `plot_aberration_corrections.py` | Reproduce the documented aberration figure |

Read a driver's arguments and [baseline research guide](../docs/research/research-guide.md)
before running it. SPICE-backed examples require kernels. Multi-day and Monte
Carlo drivers are manual campaigns, not quick installation checks.

Earlier phase, baseline, thesis and campaign drivers retain their established
locations for reproduction and external imports. In particular,
`phase6_scenario_comparison.py` is imported by the external campaign tree;
`r1_covariance_observability_validation.py` and `run_scenario_config.py` have
byte-protected provenance identities. Do not rewrite these solely to remove
legacy import bootstraps.

Phase 17 research drivers have moved to [studies/phase17](../studies/phase17/README.md).
The [path manifest](../docs/development/path-migration.json) resolves their
historical names. Generated runs belong in ignored `results/` or `outputs/`;
review scientific evidence before promoting it into a study's tracked results.
