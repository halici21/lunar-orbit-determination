# Roadmap

The current release line is an alpha research/source preview. This roadmap records
acceptance criteria rather than dates or promises. Track work in
[GitHub Issues](https://github.com/halici21/lunar-orbit-determination/issues).

## Available now

- Installable scientific library and native Windows desktop workspace.
- [Data-free, seeded batch-fit tutorial](../docs/tutorials/quickstart.md) with traceable plots and CSVs.
- [Support matrix](../docs/validation/support-matrix.md), public regression CI and searchable documentation.
- Preserved study records with current/superseded conclusions and known defects.
- [Real desktop tutorial and recorded workflow](../docs/tutorials/desktop.md), including a data-free child-process solve.
- [Source-generated API signatures and numerical contracts](../docs/site/api.md).
- Windows/Linux library qualification for Python 3.10–3.13, dependency alerts and mandatory PR dependency review.
- [Predeclared 64-trial BLS-LM/SRIF/SR-UKF experiment](../docs/research/paired-benchmark.md), with raw inputs and caveats.
- Python-bundled Windows portable preview and [gated release automation](../docs/development/releases.md).

## Next research milestones

| Work | Acceptance evidence |
|---|---|
| Broaden the paired comparison's physical coverage | Additional predeclared geometries, real visibility, force-model mismatch and independent truth; the published Level-0 campaign is the starting point |
| Independent propagation/observable comparison | Aligned frames, time scales, constants and force/measurement models; stored external reference with reproducible inputs |
| Stronger consistency assessment | Sufficient independent trials, documented degrees of freedom, residual/innovation diagnostics and justified NIS/NEES interpretation |

## Next delivery milestones

| Work | Acceptance evidence |
|---|---|
| Installed/signed Windows application | Installer/update lifecycle, signing and additional clean Windows editions; the portable alpha is already available |
| Additional library platforms | Explicit passing macOS/Python 3.14+ matrix before expanding support claims |
| [Archived research release DOI](https://github.com/halici21/lunar-orbit-determination/issues/15) | Owner-authorized Zenodo integration, archived release record and verified DOI added to citation metadata |

The one-seed quickstart and bounded paired campaign have distinct scopes. Passing a software
milestone does not close a scientific qualification item. Long campaigns remain
manual until their scope and resource requirements are explicitly agreed.
