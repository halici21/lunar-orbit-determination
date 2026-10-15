# Roadmap

The current release line is an alpha research/source preview. This roadmap records
acceptance criteria rather than dates or promises. Track work in
[GitHub Issues](https://github.com/halici21/lunar-orbit-determination/issues).

## Available now

- Installable scientific library and native Windows desktop workspace.
- [Data-free, seeded batch-fit tutorial](docs/tutorials/quickstart.md) with traceable plots and CSVs.
- [Support matrix](docs/validation/support-matrix.md), public regression CI and searchable documentation.
- Preserved study records with current/superseded conclusions and known defects.

## Next research milestones

| Work | Acceptance evidence |
|---|---|
| Paired BLS-LM / SRIF / SR-UKF comparison | Predeclared scenarios, matched data and uncertainty assumptions, explicit Q/R, multiple seeds, error distributions and failure cases |
| Independent propagation/observable comparison | Aligned frames, time scales, constants and force/measurement models; stored external reference with reproducible inputs |
| Stronger consistency assessment | Sufficient independent trials, documented degrees of freedom, residual/innovation diagnostics and justified NIS/NEES interpretation |

## Next delivery milestones

| Work | Acceptance evidence |
|---|---|
| Standalone Windows distribution | Clean-machine installation, packaged child-process/backend execution, cancellation/shutdown tests and scientific data setup |
| Broader library platform qualification | Explicit Python/OS matrix with passing library checks; desktop qualification remains separately stated |
| Archived research release DOI | Owner-authorized Zenodo integration, archived release record and verified DOI added to citation metadata |

The small quickstart is not the estimator benchmark above. Passing a software
milestone does not close a scientific qualification item. Long campaigns remain
manual until their scope and resource requirements are explicitly agreed.
