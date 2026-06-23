# Result Artifact Map

Every published number, plot, CSV, thesis result, README metric, or presentation
claim must be traceable through this chain:

```
claim -> figure/table -> CSV/PNG artifact -> scenario config -> producing script -> random seed -> commit hash (if known)
```

## Apply to
- thesis figures and tables
- presentation / slide numbers
- README metrics
- protected baseline outputs (see `baseline-registry.md`)
- exploratory outputs (label them as exploratory)

## Rules
- Generated artifacts live under `python_port/results/`; they are produced by
  scripts in `python_port/examples/`, not authored by hand.
- If any link in the chain is unknown, state it explicitly — do not imply
  traceability that does not exist.
- Do not invent artifacts, scripts, seeds, or commit hashes. Unknown -> mark as
  `expected / verify in repo` or `unknown`.
