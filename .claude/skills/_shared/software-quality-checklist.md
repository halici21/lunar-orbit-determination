# Software Quality Checklist

For maintainability, robustness, and engineering quality.

Check:
- clear module ownership; no duplicated logic without reason
- no hidden global state; no import-time side effects; no circular imports
- separation between science logic, experiment scripts, and UI code
- meaningful names; explicit units where relevant
- explicit frame / time-system assumptions; no unexplained magic numbers
- no silent exception swallowing; helpful error messages
- explicit numerical tolerances; deterministic random seeds where required
- no unintended mutation of input arrays
- graceful handling of missing SPICE kernels, empty measurements, invalid configs

## Critical rule
A code-quality change is only safe when it preserves scientific behavior or
explicitly documents the intended behavior change.
