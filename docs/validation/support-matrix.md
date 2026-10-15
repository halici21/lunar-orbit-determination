# Support and qualification matrix

Implementation support is narrower than a universal accuracy claim. Follow the
linked evidence for exact assumptions, profiles and external inputs.

| Capability | Estimator / entry point | Inputs | Evidence and boundary |
|---|---|---|---|
| Geometric range / azimuth / elevation | BLS-LM, SRIF, SR-UKF | Synthetic geometry or configured ephemeris | [Batch tests](../../tests/test_estimators.py), [filter tests](../../tests/test_filters.py); [quickstart](../tutorials/quickstart.md) covers BLS-LM/SRIF only |
| One-way light time and stellar aberration for position observables | BLS-LM / SRIF; UKF position profiles rejected | Consistent receive/transmit history and frame/ephemeris inputs | [Jacobian model](../theory/one_way_light_time_jacobians.md), [runtime support gate](../../src/lunar_od/filters.py); profile-specific analytic/hybrid limits apply |
| Range-rate and counted Doppler | Batch / sequential paths subject to selected radiometric model | Matched count interval, event model, history and station-state method | [Counted-Doppler contract](../theory/two_way_counted_doppler.md), [filter implementation](../../src/lunar_od/filters.py); geometric rate and counted Doppler are different observables |
| Converged two-way range | BLS-LM / SRIF; UKF rejected | Pre/post-roll history, event-epoch station states, fixed delay | [Two-way range contract](../theory/two_way_range.md), [integration tests](../../tests/test_two_way_range_integration.py); not a complete operational DSN calibration model |
| SRIF square-root formal handoff | SRIF only | Posterior square-root information and documented process noise | [Scenario implementation](../../src/lunar_od/scenarios.py), [scenario tests](../../tests/test_scenarios.py); other estimators reject this start mode |
| SPICE geometry, visibility and frames | Ephemeris / station pipeline | External documented SPICE kernels | [Windows data setup](../windows-quickstart.md), [frame tests](../../tests/test_frame_spice_validation.py); public data-free CI does not qualify every kernel-backed campaign |
| High-degree gravity / SRP solve-for / field-error studies | Study-specific paths | External gravity data plus each study's saved configuration | [Authoritative status](../../studies/phase17/reports/phase17_authoritative_status.md), [defect ledger](../../studies/phase17/reports/phase17_harden_defect_ledger.md); uncertainty values and superseded results must be checked there |
| Native desktop workspace | PyQt5 / Windows reference | Python 3.13.2, validated dependencies; kernels for scientific runs | [Public verification](verification.md), [desktop workflow](../../desktop_app/RESULT_WORKFLOW.md); process completion and estimator convergence are separate |
| Windows portable desktop | Bundled CPython / PyQt5 / scientific backend | Windows x64; writable extracted directory | [Portable qualification](../../tools/qualify_windows_portable.py), [release gate](../development/releases.md); imports, child processes, cancellation, tutorial and shutdown; external kernels remain separate |

## Verification levels

- **Tutorial:** seeded, same-model synthetic recovery; no operational claim.
- **Public CI:** bounded regressions using committed fixtures with external data isolated.
- **Research qualification:** the exact assumptions and recorded outcomes of each study.
- **External / operational validation:** not established by passing the above checks.

## Library platform matrix

[Library qualification](../../.github/workflows/library.yml) builds a wheel,
checks every package import outside the checkout, compares propagation with an
analytic circular orbit, checks dependency consistency and runs the
[kernel-independent regression tier](../../tools/run_library_checks.py).

| OS runner | Python versions | Scope |
|---|---|---|
| Ubuntu 24.04 x64 | 3.10, 3.11, 3.12, 3.13 | Scientific library wheel and data-free regressions |
| Windows Server 2022 x64 | 3.10, 3.11, 3.12, 3.13 | Scientific library wheel and data-free regressions |
| Windows desktop reference | 3.13.2 x64 | Pinned source/portable desktop checks |

The library matrix resolves dependencies compatible with each Python version;
it does not assert identical floating-point outputs across different dependency
versions. Four explicitly named integration cases that unconditionally load
external kernels belong to the `--with-kernels` tier, not the data-free matrix.
Other existing kernel-dependent cases may report explicit skips.
macOS and Python 3.14+ remain unqualified here. Package metadata's Python 3.10+
range does not expand this tested matrix or desktop qualification.

The [paired estimator experiment](../research/paired-benchmark.md) supplies a
separate bounded scientific comparison: 32 seeds per noise block, Level-0 truth,
matched priors and full failure/termination reporting. It does not qualify
external frames, real visibility or mission performance.
