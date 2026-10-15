# D1 Measurement-Model Safety Diagnostics

Status: D1 baseline evidence with post-D1 P0A, P0B-1, and P0B-2 resolution
updates, 2026-07-15.

This report covers FA-01, FA-02, FA-03A, FA-03B, and FA-06 only. Sections 1-13
preserve the D1 pre-fix measurements and explicitly label later safety-contract
updates; Section 14 summarizes the completed P0A/P0B-1/P0B-2 status. It does
not execute the D2 frame, PCHIP, visibility, or lunar-frame campaigns.

## 1. Environment

| Field | Value |
|---|---|
| Repository | `C:/Users/erayh/Documents/Python/Grad/python_port_measurement_fix` |
| Branch | `fix/measurement-model-safety` |
| HEAD | `eb92461f781c3fccd39012e3a02cd6ace64d893b` |
| Python | 3.13.12, `C:/Users/erayh/miniforge3/python.exe` |
| pytest | 9.0.3 |
| Platform | Windows 11 / win32 |
| Date | 2026-07-13 |
| Production files changed | none |

The main-worktree Phase13G commit `f360687` was not merged or cherry-picked.

## 2. Pytest import provenance

Issue: **FA-06**.

The parent config is
`C:/Users/erayh/Documents/Python/Grad/pytest.ini`. It declares both
`python_port` and `python_port/desktop_app` under `pythonpath`. A plain pytest
session can therefore import the main worktree even when the current directory
is the isolated worktree.

The verified D1 policy was used for every pytest invocation:

```powershell
python -m pytest --override-ini=pythonpath= ...
```

Focused-session evidence from
`test_fa06_pytest_session_imports_isolated_repository`:

| Field | Observed |
|---|---|
| pytest rootdir | `C:/Users/erayh/Documents/Python/Grad` |
| pytest config | `C:/Users/erayh/Documents/Python/Grad/pytest.ini` |
| repository root | `C:/Users/erayh/Documents/Python/Grad/python_port_measurement_fix` |
| `lunar_od.__file__` | isolated `python_port_measurement_fix/lunar_od/__init__.py` |
| effective pytest `pythonpath` | `[]` |
| relevant `sys.path` entry | isolated worktree only |

The diagnostic test asserts semantically that the resolved package is below
the repository root; it contains no machine-specific expected path.

The earlier provenance experiment established that plain `PYTHONPATH` and
`-p no:pythonpath` were insufficient on their own. D1 did not repeat an unsafe
plain-pytest negative test. Plain pytest remains unsafe in this worktree;
`--override-ini=pythonpath=` is required.

Clearing the parent config also clears the path used by
`tests/test_pytest_desktop_app.py` for its bare `from app` import. The first D1
full-suite attempt therefore stopped during collection with
`ModuleNotFoundError: app`. The successful full run kept the required override
and supplied isolated paths for that process only:

```powershell
$env:PYTHONPATH="$PWD;$PWD\desktop_app"
python -m pytest --override-ini=pythonpath= tests/
```

Before that run, both imports were checked:

```text
.../python_port_measurement_fix/lunar_od/__init__.py
.../python_port_measurement_fix/desktop_app/app.py
```

The parent `pytest.ini` and repository configuration were not changed.

## 3. D1 scope

Executed:

- FA-01 UKF position profile-physics mismatch.
- FA-02 one-way solver metadata mismatch.
- FA-03A nonconverged light-time results reaching observable paths.
- FA-03B legacy event-history extrapolation.
- FA-06 pytest/import provenance defect and guard.

Not executed: FA-04, FA-05, FA-07, frame interpolation campaigns, PCHIP,
visibility parity, lunar-frame cadence, and all production safety fixes.

## 4. Tests added or extended

At the D1 baseline, `tests/test_measurement_model_safety_diagnostics.py` was
added with seven passing current-behavior tests:

| Test | Evidence |
|---|---|
| `test_fa06_pytest_session_imports_isolated_repository` | pytest-session provenance |
| `test_fa01_current_ukf_operator_is_geometric_for_cn_profiles` | generated profile vs UKF operator |
| `test_fa01_config_acceptance_and_m3_rejection_are_explicit` | accepted inconsistent configs; retained M3 rejection |
| `test_fa02_generation_metadata_reports_range_rate_solver_defaults` | actual vs reported solver contract |
| `test_fa03a_one_way_nominal_consumes_nonconverged_last_iterate` | one-way nominal/Jacobian split |
| `test_fa03a_counted_observable_and_jacobian_consume_nonconverged_endpoints` | counted endpoint flags ignored |
| `test_fa03b_history_domain_matrix_documents_current_behavior` | five-path boundary matrix |

A single new file was used because each D1 assertion crosses two or more of
`measurements`, `filters`, `radiometrics`, and `scenario_config`, and all tests
share one explicit current-defect vocabulary. Existing owner tests were not
duplicated or modified. No helper was added to the production package.

## 5. FA-01 evidence

**Source symbols:** `scenario_config._validate_cross_field_rules`,
`measurements.generate_position_measurements`,
`filters._position_measurement_from_state`.

The test generated geometric, CN, CN+S local-MCI, and CN+S SSB measurements
from the same state history, station, receive epoch, and identity frame
transform. The SPICE call boundary was deterministic; the production profile
routing and observable implementations were used.

The table reports `UKF operator - generated selected profile`. This synthetic
case is diagnostic amplification, not a general campaign error estimate.

| Selected profile | Range [m] | Az [rad] | Az [arcsec] | El [rad] | El [arcsec] |
|---|---:|---:|---:|---:|---:|
| geometric | 0 | 0 | 0 | 0 | 0 |
| one-way CN | +817.226390 | -4.243459e-6 | -0.875276 | +3.767581e-6 | +0.777119 |
| CN+S local-MCI | +817.226390 | -6.913887e-5 | -14.260915 | +5.101278e-5 | +10.522141 |
| CN+S SPICE-SSB | +817.226390 | +1.323117e-5 | +2.729124 | -6.679996e-5 | -13.778481 |

For every profile, the UKF internal operator equaled the geometric observable
to the test tolerance. Config validation accepted UKF with all three
non-geometric profiles and `jacobian_model="implicit_light_time"`. M3 two-way
range with UKF remained explicitly rejected.

**Classification:** FA-01 confirmed, CRITICAL configuration/physics mismatch.

**Recommended fix:** first hard-reject non-geometric position profiles for UKF;
implement profile-aware sigma-point histories only in a later, separately
validated change.

## 6. FA-02 evidence

**Source symbols:** `measurements.solve_one_way_light_time`,
`measurements.measurement_model_metadata`,
`radiometrics.RangeRatePhysicsConfig`.

The test read the one-way defaults from the live function signature and read
metadata attached by normal position generation:

| Contract | Tolerance [s] | Max iterations |
|---|---:|---:|
| Actual one-way solver | `1e-12` | 10 |
| Reported position metadata | `1e-10` | 20 |
| Range-rate config defaults | `1e-10` | 20 |

The generated metadata exactly matched `RangeRatePhysicsConfig` defaults, not
the one-way solver used by the generated observable.

**Classification:** FA-02 confirmed, traceability/reproducibility defect.

**Recommended fix:** give the one-way profile an explicit solver-policy owner
and emit metadata from that same object or shared constants.

## 7. FA-03A evidence and P0B-1 resolution

### One-way CN

**Source symbols:** `solve_one_way_light_time`,
`_apparent_position_observable`, `one_way_light_time_range_sensitivity`.

A deterministic high-velocity synthetic target and `max_iter=1` produced:

| Field | Observed |
|---|---:|
| `solution.converged` | false |
| iterations | 1 |
| returned transmit epoch | -0.950657671315 s |
| independent equation residual | -2.501730714e-3 s |
| equivalent range residual | -750000.000 m |

The nominal observable returned finite range/angles from the last transmit
epoch. The sensitivity/Jacobian helper raised `RuntimeError` for the same
nonconverged solve. The large residual is intentionally forced diagnostic
evidence, not a nominal orbit error estimate.

### Legacy counted Doppler

**Source symbols:** `solve_two_way_light_time`,
`two_way_counted_doppler_observable`,
`two_way_counted_doppler_initial_state_jacobian`.

Both count endpoints returned `converged=False` with `max_iter=1`. The final
observable exactly matched the scaled difference of those two last-iterate
round-trip times within floating-point spacing:

| Field | Observed |
|---|---:|
| endpoint flags | `[False, False]` |
| counted result | 8728007.725983271 m/s |
| maximum independent leg-equation residual | 6.031153714e-4 s |
| equivalent range residual | 180809.439637 m |
| counted initial-state Jacobian | finite; no convergence exception |

**D1 classification:** FA-03A confirmed at the D1/P0A baseline. The strictness
split was narrower than a generic "all sensitivity helpers are strict"
statement: the one-way Jacobian rejected nonconvergence, while the one-way
nominal and legacy counted nominal/Jacobian paths consumed invalid final
iterates.

**D1 recommendation:** enforce convergence and a separately evaluated equation
residual in both nominal and derivative paths; never publish the last iterate
as an observable.

### P0B-1 implemented contract

P0B-1 resolves FA-03A. A one-way or round-trip light-time solution is valid
only when both of these independently checked conditions pass:

1. the fixed-point iteration update is within its update tolerance;
2. the final light-time equation, freshly re-evaluated at the returned event
   epochs, is within its equation-residual tolerance.

The raw `solve_one_way_light_time` and `solve_two_way_light_time` APIs may
still return diagnostic nonconverged solution objects. Those objects expose
update status/residuals and final equation residuals; they are not valid
observable values. Production consumers now enforce the boundary:

- `_apparent_position_observable`, one-way sensitivities, and the local and
  initial-state Jacobian chains reject invalid results with
  `LightTimeConvergenceError`. One-way nominal and Jacobian paths therefore
  use the same failure policy.
- `two_way_counted_doppler_observable` validates `count-start` and `count-end`
  separately. The counted initial-state Jacobian applies the same endpoint
  checks and raises `RoundTripLightTimeConvergenceError` on failure.
- Counted diagnostics distinguish uplink and downlink update/equation failures.
  Equation residuals are reported in seconds and equivalent metres.

Normal converged numerical behavior is intended to remain unchanged. The
forced-failure tests retain the independent D1 residual calculations as the
oracle and now assert controlled rejection:

| P0B-1 test | Contract |
|---|---|
| `test_p0b1_one_way_nonconverged_last_iterate_is_rejected` | raw diagnostic result retained; nominal and sensitivity consumers reject |
| `test_p0b1_counted_observable_and_jacobian_reject_nonconverged_endpoints` | count-start/count-end and counted Jacobian rejection |
| `test_p0b1_dual_criterion_rejects_equation_residual_after_updates_converge` | equation closure remains mandatory after the update criterion passes |
| `test_p0b1_equation_tolerances_must_be_finite_and_positive` | configuration validation |

P0B-1 validation completed with `16 passed` in the focused safety file and
`523 passed, 28 skipped, 1 warning, 8 subtests passed` in the full suite. The
warning is the existing test-only numerical-Jacobian performance warning.
P0B-1 did not address FA-03B; P0B-2 resolves it as documented below. The legacy
counted model is still single-bounce; a four-event counted-Doppler model remains
future work. M3 two-way range behavior is unchanged.

## 8. FA-03B history-domain matrix and P0B-2 resolution

**Source symbols:** `radiometrics.interp_state_history`, `_interp_state`,
`_interp_vector`, `_interp_matrix`; `measurements._apparent_position_observable`;
`filters._two_way_local_histories`, `_interp_pass_values`.

The following matrix is the preserved **D1 pre-fix baseline**, not current
production behavior:

| Path | Before start | Exact start | Exact end | After end |
|---|---|---|---|---|
| One-way observable | silent extrapolation | interpolation | silent extrapolation | silent extrapolation |
| One-way Jacobian | silent extrapolation | interpolation | silent extrapolation | silent extrapolation |
| Counted observable | silent extrapolation | interpolation | interpolation | silent extrapolation |
| Counted Jacobian | silent extrapolation | interpolation | interpolation | silent extrapolation |
| UKF counted local history | silent extrapolation | interpolation | interpolation | silent extrapolation |

For the one-way upper-bound case, a physical transmit event exactly at the
history end requires a receive epoch after the end. The final event is at the
boundary, but the solver's initial receive-epoch target probe is already
outside the history; the path is therefore classified as silent extrapolation.

For counted paths, the cases were constructed from the full event envelope:
the earliest station transmit event was before/exactly at the lower bound, and
the latest receive endpoint was exactly at/after the upper bound. For the UKF
path, local-history start/end were aligned with the source pass bounds and the
linearly extrapolated Earth positions were verified numerically.

At D1, no tested legacy path raised a controlled history-domain exception,
recorded a drop, or persisted extrapolation metadata.

**D1 classification:** FA-03B confirmed.

**D1 recommendation:** define required pre-roll/post-roll per observable, reject
unsupported spacecraft/event epochs before solving, and persist structured
failure/drop metadata. Preserve exact-boundary acceptance explicitly.

### P0B-2 implemented contract

P0B-2 resolves FA-03B without changing the low-level interpolation algorithms
or extending source histories. For a requested epoch `q` and compared support
bound `b`:

```text
S = max(1, abs(q), abs(b))
policy_ulp = nextafter(S, +infinity) - S
```

Support is the closed interval `[t_start,t_end]`. Exact endpoints and requests
at most two policy ULP outside support use the endpoint sample. Three or more
policy ULP outside support raises `HistoryDomainError` before interpolation or
extrapolation. The first unsupported intermediate solver probe stops the solve;
the guard never clips a solver event variable.

| Path | Current P0B-2 behavior |
|---|---|
| One-way observable and Jacobian | Every transmit-state probe, final state re-query, and transmit STM lookup is guarded; nominal/sensitivity/Jacobian failure policy is identical. |
| Counted observable | `count-start` and `count-end` are separate; spacecraft state, Earth position/velocity, and transform lookups identify uplink/downlink context. |
| Counted Jacobian | The nominal guards also apply, and the spacecraft STM at the reflection event is independently guarded. |
| Counted UKF local history | Earth position, Earth velocity, and transform source intervals are preflighted before local spacecraft propagation or source re-sampling. The preflighted envelope is anchored at the clock-corrected count endpoints — min/max of the raw versus corrected count-start/count-end, computed with the observable's own clock correction — so nonzero station clock offset/drift of either sign is covered (P0B-2E2). Accepted representation-only offsets alter source lookup only; physical local times are unchanged. |
| Direct estimator/observability consumers | Unsupported records raise the same controlled error; these layers do not silently skip rows. |

Generation catches `HistoryDomainError` per visible candidate, emits one
ordered `HistoryDomainDropRecord`, and omits the row. Candidate RNG draws happen
before physics evaluation: three draws for position and four for range-rate,
so later surviving seeded rows and the RNG tail match a fully supported
pre-rolled run. Noise-disabled generation consumes no draws. Metadata retains
ordered records, drop count, and maximum actual required pre/post-roll.

Scenario handling is family-local because one `build_measurement_arcs` call and
one `ScenarioResult` own one measurement family. Partial drops leave surviving
arcs estimable. If all eligible arcs in the selected family are domain-empty,
`build_measurement_arcs` raises an aggregate `HistoryDomainError` before any
estimator call. Successful results carry total/position/range-rate drop counts,
maximum pre/post-roll, an all-empty flag, and detailed records. The summary CSV
appends the six scalar fields; detailed records remain in result metadata.

No automatic pre-roll/post-roll is created. The legacy counted model remains
single-bounce and rejects nonzero transponder delay; four-event counted Doppler
remains future work. M3 retains its separate `TwoWayEventHistoryError`, event
provider, generation behavior, and nonzero-delay model.

Key safety tests now cover exact/1/2-policy-ULP acceptance, symmetric 3+-ULP
rejection and mutation detection, first unsupported solver probes, all five
counted history names, endpoint/leg diagnostics, UKF preflight ordering,
generation drop/RNG determinism, estimator/observability propagation,
family-local empty-arc handling, CSV aggregates, and unchanged M3 suites.

## 9. Findings confirmed

| Issue | Result |
|---|---|
| FA-01 | Confirmed at D1; **resolved by P0A hard rejection** |
| FA-02 | Confirmed at D1; **resolved by P0A one-way metadata ownership** |
| FA-03A | Confirmed at D1/P0A baseline; **resolved by P0B-1 strict dual-criterion enforcement** |
| FA-03B | Confirmed at D1; **resolved by P0B-2 closed-support enforcement, deterministic drops, and scenario/report transport** |
| FA-06 | Confirmed: parent pytest config requires explicit provenance override |

The D1 defect evidence was expressed as passing tests that asserted baseline
behavior. P0A/P0B-1/P0B-2 converted resolved findings into passing
safety-contract tests. No test is intentionally failing or marked xfail.

## 10. Findings reclassified

No issue was dismissed. Two claims were refined:

1. At D1, FA-03A derivative behavior was model-specific: one-way sensitivity
   was strict while one-way nominal and legacy counted nominal/Jacobian paths
   were permissive. P0B-1 replaces that split with one strict consumer policy.
2. The D1 FA-03B one-way exact-upper-event case included an earlier
   out-of-domain receive-epoch probe, so exact support for the final physical
   event alone was insufficient. P0B-2 intentionally guards every intermediate
   probe; the same construction now raises before model evaluation.

The full-suite `from app` collection failure is classified as an FA-06 test
harness dependency exposed by clearing the parent config, not as a production
measurement defect.

## 11. Recommended production fixes

Apply in separate production patches, with these D1 tests converted from
current-defect assertions to safety-contract assertions:

1. **P0.1 (completed by P0A):** reject UKF with CN/CN+S position profiles.
2. **P0.2 (completed by P0B-1):** enforce convergence plus independently
   evaluated equation-residual closure for one-way and counted
   nominal/Jacobian paths.
3. **P0.3 (completed by P0B-2):** prohibit silent event-history extrapolation,
   preserve deterministic candidate/RNG behavior, and carry explicit
   pre-roll/domain metadata through family-local scenario reporting.
4. **P0.4 (completed by P0A):** source one-way metadata from the actual
   one-way solver policy.

P0A separately rejects legacy counted Doppler when `transponder_delay_s != 0`.
A fixed scalar delay cancels directly in the endpoint round-trip-light-time
difference, but nonzero delay can still change counted Doppler indirectly
through separate `t2u`/`t2d` spacecraft states and uplink/downlink event
geometry; support returns with a future four-event model.

Do not combine these physics/safety changes with interpolation refactoring.

## 12. D2 diagnostics not yet executed

- FA-04 bare `MOON_PA` alias/default bypass.
- FA-05 ECEF/ITRF93 naming work.
- FA-07 M3 CSV/reporting traceability.
- 1--240 s exact-SPICE frame interpolation sweep.
- PCHIP position/velocity consistency.
- Receive/uplink visibility parity.
- Lunar-frame cadence and kernel-load-order campaign.
- Shared station-provider architecture.
- Four-event counted-Doppler implementation.

No conclusion about these D2 items is inferred from D1.

## 13. Regression results

The following runs are the historical D1 baseline. Every command included
`--override-ini=pythonpath=`.

| Run | Result | Runtime |
|---|---|---:|
| D1 diagnostic finalization run | 7 passed | 2.07 s |
| Related owner/integration regression | 152 passed, 11 skipped, 1 warning, 5 subtests passed | 26.59 s |
| Full suite, final provenance-safe run | 514 passed, 28 skipped, 1 warning, 8 subtests passed | 38.94 s |

Full regression: **0 failed, 0 deselected**. The warning is the existing
`_range_rate_numerical_initial_jacobian` runtime warning in
`tests/test_estimators.py`. The 28 skips are existing gated/optional tests.

The first full-suite attempt with the required override but without the
isolated `desktop_app` environment path failed during collection as described
in section 2. It was not treated as the acceptance run. The final run verified
both isolated imports before pytest and completed successfully.

During D1 diagnostic implementation, before this baseline finalization, no
commit or push was made.

P0B-1 was then verified on the P0A-based implementation worktree:

| Run | Result | Runtime |
|---|---|---:|
| P0B-1 focused measurement-safety diagnostics | 16 passed | 1.15 s |
| P0B-1 full suite | 523 passed, 28 skipped, 1 warning, 8 subtests passed | 23.13 s |

The P0B-1 full regression also had **0 failed and 0 deselected**. These
results validate the implementation consistency of the strict failure policy;
P0B-1 alone did not close FA-03B or change the M3 model.

The complete patch through P0B-2D1 was then exported and reconstructed on a
detached checkout. The technically isolated acceptance runs were:

| Run | Result | Runtime |
|---|---|---:|
| P0B-2D1 focused scenario/reporting | 19 passed, 9 skipped | 10.94 s |
| P0B-2 related history-domain + M3 | 201 passed, 20 skipped, 1 warning, 26 subtests passed | 18.09 s |
| P0B-2 full suite | 557 passed, 28 skipped, 1 warning, 29 subtests passed | 28.10 s |

All three runs had **0 failed and 0 deselected**. The warning is the same
existing test-only numerical-Jacobian performance warning. The related/full
runs include unchanged M3 suites. The validation report records a governance
caveat that the D1 writer and technically isolated validator were the same
Codex session; detached patch reconstruction, fresh imports, and external-only
artifacts provided technical rather than personnel independence.

## 14. Post-D1 safety resolution status

The P0A, P0B-1, and P0B-2 measurement-safety patches changed the status of the
D1 findings as follows; historical tables above remain the pre-fix evidence
record where explicitly labeled.

| Finding | Current status |
|---|---|
| FA-01 | confirmed at eb92461/D1; **resolved by P0A hard rejection** — shared `filters.validate_ukf_measurement_support` enforced at `scenario_config._validate_cross_field_rules` (loader) and at `run_lunar_ukf` entry (runtime defense-in-depth for direct `ScenarioConfig` construction). UKF + geometric, and BLS-LM/SRIF + CN/CN+S, remain accepted; the M3 two_way_range UKF rejection is preserved. |
| FA-02 | confirmed at eb92461/D1; **resolved by P0A metadata correction** — `ONE_WAY_LIGHT_TIME_TOLERANCE_S = 1e-12` / `ONE_WAY_LIGHT_TIME_MAX_ITERATIONS = 10` are now the single source of truth for the seven one-way solver signatures, and `measurement_model_metadata` branches by measurement type (position -> one-way constants; range_rate -> `RangeRatePhysicsConfig` values). |
| Legacy counted-Doppler nonzero delay | **short-term rejection implemented (P0A)** in `RangeRatePhysicsConfig.__post_init__` (`mode='two_way_counted_doppler'` + any nonzero delay -> `ValueError`; +/-0.0 accepted; negative/nonfinite rejected by the pre-existing general validation). The fixed scalar delay term cancels directly in the endpoint RTLT difference, but nonzero delay can still affect counted Doppler through the t2u/t2d separation, spacecraft motion during the delay, and the resulting uplink/downlink event geometry — the four-event counted-Doppler model (CD-4) remains future work. M3 `TwoWayRangeConfig` nonzero-delay support is unchanged. |
| FA-03A | confirmed at eb92461/D1 and the P0A baseline; **resolved by P0B-1**. Validity now requires update and independently evaluated final equation-residual tolerances; raw solvers may return diagnostic failures, while all nominal/sensitivity/Jacobian consumers reject them with controlled, unit-bearing diagnostics. |
| FA-03B | confirmed at eb92461/D1; **resolved by P0B-2**. Shared closed support accepts exact/<=2-policy-ULP endpoint cases and rejects 3+; one-way/counted/UKF production probes are guarded before interpolation; deterministic generation drops and family-local scenario/CSV aggregates report actual deficiencies without automatic history extension. P0B-2E2 additionally corrected the UKF local-envelope demand: the locally built grid now covers the clock-corrected count endpoints (both offset signs) instead of only the raw count window, so nonzero station clock offset/drift no longer produces a spurious guard rejection; the guard, ULP policy, and source-history ownership are unchanged. |

The FA-01/FA-02 D1 current-defect assertions were converted into P0A
safety-contract assertions in
`tests/test_measurement_model_safety_diagnostics.py`; the operator-mismatch
measurement is retained as the rationale evidence motivating the rejection.
One collateral fixture was adjusted:
`tests/test_filters.py::test_two_way_long_arc_noise_clock_and_model_mismatch_with_station_biases`
lost its now-rejected `transponder_delay_s=4e-6` truth-mismatch knob (clock
offset/drift and mu mismatches remain); delay-mismatch campaigns return with
the four-event model.

P0B-1 converted the FA-03A D1 current-defect assertions into strict
nonconvergence safety-contract tests. The focused coverage checks the dual
update/equation-residual criterion, one-way nominal/Jacobian parity, separate
count-start/count-end validation, uplink/downlink diagnostics, and seconds/metres
unit reporting. P0B-2 then converted the FA-03B D1 matrix into strict boundary,
drop/RNG, estimator/observability, UKF preflight, family-local empty-arc, and
reporting tests. P0B-2E2 added clock-corrected UKF local-envelope unit coverage
(zero/positive/negative offsets, corrected-endpoint containment with outward
corrected endpoints as exact anchor nodes, and
source-boundary rejection with clock-sized post-roll), and the long-arc UKF
campaign test's positive station-clock truth mismatch serves as its end-to-end
regression. M3 production behavior remains unchanged.
