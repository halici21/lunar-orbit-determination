# D1 Measurement-Model Safety Diagnostics

Status: D1-only executable validation report, 2026-07-13.

This report covers FA-01, FA-02, FA-03A, FA-03B, and FA-06 only. It does not
contain a production fix, and it does not execute the D2 frame, PCHIP,
visibility, or lunar-frame campaigns.

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

Added `tests/test_measurement_model_safety_diagnostics.py` with seven passing
tests:

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

## 7. FA-03A evidence

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

**Classification:** FA-03A confirmed. The strictness split is narrower than a
generic "all sensitivity helpers are strict" statement: the one-way Jacobian
rejects nonconvergence, while the legacy counted Jacobian currently consumes
the nonconverged solution too.

**Recommended fix:** enforce convergence and a separately evaluated equation
residual in both nominal and derivative paths; never publish the last iterate
as an observable.

## 8. FA-03B history-domain matrix

**Source symbols:** `radiometrics.interp_state_history`, `_interp_state`,
`_interp_vector`, `_interp_matrix`; `measurements._apparent_position_observable`;
`filters._two_way_local_histories`, `_interp_pass_values`.

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

No tested legacy path raised a controlled history-domain exception, recorded a
drop, or persisted extrapolation metadata.

**Classification:** FA-03B confirmed.

**Recommended fix:** define required pre-roll/post-roll per observable, reject
unsupported spacecraft/event epochs before solving, and persist structured
failure/drop metadata. Preserve exact-boundary acceptance explicitly.

## 9. Findings confirmed

| Issue | Result |
|---|---|
| FA-01 | Confirmed: accepted UKF profiles use geometric position physics |
| FA-02 | Confirmed: one-way metadata reports range-rate solver defaults |
| FA-03A | Confirmed: nonconverged last iterates reach nominal observable paths |
| FA-03B | Confirmed: legacy paths silently extrapolate outside history support |
| FA-06 | Confirmed: parent pytest config requires explicit provenance override |

All defect evidence is expressed as passing tests that assert current behavior.
No test is intentionally failing or marked xfail.

## 10. Findings reclassified

No issue was dismissed. Two claims were refined:

1. FA-03A derivative behavior is model-specific: one-way is strict; legacy
   counted nominal and Jacobian paths are both permissive.
2. FA-03B one-way exact-upper-event evaluation still performs an earlier
   out-of-domain receive-epoch probe, so its complete path is not boundary-only.

The full-suite `from app` collection failure is classified as an FA-06 test
harness dependency exposed by clearing the parent config, not as a production
measurement defect.

## 11. Recommended production fixes

Apply in separate production patches, with these D1 tests converted from
current-defect assertions to safety-contract assertions:

1. **P0.1:** reject UKF with CN/CN+S position profiles.
2. **P0.2:** enforce convergence plus equation-residual closure for one-way and
   counted nominal/Jacobian paths.
3. **P0.3:** prohibit silent event-history extrapolation and add explicit
   pre-roll/domain metadata.
4. **P0.4:** source one-way metadata from the actual one-way solver policy.

Separately, reject legacy counted Doppler when `transponder_delay_s != 0`
until a four-event model exists. A fixed scalar delay cancels directly in the
endpoint round-trip-light-time difference, but nonzero delay can still change
counted Doppler indirectly through separate `t2u`/`t2d` spacecraft states and
uplink/downlink event geometry.

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

Every pytest command included `--override-ini=pythonpath=`.

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

## 14. P0A resolution status (post-D1 production patch)

The P0A measurement-safety patch on this branch changed the status of the
D1 findings as follows; the D1 sections above are preserved as the
pre-patch evidence record.

| Finding | Status after P0A |
|---|---|
| FA-01 | confirmed at eb92461/D1; **resolved by P0A hard rejection** — shared `filters.validate_ukf_measurement_support` enforced at `scenario_config._validate_cross_field_rules` (loader) and at `run_lunar_ukf` entry (runtime defense-in-depth for direct `ScenarioConfig` construction). UKF + geometric, and BLS-LM/SRIF + CN/CN+S, remain accepted; the M3 two_way_range UKF rejection is preserved. |
| FA-02 | confirmed at eb92461/D1; **resolved by P0A metadata correction** — `ONE_WAY_LIGHT_TIME_TOLERANCE_S = 1e-12` / `ONE_WAY_LIGHT_TIME_MAX_ITERATIONS = 10` are now the single source of truth for the seven one-way solver signatures, and `measurement_model_metadata` branches by measurement type (position -> one-way constants; range_rate -> `RangeRatePhysicsConfig` values). |
| Legacy counted-Doppler nonzero delay | **short-term rejection implemented (P0A)** in `RangeRatePhysicsConfig.__post_init__` (`mode='two_way_counted_doppler'` + any nonzero delay -> `ValueError`; +/-0.0 accepted; negative/nonfinite rejected by the pre-existing general validation). The fixed scalar delay term cancels directly in the endpoint RTLT difference, but nonzero delay can still affect counted Doppler through the t2u/t2d separation, spacecraft motion during the delay, and the resulting uplink/downlink event geometry — the four-event counted-Doppler model (CD-4) remains future work. M3 `TwoWayRangeConfig` nonzero-delay support is unchanged. |
| FA-03A | **open** — the D1 current-defect tests in Section 7 remain passing evidence until the P0B enforcement patch. |
| FA-03B | **open** — the D1 history-domain matrix in Section 8 remains passing evidence until the P0B enforcement patch. |

The FA-01/FA-02 D1 current-defect assertions were converted into P0A
safety-contract assertions in
`tests/test_measurement_model_safety_diagnostics.py`; the operator-mismatch
measurement is retained as the rationale evidence motivating the rejection.
One collateral fixture was adjusted:
`tests/test_filters.py::test_two_way_long_arc_noise_clock_and_model_mismatch_with_station_biases`
lost its now-rejected `transponder_delay_s=4e-6` truth-mismatch knob (clock
offset/drift and mu mismatches remain); delay-mismatch campaigns return with
the four-event model.
