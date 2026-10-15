# PHASE 17-HARDEN — PHYSICAL-CONTRACT HARDENING AND SCIENTIFIC INFRASTRUCTURE CLOSURE

> **The question, answered first.**
>
> **Why did the pipeline let impossible measurements look valid?** Physical assumptions lived
> inside individual call sites and were never stated or checked at interfaces:
> - an Earth callable carried no epoch, so an Earth table 504000 s early passed silently;
> - an out-of-range table returned its last sample instead of failing;
> - each ΔDOR / surrogate row builder wrote its own visibility test, and each one tested only elevation.
>
> Every one of these produced plausible numbers, and no oracle compared them to physics.
>
> **What changed.** The failure classes now fail loudly or are fenced off:
> - Earth providers carry and are checked for epoch, frame, centre and units;
> - ephemerides raise outside their coverage;
> - there is one visibility contract (elevation AND lunar occultation), and every row builder defaults to it;
> - the STM is column-major everywhere, and the transposed layout is caught by an FD oracle;
> - superseded physics runs only with an explicit opt-in and warns when it does.
>
> Six historical-defect mutants (A–F) are permanent tests. Each one also demonstrates that the
> original defect is physically wrong. The range-Jacobian discrepancy has a known root cause:
> the variational equations omit the SRP position gradient, a documented approximation. That
> gradient is concentrated in penumbra.
>
> **Verdict:** see §56. **All numbers are `SYNTHETIC_CAMPAIGN_TRUTH`.**

---

## 1. Executive Summary

HARDEN added no new science. It produced:
- a 23-entry defect ledger;
- a 17-entry physical-contract inventory;
- production contract checks at the measurement inputs;
- a single visibility contract with legacy quarantine;
- a path resolver and cross-checkout import isolation;
- 55 new tests (54 in `tests/test_harden_*.py`, 1 in `test_r1od2_ddor_contracts.py`);
- the authoritative Phase 17 status page.

It also found and fixed two defects the brief did not list:
- **HARDEN-D12.** The external campaign module prepends a sibling checkout to `sys.path` before importing `lunar_od`.
- **HARDEN-D04, Earth-LOS extension.** 230 of 590 Earth-LOS surrogate rows (39%) were occulted.

A first attempt put the Earth contract inside the byte-frozen `src/lunar_od/two_way_range.py`. That tripped the
qualified R3-P21-M3 gate, so the module was restored byte for byte and the contract moved one layer up (§41).

## 2. Why These Defects Were Possible

1. **Implicit time origin.** Scenario seconds were passed around without the ET they count from. The Earth
   table counted from `phase7_ET0 − 504000 s`; the Earth rotation counted from `et0`. Nothing compared the two.
2. **Silent coverage handling.** An end-of-table clamp and PCHIP/linear extrapolation returned finite, plausible values.
3. **Duplicated physics.** Visibility existed in production (elevation + occultation). Analysis builders
   re-implemented it as elevation only, and no test compared the two.
4. **Unchecked storage convention.** The STM is column-major, but the reshape order was a per-call-site choice.
5. **No physical oracle at interfaces.** Checks were self-consistency checks (model vs its own Jacobian), which
   pass whatever the geometry is.
6. **Environment leakage.** A workspace `pytest.ini` and a campaign module could each put another checkout's
   `lunar_od` on the import path, so tests could validate the wrong code.

## 3. Defect Ledger

The full ledger is `studies/phase17/reports/phase17_harden_defect_ledger.md` (= `studies/phase17/results/phase17_harden_defects.json`, single
source `studies/phase17/scripts/phase17_harden_ledger.py`). Summary:

| status | IDs |
|---|---|
| fixed (this phase) | D04, D06, D07, D09, D10, D15, D16, D17 |
| fixed earlier, hardened here | D01, D02, D03 |
| quarantined legacy-only | D08, D13 |
| archived | D11 |
| mitigated, residual documented | D12, D14 |
| root cause known, documented approximation | D05 |
| known documented approximation | D23 |
| open follow-up (not on the current-science path, or bounded there) | D18, D19, D20, D21, D22 |

## 4. Root-Cause Pattern

All material defects share one pattern: **a physical assumption held by convention, not by an interface check.**
The repair is the same in each case: attach the assumption to the data (epoch/frame/centre/units tags, coverage
spans, `row_rule`), check it where the data is consumed, and pin it with a test that fails on the historical defect.

## 5. Repository / Instruction State

| field | value |
|---|---|
| branch | `feature/phase17-r-k-srp-estimation` |
| start HEAD / tree | `57ca034a287a5deb177d013d457ae3a76bb89985` / `e812e5e99d42cb3b9ced69d02509ee9fcb7ceb56` |
| main = origin/main | `fea476f81dad07b3914e53eab709e10fa6e9d10b` (unchanged) |
| entering gates | PHASE17_FXR_GATE = PASS; PHASE17_R1O_D2_GATE = PASS |
| instructions | worktree `CLAUDE.md` and workspace `CLAUDE.md` read; `CLAUDE_OPUS_5_5_REPO_INSTRUCTIONS.md` not found |

## 6. Task Checklist / Completion State

`studies/phase17/results/phase17_harden_checklist.md`:
- all 27 REQUIRED items are done;
- 4 RECOMMENDED_FOLLOWUP items and 1 OPTIONAL_CLEANUP item are recorded open.

## 7. Subagents Used and Why

Two read-only agents, within the brief's limit of 3:
1. **Physics-contract audit sweep** (time / ephemeris / frame / visibility search). Its findings became D08, D12–D14 and D18–D22.
2. **FA-06 / R2 non-pass investigation.** Its findings became D07 and D09.

The lead verified every finding against source code and made every shared edit.

## 8. Time Contracts

- **Earth epoch.** `EarthEphemeris.et_origin` and `MoonCenteredEphemeris.et0_s` carry the ET of t = 0.
  `require_ephemeris_contract` raises `EphemerisEpochMismatchError` when the origin differs from the model's `et0`.
- **Windowed arcs.** They keep the absolute epoch (`test_canonical_windowed_arc_keeps_one_epoch`;
  `test_nonzero_arc_start_and_long_arc_keep_the_measurement_epoch`).
- **Event ledgers.** They record local t and absolute ET for every event (§17–§20).

Open, low materiality:
- D20: the legacy `analyze_visibility_gap` path has three time origins. It is unused by Phase 17.
- D21: `earth_position_mci` has an implicit campaign origin.

## 9. Ephemeris Contracts

- **Tagged providers** must match `et0`, J2000, Moon-centred, metres.
- **Legacy providers** (`legacy_reproduction_only`) are accepted with `LegacyEphemerisWarning`.
- **Untagged providers** pass through the tag check but must still give a Moon–Earth distance in [3.40e8, 4.20e8] m.
  Residual D14: an untagged callable with a wrong epoch but a physical distance still passes. Every Phase-17
  science builder passes tagged providers, and a test enforces this.

## 10. Frame Contracts

A frame other than J2000 raises (mutant E). The J2000→ITRF93 transform is orthonormal with det +1, and station
inertial velocity equals d(position)/dt to 1e-5 relative (`test_station_velocity_is_the_time_derivative_of_station_position`).

## 11. Origin Contracts

A centre other than MOON raises (mutant E). The Moon–Earth distance bound catches a wrong-body provider.
Third-body acceleration is direct minus indirect about the Moon, at the correct epoch (§22).

## 12. Unit Contracts

A units tag that is not "m, m/s" raises (mutant F). An untagged km provider fails the distance bound (mutant F).
SRP is linear in K with K in m²/kg (§21).

## 13. Coverage / Interpolation Contracts

- **Fixed:**
  - canonical Earth raises outside [t_min, t_max] (FXR);
  - `MoonCenteredEphemeris` raises beyond a 10 s margin (D15);
  - ΔDOR sensitivity raises when t_tx is outside the grid (D16);
  - the range generator raises when events fall outside the propagated history (pre-existing).
- **Bounded extrapolation in current science:**
  - ΔDOR station B receive up to ~0.03 s past the last node, ~1e-5 m;
  - range t1 uplink, ~1 cm, documented.
- **Open, not on the current-science path:**
  - D18: numba lerp, used only by the UKF sigma propagator;
  - D19: the filters/radiometrics edge helpers.

## 14. Visibility Architecture

`phase17_r1m_core.station_visibility_mask` wraps the production `analyze_visibility_gap_with_transforms`
(elevation ≥ mask AND the station→spacecraft segment is outside the lunar sphere). It is the one visibility contract.
Every builder defaults to `DDOR_ROW_RULE_VISIBLE`:
- `r1o_core.build_ddor_arc`;
- `r1od_campaign.build_ddor_rows`;
- `r1od2_core.build_ddor_rows`;
- `r1o_r_requalification.earth_los_rows`, which uses production `is_occulted_by_moon` from spacecraft to Earth.

`DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY` must be named explicitly and emits `SupersededVisibilityWarning`.
Rule names come from one place (`test_row_rule_names_are_single_sourced`).

## 15. ΔDOR Occultation Repair

Canonical Goldstone–Canberra: 263 elevation-only rows become 170 visible rows. The R1O-D builder's default rows are
bitwise D2's (`test_r1od_builder_default_is_the_d2_visibility_contract`). The visibility-rule builder requires the
MEASUREMENT_EPOCH Earth anchor; FROZEN raises.

## 16. FXR / R1O-D Erratum

0.3204 is `EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`. Authoritative values:

| quantity | value |
|---|---:|
| range | 0.091508 |
| combined | 0.172351 |
| Δ`f_perp` | +0.0808 |
| σ_K/K | 7.46% |

Erratum blocks were added to:
- the FXR erratum;
- the FXR report;
- the R1O-D report;
- the R1O scientific erratum;
- the R1O feasibility report;
- the R1O-R report.

Surrogates re-evaluated under the contract:

| surrogate | before | after | legacy-path reproduction of FXR |
|---|---:|---:|---|
| ΔDOR-like 5 nrad | 0.3203 | 0.1722 | bitwise |
| ΔDOR-like 1 nrad | 0.3619 | 0.1381 | bitwise |
| Earth LOS | 0.0920 | 0.0919 | bitwise |

The visibility-correct surrogate differs from production by −1.7e-4 in `f_perp`
(`studies/phase17/results/phase17_harden_visibility_requalification.json`).

## 17. Range Event Audit

`studies/phase17/results/phase17_harden_range_event_ledger.json` records t1/t2u/t2d/t3, local and absolute ET, and the
equation residuals. Receive tags whose chain precedes the grid are dropped, not extrapolated.

## 18. Doppler Event Audit

`studies/phase17/results/phase17_harden_doppler_event_ledger.json` records the count-interval events. Counted Doppler
interpolates the Earth linearly (documented design R3-P22). That gives 2.62 m at mid-nodes and a ~1.5e-6 m/s count-rate error (D23).

> **Erratum (2026-10):** the ~1.5e-6 m/s count-rate figure differenced only the symmetric receive endpoints. The
> asymmetric uplink transmit epochs give a 1.67–1.83e-3 m/s value offset (CD1 closure §49.1). D23 is **FIXED** by
> Phase 17-HARDEN-D23 (cubic-Hermite Earth): see `studies/phase17/reports/phase17_harden_d23_repair_qualification.md`.

## 19. ΔDOR Event Audit

`studies/phase17/results/phase17_harden_ddor_event_ledger.json` records the common transmit event. Both light-time equations
hold to < 1e-12 s (`test_common_transmit_event_satisfies_both_light_time_equations`).

## 20. Optical Event Audit

`studies/phase17/results/phase17_harden_optical_event_ledger.json`. Optical rows are Earth-independent (bitwise on both fixtures, FXR).

## 21. SRP / Shadow Audit

- ν ∈ [0, 1], monotone and continuous through penumbra (max step < 0.05 on a 100 m scan).
- Umbra gives zero force.
- The acceleration is exactly K × kernel, so K = 0 gives zero.

(`test_shadow_entry_exit_is_continuous_bounded_and_monotone`, `test_srp_is_linear_in_k_and_vanishes_at_k_zero`)

## 22. Third-Body Audit

`third_body_acceleration` equals the difference of two point-mass accelerations to 1e-12, and agrees with the tidal
approximation to within 5% at lunar orbit radii. The Earth used is SPICE at the arc epoch, and |r_E| ∈ [3.40e8, 4.20e8] m.

## 23. STM / Sensitivity Audit

- **Production and analysis:** 25 reshape sites, all `order="F"`.
- **Deliberate C order:** 3 sites, all labelled defect-reproduction oracles (`r1o_r_requalification`, `r1o_r_stm_oracle`, tests).
- **Checks:** Φ matches FD while Φᵀ fails (mutant D; `test_r1o_stm_layout.py`). The optical K-sensitivity requires a 6×6 Φ (D17).

## 24. Range Jacobian Root Cause

**Root cause:** `dynamics_jacobian_a_matrix` builds A from gravity gradients only. The SRP position gradient
K·∂g/∂r is omitted, which is the documented Phase 13/16 approximation (mean 4.02e-13 of the gravity gradient).
The gradient is not uniform:
- penumbra maximum: 1.09e-11 s⁻²;
- median: 7.7e-19 s⁻².

Its effect on Φ accumulates secularly.

**Evidence from interventions** (`studies/phase17/results/phase17_harden_range_jacobian_decomposition.json`, 4 orbits, rtol 1e-13, h = 10 m / 0.01 m/s):
- **Event derivative.** The light-time/geometry chain evaluated on re-propagated states (LIN) agrees with the
  analytic event derivative to 1e-8 – 9e-8, and the error falls with h. The event derivative is exact.
- **Penumbra STM error.** With SRP and conical penumbra, the error is 2.3e-6 – 1.0e-5. Turning the shadow off
  (NO_SHADOW), or setting K = 0, drops it to 3e-8 – 5e-7, which is the integrator/FD floor. Switching off the
  mechanism removes the discrepancy.
- **First-order prediction.** δΦ = Φ ∫ Φ⁻¹ B Φ dτ with B = [[0,0],[G_SRP,0]], using the production kernel, reproduces
  the observed error. At dt = 1 s: correlation 1.000 / 0.999 / 1.000 / 0.999 on x0 / z0 / vy0 / vz0, and the vx0
  residual (2.2e-7) is at the floor. With the FD oracle at rtol 1e-14 two columns fit only partly (y0 0.908, vx0 0.737). That residual was
tested, not assumed (`studies/phase17/results/phase17_harden_range_jacobian_oracle_convergence.json`):
- **Quadrature is not the cause.** Refining the prediction's quadrature to dt = 0.25 s and 0.1 s leaves the y0 residual at 9.9e-7.
- **Oracle tolerance is the cause.** Tightening the oracle's integrator tolerance removes it. The y0 residual goes
  9.5e-6 (rtol 1e-13) → 1.0e-6 (1e-14) → 3.8e-8 (3e-15), i.e. it scales with rtol.

At rtol 3e-15, **all six columns** match the prediction (correlation 0.986–1.000, residual 2.7e-8 – 1.0e-7).
The earlier residual was the oracle's own integration error at the non-smooth (penumbra-kinked) right-hand side.
The discrepancy is fully attributed.

## 25. Range Jacobian Closure

**Closure outcome:** *mathematically understood approximation identified and bounded* (brief §41).
- **Bound:** 1e-5 relative over 4 orbits at rtol 1e-13, growing with the number of penumbra passes. This is consistent
  with the 1e-6 to 3e-5 R1O-D2 figure for 15-orbit arcs.
- **Not repaired in HARDEN:** `dynamics.py` is R2-protected, and adding the gradient changes the force model's variational content.
- **Follow-up:** F1.
- **Regression tests:** `test_omitted_srp_position_gradient_is_the_range_jacobian_root_cause`,
  `test_range_jacobian_root_cause_evidence_is_consistent`.

## 26. ΔDOR Derivative Qualification

Against re-propagation at rtol 1e-13, best step (`studies/phase17/results/phase17_harden_derivatives.json`):
- state columns ≤ 1.2e-6;
- K 1.1e-4, at the K FD floor (R1O-D2 refined it with an rtol 1e-14 oracle).

## 27. Doppler Derivative Qualification

Counted Doppler: state columns ≤ 1.3e-6; K 2.2e-4.

## 28. Optical Derivative Qualification

Optical, with attitude frozen: state columns ≤ 4.2e-6; K 3.2e-4.

## 29. Legacy Path Quarantine

- **Opt-in required.** 11 historical scripts raise `LegacyReproductionRequired` unless `PHASE17_LEGACY_REPRODUCTION=1`
  (parametrised test).
- **Superseded fixtures warn.** The legacy Earth fixture and the legacy row rule warn when used.
- **Always warns:** `bridge_reduced_rows`.
- **No current-science leak:** D2's `install_legacy_earth_guard` makes the campaign Earth table raise.

## 30. Orphan Script Disposition

The two dead R1O-D scripts are archived byte for byte in `archive/phase17_orphans/` with a README and
`MANIFEST.sha256`. The LF-normalised hashes match the hygiene report. A test proves they are gone from `examples/`.

## 31. External Input / Path Hardening

- **Resolver.** `src/lunar_od/project_paths.py` provides `repo_root`, `artifacts_dir`, `project_root`, `campaign_root`,
  `gravity_data_root` and `ephemeris_mat`, overridable through `LUNAR_OD_*` environment variables.
- **Literals.** Zero machine-specific literals remain in tracked `.py` files outside `archive/`.
- **Campaign import.** `import_campaign_module()` neutralises the external module's sibling-checkout `sys.path` injection (D12).

## 32. Historical-Defect Mutant Tests

`tests/test_harden_mutants.py`:

| mutant | caught | would it have failed before the repair? |
|---|---|---|
| A epoch −504000 s | EphemerisEpochMismatchError via the checked generator | YES: the SPICE oracle shows the Earth wrong by > 1e7 m, and the old path accepted it |
| B clamp | coverage errors (canonical and MoonCenteredEphemeris) | YES: the clamped table holds the Earth still while SPICE moves > 100 km in 1 h |
| C occultation | 0 far-side rows from all three builders; the zero-Moon-radius mutant makes the far side visible | YES: the elevation-only rule (the historical code) forms far-side rows in all three builders |
| D STM order | the C-order Φ fails FD by > 1e-2 relative | YES (the R1O defect) |
| E frame / centre | EphemerisContractError | YES: no metadata check existed |
| F units | EphemerisContractError (tag) and the distance bound (untagged km) | YES |

## 33. Boundary Tests

`tests/test_harden_boundaries.py` covers:
- coverage edges (±1e-6 s);
- elevation mask at 10° ± 1e-4°;
- lunar limb at ±50 m and ±2 km against an independent segment–sphere oracle;
- shadow entry and exit;
- nonzero arc start;
- light-time convergence.

## 34. Long-Arc Test

A 15-orbit canonical Earth matches SPICE to < 1 mm at the start, middle and end, and raises beyond.

## 35. Windowed-Arc Test

`test_canonical_windowed_arc_keeps_one_epoch` (FXR). A 5000 s start offset keeps the absolute epoch (HARDEN).

## 36. Cross-Measurement Consistency

- **At grid nodes:** range, Doppler and ΔDOR station states are identical (0.0).
- **At mid-nodes, range vs Doppler:** they differ by 2.62 m, which is exactly the linear-Earth interpolation error of counted Doppler.
- **At mid-nodes, range vs ΔDOR:** they differ by 2.5e-5 m.
- **Orientation transform and spacecraft nominal:** shared.

## 37. Randomized Contract Stress

With seed 20260926, 4000 random segments (≥ 3500 outside the guard band) were tested. Production
`is_occulted_by_moon` agrees with the independent oracle on every one of them. The third-body check also used 20 random states.

## 38. Existing Test Failure Investigation

The entering full suite had 2 failed and 10 errors:
- `test_fa06_pytest_session_imports_isolated_repository`: the workspace `pytest.ini` pythonpath pointed at a sibling checkout (D09).
- `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change`: a stale `dynamics.py` hash (D07).
- 10 `test_r2_measurement_fidelity.py` fixture errors: cascaded from the same gate.

All were resolved in cbc3af6.

## 39. Cross-Worktree Isolation

- **Configuration:** a worktree-local `pytest.ini` (no pythonpath) and `tests/conftest.py`, which pins ROOT and `desktop_app`.
- **Regression:** `tests/test_harden_import_isolation.py`. It fails under `-c ../pytest.ini`, and it caught D12.
- **Result:** **PASS**.

## 40. R2 Protection-Gate Resolution

A documented re-freeze, `PHASE17_HARDEN_REFROZEN_CURRENT_TREE_SHA256`:
- **old hash:** d0f7ab02 (fc13cc4);
- **new hash:** b369899a;
- **authorising commit:** 0e785f6 (qualified SRP-gradient documentation), blob c973822f.

The R3-P21-M3 gate (`two_way_range.py`) was **not** re-frozen. HARDEN is not authorised to re-freeze it, so the module was restored instead (§41).

## 41. Production Changes

| file | change | root cause / regression |
|---|---|---|
| `src/lunar_od/ephemeris.py` | contract API; tagged `MoonCenteredEphemeris`; coverage errors | D02, D03, D15 / mutants A, B, E, F |
| `src/lunar_od/measurement_contracts.py` (new) | checked range generator | D02, D14 / mutants A, F; structural test |
| `src/lunar_od/scenarios.py` | uses the checked generator | D14 |
| `src/lunar_od/delta_dor.py` | t_tx grid bound | D16 |
| `src/lunar_od/lunar_landmark_optical.py` | 6×6 Φ check; docstring | D17, D06 |
| `src/lunar_od/project_paths.py` (new) | path resolver; guarded campaign import | D10, D12 |
| `src/lunar_od/two_way_range.py` | **net unchanged**: edited in 2e8a53d, restored byte for byte in e2fbfdc | P21-M3 gate |

Before/after evidence: no committed science number changed. The legacy path reproduces FXR bitwise, D2's contract
tests are unchanged, and the full suite results are in §51.

## 42. Analysis Changes

- `phase17_r1m_core`: visibility contract, legacy opt-in, guarded campaign import.
- Row builders: `row_rule`.
- FXR requalification: names the legacy rule.
- 11 gated scripts.
- Path fixes.
- Evidence scripts: `phase17_harden_{range_jacobian, events_and_derivatives, visibility_requalification, ledger}.py`.

## 43. Test Changes

New:
- `test_harden_mutants.py` (31 tests);
- `test_harden_boundaries.py` (17);
- `test_harden_import_isolation.py`.

Updated:
- `test_r1od2_ddor_contracts.py` (explicit legacy rule + default-contract test);
- `test_r2_measurement_fidelity.py`;
- the path resolver in four tests.

## 44. Documentation Changes

- `studies/phase17/reports/phase17_authoritative_status.md`
- the defect ledger
- the contracts CSV
- six errata blocks
- the hygiene-report resolution note
- the archive README
- this report

## 45. What Is Now Impossible to Do Silently

- Hand the range model a tagged Earth with a different epoch, frame, centre or units.
- Hand it an Earth at the wrong distance (km, wrong body).
- Query a canonical or production ephemeris outside its coverage.
- Form a ΔDOR, surrogate or Earth-LOS row behind the Moon without naming the legacy rule and receiving a warning.
- Run a superseded historical script.
- Import `lunar_od` from another checkout in the test session or through the campaign module.
- Use a transposed STM in a tested chain.
- Call the unchecked range generator from `scenarios.py` or Phase-17 code.

## 46. Explicit Legacy Reproduction Paths

- `PHASE17_LEGACY_REPRODUCTION=1 python examples/<historical script>.py`
- `row_rule=DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY`
- `EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED`
- `studies/phase17/scripts/phase17_fxr_requalification.py`, which reproduces FXR bitwise.

## 47. Remaining Scientific Limitations

- Simplified dynamics: no Earth or Sun third-body gravity in the campaign truth.
- `SYNTHETIC_CAMPAIGN_TRUTH`.
- The ΔDOR gain is geometry-dependent (D2).
- The STM omits the SRP position gradient (D05).
- Counted Doppler uses a linear Earth (D23).

## 48. Remaining Technical Debt

- F1–F4 and O1: D05, D14, D18–D23.
- The public `lunar_od.generate_two_way_range_measurements` export stays unchecked, for the frozen module's own unit tests.

## 49. Remaining Unknowns

**No material physical-contract unknown remains.** Every discrepancy measured in this phase has an identified cause:
- range Jacobian (D05): the omitted SRP gradient, plus oracle integration error, both demonstrated;
- mid-node range-vs-Doppler (D23): linear Earth interpolation, matched to 1e-6 m;
- surrogate-vs-production ΔDOR (−1.7e-4 in `f_perp`): a different observable model on identical rows.

Known **residual risks** (not unknowns):
- D14: an untagged Earth callable with a wrong epoch but a physical distance passes. No current-science path does this, and it is tested.
- D12: the external campaign module still contains its `sys.path` injection. It is neutralised when imported through the guarded loader.
- D18, D19: extrapolation helpers outside, or bounded on, the current-science path.

## 50. Targeted Regression Results

| batch | result |
|---|---|
| production-contract batch | 156 passed, 9 skipped |
| D2/FXR contracts | 18 passed (then 24 with isolation) |
| affected-module batch | 374 passed |
| mutants | 29 passed, then 31 |
| boundaries | 17 passed |
| contract-layer batch (mutants, od_contracts, scenarios, scenario_config, two_way_range*, D2, FXR) | 159 passed, 9 skipped |

## 51. Full Suite Results

Final run: `python -m pytest -o addopts="" -q -rs`, from the worktree, on commit 8 of this phase.

| collected | passed | failed | errors | skipped | subtests |
|---:|---:|---:|---:|---:|---:|
| 1391 | **1362** | **0** | **0** | 29 | 146 passed |

Every skip is an opt-in or data-availability skip:
- 20 opt-in slow regressions (`LUNAR_OD_RUN_SLOW_TESTS`);
- 7 missing optional GRGM660PRIM data;
- 1 missing Phase 13G store;
- 1 bounded slow slice.

The entering suite had 2 failed and 10 errors (§38).

Intermediate full runs found two failures, each of which was fixed and then re-run:
- R3-P21-M3 byte gate: `two_way_range.py` restored (§41).
- Structural guard matched a string literal: made AST-based.

## 52. Final Diff Review

Scope: `git diff 57ca034..HEAD`, 76+ files.
- **`lunar_od/`:** 6 files.
  - 2 are new (`measurement_contracts.py`, `project_paths.py`).
  - The other 4 carry changes each tied to a ledger entry (§41).
  - `two_way_range.py` has a net-zero diff.
- **`dynamics.py` and `accelerated.py`:** untouched (R2-protected).
- **Analysis files:**
  - visibility / row-rule plumbing;
  - legacy opt-in gates;
  - path replacements;
  - evidence scripts.
- **Tests:** 3 new files; small targeted edits in 6 existing ones.
- **Docs:** errata blocks, ledger, status, this report.
- **Archive:** 2 orphans plus README and manifest.
- **Artifacts:** new `phase17_harden_*` evidence only. No committed science artifact was regenerated or altered.

No convenience refactors and no unrelated files. A control-character scan of every changed file came back clean.
**FINAL_DIFF_REVIEW = PASS.**

## 53. Phase-17 Authoritative Status

See `studies/phase17/reports/phase17_authoritative_status.md`. The current ΔDOR number is **0.172351**, not 0.3204.

## 54. What This Phase Actually Established

- Every historical Phase 17 failure class is pinned by a test that fails on the original defect.
- There is one visibility contract.
- Metadata-checked Earth inputs guard the range model.
- Coverage fails loudly.
- Test sessions import only this checkout.
- The range-Jacobian discrepancy is fully attributed to a documented, bounded approximation.

## 55. What This Phase Did NOT Establish

- Any new science.
- That the SRP-gradient approximation is acceptable for every future arc: it grows with penumbra passes.
- Protection for callers outside `scenarios.py` and Phase-17 code that pass untagged Earth callables with wrong epochs (D14).
- Anything about the external campaign module, which was not modified.

## 56. Final Closure Verdict

Every strict PASS criterion (§71) is met:

| criterion | evidence |
|---|---|
| material defects repaired, or fenced with documented residual | ledger |
| Earth-epoch mixing cannot pass silently | mutant A; tagged providers; checked generator |
| no silent clamp | mutant B; coverage errors |
| no occulted ΔDOR rows | mutant C, all builders |
| no wrong STM layout | mutant D; audit |
| range Jacobian root cause known and closed | §24–25 |
| derivatives qualified | §26–28 |
| contracts consistent | §8–13 |
| windowed arcs keep absolute time | §35 |
| no legacy leak | §29 |
| mutants catch the old failure classes | §32 |
| import isolation | §39 |
| all non-passes understood and resolved | §38, §51 |
| documentation current | §53 |
| final diff clean | §52 |

**PHASE17_HARDEN_GATE = PASS.**

## 57. Whether New Science Is Safe

**Yes, within the documented scope.** The infrastructure is **QUALIFIED_FOR_NEW_SCIENCE**. New science must:
- use the checked generator and the visibility contract (the defaults);
- cite `studies/phase17/reports/phase17_authoritative_status.md`;
- treat D05 (the SRP-gradient approximation) as a known, bounded model limit for penumbra-rich arcs.

## 58. Exact Next Action

**Stop.** HARDEN is complete (hard stop, brief §73). The next scientific phase needs its own brief and authorisation.
Recommended before any arc that crosses many penumbrae: follow-up F1, the SRP position gradient in the variational
equations. It needs authorisation because `dynamics.py` is R2-protected.

---

## Final fields

```text
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD = 57ca034a287a5deb177d013d457ae3a76bb89985
START_TREE = e812e5e99d42cb3b9ced69d02509ee9fcb7ceb56

FINAL_HEAD = the commit "docs(harden): closure report ..." (git log -1 on this branch)
FINAL_TREE = tree of FINAL_HEAD

MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b
ORIGIN_MAIN_HEAD = fea476f81dad07b3914e53eab709e10fa6e9d10b

HARDEN_INPUT_GATE = PASS (PHASE17_FXR_GATE = PASS, PHASE17_R1O_D2_GATE = PASS)

CLAUDE_REPO_INSTRUCTIONS_READ = YES (worktree + workspace CLAUDE.md; CLAUDE_OPUS_5_5_REPO_INSTRUCTIONS.md NOT FOUND)
PERSISTENT_TASK_CHECKLIST_USED = YES (studies/phase17/results/phase17_harden_checklist.md)

SUBAGENTS_USED = 2
SUBAGENT_LIST = physics-contract audit sweep (read-only); FA-06/R2 nonpass investigation (read-only)
SUBAGENT_RESULTS_RECONCILED = YES

DEFECT_LEDGER_COMPLETE = YES (23 entries)

TIME_CONTRACT_GATE = PASS
EPHEMERIS_CONTRACT_GATE = PASS
FRAME_CONTRACT_GATE = PASS
ORIGIN_CONTRACT_GATE = PASS
UNIT_CONTRACT_GATE = PASS
COVERAGE_CONTRACT_GATE = PASS

COMMON_VISIBILITY_GATE = PASS
DDOR_OCCULTATION_GATE = PASS

RANGE_EVENT_GATE = PASS
DOPPLER_EVENT_GATE = PASS
DDOR_EVENT_GATE = PASS
OPTICAL_EVENT_GATE = PASS

SRP_SHADOW_GATE = PASS
THIRD_BODY_GATE = PASS
STM_SENSITIVITY_GATE = PASS

RANGE_JACOBIAN_ROOT_CAUSE_KNOWN = YES
RANGE_JACOBIAN_GATE = PASS (mathematically understood approximation, identified and bounded)

DDOR_JACOBIAN_GATE = PASS
DOPPLER_JACOBIAN_GATE = PASS
OPTICAL_JACOBIAN_GATE = PASS

ACTIVE_SCIENCE_LEGACY_LEAK_GATE = PASS

EARTH_EPOCH_MUTANT_CAUGHT = YES
EARTH_CLAMP_MUTANT_CAUGHT = YES
DDOR_OCCULTATION_MUTANT_CAUGHT = YES
STM_ORDER_MUTANT_CAUGHT = YES
FRAME_MUTANT_CAUGHT = YES
UNIT_MUTANT_CAUGHT = YES

LONG_ARC_GATE = PASS
WINDOWED_ARC_GATE = PASS
CROSS_MEASUREMENT_GEOMETRY_GATE = PASS

ORPHAN_SCRIPT_STATUS = ARCHIVED (archive/phase17_orphans, README + SHA-256 manifest)
HARD_CODED_PATH_STATUS = RESOLVED (0 literals in tracked .py outside archive/)

CROSS_WORKTREE_IMPORT_ISOLATION = PASS
R2_PROTECTION_GATE_STATUS = PASS (documented re-freeze d0f7ab02 -> b369899a, authorised by 0e785f6; P21-M3 gate untouched)

MATERIAL_DEFECTS_FOUND = 17 (D01-D17)
MATERIAL_DEFECTS_REPAIRED = 17 (fixed, fixed-earlier-and-hardened, quarantined, archived, mitigated, or closed as a bounded approximation; residuals documented for D05, D12, D14)
MATERIAL_DEFECTS_DEFERRED = 0

LOW_MATERIALITY_DEFECTS_FOUND = 6 (D18-D23)
LOW_MATERIALITY_DEFECTS_REPAIRED = 0 (documented follow-ups; none on, or all bounded on, the current-science path)

UNKNOWN_PHYSICAL_CONTRACT_ISSUES = 0

PRODUCTION_CODE_CHANGED = YES (ephemeris, delta_dor, lunar_landmark_optical, scenarios; new measurement_contracts, project_paths; two_way_range net-zero)
ANALYSIS_CODE_CHANGED = YES
TEST_CODE_CHANGED = YES
DOCUMENTATION_CHANGED = YES

TOTAL_TESTS_COLLECTED = 1391
TESTS_PASSED = 1362
TESTS_FAILED = 0
TESTS_ERRORS = 0
TESTS_SKIPPED = 29

NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0

FINAL_DIFF_REVIEW = PASS

PHASE17_HARDEN_GATE = PASS
PHASE17_INFRASTRUCTURE_STATUS = QUALIFIED_FOR_NEW_SCIENCE
NEW_SCIENCE_SAFE_TO_START = YES

COMMITS_CREATED = 10
COMMIT_LIST = cbc3af6, 2e8a53d, 91c7a9d, af1ac3c, 4296eb8, 5e4ba7e, e2fbfdc, f624f23, 89452e8, <closure-report commit>

PUSH = NONE
MERGE = NONE
MAIN_CHANGED = NO
ORIGIN_MAIN_CHANGED = NO
```
