# Doppler / Range-Rate Model Review

Review of the range-rate / two-way counted-Doppler implementation against standard
OD and DSN formulations (Moyer, Thornton & Border, Tapley/Schutz/Born,
Montenbruck & Gill, Vallado, DSN 810-005). All file/line references are to the
current tree.

---

## 1. Summary of what the code currently implements

Two physically distinct range-rate observables, selected by
`RangeRatePhysicsConfig.mode` ([radiometrics.py](lunar_od/radiometrics.py)):

### `geometric_instantaneous`
- `instantaneous_geometric_range_rate(r_rel, v_rel)` ([radiometrics.py:101](lunar_od/radiometrics.py#L101))
  returns `ρ̇ = (ρ⃗·ρ̇⃗)/|ρ⃗|`.
- Built in the **ECEF (ITRF93) rotating frame**: the spacecraft 6-state is rotated
  to ECEF with the 6×6 `sxform`, and `ρ̇⃗ = v_sat_ecef` because the station is fixed
  in ECEF (`v_station_ecef = 0`). Earth-rotation transport of the station is thus
  captured exactly ([measurements.py:524-538](lunar_od/measurements.py#L524-L538)).
- This is the **one-way instantaneous** line-of-sight range-rate — the simplified
  textbook reference model.

### `two_way_counted_doppler`
- `two_way_counted_doppler_observable(...)` ([radiometrics.py:111](lunar_od/radiometrics.py#L111)):
  evaluates the **round-trip light-time at the two count-interval endpoints**
  `t_mid ± Tc/2` and forms the **averaged differenced range**
  `ρ_rate = [τ_RT(t_end) − τ_RT(t_start)] / Tc`. It is **not** an instantaneous
  derivative.
- Outputs either a coherent Doppler frequency
  `f_D = turnaround_ratio · f_uplink · ρ_rate` (`output_unit="hz"`) or its
  m/s-equivalent `c·ρ_rate/2` (`output_unit="mps_equivalent"`, the default OD
  observable).
- `solve_two_way_light_time(...)` ([radiometrics.py:287](lunar_od/radiometrics.py#L287))
  solves a genuine **three-epoch** problem with **separately converged uplink and
  downlink legs**:
  - downlink `t3 → t2`: `t2 = t3 − |r_sc(t2) − r_st(t3)|/c` (iterated),
  - uplink `t2 → t1`: `t1 = t2 − transponder_delay − |r_sc(t2) − r_st(t1)|/c` (iterated).
  The station is evaluated at the **transmit time t1** and the **receive time t3**
  (station motion during the light time is included), and the spacecraft is
  interpolated at the **bounce time t2**.
- **Coherent turnaround ratio**, **constant uplink frequency**, **transponder
  delay**, and a **station clock model** (constant offset + linear drift, applied
  to the receive endpoints via `_clock_corrected_receive_time`,
  [radiometrics.py:381](lunar_od/radiometrics.py#L381)) are all included.
- A `local_state_model` switch (`"ode"` default, or `"taylor3"` for `Tc ≤ 60 s`)
  controls the local spacecraft-state model around the bounce.

### Analytic Jacobians
- geometric: local analytic 4×6 `H̃` ([measurements.py:777](lunar_od/measurements.py#L777))
  mapped to the arc initial state through the STM.
- two-way: the range-rate row is replaced by the **analytic counted-Doppler
  initial-state partials** `two_way_counted_doppler_initial_state_jacobian`
  ([radiometrics.py:164](lunar_od/radiometrics.py#L164)), which differences the
  per-endpoint `round_trip_light_time_initial_state_jacobian`
  ([radiometrics.py:209](lunar_od/radiometrics.py#L209)) — this carries the
  count-interval endpoints, the implicit uplink/downlink light-time partials
  (`dt1_dx`, `dt2_dx`), the bounce-time dependence, and the STM. Wired in the
  estimators at [estimators.py:984-1008](lunar_od/estimators.py#L984-L1008). A
  finite-difference Jacobian is kept as a validation reference.

---

## 2. Mapping between code modes and literature models

| Code mode | Literature model | Notes |
|---|---|---|
| `geometric_instantaneous` | Instantaneous LOS range-rate `ρ̇ = (ρ⃗·ρ̇⃗)/\|ρ⃗\|` — Tapley/Schutz/Born Ch. 3, Vallado, Montenbruck & Gill Ch. 6 | Exact textbook form; one-way, single epoch. Correct as a simplified reference observable. |
| `two_way_counted_doppler` (`mps_equivalent`) | Differenced-range (averaged) two-way Doppler — Moyer (DESCANSO Mon. 2), Thornton & Border (DESCANSO Mon. 1) | Endpoint-differenced round-trip range over `Tc`; the operational counted-Doppler idea, expressed as a range-rate equivalent. |
| `two_way_counted_doppler` (`hz`) | Two-way coherent Doppler frequency `f_D = G·f_up·ρ_rate` — Moyer; DSN 810-005 (turnaround ratios) | Coherent frequency observable; turnaround ratio `G` configurable (default X-band 880/749). |

The two-way mode follows the correct **conceptual** Moyer construction (count over
`Tc`, coherent turnaround, three distinct epochs, separate light-time legs). It is
a **controlled / simplified** DSN model: it omits the media, relativistic, and
station-displacement terms of a full operational formulation (Section 3).

---

## 3. Implemented vs missing physical effects

Status key: **Implemented** / **Partial** / **Missing** / **N/A**.

### Propagation medium
| Effect | Status | Note |
|---|---|---|
| Tropospheric delay | Missing | not modeled |
| Ionospheric delay | Missing | not modeled |
| Solar plasma delay | Missing | not modeled |

### Relativity
| Effect | Status | Note |
|---|---|---|
| Shapiro (gravitational) delay | Missing | — |
| Relativistic time transformations (TDB/TT/proper time) | Partial | SPICE ET (≈TDB) is used for frame epochs; per-station proper-time/clock-rate relativity not modeled |
| Gravitational frequency shift | Missing | — |
| Special-relativistic Doppler terms | Missing | Newtonian differenced-range only |

### Station modeling
| Effect | Status | Note |
|---|---|---|
| Earth orientation (precession/nutation/spin) | Implemented | via SPICE `sxform("J2000","ITRF93")` (binary Earth PCK) |
| Polar motion | Implemented | included in the SPICE ITRF93 high-precision Earth PCK |
| UT1/UTC | Implemented | handled inside the SPICE ET → ITRF93 transform |
| Solid-Earth tides | Missing | station is a fixed ITRF93 point (`r_ecef_m` constant) |
| Ocean loading | Missing | as above |
| Pole tide | Missing | as above |

### Instrumentation
| Effect | Status | Note |
|---|---|---|
| Transponder delay | Implemented | between uplink/downlink legs ([radiometrics.py:328](lunar_od/radiometrics.py#L328)) |
| Station clock offset | Implemented | constant offset on receive endpoints |
| Station clock drift | Implemented | linear drift about a reference time |
| Frequency ramping / ramp tables | Missing | constant uplink frequency assumed |
| Count time-tagging convention | Partial | receive-time-tagged, symmetric `± Tc/2` endpoints; ramp/agency-specific tagging not modeled |

### Measurement type
| Observable | Status | Note |
|---|---|---|
| Range-rate equivalent [m/s] | Implemented | `output_unit="mps_equivalent"` |
| Two-way coherent Doppler frequency [Hz] | Implemented | `output_unit="hz"` |
| Total count phase | Partial | the differenced round-trip range over `Tc` *is* the cycle count; explicit accumulated-phase bookkeeping is not separately exposed |
| Integrated Doppler count | Partial | the observable is the count averaged over `Tc`; multi-interval accumulation is not modeled |

---

## 4. OD consistency review

**Consistent.** The same model is used end-to-end, because the mode lives on
`PassGeometry.range_rate_physics` and every consumer reads it:

- generation: `generate_range_rate_measurements` branches on the mode
  ([measurements.py:537-549](lunar_od/measurements.py#L537-L549));
- residual prediction: `compute_range_rate_residuals` branches on the **same**
  config and calls the **same** `two_way_counted_doppler_observable`
  ([measurements.py:637-649](lunar_od/measurements.py#L637-L649));
- the arc builder carries the physics into the pass geometry
  (`test_build_measurement_arcs_carries_two_way_range_rate_physics`).

There is **no** generation/residual mismatch: two-way generation is never paired
with an instantaneous prediction internally.

---

## 5. Jacobian review

| Mode | Jacobian | Endpoints | Light-time | Bounce time | Uplink/downlink |
|---|---|---|---|---|---|
| `geometric_instantaneous` | analytic local + STM | N/A | N/A | N/A | N/A |
| `two_way_counted_doppler` | **analytic** initial-state partials | yes (differenced) | yes (`dt1_dx`,`dt2_dx`) | yes (`t2`) | yes (`u_up`,`u_down`) |

The two-way analytic Jacobian is **not** an instantaneous approximation; it
differentiates the differenced round-trip light-time through the STM. A
finite-difference path (`_range_rate_numerical_initial_jacobian`) is retained as a
slower reference and is checked against the analytic one in tests.

---

## 6. Verification status

Existing coverage (selected):

| Area | Test |
|---|---|
| static geometry → 0 | `test_two_way_counted_doppler_zero_for_static_geometry` |
| receding tracks range-rate | `test_two_way_counted_doppler_tracks_receding_range_rate` |
| grid-density stability | `test_two_way_counted_doppler_is_stable_across_constant_acceleration_grid_density` |
| turnaround ratio + clock drift (Hz) | `test_two_way_hz_observable_responds_to_turnaround_ratio_and_clock_drift` |
| transponder delay | `test_two_way_light_time_includes_transponder_delay` |
| taylor3 guard + match | `test_taylor3_two_way_model_rejects_long_count_intervals`, `test_two_way_taylor3_local_state_model_matches_ode_observable` |
| noiseless residual closure | `test_two_way_counted_doppler_residual_closure` |
| analytic vs FD Jacobian | `test_two_way_analytic_initial_jacobian_matches_numerical_reference` |
| BLS/SRIF use analytic two-way Jacobian | `test_range_rate_bls_lm_two_way_uses_analytic_jacobian`, `test_range_rate_srif_two_way_uses_analytic_jacobian` |
| observability uses analytic light-time partials | `test_two_way_range_rate_observability_uses_analytic_light_time_partials` |
| UKF two-way recovery / consistency / mismatch | `test_lunar_ukf_two_way_range_rate_*`, `test_two_way_long_arc_noise_clock_and_model_mismatch_with_station_biases` |
| OD contracts (weight order, mps-equivalent R) | `test_two_way_doppler_uses_same_mps_equivalent_r_order` |

Added by this review (`tests/test_doppler_model_review.py`, 4 tests, all passing):

- **9A** geometric receding → `ρ̇ = speed`; approaching → `−speed`.
- **9B** transverse motion → `ρ̇ = 0`.
- **9C** two-way observable **depends on `Tc`** under non-zero jerk and collapses to
  the instantaneous rate as `Tc → 0` (it matches the analytic averaging signature
  `inst + (jerk/6)(Tc/2)²` to 0.05 m/s) — empirical proof that the mode is averaged
  differenced-range, not instantaneous.
- **9D** two-way differs from instantaneous under jerk → the modes are physically
  distinct.

---

## 7. Validation gaps

- **By construction**: media (troposphere/ionosphere/plasma), relativity
  (Shapiro, relativistic Doppler), and station displacement (tides) are not
  modeled, so they are neither implemented nor tested. These are the genuine gaps
  versus a full operational DSN model.
- **Documentation**: the `hz` observable is a coherent Doppler frequency; its
  relationship to total-count-phase / integrated-count conventions should be
  stated explicitly for users comparing against agency data types.
- **Cross-tool validation**: no SPICE/MONTE/GMAT numerical cross-check of the
  two-way observable yet (the light-time geometry is internally consistent and the
  analytic Jacobian is FD-verified, but an external reference is not wired in).

---

## 8. Recommended roadmap

### Priority 0 — Must be correct — **complete**
- ✅ Model consistency between generation and residuals (Section 4).
- ✅ Instantaneous vs averaged counted-Doppler distinction (Sections 1, 6: 9C/9D).
- ✅ Count-interval handling (`± Tc/2` endpoints, differenced range).
- ✅ Separate converged uplink/downlink light-time with three epochs.

### Priority 1 — Operational fidelity — **largely complete**
- ✅ Turnaround ratio (configurable; X/S-band ratios).
- ✅ Station velocity / frame consistency (ECEF for instantaneous, three-epoch
  station states for two-way).
- ✅ Analytic two-way Jacobian (+ FD reference).
- ◻ Document the frequency/count observable conventions (Hz vs total count phase).

### Priority 2 — High-accuracy DSN effects (not yet modeled)
- Tropospheric and ionospheric delay (mapping-function + zenith models).
- Solar plasma delay.
- Station displacement: solid-Earth tides, ocean loading, pole tide.
- Frequency ramp tables; richer clock/time-tag conventions.

### Priority 3 — Research-grade refinement
- Relativistic Doppler and Shapiro delay; gravitational frequency shift.
- Full Moyer-compatible observable (proper-time / coordinate-time transformations).
- SPICE / MONTE / GMAT numerical cross-validation of the two-way observable.

---

## 9. Final review question

**Does the current implementation behave more like textbook instantaneous
range-rate, operational two-way counted Doppler, or both depending on selected
mode?**

**Both, depending on selected mode.**

`geometric_instantaneous` is exactly the textbook instantaneous line-of-sight
range-rate `ρ̇ = (ρ⃗·ρ̇⃗)/|ρ⃗|`. `two_way_counted_doppler` is an operational-style
**averaged differenced-range** counted-Doppler observable: it solves separate
converged uplink and downlink light-times across three epochs, differences the
round-trip range over the count interval `Tc`, applies the coherent turnaround
ratio, and supports transponder delay and a station clock model — with a matching
analytic initial-state Jacobian. The added Tc-dependence test (9C) empirically
confirms the two-way path is averaged, not instantaneous. Both paths are
implemented correctly and used consistently in generation, prediction, residuals,
and the Jacobian; the two-way model is *simplified* relative to a full DSN
formulation only in the unmodeled media / relativity / station-displacement terms
catalogued in Section 3.
