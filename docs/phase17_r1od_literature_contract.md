# PHASE 17-R1O-D — ΔDOR LITERATURE CONTRACT

For every term this phase implements, approximates, or defers: the literature source, the exact
equation, the repository representation, and its status. Read this before reading `lunar_od/delta_dor.py`.

**Provenance note.** This document's literature research (the Moyer citations, equation numbers, and
quoted text below) was produced during an earlier attempt at this phase whose own `lunar_od/
delta_dor.py` implementation (function names `delta_dor_k_column`, `transmit_time_consistency_check`,
etc.) was superseded when this phase's final implementation was built from scratch. The "Repo
representation" column below has been updated to match the FINAL, shipped implementation
(`solve_common_transmit_event`, `delta_dor_spacecraft_sensitivity_full`, transmit-anchored); the
literature sourcing itself is unchanged and remains accurate. See "Resolving the anchor-convention
tension" below for how the final implementation's TRANSMIT-anchored construction was reconciled
against Moyer's own RECEPTION-anchored Section 11.4.1 formulation, which this research had already
identified as the literal textual convention.

## Primary references

- **T. D. Moyer**, *Formulation for Observed and Computed Values of Deep Space Network Data Types
  for Navigation*, JPL Publication 00-7 / DESCANSO Monograph Series 2 (2000/2003). Section 11,
  "Calculation of Precision Light Times and Quasar Delays," fetched and read directly
  (`descanso.jpl.nasa.gov/monograph/series2/Descanso2_S11.pdf`) — Eq. (11-9), (11-11), (11-12),
  (11-65), (11-66), (11-67) quoted below verbatim from the source.
- **D. W. Curkendall & J. S. Border**, *Delta-DOR: The One-Nanoradian Navigation Measurement System
  of the Deep Space Network — History, Architecture, and Componentry*, JPL IPN Progress Report
  42-193 (2013). Confirms the sign convention (§ below) and the historical/precision context used
  in R1O and repeated here.
- **DSN 810-005, Module 210**, *Delta Differential One-way Ranging*. Operational-level description
  of the ΔDOR data type, consistent with Moyer's OD-level formulation.

`DDOR_LITERATURE_CONTRACT_GATE = PASS` — every term below is traced to a numbered equation or is
explicitly marked deferred with a stated reason; none is invented.

## Term-by-term contract

| # | Term | Literature source | Repo representation | Status |
|---|---|---|---|---|
| 1 | One-way light time τ₁ = t₃(ST)_R − t₂(ET) | Moyer Eq. (11-9) | `LightTimeSolution.light_time_s` from `measurements.solve_one_way_light_time` (unmodified, frozen) | **Implemented** (pre-existing, reused) |
| 2 | Spacecraft differential delay D_S, common-reception-time IWS convention | Moyer §11.4.1, text following Eq. (11-12): *"the precision one-way light times for receivers 2 and 1 have a common reception time t₃(ST)_R, ... The transmission times ... will differ by less than the Earth's radius divided by the speed of light, or 0.02 s."* | **Shipped implementation uses the spec's transmit-anchored construction instead** (`delta_dor.solve_common_transmit_event`, a new forward+backward two-leg solver) — see "Resolving the anchor-convention tension" below for the direct numerical reconciliation against this Moyer convention (relative difference 1.4e-3 in value, ~3.3e-4 uniform scale offset in the Jacobian DIRECTION, i.e. no material effect on `f_perp`/`theta_K`) | **Implemented** (transmit-anchored; reception-anchored cross-checked independently) |
| 3 | Quasar delay definition, τ_Q = t₂(ST)_R − t₁(ST)_R | Moyer Eq. (11-65) | `delta_dor.QuasarDelay` naming (`station_b` minus `station_a`, matching "receiver 2" minus "receiver 1") | **Implemented** (definitional) |
| 4 | Quasar delay, geometric leading term r₁₂/c | Moyer Eq. (11-67): τ_Q = r₁₂/c + RLT₁₂ − [ET−TAI, TAI−UTC, UTC−ST diffs] + [antenna, solar-corona corrections]/10³c + downlink-delay diffs; r₁₂ = "distance the quasar wavefront travels from receiver 1 to receiver 2" | `delta_dor.quasar_plane_wave_delay()`: exact analytic plane-wave formula `D_Q = -(r_B - r_A) . n_hat / c` | **Implemented** (leading geometric term only — see below) |
| 5 | Relativistic light-time delay RLT₁₂ | Moyer Eq. (11-67), second term | Not implemented | **CHARACTERIZED_NOT_FULLY_IMPLEMENTED**. Order-of-magnitude bound: RLT-type Shapiro corrections for an Earth-baseline (≤1.3e7 m) VLBI delay scale as `~ (GM/c^3) * ln(...)`, of order 10^-11-10^-10 s for Earth-Sun geometry — this is far below the delay-equivalent of even the BEST-CASE tested angular noise (2 nrad * 1.04e7 m baseline / c =~ 7e-8 s). Deferred as negligible at the precision this phase tests, not because it is unimportant in general.
| 6 | Clock/timescale chain (ET−TAI, TAI−UTC, UTC−ST) per receiver | Moyer Eq. (11-67), lines 3-4 | Not implemented as separate terms; represented only implicitly via the common-mode cancellation tests (§25/§26 below), which inject synthetic per-station clock/instrument offsets directly | **CHARACTERIZED_NOT_FULLY_IMPLEMENTED** as a physical clock model; the CANCELLATION PROPERTY these terms are responsible for is tested directly. |
| 7 | Antenna corrections ∅_A, solar-corona corrections ∅_SC, downlink delays κ_D | Moyer Eq. (11-67), lines 5-6 | Not implemented | **CHARACTERIZED_NOT_FULLY_IMPLEMENTED** — media/antenna calibration physics; see `MEDIA_CALIBRATION_PHYSICS` in the main report. Represented only as an injected residual-error sweep (§32/§45 of the main report), never fabricated as a physical model. |
| 8 | ΔDOR sign convention | Curkendall & Border (2013): *"the delay time of the quasar is subtracted from that of the spacecraft's ... ΔDOR = Delay_spacecraft − Delay_quasar"* | `delta_dor.delta_dor_observable_s() = D_S - D_Q` | **Implemented**, `DDOR_SIGN_CONVENTION = QUALIFIED` |
| 9 | Observable unit | Moyer works throughout in seconds of delay (Eq. 11-9, 11-65 are both in seconds, explicit unit label "s") | `delta_dor_observable_s()` returns seconds; nrad is a derived reporting/diagnostic quantity only | `DDOR_OBSERVABLE_UNIT = SECONDS_OF_DELAY` |
| 10 | Same-wavefront ("common transmit event") physical picture | Not a separate Moyer equation; the spec's own §17 makes this a literal HARD requirement rather than a reference to Moyer's reception-anchored choice | `delta_dor.solve_common_transmit_event()`: solves t_tx BACKWARD from a fixed station-A reception time, then solves station B's reception FORWARD from that same, now-fixed t_tx (`solve_forward_one_way_light_time`, new) — both legs provably share one t_tx (§17 of the main report) | **Implemented literally**; independently cross-checked against Moyer's own reception-anchored convention (see reconciliation below) |
| 11 | K_SRP composition through the differenced light-time Jacobian | Phase 17-R's qualified `_two_way_range_k_srp_column` pattern: `H_K(t3) = H_x0(t3) . Phi(t3)^-1 . S_K(t3)`, validated to 4.6e-09/4.9e-08 (Phase 17A-R) | `delta_dor.delta_dor_spacecraft_sensitivity_full()`: the analogous 2x2 implicit-event-matrix substitution (`Phi_r` -> `S_K`), validated by composition + end-to-end FD to 8.2e-05 relative (main report §29) | **Implemented**, same substitution principle, event-matrix form rather than the single-event `Phi^-1` form (DDOR has two coupled solved events, not one) |

## Resolving the anchor-convention tension against Moyer

Section 17 of the governing phase specification ("SAME-WAVEFRONT EVENT SEMANTICS — HARD
REQUIREMENT") diagrams DDOR as anchored on a single spacecraft TRANSMIT event, with the two
stations' RECEIVE times differing. Moyer's actual OD-level "IWS" observable definition (term 2
above) anchors the opposite way: a common RECEPTION time (the correlator's data time tag), with the
two stations' independently-solved TRANSMIT times differing by up to ~0.02 s for Earth baselines.

These are not contradictory physics — they are the same light-cone geometry described from two
different anchors. The FINAL SHIPPED implementation follows the spec's own transmit-anchored
diagram literally (`solve_common_transmit_event`, a new forward+backward two-leg solver), since the
spec's §17 makes that construction an explicit HARD requirement ("Failure = HARD STOP") rather than
a reference to Moyer's own choice of anchor.

**This was not left as an untested assumption.** A direct numerical reconciliation
(`examples/phase17_r1od_anchor_convention_check.py`) computed the RECEPTION-anchored quantity
independently, using ONLY the pre-existing, already-qualified `one_way_light_time_initial_state_
sensitivity` called twice (once per station, at the same nominal epoch T) — zero new solver code —
and compared it against the shipped transmit-anchored production result on the identical campaign
epoch:

```
solved transmit times: t_tx_A = 53008.644392572 s, t_tx_B = 53008.637047297 s
difference: 0.007345 s   (within Moyer's own cited ~0.02 s Earth-baseline bound)

D_S (reception-anchored) = 7.345275021e-03 s
D_S (transmit-anchored)  = 7.334913931e-03 s
relative difference       = 1.41e-03

state Jacobian, component-by-component relative difference (reception vs transmit):
  x: 3.38e-4   y: 2.88e-4   z: 3.13e-4   vx: 3.30e-4   vy: 3.33e-4   vz: 3.41e-4
```

The two conventions' state-Jacobian VECTORS are consistently offset by the same small (~3.3e-4)
relative scale factor across all six components — i.e. they are nearly PARALLEL, not merely close in
norm. Since `f_perp`/`theta_K` depend on the Jacobian's DIRECTION relative to the six-state subspace,
not its overall scale, this means the two anchor conventions would classify K's identifiability
IDENTICALLY. The phase's central negative finding (§36-40 of the main report) is therefore robust to
this convention choice, quantitatively, not just by the qualitative "should be interchangeable"
argument this document's earlier draft offered.

## What this module does NOT implement

Per §14 of the governing spec: no raw RF sampling, no VLBI correlator, no baseband signal
processing, no PN-code generation, no antenna receiver DSP, no fringe fitting. This is the
OD-level calibrated ΔDOR observable Moyer's Section 11 describes, consuming already-solved
light-time and geometry — not the signal-processing chain that produces the calibrated delay in
the first place.
