# PHASE 17-R1O-D — PRODUCTION ΔDOR MEASUREMENT MODEL DESIGN, PHYSICS QUALIFICATION, AND K_SRP VALIDATION

## 1. Executive Summary

R1O demonstrated, with an explicitly-labeled analysis-only surrogate, that DDOR-like plane-of-sky
information could rotate K_SRP's measurement signature away from the six-state orbital-correction
subspace. This phase built the **actual production measurement physics** — a genuine common-transmit-
event solver, an analytic quasar plane-wave delay, and an implicit-event-matrix state/K Jacobian —
and asked the stricter question R1O could not answer: does the real physics retain that promise?

**It does not.** The production ΔDOR observable, built with correct differential light-time physics
(both stations tied to one shared spacecraft transmit event, not evaluated at a common nominal
epoch), gives `f_perp = 0.2121` at 5 nrad noise — *below* the range-only baseline's own `0.2523`, and
classified `MAGNITUDE_ONLY`. R1O's surrogate had shown `f_perp = 0.3633` at the same noise level on
the identical trajectory and station pair. A dedicated bridge experiment (§21) isolates the cause
precisely: R1O's surrogate evaluated both station ranges at one common nominal epoch, silently
discarding the light-time *difference* between the two station legs (of order tens of milliseconds,
translating to tens of meters of spacecraft motion at orbital speed). That approximation, not
anything about DDOR's physical principle, is what manufactured the apparent information-direction
gain. `R1O_SURROGATE_DID_NOT_GENERALIZE_TO_PRODUCTION_DDOR`, per the governing spec's own
anticipated §79 outcome.

This is not a defect to fix by tuning the model back toward the surrogate's answer — §45 explicitly
forbids that, and doing so would defeat the entire purpose of this phase, which existed precisely to
test whether the surrogate's promise survives real physics.

**A second, independent finding compounds this.** Even setting aside the direction-gain question, a
session-level DDOR calibration bias as small as 1 nanosecond — far below any literature-cited
residual floor — induces a K_SRP point-estimate shift of **22.5% of the truth value**; at 100 ns
(a plausible real VLBI-class residual) the induced shift is **22.5 times the truth value**. The
*formal* covariance barely notices this (a 1.01× inflation when bias is jointly solved for as a
nuisance parameter) — the danger is entirely in the point estimate, not the reported uncertainty,
which is exactly the distinction §57 requires this phase not to overlook.

**A third finding closes off the RF hardware assumption R1O made without evidence.** This repository
has no frozen reference-spacecraft RF design — only a schema/validation framework and test fixtures
explicitly labeled *"synthetic test fixture, not a spacecraft."* `REFERENCE_SPACECRAFT_DDOR_RF_
CAPABILITY = UNKNOWN`. R1O's claim that DDOR "requires no new spacecraft hardware" was therefore
unsupported and is corrected here.

Every physics-qualification gate this phase could run — geometric oracles, the common-transmit-event
gate, the quasar-delay oracle, local-delay numerical conditioning, the state Jacobian, and the
K-sensitivity chain — **passed**, several only after finding and fixing two genuine numerical bugs
(§17, §28) in the first implementation attempt, both documented in full below. The measurement
*physics* is qualified. The measurement's *scientific promise* for K_SRP is not.

Because the information-direction gate did not pass, §51's own sequencing correctly withholds
nonlinear BLS/SRIF recovery and future holdout testing — running them would test a solve-for
configuration already shown not to add independent K information, and §77/§79 do not ask for that.

```
PRODUCTION_DDOR_PHYSICS_QUALIFIED = YES
PRODUCTION_DDOR_INFORMATION_DIRECTION_GATE = FAIL (MAGNITUDE_ONLY, f_perp slightly WORSE)
R1O_SURROGATE_DID_NOT_GENERALIZE_TO_PRODUCTION_DDOR = YES (cause isolated: differential light-time)
REFERENCE_SPACECRAFT_DDOR_RF_CAPABILITY = UNKNOWN
DDOR_SESSION_BIAS_SENSITIVITY_CLASS = EXTREME
PRIMARY_CLASS = R1O_SURROGATE_DID_NOT_GENERALIZE_TO_PRODUCTION_DDOR
```

## 2. What R1O Actually Established

R1O built an analysis-only DDOR-like surrogate — a differenced one-way range evaluated at one common
nominal epoch for both stations, with no light-time or event solving — and found it rotated `f_perp`
from `0.2523` to `0.4419` at 1 nrad noise (`0.4246` at 2 nrad), driving fractional `sigma_K` from
9.3% to 1.2%. It explicitly labeled this `DDOR_LIKE_INFORMATION_SURROGATE`, not production DDOR, and
selected DDOR as the first candidate for this follow-on phase over lunar landmark LOS, on robustness
and heritage grounds.

## 3. R1O Numeric Erratum

**Audit finding.** The underlying machine-readable artifact `r1o_ddor_sweep.csv` is internally
consistent: every noise level's row (1–300 nrad) carries correct, self-consistent `f_perp` and
`sigma_K/K` values, verified directly against the CSV. The defect is confined to **prose and one
derived-table column**:

- R1O's executive summary (and two further prose repetitions) states *"rotates the orthogonal K
  fraction from 0.252 to 0.442 at 2 nrad noise... drives the fractional K uncertainty from 9.3% down
  to 1.2%."* These are the **1 nrad** values (`f_perp=0.4419`, `sigma_K/K=1.224%`), not the 2 nrad
  values (`f_perp=0.4246`, `sigma_K/K=2.365%`).
- The root cause is traced to `examples/phase17_r1o_comparison.py`: its primary comparison table
  built a row labeled `"O2 DDOR best-in-class (2 nrad)"` but populated it from
  `r1o_ddor_summary.json`'s `best_*` fields — which hold the single **best-tested** (1 nrad) case,
  not literally the 2 nrad row. That mislabeled string then propagated into
  `r1o_observable_comparison.csv`'s `case` column and from there into three places in the report
  prose.
- One further, smaller slip: §14/§35's printed 2 nrad row shows `sigma_K/K = 0.0242`; the CSV's exact
  value is `0.023653...`, which rounds to `0.0237`, not `0.0242`.

```
R1O_ERRATUM_CLASS = DOCUMENTATION_TRANSCRIPTION_ONLY
R1O_NUMERIC_CONSISTENCY_GATE = PASS
```

**Correction (additive; the historical artifacts and report file are left unmodified as evidence):**
the authoritative 2 nrad values are `f_perp = 0.4246`, `sigma_K/K = 2.37%` (not `0.442` / `1.2%`,
which are the 1 nrad values). This correction does not change any qualification verdict R1O reached
— 1 nrad and 2 nrad both sit in R1O's own "best-in-class" literature-cited regime — but it must not
be quoted going forward as "the 2 nrad result."

## 4. Why Production DDOR Is the Next Rational Step

Restated from the governing spec, and directly borne out by this phase's own result: a surrogate
proves an idea is worth building. It cannot prove the built thing preserves the idea. R1O's own
executive summary said as much about its own landmark surrogate's map/attitude sensitivity; this
phase found the DDOR surrogate had an even more fundamental gap — not a degradation under realistic
noise, but a construction that never modeled the light-time physics DDOR's whole informational
content depends on.

## 5. Academic DDOR Theory

The implementation follows the standard DSN differential-delay formulation (Moyer, *Formulation for
Observed and Computed Values of DSN Data Types for Navigation*; Curkendall & Border, *Delta-DOR: The
One-Nanoradian Navigation Measurement System of the DSN*, IPN PR 42-193): a spacecraft
interferometric differential delay `D_S = tau_{S,B} - tau_{S,A}` calibrated against a quasar
differential delay `D_Q` for the same baseline. The governing identity —

```
D_S = tau_{S,B} - tau_{S,A} = light_time_B - light_time_A
```

with both one-way light times traced to one common spacecraft transmit event — is the literal
algebraic content of Moyer's precision light-time formulation applied to two receivers of one
wavefront, and is what this phase's event solver computes (§13–§16).

**What is implemented, approximated, or deferred**, per the literature terms:

| Moyer/DSN term | repo representation | status |
|---|---|---|
| Spacecraft one-way light time (each leg) | `solve_one_way_light_time` (existing, qualified) + new forward solver | implemented |
| Common spacecraft transmit event | `solve_common_transmit_event` (new) | implemented, oracle-qualified |
| Quasar far-field plane-wave delay | `quasar_differential_delay_s`, fixed J2000-like catalog direction | implemented (synthetic catalog frame, §22) |
| Quasar-scan / S-Q-S switching interpolation | — | NOT modeled; a single simultaneous calibration is assumed |
| Media (troposphere/ionosphere/plasma) calibration | — | `MEDIA_CALIBRATION_PHYSICS = CHARACTERIZED_NOT_FULLY_IMPLEMENTED` |
| Station clock / instrumental delay | — | characterized only as an injectable session-bias term (§32/§33), not a physical clock model |
| Earth orientation / nanoradian-class station frame fidelity | reuses the existing SPICE ITRF93↔J2000 transform, unaudited at nanoradian precision | `EARTH_ORIENTATION_MODEL_LIMITATION = IDENTIFIED`, not quantified |
| PN-DOR / tone / correlator signal processing | — | out of scope by design (§14); this is an OD-level observable, not a signal-chain model |

```
DDOR_LITERATURE_CONTRACT_GATE = PASS
```

No claim of novelty is made for the underlying physics (§11); this table is scope documentation, not
a claim of completeness.

**Anchor-convention tension, found and reconciled.** The governing spec's own §17 diagrams DDOR as
anchored on a single spacecraft *transmit* event; Moyer's actual Section 11.4.1 anchors instead on a
common *reception* time, with each station's transmit time solved independently backward. The shipped
implementation follows the spec's transmit-anchored diagram literally (§17's "HARD REQUIREMENT"
language leaves no discretion), but this was not left as an untested assumption: a direct numerical
reconciliation (`examples/phase17_r1od_anchor_convention_check.py`; full derivation in
`docs/phase17_r1od_literature_contract.md`) computed Moyer's reception-anchored quantity independently
— using *zero* new solver code, two calls to the pre-existing, already-qualified
`one_way_light_time_initial_state_sensitivity` — and compared it against the production result on the
identical campaign epoch. The two solved transmit times differ by `0.007345 s`, within Moyer's own
cited `~0.02 s` Earth-baseline bound; `D_S` itself differs by `1.41e-03` relative; and — the decisive
number — the two conventions' **state-Jacobian vectors are offset by a uniform ~3.3e-04 relative scale
factor across all six components**, i.e. they are nearly *parallel*, not merely close in norm. Since
`f_perp`/`theta_K` depend on the Jacobian's *direction*, not its scale, both literature-consistent
anchor conventions classify K's identifiability identically. §36–40's negative finding is therefore
robust to this convention choice, quantitatively confirmed rather than assumed.

## 6. DSN Operational Context

DDOR has been flown operationally for decades and is well characterized in the cited literature at
the 1–100 nrad precision range. The operational question this phase actually had to answer was
narrower and project-specific: does DDOR's established navigational value transfer to observing a
*weak SRP force parameter* specifically, on *this* mission's tracking geometry. §51's own gating
correctly stopped this phase from proceeding to a nonlinear-estimation demonstration once the answer
to that narrower question came back negative.

## 7. Reference Spacecraft RF Capability

**Audited, not assumed** (correcting R1O's unaudited claim). `lunar_od/rf/link_budget.py` implements
carrier EIRP/CN0/free-space-loss/G-over-T calculations for a two-way coherent link — no DOR-tone
generation, PN-DOR modulation, or wideband ranging-channel structure is represented anywhere in it.
`lunar_od/reference_config/` is a schema and validation *framework* (it asserts that *if* a
transponder band is declared, uplink/downlink frequencies lie inside it) — it is not itself a frozen
spacecraft RF design. The only populated instance of that schema in this repository is
`tests/test_reference_config.py`'s `_synthetic_document()` fixture, whose own `limitations` field
reads: `"synthetic test fixture, not a spacecraft"`.

```
REFERENCE_SPACECRAFT_DDOR_RF_CAPABILITY = UNKNOWN
```

Per this phase's own governing rule: `NO_NEW_SPACECRAFT_HARDWARE_REQUIRED` must not be claimed, and
is not claimed here. Whether the mission this repository models could fly DDOR at all is genuinely
unresolved by anything in the codebase.

## 8. Entering Repository State

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 9f18d96407647150e6eb48fd2b967db152e89dc3
START_TREE   = e18399d79e1600266d832b69fe2e7011403ea60c
```

Verified identical to R1O's reported `FINAL_HEAD`. `main`/`origin/main` verified unchanged at
`fea476f81dad07b3914e53eab709e10fa6e9d10b`. `R1OD_INPUT_GATE = PASS`.

## 9. Measurement Architecture

Per §14's explicit scope boundary, this phase implements the **OD-level calibrated ΔDOR
observable** — the processed interferometric-delay quantity an orbit determination system consumes
— not raw RF sampling, correlator, or PN-code processing. The new production module is
`lunar_od/delta_dor.py`, added standalone (not wired into `estimators.py`/`filters.py`/`scenarios.py`
call sites): its qualification is complete, but its *scientific case* for solve-for integration did
not clear this phase's own information-direction gate (§36–40), so no estimator-interface change was
made or would have been justified.

## 10. Observable Definition

```
D_S = tau_{S,B} - tau_{S,A} = light_time_B - light_time_A     (spacecraft term)
D_Q = -(r_B - r_A) . s_hat / c                                 (quasar term, plane wave)
DDOR = D_S - D_Q
```

`light_time_A`/`light_time_B` are the converged one-way light times from the single common transmit
event to stations A and B respectively (§13–16), never reconstructed from a difference of absolute
epochs (§17's numerical-conditioning fix, below).

## 11. Units / Sign Convention

```
DDOR_OBSERVABLE_UNIT = seconds of differential delay
```

matching §16's preference; an angular-equivalent (nrad) is used only as a derived reporting/noise-
specification convenience, never as the internal representation. Sign convention was **qualified by
oracle**, not frozen by assumption: the station-swap oracle (§18) proves `D_S(A,B) = -D_S(B,A)` to
exact floating-point negation once both share the same transmit event.

```
DDOR_SIGN_CONVENTION = QUALIFIED
```

## 12. Time-Tag Contract

The observation epoch is defined as **station A's reception time**, a fixed input constant `T_obs`.
The transmit event `t_tx` is solved backward from it; station B's reception `t_B` is solved forward
from that same `t_tx`. This is verified directly (`tests/test_delta_dor.py::
test_time_tag_is_reception_at_station_a`), using the solver's own internal equation residual rather
than an externally reconstructed epoch difference (see §17 for why that distinction matters
numerically).

```
DDOR_TIME_TAG_CONTRACT = QUALIFIED
```

## 13. Spacecraft Interferometric Event Geometry

**Purpose.** Establish, before any sensitivity or information-geometry work, that both stations'
receptions trace to one physically shared spacecraft transmit event (§17's hard requirement).

**Method.** Two chained light-time solves: `solve_one_way_light_time` (existing, qualified,
receive-anchored) backward from `T_obs` at station A gives `t_tx`; a new
`solve_forward_one_way_light_time` (transmit-anchored, spacecraft position frozen at `t_tx`) gives
`t_B`. Both share the numerical structure of the existing solver (dual convergence criterion: update
tolerance plus an independently re-evaluated equation residual, FA-03A-style).

**Results.** On the qualified W15 campaign trajectory (Goldstone–Canberra pair, midpoint epoch):
`t_obs=53010.000 s`, `t_tx=53008.644392572 s`, `D_S=7.345237349e-03 s`. Both legs' equation residuals
are `0.0` at the printed precision (well under the `1e-11 s` internal tolerance).

**Verdict.** `PASS`. `DDOR_COMMON_TRANSMIT_EVENT_GATE = PASS`.

## 14. Common-Transmit-Event Solver

Covered in §13 and §10; see `lunar_od/delta_dor.solve_common_transmit_event`.

## 15. Quasar Delay

**Method.** `D_Q = -(r_B - r_A) . s_hat / c` for a fixed catalog unit vector `s_hat` (RA/Dec ->
Cartesian, §21/§22). Verified against a hand-computed dot product to `1.8e-18` absolute agreement,
against zero for a zero baseline, and against zero for a baseline constructed perpendicular to the
source direction.

```
QUASAR_DELAY_ORACLE_GATE = PASS
```

## 16. Spacecraft–Quasar Calibration Difference

`DDOR = D_S - D_Q`. Because `D_Q` depends only on station positions and the fixed catalog direction
— never on the spacecraft state or K — it contributes **zero** variance and **zero** K-sensitivity to
the observable's design row; it shifts the observable's *value* but not its *information content*.
This was confirmed structurally (§40) and is why the arbitrary quasar direction chosen for the
production campaign (§30) has no bearing on the information-geometry result.

## 17. Local-Delay Numerical Conditioning

**This is where the first genuine bug was found and fixed.**

An initial implementation computed `D_S = t_B - T_obs`, reasoning that both operands sit within one
baseline light-time (~tens of ms) of each other and so "cost no precision." That reasoning was
wrong: `t_B` is itself constructed as `t_tx + light_time_B`, and `t_tx` already carries the full
magnitude of `T_obs`. Precision is lost when `t_B` is *built* (rounding `light_time_B` to
`ulp(t_tx)`), not when it is later subtracted — so subtracting `T_obs` afterward cannot recover it.

**Evidence.** A conditioning sweep holding the physical scenario fixed (identical relative
geometry/velocities at every epoch, only the absolute epoch label varied across binades
`2^0` to `2^36` s) found the old formula's `D_S` drifting by up to **1.265e-04** relative — a real
degradation, growing with epoch magnitude, exactly the Phase 17C-class quantization symptom.

**Fix**, mirroring `two_way_range.py`'s own Phase 17C repair pattern exactly: difference the two
*local* light times directly, `D_S = light_time_B - light_time_A`, never touching the absolute
epochs in the final formula. Re-run of the identical conditioning sweep after the fix:

| test | before | after |
|---|---:|---:|
| max relative step across binades `2^0`–`2^36` s | 1.265e-04 | **1.377e-10** |

**Verdict.** `PASS`. `DDOR_LOCAL_DELAY_NUMERICAL_CONDITIONING_GATE = PASS`.

## 18. Basic Geometric Oracles

Zero baseline → zero `DDOR` (exact `0.0`); station swap → exact sign reversal (`d_s_ab + d_s_ba ==
0.0`, pure float negation, verified by holding `t_tx` fixed and swapping only which light time is
subtracted from which — an earlier version of this test compared two *different* transmit events
under differently-anchored tags and found a spurious ~3e-8 s residual, which was a flawed test, not a
solver defect, and was corrected); quasar plane-wave delay → exact dot-product match; rigid epoch
translation → `D_S` unchanged to `0.0` difference when the whole scenario (not just the tag) is
re-anchored identically.

```
DDOR_GEOMETRIC_ORACLE_GATE = PASS
```

## 19. Spacecraft Event Oracle

**Purpose.** An independent, non-fixed-point solve (scalar root-finding via `scipy.optimize.brentq`
on the same local-light-time-differenced quantity) against which the production solver's `t_tx` and
`D_S` are compared.

**Results.** `t_tx`: relative agreement `0.000e+00`. `D_S`: relative agreement `0.000e+00` (after the
oracle's own reference computation was corrected to use the same local-light-time-difference form —
an earlier version of the oracle used the same flawed `t_b - T_obs` construction and disagreed with
the (already-fixed) production result at `3.9e-8` relative, which was the oracle's own defect, not
production's).

```
SPACECRAFT_DOR_EVENT_ORACLE_GATE = PASS
```

## 20. Quasar Delay Oracle

Covered in §15.

## 21. R1O Surrogate Bridge

**Purpose.** Isolate, by direct construction, exactly what changed between R1O's surrogate result
and production's result — the central §79 investigation this phase's negative finding requires.

**Method.** Three points on one dial, all on the identical W15 trajectory, station pair, and epoch
set: (1) range-only baseline; (2) R1O's *original*, unmodified surrogate function, called directly;
(3) production's full common-transmit-event model.

**Results** (5 nrad, Goldstone–Canberra, W15, n=143 dual-visible epochs in both DDOR cases):

| configuration | `f_perp` | `I(K|x)` |
|---|---:|---:|
| range-only | 0.2523 | 1.1628e+06 |
| R1O's original surrogate (simultaneous-epoch approximation) | **0.3633** | 4.1191e+06 |
| Production (common-transmit-event, light-time-correct) | **0.2121** | 1.3295e+06 |

A row-by-row comparison of the surrogate's own `h_k` design-matrix column against a from-scratch
reimplementation of its exact construction (using this phase's own utility functions) agreed to
`1.2e-08` absolute — confirming the reimplementation faithfully reproduces R1O's method, and that the
divergence from production is a genuine physical-model difference, not an implementation
inconsistency on either side.

**Interpretation.** R1O's surrogate evaluated both station ranges at one common nominal epoch,
implicitly treating the spacecraft as being at the same point in its orbit for both legs. Production
correctly uses the spacecraft position at the *single shared transmit event*, but that transmit event
is reached by two receivers separated by light-times differing by tens of milliseconds — at orbital
speeds of order km/s, tens of meters of spacecraft motion the surrogate discarded entirely. That
discarded motion is evidently *not* small relative to the already-minuscule K-sensitivity angle this
observable was hoped to supply.

```
R1O_SURROGATE_REDUCED_MODE_BRIDGE = PASS (bridge executed; result is a clean isolation, not a match)
R1O_SURROGATE_DID_NOT_GENERALIZE_TO_PRODUCTION_DDOR = YES
```

**Verdict.** `CHARACTERIZATION` (the investigation this phase required; the finding is negative and
is not "fixed," per §45's explicit prohibition on tuning toward the surrogate's answer).

## 22. Clock / Instrument Calibration

Common-mode station clock and instrumental delay are addressed structurally rather than by a
dedicated numerical injection experiment (deferred, see §58): because `D_S` is a *difference* of two
one-way light times sharing the same physical transmit event, any error common to both legs (e.g. a
shared timing reference) cancels by construction in the differencing; the only calibration error that
survives is one that differs between the two station legs (a station-specific instrumental delay or
clock offset). This is the standard DDOR common-mode-rejection argument (Curkendall & Border); this
phase confirms it analytically for the implemented formula but did not inject a synthetic per-station
clock offset to measure the residual quantitatively (see §46's session-bias study for the closest
quantitative analogue — a *shared*, not per-station, calibration residual).

```
COMMON_MODE_CLOCK_CANCELLATION_GATE = CHARACTERIZED_ANALYTICALLY_NOT_NUMERICALLY_INJECTED
COMMON_MODE_INSTRUMENT_CANCELLATION_GATE = CHARACTERIZED_ANALYTICALLY_NOT_NUMERICALLY_INJECTED
```

## 23. Media Residual Treatment

No tropospheric/ionospheric/solar-plasma calibration model is implemented; this was a design decision
stated up front (§27 of the governing spec: "do not fabricate them").

```
MEDIA_CALIBRATION_PHYSICS = CHARACTERIZED_NOT_FULLY_IMPLEMENTED
```

## 24. Quasar Catalog Error

Not run: the decisive negative finding (§21, §36–40) was reached before this secondary sensitivity
study was scheduled, and — since `D_Q` was shown (§16) to carry zero K-sensitivity regardless of the
chosen quasar direction — a catalog-error sweep would characterize `D_Q`'s own value accuracy, not
K_SRP's identifiability, and was judged lower priority once the primary information question closed
negatively.

```
STATUS = NOT_RUN (secondary; information gate already closed the primary question)
```

## 25. Baseline / Earth-Orientation Error

Not run as a dedicated FD sensitivity sweep, for the same reason as §24. The qualitative limitation is
recorded: the existing SPICE ITRF93↔J2000 transform this module reuses (§9 of the R1M/R1COV/R1O
lineage) has never been independently audited for nanoradian-class angular fidelity — it was built and
qualified for two-way range/Doppler, where meter-level, not sub-meter, accuracy matters.

```
EARTH_ORIENTATION_MODEL_LIMITATION = IDENTIFIED (not quantified)
```

## 26. State Jacobian

**Purpose.** Verify the analytic implicit-event-matrix state Jacobian `d(D_S)/dx0` independently.

**Method.** The same 2×2 implicit-event-matrix construction `two_way_range_event_sensitivity`
already uses for the two-way range chain (`dy/dx0 = solve(G_y, -G_x)`), here for
`y = [t_tx, t_B]`. See `lunar_od.delta_dor.delta_dor_spacecraft_sensitivity_full`.

**Results.** Event matrix condition number `2.618` (well conditioned). Full FD verification in §28.

## 27. Event-Time Derivatives

The implicit chain explicitly includes both events' time-derivative dependence (`∂G_A/∂t_tx`,
`∂G_B/∂t_tx`, `∂G_B/∂t_B` all appear in `G_y`) — this is the exact requirement §38 sets: the state
Jacobian is not computed by freezing the solved event times and differentiating only the
instantaneous geometry at them.

## 28. FD Sweep

**This is where the second genuine bug was found and fixed — a fidelity mismatch, not a physics
error.**

An initial FD verification harness used crude piecewise-linear interpolation for both spacecraft and
station positions, while the analytic Jacobian under test uses the qualified cubic-Hermite
interpolator (`_interp_state`) throughout. Result: the state-Jacobian check passed marginally (max
relative error `3.11e-04`, against a `1e-4` threshold — a **FAIL**), and the K-sensitivity end-to-end
check **failed catastrophically** (`rel_err = 37.3`, with the FD estimate not even holding a stable
sign across step sizes).

**Diagnosis.** `d(D_S)/dK` is of order `3.5e-9` s per (m²/kg). A physically expected FD signal at
`dK ~ 1e-6` is therefore of order `3.5e-15` s — far below the ~1e-7 s *absolute* position-interpolation
noise a 90-second-cadence linear interpolant's own truncation error introduces. This is precisely the
"floor consistency" lesson from Phase 17A-R: an FD oracle cannot resolve a signal smaller than its own
interpolation floor.

**Fix.** Every position/velocity lookup in the FD harness — spacecraft trajectory and both stations —
was rebuilt on the same qualified Hermite interpolator the analytic path uses, with station velocity
taken directly from the existing qualified 6×6 J2000↔ITRF93 transform's velocity block (via
`_station_relative_state_j2000_at_receive_epoch`) rather than numerically differenced.

| check | before (linear interpolation) | after (Hermite, matched fidelity) |
|---|---:|---:|
| state Jacobian, max relative error | 3.11e-04 | **6.78e-06** |
| K-sensitivity E2E, relative error | 37.3 (sign unstable) | **8.21e-05** (sign & order-of-magnitude confirmed) |

**Verdict.** `PASS`. `DDOR_STATE_JACOBIAN_GATE = PASS`.

## 29. K_SRP Sensitivity Chain

**Purpose.** Qualify `d(D_S)/dK` by two independent routes: composition (the analytic implicit-event
chain, with `Phi_r` substituted by the trajectory's `S_K` column — the identical substitution pattern
`_two_way_range_k_srp_column` uses for range) and end-to-end finite difference (re-propagating the
full trajectory at perturbed K values and re-solving the event chain from scratch).

**Results.** Composition-chain `d(D_S)/dK = 3.747483269e-09`. End-to-end FD (5-point sweep,
`dK` from `1e-4·K` to `1e-2·K`, sized to lift the signal above the `rtol=1e-12` propagator's own
integration noise floor): best match `3.747790966e-09`, relative error `8.21e-05`, matching sign and
order of magnitude cleanly across the whole sweep.

```
DDOR_K_COMPOSITION_SENSITIVITY_GATE = PASS
DDOR_K_E2E_SENSITIVITY_GATE = PASS
DIRECT_MEASUREMENT_K_DEPENDENCE = NONE
```

Structurally confirmed (not merely inferred): `g_x` and `g_k` in the implicit-event system share the
identical `u_a`/`u_b`/`Phi_r` machinery, substituting only `S_K` for `Phi_r` — no separate K term
exists anywhere in `G_A` or `G_B`. K enters exclusively through the trajectory: `K → a_SRP → x(t_tx) →
D_S`.

## 30. Goldstone–Canberra Geometry

**Purpose.** Reproduce R1O's dual-visibility finding using the production event solver, not the
elevation-mask surrogate check.

**Results.** 143 epochs pass the qualified 10° elevation mask on both stations (Goldstone, Canberra)
across the W15 window; **all 143** production common-transmit-event solves converge, with event-matrix
condition number a constant `2.618` across every row. This count matches R1O's own surrogate
visibility count exactly, confirming both studies' independently-computed visibility masks agree on
which epochs qualify.

```
GOLDSTONE_CANBERRA_DUAL_VISIBILITY_REPRODUCTION = PASS
```

## 31. Random Noise Study

**Setup.** Range+production-DDOR combined system, W15 window, R1COV-qualified square-root covariance
throughout (§40).

| noise [nrad] | `f_perp` | `sigma_K/K` |
|---:|---:|---:|
| 1 | 0.0738 | 0.0782 |
| 2 | 0.1264 | 0.0840 |
| 3 | 0.1657 | 0.0857 |
| 5 | 0.2121 | 0.0867 |
| 10 | 0.2493 | 0.0873 |
| 30 | 0.2636 | 0.0880 |
| 100 | 0.2641 | 0.0885 |

Every single tested noise level — including the **best-in-class 1 nrad case** — leaves `f_perp`
*below* the range-only baseline's `0.2523`. Adding production DDOR at any tested precision does not
just fail to rotate K's signature; the range-only baseline is more favorably aligned on its own. This
is the decisive, complete refutation of R1O's surrogate-based projection, not a marginal or
noise-dependent effect.

```
DDOR_RANDOM_NOISE_ROBUSTNESS_CLASS = NOT_APPLICABLE (no beneficial regime found at any tested precision)
```

## 32. Systematic Residual Study

Addressed jointly with §33 below, since the meaningful systematic-error question for this observable
turned out to be the session-level calibration bias, not per-row noise.

## 33. Session-Bias Study

**Purpose.** §46's explicit three-case study: does K improvement survive when DDOR sessions carry an
unmodeled calibration bias?

**Method.** Two *distinct* formulas were required (an earlier version of this script incorrectly used
one formula for both, conflating them):

- **Case B (unmodeled bias):** the standard omitted-variable-bias result. For the 7-parameter normal
  system `N_7 = H_7^T W H_7` (range+DDOR, no bias parameter) and a bias-direction column `n_bias =
  H_7^T W · (1 on DDOR rows, 0 on range rows)`, the point-estimate shift for an unmodeled bias `b` is
  `delta_theta = solve(N_7, n_bias) · b`. This is a shift in the **estimate**, not the covariance.
- **Case C (bias solved as a nuisance parameter):** an 8-parameter joint solve (range+DDOR+bias),
  covariance from the same qualified square-root path. This inflates `sigma_K` but — by construction
  of a jointly-solved least-squares system — never biases the point estimate.

**Results.**

| session bias | Case B: induced `dK` [m²/kg] | as fraction of `K_truth` (0.01) |
|---:|---:|---:|
| 0 | 0 | 0% |
| 1 ns | -2.251e-03 | **22.5%** |
| 10 ns | -2.251e-02 | **225%** |
| 100 ns | -2.251e-01 | **2251%** |

Case A (perfectly calibrated) `sigma_K = 8.673e-04`; Case C (bias solved as nuisance) `sigma_K =
8.719e-04` — only a **1.01× inflation**, barely distinguishable from Case A.

**Interpretation.** This is the sharpest single result in the whole systematics study. The *formal
covariance* is nearly blind to the bias risk (Case C moves almost nothing), while the *point estimate*
is catastrophically sensitive to it (Case B): a bias two orders of magnitude below any literature-
cited DDOR residual floor already exceeds the truth value itself. This is exactly the failure mode
§57 warns a covariance-only closure would miss.

```
DDOR_SESSION_BIAS_SENSITIVITY_CLASS = EXTREME
```

## 34. Quasar-Separation Study

`NOT_RUN` — see §24. Since `D_Q` carries zero K-sensitivity regardless of source direction or
separation angle (§16), a quasar-separation sweep would not change the primary finding; it was
deprioritized once the information gate closed negatively.

## 35. Baseline-Orientation Study

`NOT_RUN` as a dedicated sweep — but partially addressed by construction: R1O's own window-selection
probe (inherited here) already established Goldstone–Canberra as the *only* DSN pair with genuine
≥10° dual visibility in this orbit geometry, so no alternative "poorer but valid" baseline orientation
exists to test within this campaign's station set without extending the propagated arc further, which
was judged not to change the qualitative negative finding.

## 36. Production Information Geometry

**Purpose.** The central comparison: does adding the qualified production DDOR observable to the
qualified range baseline change K's identifiability, using R1COV's qualified covariance exclusively.

**Setup.** W15 (15 orbits, 3 DSN stations, 782 range observations) + 143 Goldstone–Canberra DDOR rows
at 5 nrad (a representative operational precision, R1O's own literature anchor).

**Results.**

| system | `n` | `I(K|x)` | `f_perp` | `theta_K` [deg] | `sigma_K/K` |
|---|---:|---:|---:|---:|---:|
| range-only (control) | 782 | 1.162817e+06 | 0.2523 | 14.61 | 0.0927 |
| range + production DDOR | 925 | 1.329505e+06 | **0.2121** | 12.25 | 0.0867 |

`f_perp` **decreases**; `theta_K` decreases correspondingly (14.61° → 12.25°); `I(K|x)` increases only
1.14×. Every metric this phase's own §16 (the governing spec, not this report) requires — magnitude
vs. direction, `f_perp`, `theta_K` — agrees: this is a pure, and in fact slightly *negative*, magnitude
effect.

```
PRODUCTION_DDOR_INFORMATION_DIRECTION_GATE = FAIL
PRODUCTION_DDOR_INFORMATION_DIRECTION_CLASS = MAGNITUDE_ONLY
```

## 37. Conditional K Information

`I(K|x)` rises from `1.163e+06` to `1.330e+06` (1.14×) — see §36. Consistent with R1M/R1G's repeatedly
observed pattern: adding measurement volume without a genuinely orthogonal direction accumulates
magnitude without separating K.

## 38. Orthogonal K Fraction

`f_perp: 0.2523 → 0.2121`, a **decrease** of `0.0401`. Reported honestly rather than rounded away:
production DDOR at 5 nrad makes K's independent signature fraction *smaller*, not larger.

## 39. Principal-Angle Result

`theta_K = arcsin(f_perp): 14.61° → 12.25°`. Consistent with §38; no rotation toward orthogonality
occurred.

## 40. Qualified K Covariance

Every `sigma_K` reported in this phase comes from R1COV's qualified square-root path
(`_square_root_covariance_from_design`, imported and used unmodified), never the floored
normal-matrix inverse. Cross-checked directly on the production combined system against an
independent Schur-complement calculation:

```
QR sigma_K    = 8.672714e-04
Schur sigma_K = 8.672714e-04
relative error = 1.000e-15
R1COV_DDOR_COVARIANCE_PATH_GATE = PASS
```

## 41. BLS Nonlinear Recovery

`NOT_RUN_BY_GATE`. §51 explicitly sequences nonlinear recovery *after* the physics and information
gates both pass. Physics passed; the information-direction gate (§36) did not. Running a synthetic
correct/wrong/solve-K BLS recovery on a combined system already shown to carry no independent K
direction would test a configuration this phase has already characterized as not adding value, and
§77/§79 do not call for it — §45 specifically forbids tuning to manufacture a result, and running an
expensive nonlinear demonstration on a system known not to help would not change that finding, only
consume effort the spec's own gating logic says to withhold.

```
BLS_DDOR_SOLVE_FOR_GATE = NOT_RUN_BY_GATE
```

## 42. SRIF Confirmation

`NOT_RUN_BY_GATE`, for the same reason as §41 (depends on §41).

```
BLS_SRIF_DDOR_SCIENTIFIC_CONSISTENCY_GATE = NOT_RUN_BY_GATE
```

## 43. SR-UKF Scope Decision

Per §55's explicit instruction not to automatically extend DDOR to SR-UKF: not audited in depth, since
§41/§42 already closed the nonlinear-estimation question by gate. If a future phase revisits DDOR
(e.g. with a different baseline pair, longer span, or corrected systematics), SR-UKF measurement-
interface compatibility would need a fresh audit at that time.

```
SRUKF_DDOR_STATUS = DEFERRED_ARCHITECTURE (not reached)
```

## 44. Future Holdout

`NOT_RUN_BY_GATE`, downstream of §41.

```
DDOR_HOLDOUT_PREDICTION_GATE = NOT_RUN_BY_GATE
```

## 45. Wrong-Fixed-K Comparison

`NOT_RUN_BY_GATE`, downstream of §41.

## 46. What Changed in the System

```
PRODUCTION_CODE_CHANGED               = YES  (one new file, lunar_od/delta_dor.py)
PHYSICAL_FORCE_MODEL_CHANGED          = NO
EVENT_SOLVER_CHANGED                  = NO   (two_way_range.py's event solver untouched; a
                                              NEW, separate event solver was added for DDOR,
                                              not a modification of the existing one)
NEW_MEASUREMENT_MODEL_ADDED           = YES  (production OD-level DDOR observable)
ESTIMATOR_MEASUREMENT_INTERFACE_CHANGED = NO (estimators.py/filters.py untouched; §41/§42's
                                              NOT_RUN_BY_GATE status meant no solve-for wiring
                                              was ever justified)
K_SOLVE_FOR_MATH_CHANGED              = NO
COVARIANCE_METHOD_CHANGED             = NO   (R1COV's square-root path reused unmodified)
DEFAULT_6STATE_BEHAVIOR_CHANGED       = NO
```

`lunar_od/delta_dor.py` is entirely new, additive, and is not imported by any existing production
entry point — it is reachable only from this phase's own analysis scripts and its permanent test
file. No existing production file's bytes changed.

## 47. What Did Not Change

`lunar_od/estimators.py`, `dynamics.py`, `two_way_range.py`, `radiometrics.py`, `measurements.py`,
`filters.py`, `srp.py`, `visibility.py`, `lunar_frames.py`, and R1COV's square-root covariance helper
were all read and executed only. `git status --porcelain --untracked-files=no` was empty throughout
except for the one new production file, its test file, and analysis/artifact additions.

## 48. Scientific Interpretation

The central lesson is methodological as much as it is a result about DDOR specifically: an
*information-geometry surrogate built without the observable's defining physics can manufacture an
information-direction signal that the real physics does not contain.* R1O's surrogate discarded
exactly the physical content — differential light-time across two receivers of one wavefront — that
gives DDOR its actual value in the literature; discarding it here did not merely degrade the result,
it inverted its qualitative character, from "rotates K's signature" to "does not, and mildly reduces
`f_perp` below the control."

The session-bias finding (§33) is a second, independent lesson: formal covariance and point-estimate
bias sensitivity are not the same question, and a covariance-only closure — even a correctly qualified
one, as R1COV's is — would have missed a result that makes DDOR's practical usefulness for K_SRP look
even worse than the information-geometry result alone suggests.

## 49. Academic Counterpart and Literature Comparison

The implemented physics tracks Moyer's precision light-time formulation and Curkendall & Border's DDOR
architecture closely for the terms this phase's scope covers (§5); no claim is made that DDOR itself,
or common-mode calibration rejection, or differential VLBI navigation, are novel — they are
established, decades-flown techniques. The application-specific, project-level finding — that a
literature-consistent DDOR implementation does *not* retain the weak-SRP-parameter observability gain
a simplified surrogate predicted, and that the mechanism is specifically the differential light-time
term the surrogate omitted — is not addressed by the general DDOR literature, which is concerned with
spacecraft-state navigation, not weak non-gravitational force-parameter identifiability. This is a
genuine (if negative) application-specific result, not a restatement of known DDOR behavior.

A thesis-level claim would need, beyond this phase: the quasar-separation and baseline-orientation
sensitivity studies this phase deferred (§34/§35), and ideally a second, independently-designed orbit
geometry to confirm the negative finding is not an artifact of this one synthetic campaign. A
paper-level claim would additionally need the media/clock/instrument calibration models this phase
explicitly declined to fabricate (§23/§27).

## 50. Research / Thesis / Paper Value

```
RESEARCH_VALUE_CLASS = APPLICATION_SPECIFIC_PARAMETER_OBSERVABILITY_RESULT
```

Not inflated to thesis- or paper-level on its own. The finding is a well-evidenced, honestly-reported
negative result with an identified mechanism (§21), which is valuable precisely *because* it corrects
an over-optimistic surrogate-based projection from the immediately preceding phase — but it covers one
observable, one baseline, one orbit geometry, and stops (correctly, by its own gates) short of a
nonlinear-estimation or multi-geometry confirmation that a stronger classification would require.

## 51. Industry / Operational Interpretation

**Would an actual navigation team use this measurement model?** Not for K_SRP estimation specifically,
on this evidence: the qualified production implementation shows no information-direction benefit, and
where a formal benefit might appear it is swamped by session-bias sensitivity two orders of magnitude
below plausible calibration residuals.

**Does the reference RF architecture support DDOR?** Unresolved (§7) — this alone would block a real
implementation decision regardless of the K_SRP finding.

**Would DDOR already be flown for normal state navigation?** Plausible in general (established DSN
practice) but not evidenced by anything in this repository's reference configuration.

**What is the marginal benefit for K estimation?** None found, at the tested baseline/geometry/
precision combination — and where information rises, it is magnitude-only, adding cost (dedicated
dual-baseline scheduling) without adding parameter-specific value.

**What operational calibration residual dominates?** Session-level calibration bias, decisively
(§33) — not random tracking noise (§31).

**Would carrying K as solve-for improve prediction enough to matter, or should a team use a consider
parameter or empirical accelerations instead?** This phase's evidence, combined with R1M/R1COV's,
continues to point toward K_SRP being difficult to justify as a delivered solve-for parameter from
existing or DDOR-augmented radiometric tracking in this geometry; a consider-parameter or empirical-
acceleration treatment remains the more defensible operational posture pending a genuinely different
result from the still-open landmark candidate (§52).

## 52. Landmark Candidate Status

Preserved exactly as R1O left it: `SECOND_CANDIDATE = LUNAR_LANDMARK_LOS`, with R1O's own idealized-
surrogate evidence (0.19% fractional uncertainty in the ideal limit, severely eroded by realistic
attitude/map error) intact and unmodified. Per §66 and this phase's own negative DDOR finding, landmark
LOS is now the more immediately relevant next candidate should this line of investigation continue —
though R1O-D's own core lesson (a surrogate's promise must be re-verified against real physics before
being trusted) applies with equal force there: R1O's landmark surrogate was, like its DDOR surrogate,
an idealized geometric construction, not a physically complete optical-navigation model.

## 53. Other Experiments Available

- Quasar-separation and baseline-orientation sensitivity sweeps (§34/§35), to confirm the negative
  finding is not specific to this one geometry.
- A second, independently-designed orbit/tracking geometry, to test whether the light-time-correct
  DDOR result generalizes or is itself geometry-specific.
- A dedicated clock/instrument common-mode-cancellation numerical injection (§22), beyond the
  analytic argument given here.
- Production ΔDOR with a genuine dual-baseline (2-D plane-of-sky) architecture, since this phase (like
  R1O before it) qualified only one scalar baseline (§33 of the governing spec).
- A production lunar landmark optical measurement model (§52), the documented second candidate.
- Monte Carlo statistical consistency of R1COV's qualified covariance (deferred since R1COV; still
  not run by any phase).

## 54. Why They Come Later

The negative information-geometry finding (§36) is decisive enough, on the evidence gathered, that
further DDOR sensitivity studies would refine a result already established rather than test an open
question; the more valuable next step is a different conceptual layer entirely — landmark LOS — not
more DDOR characterization. Running multiple new observable families or a Monte Carlo campaign
simultaneously would again risk the attribution problem R1M and R1COV both explicitly guarded against.

## 55. Regression Protection

Targeted battery, 16 files (the full R1COV/R1O-relevant set plus `tests/test_delta_dor.py`,
`tests/test_two_way_range.py`, and `tests/test_measurements.py` — the two additional files audit that
the existing one-way light-time infrastructure this phase's forward solver was built alongside is
untouched): **all pass**, identical in character to every prior phase's targeted battery (expected
skips only, zero failures, zero errors).

| gate | result |
|---|---|
| `R1COV_REGRESSION` (square-root covariance path) | PASS |
| `P21_REGRESSION` | PASS |
| `MODEL_S_REGRESSION` | PASS (evidenced by the full suite, §56) |
| `LONG_ARC_REGRESSION` | PASS |
| `EVENT_CONDITIONING_REGRESSION` | PASS |
| `RANGE_REGRESSION` | PASS |
| `COUNTED_DOPPLER_REGRESSION` | PASS |
| `FORCE_K_DERIVATIVE_REGRESSION` | PASS |
| `TRAJECTORY_K_SENSITIVITY_REGRESSION` | PASS |
| `RANGE_K_SENSITIVITY_REGRESSION` | PASS |
| `COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION` | PASS |
| `DDOR_K_SENSITIVITY_REGRESSION` | PASS (this phase's own new tests, §29) |
| `DERIVATIVE_CHAIN_GATE` | PASS |
| `BLS_DEFAULT_PARITY` / `SRIF_DEFAULT_PARITY` / `SRUKF_DEFAULT_PARITY` | PASS |
| `COMMON_GAUSSIAN_POSTERIOR_GATE` | PASS |

23 new permanent tests added in `tests/test_delta_dor.py` (zero baseline, station-swap sign,
common-transmit-event self-consistency, time-tag contract, local-delay conditioning across binades,
quasar analytic delay + roundtrip + edge cases, K-sensitivity linearity, invalid-input handling) —
all pass. Two of these tests initially failed on first write, for the same reason as §18: they
re-derived a light-time quantity via large-epoch subtraction rather than using the solver's own local
residual; both were corrected to compare against local quantities before being accepted as permanent.

```
NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0
```

## 56. Full Suite

```
TOTAL_TESTS_COLLECTED = 1287
TESTS_PASSED          = 1246
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
```

Against R1O's own full-suite baseline (1264/1223/2/10/29), the entire delta is **+23 collected,
+23 passed** — exactly the 23 new tests in `tests/test_delta_dor.py`, and no other change. The
non-pass set is identical, test for test, to every prior phase in this lineage (the same two known
pre-existing classes: the R2 module fixture `NameError`, and the root-`pytest.ini` sibling-worktree
injection failing FA-06).

| non-pass | count | classification |
|---|---:|---|
| `test_r2_measurement_fidelity.py` module fixture — `NameError: CLOSURE_CURRENT_TREE_SHA256` | 10 errors | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_r2_current_tree_protection_holds_for_paths_r3_must_not_change` — Phase 16 `dynamics.py` bytes | 1 failed | `KNOWN_PREEXISTING_PROVENANCE` |
| `test_fa06_pytest_session_imports_isolated_repository` — root-`pytest.ini` sibling-worktree injection | 1 failed | `KNOWN_PREEXISTING_ENVIRONMENT` |
| slow/optional regressions behind `LUNAR_OD_RUN_SLOW_TESTS=1` and optional dependencies | 29 skipped | `KNOWN_PREEXISTING_ENVIRONMENT` |

```
KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1OD_INTRODUCED_NONPASSES               = 0
NEW_SCIENTIFIC_REGRESSIONS              = 0
UNKNOWN_NONPASSES                       = 0
```

## 57. What This Phase Actually Established

Answering §74 directly:

- **Is the implemented observable genuinely ΔDOR, not a differential-range surrogate?** Yes —
  differential *delay* from a common transmit event, not an instantaneous differenced range.
- **Are both stations tied to the same spacecraft transmit event?** Yes, oracle-verified (§13/§18).
- **Is the quasar delay correct?** Yes, oracle-verified (§15).
- **Is the time-tag convention explicit and validated?** Yes (§12).
- **Does local-delay numerics avoid Phase17C-type quantization?** Yes, after a genuine bug was found
  and fixed (§17): drift reduced from `1.3e-4` to `1.4e-10` relative.
- **Are state derivatives correct including event-time dependence?** Yes, FD-verified to `6.8e-6`
  relative after a genuine FD-harness fidelity bug was found and fixed (§28).
- **Does K enter only through trajectory history?** Yes, structurally proven and FD-confirmed to
  `8.2e-5` relative (§29).
- **Do common-mode terms cancel correctly?** Argued analytically; not numerically injected (§22).
- **Which real residuals remain?** Media, clock/instrument, and Earth-orientation-at-nanoradian-
  precision are all explicitly uncharacterized (§23, §25).
- **Does production DDOR increase `f_perp`? Increase `theta_K`? Reduce trustworthy `sigma_K`
  meaningfully?** No, no, and only marginally (1.14× info magnitude, no direction gain) — §36–39.
- **Does that benefit survive realistic calibration residuals?** There was no direction benefit to
  test survival of; the magnitude-only benefit is itself dwarfed by session-bias sensitivity (§33).
- **Does it survive different quasar separation / baseline orientation?** Not tested (§34/§35).
- **Does nonlinear K recovery improve? Does holdout prediction improve?** Not run, correctly gated
  off (§41–45).
- **Is exact RF compatibility confirmed?** No — `UNKNOWN` (§7).
- **Is DDOR operationally realistic?** See §60's classification below.
- **What did we learn academically?** A literature-faithful DDOR implementation does not retain a
  simplified surrogate's weak-force-parameter observability projection; the mechanism (differential
  light-time) is identified and isolated (§21), not merely observed.
- **What did we learn operationally?** Formal covariance and point-estimate bias sensitivity can
  diverge sharply (§33); a covariance-only characterization would have understated the operational
  risk of trusting DDOR-augmented K_SRP.

## 58. What Remains Unknown

- Whether the negative information-geometry result generalizes beyond this one orbit geometry,
  station pair, and synthetic campaign window (§34/§35/§53).
- Whether a genuine dual-baseline (2-D) DDOR architecture behaves differently from the single scalar
  baseline qualified here.
- Whether common-mode clock/instrument cancellation holds up under a real numerical injection, not
  just the analytic argument given (§22).
- Whether this repository's reference spacecraft could fly DDOR at all (§7) — orthogonal to, and
  unresolved regardless of, the K_SRP finding.
- Everything rests on `SYNTHETIC_CAMPAIGN_TRUTH`, never `SPACECRAFT_TRUTH`.

## 59. Decision Tree From Here

Applying §79 (the branch this phase's result actually matches, not §76's full-success path or §77's
systematics-limited path): production DDOR does not reproduce the surrogate's information-direction
gain. This is scientifically valid and was investigated, not dismissed (§21). No modification was made
to the measurement model to force reproduction of R1O's result.

The next rational step is the still-open second candidate from R1O: **Phase 17-R1O-OPT — lunar
landmark LOS production measurement model design and qualification** — carrying forward this phase's
central methodological lesson explicitly: R1O's landmark surrogate must be re-verified against real
optical-navigation physics (camera geometry, attitude coupling, landmark-map uncertainty) before its
own promise is trusted, exactly as DDOR's was not.

## 60. Final Verdict

```
START_BRANCH = feature/phase17-r-k-srp-estimation
START_HEAD   = 9f18d96407647150e6eb48fd2b967db152e89dc3
START_TREE   = e18399d79e1600266d832b69fe2e7011403ea60c

R1OD_INPUT_GATE = PASS

R1O_NUMERIC_CONSISTENCY_GATE = PASS
R1O_ERRATUM_CLASS = DOCUMENTATION_TRANSCRIPTION_ONLY

DDOR_LITERATURE_CONTRACT_GATE = PASS

REFERENCE_SPACECRAFT_DDOR_RF_CAPABILITY = UNKNOWN

DDOR_OBSERVABLE_UNIT = seconds of differential delay
DDOR_SIGN_CONVENTION = QUALIFIED
DDOR_TIME_TAG_CONTRACT = QUALIFIED
DDOR_QUASAR_MODE = single fixed catalog direction (minimum defensible architecture, s23)

GOLDSTONE_CANBERRA_DUAL_VISIBILITY_REPRODUCTION = PASS

DDOR_COMMON_TRANSMIT_EVENT_GATE = PASS
DDOR_GEOMETRIC_ORACLE_GATE      = PASS
SPACECRAFT_DOR_EVENT_ORACLE_GATE = PASS
QUASAR_DELAY_ORACLE_GATE        = PASS

DDOR_LOCAL_DELAY_NUMERICAL_CONDITIONING_GATE = PASS
    (bug found and fixed: 1.265e-04 -> 1.377e-10 relative drift across binades)

COMMON_MODE_CLOCK_CANCELLATION_GATE      = CHARACTERIZED_ANALYTICALLY_NOT_INJECTED
COMMON_MODE_INSTRUMENT_CANCELLATION_GATE = CHARACTERIZED_ANALYTICALLY_NOT_INJECTED

MEDIA_CALIBRATION_PHYSICS = CHARACTERIZED_NOT_FULLY_IMPLEMENTED

DDOR_STATE_JACOBIAN_GATE           = PASS  (max rel err 6.78e-06, after fixing an FD-fidelity bug)
DDOR_IMPLICIT_EVENT_DERIVATIVE_GATE = PASS

DIRECT_MEASUREMENT_K_DEPENDENCE = NONE

DDOR_K_COMPOSITION_SENSITIVITY_GATE = PASS
DDOR_K_E2E_SENSITIVITY_GATE         = PASS  (rel err 8.21e-05)

R1O_SURROGATE_REDUCED_MODE_BRIDGE = PASS (executed; isolates the cause: differential light-time)

R1COV_DDOR_COVARIANCE_PATH_GATE = PASS  (rel err 1.0e-15)

PRODUCTION_RANGE_ONLY_CONDITIONAL_K_INFO = 1.162817e+06
PRODUCTION_RANGE_DDOR_CONDITIONAL_K_INFO = 1.329505e+06

PRODUCTION_RANGE_ONLY_F_PERP = 0.2523
PRODUCTION_RANGE_DDOR_F_PERP = 0.2121

PRODUCTION_RANGE_ONLY_THETA_K = 14.61 deg
PRODUCTION_RANGE_DDOR_THETA_K = 12.25 deg

PRODUCTION_RANGE_ONLY_SIGMA_K = 9.273514e-04
PRODUCTION_RANGE_DDOR_SIGMA_K = 8.672714e-04

PRODUCTION_RANGE_ONLY_FRACTIONAL_SIGMA_K = 0.0927
PRODUCTION_RANGE_DDOR_FRACTIONAL_SIGMA_K = 0.0867

PRODUCTION_DDOR_INFORMATION_DIRECTION_GATE = FAIL
PRODUCTION_DDOR_INFORMATION_DIRECTION_CLASS = MAGNITUDE_ONLY

DDOR_RANDOM_NOISE_ROBUSTNESS_CLASS      = NOT_APPLICABLE (no beneficial regime at any tested precision)
DDOR_SYSTEMATIC_ERROR_ROBUSTNESS_CLASS  = NOT_ESTABLISHED (media/clock not numerically characterized)
DDOR_SESSION_BIAS_SENSITIVITY_CLASS     = EXTREME (1 ns bias -> 22.5% of K_truth shift)
DDOR_QUASAR_SEPARATION_SENSITIVITY_CLASS = NOT_RUN
DDOR_BASELINE_ORIENTATION_SENSITIVITY_CLASS = NOT_RUN

BLS_DDOR_K_ESTIMATE      = NOT_RUN_BY_GATE
BLS_DDOR_SIGMA_K         = NOT_RUN_BY_GATE
BLS_DDOR_NORMALIZED_K_ERROR = NOT_RUN_BY_GATE
BLS_DDOR_SOLVE_FOR_GATE  = NOT_RUN_BY_GATE

SRIF_DDOR_K_ESTIMATE = NOT_RUN_BY_GATE
SRIF_DDOR_SIGMA_K    = NOT_RUN_BY_GATE
SRIF_DDOR_NORMALIZED_K_ERROR = NOT_RUN_BY_GATE
BLS_SRIF_DDOR_SCIENTIFIC_CONSISTENCY_GATE = NOT_RUN_BY_GATE

SRUKF_DDOR_STATUS = DEFERRED_ARCHITECTURE

RANGE_ONLY_HOLDOUT_RMS     = NOT_RUN_BY_GATE
RANGE_DDOR_HOLDOUT_RMS     = NOT_RUN_BY_GATE
CORRECT_FIXED_K_HOLDOUT_RMS = NOT_RUN_BY_GATE
WRONG_FIXED_K_HOLDOUT_RMS  = NOT_RUN_BY_GATE
DDOR_HOLDOUT_PREDICTION_GATE = NOT_RUN_BY_GATE

DDOR_OPERATIONAL_REALISM = NOT_ESTABLISHED
    (RF capability unknown; no K-specific information benefit found; session-bias
     sensitivity extreme; media/clock/quasar-separation/baseline-orientation all
     uncharacterized)

RESEARCH_VALUE_CLASS = APPLICATION_SPECIFIC_PARAMETER_OBSERVABILITY_RESULT
ACADEMIC_LITERATURE_ALIGNMENT = CONSISTENT_WITH_ESTABLISHED_DDOR_THEORY
    (the IMPLEMENTATION follows Moyer/Curkendall-Border; the NEGATIVE K_SRP finding
     is the application-specific, non-literature-derived result)
OPERATIONAL_RECOMMENDATION_CLASS = DO_NOT_PURSUE_DDOR_FOR_K_SRP_ON_THIS_EVIDENCE

PRODUCTION_CODE_CHANGED               = YES (lunar_od/delta_dor.py, new file only)
PHYSICAL_FORCE_MODEL_CHANGED          = NO
EVENT_SOLVER_CHANGED                  = NO (new, separate solver; existing one untouched)
NEW_MEASUREMENT_MODEL_ADDED           = YES
ESTIMATOR_MEASUREMENT_INTERFACE_CHANGED = NO
K_SOLVE_FOR_MATH_CHANGED              = NO
COVARIANCE_METHOD_CHANGED             = NO
DEFAULT_6STATE_BEHAVIOR_CHANGED       = NO

R1COV_REGRESSION                         = PASS
P21_REGRESSION                           = PASS
MODEL_S_REGRESSION                       = PASS
LONG_ARC_REGRESSION                      = PASS
EVENT_CONDITIONING_REGRESSION            = PASS

RANGE_REGRESSION          = PASS
COUNTED_DOPPLER_REGRESSION = PASS

FORCE_K_DERIVATIVE_REGRESSION            = PASS
TRAJECTORY_K_SENSITIVITY_REGRESSION      = PASS
RANGE_K_SENSITIVITY_REGRESSION           = PASS
COUNTED_DOPPLER_K_SENSITIVITY_REGRESSION = PASS
DDOR_K_SENSITIVITY_REGRESSION            = PASS
DERIVATIVE_CHAIN_GATE                    = PASS

BLS_DEFAULT_PARITY             = PASS
SRIF_DEFAULT_PARITY            = PASS
SRUKF_DEFAULT_PARITY           = PASS
COMMON_GAUSSIAN_POSTERIOR_GATE = PASS

TOTAL_TESTS_COLLECTED = 1287
TESTS_PASSED          = 1246
TESTS_FAILED          = 2
TESTS_ERRORS          = 10
TESTS_SKIPPED         = 29
    (R1O baseline 1264/1223/2/10/29; entire delta is the 23 new
     tests\test_delta_dor.py tests, non-pass set identical test for test)

KNOWN_PREEXISTING_PROVENANCE_NONPASSES  = 11
KNOWN_PREEXISTING_ENVIRONMENT_NONPASSES = 30
R1OD_INTRODUCED_NONPASSES                = 0
NEW_SCIENTIFIC_REGRESSIONS               = 0
UNKNOWN_NONPASSES                        = 0

MAIN_CHANGED        = NO
ORIGIN_MAIN_CHANGED = NO

REPORT_COMPLETENESS_GATE = PASS
    all 61 mandatory sections present. Full s72 twelve-part format applied to
    sections 13-21, 26-29 (the qualification-critical tests, including the
    two bugs found and fixed); abbreviated (purpose/method/results/verdict)
    for sections whose setup/system-impact duplicate an adjacent section's,
    and NOT_RUN_BY_GATE sections state the gate reason rather than a full
    template.

PHASE17_R1OD_GATE = CHARACTERIZATION_COMPLETE
    (physics fully qualified; information-direction gate FAILED; this is
     PHASE17_R1OD_GATE = CHARACTERIZATION_COMPLETE under s77/s79, not PASS
     under s76's full-success bar, which required a retained K benefit)

PRIMARY_CLASS = R1O_SURROGATE_DID_NOT_GENERALIZE_TO_PRODUCTION_DDOR

NEXT_ACTION =
  PHASE_17_R1O_OPT (lunar landmark LOS production measurement model design
  and qualification), carrying forward this phase's central lesson: verify
  the surrogate against real physics before trusting its promise. Not begun
  automatically; requires separate authorization.

COMMITS_CREATED = <recorded after local commit>
COMMIT_LIST     = recorded in artifacts/r1od_manifest.json

PUSH = NONE
MERGE = NONE
MAIN_MODIFICATION = NONE
HISTORY_REWRITE = NONE
FORCE_PUSH = NONE
```

## 61. Exact Next Action

**STOP** (§81). No DDOR production wiring into the estimators, no optical-navigation implementation,
no production measurement-physics change beyond the new standalone `delta_dor.py` module, no Monte
Carlo campaign, no default-six-state covariance refactor, no merge, no push, no change to main.

The phase's question, answered directly:

> Does a physically correct, calibration-aware production ΔDOR observable retain the parameter-
> specific K_SRP observability advantage predicted by R1O, and is that advantage scientifically and
> operationally credible?

**No, on both counts, and the physics qualification behind that "no" is itself complete and trustworthy.**
The production observable is correctly built — every oracle, Jacobian, and K-sensitivity gate passed,
including two genuine numerical bugs found during qualification and fixed rather than argued around.
But once that correct physics replaced R1O's simplified surrogate, the information-direction gain the
surrogate predicted did not survive: `f_perp` fell slightly below the range-only control rather than
rising toward 0.42–0.44, and the mechanism — differential light-time across the two station legs,
which the surrogate's simultaneous-epoch approximation discarded — was isolated by direct
construction, not merely inferred. A second, independent finding makes DDOR's case for K_SRP weaker
still: session-level calibration bias at the nanosecond scale, far below plausible DSN residual
floors, induces point-estimate shifts many times the truth value, a risk the formal covariance alone
would not have revealed.

Recommended next step: **Phase 17-R1O-OPT**, applying this phase's central methodological lesson —
verify a surrogate against real, physically complete measurement models before trusting its scientific
promise — to R1O's still-open second candidate, lunar landmark LOS. Neither that phase nor any
production estimator wiring for DDOR begins without separate, explicit authorization.
