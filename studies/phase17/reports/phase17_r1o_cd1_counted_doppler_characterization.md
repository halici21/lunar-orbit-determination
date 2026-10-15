# Phase 17-R1O-CD1 — Production Counted-Doppler K_SRP Complementarity Characterization

Truth level: **SYNTHETIC_CAMPAIGN_TRUTH** (K_truth = 0.01 m²/kg, one constant scalar K). No DSN flight performance is
claimed.

Artifacts: `artifacts/phase17_r1o_cd1_*`. Code: `examples/phase17_r1o_cd1_{core,predeclare,campaign}.py`. Tests:
`tests/test_r1o_cd1_counted_doppler.py`. Literature: `studies/phase17/reports/phase17_r1o_cd1_literature_contract.md`.

> **Closure (2026-10-04, §49–§50).** HARDEN-D23 was measured over the full primary population and moves no CD1
> decision. The §32 rebuild exceedance was adjudicated against an exact rational oracle and is the rebuild route's own
> float error. `PHASE17_R1O_CD1_GATE = PASS`. The scientific verdict is unchanged. Closure code, artifacts and tests:
> `studies/phase17/scripts/phase17_r1o_cd1_closure.py`, `artifacts/phase17_r1o_cd1_closure{,_raw}.json`,
> `tests/test_r1o_cd1_closure.py`.

## 1. Executive Summary

**Counted Doppler adds a real K_SRP direction in the ideal case, but that gain does not survive a free per-pass
Doppler bias.**

**`PRIMARY_CLASS = COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE` (Case D).**

- **Ideal Doppler (D0), high-fidelity primary population, 120 cells:**
  - The predeclared population rule passes, but only just: **76.7 %** of cells gain direction (rule ≥ 75 %) and
    **72.5 %** are MODEST or STRONG (rule ≥ 50 %).
  - Median Δf_perp is **+0.068**.
- **With one free constant bias per station pass (D1, the predeclared primary nuisance model):**
  - Only **44.2 %** of cells stay positive, and median Δf_perp is **−0.006**.
  - Among cells with an ideal gain, the median retained fraction is **0.04**, so `NUISANCE_RETENTION_CLASS = LOST`.
- **Magnitude and direction.** Doppler improves σK/K enormously: the range-only median is 0.32, and range + Doppler
  gives 0.005. That is magnitude. Direction stays weak:
  - the median combined f_perp is 0.21 (ideal) and 0.13 (D1);
  - 116 of 120 cells are WEAK_K_SEPARATION in the ideal case;
  - 120 of 120 are WEAK under D1.
- **Dynamics fidelity changes the verdict.** Under simplified dynamics, the same D0 population *fails* the rule (70 %
  positive). The change is again carried mostly by the range-only baseline (correlation of Δ(Δf_perp) with
  −Δ(range f_perp): r = 0.92), as in D4.
- **Mechanism.** The Doppler K signature is the line-of-sight projection of the SRP-induced velocity perturbation
  (correlation 0.9999). The K-specific part accumulates over the whole arc, with most of it arriving late.
- **Verification:**
  - end-to-end FD gates **PASS** (state ≤ 7.2e-7, K ≤ 7.0e-6);
  - all contracts and all 104 range-baseline anchors pass exactly;
  - one analysis-layer bug was found by the rebuild and fixed (CD1-IBF-01). It did not change any decision field.
  - **The predeclared independent-rebuild tolerance is exceeded in 1 of 720 comparisons**, by 1.46e-10 against a
    1e-10 gate. That comparison is a sensitivity configuration, not a decision input, and it was adjudicated as
    round-off in the rebuild route (§32). It is reported as a strict gate failure, not relabelled.

`NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED = NO`. `ALTERNATIVE_K_PARAMETERIZATION_JUSTIFIED = YES`, by the
predeclared rule for Case D.

## 2. Scientific Question

Does the already-qualified production two-way counted-Doppler observable add a K_SRP-specific measurement direction
beyond range, under high-fidelity lunar dynamics? And does it do so robustly across epoch, orbit and physically
plausible Doppler systematics, rather than in isolated cells?

## 3. Why Counted Doppler Is Next

- **ΔDOR did not deliver.** D3 and D4 showed that ΔDOR gain is narrow, calibration-sensitive, orbit-sensitive and
  fidelity-sensitive.
- **Range + ΔDOR did not either.** D4 also showed that a relative-gain metric can move when the *range* baseline
  changes.
- **Doppler is the natural next observable.** SRP is an acceleration, whose signature appears first in velocity, and
  Doppler measures the rate of change of range. It is a routine observable, already implemented and qualified.
- **Why not change K first.** Reparameterizing K adds no measurement information. So CD1 characterizes the observable
  first and keeps K a constant scalar.

## 4. Entering Authoritative State

| item | value |
|---|---|
| branch | `feature/phase17-r-k-srp-estimation` |
| HEAD / tree | `cfa1ce7293c60024c9ee04dfeca801ef7ba72c5d` / `d9bfc226dce6d6514f5f848f9cfd92894fafdc29` |
| main = origin/main | `fea476f81dad07b3914e53eab709e10fa6e9d10b` |
| entry suite | 1430 collected / 1401 passed / 0 failed / 0 errors / 29 skipped (verified) |
| gates | HARDEN, F1 (SRP variational model QUALIFIED), D3 and D4 all PASS, from their committed reports |

## 5. Literature Contract

`studies/phase17/reports/phase17_r1o_cd1_literature_contract.md` separates three kinds of statement:

- **LITERATURE_RESULT:** DSN 810-005 Module 202 Rev. E; Moyer, DESCANSO Monograph 2, §10.2.3.1.2 and §13.3; Mazarico
  et al. 2012 (LRO); AAS 22-728 (NRHO SRP).
- **REPOSITORY_HYPOTHESIS:** Doppler observes the SRP velocity signature in a direction the initial state cannot
  reproduce.
- **REPOSITORY_RESULT:** the production-model facts.

No literature statement is used as evidence that Doppler identifies K here. `CD1_LITERATURE_GATE = PASS`.

## 6. DSN Module 202 Status

- **Official index:** it lists "CL 22-3475 (for 202E)".
- **Document:** `810-005/202/202E.pdf` (sha256 `29a66259…4708`), "Rev. E, Issue Date: July 19, 2022". Requests for
  202F and 202G redirect to a generic page.
- **Statements established from it:**
  - Doppler is related to range rate;
  - Doppler is derived from accumulated carrier phase;
  - two-way uses a coherent turnaround;
  - the ground frequency reference is more stable than a spacecraft oscillator;
  - σ_V² = thermal + phase-noise + solar-scintillation terms, which depend on T, carrier-loop SNR and the
    Sun–Earth–probe angle.
- **Scintillation constant:** X-up/X-down C_band = 2.7e-6, read from the rendered Table (31).

## 7. Repository Counted-Doppler Model

- **Path:** production `lunar_od.radiometrics.two_way_counted_doppler_observable` and `…_initial_state_jacobian`.
- **Model:** the default R3 "single_bounce_exact_station", which equals the R4 four-event model at zero transponder
  delay (R4-P08).
- **Formulation:** y = c/(2Tc)·[τ(t3e) − τ(t3s)], two round-trip light-time solves at the ground receive epochs of the
  count endpoints. This is Moyer's F2 structure, without ramps, media or relativity.
- **Scope:** used read-only. No production code was changed.

## 8. Four-Event Timing Contract

Per count endpoint, the events are t1 (ground transmit), t2u = t2d (spacecraft turnaround, zero delay) and t3 (ground
receive, fixed). The receive epochs are t_mid ± Tc/2.

- **Station states:** from the exact event-epoch J2000→ITRF93 `sxform`, with the Earth centre linearly interpolated
  (R3-P22 / HARDEN-D23). One provider is shared by both endpoints.
- **Spacecraft:** Hermite state and STM interpolation, under the FA-03B closed-support guards.
- **Independent check:** the R4 four-event module, which uses M3's uniform 3×3 implicit sensitivity, reproduced the
  production rows in every high-fidelity cell. State agreement is 7.6e-13 relative; K is 1.7e-14 of the column
  maximum.

## 9. Observable Units and Frequency Semantics

- **Units:** `mps_equivalent`, the rate of change of the one-way range in m/s. This is 202E's σ_V convention
  (σ_f = 2 f_C σ_V / c).
- **Frequencies:** X-band uplink 7.2 GHz, turnaround 880/749, so the downlink is 8.459 GHz.
- **K column:** the production Jacobian with dx/dK placed in the first STM-column slot (the HARDEN/F1 construction).
  The arc can start in eclipse, where dx/dK ≡ 0 until eclipse exit. That is physical, not a defect.

## 10. Existing Derivative Qualification

- **HARDEN:** counted-Doppler state columns ≤ 1.3e-6; K 2.2e-4.
- **F1:** state 9.4e-7; K 2.2e-4 → 8.7e-6 at the resolving tolerance (DOPPLER_DERIVATIVE_GATE PASS).
- **CD1:** re-checked end to end under high-fidelity dynamics (§31).

## 11. Predeclaration

- **File and hash:** `studies/phase17/configs/phase17_r1o_cd1_predeclaration.json`, sha256
  **`bbaa5cee8687926699465de6fe75185fde87b10df840be652ba390ff7af3571a`**.
- **Commit:** `21962b7`, before any CD1 K-information metric.
- **Enforcement:** the campaign pins the hash and refuses to run on a mismatch; tamper tests pass.
- **What was measured before the predeclaration:** only non-K quantities. These were feasibility, row counts, the R3/R4
  Jacobian agreement, and the range-K rebuild route (2.7e-9). The last is why that gate is 1e-7.
- **Code changes after the predeclaration:** none changed a definition.
  - The decision logic was refactored into unit-tested functions. It reproduced the same verdict.
  - The worker count was reduced for memory.
  - The CD1-IBF-01 fix (§32).

## 12. High-Fidelity Primary Dynamics

- **Primary model:** `phase17_r1od2_fidelity.propagate_fidelity`, used for every orbit × epoch cell.
  - Force model: GL1800F 50×50 (MOON_PA_DE440), Earth and Sun third bodies, production SRP with the F1 position
    gradient, conical penumbra.
  - STM: harmonic STM opt-in, rtol 1e-12.
- **Control:** simplified dynamics (`propagate_nom48`) on the same 120 cells, used only for decomposition.
- **Reuse:** one propagation per cell serves every schedule, count interval, noise case and nuisance case.

## 13. Epoch Grid

- **Grid:** the D3 grid (hash `a651c4de`), 24 epochs 15° apart over one synodic month, with k = 0 exactly the
  canonical epoch.
- **Recomputed per epoch:** Sun, Earth, stations, visibility, range and Doppler.

## 14. Orbit Set

The initial states are frozen by x0 sha256, as in D4:

- CANON;
- O1_LOW_BETA_ECLIPSE_RICH (G1_b00);
- O2_FULL_SUN_HIGH_BETA (G1_b80);
- O3_HIGH_ALTITUDE_500KM (G2_alt500_b20_fixD);
- O5_FAVOURABLE_GEO_MAX_FPERP (G3_i26.7_bcan).

The full 5 × 24 high-fidelity matrix ran: 120 of 120 cells are feasible.

## 15. Tracking Schedule

- **PRIMARY ALL_VISIBLE:** one row per (90 s grid node, visible DSN3 station). This is the production candidate rule.
  A row is kept only if every grid node bracketing its count interval is in the same visibility window, using
  elevation plus lunar occultation.
  - Across the 120 high-fidelity cells, 4 435 of 124 046 candidates were dropped by that rule. No history-domain
    drops occurred.
  - Per cell: 669–1 689 Doppler rows (median 912), 707–1 701 range rows, 4–25 passes.
- **SECONDARY REGULAR_SESSION:** the centred hour of each pass that lasts at least an hour (median 626 rows).
- **What was not done:** no K-informed selection and no schedule search.

## 16. Count Interval

- **Primary:** Tc = 60 s at a 90 s cadence. This is the production default and the HARDEN/F1-qualified value; Moyer
  gives "e.g. 60 s or 600 s".
- **Sensitivity:** Tc = 10 s (90 s cadence) and Tc = 540 s (540 s cadence).
- **Overlap:** counts never overlap, so the white-noise assumption holds.
- **Model:** always the counted observable, never an instantaneous range rate.

## 17. Noise Model

- **Primary (structural):** σ_D = 1.0e-4 m/s at 60 s. This is the repository DSN `sigma_range_rate_mps`, and it lies
  inside the 202E X/X scintillation span at 60 s (2.1e-5 to 1.8e-4 m/s).
- **Bracket:** 1e-5 to 1e-3 m/s.
- **Tc scalings:** T^−0.175 (scintillation) and T^−1 (thermal).
- **OPERATIONAL_LOWER_BOUND:** per row, 202E Eq. (30) at the row's Sun–Earth–spacecraft angle, RSS the H-maser floor.
  This gives 2.1e-5 to 5.4e-4 m/s, with a median of the per-cell medians of 2.1e-5.
  - Thermal noise is PARAMETRIC (the project has no spacecraft RF hardware), so this is optimistic.
- **Range:** σ 5 m, the production station value.

## 18. Nuisance Model

- **D0_IDEAL:** no nuisance.
- **D1_SESSION_CONSTANT_BIAS:** one range-rate bias per station pass. It stands in for a frequency offset or a slowly
  varying media offset over a pass.
- **D2_SESSION_BIAS_PLUS_LINEAR_DRIFT:** D1 plus a normalized linear drift. A single-row pass carries no drift column
  (CD1-IBF-01).
- **Priors:** FREE, and a labelled synthetic bracket of 1e-5, 1e-4 and 1e-3 m/s.
- **Primary:** **D1 FREE**. No operational prior is claimed.

## 19. Range-Only Baseline

- **Recomputation:** recomputed on every propagation.
- **Anchors:** it reproduces the D3 canonical simplified range f_perp at all 24 epochs, and D4's high-fidelity and
  simplified range f_perp at 80 orbit × epoch cells (k = 4…11). Both match with a difference of **0.0**.
- **Median high-fidelity range f_perp:** 0.126; per orbit 0.09–0.29.
- **Median range σK/K:** 0.32.

## 20. Doppler-Only Information

- **Median Doppler-only f_perp:** 0.209 (D0) and 0.133 (D1 FREE).
- **Per orbit (D0):** CANON 0.159, O1 0.194, O2 0.225, O3 0.225, O5 0.262.
- **Relation to range:** Doppler alone is more K-orthogonal than range alone in most cells, the exceptions being CANON
  and O5, which have strong range baselines.

## 21. Range + Doppler Information

- **Median combined f_perp:** 0.2095 (D0) and 0.133 (D1 FREE). It is **equal to Doppler-only to three decimals**.
- **What that means:** at σ_D = 1e-4 m/s against a 5 m range, Doppler dominates the information. The combined
  K-orthogonal energy comes essentially 100 % from the Doppler rows (0.994–1.000 in the representative cells).
- **Median θ_K:** 12.1°.
- **Weakest design mode:** its K component is 1.000, so K remains the weakest direction.
- **Median condition number:** 1.8e8 (range-only 2.2e8).

## 22. Epoch Robustness

Median D0 Δf_perp across orbits, by epoch:

- above **+0.10** at k = 0, 1, 10, 11, 22 and 23;
- negative at k = 15 and 16 (−0.007, −0.020);
- near zero at k = 5 (+0.005).

The gain varies with the synodic geometry, but most epochs are positive in the ideal case.

## 23. Orbit Robustness

| orbit | D0: positive / modest+ / median | D1 FREE: positive / modest+ / median |
|---|---|---|
| CANON | 0.46 / 0.38 / −0.019 (POP no) | 0.00 / 0.00 / −0.065 |
| O1 | 0.83 / 0.83 / +0.101 (POP yes) | 0.50 / 0.17 / −0.001 |
| O2 | 0.96 / 0.92 / +0.115 (POP yes) | 0.67 / 0.38 / +0.004 |
| O3 | 0.96 / 0.92 / +0.078 (POP yes) | **0.96 / 0.75 / +0.059 (POP yes)** |
| O5 | 0.62 / 0.58 / +0.028 (POP no) | 0.08 / 0.08 / −0.089 |

- **Where Doppler helps least:** the canonical orbit and the previously favourable O5, both of which have high
  range-only f_perp.
- **Where nuisance-robust gain survives:** only O3, the 500 km orbit.

## 24. Population Robustness

Over 120 high-fidelity cells, using the predeclared POP rule:

- **POP(D0) = TRUE:** 0.767 positive, 0.725 modest+, 0.642 strong, 0.192 loss, median +0.068, range −0.157 to
  +0.196.
- **POP(D1 FREE) = FALSE:** 0.442 / 0.275 / 0.158 / 0.358, median −0.006.
- **ABS:** the fraction of cells with combined f_perp ≥ 0.35 under D1 FREE is **0.00**, so ABS = FALSE.

## 25. Count-Interval Sensitivity

| Tc | scaling | D0: positive / modest+ / POP | D1 FREE: positive / median |
|---|---|---|---|
| 10 s | T^−0.175 | 0.77 / 0.72 / yes | 0.44 / −0.006 |
| 10 s | T^−1 | 0.77 / 0.73 / yes | 0.55 / +0.005 |
| 540 s | either | 0.73 / 0.67 / **no** | 0.28 / −0.03 |

- **Short counts:** these barely change the ideal geometry.
- **Long counts:** with 1/6 of the rows, the positive fraction falls below the rule, and they are more
  nuisance-fragile.

## 26. Noise Sensitivity

- **Under D0**, the population is nearly invariant across the ×100 bracket: from 1e-5 to 1e-3 m/s the positive
  fraction is 0.77–0.79 and the median is +0.067 to +0.068. Doppler dominates range throughout that span.
- **Under D1 FREE**, only the noisiest case, 1e-3 m/s, is less negative (0.65 positive, median +0.018), because there
  the range carries relatively more weight.
- **OPERATIONAL_LOWER_BOUND** (optimistic, per-row 202E scintillation):
  - D0: 0.73 positive, POP fails;
  - D1 FREE: 0.45 positive.

No noise choice rescues the nuisance case.

## 27. Nuisance Sensitivity

| model / prior | positive | modest+ | median Δ | POP |
|---|---|---|---|---|
| D0 | 0.77 | 0.72 | +0.068 | yes |
| D1, 1e-5 m/s | 0.75 | 0.70 | +0.063 | yes |
| D1, 1e-4 m/s | 0.57 | 0.35 | +0.007 | no |
| D1, 1e-3 m/s | 0.44 | 0.28 | −0.006 | no |
| **D1 FREE** | 0.44 | 0.28 | −0.006 | **no** |
| D2, 1e-5 m/s | 0.75 | 0.70 | +0.062 | yes |
| D2, 1e-4 m/s | 0.53 | 0.30 | +0.005 | no |
| D2 FREE | 0.39 | 0.26 | −0.010 | no |

- **When the gain survives:** only if every pass's bias is known a priori to about **0.1×** the white Doppler noise
  (1e-5 m/s). A prior equal to the noise level (1e-4) already removes most of it.
- **Correlations:** the median K–bias correlation is 0.48 under D1 FREE, and the median K–drift correlation is 0.19
  under D2 FREE.
- **σK/K inflation:** under D1 FREE the median rises to 0.0092, from 0.0050 under D0.
- **Retention:** median 0.04 over the 87 cells with an ideal gain above 0.02, so **LOST**.
- **Session schedule:** it is worse. REGULAR_SESSION under D1 FREE is positive in only 6 % of cells.
- **Interpretation:** a pass bias is a constant range-rate offset over a pass. The loss is consistent with the
  K-orthogonal Doppler signature being nearly constant within a pass, which is plausible because SRP velocity
  build-up is slow compared with a pass. This was not decomposed further.

## 28. Simplified vs High-Fidelity Decomposition

Changes going from SIMPLIFIED to HF, RMS over 120 cells, D0:

| quantity | RMS change |
|---|---|
| range-only f_perp | 0.025 |
| Doppler-only f_perp | 0.011 |
| combined f_perp | 0.011 |
| Δf_perp | 0.028 |

- **Correlation:** corr(Δ(Δf_perp), −Δ(range f_perp)) = **0.92**.
- **Verdict:** the simplified population fails POP(D0) (0.70 positive, 0.67 modest+), while high fidelity passes it.
  `FIDELITY_EFFECT_CLASS = FIDELITY_CHANGES_POPULATION_VERDICT`.
- **Answer to the brief's question:** the Doppler block itself is fairly fidelity-stable (±0.011). Most of the change
  in the *relative* gain comes from the range baseline moving. The same pattern appeared in D4.
- **Consequence:** the marginal pass of POP(D0) under high fidelity (76.7 % against 75 %) is partly a baseline effect.

## 29. Time Localization

These are the representative cells, CANON at k = 0, 6, 12 and 18, high fidelity, D0; this is interpretation only.

- **Accumulation:** the cumulative combined I(K|x) reaches only **9–14 %** of its final value at 50 % of the arc.
- **Largest slice:** no single 10 % slice adds more than 20–35 %, so every representative cell is **BROAD** by the
  predeclared rule.
- **Reading:** the K-specific information accumulates steadily but late, consistent with a secular SRP velocity and
  position build-up. It does not come from short geometry windows.
- **f_perp over time:** the cumulative combined f_perp peaks around mid-arc and then declines slightly
  (e.g. 0.170 → 0.159 at k = 0). As the arc grows, the state absorbs part of the signature.

## 30. Physical Signature Interpretation

- **Doppler:** the Doppler K column correlates **0.9999** with LOS·∂v/∂K, the geocentric line-of-sight projection of
  the SRP-induced velocity perturbation. Doppler therefore observes the velocity signature exactly as hypothesized.
- **Range:** the range K column correlates 0.9999 with LOS·∂r/∂K.
- **Why the direction stays weak:** the six-state includes the initial velocity, and most of the velocity signature
  is reproduced by initial-state adjustments. Only θ_K ≈ 12° remains orthogonal.
- **Pass edges:** rows within ±10 min of a pass start or end are ~29 % of rows but carry ~50 % of the Doppler
  orthogonal energy. Pass edges and handoffs are enriched.
- **Eclipse diagnostic:** uninformative. With about 30 eclipse transitions per arc and ~2 h orbits, a ±1 h window
  covers every row. This diagnostic was predeclared and is reported as non-discriminating.

## 31. End-to-End FD Spot Checks

- **Cases:** predeclared CANON k = 0 and O2 k = 12.
- **Setup:** 2-orbit high-fidelity arcs, 6 Doppler rows each.
- **Analytic side:** the production Jacobian at rtol 1e-12.
- **Oracle:** the production observable on repropagated high-fidelity trajectories at rtol 1e-13, central differences,
  two steps per column.
- **Oracle wrapper check:** the oracle wrapper at rtol 1e-12 reproduces `propagate_fidelity` **bitwise**, so the force
  model is identical.

| column | CANON k0 | O2 k12 |
|---|---|---|
| x0 / y0 / z0 | 4.3e-7 / 1.7e-7 / 2.2e-7 | 2.0e-7 / 3.7e-7 / 6.9e-8 |
| vx0 / vy0 / vz0 | 3.8e-7 / 5.7e-7 / 4.7e-7 | 7.2e-7 / 3.4e-7 / 3.7e-7 |
| K_SRP | 1.7e-6 | 7.0e-6 |

`COUNTED_DOPPLER_STATE_FD_GATE = PASS` (≤ 1e-5). `COUNTED_DOPPLER_K_FD_GATE = PASS` (≤ 1e-3).

## 32. Independent Design Reconstruction

- **f_perp:** rebuilt by a `numpy.lstsq` route for 6 configurations × 120 high-fidelity cells = 720 comparisons.
- **Doppler rows:** rebuilt through the R4 four-event module: 7.6e-13 / 1.7e-14 (gate 1e-9), **pass**.
- **Range K rows:** rebuilt through the event-epoch STM-slot route: 4.8e-8 (gate 1e-7), **pass**.
- **Whitening, nuisance ownership, visibility, Earth audit:** all pass.
- **Anchors:** 104/104 exact.

### CD1-IBF-01 (analysis-layer bug found by the rebuild, fixed, rerun)

- **Symptom:** in the first analysis, `RPD|D2|FREE` differed from the rebuild by up to 1.26e-5 in 13 cells. Every
  other configuration agreed to ≤ 1e-14.
- **Cause:**
  - A pass with a single Doppler row made its D2 drift column all-zero.
  - `numpy.linalg.qr` returns an arbitrary orthonormal column for a zero column, so the wrapper's projector removed a
    spurious direction.
  - The wrapper f_perp came out slightly low.
- **Fix:** such passes carry no drift column, because a drift is undefined on a single row. A regression test was
  added.
- **Rerun:** the 13 affected cells were recomputed (`studies/phase17/results/phase17_r1o_cd1_ibf01_rerun.json`).
  - Only `RPD|D2|*` configurations changed, by at most 1.3e-5 in f_perp.
  - Range, rows and every other configuration are **bitwise identical**.
- **Decision fields:** before and after, every one is identical: PRIMARY_CLASS, both populations, ABS, nuisance,
  fidelity, justification flags.
- **Predeclaration:** unchanged.

### Remaining strict exceedance

- **The case:** after the fix, **1 of 720** comparisons exceeds the predeclared 1e-10: O2 k = 6, `RPD|D2|FREE`,
  |diff| = **1.46e-10**. This cell has no single-row pass, so the fix does not affect it.
- **Adjudication on the same design:**
  - Wrapper QR, full SVD projector and pivoted QR all give 0.121367102360.
  - Both `lstsq` routes, with default truncation and with no truncation, give 0.121367102506.
  - No singular value is below `lstsq`'s cutoff, and the scaled design condition is 1.4e11.
- **Conclusion:** the rebuild route itself forms its residual from ill-conditioned coefficients, so the round-off
  lies in the check, not in the metric.
- **What it touches:** D2 FREE is a nuisance-sensitivity configuration and not a decision input. Its population
  median is −0.0097 either way.
- **Strict status:** by the predeclared rule, **`INDEPENDENT_REBUILD_GATE = FAIL` (1/720 at 1.46e-10)**. It is
  reported as such, not relabelled.

## 33. Absolute K Separation

| nuisance | WEAK | MODERATE | STRONG |
|---|---|---|---|
| D0 | 116 | 3 | 1 |
| D1 FREE | 120 | 0 | 0 |

- **Best cell:** O5 k = 0 under D0, combined f_perp 0.504.
- **Conclusion:** K does not become identifiable in the f_perp sense, even where σK/K is small.

## 34. Relative Gain

| population | STRONG | MODEST+ | LOSS |
|---|---|---|---|
| HF D0 | 64 % | 72.5 % | 19 % |
| HF D1 FREE | 16 % | 27.5 % | 36 % |
| SIMPLIFIED D0 | 62 % | 67 % | 23 % |

The relative and absolute labels are reported separately throughout (`phase17_r1o_cd1_primary_results.csv`).

## 35. Regression Tests

`tests/test_r1o_cd1_counted_doppler.py` has 19 tests, all passing:

- predeclaration hash and tamper refusal;
- 5 × 24 matrix and x0 freeze;
- high-fidelity primary and FD-wrapper identity;
- visibility windows and count-interval brackets;
- Tc mapping and non-overlap;
- K-history construction;
- 202E noise values and units;
- nuisance ownership, including the D2 drift;
- session renumbering;
- REGULAR_SESSION rule;
- population boundaries;
- decision tree order and partition;
- nuisance and fidelity classes;
- Phase-17 thresholds;
- canonical range anchor;
- real-geometry schedule contract;
- production and four-event row identity;
- rebuild agreement;
- the **CD1-IBF-01 regression**.

The targeted regression set also passes: counted-Doppler, measurement, F1, HARDEN, SRP, D2, D3, D4 — 403 tests.

## 36. Full Suite

`python -m pytest -o addopts="" -p no:cacheprovider`:

| | collected | passed | failed | errors | skipped |
|---|---|---|---|---|---|
| entry | 1430 | 1401 | 0 | 0 | 29 |
| CD1 | **1449** | **1420** | **0** | **0** | **29** |

- The 19 new tests are exactly the CD1 tests.
- The 29 skips are the same opt-in slow regressions as at entry.
- `NEW_SCIENTIFIC_REGRESSIONS = 0`, `UNKNOWN_NONPASSES = 0`.
- No production path was touched, so no slow tests needed re-running.

## 37. Final Diff Review

**CLEAN.** Since the entering HEAD `cfa1ce7`, the only changed paths are the CD1 files:

- `examples/phase17_r1o_cd1_{core,predeclare,campaign}.py`
- `tests/test_r1o_cd1_counted_doppler.py`
- `docs/phase17_r1o_cd1_*.md`
- `artifacts/phase17_r1o_cd1_*`

What did not change: `lunar_od/`, the measurement model, the physical model, the estimator and the K
parameterization. There are no debug leftovers. The scratch files (`%TEMP%\cd1`, the rerun driver and the
keep-awake helpers) are removed at closure.

## 38. Academic Interpretation

- **Range and Doppler respond to different parts of the SRP signature.** Range responds to the accumulated position
  displacement. Counted Doppler responds to the change in light time over Tc, which is essentially the line-of-sight
  velocity signature. CD1 confirms that the Doppler K column *is* that velocity signature (r = 0.9999).
- **Most of it is still mimicked.** The six-state contains the initial velocity, and the ideal orthogonal fraction
  stays at f_perp ≈ 0.21 (θ_K ≈ 12°).
- **What Doppler does add** in the ideal case is a small but broadly distributed extra K direction over range. It is
  far more robust across epoch than ΔDOR was: POP(D0) holds.
- **What removes it:** that extra direction is almost constant within a pass. A per-pass range-rate bias, the
  lowest-order coherent Doppler systematic, absorbs most of it.
- **Magnitude is not identifiability.** The ~60× σK/K improvement is information magnitude. Treating it as K
  identifiability would repeat the R1O error the erratum corrected.

## 39. Operational Interpretation

- **Why the ideal result is attractive:** range and coherent Doppler are routine radiometric tracking, so an ideal
  Doppler direction gain would be operationally more attractive than narrow ΔDOR geometry.
- **Why it doesn't translate:** the gain depends on knowing every pass's Doppler bias to about 1e-5 m/s, which is 0.1×
  the white noise. Real calibration of media and frequency offsets at that level over every pass is not something
  this project can claim.
- **Result:** Doppler improves the K *magnitude* estimate a great deal, but it does not make K separable from the
  state under realistic systematics.
- All numbers are SYNTHETIC_CAMPAIGN_TRUTH.

## 40. Remaining Limitations

- Linear information analysis only. No estimator was run, as the brief required.
- One constant scalar K. The range carries no nuisance; that is the D2–D4 convention.
- Doppler noise is structural, since the thermal term is PARAMETRIC. The operational lower bound is optimistic.
- DSN3 stations only, a 15-orbit arc, and five representative orbits.
- No ramps, media or relativistic terms in the observable. Linear Earth interpolation (HARDEN-D23, ~1.5e-6 m/s).
  **Erratum (closure §49.1):** the end-to-end four-event effect on the observable *value* is ~1.7–1.8e-3 m/s, not
  ~1.5e-6 m/s. It is immaterial to every CD1 metric, because those use only the partials.
- The nuisance priors are a synthetic bracket, not DSN values.

## 41. Remaining Unknowns

- Whether a physically constrained Doppler systematic model would preserve part of the gain. Examples: a bias
  correlated between passes, or a media model with a DSN-calibrated prior.
- Whether longer arcs or multi-arc linkage change the within-pass/secular balance.
- Why O3 (500 km) keeps its gain under D1 FREE while the low orbits do not. It likely comes from a slower pass
  geometry relative to the SRP signature, but that was not decomposed.
- The marginal POP(D0) (76.7 % against 75 %) and its fidelity dependence.

## 42. What This Phase Actually Established

- **Model identity:** the production counted-Doppler model is the Moyer F2 structure, and its analytic state and K
  derivatives pass end-to-end high-fidelity FD checks.
- **Ideal gain:** ideal counted Doppler adds a broadly distributed K direction across epochs and orbits
  (POP(D0) true, marginally).
- **Nuisance fragility:** a free per-pass Doppler bias removes that gain (retention 0.04). Only a 1e-5 m/s prior
  preserves it.
- **Absolute separation:** it stays WEAK in 116–120 of 120 cells.
- **Fidelity:** it changes the population verdict, mostly through the range baseline.

## 43. What This Phase Did NOT Establish

- Any nonlinear estimation result or K recovery.
- DSN flight Doppler performance, or real bias calibration levels.
- Anything about reparameterized K, stochastic K or empirical accelerations.
- That no observable could identify K. Only counted Doppler, with range, under this formulation.

## 44. Decision Tree

The predeclared order is A→E.

| case | condition | result |
|---|---|---|
| A | POP(D0) ∧ POP(D1F) ∧ ABS | **no** |
| B | POP(D0) ∧ POP(D1F) ∧ ¬ABS | **no**: POP(D1F) false |
| C | ¬POP(D0) ∧ … | **no**: POP(D0) true |
| D | POP(D0) ∧ ¬POP(D1F) | **yes** |

**`PRIMARY_CLASS = COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE`.**

## 45. Final Verdict

- **H1** (more robust than ΔDOR): **partly.** In the ideal case, yes: the population rule holds, unlike ΔDOR's narrow
  windows.
- **H2** (persists across epoch and orbit under high fidelity): **marginally**, for D0 only.
- **H3** (survives nuisance freedom): **no.**

`ROBUST_COMPLEMENTARITY_FOUND = NO`.

**Phase gate.** Every strict pass criterion of the brief is met except one: "independent rebuild passes". By the
predeclared 1e-10 tolerance, 1 of 720 comparisons fails at 1.46e-10 (§32). So **`PHASE17_R1O_CD1_GATE` is not
reported as PASS**. It stands as FAIL on that single criterion until the user accepts the adjudicated exception or
requests a remedy. The scientific verdict does not depend on it. *(Resolved in §49.2: the gate is PASS.)*

## 46. Whether Nonlinear Confirmation Is Justified

**NO.** Only Cases A and B could justify it, and CD1 reached Case D.

## 47. Whether Alternative K Parameterization Is Now Justified

**YES**, as the next *characterization* step, by the predeclared rule for Cases C, D and E. CD1 supplies its
observable basis:

- Range and counted Doppler observe K's magnitude well (σK/K ≈ 0.5 %).
- Only ≈ 21 % of the K-signature norm is orthogonal to the state subspace (θ_K ≈ 12°).
- The residual direction is killed by per-pass biases.

A reparameterization study must show that it does not simply split this weak direction into weaker ones. It should
test systematics jointly, including a per-pass Doppler bias.

## 48. Exact Next Action

**Hard stop after CD1.** Do not start nonlinear estimation.

- **Recommended next phase, separately authorized:** an *alternative K_SRP parameterization characterization*.
  - Use the CD1 range + counted-Doppler (ALL_VISIBLE, Tc 60 s) observable set and the D1 per-pass bias as the
    reference nuisance.
  - Predeclare its own population rule.
- **Before that, for the user to decide:** accept or remediate the strict rebuild-gate exceedance in §32 (1/720 at
  1.46e-10, in a non-decision configuration, adjudicated as rebuild-route round-off). *(Resolved in §49.2.)*

## 49. Closure (2026-10-04)

This section is analysis only. The predeclaration (`bbaa5cee…`), decision tree, thresholds, noise and nuisance models
are unchanged and hash-checked. No production code changed. Script: `studies/phase17/scripts/phase17_r1o_cd1_closure.py`.

### 49.1 HARDEN-D23 materiality, measured

- **Method.** Every primary cell (5 orbits × 24 epochs × {HIGH_FIDELITY, SIMPLIFIED} = 240) was rebuilt twice on
  the same propagation.
  - The first build is exactly as in CD1.
  - The second uses a reference station-state provider: production state − linear Earth + SPICE Earth at the exact
    event epoch.
  - SPICE Earth here is `spkezr EARTH / J2000 / NONE / MOON`, the source the 90 s grid was sampled from.
  - Only the D23 term changes. The exact `sxform` station term is untouched.
  - The provider is injected through the production `station_state_provider` seam.
- **Control.** An identity substitution (− linear + linear) reproduces production bitwise. Reference and linear Earth
  agree to 0.0 m at grid nodes. The closure's production rows reproduce every committed CD1 f_perp exactly
  (max |diff| 0.0) over 1,561,143 station-state events.

| quantity (240 cells) | max |
|---|---:|
| Earth position, linear vs exact | 2.80 m |
| Earth velocity, linear vs exact | 8.2e-6 m/s |
| counted-Doppler observable value | **1.83e-3 m/s** (per-cell max 1.67–1.83e-3 in every cell; per-cell RMS ≈ max, a near-constant offset) |
| state partials, relative | 2.4e-7 |
| K partial / column max | 2.4e-9 |
| f_perp, any decision configuration | **4.0e-10** (median 1.4e-11) |
| closest decision-threshold margin | 4.65e-5 (115,000× the perturbation) |

- **Decision.**
  - Class or sign changes: none.
  - Every decision field is identical with the reference Earth: PRIMARY_CLASS (Case D), POP(D0) = PASS (0.7667),
    POP(D1 FREE) = FAIL, ABS = FAIL, NUISANCE_RETENTION = LOST, simplified POP(D0) = FAIL,
    FIDELITY_CHANGES_POPULATION_VERDICT.
  - `HARDEN_D23_CD1_MATERIALITY_GATE = PASS`.
- **Correction to the HARDEN quantification (REPOSITORY_RESULT).** HARDEN's ~1.5e-6 m/s
  (`phase17_harden_events_and_derivatives.py`) differenced only the symmetric receive endpoints t_mid ± 30 s, and only
  the x component.
  - At those endpoints the parabolic linear-interpolation error cancels.
  - The four-event observable also uses the uplink transmit epochs at t_mid ± 30 s − τ (τ ≈ 2.57 s). These sit
    asymmetrically inside their grid intervals. In the canonical example the errors are 2.456 m and 2.254 m.
  - The ~0.2 m imbalance over a 60 s count is a ~1.7e-3 m/s near-constant offset in the observable value.
- **Consequence.**
  - For CD1 this is immaterial: the metrics use only the partials.
  - For any later phase that uses counted-Doppler *values* (residuals, nonlinear estimation) it is about 17× the CD1
    structural σ (1e-4 m/s). It needs a separately authorised production-model qualification first.

### 49.2 The §32 numerical floor, adjudicated exactly

The f_perp of the identical float64 design was computed with zero arithmetic error. The method uses the exact Gram
matrix of the whitened [state | nuisance | K] design and integer Bareiss elimination:
f² = det(G) / (det(G_AA)·G_bb).

| cell (`RPD|D2|FREE`) | cond | CD1 metric − exact | SVD projector − exact | `lstsq` rebuild − exact |
|---|---:|---:|---:|---:|
| O2 k = 6 (§32 exceedance) | 1.4e11 | 5.6e-17 | 9.7e-17 | **1.46e-10** |
| O5 k = 14 (2nd largest) | 9.8e10 | 8.3e-16 | 8.3e-16 | 3.6e-11 |
| O5 k = 21 (3rd largest) | 2.0e11 | 1.7e-15 | 1.7e-15 | 3.2e-11 |

- **Interpretation.**
  - The CD1 headline metric is exact to ≤ 1.7e-15.
  - The whole 1.46e-10 is the error of the *check* route: `lstsq` forms its residual from coefficients solved on a
    cond ≈ 1e11 design.
  - It is not a bookkeeping or model discrepancy, so nothing needs fixing.
  - The three largest of 720 comparisons form one continuous distribution in the same ill-conditioned nuisance-
    sensitivity configuration. No decision input is involved.
- **Status.**
  - The nominal rebuild rule stays 1e-10. This one comparison is recorded as a bounded numerical-floor exception.
  - With that, `INDEPENDENT_REBUILD_GATE = PASS_WITH_DOCUMENTED_NUMERICAL_FLOOR_EXCEPTION`, and every strict pass
    criterion of the brief is met.

## 50. Final Status

```text
PHASE17_R1O_CD1_GATE = PASS
PRIMARY_CLASS = COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE (Case D)

ROBUST_COMPLEMENTARITY_FOUND = NO

NONLINEAR_ESTIMATOR_CONFIRMATION_JUSTIFIED = NO
ALTERNATIVE_K_PARAMETERIZATION_JUSTIFIED = YES

HARDEN_D23_CD1_MATERIALITY_GATE = PASS (max f_perp change 4.0e-10 vs 4.65e-5 closest margin; 0 class changes;
                                  observable-value effect 1.8e-3 m/s recorded, not 1.5e-6)

NUMERICAL_FLOOR_EXCEPTION_STATUS = ACCEPTED_BOUNDED (1/720 at 1.46e-10 > 1e-10; exact oracle places it entirely in
                                   the lstsq check route; CD1 metric exact to 5.6e-17; nominal 1e-10 rule retained)

PRODUCTION_CODE_CHANGED = NO
MEASUREMENT_MODEL_CHANGED = NO
PHYSICAL_MODEL_CHANGED = NO
ESTIMATOR_CHANGED = NO
K_PARAMETERIZATION_CHANGED = NO

TOTAL_TESTS_COLLECTED = 1460
TESTS_PASSED = 1431
TESTS_FAILED = 0
TESTS_ERRORS = 0
TESTS_SKIPPED = 29 (the same opt-in/data-dependent set as at entry)

NEW_SCIENTIFIC_REGRESSIONS = 0
UNKNOWN_NONPASSES = 0

FINAL_HEAD = the closure commit that adds this section (a commit cannot record its own hash); its parent is
             the import-isolation commit, on CD1 campaign commit 6d805d5
FINAL_TREE = the tree of that closure commit (`git rev-parse <closure-commit>^{tree}`)

PUSH = NONE
MERGE = NONE
MAIN_CHANGED = NO
ORIGIN_MAIN_CHANGED = NO

NEXT_ACTION = hard stop. Recommended, separately authorised: an alternative K_SRP parameterization
              characterization (§47-§48). Before any phase that uses counted-Doppler VALUES, separately qualify
              or repair HARDEN-D23 (§49.1).
```

*(Later status: KP1 completed the parameterization step, and Phase 17-HARDEN-D23 repaired the Earth interpolation.
CD1's results re-qualify unchanged on the repaired path: `studies/phase17/reports/phase17_harden_d23_repair_qualification.md` §6.)*

The suite counts are from the supported plain invocation (`python -m pytest -o addopts="" -p no:cacheprovider`, no
PYTHONPATH) on the final tree.
