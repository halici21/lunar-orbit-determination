# Phase 17 closeout — D27, final stochastic-K experiment, model-mismatch pivot

- **Date:** 2026-10-12
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`), on top of D29 `e35e245`
- **Truth level:** `SYNTHETIC_CAMPAIGN_TRUTH` throughout. K0 = 0.01 m²/kg is a campaign value.
- **Commits:**
  - `f932625`: D27 code and tests.
  - `5caa519`: KGM predeclaration, committed **before** any high-fidelity GM result.
  - `b60edc5`: D27 qualification.
  - `07988e7`: model-mismatch probes.
  - The commit carrying this report: KGM results.

## 0. Final status

```
PHASE17_HARDEN_D27_GATE              = PASS (docs/phase17_harden_d27_convergence_status.md); opens HARDEN-D30
STOCHASTIC_K_RESULT                  = SUPPORTS_M3_VARIATION_RESULT (predeclared rule, applied as frozen)
COMMON_K_LEVEL_STATUS                = NOT_CLAIMED; now with a physical reason: lunar albedo + IR pressure is
                                       ~99.99 % collinear with it (s3.1)
VARIATION_RECOVERY_STATUS            = PARTIAL and geometry-dependent, as KP2: recovered CANON k0 and O1 k12,
                                       not O2 k0 (information-limited), identically under GM15 and M3
FALSE_VARIATION_STATUS               = 0 / 3 cells under GM15 (M3-projected and full 14-dof); 0 / 2 in the
                                       tau sensitivity
PROCESS_MODEL_SENSITIVITY            = LOW: tau = 1, 5, 15 orbits give the same T0 / T1 outcomes on CANON k0
M3_COMPARISON_STATUS                 = AGREES 3 / 3 cells; GM sigma_v 0.88-1.47 x M3 (never prior-sharpened);
                                       on a GM-shaped truth M3 is INCONSISTENT on CANON (chi2_err 39.8) while
                                       GM15 is consistent (full chi2_err 4.7 / 14 dof)
K_RESEARCH_THREAD_STATUS             = CLOSED_WITH_CAVEAT. Caveat: every K-variation result is truth = model in
                                       the dynamics; s3 shows the dominant real mismatch (gravity field) can
                                       masquerade as K variation
NEXT_PRIMARY_SCIENCE_QUESTION        = Is there a masquerade window in which lunar gravity-field error is
                                       statistically invisible in range + counted-Doppler residuals yet produces
                                       significant false K_SRP variation, and does GRAIL-level field error lie
                                       inside it?
RECOMMENDED_MODEL_MISMATCH_TARGET    = Moon-fixed gravity-field model error (omission above the estimation
                                       degree, coefficient and frame/tide-convention differences), separated
                                       from Sun-fixed nonconservative (SRP-variation) error
MINIMUM_NEXT_EXPERIMENT              = s3.4: T0 (constant K), 4 KP2 cells, truth field GL1800F +
                                       alpha (GRGM1200L - GL1800F), alpha in {1, 0.3}, estimated with GL1800F by
                                       the unchanged KP2 M3 solver: 8 nonlinear solves
NEXT_PHASE_RECOMMENDATION            = run s3.4 (predeclared) before any further K work; no new K
                                       parameterisation, no Monte Carlo, no tracking-design search
```

## 1. Work package A — HARDEN-D27 (summary)

At damping exhaustion, BLS-LM now evaluates the undamped Gauss-Newton decrement of its scored objective at the
returned point, at the caller's tolerance.

- **≤ 0.1 posterior σ:** `Converged` / `STATIONARY_AT_DAMPING_LIMIT`.
- **Otherwise:** `DampingLimit`, which now means "not certified stationary".
- **Effects:** general across BLS-LM paths, estimates bitwise unchanged.
  - The frozen high-fidelity KP2 production solution is now certified (decrement 1.9e-3).
  - Genuine failures stay failures (2511; truth-start stall 3.68).
- **New open item, HARDEN-D30.** Integration noise in the acceptance cost (~5 χ² at rtol 1e-11) rejects small
  damped steps. No authoritative result is affected.
- **Full report:** `docs/phase17_harden_d27_convergence_status.md`.

## 2. Work package B — final stochastic-K experiment (KGM)

### 2.1 Design (frozen: `artifacts/phase17_kgm_predeclaration.json`, sha256 `6fd5419f…`)

- **Model GM15.** K_j = K0 (1 + f_j) on 15 one-orbit segments, realised exactly by segmented production
  propagation. It is the KP2 construction generalised; anchor: N = 3 reproduces KP2 **bitwise**, and the 15 K
  columns sum to the scalar column to 7e-9 (`artifacts/phase17_kgm_anchor.json`).
- **Stochastic model.** First-order Gauss-Markov deviations with a **flat (no-information) common level**. The prior
  information is Λ = C⁻¹ − C⁻¹11ᵀC⁻¹/(1ᵀC⁻¹1), of rank 14. The GM prior constrains variation only, so
  common-level identifiability is untouched. It is zero-mean, so it can only shrink a variation, never create
  one.
- **Hyperparameters, declared from rules.**
  - τ = T_arc/3 = 5 orbits: M3's resolution, the like-for-like comparison.
  - σ = 0.30: ≥ 3.7× every KP2 posterior σ_v, taken from KP2 posteriors, not from the truths.
  - The implied prior σ on the M3-projected variation is 0.18–0.22, against KP2 posterior σ_v of 0.007–0.081.
- **Truths.**
  - T0: constant.
  - T1: the KP2 step.
  - T3_GM: one GM realisation with σ 0.10 and **τ 2 orbits, deliberately not the model τ**.
- **Cells.** One per KP2 behaviour: CANON k0 (strong), O2 k0 (information-limited), O1 k12 (marginal).
- **Seeds and comparison.** The KP2 seeds are reused, and M3 on T0/T1 is read from the frozen KP2 results. M3 on
  T3_GM was solved here.
- **Runs.** 16 high-fidelity solves, all converged (5–7 evaluations).

### 2.2 Results (`artifacts/phase17_kgm_summary.json`, `phase17_kgm_raw_results.json`)

| cell | T1 GM15 χ²_err / χ²_detect | T1 M3 (KP2) | GM σ_v / M3 σ_v | DFS / 14 | T0 GM χ²_detect (M3-proj. / full 14-dof) | agree |
|---|---|---|---:|---:|---|---|
| CANON k0 | 3.6 / **89.8** recovered | 4.4 / 193 recovered | 1.47 | 0.86 | 0.21 / 7.5 | yes |
| O2 k0 | 2.0 / 0.9 missed | 0.9 / 1.3 missed | 0.88 | 0.29 | 0.18 / 1.2 | yes |
| O1 k12 | 1.4 / **22.9** recovered | 1.0 / 25.4 recovered | 1.06 | 0.60 | 1.54 / 6.1 | yes |

Thresholds: χ²₂(0.999) = 13.82 and χ²₁₄(0.999) = 36.12. The common-level z on T0 is −1.4, 0.8 and 1.0: consistent,
and **reported only**.

**Classification (predeclared).** Agreement 3/3, false variations 0/3, prior flags 0/3, T1 converged 3/3 →
**`SUPPORTS_M3_VARIATION_RESULT`**.

- **Not prior-manufactured.** Wherever GM recovers, the data determine 60–86 % of the 14 variation modes (DFS). Its
  σ_v is equal to or larger than M3's in the recovering cells; the GM prior never sharpened the variation.
- **Where the prior dominates (O2, DFS 0.29), GM, like M3, detects nothing.** The prior shrinks; it does not create.
- **GM-shaped truth (T3_GM, reported, not part of the class).**
  - **CANON k0.** GM15 reconstructs the variation with skill 0.73 and is consistent (full χ²_err 4.7, detect 325).
    M3 detects strongly (233) but is **inconsistent** (χ²_err 39.8): three segments cannot represent within-third
    variation.
  - **O1 k12.** GM skill 0.45 (full detect 25.6, below the 14-dof threshold). M3 does not detect (10.9).
  - **O2.** Neither detects (skill −0.17).
  - The stochastic representation is the more faithful one when the truth is not piecewise-thirds. Recovery remains
    geometry-limited exactly where KP2 found it to be.
- **τ sensitivity (CANON k0, predeclared, no selection).** τ = 1 orbit gives T1 detect 86.0 and T0 full detect 8.0;
  τ = 15 orbits gives 96.5 and 6.7. The outcome does not depend on τ.

### 2.3 K-thread closeout

The time variation of K is partially recoverable from range + counted Doppler with free per-pass biases. That holds
under the deterministic M3 representation and under a stochastic GM representation that neither knows the
segmentation nor carries information on the level. The result is geometry-dependent (O2 full Sun: no). The common K
level is not claimed (§3.1 adds the physical reason).

**`K_RESEARCH_THREAD_STATUS = CLOSED_WITH_CAVEAT`.** Every K result so far is truth = model in the dynamics. §3
shows that the dominant real mismatch can alias into exactly the K-variation directions. The thread is not reopened
for a new K parameterisation; the caveat is the next phase's question.

## 3. Work package C — model-mismatch pivot

Linear probes, labelled DIAGNOSTIC (`examples/phase17_mm_probe.py`; `artifacts/phase17_mm_probe*.json`).

- **Signature construction.** Each candidate mismatch's trajectory signature is built by STM convolution,
  S(t) = Φ(t)∫Φ(τ)⁻¹[0; δa(τ)]dτ. It is checked by applying the same construction to the production SRP kernel
  against production dx/dK: 0.03–7 % relative (trapezoid across shadow edges), adequate for directions and orders of
  magnitude.
- **Evaluation.** Columns go through the production range and counted-Doppler Jacobians on the KP2 estimator's
  design: x₀ with its a priori, M3 K, and one free bias per pass.
- **Metrics.**
  - Separability sine against that design.
  - Unabsorbed residual χ², i.e. detectability.
  - What the mismatch does to an SRP-only M3 solve: false-variation χ² (2 dof), level and state bias.

### 3.1 Lunar albedo + thermal IR pressure — rejected as the target, explains the K level

The force model is a Lambertian sub-spacecraft-point approximation (A_b 0.12, T_ss 390 K, T_n 95 K) on the same
cannonball. Its mean magnitude is 0.06–0.50 × SRP; at the subsolar point it is ≈ SRP.

| metric (10 cells, simplified; CANON k0 also high fidelity) | value |
|---|---|
| sine against x₀ + M3 K + biases | **0.001–0.032** (HF 0.012): ≥ 99.9 % absorbed |
| σ(c_PR) if estimated alongside M3 | 0.12–8.3: not estimable |
| 10 % mismodel → false variation χ² | ≤ 2.6 (none) |
| 10 % mismodel → common-level bias | −1.4 % … −2.5 % of K = 0.4–2.8 σ |
| 10 % mismodel → residual χ² / state bias | ≤ 0.7 / < 0.05 m |

Planetary radiation pressure is invisible to the data and lands almost entirely on the common K level. That level
is the direction already not claimed, and it now has a physical reason: in this geometry SRP K is degenerate with
lunar albedo/IR pressure, so an absolute K is an "effective radiation-pressure" coefficient at best. The variation
directions are untouched. Another radiation-pressure scale would be one more tunable parameter that teaches nothing,
so it fails the selection criteria.

### 3.2 Gravity-field error — selected

| probe (simplified; CANON k0 also high fidelity) | mean accel / SRP | sine | residual χ² | false-variation χ² | ratio FV / resid |
|---|---:|---:|---:|---:|---:|
| GL1800F degrees 51–100 omitted (10 cells) | 450–810 (O3 500 km: ~0) | 0.05–0.47 | 6e6–6e7 | 1e6–7e7 | 0.04–1.9 |
| GL1800F degrees 101–200 omitted (4 cells) | 20–38 | 0.02–0.11 | 6e4–4e5 | 4e3–9e4 | 0.03–0.51 |
| GRGM1200L − GL1800F, same degree 100 or 200 (4 cells) | 0.84–2.8 | 0.02–0.37 | 1.0e3–3.4e3 | 16–1380 | 0.02–0.81 |

- **Magnitude.** At 100 km, Moon-fixed field error is 1–3 orders of magnitude larger than the SRP signal whose
  10 % variation KP2 recovers. Its linear alias into K variation is enormous.
- **The decisive number is the ratio.** False-variation χ² and residual χ² both scale as (error amplitude)². Their
  ratio (0.02–0.81, ≥ 0.1 in most cells) is amplitude-independent.
  - A field error scaled until it produces χ²_detect = 13.82 adds only 13.82 / ratio ≈ **17–700 χ²** (CANON
    13–110) to ~1,800–2,600 residuals.
  - That is at or below the χ² test's own scatter (√(2N) ≈ 60–72).
  - So a **masquerade window exists in the linear model**: gravity error invisible in residuals that still
    "detects" a K variation at 99.9 %.
- **Field-difference caveat.** The GRGM1200L − GL1800F difference is the same at degree 100 and 200. It is
  dominated by degrees 2–6 relative differences of 1e-4…6e-3; C21 is 2.3e-10 in GL1800F against −6.9e-9 in
  GRGM1200L. That is the signature of **frame-realisation / tide-system conventions**, not formal commission error.
  It is still a realistic Moon-fixed mismatch (which convention an operational model inherits), but it is not a
  GRAIL error estimate.

### 3.3 Decision

| criterion | gravity-field error (selected) | albedo / IR | empirical RTN accelerations | per-pass measurement bias |
|---|---|---|---|---|
| physical relevance at LLO | dominant (20–800× SRP for omitted degrees; ~3× for field conventions) | real, ~SRP size | generic absorber | real; CD1 / KP2 characterised |
| observable signature | Moon-fixed, ground-track-locked; detectable at full size | none (≥ 99.9 % absorbed) | by construction anything | per-pass constant |
| distinct from state and nuisance | partly (sine 0.02–0.47) | no | tunable | KP2: MINOR for variation |
| beyond the common-K question | yes: decides whether the K-variation finding survives model error | no (collapses onto the K level) | no physics learned | already done |
| validation path in this stack | yes: production harmonic propagation, two independent GRAIL fields in repo | yes | yes | done |
| real / independent data | GRAIL fields (GL1800F, GRGM1200L, GRGM660PRIM in repo); LRO / GRAIL DSN tracking and reconstructed orbits (PDS, NAIF) | weak | — | — |

**Why this target.** It is the only candidate that is dominant, has a distinct (Moon-fixed vs Sun-fixed)
signature, and whose answer is about **model error**: whether the one positive K finding survives it. It needs no
new tunable parameter: the error amplitude α is the controlled variable.

**Main confounders.**

- Omission of the truth field itself (degree truncation of the "truth").
- Nonlinearity at large error amplitude: the linear alias numbers are indicative only above α ~ 0.1.
- HARDEN-D30 / integrator noise floor near convergence; the KP2 solver's 1e-2 σ criterion.
- Per-pass Doppler bias (absorbs part of any per-pass-constant signature).
- Initial-state error.
- Altitude (O3 at 500 km is immune).
- Frame-convention vs coefficient error inside the field difference (§3.2 caveat).
- Albedo/IR (lands on the level, not the variation).

### 3.4 Minimum next experiment (falsification)

- **Hypothesis H_mm.** A Moon-fixed gravity-field error whose residual signature is statistically undetectable
  produces a significant false M3 K variation.
- **Design.**
  - Truth: K constant (T0), KP2 seeds, cells CANON k0, O1 k12, O2 k0, O5 k0.
  - Truth field: GL1800F 50×50 + α (GRGM1200L 50×50 − GL1800F 50×50), α ∈ {1, 0.3}.
  - Estimator: the KP2 M3 solver unchanged, with GL1800F 50×50.
  - 8 nonlinear solves, plus the existing truth = model KP2 T0 baseline.
- **Measure.**
  - χ²_detect (false variation, 13.82).
  - Residual excess against the baseline: Δχ², whitened rms, and a run test per pass.
  - Level and state bias.
- **Outcomes.**
  - **H_mm survives (masquerade)** if, at some α, any cell has χ²_detect ≥ 13.82 while Δχ² ≤ 2√(2N). Then K
    variation from tracking is not claimable without independent gravity-field knowledge.
  - **H_mm is falsified** if false variation appears only with a clearly detectable residual excess, or never. Then
    the data distinguish Moon-fixed from Sun-fixed error and the K-variation finding is robust to this class.
- **Predeclare before running:** α values, cells, thresholds and the residual test. Linear prediction:
  α = 1 is detectable with large false variation (CANON, O1, O2); α = 0.3 puts CANON near the window.

**Real / independent data path.**

- **In the repo now:** three independent GRAIL fields allow model-to-model error proxies without new data.
- **Next:** LRO or GRAIL DSN radiometric tracking (PDS Geosciences radio-science archives) with the NAIF
  reconstructed LRO orbit (LOLA-improved, ~10 m) as independent truth. The minimum experiment does not need it, so
  no stop condition applies.

## 4. Traceability of authoritative results

| result | status after this task |
|---|---|
| KP1 `ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS` | unchanged |
| KP2 `M3_VARIATION_PARTIALLY_RECOVERABLE` | unchanged; independently supported by a stochastic representation (§2) |
| BLS1 / D25 / D28 / D29 | unchanged; BLS1's "status label is not a convergence indicator" caveat is resolved by D27 |
| §1 canonical K_SRP baseline | unchanged (no estimate changed) |
| common K level | not claimed; reason strengthened (§3.1) |

## 5. Verification

- **Focused (worktree):** D27, LM consistency, estimators, BLS1, D25/D28/D29, K-SRP BLS and scenarios. All pass.
- **Full suite:** clean detached worktree of `fd24147` (only committed files; the concurrent desktop-app edits
  in the shared worktree are excluded): **1497 passed, 29 skipped, 0 failed** (exit 0). Record:
  `artifacts/phase17_closeout_verification.txt`.
