# Phase 17 — authoritative status

**This is the single place to find which Phase 17 number is current.** Where any other document disagrees
with this page, this page wins. Every phase report keeps its historical text and carries erratum blocks
that point here.

- **Issued by:** Phase 17-HARDEN (2026-09-26). **Updated by:** Phase 17-F1 (2026-09-26): SRP position gradient in
  the variational equations; canonical numbers below re-qualified, every class preserved.
- **Updated by:** the Phase 17-R1O-CD1 closure (2026-10-04). It adds the D3, D4 and CD1 status rows (§3) and corrects
  the HARDEN-D23 size (§4). No §1 number changed: D3, D4 and CD1 answer different questions with their own configurations.
- **Updated by:** Phase 17-KP1 (2026-10-04). It adds the KP1 row (§3). No §1 number changed.
- **Updated by:** Phase 17-HARDEN-D23 (2026-10-07). The counted-Doppler Earth interpolation is repaired (§3, §4). No §1
  number changed.
- **Updated by:** Phase 17-KP2 (2026-10-08). It adds the KP2 row (§3) and HARDEN-D25 (§4). No §1 number changed.
- **Updated by:** Phase 17-HARDEN-BLS1 (2026-10-10). It repairs production BLS K_SRP recovery (HARDEN-D26, §3) and adds
  HARDEN-D27 (§4). No §1 number and no KP2 classification changed.
- **Updated by:** Phase 17-HARDEN-D25 (2026-10-11). Posterior covariance now linearised on the solve's own dynamics
  (§3, §4); supersedes uncertainty values of the gravity campaign stages G/I (§3 row). No §1 number, no estimate and no
  KP2 classification changed.
- **Updated by:** Phase 17-HARDEN-D28 (2026-10-12). No-prior posteriors are data-only (§3, §4); D25 owner-accepted
  (gate PASS). No §1 number, no estimate and no KP2 classification changed; no authoritative number superseded.
- **Updated by:** Phase 17-MM-G (2026-10-16). GL1800F-internal omission as mismatch truth: `MMG_INCONCLUSIVE` as computed
  (no solve converged; L2 coarse-grid gate failed, repaired); no masquerade from GL1800F's own truncation (§3); no number in
  §1 and no production code changed.
- **Updated by:** Phase 17-FE-P (2026-10-15). External gravity prior (GL1800F published sigmas): `FE_PRIOR_TOO_WEAK_TO_SUPPRESS`
  (§3; as-computed gate veto INCONCLUSIVE via P2; owner accepted the P2 adjudication 2026-10-21, gate PASS); no number
  in §1 and no production code changed.
- **Updated by:** Phase 17-FE-L (2026-10-15). Field-error parameterisation (linear): `FE_SUPPRESSES_BUT_COSTS_VARIATION` (§3);
  no number in §1 and no production code changed.
- **Updated by:** Phase 17-MM (2026-10-14). Gravity-field masquerade test: `H_MM_SURVIVES_MASQUERADE` (§3); no number in §1 and
  no production code changed.
- **Updated by:** Phase 17-HARDEN-D29 (2026-10-12). The SRIF K path solves data-only state + the required K prior
  (§3, §4); no authoritative consumer, no §1 number and no KP2 classification changed.
- **Updated by:** Phase 17-HARDEN-D27 (2026-10-12). BLS-LM damping exhaustion is certified by the Gauss-Newton
  decrement (§3, §4); HARDEN-D30 opened. No §1 number, no estimate and no KP2 classification changed.
- **Updated by:** the Phase 17 K closeout (2026-10-12). KGM row and K-thread closeout / model-mismatch pivot (§3). No §1
  number, no estimate and no KP2 classification changed.
- **Truth level:** every number is `SYNTHETIC_CAMPAIGN_TRUTH`. K_truth = 0.01 m²/kg is a campaign value, not a spacecraft property.
- **Dynamics:** the inherited simplified model (Moon point mass, lunar J2 and cannonball SRP) unless a row says otherwise.
  Since F1 the STM and dx/dK linearise the SRP force exactly (distance, direction and penumbra terms).

## 1. Canonical K_SRP baseline (current)

Canonical arc: 15 orbits, DSN-3, 90 s cadence, 960 range observations. Earth at the measurement epoch
(FXR canonical fixture); production visibility (elevation + lunar occultation).

| quantity | authoritative value | source |
|---|---:|---|
| range-only `f_perp` / θ_K / σ_K/K | **0.091502** / 5.25° / **25.40%** | F1 re-qualification of FXR/GEO `G0_consistent` |
| range + production ΔDOR, Goldstone–Canberra, 5 nrad: `f_perp` | **0.172343** | F1 re-qualification of R1O-D2 CANON/GC |
| gain Δ`f_perp` | **+0.0808** (+0.080841; STRONG_DIRECTION_GAIN under the R1O-D threshold) | F1 / R1O-D2 |
| σ_K/K (range + ΔDOR) | **7.46%** (7.4565%) | F1 / R1O-D2 |
| ΔDOR rows (both stations visible) | 170 (of 263 elevation-only rows) | R1O-D2; HARDEN |

Source of every F1 number: `studies/phase17/results/phase17_f1_science_materiality.json` (`comparison`).

**Superseded digits (pre-F1, omitted SRP gradient):** range-only 0.091508, combined 0.172351, gain +0.080843.
They appear in the FXR, R1O-D2, HARDEN and GEO documents and in the erratum blocks of six reports; those
blocks point here, and the values in this table replace them. The change is in the 5th–6th significant digit.
It moves no class in the D2 primary population (18 cells), the 5 frozen GEO spot cases or the D2 high-fidelity
cases (`studies/phase17/reports/phase17_f1_srp_variational_gradient_qualification.md`).

**Historical intermediate (do not cite as current):** `f_perp` **0.3204**, σ_K/K 3.97%, labelled
`EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`. Its Earth epoch is correct (FXR), but 93 of its 263
ΔDOR rows had the spacecraft behind the Moon (defect HARDEN-D04).

## 2. Scope of the ΔDOR result (R1O-D2)

The gain is **strongly geometry-dependent**. Predeclared outcome:
`DDOR_K_INFORMATION_GAIN_IS_STRONGLY_GEOMETRY_DEPENDENT`.

- **Where it is STRONG:** only on Goldstone–Canberra at the canonical epoch.
- **Where it is absent:** at the other three lunar-phase epochs, and on Goldstone–Madrid.
- **Calibration:** unconstrained per-session calibration biases remove it. A DSN-level bias prior keeps about half.
- **Identifiability:** K remains the weakest design direction everywhere.

See `studies/phase17/reports/phase17_r1od2_ddor_robustness_qualification.md`.

## 3. Phase-by-phase status

| phase | current status | what is superseded | where |
|---|---|---|---|
| R1 / R1G | historical context | geometry numbers on the published fixture | their reports; FXR erratum |
| R1M | qualitative conclusion PARTIAL: range-only multi-arc does not separate K | absolute numbers (published Earth fixture) | FXR erratum |
| R1COV | QR square-root covariance algorithm VALID | canonical numbers | FXR erratum |
| R1O | surrogate conclusions superseded twice (FXR epoch, then occultation) | ΔDOR-like 0.3203 → **0.1722**; Earth LOS 0.0920 → **0.0919** (230/590 rows were occulted) | `studies/phase17/results/phase17_harden_visibility_requalification.json` |
| R1O-R | column-major STM repair VALID | baseline numbers | FXR erratum; `tests/test_r1o_stm_layout.py` |
| R1O-D | production ΔDOR model + software qualification VALID | all campaign numbers; current = §1 | R1O-D report erratum blocks |
| R1O-OPT | MARGINAL_DIRECTION_GAIN confirmed (optical rows are Earth-independent) | numbers combined with the old range block | FXR erratum |
| GEO | `GEOMETRY_CHANGES_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY` (range-only; unaffected by D04) | — | GEO report |
| FXR | Earth-epoch repair VALID. Its R1O-D ΔDOR number is the historical intermediate | 0.3204 | FXR report + erratum blocks |
| R1O-D2 | **current ΔDOR science** | — | D2 report |
| HARDEN | infrastructure and contracts; no new science | — | `docs/phase17_harden_*` |
| F1 | SRP position gradient in the variational equations QUALIFIED; `NUMERICAL_BASELINES_UPDATED_CLASSES_PRESERVED` | canonical 6-digit values (§1) | `studies/phase17/reports/phase17_f1_srp_variational_gradient_qualification.md` |
| R1O-D3 | **current ΔDOR tracking-design science**: `DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS`. Under one-hour operational sessions the canonical GC gain is MODEST (Δ`f_perp` +0.0328), not the dense §1 value. Nonlinear confirmation NOT justified | — (§1 stays the dense/ideal D2 configuration) | `studies/phase17/reports/phase17_r1od3_tracking_design_characterization.md` |
| R1O-D4 | D3's exploratory GM FIRST_SESSION window **not confirmed**: `GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE`. It exists in simplified dynamics (85°–115°) but no high-fidelity orbit has the required 30° window (0/5); the shift follows the range-only baseline (r = 0.976). Nonlinear confirmation NOT justified; the ΔDOR scheduling branch has no remaining hypothesis | D3 secondary hypothesis | `studies/phase17/reports/phase17_r1od4_gm_window_confirmatory_qualification.md` |
| R1O-CD1 | **current counted-Doppler science**: `COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE` (Case D). Over 120 high-fidelity cells, ideal Doppler passes the population rule (76.7 % positive); one free bias per pass fails it (median Δ`f_perp` −0.006, retention LOST); every cell is WEAK in absolute separation; σ_K/K ≈ 0.5 % is magnitude, not direction. `ALTERNATIVE_K_PARAMETERIZATION_JUSTIFIED = YES`, nonlinear confirmation NOT justified. Gate PASS after closure (D23 immaterial; one bounded rebuild numerical-floor exception) | — | `studies/phase17/reports/phase17_r1o_cd1_counted_doppler_characterization.md` §1, §49–§50 |
| KP1 | **current K-parameterization science**: `ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS`, for the *variation* of K only. Under range + counted Doppler with a free bias per pass (high fidelity, 120 cells), the three-segment model's variation directions separate from state and constant K at MODERATE+ in 57.5 % of cells, with internal correlations ≤ 0.90 everywhere. Two halves give PARTIAL (25.8 %), linear drift NO_NEW_INFORMATION. The **K level stays WEAK**: every block's weakest, common-mode-like combination (≈ 0.085) is below the scalar f_perp (0.133). Orbit-dependent (the 500 km orbit: 0/24); fidelity-robust. Linear and local at a constant truth; the next K-fitting phase needs a HARDEN-D23 repair first | — | `studies/phase17/reports/phase17_kp1_alternative_k_parameterization_characterization.md` |
| HARDEN-D23 | counted-Doppler Earth **repaired**: the production station provider uses the same cubic-Hermite Earth as range and ΔDOR (R3-P22's linear policy superseded). Against a direct-SPICE oracle (frozen model S solver, 239,265 CD1/KP1 rows) production values agree to max 7.6e-8 m/s, rms 1.6e-8 (ET-exact oracle: max 5.4e-9); the historical offset (1.83e-3 m/s) is reproduced and removed 2.4e4×. CD1 and KP1 re-qualified **unchanged** (f_perp ≤ 4e-10, principal sines ≤ 1e-9, 0 class flips). Counted-Doppler **values are qualified for estimation at the 1e-7 m/s level**. Gate **PASS** (owner accepted 2026-10-07 the adjudicated exception: the frozen epoch-contract diagnostic measured the oracle's ET rounding, 6.4e-5 m; exact-epoch scan 2.4e-7 m ≤ 1e-6; 8 of 9 frozen criteria passed as computed) | — | `studies/phase17/reports/phase17_harden_d23_repair_qualification.md` |
| KP2 | **current nonlinear K-recovery science**: `M3_VARIATION_PARTIALLY_RECOVERABLE`. A nonlinear batch solve on production range + counted-Doppler **values** (free bias per pass, high fidelity, 30 trials) recovers the M3 **time variation** in 15/20 variation trials (8/12 in KP1-separable cells, 7/8 in KP1-weak cells), consistently wherever it converges (χ² and NEES 28/28), with 0/10 false detections; the Doppler bias impact is MINOR (σ_v ×1.24). Misses: O2 full Sun (information-limited, σ_v ≈ 8 %) and two O2 k12 estimator-convergence cases. **K level NOT claimed.** Variation magnitude σ_v, not KP1's direction metric, predicts recovery. Gate: 2 frozen verification criteria failed as computed and were adjudicated (FD oracle floor; production BLS non-convergence, agreement when warm-started); **owner accepted 2026-10-15, gate PASS** (after BLS1 + D25 the frozen PRODUCTION_BLS job passes both criteria as computed) | — | `studies/phase17/reports/phase17_kp2_nonlinear_m3_recovery_confirmation.md` |
| HARDEN-BLS1 | production BLS K_SRP solve-for **repaired and QUALIFIED** (scalar K, range only, the problem production solves; it has no M3 / counted-Doppler / per-pass-bias capability). Root cause: three defects in the K path, each necessary by ablation (normal matrix formed in bookkeeping units → false singularity; negative-K trials discarded with λ raised; early loose-tolerance Jacobian) — HARDEN-D26. From the declared a priori, without seeding, the frozen KP2 job now reaches the KP2 solution to ≤ 6.2e-5 σ (was 347 σ); independent stationarity ≤ 4.1e-3 σ. KP2 result **unchanged** (`M3_VARIATION_PARTIALLY_RECOVERABLE`). KP2 PRODUCTION_BLS gate: estimate criterion now PASS, σ criterion still FAIL, wholly HARDEN-D25 (σ ratio 0.917 → 1.0000 with the harmonic field); D25 immaterial to every KP2 classification. Posterior σ not qualified until D25; status label not a convergence indicator (D27). K level NOT claimed | — | `studies/phase17/reports/phase17_harden_bls1_production_bls_recovery.md` |
| HARDEN-D25 | two-way-range posterior covariance **repaired and QUALIFIED**: the posterior pass now gets the solve's own dynamics (harmonic field with matched STM; fixed SRP). Blast radius wider than recorded in KP2: BLS K path (field), BLS six-state (field + fixed SRP), SRIF six-state (fixed SRP); defaults never affected. Estimates bitwise identical in all 8 tested configurations (covariance-only); independent assembly 2.5e-13; frozen KP2 σ criterion now 1.000000–1.000001 (was 0.917 K, 0.972 z), end to end as well. **Superseded uncertainty values** (estimates unchanged): gravity campaign stage G G100 overconfidence **1.93× → 1.50×** (ANEES 11.58 → 9.01, still outside the band; conclusion stands), formal 3D σ 64.41 → 63.84 m; stage I trace(P⁻¹C) 5.96 → 4.06, largest generalized eigenvalue 4.28 → 1.74 (still directional). Predeclared FD oracle G4 FAILED as computed (K step in the integrator-noise floor), adjudicated post hoc; **owner accepted 2026-10-12, gate PASS** | stage G/I uncertainty numbers in `od_covariance_campaign` reports (files not modified) | `studies/phase17/reports/phase17_harden_d25_posterior_covariance_consistency.md` |
| HARDEN-D28 | no-explicit-prior covariance contract **repaired and QUALIFIED**: with no prior the objective is data-only (MATLAB-heritage 15 km / 5 m/s diagonal = step regularizer, never a prior) and the posterior is now data-only too; explicit priors unchanged (bitwise); SRIF K path unchanged by design (its objective carries the full prior, D29). Estimates bitwise unchanged (17 cases). Effect: ≤ 3.4e-4 in σ on a 3.3 h single-station arc, ≤ 1.5e-6 on KP2 (production now agrees with the KP2 solver to 5e-8), ≤ 3.7e-8 on scenario hand-offs; **no authoritative number superseded** | — | `studies/phase17/reports/phase17_harden_d28_prior_covariance_contract.md` |
| HARDEN-D29 | SRIF K_SRP prior contract **repaired**: without an explicit state prior the SRIF K objective is now data-only state + the required K prior (the BLS contract its docstring declares); the 15 km / 5 m/s diagonal is a step regularizer in every path. Explicit-prior and six-state paths bitwise unchanged; estimate change ≤ 2.1e-4 σ (test arc); BLS and SRIF now solve the same problem (estimates 4.3e-4 σ). Gate FAIL as computed on one posterior sub-criterion (1.44e-6 vs 1e-6 = linearisation-point dependence), adjudicated; **owner accepted 2026-10-15, gate PASS**. No authoritative consumer | — | `studies/phase17/reports/phase17_harden_d29_srif_k_prior_contract.md` |
| HARDEN-D27 | BLS-LM convergence/status contract **repaired and QUALIFIED** (gate PASS): at damping exhaustion the undamped Gauss-Newton decrement of the scored objective is evaluated at the returned point (caller's tolerance); ≤ 0.1 posterior σ → `Converged` / `STATIONARY_AT_DAMPING_LIMIT`, else `DampingLimit` (now: not certified stationary). General (position, range-rate, two-way, K path). Estimates bitwise unchanged (BLS1 a-priori K bitwise). The frozen high-fidelity KP2 production solution (was `DampingLimit`) has decrement **1.9e-3** → certified; genuine failures kept (corrupted Jacobian 2511; truth-start stall 3.68). Conservative where the cost noise floor exceeds 0.1 σ (simplified rtol 1e-11: 0.159, uncertified). Opens HARDEN-D30 | BLS1's "status label not a convergence indicator" caveat | `studies/phase17/reports/phase17_harden_d27_convergence_status.md` |
| KGM | **final stochastic-K experiment**: `SUPPORTS_M3_VARIATION_RESULT` (predeclared, sha `6fd5419f…`). A 15-segment first-order Gauss-Markov K with a flat common level (prior on the variation only) recovers the KP2 step variation where M3 did (CANON k0, O1 k12) and not where M3 did not (O2 k0): agreement 3/3, 0/3 false variations, data-driven (DFS 0.60–0.86 where it recovers; σ_v 0.88–1.47× M3). τ = 1/5/15 orbits: same outcome. On a GM-shaped truth M3 is inconsistent on CANON (χ²_err 39.8), GM consistent. Common level NOT claimed | — | `studies/phase17/reports/phase17_k_closeout_and_model_mismatch_pivot.md` §2 |
| K closeout / pivot | **K thread `CLOSED_WITH_CAVEAT`** (truth = model dynamics). Linear probes: lunar albedo/IR pressure is ≥ 99.9 % absorbed, landing on the common K level (the physical reason it is not claimable); Moon-fixed gravity-field error is 1–800× SRP at 100 km and has an amplitude-independent false-variation / residual-χ² ratio of 0.02–0.81, i.e. a linear **masquerade window**. Next question and minimum experiment (8 nonlinear T0 solves, truth field GL1800F + α(GRGM1200L − GL1800F)) in §3.4 of the report | — | `studies/phase17/reports/phase17_k_closeout_and_model_mismatch_pivot.md` §3 |
| MM (gravity masquerade) | **`H_MM_SURVIVES_MASQUERADE`** (predeclared, as computed): with K constant and a Moon-fixed field error of 0.3 × (GRGM1200L − GL1800F) at degree 50, the unchanged KP2 M3 solver reports a false K variation (CANON k0: χ²_detect 28.8, f̂ ≈ −9 / −3 / −5 %) while the residual excess stays inside the χ² scatter (Δχ² 75 ≤ 122) and the per-pass runs test is blind. **A K time variation from tracking is therefore not claimable without independent gravity-field knowledge.** KP2's classification stands only under truth = model. The field difference is convention-dominated, not a GRAIL error estimate; one draw per cell; O2 α = 1 solver failure recorded | KP2's external validity (not its classification) | `studies/phase17/reports/phase17_mm_gravity_masquerade_falsification.md` |
| FE-L (field-error parameterisation, linear) | **`FE_SUPPRESSES_BUT_COSTS_VARIATION`** (predeclared): estimating degree 2–6 field corrections (45 coefficients, data-only) alongside M3 removes the MM masquerade in all 10 cells (false-variation χ² < 13.82 at α = 0.3 and 1), but no degree N keeps the true-variation noncentrality ≥ 13.82 everywhere (N = 4 already fails in 4 of 8 eligible cells). Low-degree field error and K variation are nearly degenerate in range + counted Doppler; data-only field estimation does not make a K variation claimable. Next lever: external field prior (not started) | — | `studies/phase17/reports/phase17_fe_field_error_parameterisation_linear.md` |
| FE-P (external gravity prior, linear) | **`FE_PRIOR_TOO_WEAK_TO_SUPPRESS`** (predeclared science rule on the primary ×1 GL1800F published-sigma diagonal prior, N = 4 and 6): the prior re-fixes the field at GL1800F (posterior/prior σ = 1.000), so the false variation of the MM mismatch (median 5700 σ_GL1800F) is unchanged from the fixed-field design (8 of 10 cells ≥ 13.82 at α = 1); ×10 and ×100 classify the same. The as-computed rule gives `FE_PRIOR_INCONCLUSIVE` because gate P2 failed (weak-limit multiplier ×1e8 not asymptotic; scale⁻² convergence to 2.4e-8 at ×1e12): **owner accepted 2026-10-21, gate PASS**. Secondary: errors drawn at ×1 / ×10 the published σ give no false variation (0 / 200 draws ≥ 13.82); the stress width |ΔC| partly resolves (7 of 10 cells). Two more linear residual-invisible masquerade cells found (O2 k12 α = 0.3, O3 k0 α = 1; not nonlinearly confirmed). FE route closed; next = model-error attribution / integrity | — | `studies/phase17/reports/phase17_fep_external_gravity_prior.md` |
| MM-G (GL1800F-internal omission truth) | **`MMG_INCONCLUSIVE`** (predeclared rule, as computed): truth GL1800F 100×100 vs estimator 50×50; all 4 nonlinear KP2 solves stop with "non-positive K" (omission ≈ 2.5e-5 m/s², ~500× SRP), and the predeclared L2 linearity gate failed (90 s grid under-samples degree 51–100; repaired with a 6× sub-grid, 5.9e-3; the MM/FE-L/FE-P GR−GL column passes on the coarse grid, 3.1e-3, so no upstream result changes). Linear: the omission is grossly visible (residual 3e6–6e7 vs window ~120) or negligible (O3); no masquerade; FE N = 4/6 and the ×1 prior do not absorb it. Post-hoc: masquerade is an amplitude window (fv/residual ≥ ~0.11); GL1800F-internal shapes have windows far below their real amplitude (51–100: ~150–700×; a 100×100 estimator's 101–200 remainder: ~7–11×) | — | `studies/phase17/reports/phase17_mmg_gl1800f_internal_omission.md` |

## 4. Known approximations that remain (documented, quantified)

The STM's former omission of the SRP position gradient (HARDEN-D05) is **fixed** by Phase 17-F1.

| item | size | ledger |
|---|---|---|
| ~~counted Doppler interpolates the Earth linearly~~ **FIXED (HARDEN-D23, 2026-10-07)** | historical: 1.67–1.83e-3 m/s value offset (superseded quantification ~1.5e-6 m/s). Now cubic Hermite: Earth ≤ 2.4e-7 m, values vs direct SPICE ≤ 7.6e-8 m/s | HARDEN-D23 |
| exact station transform at rounded absolute ET (counted Doppler, range, ΔDOR) | counted-Doppler value rms ≤ 1.0e-7 m/s (max 3.6e-7); ~20× below the optimistic operational noise floor | HARDEN-D24 |
| ~~BLS-LM K_SRP posterior design omits the lunar harmonic field~~ **FIXED (HARDEN-D25, 2026-10-11)**, together with the six-state BLS (field + fixed SRP) and SRIF (fixed SRP) posterior passes | historical: σ_K +9 %, σ_z +3 % (KP2); stage G G100 ANEES 11.58 vs 9.01 repaired. Now: posterior = independent rebuild to 2.5e-13 | HARDEN-D25 |
| ~~default weak state prior (15 km, 5 m/s) in the posterior but not in the fitted objective~~ **FIXED (HARDEN-D28, 2026-10-12)**: no-prior posteriors are data-only | historical σ ≤ 3.4e-4 (3.3 h fixture), ≤ 1.5e-6 (KP2) | HARDEN-D28 |
| ~~SRIF K_SRP objective pulls toward the default state regularizer~~ **FIXED (HARDEN-D29, 2026-10-12)** | historical: ≤ 2.1e-4 σ on a 3.3 h arc; no authoritative consumer | HARDEN-D29 |
| ~~BLS-LM K_SRP may end `DampingLimit` at a verified stationary point~~ **FIXED (HARDEN-D27, 2026-10-12)** | historical: high-fidelity KP2 job labelled `DampingLimit` at decrement 1.9e-3 σ; now certified `STATIONARY_AT_DAMPING_LIMIT` | HARDEN-D27 |
| BLS-LM acceptance cost noise (6-state propagation at the caller's rtol) rejects damped steps below it | simplified rtol 1e-11: ~5 χ² cost noise; truth-start K solve stalls 1.95 σ off (labelled `DampingLimit`, correctly uncertified); no authoritative result affected | HARDEN-D30 |
| station B receive up to ~0.03 s past the last grid node (ΔDOR) | ~1e-5 m | HARDEN-D16 |

## 5. How to reproduce a historical number

Historical scripts refuse to run unless `PHASE17_LEGACY_REPRODUCTION=1` is set. The elevation-only row rule
must be named explicitly (`DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY`) and warns when used. The published FXR
artifacts reproduce bitwise through `studies/phase17/scripts/phase17_fxr_requalification.py`, which names that rule.

For the full defect list, see `studies/phase17/reports/phase17_harden_defect_ledger.md`.
