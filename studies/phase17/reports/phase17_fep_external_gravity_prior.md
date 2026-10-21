# Phase 17-FE-P — External gravity prior and the gravity / K-variation degeneracy

- **Date:** 2026-10-15
- **Predeclaration:** `studies/phase17/configs/phase17_fep_predeclaration.json`, committed in `3f46c6c` before any FE-P number
- **Evidence:**
  - `studies/phase17/results/phase17_fep_summary.json`
  - `studies/phase17/results/phase17_fep_p2_supplementary_1e12_1e13.json`
  - `studies/phase17/scripts/phase17_fe_prior.py`
  - `tests/test_fep_prior.py`
- **Follows:** Phase 17-FE-L (`FE_SUPPRESSES_BUT_COSTS_VARIATION`)

## 0. Final status

```
PHASE17_FE_P_GATE              = P1 P3 P4 P5 PASS; P2 FAIL-as-computed (oracle: the predeclared x1e8 weak-limit multiplier is
                                 not yet asymptotic; deviation follows scale^-2 exactly and reaches 2.4e-8 at x1e12, §4).
                                 Adjudicated QUALIFIED - owner accepted 2026-10-21, gate PASS
GRAVITY_PRIOR_SOURCE           = GL1800F per-coefficient uncertainty columns (jggrx_1800f_sha.tab, the estimator's own file)
GRAVITY_PRIOR_SEMANTICS        = zero-mean Gaussian on dC_nm, dS_nm about the GL1800F values the estimator uses (published
                                 sigma; the file does not state formal vs calibrated)
PRIOR_COVARIANCE_FORM          = diagonal (no coefficient covariance in the repository) - APPROXIMATION
PRIMARY_PRIOR_SCALE            = x1
PRIOR_SENSITIVITY_SET          = x10, x100 (+ secondary STRESS width |dC(GRGM1200L - GL1800F)|, not an uncertainty)
PRIOR_IMPLEMENTATION_STATUS    = QUALIFIED (normal equations 3e-16 / unit test 1e-13; strong limit 6e-12; sigma map exact;
                                 FE-L reproduced bit-for-bit; weak limit by scale^-2 convergence, see gate line)
PHASE17_FE_P_OUTCOME           = FE_PRIOR_TOO_WEAK_TO_SUPPRESS (predeclared science rule on x1).
                                 As-computed predeclared rule with the gate veto: FE_PRIOR_INCONCLUSIVE (P2); the owner accepted
                                 the P2 adjudication on 2026-10-21, so the outcome stands as TOO_WEAK (x1, x10 and x100 all give
                                 TOO_WEAK, so the P2 issue could not change the decision)
FALSE_VARIATION_SUPPRESSION    = NOT SUPPRESSED at x1: identical to the fixed-field design (N = 0) to <= 3e-4 relative;
                                 8 of 10 cells keep fv_chi2 >= 13.82 at alpha = 1
REAL_VARIATION_PRESERVATION    = PRESERVED (trivially: the prior keeps the field fixed, noncentrality = N = 0 value)
RESIDUAL_BEHAVIOR_STATUS       = unchanged from N = 0 (prior penalty <= 0.1): the prior neither absorbs the mismatch nor pushes
                                 it into residuals. Linear residuals show the masquerade is wider than MM's single case (§3.3)
GEOMETRY_DEPENDENCE_STATUS     = secondary / geometry-dependent: only the STRESS width partly resolves (N = 6: 7 of 10 cells;
                                 O1 k0 and O5 k12 stay masqueraded, O1 k12 loses detection)
COMMON_K_LEVEL_STATUS          = NOT_CLAIMED
NONLINEAR_CONFIRMATION_STATUS  = NOT TRIGGERED (outcome is not RESOLVES)
K_RESEARCH_IMPLICATION         = the parameterisation route is closed. A realistic prior does not break the degeneracy; it
                                 holds the field at GL1800F. Whether a tracking-derived K variation is credible then depends
                                 on whether the true field error is GL1800F-sigma-sized (masquerade negligible, §3.4) or
                                 convention-sized like GRGM1200L - GL1800F (masquerade present, no tracking-only remedy)
NEXT_PHASE_RECOMMENDATION      = close FE; move to model-error attribution / integrity: (1) characterise which real model
                                 errors (frame/convention, tide, GM/R, unmodelled forces) are larger than the published field
                                 sigma; (2) then ask whether measurement-geometry / observable diversity separates them
```

## 1. Prior construction (recorded before scoring)

- **Source.** Columns 5–6 of `python_port/data/gravity/gl1800f/jggrx_1800f_sha.tab` give the uncertainty of C_nm
  and S_nm. This is the same file and normalisation the estimator field is loaded from:
  - fully normalised;
  - R = 1738 km;
  - GM = 4902.8003063302 km³/s²;
  - frame MOON_PA_DE440;
  - degree 2–3 without the permanent tide.

  The mapping to the FE corrections is therefore unambiguous.
- **Prior rows.** One whitened prior row per FE coefficient: 1/(scale·σ_nm) on that column, right-hand side 0. The
  state prior, M3, the free per-pass Doppler bias, measurement weights and the cells are unchanged from FE-L.
- **Scale of the problem, recorded before scoring.** At degrees 2–6, the MM mismatch GRGM1200L − GL1800F
  (rescaled to GL μ/R) is |ΔC|/σ = median 5696, min 82.9 (S22), max 65362. The controlled mismatch lies far
  outside GL1800F's own uncertainty.

## 2. Validation

| gate | criterion | value | status |
|---|---|---|---|
| P1 | AᵀA = data information + diag(1/σ²) on the FE block (one cell, N = 6) | 3.0e-16 relative | PASS |
| P2 | σ×1e8 reproduces FE-L data-only (same N) within 1e-6 | worst 0.39 | **FAIL as computed** |
| P3 | σ×1e-8 reproduces N = 0 within 1e-6 | 6.3e-12 | PASS |
| P4 | σ vector = independent re-parse of the `.tab` lines; spot values C20, C21, S21 | exact | PASS |
| P5 | no-prior designs reproduce the committed FE-L cell metrics (N = 0, 4, 6) within 1e-8 | 0.0 | PASS |

`tests/test_fep_prior.py` checks two things independently of the campaign:
- the σ spot values;
- the prior-row normal equations, to 1e-13 on a synthetic design.

## 3. Results

### 3.1 Primary prior (×1) and sensitivity (×10, ×100)

The table gives the false-variation χ² at α = 0.3 / α = 1, then the T1 / T2 noncentrality. Each N cell lists the
×1 value; where ×100 differs it follows in brackets. N = 0 is the fixed-field design.

| cell | N = 0 | N = 4, ×1 [×100] | N = 6, ×1 [×100] |
|---|---|---|---|
| CANON k0 | 22.5 / 250 · 251 / 190 | 22.5 / 250 [19.2 / 214] · 251 / 190 | 22.5 / 250 [18.6 / 206] · 251 / 190 |
| CANON k12 | 0.45 / 4.9 · 17.8 / 19.7 | identical | identical |
| O1 k0 | 1324 / 14713 · 327 / 213 | 1324 / 14709 [465 / 5168] | 1324 / 14708 [387 / 4304] |
| O1 k12 | 134 / 1492 · 16.9 / 12.3 | 134 / 1491 [25.4 / 282] | 134 / 1491 [23.0 / 256] |
| O2 k0 | 106 / 1179 · 3.3 / 1.9 | 106 / 1178 [9.8 / 109] | 106 / 1178 [8.1 / 90] |
| O2 k12 | 141 / 1572 · 37.4 / 31.6 | 141 / 1572 [70.5 / 784] | 141 / 1571 [64.4 / 715] |
| O3 k0 | 12.7 / 141 · 280 / 467 | identical [10.7 / 119] | identical [10.2 / 113] |
| O3 k12 | 0.86 / 9.6 · 98 / 126 | identical | identical |
| O5 k0 | 5.3 / 58 · 40 / 25 | identical [4.9 / 54] | identical [4.7 / 52] |
| O5 k12 | 100 / 1112 · 137 / 106 | identical [89 / 990] | identical [88 / 978] |

Field-parameter diagnostics:
- At ×1 the posterior/prior σ is 1.000 for every coefficient in every cell: the data add no field information
  beyond GL1800F's.
- At ×1 the max |corr(K, field)| is ≤ 0.009.
- Even at ×100 the median ratio is ≥ 0.998, the minimum ratio 0.66, and the max |corr| 0.40.

Classification (predeclared science rule, both N):

| config | SUPPRESSES | PRESERVES | cells resolved | outcome |
|---|---|---|---|---|
| ×1 | no | yes | 2 (CANON k12, O3 k12: no masquerade at N = 0 either) | TOO_WEAK |
| ×10 | no | yes | 2 | TOO_WEAK |
| ×100 | no | yes | 2 | TOO_WEAK |

### 3.2 Secondary stress width |ΔC_GR−GL| (not an uncertainty)

When the prior width is set to the mismatch's own coefficient magnitudes, the result is PARTIALLY_RESOLVES:
- 6 of 10 cells at N = 4 and 7 of 10 at N = 6 (CANON k0/k12, O2 k0, O2 k12, O3 k0/k12, O5 k0);
- O2 k0 counts only because it has no eligible truth;
- O1 k0 (fv 435 at α = 1, N = 6) and O5 k12 (20.5) stay above threshold;
- O1 k12 loses detection (T1 10.0).

This width presupposes knowing the error's magnitude coefficient by coefficient. It is a bracket, not a realistic
prior, and it does not change the primary outcome.

### 3.3 Residual behaviour

At ×1 the data-row residual of the mismatch is the N = 0 value (prior penalty ≤ 0.1). The prior does not move
the mismatch into the residuals; it only stops the field from absorbing it.

The linear residuals also agree with the nonlinear MM residual excess:

| cell | linear residual | MM nonlinear Δχ² |
|---|---|---|
| CANON k0 | 59.9 | 74.8 |
| O2 k0 | 148 | 153 |
| O1 k12 | 298 | 261 |
| O5 k0 | 55 | 36 |

All values are at α = 0.3.

**Observation (new, linear, not nonlinearly confirmed).** Two more cells pair a significant false variation with
a residual far below the MM windows (115–145):

| cell | α | fv_chi2 | residual |
|---|---|---|---|
| O2 k12 | 0.3 | 141 | 39.5 |
| O3 k0 | 1 | 141 | 17 |

Neither cell was in the MM run set. FE-L §3 reading 1 says CANON k0 is the only such case. That statement does not
hold for these two cells in FE-L's own data, which are unchanged and reproduced here (P5). FE-L's decision rule
does not use residuals, so the FE-L outcome stands. The MM conclusion (`H_MM_SURVIVES_MASQUERADE`) is strengthened,
not weakened. An addendum to the FE-L report records this.

### 3.4 Prior-consistent masquerade risk (secondary diagnostic)

Each case draws random field errors δC, δS ~ N(0, (scale·σ)²) over degrees 2–50: 20 draws per cell, 200 in total.
The false-variation χ² under the standard fixed-field design (N = 0):

| scale | median | max | fraction ≥ 13.82 |
|---|---|---|---|
| ×1 | 3.5e-5 | 0.0074 | 0 |
| ×10 | 3.4e-3 | 0.24 | 0 |
| ×100 | 0.43 | 54.7 | 0.04 |

Under the ×1 N = 4 / N = 6 configurations the values are identical to ~1e-4. A field error at the published
GL1800F uncertainty, or ten times it, cannot produce a significant false M3 variation in any cell. Risk appears
only at ~×100, where 4 % of draws cross the threshold.

## 4. The P2 gate

The predeclared weak-limit multiplier ×1e8 assumed the prior would then be weaker than every data-only field
direction. It is not. The N = 6 data-only solution has field directions whose posterior σ is of order 1e9 σ_GL, so at
×1e8 the prior still adds information.

The deviation from the FE-L data-only value shrinks as scale⁻², which is what a correctly implemented prior must do:

| multiplier | 1e8 | 1e9 | 1e10 | 1e11 | 1e12 | 1e13 |
|---|---|---|---|---|---|---|
| max relative deviation | 0.39 | 2.3e-2 | 2.4e-4 | 2.4e-6 | 2.4e-8 | 2.3e-10 |

This is an oracle-scale issue, not an implementation defect. The gate is kept FAIL as computed; **the owner accepted
the adjudication on 2026-10-21 (gate PASS)**. Its effect on the decision is nil:
- the primary ×1 prior is in the opposite regime (posterior/prior = 1.000);
- ×1, ×10 and ×100 classify identically.

## 5. Answer

**Does realistic independent gravity knowledge break the gravity-error / K-variation degeneracy for this tracking
design?**

**No.** The published GL1800F uncertainty is so tight that the prior simply re-fixes the field at GL1800F. It
leaves the false variation from a convention-sized field error (GRGM1200L − GL1800F, median 5700 σ) exactly where
the fixed-field design had it. It costs no real variation, but it suppresses nothing. Supported: **D** (too weak
to prevent false attribution of a mismatch of the MM kind).

E is not supported: the prior is defensible and maps unambiguously. A, B and C are not supported at the physical
width; B-like behaviour appears only under the labelled stress width.

The complement matters for interpretation. Coefficient error of the size GL1800F publishes is not a masquerade
risk at all (§3.4). The open problem is model error outside that uncertainty model, such as frame, tide or
convention differences between gravity solutions. Coefficient priors cannot address it; it needs attribution and
integrity work, not further K or prior tuning.

## 6. Not claimed

- The common K level.
- That GRGM1200L − GL1800F is a realistic gravity uncertainty.
- That formal-σ changes show separability.
- That any geometry result generalises.
- Any nonlinear result: this is a linear study.
