# Phase 17-FE-L — Low-degree gravity-field-error parameterisation (linear study)

- **Date:** 2026-10-15
- **Predeclaration:** `artifacts/phase17_fe_predeclaration.json`, committed in `e2e2187` before any FE number
- **Evidence:** `artifacts/phase17_fe_summary.json`, `artifacts/phase17_fe_cells/*.json`,
  `artifacts/phase17_fe_gate_v{1,3}.json`, `examples/phase17_fe_linear.py`
- **Follows:** Phase 17-MM (`H_MM_SURVIVES_MASQUERADE`)

## 0. Final status

```
PHASE17_FE_L_OUTCOME   = FE_SUPPRESSES_BUT_COSTS_VARIATION   (predeclared rule, as computed)
SUPPRESSES             = N = 6 only (45 coefficients): false-variation chi2 < 13.82 in all 10 cells for
                         alpha = 0.3 and alpha = 1 (worst O1 k0 alpha = 1: 11.9). N = 4 fails in one cell
                         (O1 k0 alpha = 1: 85.4)
PRESERVES              = no N: estimating the field corrections removes the K-variation information in most
                         cells (N = 4: CANON k12 17.8 -> 7.0, O1 k12 16.9 -> 3.3, O5 k0 40.0 -> 7.9;
                         N = 6: 15 of the 18 preserve-eligible (cell, truth) pairs fall below 13.82)
MEANING                = with range + counted Doppler on these arcs, a K time variation and a low-degree (2-6)
                         Moon-fixed field error are nearly degenerate: removing the masquerade by estimating the
                         field costs the K-variation signal as well. Data-only field estimation does not rescue it
NONLINEAR CONFIRMATION = not justified (the rule requires FE_RESOLVES_MASQUERADE)
GATES                  = V1 PASS (convolution vs propagated FD: C22 1.9e-3, C31 4.5e-4 <= 2e-2);
                         V3 PASS (partial linearity 3.6e-12); V2 linear N = 0 CANON k0 alpha = 0.3: 22.5 vs
                         nonlinear MM 28.8 (one noise draw), consistent
NOT CLAIMED            = the common K level; a GRAIL error estimate; any nonlinear result
```

## 1. Design (as predeclared)

- **Base design.** For each of the 10 KP2 cells the high-fidelity model trajectory (GL1800F 50×50, constant K0)
  carries the KP2 estimator design:
  - x₀ with its a priori;
  - M3 K (3 segments);
  - one free bias per Doppler pass;
  - range at 5 m and counted Doppler at 1e-4 m/s.
- **Field-error parameters.** Corrections ΔC_nm and ΔS_nm, fully normalised, for degrees 2..N with
  N ∈ {2, 4, 6} (5, 21 and 45 parameters), data-only (no prior).
  - Their acceleration partials are exact: the engine is linear in the coefficients, so ∂a/∂C_nm is the
    acceleration of a unit-coefficient model. V3 checks this.
  - Trajectory columns are STM convolutions, checked against propagated finite differences (V1).
- **Mismatch.** α·(GRGM1200L − GL1800F) at degree 50, α ∈ {0.3, 1}, as in Phase 17-MM.
- **Metrics.**
  - Linear false-variation χ² (2 dof) of the mismatch.
  - Its unabsorbed residual χ².
  - The noncentrality of the KP2 truths T1_STEP and T2_MIDDLE: the expected χ²_detect of a real ±10 % variation.

## 2. Results

False-variation χ² at α = 0.3 / α = 1, then T1 / T2 noncentrality (the eligible-to-preserve threshold is 13.82
at N = 0):

| cell | N = 0 | N = 2 | N = 4 | N = 6 |
|---|---|---|---|---|
| CANON k0 | 22.5 / 250 · 251 / 190 | 8.3 / 92 · 117 / 119 | 0.03 / 0.36 · **33 / 33** | 0.01 / 0.06 · 15 / 15 |
| CANON k12 | 0.45 / 5.0 · 17.8 / 19.7 | 28.0 / 311 · 16.3 / 18.6 | 0.00 / 0.01 · 7.0 / 7.4 | 0.00 / 0.01 · 2.3 / 2.5 |
| O1 k0 | 1324 / 14713 · 327 / 213 | 55 / 612 · 72 / 69 | 7.7 / **85** · 44 / 42 | 1.1 / 11.9 · 15.0 / 12.8 |
| O1 k12 | 134 / 1491 · 16.9 / 12.3 | 2.8 / 31 · 10.1 / 7.6 | 0.08 / 0.87 · 3.3 / 3.2 | 0.00 / 0.02 · 1.4 / 1.4 |
| O2 k0 | 106 / 1179 · 3.3 / 1.9 (not eligible) | 0.64 / 7.1 · 1.4 / 1.1 | 0.02 / 0.18 · 0.8 / 0.8 | 0.03 / 0.30 · 0.3 / 0.2 |
| O2 k12 | 141 / 1572 · 37.4 / 31.6 | 2.2 / 24 · 8.2 / 8.5 | 0.11 / 1.2 · 3.9 / 4.3 | 0.02 / 0.25 · 1.2 / 1.2 |
| O3 k0 (500 km) | 12.7 / 141 · 280 / 467 | 0.22 / 2.4 · 189 / 196 | 0.00 / 0.01 · **52 / 59** | 0.00 / 0.00 · 14.8 / 19.0 |
| O3 k12 (500 km) | 0.86 / 9.6 · 98 / 126 | 0.61 / 6.8 · 51 / 42 | 0.01 / 0.06 · **14.9 / 16.1** | 0.00 / 0.00 · 7.3 / 8.1 |
| O5 k0 | 5.3 / 58 · 40 / 25 | 2.5 / 28 · 21 / 19 | 0.05 / 0.54 · 7.9 / 7.4 | 0.00 / 0.03 · 2.6 / 2.7 |
| O5 k12 | 100 / 1112 · 137 / 106 | 5.3 / 59 · 85 / 67 | 0.01 / 0.08 · **22.5 / 21.7** | 0.00 / 0.00 · 5.8 / 5.8 |

σ_v inflates with N, by about ×4–8 at N = 4 and ×6–14 at N = 6 (`phase17_fe_summary.json`).

## 3. Reading

1. **The linear model reproduces Phase 17-MM.** At N = 0, CANON k0 α = 0.3 is the only cell with a significant
   false variation (22.5) while its residual stays near the window (χ² 60 vs ~122). Every other large false
   variation comes with a large, detectable residual. This matches the nonlinear MM outcome, and V2 is
   consistent (22.5 vs 28.8 on one draw).
2. **Degrees 2–4 carry most of the masquerade, but not all.** N = 4 removes it everywhere except O1 k0 at full
   amplitude (85). N = 6 removes it everywhere.
3. **The same coefficients carry the K variation.** At N = 4 the true-variation noncentrality falls below
   detection in 4 of the 8 eligible cells; at N = 6 almost everywhere. Low-degree field error and segment-wise
   SRP K variation are nearly degenerate in range + counted Doppler: both are slowly varying, Moon-related
   along-track and radial forcing over a 15-orbit arc.
4. **A partial parameterisation can make things worse.** In CANON k12 the false variation rises from 0.45 (N = 0)
   to 28.0 (N = 2). Adding a subset of field parameters re-routes the mismatch into K.
5. **Where it does work (post-hoc, labelled, not the predeclared decision).** At N = 4, five cells are both
   suppressed and preserved: CANON k0, O3 k0, O3 k12, O5 k12 (and O1 k0 at α = 0.3 only). The 500 km orbit is
   the most robust, as expected from its weaker field sensitivity. The resolving power is geometry-dependent, not
   general.

## 4. Consequence and next question

Data-only estimation of the field does not make a tracking-derived K variation claimable: it trades the
masquerade for the signal. The remaining lever is **external gravity information**. With a prior on ΔC/ΔS at a
realistic level, the question becomes whether there is a prior width that suppresses the masquerade and keeps the
variation. Candidate widths are a GRAIL formal-error scale, or the GL1800F − GRGM1200L convention difference. This
is a cheap extension of the same linear machinery (one prior row per coefficient). It is **not started**: it
needs a predeclared prior width and rule.

## 5. Verification

- Gates V1 and V3 pass (§0). Production code is unchanged in this phase.
- `tests/test_mm_truth_field.py` and the full test suite are unaffected (analysis-layer script only).
