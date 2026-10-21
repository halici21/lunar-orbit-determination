# Phase 17-MM-G — GL1800F-internal omission error as the mismatch truth

- **Date:** 2026-10-16
- **Predeclaration:** `studies/phase17/configs/phase17_mmg_predeclaration.json`, committed in `0bae12c` before any MM-G number
- **Evidence:** `studies/phase17/scripts/phase17_mmg_omission.py`, plus these records in `studies/phase17/results/`:

  | artifact | content |
  |---|---|
  | `phase17_mmg_linear.json` | coarse linear stage, gate L2 FAIL |
  | `phase17_mmg_gate_L2_fine.json` | L2 on the fine grid |
  | `phase17_mmg_linear_fine.json` | fine linear stage |
  | `phase17_mmg_upstream_mm_convolution_check.json` | upstream convolution check |
  | `phase17_mmg_jobs/*.json` | nonlinear jobs |
  | `phase17_mmg_summary.json` | outcome |
  | `phase17_mmg_amplitude_window_posthoc.json` | post-hoc amplitude windows |

- **Owner request:** use GL1800F only, with no GRGM1200L difference.

## 0. Final status

```
PHASE17_MMG_OUTCOME      = MMG_INCONCLUSIVE (predeclared rule, as computed): all 4 nonlinear solves stopped with
                           "non-positive K candidate with no step to reduce", and the predeclared gate L2 failed
SCIENCE READING          = the GL1800F 51..100 omission is NOT a masquerade on these orbits: it is either negligible
                           (O3, 500 km) or grossly visible (linear residual chi2 3e6-6e7 vs window 113-145; the KP2
                           estimator cannot even fit it with a positive K). Not a nonlinear "detectable" result: the
                           rule needs converged solves, and none converged
GATES                    = L1 PASS (0.0); L2 FAIL with the predeclared FE-L convolution (3.41, independent of amplitude:
                           a sampling defect, the 90 s grid under-samples degree 51-100 along-track variation);
                           L2 PASS with a 6x sub-grid convolution (5.9e-3; 12x: 3.3e-3). Upstream: the MM / FE-L / FE-P
                           GRGM - GL column passes the same check on the coarse grid (3.1e-3) -> no upstream defect
FE-L / FE-P REDO         = for the omission, FE N = 4 / 6 data-only does not suppress (fv 1e4-2.5e6); the x1 GL1800F
                           prior equals N = 0. Degree 2-6 corrections cannot absorb degree 51-100 content
POST-HOC (labelled)      = the residual omission of a 100x100 estimator (GL1800F 101..200): fv 500-18000, residual
                           6e3-1.6e4, still visible; amplitude-window analysis (section 4): almost every error shape,
                           GL1800F-internal ones included, has an amplitude window where it masquerades
COMMON_K_LEVEL_STATUS    = NOT_CLAIMED
```

## 1. Design (as predeclared)

- **Truth.** GL1800F 100×100, the same file, GM, R and frame as the estimator. The omission error is GL1800F
  degrees 51..100. A secondary linear-only truth uses 200×200.
- **Estimator.** Unchanged: KP2 M3, free per-pass Doppler bias, GL1800F 50×50, HIGH_FIDELITY.
- **Stage 1 (linear, 10 cells).** The omission column on the FE-P designs, scored as:
  - N = 0;
  - FE N = 4 / 6 data-only;
  - FE N = 4 / 6 with the ×1 prior.

  Linear masquerade means fv ≥ 13.82 with residual ≤ 2√(2N).
- **Stage 2 (nonlinear).** The 4 MM cells, with the MM seeds and baselines, plus any linear-flagged cell. None was
  flagged, so the run set is the 4 MM cells.

## 2. Gates

| gate | value | status |
|---|---|---|
| L1: GR−GL column on the FE-P designs | 0.0 | PASS |
| L2: predeclared convolution vs propagated difference (CANON k0, D = 100) | 3.41 | **FAIL** |
| L2 diagnosis: ε = 0.01 / 0.1 scaled omission | 3.411 / 3.411; error ∝ ε | not nonlinearity |
| L2 on the fine grid, sub = 6 / 12 (spline x, Φ; exact acceleration at sub-steps) | 5.9e-3 / 3.3e-3 | PASS |
| Upstream: GR−GL column, coarse grid (full / deg 2–6 / deg 2–20) | 3.1e-3 / 1.3e-3 / 2.8e-3 | PASS |

The FE-L convolution samples the acceleration only at the 90 s measurement grid. At 83–115 km altitude a
degree-100 term has a ≈ 109 km wavelength, which the orbit crosses in ≈ 70 s, so the trapezoid aliases.

The GR−GL mismatch is dominated by degrees 2–6, which alone give 8.46 m of the 8.40 m position difference, so
the coarse grid is adequate for it. The FE-L, FE-P and MM linear numbers stand.

The fine-grid correction is a post-predeclaration oracle repair. It was re-gated under the unchanged L2 criterion.

## 3. Results

### 3.1 Linear (fine grid), GL1800F 51..100 omission

| cell | fv_chi2 N = 0 | residual N = 0 | window | FE N = 6 data-only fv | ×1 prior N = 6 fv | level bias / σ |
|---|---|---|---|---|---|---|
| CANON k0 | 1.55e7 | 8.2e6 | 122 | 1.1e5 | = N = 0 | −343 |
| CANON k12 | 2.09e7 | 3.9e7 | 113 | 1.1e4 | = N = 0 | −492 |
| O1 k0 | 1.18e7 | 1.8e7 | 122 | 1.8e5 | = N = 0 | −643 |
| O1 k12 | 8.4e6 | 5.6e7 | 115 | 3.5e4 | = N = 0 | −723 |
| O2 k0 | 1.03e7 | 2.0e7 | 145 | 1.0e5 | = N = 0 | −66 |
| O2 k12 | 1.4e6 | 2.9e6 | 118 | 9.5e4 | = N = 0 | +189 |
| O3 k0 / k12 (500 km) | 0.00 | 0.0 | 132 / 126 | 0 | 0 | 0.0 |
| O5 k0 | 1.65e7 | 6.5e6 | 122 | 2.3e4 | = N = 0 | −274 |
| O5 k12 | 7.2e7 | 3.4e7 | 114 | 1.8e4 | = N = 0 | +1414 |

The 101..200 increment changes little (D200 ≈ D100). There is no linear masquerade at D = 100 or D = 200.

### 3.2 Nonlinear

All four solves (CANON k0, O1 k12, O2 k0, O5 k0) stopped with "non-positive K candidate with no step to
reduce". This is the failure mode of MM O2 α = 1. Two numbers explain it:
- the omission acceleration is ≈ 2.5e-5 m/s² rms;
- the SRP acceleration is ≈ 5e-8 m/s².

The error is ~500× the SRP signal, so the K-only parameterisation cannot absorb it without negative K. By the
predeclared rule this is INCONCLUSIVE. Operationally the misfit announces itself through solver failure; it is
not a hidden false variation.

### 3.3 Post-hoc: what a 100×100 estimator would leave (GL1800F 101..200, linear only)

This variant is labelled and post-hoc.
- **N = 0:** fv 511–18391 with residual 6.2e3–1.6e4 against a window of 113–145. Large and visible; no
  masquerade.
- **FE N = 6 data-only:** fv 22–2045.

The design partials are the 50×50 model's, a first-order approximation for this question.

## 4. Post-hoc: masquerade is an amplitude window, not a property of one error

fv_chi2 and the residual chi2 both scale with the square of the error amplitude β. An error shape can therefore
masquerade for some β exactly when

  fv / residual ≥ 13.82 / window ≈ 0.11.

When it can, the window runs over

  β ∈ [√(13.82/fv), √(window/residual)].

Ratios and windows (β relative to the column as computed):

| cell | GR−GL (MM) | GL1800F 51..100 | GL1800F 101..200 |
|---|---|---|---|
| CANON k0 | 0.38 · [0.24, 0.43] | 1.9 · [0.0009, 0.0039] | 0.27 · [0.056, 0.087] |
| CANON k12 | 0.0014 · closed | 0.54 · [0.0008, 0.0017] | 0.65 · [0.048, 0.11] |
| O1 k0 | 6.1 · [0.031, 0.22] | 0.66 · [0.0011, 0.0026] | 0.61 · [0.049, 0.11] |
| O1 k12 | 0.45 · [0.096, 0.19] | 0.15 · [0.0013, 0.0014] | 0.17 · [0.12, 0.14] |
| O2 k0 | 0.72 · [0.11, 0.30] | 0.50 · [0.0012, 0.0027] | 0.46 · [0.053, 0.12] |
| O2 k12 | 3.6 · [0.094, 0.52] | 0.49 · [0.0031, 0.0064] | 0.033 · closed |
| O3 k0 | 8.3 · [0.31, 2.8] | 2.0 · [54, 240] | 0.12 · [6.7e7, 7.2e7] |
| O3 k12 | 0.12 · [1.2, 1.2] | 0.043 · closed | 0.22 · [1.1e8, 1.6e8] |
| O5 k0 | 0.095 · closed | 2.5 · [0.0009, 0.0043] | 0.30 · [0.08, 0.13] |
| O5 k12 | 0.42 · [0.11, 0.21] | 2.1 · [0.0004, 0.0018] | 2.0 · [0.027, 0.11] |

Reading:
- MM's own cases are consistent with this. CANON k0 α = 0.3 lies inside its window [0.24, 0.43]. O3 k0 α = 1
  lies inside [0.31, 2.8], the cell FE-P §3.3 flagged.
- The GL1800F-internal shapes have open windows in almost every low-orbit cell. The real error amplitudes sit
  far above them:
  - the 51..100 omission is ~150–700× above the top of its window;
  - the 101..200 omission, for a 100×100 estimator, is ~7–11× above.
- Hence the integrity risk is not a particular gravity model. It is an unmodelled error of **intermediate**
  amplitude: large enough to bias K, small enough to hide in the residuals. Examples would be the remaining
  omission of a higher-degree estimator, or a commission error well above the published σ (FE-P §3.4: ×100 σ
  gives 4 % of draws ≥ 13.82).
- Which real errors fall in such a window, for which truncation and altitude, is an attribution / integrity
  question. It is not computed here.

## 5. Answer to the owner's question

With GL1800F alone, the gravity error a GL1800F 50×50 estimator really makes on these orbits does not masquerade
as K variation. On the low orbits (CANON: 83–115 km) it is so large that the fit fails visibly. At 500 km it is negligible. The FE
corrections (N = 4, 6) and the GL1800F prior cannot absorb it either.

The MM masquerade remains a real mechanism, but it needs an error of intermediate size (§4). The GRGM−GL
difference at α = 0.3 happens to be one; GL1800F's own truncation at 50 is not.

As computed, the predeclared outcome is MMG_INCONCLUSIVE. Two reasons:
- no solve converged;
- the predeclared L2 gate failed, a coarse-grid sampling defect specific to the high-degree columns, repaired and
  re-gated.

## 6. Not claimed

- The common K level.
- That the GL1800F omission is the total real field error.
- Any nonlinear MM-G result.
- That the post-hoc amplitude windows generalise beyond these 10 cells.
