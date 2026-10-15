# Phase 17-MM — Gravity-field model-mismatch masquerade test

- **Date:** 2026-10-12 (run) – 2026-10-14 (analysis)
- **Branch:** local `phase17-harden-bls1` (worktree `python_port_phase17r0`)
- **Predeclaration:** `artifacts/phase17_mm_predeclaration.json`, committed in `a735e2b` before any mismatch solve
- **Evidence:** `artifacts/phase17_mm_summary.json`, `artifacts/phase17_mm_jobs/*.json`,
  `examples/phase17_mm_experiment.py`, `tests/test_mm_truth_field.py`
- **Origin:** the minimum falsification experiment of `docs/phase17_k_closeout_and_model_mismatch_pivot.md` §3.4

## 0. Final status

```
PHASE17_MM_OUTCOME   = H_MM_SURVIVES_MASQUERADE  (predeclared rule, as computed)
DECIDING RUN         = CANON k0, alpha = 0.3: false M3 variation chi2_detect = 28.8 (>= 13.82),
                       residual excess delta_chi2 = 74.8 <= 2 sqrt(2N) = 122.4; per-pass runs test blind
                       (range Z -0.10, Doppler Z -0.31; 0 of 23 + 23 passes with |z| > 3)
CONSEQUENCE          = a K time variation recovered from tracking (KP2 M3_VARIATION_PARTIALLY_RECOVERABLE) is
                       NOT claimable without independent knowledge of the Moon-fixed gravity field: a field error
                       invisible in the residuals produces a false variation of the same size KP2 recovers
HARNESS              = baselines reproduce KP2's stored costs to 4.4e-16; truth field = a_GL + alpha (a_GR - a_GL)
                       to 4.4e-11
NOT CHANGED          = KP2's classification under truth = model; production code (untouched in this phase)
NOT CLAIMED          = the common K level; any statement about real GRAIL commission error (§4)
```

## 1. Hypothesis and design (as predeclared)

**H_mm.** A Moon-fixed gravity-field error whose residual signature is statistically undetectable produces a
significant false M3 K variation.

- **Truth:** K constant (T0, K0 = 0.01 m²/kg).
- **Truth field:** GL1800F 50×50 + α·(GRGM1200L 50×50 − GL1800F 50×50), α ∈ {1, 0.3}. It is built on GL1800F's
  μ, R and frame (`MOON_PA_DE440`), with GRGM1200L rescaled to them. Its perturbing acceleration equals
  a_GL + α(a_GR − a_GL) to 4.4e-11; `tests/test_mm_truth_field.py` pins this.
- **Estimator:** the KP2 M3 solver, unchanged (free per-pass Doppler bias, high fidelity), with GL1800F 50×50.
- **Cells:** CANON k0, O1 k12, O2 k0, O5 k0.
- **Draws:** each run reuses its cell's KP2 T0 draws (a priori offset, range noise, per-pass bias, Doppler noise),
  so every run is a paired comparison with that cell's existing truth = model solution.
- **Rule:**
  - **masquerade** if χ²_detect ≥ 13.82 and Δχ² ≤ 2√(2N), on a converged solve;
  - **falsified** if every detection carries a detectable residual excess, or no run detects;
  - **inconclusive** if a deciding run failed.

## 2. Results

| cell | α | χ²_detect (≥ 13.82) | Δχ² (≤ 2√(2N)) | masquerade | runs Z range / Doppler | Doppler wrms (base) | level error (z) | max \|x₀ err\| / σ |
|---|---|---:|---:|---|---|---|---|---:|
| CANON k0 | 1 | 269.9 | 714.9 (122.4) | no: detectable | 0.40 / **−7.55** | 1.325 (0.992) | −16.9 % (−17.9) | 60.9 |
| O1 k12 | 1 | 1532.7 | 3183.4 (115.2) | no: detectable | −0.90 / **−18.45** | 2.197 (0.953) | +8.7 % (+6.2) | 47.5 |
| O2 k0 | 1 | — | — | solver failed (non-positive K, §3) | — | — | — | — |
| O5 k0 | 1 | 68.5 | 549.6 (122.4) | no: detectable | 0.68 / **−6.27** | 1.260 (0.999) | −18.6 % (−18.4) | 50.1 |
| **CANON k0** | **0.3** | **28.8** | **74.8 (122.4)** | **YES** | −0.10 / −0.31 | 1.031 (0.992) | −5.8 % (−6.1) | 17.5 |
| O1 k12 | 0.3 | 147.3 | 260.7 (115.2) | no: detectable | −0.74 / −4.32 | 1.109 (0.953) | +3.9 % (+2.8) | 13.7 |
| O2 k0 | 0.3 | 110.0 | 153.3 (144.9) | no: marginally detectable | −0.23 / −1.23 | 1.012 (0.952) | +12.8 % (+3.6) | 14.9 |
| O5 k0 | 0.3 | 9.7 | 36.4 (122.4) | no: no detection | 1.16 / −0.05 | 1.017 (0.999) | −5.7 % (−5.6) | 15.0 |

Baseline (truth = model) runs-test Z lie in −0.67 … +1.34. Range wrms never moves by more than 0.5 %; the
signature is in the Doppler.

**The masquerade case (CANON k0, α = 0.3):**

- **False variation:** segment fractions f̂ = (−9.3 %, −3.4 %, −4.6 %) ± (1.4, 1.1, 1.0) %, i.e. an apparent K
  step of ≈ 5 % at the first segment boundary.
- **Variation coordinates:** v̂ = (−0.041 ± 0.009, −0.015 ± 0.007).
- **Comparison:** that is the size of the 10 % variations KP2 counts as recovered.
- **Visibility:** the field error that produces it adds Δχ² = 75 to 1,873 residuals, about 1.7 σ of the χ² test's
  own scatter (√(2N) = 61). The runs test is blind (no pass flagged), and Doppler wrms rises 4 %.
- **Hidden by the fit:** the same error moves the common level by −5.8 % (6 σ) and the initial state by up to
  17.5 σ. The fit absorbs it into the state and K rather than leaving it in the residuals.

The linear prediction (report §3.2 / §3.4) held:

- At α = 1 every converged cell is detectable, with a large false variation.
- At α = 0.3, CANON sits inside the window.
- O1 and O2 stay detectable at α = 0.3.
- O5 at α = 0.3 is undetectable but also produces no significant false variation.

## 3. The failed solve (O2 k0, α = 1)

The KP2 solver stopped with "non-positive K candidate with no step to reduce": the field error drove K through
zero on the full-Sun orbit. The solve is recorded as failed and not used. It does not decide the outcome, which
the converged CANON α = 0.3 run already settles. A failure of this kind is itself a symptom of gross mismatch:
residual excess at α = 1 is 5–28 × the threshold in the converged cells.

## 4. What this does and does not say

- **Does:** within this error class, residual tests on the tracking data cannot tell a ~5 % K step from a
  Moon-fixed gravity error at 30 % of the GL1800F − GRGM1200L difference. A K-variation detection from tracking
  therefore needs independent gravity-field knowledge, or a field-error parameterisation in the estimator, before
  it can be claimed.
- **Does not say anything about GRAIL accuracy.** The GL1800F − GRGM1200L difference at degree 50 is dominated by
  degrees 2–6 and frame / tide-convention terms (C21: 2.3e-10 vs −6.9e-9), not formal commission error (report
  §3.2). α = 0.3 of a convention difference is a plausible operational inconsistency, not a GRAIL error estimate.
- **Single realisation.** One noise draw per cell (the KP2 draws). The window is a linear-model property (ratio of
  false-variation to residual χ², amplitude-independent), and this run confirms it nonlinearly in one cell. Its
  frequency across draws and fields is not measured.
- **Unchanged:** KP2's classification is a statement under truth = model and stands as such. The common K level
  remains unclaimed, now with a second reason: gravity error also biases the level by 6 σ here.

## 5. Next questions (not started)

- **Estimator-side:** add a low-degree field-error parameterisation (e.g. C21/S21 and degree-2–6 corrections)
  alongside M3 and test whether it removes the false variation without killing the true one.
- **Data-side:** real LRO / GRAIL DSN tracking against the NAIF reconstructed orbit (K-closeout report §3.4).

## 6. Verification

- Production code is unchanged in this phase (it is identical to the Phase 17 closeout tree, verified 1497 passed,
  29 skipped, 0 failed).
- `tests/test_mm_truth_field.py`: 2 passed (truth-field construction; frozen rule).
