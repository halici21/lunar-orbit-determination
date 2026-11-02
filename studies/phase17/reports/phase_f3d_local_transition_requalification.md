# Phase F3d: F3 / F3b with accurate local transitions

**Decision as computed: F3D_NOT_QUALIFIED.** Two gates failed: G_PHI, and F3's G1 on the σ = 0 control.

Information only: with either set of transitions, F3's and F3b's conclusions are unchanged.
- σ* = 1e-9 either way.
- F3 is F3_NO_DMC_GAIN either way.
- F3b is F3B_MIXED either way, with shares within 0.04.

Files:
- predeclaration `configs/phase_f3d_local_transition_requalification_predeclaration.json` (`2e7f2a4`)
- code `ce79ad9`
- driver `scripts/phase_f3d_local_transition_requalification.py`
- results `results/phase_f3d_local_transition_requalification.json`, `phase_f3d_sweep.json`, `phase_f3d_split.json`

## 1. Why

C6 found that F3 forms each 20 s transition as Φ(t_b)·Φ(t_a)⁻¹ from its finite-difference Φ(t, t0) (report C6 §6).
- cond(Φ) reaches 8.5e11 by the end of the arc.
- After day 1 the local transitions are no longer symplectic: the error is 0.6–0.8.

State propagation telescopes, so σ = 0 is exact. With σ > 0, however, the DMC noise injected mid-arc is mapped through these transitions.

## 2. What was done

F3d replaces `phase_f3_core.transition` without editing any F3 file.
- Φ(t_b, t_a) is integrated by RK4 with steps of 1 s or less.
- The gravity gradient is linearly interpolated from C6's cached full gradient (GL1800F N = 300, plus the Moon point mass, Earth and Sun) at 1 s nodes.
- Everything else is F3's code, run unchanged: the sweep and the selection, H1/H2 and decision rules. F3b's split is rerun at the new σ*.
- G_LOC passed: the closed-form oscillator transition is matched to 1e-10.

## 3. Gates

| gate | result |
|---|---|
| G_LOC | PASS |
| G_PHI (Φ_acc vs Φ_FD, ≤ 1e-3 per column, fit arc) | **FAIL**: 1.0–1.1 % in every column, 2.0 % including 263 |
| G1 (F3's σ = 0 control) | **FAIL**: epoch deviation 0.087 σ (limit 0.05); fit and 263 RIC within 0.02 m of F2p (PASS) |

G1's failure follows from G_PHI. With accurate transitions, the σ = 0 smoother no longer telescopes to Φ_FD, so it is not identical to the batch built on h0 = hx·Φ_FD.

The 1 % disagreement is uniform across columns. That points to the gradient model, not to the FD Φ, which is a difference of nonlinear propagations of F2's own dynamics. Candidates:
- the omitted SRP position gradient (K = 0.0745, including the penumbra term);
- a frame or time difference in the harmonic rotation.

It is the same size and sign of problem as C6's K-column residual (1.07e-3). Not resolved.

## 4. Sweep, F3 (FD transitions) vs F3d (accurate transitions)

| σ (m/s²) | 263 chi² F3 → F3d | 263 RIC F3 → F3d (m) | mean Doppler NIS |
|---|---|---|---|
| 0 | 69 900 → 69 832 | 1.88/5.05/2.90 → 1.88/5.05/2.92 | 1.21 |
| **1e-9** | **3 643 → 4 119** | 1.89/7.52/1.06 → 1.89/7.77/1.03 | 1.17 |
| 3e-9 | 99 225 → 111 761 | 1.74/17.35/7.90 → 1.74/17.94/7.83 | 1.12 |
| 1e-8 | 317 626 → 335 373 | 1.69/27.30/17.93 → 1.69/27.79/17.91 | 1.09 |
| 3e-8 | 203 134 → 197 555 | 1.73/25.03/24.05 → 1.71/24.84/24.05 | 1.07 |

As computed, with the hypotheses not read because G1 failed:
- σ* = 1e-9.
- H1 false and H2 false, which would give F3_NO_DMC_GAIN.
- σ*'s 263 chi² is 13 % above F3's.

## 5. F3b split at σ* = 1e-9

| part | share F3b → F3d |
|---|---|
| STATE | −1.72 → −1.69 |
| CONST | 0.46 → 0.43 |
| DYN | 0.46 → 0.43 |
| MEAS | 0.002 → 0.002 |
| STATE+DYN | 1.000 → 1.000 |

F3b's gates pass and its decision is again F3B_MIXED.

## 6. Reading

The defect is real: F3's local transitions are not symplectic after day 1.

Its effect on F3's and F3b's science is small. The two sets of transitions bracket the answer:
- the FD set is globally exact but locally wrong;
- the gradient set is locally right but about 1 % off globally.

Across them, σ* and both decisions are the same, and every number moves by 13 % or less. F3's and F3b's records stand with this caveat attached.

A clean qualification needs local transitions that are both locally accurate and consistent with F2's dynamics. Options:
- find the 1 % gradient-model gap;
- obtain Φ(t_b, t_a) from short nonlinear FD propagations of F2's own Problem at each node. That costs roughly as much as F3's reference build, about 5 h.
