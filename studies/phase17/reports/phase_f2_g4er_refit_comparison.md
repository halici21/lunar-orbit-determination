# Phase F2 — F2g/h/i refit on the G4e-R numerics, compared with the historical records

2026-10-30. Code `fd5ef6a` (branch `fix/f2-g4e-sensitivity-station-epoch`), which contains:
- the G4e-R fixes `a8f1bef`: station-epoch quantisation, FD sensitivity numerics, shadow guard;
- the TRF merge path `f8692a3`;
- stop-on-certified-stationarity `6923a0f`.

The historical records are unchanged. The refit records sit beside them under new names.

## What changed between the historical records and these refits

Two things changed at once, so the differences below cannot be split between them:

1. **Solver:** legacy LM (`historical_legacy`) became bounded TRF (`bounded_trf`). TRF stops at the first accepted point
   that passes `posterior_diagnostics` with stationarity_sigma 0.1. That point has at most Δχ² 0.01 left to gain.
2. **Numerics and stations:**
   - exact-epoch station states;
   - the FD sensitivity columns and the shadow guard repaired in G4e-R
     (`phase_f2_g4er_root_cause_and_requalification.md`).

Model, data, priors, weights, starts and prediction days are those of the original predeclarations
(`studies/phase17/configs/phase_f2{g,h,i}_*_predeclaration.json`). G4e-R was rerun under the refit code identity and
**passes**: G0R, G4eR_dyn, G4eR_meas and G4eR_jump all pass (`results/phase_f2_g4er_checks_fd5ef6a.json`).

## Results

`results/phase_f2_refit_compare_trf_g4er.json` (built by `scripts/phase_f2_refit_compare.py`).
- LOLA R/A/C are the radial, along-track and cross-track RMS differences from the LOLA-based reference ephemeris over
  the fit span.
- The prediction is the third day (2009-262), propagated from the fit.

### F2g — two starts, 4 empirical terms with σ 1e-8 m/s² (`results/phase_f2g_lro_empirical_trf_g4er.json`)

| | S1 historical | S1 refit | S2 historical | S2 refit |
|---|---|---|---|---|
| status | DampingLimit | **STATIONARY** (3 lin.) | DampingLimit | **STATIONARY** |
| range / Doppler post-fit | 1.229 m / 2.52 mm/s | 0.980 m / 2.38 mm/s | **114.0 m / 121 mm/s** | 0.980 m / 2.38 mm/s |
| K_SRP ± σ | 0.0795 ± 3.8e-4 | 0.0739 ± 4.5e-4 | 0.0000 (bound) ± 6.4e-4 | 0.0739 ± 4.5e-4 |
| fit vs LOLA R/A/C [m] | 1.6 / 60.1 / 78.1 | 1.0 / 13.4 / 19.2 | 59.9 / 264.8 / 155.3 | 1.0 / 13.4 / 19.2 |
| prediction range / Doppler RMS | 4.20 m / 2.96 mm/s | **1.70 m / 1.79 mm/s** | — | — |
| prediction vs LOLA R/A/C [m] | 2.8 / 8.0 / 62.1 | 1.7 / 6.5 / 6.2 | — | — |

| gate | historical | refit |
|---|---|---|
| G0 | PASS | PASS (G0R) |
| G1 (certified stationary) | FAIL | **PASS** |
| G2 (start difference ≤ 1 m) | FAIL, 447.2 m | **PASS, 0.021 m** |
| G3 (post-fit) | FAIL | **PASS** |
| G4e / G4e-R | G4e FAIL | **G4e-R PASS** |

- **All F2g gates pass for the first time.**
- Historically, S2 was trapped at K = 0 with 114 m residuals. Now both starts reach the same solution, 2 cm apart.
- The empirical terms change too, beyond their σ: C_N goes from +9.4e-9 to +2.6e-8 and S_N from +4.4e-8 to −5.7e-8.
  The 4-term along-track plus normal model still converges to the same physics.

### F2h — along-track-only model, start S1 (`results/phase_f2h_lro_alongtrack_empirical_trf_g4er.json`)

| | historical | refit |
|---|---|---|
| status | DampingLimit | **STATIONARY** (3 lin.) |
| range / Doppler post-fit | 0.984 m / 2.43 mm/s | 0.981 m / 2.43 mm/s |
| K_SRP ± σ | 0.0742 ± 1.8e-4 | 0.0749 ± 3.5e-4 |
| C_T / S_T [m/s²] | −1.94e-8 / +2.44e-7 | −2.00e-8 / +2.48e-7 |
| fit vs LOLA R/A/C [m] | 1.7 / 20.4 / 27.9 | 1.7 / 19.5 / 26.8 |
| prediction range / Doppler RMS | 1.63 m / 1.81 mm/s | 1.75 m / 1.79 mm/s |
| prediction vs LOLA R/A/C [m] | 2.3 / 6.6 / 10.5 | 2.2 / 6.0 / 9.3 |
| gates | G1 FAIL, G3 PASS | **G1 PASS**, G3 PASS |

- F2h barely moves. ΔK = +0.0007, which is 2σ of the historical σ.
- **The formal K σ doubles, from 1.8e-4 to 3.5e-4.** The repaired K/empirical partials change by up to tens of
  percent (G4e-R report), so the historical formal σ is not comparable. It should not be quoted as a K precision.

### F2i — basin robustness: S2 = S1 + (100 m, 0.1 m/s) (`results/phase_f2i_part_s2_trf_g4er.json`)

The historical F2i record is on `9dc737f` (branch `phase-f1-lro-residuals`, not an ancestor of this branch):
`studies/phase17/results/phase_f2i_lro_basin_robustness.json`.

| | historical S1R | historical S2 | refit S2 |
|---|---|---|---|
| status | DampingLimit (11 it.) | DampingLimit (28 it.) | **STATIONARY** (5 lin.) |
| range / Doppler post-fit | 0.984 m / 2.43 mm/s | 0.982 m / 2.43 mm/s | 0.981 m / 2.43 mm/s |
| K_SRP ± σ | 0.07424 ± 1.8e-4 | 0.07447 ± 2.1e-4 | 0.07493 ± 3.5e-4 |
| fit vs LOLA R/A/C [m] | 1.7 / 20.4 / 27.9 | 1.6 / 20.1 / 27.4 | 1.7 / 19.5 / 26.8 |

**S1R was not rerun.** On the TRF path, F2i S1R is the same computation as F2h S1: same priors F2H_PRIORS, same start,
same arc. The legacy-only step limit and Gauss–Newton option are off on TRF. F2i S2 was therefore compared against F2h
S1 (same code identity), and `--f2i-merge` was not run. Every difference between the two fits is far below its formal
σ:

| quantity | F2i S2 − F2h S1 | F2h S1 σ |
|---|---|---|
| χ² | 11 012.658 − 11 012.570 = 0.087 | — |
| initial position / velocity | 0.45 mm / 7.1e-5 mm/s | — |
| K_SRP | 2.2e-6 | 3.5e-4 |
| range biases (103 / 105 / 126) | 0.15 / 0.16 / 0.16 mm | 164 / 177 / 170 mm |
| Doppler bias | 2.4e-6 mm/s | 0.038 mm/s |
| C_T / S_T | −4.9e-13 / 1.0e-11 m/s² | 3.8e-10 / 2.1e-9 m/s² |
| along-track RMS vs LOLA | 0.31 mm | — |

**G2 as predeclared** (largest difference between the two estimated trajectories) **was not computed.**
- The full F2h record drops `est_positions`.
- An initial-state difference of 0.45 mm and 7e-5 mm/s cannot reach the 1 m threshold over the 2-day arc:
  ~7e-8 m/s × 1.7e5 s ≈ 0.012 m before dynamics amplification.
- The historical value was 0.991 m, a marginal PASS.
- The χ² difference of 0.087 is larger than the 0.01 certification bound for each fit. Each fit's local
  model predicts less than 6e-4 χ² left to gain (GN decrements 0.017 and 0.023). The difference is therefore not
  explained by the linear model: it is nonlinearity or integration noise at the 1e-5 relative level, not a sign that
  either fit stopped early.

## Evaluation counts (stop rule)

| fit | linearizations | SciPy status | remaining GN decrement |
|---|---|---|---|
| F2g S1 | 3 | −2 (certified stop) | 0.016 |
| F2g S2, 1-day stage | 4 | −2 | — |
| F2h S1 | 3 | −2 | 0.017 |
| F2i S2 | 5 | −2 | 0.023 |

- At their third linearization, F2g S1 and F2h S1 reproduce run 1's χ² (10 764.871 / 11 012.570) exactly. Run 1 had
  no stop rule and was killed on 2026-10-27; its later linearizations gained 0.01–0.03 χ²
  (`phase_f2_trf_wasted_evaluations.md`).
- Wall times in the records (5–24 h) are **not a cost measure**. They include ~13 h of hibernation (2026-10-29 01:09 →
  14:07) and CPU sharing with G4e-R and another session's F2 run.

## Not claimed

- **The change of solver and the change of numerics are not separated.** No run used legacy LM with the new numerics,
  or TRF with the old numerics.
- The formal σ are local and unconstrained, not calibrated. Historical and refit σ differ by up to 2× and should not
  be compared as precisions.
- F2i G2 was not evaluated as predeclared (see above), and no F2i merged record exists on this code.
- These are the same three tracking days (2009-260…262). Nothing here says how the model behaves on other arcs.

## Run notes

- Runs 1–3 died without results:
  - run 1: session end and sleep, 2026-10-27;
  - runs 2 and 3: their Git Bash wrapper exited with 0xC000013A, cause not identified.
- Run 4 launched Python from Task Scheduler through `cmd`, with no bash, and completed.
- Full records, including final linearizations, are in `outputs/numerical-core/trf/` (gitignored). The committed
  copies drop `final_linearization` and `est_positions` for size.
