# Phase F2 G4e-R — G4e root cause and requalification of the F2g/h/i partials

2026-10-27. Base `b77cca6` (`codex/numerical-core-reliability`), branch `fix/f2-g4e-sensitivity-station-epoch`.
Predeclaration: [`configs/phase_f2_g4er_predeclaration.json`](../configs/phase_f2_g4er_predeclaration.json), written
before any G4e-R number. Result: [`results/phase_f2_g4er_checks.json`](../results/phase_f2_g4er_checks.json).

The historical G4e record (`results/phase_f2g_checks.json`, FAIL 1.5e-3) and its threshold are unchanged. The
adjudication proposed in [`phase_f2_lro_od.md`](phase_f2_lro_od.md) ("oracle floor; accept") is **withdrawn**: the
failure had a defect behind it, and the gate had a structural gap.

## Root cause (read-only analysis, historical code reproduced bit for bit)

1. **Station epoch quantisation (defect).** `phase_f1_lro_residuals.Setup._station` read SPICE at `et0 + t`. With
   et0 ≈ 3.06e8 s, ulp(et) ≈ 6e-8 s, so the station state steps by ~17 µm whenever an event epoch crosses an ulp
   boundary. In counted Doppler (c/(2Tc)·Δτ, cancellation factor 217–647) one such step is a 1.66e-6 m/s jump.
   99.9 % of the failing Doppler column 0 came from one row (row 5): its end-b uplink t1 moved by one ulp. The
   Doppler partial chain itself is correct; measurement-only differences fall as 1/h to ≤ 4.5e-6 at h = 1e-5.
2. **The gate did not test dynamics (structural gap).** The G4e "analytic" columns and their oracle came from the
   same ± propagations of `fd_history`. G4e therefore tested only the measurement layer, never dx/dp.
3. **Noisy FD sensitivities.** Against a converged reference (rtol 1e-12, h 1e-7), the F2g columns (rtol 1e-10, empirical
   step 1e-9) were 14–76 % off for the empirical terms and 4.3 % off for K_SRP. This is adaptive-step noise of DOP853
   dominating a small perturbation.
4. **Shadow guard rejected the F2 arc.** On the two-day N = 300 arc, the angle-only 1e-9 rad shadow-contact guard
   refused every tolerance: residuals were 3.4e-9, 2.2e-9 and 1.1e-9 rad at rtol 1e-10, 1e-11 and 1e-12. The cut
   was only ~1e-6 s from the contact.

## Changes

| file | change |
|---|---|
| `scripts/phase_f1_lro_residuals.py` | `Setup._station` adds the Fast2Sum rounding `t − (et − et0)` × station velocity. `EXACT_EPOCH_STATIONS = True`; `False` reproduces the historical records. |
| `scripts/phase_f2_lro_od.py` | Numerics are explicit per backend (`_numerics_for_backend`). TRF (qualified): rtol 1e-12, atol 1e-13, empirical FD step 1e-8. Legacy: unchanged (1e-10 / 1e-11 / 1e-9). Pool workers receive them. They are recorded in `numerical_profile`. `fd_history(keep=)` exposes the ± histories. |
| `scripts/phase_f2g_checks.py` | The historical G4e runs explicitly on legacy numerics, so its oracle still matches its record. |
| `src/lunar_od/dynamics.py` | `_shadow_contact_check` judges a cut by its **time offset** from the final trajectory's contact (angular residual / contact-function rate, limit 1e-3 s). The 1e-9 rad angle test remains as a sufficient pass, and stays the only test at a tangent (rate 0). |
| `scripts/phase_f2_g4er_checks.py` | New gate G4e-R. |

## G4e-R as computed (LRO arc, N = 300, range + Doppler, 4 empirical terms, 5 336 s)

| gate | threshold | result |
|---|---|---|
| G0R empirical = 0 vs production | bit-equal | **PASS** (max diff 0.0 m) |
| G4eR_dyn fd_history vs independent-step reference (position, 11 columns) | ≤ 1e-2 | **PASS**, max 1.08e-3 (C_N); state ≤ 4.8e-5, K 1.7e-4 |
| G4eR_meas range / Doppler partials vs FD of the full measurement | ≤ 1e-3 | **PASS**, range ≤ 3.9e-7, Doppler ≤ 6.1e-5 |
| G4eR_jump historical jump row, neighbour step | ≤ 1e-7 m/s | **PASS**, 4.1e-8 m/s (historical 1.66e-6), detrended σ 1.0e-8 |

The worst Doppler column improved from 1.5e-3 (historical G4e) to 6.1e-5, a factor of ~25. The K column's dynamical
error improved from 4.3 % to 1.7e-4.

## Tests

- New tests:
  - `tests/test_f1_station_exact_epoch.py`: the exact provider stays continuous across ulp boundaries (< 3e-7 m), the quantised one does not (> 1e-5 m).
  - `tests/test_shadow_contact_guard.py`: a contact cut passes; a 5 mm along-track offset passes (the old guard refused it); a cut 10 s away fails; a tangent keeps the angle test.
  - `tests/test_f2_qualified_adapter.py`: numerics per backend, and worker forwarding.
- Full suite (`--continue-on-collection-errors`, environment without PyQt5/matplotlib):

  | | passed | failed | errors | skipped |
  |---|---|---|---|---|
  | this branch | 1 621 | 19 | 3 | 39 |
  | baseline `b77cca6` | 1 617 | 19 | 3 | 39 |

  The failing sets are identical. All of them are pre-existing or environmental.

## Not claimed

- The F2g/h/i fits were **not** rerun with the new numerics and stations. Their records stand as they are, made at
  the legacy numerics with quantised stations.
- G1/G2/G3 stay open. G4e-R qualifies partials only and says nothing about orbit accuracy.
- Rerunning F2g/h/i on the TRF path is the next step. Expect slightly different post-fit values: the station fix is ~µm,
  but the empirical and K partials change by up to tens of percent.
