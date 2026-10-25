# Phase C4-I — Production planetary ephemeris DE440

- **Date:** 2026-10-23
- **Predeclaration:** `studies/phase17/configs/phase_c4i_de440_predeclaration.json`, committed before any number.
- **Baseline:** `studies/phase17/results/phase_c4i_baseline.json`, recorded from the pre-C4-I loader before any
  edit (`phase_c4i_baseline.py`).
- **Evidence:** `studies/phase17/results/phase_c4i_de440.json`, from `phase_c4i_de440_gates.py`.
- **Authorisation:** owner decision 2026-10-23, "DE440 after D2". The C4 survey classified the DE440 − DE421
  Moon-centred Earth difference as MATERIAL (OD 4.82σ).

## 0. Final status

```
STATUS     C4I_NOT_QUALIFIED as computed: G5 FAIL (2 fixture-behaviour tests); G1 G2 G3 G4 G6 PASS
PROPOSED   pin the two fixture tests to DE421 like the record tests (owner adjudication, s4)
OUTCOME    adjudication ACCEPTED by the owner 2026-10-25 -> C4I_QUALIFIED; both modules pinned to DE421 (PR #55)
DEFAULT    production planetary ephemeris is now DE440; DE421 selectable (ephemeris="DE421" or
           LUNAR_OD_PLANETARY_EPHEMERIS=DE421) and reproduces every pre-C4-I record bit for bit
DEFERRED   the D2 PropagationProfile default flip (D2 adjudications open; 3-10x runtime; Phase E first)
```

## 1. What changed

- **`spice_loader.py`** (not frozen):
  - New names: `PLANETARY_EPHEMERIS_KERNELS`, `DEFAULT_PLANETARY_EPHEMERIS = "DE440"`,
    `planetary_ephemeris()` and `required_kernels(ephemeris)`.
  - The selected `.bsp` occupies the slot `de421.bsp` held, so the load order (and with it the generic
    `MOON_PA` binding) is unchanged.
  - `required_kernel_paths` and `load_spice_kernels` take `ephemeris=`.
  - Selection order: argument, then `LUNAR_OD_PLANETARY_EPHEMERIS`, then the default. An unknown name raises.
- **`__init__.py`:** exports the new names.
- **Tests:**
  - New `tests/conftest.py` fixture `de421_ephemeris` (module scope). It sets the variable and reloads kernels,
    and on teardown restores the default.
  - The external `od_covariance_campaign` module is imported on DE421, because it samples its Earth table at
    import time and is a DE421-era record generator.
- **No production numerics** other than the ephemeris changed: GM constants, Earth orientation, lunar PA,
  propagators and estimators are untouched.

## 2. Gates as computed

| gate | result | number |
|---|---|---|
| G1 DE421 reproduction | **PASS** | `required_kernels("DE421")` equals the pre-C4-I list; Moon→Earth at 25 epochs 2025–2027 bit-identical |
| G2 DE440 default | **PASS** | only `de440.bsp` loaded; \|ΔEarth\| 3.15–3.89 m (≤ 5 m; C4 S2 3.92 m) |
| G3 environment override | **PASS** | `LUNAR_OD_PLANETARY_EPHEMERIS=DE421` bit-identical to G1; `"DE999"` raises |
| G4 frame contract | **PASS** | `test_de440_kernel_contract.py` 11 passed |
| G5 suite, default | **FAIL** | 1 903 passed, **2 failed**, 29 skipped (two runs, identical) |
| G6 suite, DE421 | **PASS** | 1 905 passed, 0 failed: the full pre-C4-I pass set |

### Pinned record tests (records rule)

Each failed under DE440 only because it reproduces a value computed on DE421. Each is pinned with
`pytestmark = ...usefixtures("de421_ephemeris")` and a comment naming the record:

| module | record |
|---|---|
| `test_ephemeris.py` | MATLAB `ephemeris_data.mat` / SPICE sampling fixture |
| `test_spice_snapshots.py` | MATLAB SPICE snapshot fixture |
| `test_fxr_earth_epoch_fixture.py` | published FXR range baseline |
| `test_r1o_cd1_counted_doppler.py` | R1O-CD1 / D3 range baseline |
| `test_r1od3_tracking_design.py` | canonical R1O-D2 result |
| `test_r1od4_gm_confirmation.py` | R1O-D3 anchor values |
| `test_r2_measurement_fidelity.py` | R2/R3 campaign golden values (the estimate-shift pin 0.018747738770222504) |

In the R2 module the `p04b_a07` comparisons are part of that frozen campaign. Under DE440 they fail at the
1e-8 / 2e-9 m/s level: production is 6.66e-8 from truth vs model S at 6.41e-8. This is recorded here so the pin
does not hide it.

### G5 failures, not pinned (not records)

1. **`test_estimator_lm_consistency::test_exact_optimum_start_neither_wanders_nor_claims_singularity`**
   - The test starts BLS-LM on the exact truth of its own noise-free fixture.
   - On DE421 every candidate is rejected and the start is returned untouched (`STATIONARY_AT_DAMPING_LIMIT`).
   - On DE440 a round-off candidate (max 9.3e-10 m, relative 1.7e-13) scores lower and is accepted, so the run
     ends `Converged / STEP_NORM`.
   - Not a defect: the test requires bit equality at the round-off floor, which holds only for its DE421 geometry.
2. **`test_harden_bls1_k_srp_recovery::test_bound_active_step_lets_the_orbit_converge_when_k_sits_at_zero`**
   - On DE440 the solve still converges, 0.25 mm from the truth, with K 3.7e-8 (approached, not clipped).
   - But it never produces a negative-K trial, so the bound-active path the test exists for is not reached.
   - The fixture was tuned on DE421.

## 3. Proposed adjudication (owner decision)

- Pin both fixture tests to DE421 as well. Their scenarios are frozen DE421-era fixtures whose purpose is to
  exercise a code path, which they still do on DE421.
- A DE440 counterpart with a re-tuned scenario can follow.
- With that: **C4I_QUALIFIED**.

## 4. Consequences and use

- New work (Phase F real data, external comparisons) runs on DE440 by default.
- To reproduce any pre-C4-I result, set `LUNAR_OD_PLANETARY_EPHEMERIS=DE421`. G6 shows this restores the whole
  suite.
- The frozen Phase 17 records and reports keep their DE421 numbers. Their scripts are unchanged and pick up the
  variable.

## 5. Limits

- GM constants still come from `gm_de431.tpc.txt`. `gm_de440.tpc` is not on disk; the GM differences are
  far below the ephemeris change (follow-up).
- The NAIF predict Earth PCK limitation (C3 T2B) is unchanged.
- The `PropagationProfile` default flip is not part of C4-I.
