# Phase F2 — First LRO orbit determination from real tracking (12 h, range only)

- **Date:** 2026-10-24
- **Predeclaration:** `studies/phase17/configs/phase_f2_lro_od_predeclaration.json` (`d6d9f0a`); amendment A1
  (`87fbe5d`) before any gate number.
- **Evidence:** `studies/phase17/results/phase_f2_lro_od.json`, from `studies/phase17/scripts/phase_f2_lro_od.py`.
- **Run history:**
  - Run 1 crashed: K_SRP reached 6e-4 and the K central difference went negative. Fixed in `4dd8573` with a
    forward difference at the bound.
  - Run 2 hung: worker stdout blocked under the process pool. Fixed in `2445f3f` and `6260407` by writing progress
    to log files.
  - Neither run computed a gate. Run 3 is the result.

## 0. Final status

```
STATUS      F2_NOT_QUALIFIED as computed: G1 FAIL, G2 FAIL, G3 FAIL; G4 PASS (partials 9.3e-7)
BEST FIT    N=100 (- start): post-fit RMS 14.46 m, 198 m RMS / 423 m max from the LOLA orbit
            (along-track 164 m RMS), but with K_SRP = 1.78 m^2/kg (~180x physical) and station biases
            100+ m away from F1's well-determined delays: the solution absorbs model error
OTHER RUNS  N=100 (+ start) and N=60 stall in another basin (121 m / 170 m RMS, ~1.6 km from LOLA, K -> 0)
CAUSES      (1) dynamics: degree <= 100 gravity cannot represent LRO's ~50 km orbit (LOLA used GRGM900C to 600);
            (2) observability: range only, two of three stations have a single pass whose bias absorbs that
                whole pass, so the problem has several minima from a 1 km / 1 m/s a priori
```

## 1. Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence (N=100, both starts) | **FAIL** | + start: MaxIter at 121.18 m; − start: DampingLimit (stationary, last step 3 µm) at 14.46 m |
| G2 start independence | **FAIL** | the two N=100 solutions differ by up to 2 090 m in position and 319 m in a bias |
| G3 post-fit RMS ≤ 2 m (N=100) | **FAIL** | 121.2 m (+ start, the gate's reference run) and 14.46 m (− start) |
| G4 partials vs full-model FD (N=60) | **PASS** | column errors ≤ 9.3e-7 (worst element 1.3e-4) |

G4 confirms the A1 FD-STM sensitivities and the analytic light-time partials on real geometry. The failures are
in the problem, not in the derivatives.

## 2. Solutions

| run | status | post-fit RMS | K_SRP [m²/kg] | biases 103 / 105 / 126 [m] | vs LOLA: R / A / C RMS [m] | total RMS / max [m] |
|---|---|---|---|---|---|---|
| N100 − | DampingLimit, 10 it. | **14.46 m** | 1.78 ± 0.0019 | 374.6 / 355.1 / 443.9 | 78.9 / 164.4 / 78.3 | **198 / 423** |
| N100 + | MaxIter, 30 it. | 121.18 m | 5e-6 | 631.5 / 626.3 / 763.4 | 656.5 / 1337.6 / 670.6 | 1634 / 2047 |
| N60 + | MaxIter, 30 it. | 169.62 m | 2e-6 | 426.4 / 514.3 / 381.0 | 618.4 / 1284.2 / 648.9 | 1566 / 1985 |

For comparison, F1 measured the station delays against the LOLA orbit itself: 487.8–490.8 m (Dongara),
476.9 m (South Point), 474.5 m (Kiruna).

### Why

1. **Single-pass stations.**
   - PAD 105 and 126 each have one pass, so their free biases absorb the whole mean of that pass. The residual
     mean of those passes is exactly 0 in every run.
   - The orbit is then effectively fitted to Dongara's four passes plus the shape, not the level, of the other
     two. That is weak along-track and cross-track information for 10 unknowns.
2. **Model error larger than the data noise.**
   - Propagated from the LOLA state, N=100 departs from the LOLA orbit by 266 m over the arc (A1 smoke test).
   - The best fit reduces the residual to 14 m only by putting a 180× solar-pressure coefficient and 100 m bias
     errors to work as empirical forces.
3. **Several minima.**
   - From ±(1 km, 1 m/s) the damped Gauss–Newton path splits after iteration 1. Both starts first push K to
     5–15 m²/kg.
   - The − start reaches the 14 m basin. The + start and N=60 end in a basin with K at its bound, crawling with
     millimetre steps under high damping.

## 3. What this does and does not show

- **Shown:** lunar_od can ingest real LRO tracking and run a full real-data fit end to end (UTDF, time scales,
  stations, media, SSB light time, harmonic dynamics with shadow events, partials). The partials are verified on
  the real geometry (G4).
- **Not shown:** meter-level real-data OD. The failure has two named causes, gravity resolution and observability.
  Neither is a defect found in the measurement chain, which F1 already verified at 0.5–0.85 m against the
  precision orbit.

## 4. Recommendations (next steps, ordered)

1. **Constrain the station biases with F1's calibrated values.** The delays are stable over 25 days (F1b). An F2
   rerun with F1-derived bias priors (σ ~3 m) removes the single-pass degeneracy.
2. **Add Doppler.** The UTDF files carry cumulative bias counts. Doppler gives the along-track information range
   lacks; this is F1c, the Doppler reader and residuals against the LOLA orbit.
3. **Gravity above degree 100.**
   - The 6-state numba path makes N=200–300 feasible in the residual.
   - The sensitivity propagator needs the numba harmonic acceleration first. That is Phase E, and it changes
     frozen `dynamics.py`, so it needs owner approval.
   - The harmonic STM path is qualified only to degree 100.
4. **A K_SRP prior from LRO's area-to-mass ratio** (~0.01 m²/kg), so K cannot act as an empirical force.
5. **Longer arcs or more stations per arc** to separate the orbit from the biases.

## 5. Limits

- One 12 h arc, range only, cannonball SRP, no momentum-dump modelling, no empirical accelerations.
- The a priori is 1 km / 1 m/s off the reference, a realistic but demanding start.
- FD sensitivities (A1).

## 6. F2b — F2 with station-bias and K_SRP priors (N = 100)

- **Predeclaration:** `studies/phase17/configs/phase_f2b_lro_od_priors_predeclaration.json` (`17bc923`).
- **Evidence:** `studies/phase17/results/phase_f2b_lro_od_priors.json`.
- **Priors:** biases from F1's per-station means (σ 3 m); K_SRP 0.010 ± 0.005 m²/kg.
- **Status:** all three gates FAIL as computed.
  - G1: + Converged, − DampingLimit.
  - G2: the two starts differ by 2 120 m.
  - G3: 126.7 m and 14.66 m against the 20 m limit.

| start | status | post-fit RMS | K_SRP | biases 103 / 105 / 126 [m] | vs LOLA R / A / C RMS [m] | total RMS / max [m] |
|---|---|---|---|---|---|---|
| + | Converged, 20 it. | 126.68 m | 0.0 | 696.4 / 692.2 / 826.0 | 654.3 / 1334.4 / 700.6 | 1643 / 2056 |
| − | DampingLimit, 11 it. | 14.66 m | 1.556 | 406.2 / 386.9 / 476.2 | 66.5 / 152.9 / 94.3 | 191.5 / 352.5 |

- The priors change almost nothing: about 2 380 range rows outweigh four pseudo-observations.
- With degree-100 gravity, the data still prefer a non-physical K and biased delays.
- **Conclusion: the dynamics, not the regularisation, limit F2.**

## 7. F2c — degree 200 and 300 (with F2b's priors)

- **Predeclaration:** `studies/phase17/configs/phase_f2c_lro_od_degree_predeclaration.json` (`07a643b`). It
  discloses a smoke test: propagating the LOLA epoch state departs from the LOLA orbit by 266 / 42 / 31 / 27 m at
  N = 100 / 150 / 200 / 300.
- **Evidence:** `studies/phase17/results/phase_f2c_lro_od_degree.json`.
- **Status:** all three gates FAIL as computed.
  - G1: DampingLimit for both N=200 runs.
  - G2: the two starts differ by 1 964 m.
  - G3: the + runs end at 130.5 m and 130.9 m against the 5 m limit.

| run | status | post-fit RMS | K_SRP [m²/kg] | biases 103 / 105 / 126 [m] (±0.71) | vs LOLA R / A / C RMS [m] | total RMS / max [m] |
|---|---|---|---|---|---|---|
| **N200 −** | DampingLimit (stationary from iteration 7) | **0.889 m** | **0.0132 ± 0.0017** | **482.2 / 470.3 / 473.1** | **3.2** / 90.4 / 91.6 | 128.7 / 178.9 |
| N200 + | DampingLimit, 16 it. | 130.51 m | 0.0 | 687.4 / 691.9 / 744.2 | 647.4 / 1327.7 / 705.1 | 1637 / 2040 |
| N300 + | DampingLimit, 28 it. | 130.94 m | 4e-5 | 688.5 / 693.3 / 743.7 | 646.7 / 1326.7 / 708.4 | 1637 / 2040 |

### The N = 200 (−) solution: the first physically consistent real-data orbit

- **Post-fit residuals 0.55–1.26 m RMS per pass**, with per-pass means ≤ 0.61 m. This equals F1's measurement-chain
  floor (0.50–0.85 m detrended against the precision orbit). The solve stopped at iteration 7 and was stationary
  thereafter.
- **K_SRP = 0.0132 ± 0.0017 m²/kg.** This is physical: it agrees with LRO's area-to-mass ratio, ~0.01. The prior
  carried weight 0.005 and did not force it.
- **Station biases 482.2 / 470.3 / 473.1 m (±0.7 m).**
  - Dongara is 7 m below F1's mean of the four passes (489.2 m). South Point (−6.6 m) and Kiruna (−1.4 m) are within
    a few metres of their priors.
  - This is consistent with F1's 3 m within-day spread and F1b's along-track signature of the reference.
- **Against the LOLA GRGM900C orbit:**
  - radial **3.2 m RMS** (6.2 m max);
  - along-track 90 m RMS, cross-track 92 m RMS.
  - Earth-based range over 12 h measures mainly the line-of-sight component. On LRO's polar orbit, cross-track and
    along-track positions are weakly observed, so they move freely within the data's null space. The 0.9 m fit does
    not constrain them to better than ~100 m.
  - Doppler and longer arcs carry that information (recommendations 2 and 5).

### The + starts

- From +(1 km, 1 m/s), N = 200 and N = 300 fall into the same basin as F2/F2b's + starts: K → 0, ~130 m post-fit,
  ~1.6 km from LOLA.
- **The basin is a property of the start and the damping path, not of the degree.** A better a priori, or a
  globalisation strategy (multiple starts, a homotopy in K), is needed for robustness.

## 8. F2 series summary and next steps

| | N=100 (F2/F2b) | N=200 (F2c) |
|---|---|---|
| best post-fit RMS | 14.5 m (non-physical K) | **0.89 m** (physical K) |
| best radial vs LOLA | 67–79 m RMS | **3.2 m RMS** |
| best total vs LOLA | 191–198 m RMS | 129 m RMS (along/cross-track unobserved) |

Next steps:
1. **Doppler (F1c):** cumulative bias counts in UTDF, for the along/cross-track information.
2. **Arc robustness:** a closer a priori or a multi-start strategy, so the + basin is avoided.
3. **Longer arc:** two days or more, more passes per station.
4. **Phase E (owner approval):** the numba harmonic acceleration in the sensitivity propagator, so the analytic STM
   replaces the A1 FD sensitivities.

## 9. F2d — combined range + Doppler, N = 200

- **Predeclaration:** `studies/phase17/configs/phase_f2d_lro_od_range_doppler_predeclaration.json` (`bfe9c9a`).
- **Evidence:** `studies/phase17/results/phase_f2d_lro_od_range_doppler.json`.
- **Data:** 2 473 ranges (σ 1 m) and 2 416 Doppler intervals (σ 2 mm/s).
- **Estimated:** initial state, K_SRP, 3 range biases and 1 global Doppler bias.
- **Disclosed smoke test:** from the LOLA epoch state at N = 200, the prefit was 14.1 m (range) and 11.9 mm/s (Doppler).
- **Runtime:** the run spent about a day suspended while the PC slept (iteration 22 is time-stamped 91 654 s). That
  does not affect the results.

### Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence | **FAIL** | − start: DampingLimit (the last accepted relative improvement was not below 1e-6, so not STATIONARY); + start: STATIONARY, in the wrong basin |
| G2 start independence | **FAIL** | the solutions differ by up to 1 983 m |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **FAIL** | + start: 148.1 m / 333 mm/s; − start passes both (0.916 m / 2.31 mm/s) |

### Solutions

| start | range RMS | Doppler RMS | Doppler bias [mm/s] | K_SRP | biases 103 / 105 / 126 [m] | vs LOLA R / A / C RMS (max) [m] | total RMS / max |
|---|---|---|---|---|---|---|---|
| **−** | **0.916 m** | **2.31 mm/s** | −8.05 ± 0.05 | 0.0084 ± 0.0012 | **487.2 / 475.3 / 477.7** | **1.95 (3.7)** / 84.0 (103.8) / 85.6 (127.5) | 119.9 / 164.2 |
| + | 148.1 m | 333 mm/s | +235.5 | 0.0 | 746.0 / 735.8 / 811.6 | 654.9 / 1331.2 / 704.1 | 1642 / 1973 |

### Reading

- **Range + Doppler, the − start:**
  - Every pass is at its noise floor: range 0.6–1.24 m, Doppler 2.3 mm/s.
  - K_SRP is physical.
  - The station delays now agree with F1's direct measurement against the LOLA orbit (489.2 / 476.9 / 474.5 m)
    to **2.0 / 1.6 / 3.2 m**. In F2c the gap was 7 m.
  - Radial against LOLA improves from 3.2 to **1.95 m RMS**.
- **Along-track and cross-track stay at ~85 m** (90 m in F2c).
  - The two are nearly equal and the radial is metre-level. That is the signature of a rotation of the orbit about
    the Earth–Moon line of sight. Earth-based range and Doppler both measure only line-of-sight quantities, so over
    a 12 h arc that rotation is almost unobservable. Only the lunar gravity field's slow action breaks it.
  - This is the classic weakness of Earth-tracked lunar orbiters. The LOLA precision orbits close it with altimetric
    crossovers.
  - A real-data fit can therefore resolve it only with longer arcs, a different observable (altimetry, ΔDOR,
    optical), or a stronger dynamical constraint.
- **Doppler bias.** The fitted −8.05 mm/s differs from F1c's −11.1 mm/s, which was measured against the LOLA orbit.
  About 3 mm/s of the line-of-sight velocity difference between the two orbits is absorbed there, consistent with
  the ~85 m rotation.
- **The + start** stays in the same wrong basin as in F2–F2c, even with Doppler. Basin selection from ±1 km starts
  is a property of the damped Gauss–Newton path.

## 10. Real-data status after F2d (summary)

| | value |
|---|---|
| post-fit range / Doppler | 0.92 m / 2.3 mm/s (at the measurement floor) |
| radial vs NASA/LOLA orbit | **1.95 m RMS** |
| along / cross-track vs LOLA | ~85 m (unobservable rotation about the line of sight over 12 h) |
| station delays vs F1 direct measurement | ≤ 3.2 m |
| K_SRP | physical (0.008–0.013 m²/kg) |

Next steps, in order:
1. **Longer arc:** several days within the same SPK arc (2009-09-15 to 10-23), more TRK days. A download is needed
   and will be reported first.
2. **A multi-start or homotopy strategy** for basin robustness.
3. **ΔDOR or optical** (the project has both models) as the observable for the line-of-sight rotation.
4. **Phase E (owner approval):** an analytic harmonic STM with numba, replacing the A1 FD sensitivities.

## 11. F2e — two-day range + Doppler arc (2009-260/261) with a one-day prediction test (2009-262)

- **Predeclaration:** `studies/phase17/configs/phase_f2e_lro_two_day_arc_predeclaration.json` (`0b42ab7`).
- **Amendment A1** (`5ac6910`, before any fit): Doppler intervals with an observed |range rate| above 3 km/s are
  physically impossible and are discarded. This removes all 482 PAD 105 intervals of 2009-261 (a constant
  15.84 km/s) and one PAD 103 counter rollover.
- **Evidence:** `studies/phase17/results/phase_f2e_lro_two_day_arc.json`.
- **Data:** 4 252 ranges and 4 158 Doppler intervals over 36.6 h.
- **Runs:**
  - A serial run was stopped at iteration 3 (owner instruction) and restarted in parallel mode, `--workers 14`,
    `04864b2`. The serial run's logs are kept in `lro_data/f2e_serial_aborted/`.
  - The pooled residual and Jacobian are bit-identical to the serial ones
    (`studies/phase17/results/phase_f2_parallel_equivalence.json`).
  - The iterates differ from the aborted serial run in the fourth digit (S1 iteration 1: 2.715 vs 2.713 m). The LM
    solve now runs in the main process instead of a process-pool worker, so the BLAS reduction order of the normal
    equations differs. The ill-conditioned step amplifies that rounding; the residual model is identical.

### Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence | **FAIL** | S1: DampingLimit (stationary at 2.263 m; the last accepted relative improvement was not below 1e-6); S2: MaxIter |
| G2 start independence (≤ 1 m) | **FAIL** | 214.3 m |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **FAIL** | S1 2.263 m / 3.70 mm/s; S2 17.65 m / 39.2 mm/s |

### Solutions

| start | range RMS | Doppler RMS | Doppler bias [mm/s] | K_SRP | biases 103 / 105 / 126 [m] | vs LOLA R / A / C RMS (max) [m] |
|---|---|---|---|---|---|---|
| **S1** (F2d solution) | **2.263 m** | 3.70 mm/s | −8.14 | 0.0357 ± 0.0002 | **489.7 / 477.7 / 476.8** | **3.16 (7.5)** / 63.8 (110.9) / 79.6 (127.0) |
| S2 (S1 + 100 m, 0.1 m/s) | 17.650 m | 39.2 mm/s | +15.27 | 0.0 | 522.3 / 517.5 / 511.3 | 61.3 / 144.2 / 163.8 |

S1 per pass:
- range RMS 0.71–2.59 m at Dongara and 1.89 m at South Point;
- the two Kiruna passes: 4.12 and 3.59 m, with means of +3.6 and −3.4 m.

### Prediction test (S1 propagated over 2009-262, not fitted)

| | value |
|---|---|
| range RMS / mean | 20.1 m / −9.9 m |
| Doppler RMS / mean | 21.0 mm/s / +20.4 mm/s |
| vs LOLA R / A / C RMS (max) | 11.9 (19.2) / 38.4 (79.0) / 64.3 (92.7) m |

| pass | hours after fit end | range mean [m] | range RMS [m] |
|---|---|---|---|
| 105 | 14.3 | −4.36 | 12.61 |
| 103 | 16.1 | −7.94 | 16.26 |
| 103 | 18.0 | −9.19 | 19.24 |
| 103 | 19.9 | −11.81 | 22.45 |
| 126 | 21.8 | −11.17 | 21.55 |
| 126 | 23.7 | −14.24 | 25.31 |

The prediction error grows almost linearly, by about 1 m per hour in range RMS.

### Reading

1. **Station delays.** The two-day fit puts them within **0.5 / 0.8 / 2.3 m** of F1's direct measurement against
   the LOLA orbit. Range calibration is effectively recovered from the data.
2. **Along-track and cross-track** improve with the longer arc: 84 → 64 m and 86 → 80 m against F2d. The radial
   stays at the 3 m level. The line-of-sight rotation weakness is reduced, but not removed, by a second day.
3. **The dynamics model shows its limits over two days.**
   - The post-fit range rises from 0.92 m (12 h) to 2.26 m (36.6 h), with systematic pass means at Kiruna
     (±3.5 m).
   - K_SRP rises to 0.036, 3.6× nominal, acting again as an empirical force.
   - The prediction diverges at ~1 m/h.
   - Candidate missing forces: gravity above degree 200 (F2c smoke test: 31 m at N=200 vs 27 m at N=300 from the
     LOLA state over 12 h); LRO's box-wing SRP and attitude (a cannonball with one K); thermal and planetary albedo
     radiation; momentum-wheel desaturations (not modelled; LRO performed them regularly); Earth albedo.
4. **S2** again lands in a different basin. Even a 100 m / 0.1 m/s offset from a good solution diverges over a
   two-day arc. Basin robustness needs a better-damped globalisation (a trust region with a geodesic step, or a
   homotopy in arc length), or a priori covariance.

### Next steps

1. **Dynamics fidelity:**
   - N = 300 in the residual (6-state numba path; 21.6 s per propagation);
   - empirical once-per-revolution accelerations (standard in lunar OD: GEODYN, MONTE);
   - momentum-dump epochs from the LRO SFF files in the same archive (`data/sff`).
2. **Arc strategy:** grow the arc from the converged one-day solution (12 h → 24 h → 36 h), instead of jumping to
   the full span from a distant start.
3. **Phase E (owner approval):** a numba harmonic STM. The parallel pool already gives 6×; the analytic STM removes
   the 15 FD propagations entirely.

## 12. F2f — two-day arc at degree 300; arc continuation for the second start

- **Predeclaration:** `studies/phase17/configs/phase_f2f_lro_degree300_continuation_predeclaration.json` (`80ebb38`).
- **Evidence:** `studies/phase17/results/phase_f2f_lro_degree300_continuation.json`, merged from
  `phase_f2f_part_s1.json` and `phase_f2f_part_s2.json`.
- **First attempt:** it ran the two starts as threads of one process and failed while loading SPICE kernels
  (CSPICE is not thread-safe). No computation ran. The starts then ran as separate processes, 7 workers each
  (`e814c36`).
- **Maneuver check:** the archive's FDF-25 files list MOI1–4 and SK02; none falls in the arc. Momentum
  desaturations are not in the archive.

### Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence | **FAIL** | S1 DampingLimit (stationary at 1.506 m; the last accepted relative improvement was not below 1e-6); S2 DampingLimit |
| G2 start independence (≤ 1 m) | **FAIL** | 262 m |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **FAIL** | S1 passes (1.506 m / 3.82 mm/s); S2 14.76 m / 35.8 mm/s |

### S1: degree 300 against F2e's degree 200 (same start, same data)

| | F2e (N=200) | **F2f (N=300)** |
|---|---|---|
| range / Doppler post-fit | 2.263 m / 3.70 mm/s | **1.506 m** / 3.82 mm/s |
| Kiruna pass means | +3.6 / −3.4 m | **+0.5 / −0.5 m** |
| station delays vs F1 (489.2 / 476.9 / 474.5) | 0.5 / 0.8 / 2.3 m | **0.8 / 1.3 / 0.04 m** |
| K_SRP | 0.036 | 0.040 |
| vs LOLA R / A / C RMS | 3.2 / 63.8 / 79.6 m | 3.5 / **21.8 / 25.4** m |
| prediction range RMS (2009-262) | 20.1 m | **12.1 m** |
| prediction vs LOLA R / A / C | 11.9 / 38.4 / 64.3 m | 12.7 / 39.1 / **9.4** m |

Prediction range RMS per pass: 9.3 m at 14.3 h after the fit end, rising to 14.6 m at 23.7 h; the means stay within
4 m.

### Reading

- **Gravity degree was the dominant missing physics.**
  - Going from 200 to 300 cut the along-track error to a third (64 → 22 m) and the cross-track error to under a
    third (80 → 25 m).
  - It removed the Kiruna pass biases and brought the post-fit range within the 2 m target.
  - The prediction improved from 20 to 12 m.
  - The line-of-sight rotation found in F2d was largely gravity-induced, not only an observability limit.
- **K_SRP stays high** (0.040 against ~0.014 expected from the 1 280 kg mass) and the prediction still grows. A
  residual non-gravitational signal remains: box-wing SRP and attitude, thermal radiation, desaturations. This is
  F2g's target, with empirical once-per-revolution accelerations.
- **S2**, even with arc continuation, converges to a wrong basin on the one-day stage (15.4 m) and keeps it on two
  days (14.8 m). The basin problem is independent of the arc length and the degree. It needs a different
  globalisation (multi-start, or a trust region on a scaled parameter space).

## 13. F2g — empirical once-per-revolution accelerations (N = 300)

- **Predeclaration:** `studies/phase17/configs/phase_f2g_lro_empirical_predeclaration.json` (`7ecc20d`).
- **Code:**
  - The empirical acceleration is a = (C_T cos u + S_T sin u)·T + (C_N cos u + S_N sin u)·N, with prior
    N(0, 1e-8 m/s²) on each coefficient.
  - It is added at study level on top of production's own RHS (`dynamics._build_state_rhs`) and its shadow-event
    integrator. Frozen `dynamics.py` is unchanged.
  - The FD sensitivities gain 4 columns (23 propagations per iteration, parallel). The light-time partials carry
    the extra columns through `position_sensitivity`.
- **Evidence:**
  - `studies/phase17/results/phase_f2g_lro_empirical.json` (merged from `phase_f2g_part_s1.json` and
    `phase_f2g_part_s2.json`);
  - `studies/phase17/results/phase_f2g_checks.json`.
- **SFF:**
  - The archive's FDF-25 files hold thruster maneuvers only (LOI, MCC, MOI, SK). None falls inside the arc, and
    there are no desaturation records.
  - The six maneuver files in the window were downloaded (reported first, ~35 KB).
  - MOI1 gives an average mass of 1 279.9 kg, so a cannonball with ~14 m² and Cr ~1.3 means K ≈ 0.014 m²/kg.
- **Note:** a first run of the check script was killed by its own 3 500 s timeout while F2f held all cores. It was
  rerun without a timeout, and the result above is from that rerun.

### Gates as computed

| gate | result | number |
|---|---|---|
| G0 equivalence (empirical = 0 vs production) | **PASS** | bit-identical on 6 590 epochs (max diff 0.0) |
| G4e partials (≤ 1e-3) | **FAIL** | range columns ≤ 3.6e-6; Doppler columns 3.7e-5, 3.1e-4, **1.1e-3, 1.5e-3** |
| G1 convergence | **FAIL** | S1 DampingLimit (stationary at 1.229 m, sub-µm steps); S2 DampingLimit |
| G2 start independence | **FAIL** | 447 m |
| G3 post-fit | **FAIL** | S1 passes (1.229 m / 2.52 mm/s); S2 114.0 m / 121 mm/s |

**G4e.** A 1e-9 m/s² empirical step changes a 5 s counted Doppler by only ~1e-4 m/s. At rtol 1e-10 the FD oracle
itself carries about 1e-3 relative noise. The range columns of the same parameters agree to 1e-6, and the Doppler
partial is the difference of two range partials. Proposed adjudication: oracle floor; accept.

### S1 against F2f (same start, same data, N = 300)

| | F2f | **F2g** |
|---|---|---|
| range / Doppler post-fit | 1.506 m / 3.82 mm/s | **1.229 m / 2.52 mm/s** |
| per-pass range RMS | 0.85–2.08 m | 0.53–2.06 m |
| station delays vs F1 | 0.8 / 1.3 / 0.04 m | 1.7 / 1.8 / 0.5 m |
| K_SRP | 0.040 | 0.0795 ± 0.0004 |
| empirical [m/s²] | — | C_T −1.18e-8 ± 4e-10, **S_T +2.71e-7 ± 2e-9**, C_N 0.94e-8 ± 7.8e-9, S_N 4.40e-8 ± 4.1e-9 |
| vs LOLA R / A / C, fit arc | 3.5 / **21.8 / 25.4** m | **1.6** / 60.2 / 78.2 m |
| prediction range RMS (2009-262) | 12.1 m | **4.2 m** |
| prediction Doppler RMS | 16.4 mm/s | **3.0 mm/s** |
| prediction vs LOLA R / A / C | 12.7 / 39.1 / 9.4 m | **2.8 / 8.0** / 62.1 m |

Prediction range RMS per pass: 2.2 m at 14.3 h after the fit end, rising to 7.0 m at 23.7 h.

### Reading

1. **The data demand an along-track once-per-rev signal.**
   - S_T = 2.7e-7 m/s² is 27σ of its prior, and the Doppler residual drops from 3.8 to 2.5 mm/s.
   - The **one-day prediction against independent data improves threefold** (12.1 → 4.2 m range, 16.4 → 3.0 mm/s
     Doppler). In prediction, radial and along-track agree with the LOLA orbit to 2.8 / 8.0 m.
   - By the measure that matters operationally (prediction of unseen tracking), F2g is the best model so far.
2. **Over the fit arc, along-track and cross-track against LOLA degrade** (22/25 → 60/78 m), and K_SRP doubles.
   - The along- and cross-track empirical terms give the orbit freedom to rotate about the Earth line of sight,
     the direction Earth-based range and Doppler do not observe (F2d). The data cost of that rotation is near
     zero, so the fit moves along it.
   - K_SRP and S_T are correlated; both describe along-track energy change.
   - Better data fit is therefore not better geometry here. The cross-track error in prediction (62 m) shows the
     rotation persists.
3. **Consequence.** Empirical accelerations need a constraint that this tracking geometry cannot provide by itself:
   - a tighter prior on N (cross-track), or estimating along-track terms only;
   - or a second observable direction (ΔDOR, optical, or LOLA altimetry crossovers, as the reference orbit used).
4. **S2** again ends in a wrong basin. The one-day stage reaches 7.2 m and the two-day stage 114 m. Its empirical
   terms run large (C_N, S_N ~ ±2.9e-7) and its RIC errors reach 265 m. Basin robustness remains the open
   estimator problem.

### Next steps

1. An along-track-only empirical model (C_T, S_T), with N held at zero or under a tight prior: tests whether the
   prediction gain survives without the line-of-sight rotation.
2. Multi-start / globalisation for the S2 basin.
3. Phase E (owner approval): production empirical accelerations and the numba harmonic STM.

## 14. F2h — along-track-only empirical once-per-revolution accelerations (N = 300)

- **Predeclaration:** `studies/phase17/configs/phase_f2h_lro_alongtrack_empirical_predeclaration.json` (`07280b2`).
- **Evidence:** `studies/phase17/results/phase_f2h_lro_alongtrack_empirical.json`.
- **Change from F2g:** C_N and S_N are pinned by a 1e-11 m/s² prior. C_T and S_T stay free under the 1e-8 m/s²
  prior. Run for S1 only.
- **Run history:**
  - A first launch never reached its own entry point (shell working-directory slip) and briefly started the default
    F2 runs. It was stopped at once; no output was written.
  - A launch with 14 workers failed with MemoryError while the pool was starting. The commit charge was 30.6 / 48 GB
    from other applications. Its log is kept in `lro_data/f2h_memoryerror_14workers.txt`.
  - The run reported here used 10 workers.

### Gates and hypotheses as computed

| item | result | number |
|---|---|---|
| G1 convergence | **FAIL** | DampingLimit, stationary at 0.984 m (last step 4e-12 m) |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **PASS** | 0.984 m / 2.43 mm/s |
| H1 along / cross vs LOLA ≤ F2f (21.8 / 25.4 m) | partly | along-track 20.4 m (met), cross-track 27.9 m (not met) |
| H2 one-day prediction ≤ 6 m | **met** | 1.63 m |
| H3 K_SRP toward ~0.014 | not met | 0.0742 ± 0.0002 |

### Results

| | F2f (N=300) | F2g (+4 empirical) | **F2h (+along-track empirical)** |
|---|---|---|---|
| range / Doppler post-fit | 1.506 m / 3.82 mm/s | 1.229 m / 2.52 mm/s | **0.984 m / 2.43 mm/s** |
| per-pass range RMS | 0.85–2.08 m | 0.53–2.06 m | 0.50–1.67 m |
| station delays (vs F1 489.2 / 476.9 / 474.5) | 488.4 / 475.6 / 474.6 | 487.5 / 475.1 / 475.0 | 486.3 / 473.8 / 472.7 |
| empirical [m/s²] | — | C_T −1.2e-8, S_T 2.71e-7, C_N 0.9e-8, S_N 4.4e-8 | C_T −1.94e-8 ± 4e-10, **S_T 2.44e-7 ± 1.2e-9**, C_N, S_N ≈ 0 |
| vs LOLA R / A / C RMS (fit arc) | 3.5 / 21.8 / 25.4 m | 1.6 / 60.2 / 78.2 m | **1.7 / 20.4 / 27.9 m** |
| total vs LOLA RMS / max | — | 98.6 / 161.8 m | **34.6 / 70.1 m** |
| **prediction range RMS / mean** (2009-262) | 12.1 / −1.6 m | 4.2 / −2.3 m | **1.63 / +0.29 m** |
| prediction Doppler RMS / mean | 16.4 mm/s | 3.0 mm/s | **1.81 / +0.35 mm/s** |
| prediction vs LOLA R / A / C | 12.7 / 39.1 / 9.4 m | 2.8 / 8.0 / 62.1 m | **2.3 / 6.6 / 10.5 m** |

Prediction range RMS per pass, 14.3–23.7 h after the fit end: 3.15 (South Point), 0.89, 0.56, 1.23, 1.03 and
1.61 m.

### Reading

1. **The along-track once-per-rev signal is real, and the cross-track terms were the problem.**
   - With only C_T and S_T, the fit improves to the noise floor (0.98 m / 2.4 mm/s).
   - The orbit goes back to F2f's geometry against LOLA (along-track 20 m, cross-track 28 m), so F2g's
     line-of-sight rotation is gone.
   - The estimated S_T = 2.44e-7 m/s² is stable between F2g and F2h (2.71e-7).
2. **One-day prediction against unseen tracking: 1.6 m range, 1.8 mm/s Doppler.**
   - That is the measurement noise level.
   - Against the LOLA precision orbit the prediction agrees to 2.3 / 6.6 / 10.5 m (R / A / C).
   - This is the first result of this project at the level of operational lunar OD with S-band ground tracking.
3. **K_SRP stays at 0.074.** The along-track empirical term and K both describe along-track energy change. The
   remaining along-track signal is some combination of:
   - SRP with LRO's true geometry and attitude;
   - thermal and albedo radiation;
   - desaturation effects.
   K at 0.074 should be read as an effective parameter, not a physical area-to-mass ratio.
4. **What is still open:**
   - G1: the damped Gauss–Newton ends at the damping limit, not with a declared convergence;
   - the basin robustness of other starts (F2e–F2g S2);
   - the cross-track error over the fit arc (28 m), which is the remaining line-of-sight weakness of Earth tracking.

### Next steps

1. Basin robustness (multi-start screening, or trust-region on scaled parameters), so that G1/G2 can pass.
2. Longer arcs (3–5 days) with F2h's model and a 1–2 day prediction test.
3. A box-wing SRP model instead of cannonball plus empirical terms (needs LRO geometry and attitude).
4. Phase E (owner approval): production empirical accelerations and the numba harmonic STM.

## 15. F2i — basin robustness (step limit, Gauss–Newton stationarity test; F2h model)

- **Predeclaration:** `studies/phase17/configs/phase_f2i_lro_basin_robustness_predeclaration.json` (`45e14be`).
- **Evidence:** `studies/phase17/results/phase_f2i_lro_basin_robustness.json`, merged from `phase_f2i_part_s1r.json`
  and `phase_f2i_part_s2.json`.
- **Changes from F2h:**
  - each LM step is capped at 200 m / 0.2 m/s;
  - a run that reaches the damping limit is STATIONARY when the undamped Gauss–Newton decrement g^T A^-1 g < 0.01
    (χ² units);
  - S2 starts on the two-day arc directly, with no one-day stage;
  - two processes with 5 workers each.

### Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence | **FAIL** | both DampingLimit: Gauss–Newton decrement 10.09 (S1R) and 0.46 (S2) > 0.01 |
| **G2 start independence (≤ 1 m)** | **PASS** | **0.991 m** (first pass of this gate in the F2 series) |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **PASS** | S1R 0.984 m / 2.43 mm/s; S2 0.982 m / 2.43 mm/s |

### The two solutions

| | S1R (F2h start) | S2 (+100 m, +0.1 m/s) |
|---|---|---|
| iterations | 11 | 28 |
| range / Doppler post-fit | 0.984 m / 2.426 mm/s | 0.982 m / 2.432 mm/s |
| station biases 103 / 105 / 126 | 486.340 / 473.805 / 472.673 m | 486.329 / 473.803 / 472.644 m |
| K_SRP | 0.07424 | 0.07447 |
| C_T, S_T [m/s²] | −1.939e-8, 2.444e-7 | −1.936e-8, 2.441e-7 |
| vs LOLA R / A / C RMS | 1.67 / 20.44 / 27.90 m | 1.64 / 20.05 / 27.36 m |

S1R reproduces F2h exactly (same start, same model). Its one-day prediction is 1.63 m / 1.81 mm/s, as in F2h.

### Reading

1. **The step cap was never active.** The largest position steps were 41.6 m (S1R) and 98.2 m (S2), all below
   200 m. The basin fix in F2i therefore does not come from step limiting.
2. **What changed for S2 is the model.**
   - F2e–F2g S2 used the full or no empirical set and a one-day continuation stage, and ended at 7–150 m.
   - F2i S2 uses F2h's along-track-only empirical model directly on the two-day arc.
   - It descended through a slow plateau near 3.15 m (iterations 8–15) and then reached the same minimum as S1R
     within 0.99 m.
   - Removing the cross-track empirical freedom shrinks the null space: the cost surface has one dominant minimum
     at this offset, and a damped Gauss–Newton path finds it.
3. **G1:** the stationarity test as declared does not pass.
   - At both end points the undamped Gauss–Newton step predicts a decrease (10.1 and 0.46 χ²) that no damped trial
     step realises.
   - The total χ² is ~1.0e4, so these are 1e-3 and 5e-5 of the cost. That is inside the noise of a cost evaluated
     at rtol 1e-10 with FD sensitivities, the same situation production's HARDEN-D30 met and handled by rescoring
     at a tighter tolerance.
   - The 0.01 threshold was too strict for this cost noise. A relative threshold (decrement < 1e-3 of χ²), or a
     tight-tolerance rescore, would classify both runs as stationary. Proposed as an owner adjudication, not
     applied.

### Status of the F2 series

- **Best real-data LRO solution so far:** F2h/F2i.
  - two-day range + Doppler, 0.98 m / 2.4 mm/s post-fit;
  - the solution is independent of a 100 m / 0.1 m/s start offset (G2);
  - one-day prediction 1.6 m / 1.8 mm/s;
  - 1.7 / 20 / 28 m vs the LOLA orbit.
- **Open:**
  - G1 classification (cost noise);
  - the cross-track error (~28 m) against LOLA;
  - K_SRP as an effective rather than physical parameter.

## 16. F2j — three-day arc (2009-260..262) with a one-day prediction test (2009-263; F2h model)

- **Predeclaration:** `studies/phase17/configs/phase_f2j_lro_three_day_arc_predeclaration.json` (`9cc3873`).
- **Evidence:** `studies/phase17/results/phase_f2j_lro_three_day_arc.json`.
- **Diagnostic (not a gate):** `studies/phase17/results/phase_f2j_day263_lola_residuals.json`. This is the F1 slope
  diagnostic (`phase_f1_slope_diagnostic.py`) run on 2009-263: real range minus range computed from the LOLA orbit.
- **Model:** F2h's.
  - N = 300.
  - Along-track empirical C_T and S_T; C_N and S_N pinned at 1e-11 m/s².
  - `F2H_PRIORS`.
  - Start from F2h's S1 state. K and the empirical terms start at 0.01 and 0.
- **Termination:** the relative rule. At the damping limit a run is STATIONARY when the Gauss–Newton decrement is
  below 1e-3·χ².
- **Run:**
  - 8 workers, 18 iterations.
  - Elapsed 36 835 s. This includes about 3 h when the laptop was in Modern Standby (lid closed, 21:20–00:25).
    The run is deterministic, so the pause does not change any number.

### Gates as computed

| gate | result | number |
|---|---|---|
| G1 convergence | **FAIL** | DampingLimit at iteration 18; Gauss–Newton decrement 977.4 = 0.057·χ² (χ² = 17 154) > 1e-3·χ² |
| G3 post-fit (range ≤ 2 m, Doppler ≤ 5 mm/s) | **PASS** | 1.086 m / 2.17 mm/s |

### Results

| | F2h (two-day arc 260–261) | **F2j (three-day arc 260–262)** |
|---|---|---|
| range / Doppler post-fit | 0.984 m / 2.43 mm/s | **1.086 m / 2.17 mm/s** |
| per-pass range RMS | 0.50–1.67 m | 0.42–1.60 m (16 passes) |
| station delays 103 / 105 / 126 (F1: 489.2 / 476.9 / 474.5) | 486.3 / 473.8 / 472.7 m | 486.1 / 475.2 / 472.0 m (σ 0.07 m) |
| global Doppler bias | −8.03 mm/s | −7.99 ± 0.03 mm/s |
| K_SRP | 0.0742 | 0.0756 ± 0.00005 |
| C_T, S_T [m/s²] | −1.94e-8, 2.44e-7 | −1.92e-8 ± 1.4e-10, **2.56e-7 ± 5.5e-10** |
| vs LOLA R / A / C RMS (fit arc) | 1.7 / 20.4 / 27.9 m | **1.6 / 16.2 / 23.2 m** |
| vs LOLA total RMS / max (fit arc) | 34.6 / 70.1 m | 28.3 / 70.4 m |
| prediction day | 2009-262 | 2009-263 |
| prediction range RMS / mean | 1.63 / +0.29 m | **7.12 / +0.06 m** |
| prediction Doppler RMS / mean | 1.81 / +0.35 mm/s | **24.4 / −6.25 mm/s** |
| prediction vs LOLA R / A / C RMS | 2.3 / 6.6 / 10.5 m | 2.3 / 5.8 / 10.4 m (max 3.5 / 13.2 / 15.5 m) |

Prediction range per pass on 2009-263:

| PAD | n | hours after fit end | mean | RMS |
|---|---|---|---|---|
| 103 | 451 | 14.3 | +0.28 m | 4.79 m |
| 103 | 435 | 16.2 | −0.64 m | 5.51 m |
| 103 | 451 | 18.1 | −0.02 m | 6.51 m |
| 103 | 1 | 20.3 | +0.36 m | — |
| 103 | 1 | 20.8 | −11.79 m | — |
| 103 | 225 | 21.8 | +4.42 m | 6.67 m |
| 126 | 444 | 23.7 | −1.56 m | 10.52 m |

Diagnostic: the 2009-263 tracking against the LOLA orbit, after a per-pass bias and slope, leaves 0.33–0.92 m
RMS. The slopes are −13 to −17 m/h, the same along-track signature F1/F1b found in LOLA on other days. The
prediction-day data are therefore as clean as the fit-arc data.

### Reading

1. **The fit-arc solution improves with the third day.**
   - Post-fit is at the noise floor (1.09 m / 2.2 mm/s).
   - Along-track and cross-track against LOLA improve from 20 / 28 m to 16 / 23 m.
   - The station delays, the Doppler bias and C_T agree with F2h within ~1.4 m, 0.04 mm/s and 1 %.
   - S_T rises by 5 % (2.44e-7 → 2.56e-7 m/s²), many times its formal σ. A single constant once-per-rev amplitude
     is not quite the same over two and three days.
2. **The one-day prediction is worse than F2h's in the tracking data, but not against LOLA.**
   - Range is 7.1 m RMS and Doppler 24 mm/s, against F2h's 1.6 m / 1.8 mm/s.
   - The pass means stay near zero (−1.6 … +4.4 m). The error is a within-pass shape that grows with time from
     4.8 m to 10.5 m RMS.
   - The data are not the cause: against LOLA the same passes are clean to 0.3–0.9 m.
   - Against LOLA the predicted orbit is as good as F2h's (2.3 / 5.8 / 10.4 m). LOLA carries its own along-track
     error (F1b, and the −13 to −17 m/h slopes above), so a metre-level comparison with it does not settle which
     orbit is closer to the truth on 2009-263.
   - The Doppler prediction mean of −6.25 mm/s on top of the fitted global bias suggests that the bias, or the
     orbit's velocity, differs on 2009-263. This is not resolved here.
   - F2h predicted a different day (2009-262), so the two prediction rows are not a like-for-like comparison.
3. **G1 fails, and not because of the threshold.**
   - Iteration 17 still took a 27 m step at λ = 1e-5.
   - At iteration 18 no damped trial up to λ = 1e10 lowered the cost, although the undamped Gauss–Newton step
     predicts a 5.7 % decrease.
   - F2i's decrements were 1e-3 and 5e-5 of χ², inside the cost noise. 5.7 % is not cost noise.
   - A predicted decrease that no small step realises points to a Jacobian that is wrong in some direction near
     this point, or to a cost that is not smooth there: for example an observation or a shadow crossing that
     switches with the state. The FD sensitivities (A1) are the first suspect. This is not diagnosed here.
4. **What this means for the F2 series.**
   - F2h/F2i (two-day arc) stays the best predictor: 1.6 m / 1.8 mm/s one day ahead.
   - Three days of tracking give a better fit-arc orbit against LOLA, but the constant once-per-rev along-track
     model does not carry the prediction as well. The remaining force is not constant over three days, or not
     once-per-rev.

### Next steps

1. Diagnose G1 at the F2j end point: compare the FD Jacobian with a tighter-step or analytic one along the
   Gauss–Newton direction, and scan the cost along that direction.
2. Piecewise-constant (per-day) along-track empirical terms over the three-day arc, then the same 2009-263
   prediction test.
3. Check the Doppler bias on 2009-263 separately: a per-day bias fit against the LOLA orbit, as in F1c.
4. Phase E (owner approval): the analytic numba harmonic STM, which removes the FD Jacobian; it is expected to be several
   times faster per iteration (not measured).

### 16.1 F2j-G1 — why the fit stopped: integrator jitter at rtol 1e-10

- **Predeclaration:** `studies/phase17/configs/phase_f2j_g1_diagnostic_predeclaration.json`.
  - D0–D3 predeclared in `5de0195`.
  - Amendment A1 (D4) in `b3e3a77`, written after the D0–D2 numbers.
  - Amendment A2 (D5) in `9e53a0b`, written after D4.
- **Evidence:**
  - `studies/phase17/results/phase_f2j_g1_diagnostic.json` (D0–D3; script `phase_f2j_g1_diagnostic.py`);
  - `studies/phase17/results/phase_f2j_g1_rowdiff.json` (D4; script `phase_f2j_g1_rowdiff.py`);
  - `studies/phase17/results/phase_f2j_g1_propagation_jitter.json` (D5; script
    `phase_f2j_g1_propagation_jitter.py`).
- No production code changed. F2j's result and its G1 FAIL stand.

#### D0–D3: cost scan and an independent FD Jacobian

- **D0** reproduces F2j's end point exactly: χ² = 17153.6527, Gauss–Newton decrement 977.41. The Gauss–Newton
  step is 22.5 m in position.
- **D1/D2:** actual cost against the quadratic model.

| step | position step | actual χ² | model χ² |
|---|---|---|---|
| GN α = −0.01 | 0.22 m | 17 305.9 | 17 173.3 |
| GN α = +0.01 | 0.22 m | **17 097.5** | 17 134.2 |
| GN α = 0.1 | 2.2 m | 18 382.8 | 16 967.9 |
| GN α = 0.3 | 6.7 m | 20 125.4 | 16 655.2 |
| GN α = 1 | 22.5 m | 18 551.2 | 16 176.2 |
| LM λ = 1e-5 | 0.18 m | 17 993.9 | 16 977.9 |
| LM λ = 1e-2 | 1.0 mm | 17 789.8 | 17 066.9 |
| LM λ = 1 | 0.12 mm | 18 388.7 | 17 118.6 |
| LM λ = 1e2 | 9 µm | 18 066.8 | 17 148.5 |
| LM λ = 1e4 | 0.1 µm | 18 281.4 | 17 153.6 |

  - Steps of 0.1 µm to 1 mm raise the cost by 600–1200 χ².
  - The central slope at α = ±0.01 is −10 420 per unit α. The model gives −1 955.
- **D3:** residual-level central differences of the 11 dynamical columns.
  - At the A1 step h, they agree with the solver's columns to 1e-7…4e-4. This is not an independent check: the
    solver's columns come from the same ±h propagations.
  - At h/4, the state columns agree to 1e-5…8e-4. The K and empirical columns differ by 7–31 %.
  - The gradient differs by 2e-4 and the decrement is 1032.
  - The Gauss–Newton steps from both FD Jacobians also raise the actual cost: 17 723 / 18 060 (h, α = 1 / 0.1) and
    18 027 / 17 829 (h/4).

#### D4: which rows carry the jump

| case | χ² | Δχ² | range Δr RMS | max \|Δr\| | rows with \|Δr\| > 1 cm | share of top 25 rows |
|---|---|---|---|---|---|---|
| repeat at the end point | 17 153.6527 | 0 | 0 | 0 | 0 | — |
| x0[0] + 1 µm | 17 633.5 | +479.8 | 0.28 m | 0.75 m | 8 699 | 14 % |
| 0.01 × GN step | 17 097.5 | −56.2 | 0.09 m | 0.22 m | 8 600 | 31 % |

The evaluation is deterministic. A 1 µm change of the start state moves the range residuals by 0.28 m RMS, spread
over all passes and stations. That is not a measurement-model discontinuity; it is the propagated orbit.

#### D5: the propagated orbit itself

All cases use F2g's empirical propagator at the F2j end point and compare positions on the 20 s grid.

| comparison | position difference RMS / max |
|---|---|
| x0[0] + 1 µm, rtol 1e-10 | **0.35 / 0.75 m** |
| the same, with the base's shadow cuts frozen | 0.16 / 0.35 m |
| x0[0] + 1 µm, rtol / atol 1e-12 / 1e-13 | **0.37 / 0.79 mm** |
| rtol 1e-10 against rtol 1e-12, same start | **0.68 / 1.39 m** |

- All cases find 128 shadow contacts.
- A 1 µm start change shifts a contact time by up to 0.46 ms, far more than the ~1e-9 s the geometry implies. The
  contact pre-pass carries its own integration jitter.
- At rtol 1e-12, the 1 µm change moves the orbit by under 1 mm. That is the size linear dynamics allows over three
  days.

#### Reading

1. **By the predeclared D0–D3 rules: UNEXPLAINED, as computed.**
   - H_J is not supported: both FD Jacobians match the solver's gradient to 2e-4, far inside 10 %.
   - H_Q is not supported: the small-α slope is not within 20 % of the model.
   - H_N's first condition holds (slope off by a factor of 5). Its second condition, FD gradients at h and h/4
     disagreeing by more than 10 %, does not hold, because the gradient is dominated by the state columns and their
     steps are large against the jitter. That condition was the wrong test.
2. **D4 and D5 (amendments) identify the mechanism: integrator jitter at rtol 1e-10.**
   - At F2's integration tolerance, the three-day orbit is a discontinuous function of the parameters at the
     0.3–0.8 m level.
   - About half of it comes from the shadow-segment restarts: the cut times jitter and the integrator restarts at
     shifted epochs. Freezing the cuts halves it.
   - The rest comes from the adaptive step control.
   - At rtol 1e-12 it falls by a factor of ~1000.
3. **Consequence for F2j G1.**
   - Residual jitter of ~0.3 m over ~14 000 rows (6 989 range, 6 851 Doppler) makes the cost noisy at the level of several hundred χ².
   - A Gauss–Newton decrement of 977 is inside that noise.
   - The LM line search accepted points where the jitter happened to be favourable, and ended at one that no nearby
     point beats. G1's damping-limit end is a cost-noise floor, not a Jacobian error and not a true minimum.
   - F2i's two-day decrements (1e-3 and 5e-5 of χ²) are the same effect, smaller on a shorter arc.
4. **A second consequence: rtol 1e-10 is a metre-level orbit error on a three-day arc.**
   - Against rtol 1e-12, the rtol 1e-10 orbit differs by 0.68 m RMS and 1.39 m at most.
   - That is comparable to the 1.09 m range post-fit.
   - It is a candidate contributor to the F2j fit and prediction numbers. How much of the 7.1 m prediction it
     explains is not measured here.
5. **Cost of the fix:** all five D5 propagations, the two rtol 1e-12 ones included, ran in parallel in about 7
   minutes. A tighter tolerance looks affordable, but a per-propagation timing was not recorded.

#### Next steps

1. Re-run F2j at rtol / atol 1e-12 / 1e-13 (new predeclaration), with the same start and gates. Expect G1 to be
   decidable and the fit and prediction numbers to move at the metre level.
2. Rescore F2h's two-day solution at the tight tolerance. This shows how much the best predictor so far moves.
3. Then the §16 next steps: per-day empirical terms and the 2009-263 Doppler bias.


## 17. F2k — tight integrator tolerance (rtol / atol 1e-12 / 1e-13)

Predeclaration: `studies/phase17/configs/phase_f2k_tight_tolerance_predeclaration.json` (`87f7ef3`), written
after §16.1 and before any F2k number. The only change is the integrator tolerance of every propagation, in the
main process and in every worker (`set_tolerance`, passed through the worker initializer). With the default
tolerance the code path is bitwise F2j's. Two parts: R rescores the stored F2h and F2j end points without
refitting; K re-runs F2j's fit.

### 17.1 R — F2h and F2j end points rescored

Script `studies/phase17/scripts/phase_f2k_tight_tolerance.py`, result `studies/phase17/results/phase_f2k_rescore.json`
(`7118cf2`). At 1e-10 / 1e-11 the rescoring reproduces the stored post-fit and prediction RMS of both points
exactly (relative difference 0.0), so the comparison below is a pure tolerance change at a fixed parameter vector.

| point | tolerance | χ² | range post-fit (m) | Doppler post-fit (mm/s) | RIC fit (m) | prediction range RMS / mean (m) | prediction Doppler RMS / mean (mm/s) | RIC prediction (m) |
|---|---|---|---|---|---|---|---|---|
| F2h (fit 260–261, predict 262) | 1e-10 | 11002.28 | 0.9840 | 2.426 | 1.67 / 20.44 / 27.90 | 1.632 / 0.29 | 1.811 / 0.35 | 2.33 / 6.59 / 10.52 |
| | 1e-12 | 11018.87 | 0.9841 | 2.429 | 1.67 / 20.45 / 27.90 | 1.633 / 0.29 | 1.812 / 0.36 | 2.33 / 6.59 / 10.52 |
| F2j (fit 260–262, predict 263) | 1e-10 | 17153.65 | 1.0856 | 2.172 | 1.57 / 16.15 / 23.18 | 7.125 / 0.07 | 24.39 / −6.25 | 2.26 / 5.83 / 10.39 |
| | 1e-12 | 19177.04 | 1.2094 | 2.177 | 1.57 / 16.15 / 23.18 | 8.188 / 0.26 | 24.76 / −7.39 | 2.25 / 6.91 / 10.39 |

F2j prediction per pass on 2009-263, range RMS (m), 1e-10 → 1e-12: PAD 103 at 14.3 h 4.79 → 5.90, 16.2 h
5.51 → 6.61, 18.1 h 6.51 → 7.66, 20.3 h 0.36 → 0.68, 20.8 h 11.79 → 13.40 (mean −11.79 → −13.40), 21.8 h
6.67 → 7.68; PAD 126 at 23.7 h 10.52 → 11.62. F2h's per-pass numbers on 262 are unchanged to the printed
digits.

#### Reading

1. **F2h is tolerance-proof.** On the two-day arc the tight tolerance moves χ² by +16.6 (0.15 %) and the
   prediction by under 1 mm. The best predictor so far stands as reported.
2. **F2j's end point is tuned to the loose tolerance's noise.** At its own parameter vector the tight tolerance
   adds 2023 χ² (+12 %) and 0.12 m of range post-fit. This is what §16.1 predicts: the LM search at 1e-10 settled
   where the jitter happened to be favourable, and that point is not a minimum of the accurate model.
3. **F2j's 7.1 m prediction is not an integration artefact.** Under the tight tolerance at the same parameters it
   gets worse (8.19 m), not better; the change, 1.06 m, is about 15 % of its value. By the predeclared reading the
   prediction failure belongs to the model or the data, not to the integrator. Whether a fit made at the tight
   tolerance predicts better is the question part K answers.
4. Fit-span RIC against LOLA does not move at either point (≤ 0.01 m), so the integration error that changes the
   residuals is far below the decametre along-track / cross-track offsets to LOLA.
5. Wall time of each case was 4.4–5.0 h. The computer spent most of that span in Modern Standby, so it is not a
   measure of the tolerance cost.

### 17.2 K — F2j's fit re-run at rtol / atol 1e-12 / 1e-13

Result `studies/phase17/results/phase_f2k_lro_three_day_arc_tight_tol.json`. Same data, model, priors, weights, start
and relative Gauss–Newton rule as F2j (§16); only the integrator tolerance differs.

**Run history (amendment A1, `eaabed0`).** The first K run died in iteration 10 on 2026-10-27 21:40: a pool worker
was killed when the laptop, just unplugged, was put to sleep (`BrokenProcessPool`, no result). The restart added an
exact-resume checkpoint and ran outside the session's process tree. Its iterations 1–9 reproduce the dead run's
logged iterations 1–9 exactly (range RMS, K, step, λ), so the evaluation is deterministic, as A1 required. The
checkpoint was not needed: the restart ran to the end in one process. Its 8.6 h wall time includes about 2 h of
Modern Standby and is not a measure of the tolerance cost.

#### Gates as computed

| gate | rule | value | outcome |
|---|---|---|---|
| G1 | Converged or STATIONARY (decrement < 1e-3 χ²) | STATIONARY at iteration 12; Gauss–Newton decrement 9.4e-4 χ² (5.8e-8 of χ²) | **PASS** |
| G3 | range ≤ 2 m, Doppler ≤ 5 mm/s | 1.030 m, 2.143 mm/s | **PASS** |

#### Results against F2j

| quantity | F2j (rtol 1e-10) | F2k (rtol 1e-12) |
|---|---|---|
| end | DampingLimit, iteration 18, decrement 977 χ² | STATIONARY, iteration 12, decrement 9.4e-4 χ² |
| χ² (own tolerance) | 17153.65 (19177.04 rescored at 1e-12, §17.1) | 16110.91 |
| range / Doppler post-fit | 1.086 m / 2.172 mm/s | 1.030 m / 2.143 mm/s |
| K_SRP (m²/kg) | 0.07558 | 0.07451 (σ 1.1e-4) |
| C_T / S_T (m/s²) | −1.918e-8 / 2.562e-7 | −1.908e-8 / 2.548e-7 |
| range biases 103 / 105 / 126 (m) | 486.08 / 475.22 / 472.01 | 485.08 / 474.19 / 471.03 |
| Doppler bias (mm/s) | −7.99 | −7.97 |
| RIC vs LOLA, fit arc, RMS (m) | 1.57 / 16.15 / 23.18 | 2.01 / 9.92 / 12.26 |
| RIC vs LOLA, fit arc, max (m) | 3.75 / 45.04 / 54.09 | 4.05 / 30.64 / 35.89 |
| prediction 263, range RMS / mean (m) | 7.12 / 0.07 | 5.21 / 1.31 |
| prediction 263, Doppler RMS / mean (mm/s) | 24.39 / −6.25 | 23.81 / −4.25 |
| RIC vs LOLA, 263, RMS (m) | 2.26 / 5.83 / 10.39 | 1.88 / 5.05 / 2.90 |
| RIC vs LOLA, 263, max (m) | 3.45 / 13.23 / 15.49 | 2.77 / 11.35 / 4.67 |

The epoch state moved by 11.6 / 16.0 / 11.0 m and 4 / 11 / −10 mm/s from F2j's.

Prediction per pass on 263, range RMS / mean (m), F2k (F2j in brackets): PAD 103 at 14.3 h 3.53 / 1.44 (4.79),
16.2 h 3.85 / 0.76 (5.51), 18.1 h 4.77 / 1.31 (6.51), 20.3 h 1.36 / 1.36 (0.36), 20.8 h 6.96 / −6.96 (11.79),
21.8 h 5.89 / 4.60 (6.67); PAD 126 at 23.7 h 7.41 / 0.06 (10.52).

#### Reading

1. **F2j's G1 failure was the integrator.** With the only change being rtol 1e-10 → 1e-12, the same fit ends
   STATIONARY with a decrement six orders of magnitude below the threshold, in 12 iterations instead of 18. The
   §16.1 diagnosis (cost noise from integrator jitter) is confirmed by the outcome it predicted.
2. **The accurate fit is a different, better solution.** χ² is 1043 lower than F2j's own and 3066 lower than F2j's
   point rescored at the same tolerance. Against LOLA the along-track and cross-track errors on the fit arc fall
   from 16.2 / 23.2 m to 9.9 / 12.3 m RMS; radial rises from 1.6 to 2.0 m. This is the best fit-arc agreement with
   LOLA in the F2 series (F2h: 1.7 / 20.4 / 27.9 m).
3. **The prediction improves against LOLA, much less against the data.** On 263 the orbit is 1.9 / 5.1 / 2.9 m
   from LOLA (cross-track 10.4 → 2.9 m), better than F2h's own next-day prediction against LOLA (2.3 / 6.6 /
   10.5 m). Against the 263 measurements the range RMS falls from 7.1 to 5.2 m, but the Doppler RMS stays at
   24 mm/s.
4. **The 263 Doppler misfit does not look like an orbit error.** An orbit within a few metres of LOLA has velocity
   errors of the order of mm/s, not 24 mm/s. Together with §16's day-263 range diagnostic (per-pass slopes of
   −13 to −17 m/h against LOLA, or equivalently a ~3 ms timing offset), this points at the day-263 data or at a
   day-specific measurement-model term rather than at the dynamics. Not tested here: scoring the LOLA orbit itself
   against the 263 Doppler would settle it.
5. **Consequence for the F2 series.** The rtol 1e-10 numbers of F2j are superseded by F2k's. F2h's are not
   (§17.1: unchanged at 1e-12). Any further multi-day fit uses rtol ≤ 1e-12.

#### Next steps

1. Score the LOLA orbit against the 263 range and Doppler with the same measurement model (no fit). If LOLA also
   gives ~24 mm/s, the prediction test on 263 measures the data, and the LOLA comparison is the prediction metric
   for that day.
2. Per-day empirical terms on the three-day arc at rtol 1e-12, with the same 263 test.
3. Phase E (owner approval): analytic harmonic STM. At rtol 1e-12 one F2k iteration took 20–28 min of wall time
   with the CPU to itself.

### 17.3 F2l — the LOLA orbit scored against the 260–263 data (no fit)

Predeclaration `studies/phase17/configs/phase_f2l_lola_orbit_scoring_predeclaration.json` (`501fe63`), written
before any number. Script `studies/phase17/scripts/phase_f2l_lola_scoring.py`, result
`studies/phase17/results/phase_f2l_lola_scoring.json`. The LRO reference ephemeris used as "LOLA" in every RIC
comparison is sampled on the 20 s grid and passed through `predict()`'s measurement code unchanged (light time,
media, counted Doppler, F2e A1 bound), with F2k's range and Doppler biases.

**Reproduction check: PASS.** F2k's orbit through the same script reproduces F2k's stored 263 prediction RMS to a
relative 1.1e-16.

| day | LOLA range RMS / mean / about mean (m) | LOLA Doppler RMS / mean / about mean (mm/s) |
|---|---|---|
| 260 (fitted by F2k) | 4.83 / 3.80 / 2.98 | 4.03 / −3.22 / 2.43 |
| 261 (fitted) | 4.59 / 3.05 / 3.44 | 4.53 / −3.24 / 3.16 |
| 262 (fitted) | 5.37 / 3.87 / 3.73 | 4.20 / −3.55 / 2.23 |
| 263 (predicted) | 5.02 / 3.65 / 3.45 | **23.52** / −3.01 / 23.33 |

**Verdict by the predeclared rule: H_DATA.** LOLA's 263 Doppler RMS is 23.5 mm/s (≥ 10) while each of 260–262 is
≤ 4.5 mm/s (≤ 5). The reference orbit misfits the 263 Doppler exactly as much as F2k's (23.8 mm/s), so F2k's 263
Doppler misfit is not an orbit error.

Doppler per pass on 263, RMS (mm/s), LOLA / F2k:

| pass (PAD, UTC 2009-09-20) | n | LOLA | F2k |
|---|---|---|---|
| 103, 03:18–04:08 | 440 | 4.15 | 3.47 |
| 103, 05:13–06:01 | 425 | 3.73 | 3.88 |
| 103, 07:04–07:54 | 439 | 3.85 | 4.66 |
| **103, 10:50–11:20** | 220 | **69.3** (mean −0.1) | **69.5** |
| 126, 12:44–13:33 | 434 | 3.97 | 7.77 |

#### Reading

1. **One pass carries the whole 263 Doppler misfit.** PAD 103's 10:50–11:20 pass scatters by 69 mm/s about a
   near-zero mean against both orbits; every other 263 pass is at the fit days' 4 mm/s level. Its UTDF headers
   (mode 9830, flags 11, PAD 103 transmit and receive) match the clean passes; it is short (30 min against ~50)
   and has a 470 s gap. The PAD 103 pass just before it (08:57–09:50) is the only one with flags 9 instead of 11;
   its Doppler is already removed by the F2e A1 physical bound and it gives only two range points. Why the 10:50
   pass is bad is not determined here.
2. **Without that pass (post hoc, not predeclared)** the 263 Doppler RMS is 3.9 mm/s for LOLA and 5.2 mm/s for F2k.
   The remaining F2k excess is the last pass (PAD 126, 23.7 h after the fit end: 7.8 against LOLA's 4.0 mm/s), a
   genuine prediction degradation at the end of the day.
3. **Range on 263 is also at the reference orbit's level.** With F2k's biases LOLA gives 4.6–5.4 m RMS on every
   day, fitted or not, so F2k's 5.2 m range prediction on 263 equals what the reference orbit itself achieves in
   our measurement model. The day-to-day constant ~+3.5 m LOLA range mean says the biases were estimated with our
   orbit, not LOLA's; the ~3 m scatter about the mean is the reference orbit's level in our model.
4. **Consequence for F2j / F2k.** The "7.1 m / 24 mm/s" prediction failure of §16 was mostly one bad Doppler pass
   plus a range level that even the reference orbit does not beat. Against LOLA, F2k's 263 prediction
   (1.9 / 5.1 / 2.9 m) is the best in the series. The constant once-per-rev model is not shown to fail over three
   days.

#### Next steps

1. Exclude PAD 103 2009-263 10:50–11:20 Doppler from prediction scoring by a predeclared data-screening rule
   (for example a per-pass Doppler scatter bound, applied to every day), not by hand.
2. Look at the bad pass's residual time series and raw counts (count-rate continuity, the 470 s gap) to find
   the cause.
3. Per-day empirical terms remain an option, but the case for them made in §16 is weakened by this section.

### 17.4 F2m — per-pass Doppler screening rule

Predeclaration `studies/phase17/configs/phase_f2m_doppler_pass_screening_predeclaration.json` (`4aa562f`), written
after F2l with its per-pass numbers known (disclosed there). Rule: reject a Doppler pass whose residual RMS about
the pass mean, against F2k's own orbit, exceeds 5 × the median of that statistic over all passes of 260–263. The
same rule against the LOLA orbit is the cross-check. Script `studies/phase17/scripts/phase_f2m_doppler_screening.py`,
result `studies/phase17/results/phase_f2m_doppler_screening.json`.

| | F2k orbit (applied) | LOLA orbit (cross-check) |
|---|---|---|
| median per-pass scatter | 1.91 mm/s | 2.01 mm/s |
| threshold (5 × median) | 9.57 mm/s | 10.05 mm/s |
| rejected | PAD 103, 2009-263 10:50–11:20, n = 220, 69.4 mm/s | the same pass, 69.3 mm/s |

The two rejected sets agree. **No fit-day pass is rejected**, so by the consequence declared in advance F2k's fit
(`5516b19`) stands unchanged and only the 263 scoring changes.

Doppler RMS / mean after screening (mm/s; before screening only 263 differs):

| day | F2k | LOLA |
|---|---|---|
| 260 | 2.08 / −0.30 | 4.03 / −3.21 |
| 261 | 2.85 / 0.13 | 4.53 / −3.24 |
| 262 | 1.60 / 0.18 | 4.20 / −3.55 |
| 263 (predicted) | 5.22 / −4.46 (before: 23.81 / −4.25) | 3.93 / −3.37 (before: 23.52 / −3.01) |

#### Reading

1. **The rule removes exactly the bad pass and nothing else.** The threshold (~10 mm/s) sits between the bad
   pass (69 mm/s) and every other pass, fitted or predicted, against either orbit: the largest kept scatter
   about a pass mean is 3.5 mm/s (F2k) and 3.8 mm/s (LOLA).
2. **F2k's screened 263 Doppler prediction is 5.2 mm/s, mostly a mean offset.** Its mean is −4.5 mm/s and its
   scatter about that mean is 2.7 mm/s; on the fit days the mean is within ±0.3 mm/s. LOLA's mean is −3.2 to
   −3.6 mm/s on every day (the Doppler bias was estimated with our orbit), so relative to LOLA the F2k 263 mean
   drifts by about −1 mm/s. The largest single contribution is the last 263 pass (PAD 126, 23.7 h after the fit
   end; 7.8 mm/s in F2l).
3. **The same rule cannot screen range across fit and prediction days** (reported only, not applied). Against
   F2k's orbit the median range-pass scatter is 0.44 m, set by the fitted days, and every 263 range pass
   (3.2–7.4 m, the prediction error) would be rejected. Against LOLA no range pass would be. A pass-scatter rule
   on residuals of one's own orbit is valid only where the orbit's own error stays well below the threshold; for
   Doppler on 263 it does (the largest kept pass scatter about its mean is 3.5 mm/s against 9.6), for range it
   does not.
4. **Updated F2k prediction on 263 (screened):** range 5.2 m RMS, Doppler 5.2 mm/s RMS, RIC vs LOLA 1.9 / 5.1 /
   2.9 m. For comparison F2h's two-day fit predicted its next day (262, nothing screened there) to 1.6 m /
   1.8 mm/s and 2.3 / 6.6 / 10.5 m against LOLA. Against the data the three-day fit predicts its next day worse
   than the two-day fit did; against LOLA it predicts better. The range part of that gap is at the level the
   LOLA orbit itself reaches on 263 (5.0 m), so it reflects the 263 data in our model as much as the orbit.

#### Next steps

1. Find the cause of the bad pass (residual time series, raw count continuity, the 470 s gap).
2. Adopt the Doppler pass-screening rule for future prediction scoring (fit days included, F2k's orbit, 5 ×
   median), recorded in each phase's predeclaration.
3. The range-level question on 263 (LOLA itself 5.0 m with F2k's biases; §16's −13 to −17 m/h per-pass slopes /
   ~3 ms timing offset) is open.
