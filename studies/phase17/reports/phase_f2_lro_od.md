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
