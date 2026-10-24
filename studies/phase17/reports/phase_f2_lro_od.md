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
