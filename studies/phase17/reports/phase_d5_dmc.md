# Phase D5: dynamic model compensation on the iterated extended Kalman smoother

**Outcome as computed: `D5_NOT_QUALIFIED`.** G1, G2, G4 and G6 pass. G3 and G5 fail. **D5-R (§7): `D5R_QUALIFIED` at σ = 2e-6 m/s².**

- **G3 fails on its accuracy ratio.** On the matched control, DMC costs 2.7–6× position RMS against the predeclared ≤ 2.0×.
- **G5 fails in one cell.** For position M1 the NEES is 3.99, just below the 4.19 band edge.

**What DMC does on mismodelled dynamics.** It cuts the iterated smoother's position error **10–185×** and brings NEES from 10⁵–10⁸ down to 4–6.7. It recovers the injected along-track acceleration to 6% (position) and 0.03% (range rate).

- Predeclaration: [`phase_d5_dmc_predeclaration.json`](../configs/phase_d5_dmc_predeclaration.json). Committed at `c581247`, before the EKF hook, the tests or any study number.
- Driver: [`phase_d5_dmc.py`](../scripts/phase_d5_dmc.py).
- Results: [`phase_d5_dmc.json`](../results/phase_d5_dmc.json).
- Code: [`src/lunar_od/dmc.py`](../../../src/lunar_od/dmc.py) (new), `dmc=` in [`src/lunar_od/ekf.py`](../../../src/lunar_od/ekf.py), tests in [`tests/test_dmc.py`](../../../tests/test_dmc.py).
- `dynamics.py` and `filters.py` are unchanged.

## 1. Model

**State.** The state is `[r, v, w, biases]`. `w` is a first-order Gauss–Markov acceleration in MCI (Tapley, Schutz & Born 2004, §4.9):

    r'' = a_model(r, t) + w,    w' = −w/τ + u,    E[u uᵀ] = (2σ²/τ) I δ(t−s)

**Propagation.** `propagate_dmc_state` integrates `f3body_moon + w` and its 9×9 STM with DOP853, using

    A₉ = [[0, I, 0], [G, 0, I], [0, 0, −I/τ]]

where G comes from `dynamics_jacobian_a_matrix`.

**Process noise.** `dmc_process_noise` is the exact van Loan discretisation of the linear [r, v, w] chain. The gravity gradient is left out of Q.

**EKF hook.** `run_lunar_ekf(dmc=…)`:

- inserts `w` after `v`;
- hands the shared measurement function the state without `w`;
- sizes the bias configuration as it would be without DMC;
- requires `process_noise=None`.

With `dmc` set, the iterated smoother takes its IEKS (process-noise) branch. `dmc=None` leaves every D3/D4 path unchanged.

**Tuning (predeclared, not changed afterwards).** σ = 5e-6 m/s², τ = 1800 s, w₀ ~ N(0, σ²I).

## 2. Experiment

**Arc.** D3's synthetic arc lengthened to 3 h (181 epochs, 60 s spacing, 3 stations):

- Moon point mass;
- rtol 1e-12 / atol 1e-13;
- prior S2km;
- 20 seeds per cell;
- noise as in D3.

**Truth cases.**

- C0: truth model equals the filter model.
- M1: a constant 2e-6 m/s² along-track push.
- M2: lunar J2 = 2.03e-6, i.e. 0.01 × the Moon's J2. The filter has none.

**Estimators.** All three get the same observations, x_guess and P0 on [r, v].

- **IS:** the D4-R iterated smoother without process noise.
- **IS_DMC:** the iterated smoother with DMC.
- **EKF_RTS_DMC:** a single EKF+RTS pass with DMC.

## 3. Results

Position RMS is taken over all update epochs and seeds. NEES is the mean over 20 seeds of the [r, v] NEES of the smoothed first-epoch state. The predeclared band is [4.19, 8.18].

| Cell | IS RMS (m) | IS_DMC RMS (m) | IS_DMC / IS | IS NEES | IS_DMC NEES | EKF_RTS_DMC RMS (m) |
|---|---|---|---|---|---|---|
| position / C0 | 0.168 | 0.458 | 2.73 | 5.83 | 4.32 | 22.1 |
| position / M1 | 28.36 | 0.458 | 0.016 | 7.0e6 | **3.99** | 16.6 |
| position / M2 | 4.16 | 0.422 | 0.101 | 7.1e5 | 4.30 | 9.36 |
| range_rate / C0 | 0.0262 | 0.158 | 6.03 | 5.62 | 6.68 | 6.45 |
| range_rate / M1 | 28.47 | 0.154 | 0.0054 | 4.3e8 | 6.34 | 5.24 |
| range_rate / M2 | 5.04 | 0.146 | 0.029 | 4.6e7 | 5.01 | 4.92 |

**Estimated along-track `w`.** Mean over the arc and seeds:

| Cell | Mean along-track `w` (m/s²) | Against the injected push |
|---|---|---|
| position / M1 | 1.88e-6 | 2e-6 injected |
| range_rate / M1 | 2.0006e-6 | 2e-6 injected |
| C0 and M2 | ≤ 8e-8 | M2's J2 error is not along-track on average |

**Convergence.**

- IS_DMC: 120/120 runs Converged in ≤ 3 iterations.
- IS on range_rate / M1: 4 of 20 runs ended NoDescent. A biased model has no zero-residual minimum near the start, and the D4-R line search stops there.
- The single-pass EKF_RTS_DMC from a 2 km prior is far from the iterated solution. This is the same nonlinearity that D3 reported without DMC.

## 4. Gates

| Gate | Result | Evidence |
|---|---|---|
| G1 STM | **PASS** | `test_dmc_stm_matches_central_finite_differences`: all 9 columns ≤ 1e-6 relative over 1 h with lunar J2 and an Earth third body; the w block is exactly e^{−Δt/τ}. `test_dmc_with_zero_acceleration_follows_the_production_propagator`: ≤ 1 mm vs `propagate_state`. |
| G2 noise and invariance | **PASS** | `test_van_loan_noise_matches_the_gauss_markov_limits`: Q_ww = σ²(1 − e^{−2Δt/τ}) to 1e-10, and Q → Q_c Δt for small Δt. `tests/test_ekf_smoother.py` is unchanged and passes. `test_negligible_dmc_reproduces_the_plain_ekf` also passes. |
| G3 matched control | **FAIL** | NEES is in band (4.32 and 6.68), but the RMS ratio is 2.73 for position and 6.03 for range rate, against the predeclared ≤ 2.0. |
| G4 mismodel accuracy | **PASS** | IS_DMC / IS ratios are 0.0054–0.101, against ≤ 0.5. |
| G5 mismodel consistency | **FAIL** | position / M1 NEES is 3.99, below 4.19. The other three cells are in band (4.30, 6.34, 5.01). |
| G6 convergence | **PASS** | 120/120 runs Converged. |

## 5. Reading the two failures (not an adjudication)

**G3: the ratio rule penalised the cost DMC is expected to have.** On a perfect force model, DMC adds 3 free accelerations per correlation time. The smoother then gives up the information that the orbit is purely deterministic. This raises position error even though the covariance stays honest: NEES is in band in both C0 cells.

How large the penalty is depends on how much the measurements constrain the orbit without dynamics. It is larger for range rate (6×), where the deterministic model carries more of the position information. The predeclared 2.0× was a guess at that cost, not a derived bound. As computed, it is exceeded.

**G5: position / M1 sits slightly below the band.** Mean NEES is 3.99 (median 3.31). A low NEES means a conservative covariance: σ = 5e-6 m/s² is 2.5× the injected 2e-6 push, so `w` is given more freedom than this truth uses. The same tuning gives 4.30 on position / M2.

**Neither failure was tuned away.** Retuning (for example σ chosen per arc) or replacing the G3 ratio rule would be a new predeclared phase. That decision belongs to the owner.

## 6. Runtime (reported, not gated)

The run used 12 worker processes while another session's 12-process pool was also running on the same 16 cores. Wall-clock times inside the results JSON are badly contaminated by this:

- position / C0 IS_DMC: median 3414 s;
- other cells: medians 18–42 s.

Rerunning position / C0 seed 0 alone gave an IS_DMC time of **10.9 s**, with bit-identical numbers (pos RMS 0.4205708656175265 for IS_DMC and 0.16506796627629902 for IS). The comparable IS time is about 4–5 s.

The DMC path integrates a Python right-hand side with DOP853. It is roughly 2–3× slower than the IS path when it has the CPU to itself.

## 7. D5-R: tuning sweep and a derived matched-control criterion

**Outcome as computed: `D5R_QUALIFIED` at σ = 2e-6 m/s²**, the only one of the four swept values that passes. D5's own outcome stays `D5_NOT_QUALIFIED` as computed.

- Predeclaration: [`phase_d5r_dmc_tuning_predeclaration.json`](../configs/phase_d5r_dmc_tuning_predeclaration.json). Committed at `86f3668`, before any D5-R number.
- Driver: `phase_d5_dmc.py --d5r`.
- Results: [`phase_d5r_dmc_tuning.json`](../results/phase_d5r_dmc_tuning.json).

**What changed from D5.**

- **σ is swept.** The values are 1e-6, 2e-6, 5e-6 and 1e-5 m/s², with τ = 1800 s.
- **The cells are D5's.** Cells, seeds, observations and x_guess are the same as in D5.
- **G3 is replaced by G3R.** G3R asks whether the matched-model accuracy loss is the loss the DMC covariance reports. It compares the empirical position-RMS ratio IS_DMC/IS with the ratio of their smoothed position σ over the arc, and requires empirical/predicted ∈ [0.8, 1.25] plus NEES in band. D5's ≤ 2.0 rule is still reported.

**G0.** At σ = 5e-6, D5's IS and IS_DMC numbers are reproduced to ≤ 1e-6 relative: **PASS**.

### Gates per σ

| σ (m/s²) | G3R | G4 | G5 | G6 | All |
|---|---|---|---|---|---|
| 1e-6 | PASS | PASS | **FAIL** (pos/M1 9.38, pos/M2 20.06) | PASS | no |
| **2e-6** | PASS | PASS | PASS | PASS | **yes** |
| 5e-6 (= D5) | PASS | PASS | **FAIL** (pos/M1 3.99) | PASS | no |
| 1e-5 | **FAIL** (pos/C0 NEES 4.13) | PASS | **FAIL** (pos/M1 3.71, pos/M2 3.92) | PASS | no |

### Matched control: is the cost honest?

| σ | Position: empirical / predicted ratio | Range rate: empirical / predicted ratio | D5's G3 (≤ 2.0) |
|---|---|---|---|
| 1e-6 | 2.03 / 2.07 = 0.98 | 5.41 / 5.48 = 0.99 | fail |
| 2e-6 | 2.29 / 2.33 = 0.98 | 5.86 / 5.75 = 1.02 | fail |
| 5e-6 | 2.73 / 2.75 = 0.99 | 6.03 / 5.85 = 1.03 | fail |
| 1e-5 | 3.11 / 3.11 = 1.00 | 6.06 / 5.87 = 1.03 | fail |

**The covariance predicts the matched-model cost to within 3% at every σ.** The cost is real and unavoidable at these σ, but it is not hidden: it is 2–3× for position measurements and 5.4–6× for range rate. Even σ = 1e-6 does not meet D5's 2.0× guess.

### IS_DMC position RMS and NEES at σ = 2e-6

| Cell | IS RMS (m) | IS_DMC RMS (m) | IS_DMC NEES |
|---|---|---|---|
| position / C0 | 0.168 | 0.383 | 4.54 |
| position / M1 | 28.4 | 0.404 | 5.06 |
| position / M2 | 4.16 | 0.385 | 7.24 |
| range_rate / C0 | 0.0262 | 0.153 | 6.27 |
| range_rate / M1 | 28.5 | 0.147 | 5.91 |
| range_rate / M2 | 5.04 | 0.143 | 5.41 |

### What the sweep shows

**The consistency window is narrow.**

- **Too small: σ = 1e-6.** The smoother is overconfident when the missing acceleration is larger than σ allows (pos/M2 NEES 20).
- **Too large: σ ≥ 5e-6.** It turns conservative for position measurements.

**Accuracy is not where the window bites.** G4 passes at every σ.

**σ = 2e-6 equals the injected M1 push.** The passing tuning is the one matched to the size of the unmodelled acceleration. This is the expected DMC behaviour, but it means:

- σ must be set from an a-priori estimate of the force-model error level, for example the size of the omitted terms;
- 2e-6 is this experiment's value, not a universal default.

**Runtime.** Runtimes in the JSON are again contaminated by another session's 12-process pool running at the same time. Section 6 gives the uncontaminated reference.
