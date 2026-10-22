# Phase C2-I — Station time and the station clock consider parameter: implementation and qualification

- **Date:** 2026-10-22
- **Predeclaration:** `studies/phase17/configs/phase_c2i_station_time_implementation_predeclaration.json`.
  - Committed in `5a8f5d9` before any C2-I code or number.
  - Amendment A1 was committed in `0fe1396`, after G2/G3 and before any G1 or G4–G9 number. It covers the G6
    status clause, the G5 step sweep and the midpoint-rate evaluation of the interval difference.
- **Basis:** `studies/phase17/reports/phase_c2_time_scale_materiality.md`. The owner's decisions are in its §7.
- **Code:**
  - `src/lunar_od/station_time.py` (new).
  - `src/lunar_od/_erfa_dtdb_table.py` (new, generated).
  - `src/lunar_od/consider_covariance.py` (new).
  - `src/lunar_od/relativistic_light_time.py` (extended; no frozen identity).
- **Unchanged:** `estimators.py` and `measurement_ingestion.py` keep their bytes, so no re-freeze is needed.
  `two_way_range.py`, `radiometrics.py` and `delta_dor.py` are also untouched.
- **Evidence:**
  - `studies/phase17/results/phase_c2i_g2_g3.json`.
  - The first-run record `phase_c2i_g2_g3_run1_calendar_defect.json`.
  - `phase_c2i_qualification.json`, `phase_c2i_estimator_gates.json`.
  - The pyerfa reference `phase_c2i_erfa_reference.npz` (sha256 `ac663373…ca91c`).
- **Scripts:** `studies/phase17/scripts/phase_c2i_*.py`. The `*_sandbox.py` scripts run only in
  `agent-sandbox/c2-erfa-reference`.

## 0. Final status

```
GATES (as computed)     G1 PASS (bit identity 0 difference; full suite 1,878 passed, 0 failed)
                        G2 PASS | G3 PASS (run 1 FAIL: a calendar defect, fixed, s2.3)
                        G4 range, delta-DOR, epoch parts PASS; counted Doppler FAIL as computed (3.0e-9 vs 1e-9 m/s)
                        G5 FAIL as computed (state partials identical to C1-I's owner-accepted record; clock
                           partials fail on the arc's last range row and at the predeclared Doppler steps)
                        G6 FAIL as computed (BLS-LM 1.51 m, different status; SRIF 2.6 cm)
                        G7 PASS (5.2 %) - but see s3.3 | G8 PASS | G9 informational
OUTCOME (rule)          C2I_NOT_QUALIFIED as computed
ADJUDICATIONS           G4 Doppler, G5, G6 QUALIFIED - owner accepted 2026-10-22 (s4, s6)
OUTCOME                 C2I_QUALIFIED (G1-G8 PASS, three of them by owner-accepted adjudication)
```

Three of the failing thresholds were set by me without the numerical floors that were already known (C1
lesson 5): the counted-Doppler evaluation floor (G4), C1-I's own finite-difference floor (G5, which C1-I's
identical numbers failed in the same way) and the non-convergence of the C1-I closure (G6, partly corrected
by A1). The evidence below is post-hoc and labelled as such; no threshold was changed.

## 1. What was built

**`station_time.py`**
- **`dtdb` / `dtdb_rate`:** a port of ERFA `eraDtdb` and its analytic derivative.
  - The 787 + 10 + 4 coefficients are generated from ERFA 2.0.1 `dtdb.c` by
    `phase_c2i_generate_dtdb_table.py`.
  - The source sha256 is pinned: `228bd7c7…`, from `pyerfa-2.0.1.5.tar.gz`, sha256 `17d6b24f…`, matching
    PyPI. The BSD-3 notice is in `LICENSES/ERFA.txt` and listed in `pyproject.toml`.
- **`tdb_minus_tt` / `tdb_minus_tt_rate`:** evaluated at the production epochs `et0 + t`, with the epoch kept
  in two parts.
- **`utc_to_tdb_seconds`:** converts UTC tags to TDB seconds past `et0`.
  - TAI−UTC comes from the loaded LSK; TT = TAI + 32.184 s; TDB = TT + `dtdb` at the site.
  - `ut` is UTC.
- **`spice_utc_chain_tdb_seconds`:** the SPICE chain, used for comparisons.

**`relativistic_light_time.py`**
- `LightTimeModel.station_time` takes `DELTET_GEOCENTRIC` (the default: the C1-I relation, bit-identical)
  or `FULL_TOPOCENTRIC`.
  - Under `FULL_TOPOCENTRIC`, each station interval is converted with D_site(t3) − D_site(t1), evaluated as
    round trip × the station-time rate at the interval midpoint (A1).
  - `FULL_TOPOCENTRIC` requires `interval_time_scale = 'TT'` and the station site.
- **`DSN_C2_LIGHT_TIME_MODEL`** is new. `DSN_LIGHT_TIME_MODEL` is unchanged.
- **Delta-DOR:** `delta_dor_station_time` gives the delay as the two station clocks measure it.
  - Formula: D_TT = D_TDB − [D_B(t_B) − D_A(t_A)] + b_A − b_B.
  - This carries the −(V⊕·b)/c² term (owner decision, C2 report s5.2).
- **Clock offsets:** per-station clock offsets use the production convention (the receive tag `t` is read as
  `t + b`).
  - Analytic partials come from the implicit event equations: `two_way_range_clock_partial`, the
    counted-Doppler `with_clock_partial`, and `delta_dor_clock_partials`.
- **Pass metadata:**
  - It gains `station_time`; older metadata without the key reads as `DELTET_GEOCENTRIC`.
  - It optionally gains `station_clock_offsets_s`.
  - `pass_station_kwargs` supplies sites and offsets to the estimator path, so BLS-LM and SRIF see the C2
    model through the existing `two_way_light_time_ssb` dispatch.

**`consider_covariance.py`**
- QR of `[A_x | A_c]` gives P_x, S = −R_xx⁻¹R_xc and P_consider = P_x + S P_cc Sᵀ, without forming the normal
  matrix.
- The a priori σ of the consider parameter is a required input (owner decision s5.3).

**Tests:** `tests/test_station_time.py` (11) and `tests/test_relativistic_light_time_c2.py` (12).

## 2. Gates

### 2.1 G1 — default invariance

- Bit identity on all 10 KP2 cells against the branch-base module (`d24e380`, loaded from git): two-way
  range, counted Doppler and delta-DOR under `DSN_LIGHT_TIME_MODEL` and `LightTimeModel()`, max |diff| =
  **0** — PASS.
- Full suite: **1,878 passed, 29 skipped, 0 failed** (C1-I: 1,855 passed; +23 new tests) - PASS. G1 therefore **PASS**.

### 2.2 G2, G3 (`phase_c2i_g2_g3.json`)

| gate | result | value |
|---|---|---|
| G2 value vs pyerfa (76,716 inputs: C2 epochs geocentric and at stations, 2,000 epochs 1950–2100 × 10 sites) | **PASS** | 1.1e-18 s (limit 1e-15) |
| G2 analytic rate vs h = 1 s difference (56,716) | **PASS** | 0.92 of the limit; the difference carries eraDtdb's own ~1.6e-7 s epoch-sum quantisation |
| G3a TAI−UTC vs erfa.dat (54 boundary instants + 1,000 random) | **PASS** | 0 mismatches |
| G3b UTC chain vs pyerfa (3,000 random tags 1980–2050, 3 DSN sites) | **PASS** | 1.46e-11 s (limit 1e-10) |
| G3c the same on the 28,358 C2 survey epochs | **PASS** | 1.46e-11 s |

### 2.3 Run 1 of G3: a real defect, fixed

The first run computed **G3 FAIL**: dates 1–2 days late, 27 leap-boundary mismatches.
- `_mjd` used Python floor division where eraCal2jd truncates toward zero: `(month − 14) / 12` is −1 for
  January–February and 0 otherwise.
- The fix is one line; the thresholds were not changed.
- The record is kept: `phase_c2i_g2_g3_run1_calendar_defect.json`. The gate did its job.

### 2.4 G4 — survey parity (`phase_c2i_qualification.json`)

| observable | result | value | limit |
|---|---|---|---|
| two-way range, DSN_C2 − DSN vs survey T2_RATE + T3_RATE | **PASS** | 4.4e-6 m | 1e-5 m |
| delta-DOR vs T3_DIFFERENTIAL + TT_INTERVAL_INFO | **PASS** | 4.0e-17 s | 1e-12 s |
| epoch parts (UTC tags minus the SPICE chain vs the survey's ERFA − SPICE + topo) | **PASS** | 1.45e-11 s | 1e-10 s |
| counted Doppler vs survey | **FAIL as computed** | 2.3–3.0e-9 m/s in every cell | 1e-9 m/s |

The change itself is 0.045 m (range) and 3.3 µm/s (Doppler) at most; the delta-DOR change is −2.78 … −3.30 µs.

### 2.5 G5 — partials vs finite differences

| partial | value (max relative, per component) | limit | result |
|---|---|---|---|
| range | 1.48e-4 | 1e-6 | FAIL as computed |
| counted Doppler | 4.16e-4 | 1e-5 | FAIL as computed |
| delta-DOR | 1.01e-5 | 1e-6 | FAIL as computed |
| clock, range | 1.50e-4 | 1e-6 | FAIL as computed |
| clock, Doppler | 3.30e-4 | 1e-5 | FAIL as computed |
| clock, delta-DOR A / B | 2.1e-10 / 2.2e-16 | 1e-6 | PASS |

### 2.6 G6 — closure (`phase_c2i_estimator_gates.json`)

| estimator | C1-I model on its truth | DSN_C2 on its truth | position difference | result |
|---|---|---|---|---|
| BLS-LM | DampingLimit, 11.59 m | Converged (STEP_NORM), 12.98 m | **1.51 m**, status differs | FAIL as computed |
| SRIF | MaxIter, 0.111 m | MaxIter, 0.086 m | 2.6 cm | FAIL as computed |

### 2.7 G7, G8, G9

- **G7 PASS:** the largest component (index 3) agrees within 5.2 % (z_nl 0.362, z_lin 0.382). See s3.3 for
  what it actually tests.
- **G8 PASS:**
  - (a) S against the bias response: 1.4e-12 relative.
  - (b) symmetric; inflation PSD (min/max eigenvalue −1.7e-15); P_cc = 0 gives P_x bit for bit.
  - (c) with σ_b = 1 µs on CANON k0 RANGE+DOPPLER at DSN_REALISTIC: |S b|/σ = 0.142213, against the survey's
    0.142213 (2e-7 relative). The consider σ inflation at 1 µs is at most **1.0 %**.
- **G9 (informational)** — OD impact of the C2 model change over the population (max |δθ/σ|):

| design | DSN_REALISTIC | PROJECT |
|---|---|---|
| RANGE_ONLY | 0.46 | 0.09 |
| RANGE+DOPPLER | 0.19 | 0.04 |

## 3. Post-hoc diagnostics (informational; no verdict changed)

### 3.1 G4 Doppler: the evaluation floor (`phase_c2i_g4_g5_diagnostics.json`, D3)

| comparison (vs the exact interval difference D_site(t3) − D_site(t1)) | CANON k0 | O5 k0 |
|---|---|---|
| the module, term by term (midpoint rate, A1) | 1.6e-10 m/s | 1.4e-10 m/s |
| the survey's receive-epoch form | 4.1e-10 m/s | 4.2e-10 m/s |
| the difference of two module Doppler **values** (what G4 compared) | 2.6e-9 m/s | 2.7e-9 m/s |

A counted-Doppler value is c/(2Tc) × a difference of two ~2.56 s scaled round trips. Its rounding floor is
c/(2Tc) × ulp(round trip) = **1.1e-9 m/s per value**, and G4's difference of two values reaches 2–3e-9.

The module's station-time change is exact to 1.6e-10 m/s: more exact than the survey's own form (4e-10).

### 3.2 G5: the C1-I floor, and the last row of an arc

- **State partials.** The G5 state numbers are the same, to the digits printed, as C1-I's exact-station,
  wide-K record (`phase_c1i_qualification_exact_stations_wide_k.json`):

  | cell | range | Doppler | delta-DOR |
  |---|---|---|---|
  | CANON k0 | 8.948e-6 | 4.623e-5 | 1.013e-5 |
  | O2 k12 | 1.476e-4 | 4.158e-4 | — |
  | O3 k0 | 6.88e-5 | 7.23e-5 | — |

  The owner accepted that record (C1-I G5 adjudication, row-normalised errors). The station-time relation
  adds nothing measurable to the partial error.
- **Clock partials** (D4):
  - Range: every row except the arc's **last** range row is within ≤ 1.1e-7. On the last row the difference
    is 0.030 m/s at every offset step (O2 k12, O3 k0), i.e. systematic. In the last Hermite interval, the
    interpolated velocity differs from the derivative of the interpolated position. That row is also C1-I
    G5's worst state row ([19, 3]). It is a property of the shared production interpolator, not of C2-I.
  - Doppler: with a 0.1 s offset step added (post-hoc), ≤ 5.4e-6 relative (limit 1e-5); row-normalised
    ≤ 3e-8. At the predeclared steps (≤ 1e-2 s) the finite difference of a ~1e-3 m/s-per-s partial sits in the
    1.1e-9 m/s evaluation floor of s3.1.

### 3.3 G6 and G7 (`phase_c2i_estimator_diagnostics.json`)

- **D2 — closure from the truth start.**
  - BLS-LM: **identical** for both models (DampingLimit, 7.6 mm, 21 iterations).
  - SRIF: MaxIter, 0.106 m (C1-I model) against 0.101 m (C2).
  - The models are consistent. G6's 1.5 m comes from where two non-converged LM paths stop, starting 100 m
    away on an integrator-limited cost (C1-I G6, Phase D).
- **D1 — what G7 tests.**
  - The station-time change alone is −5 … +45 mm.
  - G7's data-minus-model is ±1.8 m, because the construction (inherited from C1-I G7) evaluates the model on
    `propagate_state_with_k_sensitivity` while the truth comes from `propagate_state`. At the same
    tolerances (rtol 1e-11) the two propagators differ by **1.96 m** in position over the arc.
  - G7 therefore mostly validates the bias mapping of that mismatch, not of the station-time change.
  - The 1.96 m propagator mismatch is a pre-existing finding for Phase D.

## 4. Proposed adjudications (owner decision)

1. **G4 Doppler → QUALIFIED.** The predeclared 1e-9 m/s is below the 1.1e-9 m/s per-value evaluation floor of
   the observable. The change itself matches the exact interval difference within 1.6e-10 m/s (s3.1).
2. **G5 → QUALIFIED.**
   - The state partials reproduce the owner-accepted C1-I record digit for digit.
   - The clock partials pass at ≤ 1.1e-7 except on an arc's last range row, where the production Hermite
     interpolator's velocity/position inconsistency is systematic.
   - The Doppler clock partial passes with a 0.1 s step (s3.2).
3. **G6 → QUALIFIED.**
   - From the truth start, BLS-LM closes identically under both models and SRIF better under C2 (s3.3).
   - The 1 mm comparison between non-converged runs measures path dependence at the integrator floor, not
     the model.
4. **Note on G7:** it passes but tests mostly the propagator mismatch. A clean G7 needs the truth and the model
   on one propagator. That belongs with the Phase D integrator work.

With 1–3 accepted the outcome is **C2I_QUALIFIED**.

## 5. Opened items

- `propagate_state` vs `propagate_state_with_k_sensitivity`: a 1.96 m position difference at the same
  tolerances (D1). Proposed as a Phase D defect; it also affects the C1-I G7 construction.
- Last-interval Hermite inconsistency of the shared interpolator (s3.2): 0.03 m/s in dρ/dt on the arc's last
  row.
- Not done in C2-I (out of scope as predeclared):
  - scenario/desktop plumbing;
  - switching the generator default to `DSN_C2_LIGHT_TIME_MODEL`;
  - wiring `utc_to_tdb_seconds` into the CSV reader;
  - the clock σ value itself, which must come from DSN 810-005 or a clock-offset product.

## 6. Owner decision (2026-10-22)

The owner accepted the three adjudications of s4: G4 Doppler, G5 and G6 are QUALIFIED.

- **G1–G8 PASS**: five as computed, three by owner-accepted adjudication.
- **Outcome: C2I_QUALIFIED.**
- G7 passes as computed. Its caveat (s3.3) and the opened items of s5 stand.
