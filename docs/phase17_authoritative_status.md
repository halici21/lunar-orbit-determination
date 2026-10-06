# Phase 17 — authoritative status

**This is the single place to find which Phase 17 number is current.** Where any other document disagrees
with this page, this page wins. Every phase report keeps its historical text and carries erratum blocks
that point here.

- **Issued by:** Phase 17-HARDEN (2026-09-26). **Updated by:** Phase 17-F1 (2026-09-26): SRP position gradient in
  the variational equations; canonical numbers below re-qualified, every class preserved.
- **Updated by:** the Phase 17-R1O-CD1 closure (2026-10-04). It adds the D3, D4 and CD1 status rows (§3) and corrects
  the HARDEN-D23 size (§4). No §1 number changed: D3, D4 and CD1 answer different questions with their own configurations.
- **Updated by:** Phase 17-KP1 (2026-10-04). It adds the KP1 row (§3). No §1 number changed.
- **Updated by:** Phase 17-HARDEN-D23 (2026-10-07). The counted-Doppler Earth interpolation is repaired (§3, §4). No §1
  number changed.
- **Truth level:** every number is `SYNTHETIC_CAMPAIGN_TRUTH`. K_truth = 0.01 m²/kg is a campaign value, not a spacecraft property.
- **Dynamics:** the inherited simplified model (Moon point mass, lunar J2 and cannonball SRP) unless a row says otherwise.
  Since F1 the STM and dx/dK linearise the SRP force exactly (distance, direction and penumbra terms).

## 1. Canonical K_SRP baseline (current)

Canonical arc: 15 orbits, DSN-3, 90 s cadence, 960 range observations. Earth at the measurement epoch
(FXR canonical fixture); production visibility (elevation + lunar occultation).

| quantity | authoritative value | source |
|---|---:|---|
| range-only `f_perp` / θ_K / σ_K/K | **0.091502** / 5.25° / **25.40%** | F1 re-qualification of FXR/GEO `G0_consistent` |
| range + production ΔDOR, Goldstone–Canberra, 5 nrad: `f_perp` | **0.172343** | F1 re-qualification of R1O-D2 CANON/GC |
| gain Δ`f_perp` | **+0.0808** (+0.080841; STRONG_DIRECTION_GAIN under the R1O-D threshold) | F1 / R1O-D2 |
| σ_K/K (range + ΔDOR) | **7.46%** (7.4565%) | F1 / R1O-D2 |
| ΔDOR rows (both stations visible) | 170 (of 263 elevation-only rows) | R1O-D2; HARDEN |

Source of every F1 number: `artifacts/phase17_f1_science_materiality.json` (`comparison`).

**Superseded digits (pre-F1, omitted SRP gradient):** range-only 0.091508, combined 0.172351, gain +0.080843.
They appear in the FXR, R1O-D2, HARDEN and GEO documents and in the erratum blocks of six reports; those
blocks point here, and the values in this table replace them. The change is in the 5th–6th significant digit.
It moves no class in the D2 primary population (18 cells), the 5 frozen GEO spot cases or the D2 high-fidelity
cases (`docs/phase17_f1_srp_variational_gradient_qualification.md`).

**Historical intermediate (do not cite as current):** `f_perp` **0.3204**, σ_K/K 3.97%, labelled
`EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`. Its Earth epoch is correct (FXR), but 93 of its 263
ΔDOR rows had the spacecraft behind the Moon (defect HARDEN-D04).

## 2. Scope of the ΔDOR result (R1O-D2)

The gain is **strongly geometry-dependent**. Predeclared outcome:
`DDOR_K_INFORMATION_GAIN_IS_STRONGLY_GEOMETRY_DEPENDENT`.

- **Where it is STRONG:** only on Goldstone–Canberra at the canonical epoch.
- **Where it is absent:** at the other three lunar-phase epochs, and on Goldstone–Madrid.
- **Calibration:** unconstrained per-session calibration biases remove it. A DSN-level bias prior keeps about half.
- **Identifiability:** K remains the weakest design direction everywhere.

See `docs/phase17_r1od2_ddor_robustness_qualification.md`.

## 3. Phase-by-phase status

| phase | current status | what is superseded | where |
|---|---|---|---|
| R1 / R1G | historical context | geometry numbers on the published fixture | their reports; FXR erratum |
| R1M | qualitative conclusion PARTIAL: range-only multi-arc does not separate K | absolute numbers (published Earth fixture) | FXR erratum |
| R1COV | QR square-root covariance algorithm VALID | canonical numbers | FXR erratum |
| R1O | surrogate conclusions superseded twice (FXR epoch, then occultation) | ΔDOR-like 0.3203 → **0.1722**; Earth LOS 0.0920 → **0.0919** (230/590 rows were occulted) | `artifacts/phase17_harden_visibility_requalification.json` |
| R1O-R | column-major STM repair VALID | baseline numbers | FXR erratum; `tests/test_r1o_stm_layout.py` |
| R1O-D | production ΔDOR model + software qualification VALID | all campaign numbers; current = §1 | R1O-D report erratum blocks |
| R1O-OPT | MARGINAL_DIRECTION_GAIN confirmed (optical rows are Earth-independent) | numbers combined with the old range block | FXR erratum |
| GEO | `GEOMETRY_CHANGES_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY` (range-only; unaffected by D04) | — | GEO report |
| FXR | Earth-epoch repair VALID. Its R1O-D ΔDOR number is the historical intermediate | 0.3204 | FXR report + erratum blocks |
| R1O-D2 | **current ΔDOR science** | — | D2 report |
| HARDEN | infrastructure and contracts; no new science | — | `docs/phase17_harden_*` |
| F1 | SRP position gradient in the variational equations QUALIFIED; `NUMERICAL_BASELINES_UPDATED_CLASSES_PRESERVED` | canonical 6-digit values (§1) | `docs/phase17_f1_srp_variational_gradient_qualification.md` |
| R1O-D3 | **current ΔDOR tracking-design science**: `DDOR_TRACKING_GAIN_REMAINS_CONFINED_TO_NARROW_GEOMETRIC_WINDOWS`. Under one-hour operational sessions the canonical GC gain is MODEST (Δ`f_perp` +0.0328), not the dense §1 value. Nonlinear confirmation NOT justified | — (§1 stays the dense/ideal D2 configuration) | `docs/phase17_r1od3_tracking_design_characterization.md` |
| R1O-D4 | D3's exploratory GM FIRST_SESSION window **not confirmed**: `GM_WINDOW_IS_DYNAMICS_FIDELITY_FRAGILE`. It exists in simplified dynamics (85°–115°) but no high-fidelity orbit has the required 30° window (0/5); the shift follows the range-only baseline (r = 0.976). Nonlinear confirmation NOT justified; the ΔDOR scheduling branch has no remaining hypothesis | D3 secondary hypothesis | `docs/phase17_r1od4_gm_window_confirmatory_qualification.md` |
| R1O-CD1 | **current counted-Doppler science**: `COUNTED_DOPPLER_K_INFORMATION_IS_SYSTEMATICALLY_FRAGILE` (Case D). Over 120 high-fidelity cells, ideal Doppler passes the population rule (76.7 % positive); one free bias per pass fails it (median Δ`f_perp` −0.006, retention LOST); every cell is WEAK in absolute separation; σ_K/K ≈ 0.5 % is magnitude, not direction. `ALTERNATIVE_K_PARAMETERIZATION_JUSTIFIED = YES`, nonlinear confirmation NOT justified. Gate PASS after closure (D23 immaterial; one bounded rebuild numerical-floor exception) | — | `docs/phase17_r1o_cd1_counted_doppler_characterization.md` §1, §49–§50 |
| KP1 | **current K-parameterization science**: `ALTERNATIVE_K_PARAMETERIZATION_ADDS_DISTINGUISHABLE_DIRECTIONS`, for the *variation* of K only. Under range + counted Doppler with a free bias per pass (high fidelity, 120 cells), the three-segment model's variation directions separate from state and constant K at MODERATE+ in 57.5 % of cells, with internal correlations ≤ 0.90 everywhere. Two halves give PARTIAL (25.8 %), linear drift NO_NEW_INFORMATION. The **K level stays WEAK**: every block's weakest, common-mode-like combination (≈ 0.085) is below the scalar f_perp (0.133). Orbit-dependent (the 500 km orbit: 0/24); fidelity-robust. Linear and local at a constant truth; the next K-fitting phase needs a HARDEN-D23 repair first | — | `docs/phase17_kp1_alternative_k_parameterization_characterization.md` |
| HARDEN-D23 | counted-Doppler Earth **repaired**: the production station provider uses the same cubic-Hermite Earth as range and ΔDOR (R3-P22's linear policy superseded). Against a direct-SPICE oracle (frozen model S solver, 239,265 CD1/KP1 rows) production values agree to max 7.6e-8 m/s, rms 1.6e-8 (ET-exact oracle: max 5.4e-9); the historical offset (1.83e-3 m/s) is reproduced and removed 2.4e4×. CD1 and KP1 re-qualified **unchanged** (f_perp ≤ 4e-10, principal sines ≤ 1e-9, 0 class flips). Counted-Doppler **values are qualified for estimation at the 1e-7 m/s level**. Gate: PASS on 8 of 9 frozen criteria; the epoch-contract diagnostic measured the oracle's ET rounding (6.4e-5 m) and awaits owner acceptance of the adjudication (exact-epoch scan 2.4e-7 m ≤ 1e-6) | — | `docs/phase17_harden_d23_repair_qualification.md` |

## 4. Known approximations that remain (documented, quantified)

The STM's former omission of the SRP position gradient (HARDEN-D05) is **fixed** by Phase 17-F1.

| item | size | ledger |
|---|---|---|
| ~~counted Doppler interpolates the Earth linearly~~ **FIXED (HARDEN-D23, 2026-10-07)** | historical: 1.67–1.83e-3 m/s value offset (superseded quantification ~1.5e-6 m/s). Now cubic Hermite: Earth ≤ 2.4e-7 m, values vs direct SPICE ≤ 7.6e-8 m/s | HARDEN-D23 |
| exact station transform at rounded absolute ET (counted Doppler, range, ΔDOR) | counted-Doppler value rms ≤ 1.0e-7 m/s (max 3.6e-7); ~20× below the optimistic operational noise floor | HARDEN-D24 |
| station B receive up to ~0.03 s past the last grid node (ΔDOR) | ~1e-5 m | HARDEN-D16 |

## 5. How to reproduce a historical number

Historical scripts refuse to run unless `PHASE17_LEGACY_REPRODUCTION=1` is set. The elevation-only row rule
must be named explicitly (`DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY`) and warns when used. The published FXR
artifacts reproduce bitwise through `examples/phase17_fxr_requalification.py`, which names that rule.

For the full defect list, see `docs/phase17_harden_defect_ledger.md`.
