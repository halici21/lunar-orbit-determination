# Phase 17 — authoritative status

**This is the single place to find which Phase 17 number is current.** Where any other document disagrees
with this page, this page wins. Every phase report keeps its historical text and carries erratum blocks
that point here.

- **Issued by:** Phase 17-HARDEN (2026-09-26).
- **Truth level:** every number is `SYNTHETIC_CAMPAIGN_TRUTH`. K_truth = 0.01 m²/kg is a campaign value, not a spacecraft property.
- **Dynamics:** the inherited simplified model (Moon point mass, lunar J2 and cannonball SRP) unless a row says otherwise.

## 1. Canonical K_SRP baseline (current)

Canonical arc: 15 orbits, DSN-3, 90 s cadence, 960 range observations. Earth at the measurement epoch
(FXR canonical fixture); production visibility (elevation + lunar occultation).

| quantity | authoritative value | source |
|---|---:|---|
| range-only `f_perp` / θ_K / σ_K/K | **0.091508** / 5.25° / **25.40%** | FXR; bitwise = GEO `G0_consistent` |
| range + production ΔDOR, Goldstone–Canberra, 5 nrad: `f_perp` | **0.172351** | R1O-D2 `artifacts/phase17_r1od2_summary.json` (CANON/GC) |
| gain Δ`f_perp` | **+0.0808** (STRONG_DIRECTION_GAIN under the R1O-D threshold) | R1O-D2 |
| σ_K/K (range + ΔDOR) | **7.46%** | R1O-D2 |
| ΔDOR rows (both stations visible) | 170 (of 263 elevation-only rows) | R1O-D2; HARDEN |

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

## 4. Known approximations that remain (documented, quantified)

| item | size | ledger |
|---|---|---|
| STM omits the SRP position gradient (range-Jacobian state columns) | relative error 2e-6 to 1e-5 over 4 orbits; concentrated in penumbra | HARDEN-D05 |
| counted Doppler interpolates the Earth linearly | 2.62 m at mid-nodes; ~1.5e-6 m/s count rate | HARDEN-D23 |
| station B receive up to ~0.03 s past the last grid node (ΔDOR) | ~1e-5 m | HARDEN-D16 |

## 5. How to reproduce a historical number

Historical scripts refuse to run unless `PHASE17_LEGACY_REPRODUCTION=1` is set. The elevation-only row rule
must be named explicitly (`DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY`) and warns when used. The published FXR
artifacts reproduce bitwise through `examples/phase17_fxr_requalification.py`, which names that rule.

For the full defect list, see `docs/phase17_harden_defect_ledger.md`.
