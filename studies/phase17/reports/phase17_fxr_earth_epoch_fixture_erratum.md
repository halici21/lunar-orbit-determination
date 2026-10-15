# ERRATUM — PHASE 17 EARTH-EPOCH FIXTURE (R1M, R1COV, R1O, R1O-D, R1O-OPT, R1O-R)

**Status:** authoritative map between historical and corrected Phase 17 results.
**Issued by:** Phase 17-FXR. The full evidence is in `studies/phase17/reports/phase17_fxr_earth_epoch_fixture_requalification.md`.
**Evidence artifacts:** `studies/phase17/results/phase17_fxr_epoch_audit.json`,
`studies/phase17/results/phase17_fxr_requalification.json`, `studies/phase17/results/phase17_fxr_g5_audit.json`.
**Truth level:** every number below is `SYNTHETIC_CAMPAIGN_TRUTH` (K_truth = 0.01 m²/kg is a campaign
setting, not a spacecraft property). The dynamics are the inherited simplified model: Moon point
mass, lunar J2 and cannonball SRP, with no Earth or Sun third-body gravity.

> ### ERRATUM (Phase 17-R1O-D2, enforced by Phase 17-HARDEN): lunar occultation
>
> **0.3204 is superseded.** The range + production ΔDOR 5 nrad value 0.3204 (and its surrogate twin 0.3203) is labelled
> **`EPOCH_CORRECTED_BUT_ELEVATION_ONLY_HISTORICAL_INTERMEDIATE`**. FXR corrected its Earth epoch, but
> its ΔDOR rows were still accepted on station elevation alone, without a lunar-occultation test. As a result,
> **93 of 263 rows (35%) had the spacecraft behind the Moon** for at least one station.
>
> **Authoritative values** (Phase 17-R1O-D2, canonical Goldstone–Canberra, 5 nrad, both stations visible under the
> production elevation + occultation contract):
>
> | quantity | value |
> |---|---:|
> | range-only `f_perp` | **0.091508** |
> | range + ΔDOR `f_perp` | **0.172351** |
> | gain Δ`f_perp` | **+0.0808** (STRONG_DIRECTION_GAIN) |
> | σ_K/K | **7.46%** |
>
> **R1O surrogates under the same contract** (Phase 17-HARDEN, `studies/phase17/results/phase17_harden_visibility_requalification.json`):
>
> | surrogate | 5 nrad `f_perp` | 1 nrad `f_perp` | rows |
> |---|---:|---:|---|
> | ΔDOR-like | 0.1722 | 0.1381 | 170 kept of 263 |
> | Earth LOS | 0.0919 | — | 360 kept; 230 of 590 were occulted |
>
> The R1O-D-era bridge surrogate uses an Earth frozen at t = 0 and is historical only. On the legacy (elevation-only) path, every FXR number
> still reproduces bitwise. The single current status is in `studies/phase17/reports/phase17_authoritative_status.md`. The defect is
> HARDEN-D04 in `studies/phase17/reports/phase17_harden_defect_ledger.md`. The text below is kept unchanged as the historical record.


Historical numbers are **not deleted** anywhere. Each affected report keeps its text and carries an
erratum block that points here.

---

## 1. What was wrong

The Phase 17 range-analysis fixture (`examples/phase17_r1m_core.build_range_arc`) built its
measurement geometry from two different epochs:

| quantity | epoch used |
|---|---|
| spacecraft state, propagation t = 0 | `phase7_et0` = 857806356.99998 ET (2027-03-08T19:31:27.814 UTC) |
| Sun (SRP) | `phase7_et0 + t` |
| Earth rotation (ITRF93 → J2000, station placement) | `phase7_et0 + t` |
| **Earth inertial position and velocity** (station centre, visibility, light time) | **`C.ET0 + t` = `phase7_et0 − 504000 s + t`** |

The Earth came from `od_gravity_covariance_campaign.get_earth_pos`, a table whose t = 0 is the
campaign's canonical ET0 (`ephemeris_data.mat:first_jd`). That is **504000 s = 5.8333 days** before
the measurement epoch. The table covers only 0 … 28260 s (3.9986 orbits). `numpy.interp` holds both
position *and* velocity constant beyond that. So on the 15-orbit arc the Earth froze during orbit 4,
and **75% of the 782 published observations** were formed against a frozen Earth.

The range line of sight was therefore drawn to an Earth displaced by **71° at t = 0, growing to 83°**,
as seen from the Moon. The effect on the Sun–Moon–Earth angle:

- correct epoch: 174.4° (Earth almost exactly anti-sunward);
- published fixture: 114.2°.

For windowed multi-arc arcs (R1M, R1O confirmation), each window restarted the table at its own
t = 0, so the offset there was 504000 s **plus** the window start.

It was an analysis-fixture defect. Production physics, SPICE, and the range, ΔDOR and optical
models are all unaffected. The production range model receives Earth as an untagged callable of
scenario seconds and cannot detect a mis-originated table.

**Repair.** The Earth the measurement model sees is now an epoch-tagged SPICE ephemeris at the same
ET origin the production model uses for the Earth rotation (`pass_geo.et0_s`). Its coverage is
explicit, and a request outside that coverage raises `EarthEphemerisCoverageError` instead of
clamping. The defective fixture remains available only as
`EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED`, which is never the default and emits
`SupersededEarthFixtureWarning`. Protection: `tests/test_fxr_earth_epoch_fixture.py`.

## 2. What it did to the canonical range baseline

Configuration: 15 orbits, Goldstone + Madrid + Canberra, 90 s cadence. All four arms have an
identical truth trajectory, STM and dx/dK. Only the Earth handed to the measurement model differs.

| arm | obs | `f_perp` | `theta_K` | `I(K\|x)` | `sigma_K/K` |
|---|---:|---:|---:|---:|---:|
| A published (offset + clamp) | 782 | **0.252292** | 14.61° | 1.1628e+06 | **9.27%** |
| B offset only | 768 | 0.206667 | 11.93° | 7.2905e+05 | 11.71% |
| C clamp only | 968 | 0.085728 | 4.92° | 1.2335e+05 | 28.47% |
| **D corrected (canonical)** | **960** | **0.091508** | **5.25°** | **1.5497e+05** | **25.40%** |

Repairing the epoch moves `f_perp` by −0.115 (with no clamp) and −0.167 (with the clamp).
Repairing the clamp moves it by +0.006 and −0.046 respectively. **The epoch offset is the dominant
defect**; the clamp is secondary, and the two interact (+0.051).

**Authoritative canonical K_SRP range baseline from now on:** arm D — 960 obs,
`f_perp = 0.091508`, `theta_K = 5.25°`, `I(K|x) = 1.5497e+05`, `sigma_K = 2.5402e-03 m²/kg`
(25.40%). This is bitwise identical to GEO's `G0_consistent`.

## 3. Cross-phase table

Labels (FXR s40): **MQ** mathematical qualification · **SQ** software qualification ·
**SSR** synthetic science result · **HNB** historical numeric baseline.

| Phase | Historical quantity / result | Historical fixture | Corrected (FXR) | Absolute changed? | Relative conclusion changed? | Current status | Action |
|---|---|---|---|---|---|---|---|
| **R1M** | G0–G3 `f_perp` 0.009 / 0.225 / 0.243 / 0.269. `I(K\|x)` 1.22 / 2.66e3 / 9.36e3 / 3.52e4. Best case M3-linked 4.90e4 (σ_K/K 45%). `f_perp` saturates at 0.28–0.30. Ordering pure F3 < contiguous G3 < linked M3 | offset + clamp. Windows offset by 504000 s + start | G0–G3 `f_perp` 0.051 / 0.067 / 0.073 / 0.059. `I(K\|x)` 26.9 / 164 / 615 / 1856. M3-linked 780 (σ_K/K 358%). F3-linked 1.25e4 (σ_K/K **89%**, best). `f_perp` ≤ 0.088. Pure F3 570 < G3 1856, but **M3-linked 780 < G3** | YES | **PARTIAL**: core conclusions confirmed; the "M3-linked beats contiguous G3" ordering is not | HNB superseded. SSR partially confirmed | erratum block; corrected tables in FXR §16 |
| **R1COV** | QR covariance = exact oracle (≤1.5e-13). Floored normal path overconfident (G0 ×261). G0/G1/G3 σ_K 0.904 / 0.0194 / 0.00533. M3-linked 0.00452 (45%) | offset + clamp (numbers only) | QR = oracle (≤3.4e-13) on corrected designs. cond(info) = cond(A)² still. Floored path still overconfident (G0 ×65, G1 ×45, G3 ×51, M3 ×75). σ_K 0.193 / 0.0781 / 0.0232. M3-linked 0.0358 (358%) | YES (numbers) | NO | MQ **valid, unchanged**. HNB superseded | erratum block |
| **R1O** | Original ranking already superseded by R1O-R (STM). R1O-R corrected: range 0.2523; landmark 0.2530 (+0.0007); ΔDOR-like 1 nrad 0.0701, 5 nrad 0.2070; Earth-LOS 0.2578. "No candidate improves on range" | offset + clamp | range 0.0915; landmark **0.2524 (+0.161)**; ΔDOR-like 1 nrad **0.3619 (+0.270)**, 5 nrad **0.3203 (+0.229)**; Earth-LOS 0.0920 (+0.0005) | YES | **YES (CHANGED)**: both surrogates raise `f_perp` above the corrected range baseline. The LOS negative control stays null | original ranking still invalid. R1O-R's corrected R1O conclusion superseded | erratum block |
| **R1O-D** | Production ΔDOR 5 nrad: `f_perp` 0.2523 → 0.2121, gain −0.040, **MAGNITUDE_ONLY**. σ_K/K 9.27% → 8.67%. 1 ns session bias → K shift 22.5% of truth | offset + clamp **+ Earth frozen at t = 0 for the whole arc** | `f_perp` 0.0915 → **0.3204**, gain **+0.229**, **STRONG_DIRECTION_GAIN**. σ_K/K 25.40% → **3.97%** (1 nrad: 0.83%). 1 ns bias → 2.5%. Epoch-only repair (frozen anchor kept): +0.245 STRONG | YES | **YES (CHANGED)**. Cause isolated: flips iff the Earth epoch is corrected | production ΔDOR SQ valid. Relative SSR **changed** | erratum block; focused follow-up (FXR §44) |
| **R1O-OPT** | range (10 s, Canberra) 0.2950, σ_K/K 3.50%; optical-only 0.2617; best combined 0.3253; improvement ×1.18; **MARGINAL_DIRECTION_GAIN** | offset + clamp (range block only) | range (10 s, **Goldstone**) 0.0861, σ_K/K 12.82%; optical-only 0.2617 (bitwise-identical rows); best realistic combined 0.1405 (0.1 px); improvement ×1.79; **MARGINAL_DIRECTION_GAIN** | YES | NO (class confirmed) | production optical SQ valid. SSR confirmed | erratum block |
| **R1O-R** | STM must be unpacked `order="F"`; historical defect reproducible bitwise. Context baseline 0.2523 | offset + clamp (context numbers) | STM repair valid on both fixtures (bitwise equivalence, nom48 fixture-independent, 5/5 STM tests pass). Its R1O-corrected table: see R1O row | context numbers only | STM: NO. R1O conclusion: see R1O row | SQ **valid** | erratum block |
| **GEO** | Canonical consistent `f_perp` 0.091508, σ_K/K 25.40%, 960 obs; 48-case class `GEOMETRY_CHANGES_INFORMATION_MAGNITUDE_WITHOUT_BREAKING_K_STATE_DEGENERACY` | already consistent | FXR canonical = GEO `G0_consistent` **bitwise**. G5 confirmed as a physical K-truth sweep, invariant to ~1e-4 relative (not "exactly") | NO | NO | SSR valid (range-only) | wording note only |

## 4. What survives, by type

- **Mathematical (MQ):** R1COV's QR/square-root covariance recovery; cond(HᵀWH) = cond(A)²; the
  Schur identity `I(K|x) = ‖b_K,⊥‖²`; the floor artifact of the eigenvalue-clipped normal path.
  All re-verified on the corrected designs.
- **Software (SQ):** R1O-R's column-major STM contract; R1O-D's production ΔDOR event solver,
  Jacobian and K chain; R1O-OPT's production optical model. These are geometry-independent
  qualifications.
- **Synthetic science (SSR):**
  - R1M: K is never separated from the state by range-only multi-arc processing. Rank stays 6/7,
    the weakest mode stays K-aligned, and nothing is practically identifiable.
  - R1O-OPT: marginal optical direction gain.
  - The Earth-LOS negative control.
  - GEO's range-only geometry result.
- **Changed (SSR):** the ΔDOR relative conclusion (R1O-D and the R1O surrogate). It is **not
  magnitude-only** on the consistent geometry.
- **Superseded (HNB):** every absolute range-baseline number quoted from the published fixture,
  listed in FXR §28.

## 5. What this does NOT establish

That ΔDOR makes K_SRP identifiable for a real mission. The corrected ΔDOR result is:

- synthetic;
- single-arc, one quasar, one station pair;
- simplified dynamics with no third-body gravity;
- free of process noise;
- a formal covariance only.

It is also a *relative-direction* gain. The normal-matrix rank (6/7) and the K-alignment of the
weakest mode are unchanged. Before K parameterization, a focused ΔDOR follow-up must test whether
the gain survives those choices.
