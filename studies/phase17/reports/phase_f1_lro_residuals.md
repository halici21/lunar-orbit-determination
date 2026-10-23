# Phase F1 — Real LRO two-way range against NASA's precision orbit (measurement-chain residuals)

- **Date:** 2026-10-24
- **Predeclaration:** `studies/phase17/configs/phase_f1_lro_residuals_predeclaration.json` (`c9238d9`), before any
  residual.
- **Reader:** `src/lunar_od/utdf.py` with `tests/test_utdf.py`.
- **Evidence:**
  - `studies/phase17/results/phase_f1_lro_residuals.json` (gates, from `phase_f1_lro_residuals.py`);
  - `studies/phase17/results/phase_f1_slope_diagnostic.json` (diagnostic, from `phase_f1_slope_diagnostic.py`).
- **Data:**
  - PDS LRO-L-RSS-1-TRACKING-V1.0, `trk/lro_no_01/2009260` (USN S-band UTDF, 2009-09-17);
  - reference orbit `lro_no_01_grgm900c_l600.bsp` (LOLA team, GEODYN, GRGM900C).
  - Kept outside the repository in `<workspace>/lro_data`.

## 0. Final status

```
GATES       R1 decode PASS (146 files, 7 711 records) | R2 coverage PASS | R3 interpolation PASS (8.6e-5 m)
            R4 light time PASS (0 failures in 5 variants x 2 473 ranges)
H1          NOT MET as computed: de-biased RMS 2.4-3.1 m per pass (limit 3 m), driven by a common slope;
            with the slope (or a time offset) removed the RMS is 0.50-0.85 m
H2          MET: media align the per-pass mean residuals of one station (spread 12.5 m without, 3.3 m with)
H3          MET: the mean residual is a per-station constant (uncalibrated station + transponder delay)
FINDING     a common signature on all 6 passes / 3 stations: dt = -3.2 ms (-2.7..-3.8) as a time-tag offset,
            or equivalently ~5 m of along-track error in the reference orbit; one day cannot separate them
```

## 1. Data

| | |
|---|---|
| range records with the R-valid flag | 2 473 |
| Dongara (PAD 103) | 1 690 |
| South Point (PAD 105) | 378 |
| Kiruna (PAD 126) | 405 |
| PAD 119 | excluded: not in the archive's station table |
| passes | 6, each ~45 min |
| elevations | 6°–50° |

## 2. Residuals (observed − computed), M_FULL

M_FULL is: DSN_C2 light time, DE440, high-precision Earth PCK, solid tide, Niell troposphere with
standard-atmosphere ZHD, and S-band climatological ionosphere.

| PAD | n | elevation | mean [m] | de-biased RMS [m] | detrended RMS [m] | slope [m/h] |
|---|---|---|---|---|---|---|
| 103 | 432 | 35–42° | 490.78 | 2.42 | 0.85 | −9.8 |
| 103 | 448 | 45–51° | 489.54 | 2.99 | 0.70 | −12.1 |
| 103 | 420 | 27–36° | 488.50 | 2.79 | 0.68 | −12.0 |
| 103 | 390 | 6–15° | 487.81 | 3.08 | 0.50 | −14.5 |
| 105 | 378 | 6–15° | 476.89 | 2.51 | 0.62 | −12.0 |
| 126 | 405 | 22–24° | 474.51 | 2.85 | 0.65 | −12.8 |

- **Mean ~475–491 m.** This is the raw RTLT's uncalibrated round-trip delay: 3.17–3.27 µs of station plus
  transponder delay. It is constant per station: Dongara's four passes agree to 3.0 m, and Kiruna and South Point
  sit 13–16 m lower. The archive does not carry the calibration, so a per-station bias is a required solve-for
  in F2.
- **Detrended scatter 0.50–0.85 m.** This is the level at which the whole chain agrees with NASA's orbit inside
  a pass. The chain covers UTC → TDB, station frames, SSB light time with Shapiro, media and DE440.

## 3. Model variants (per-pass mean / detrended RMS)

| variant | Dongara means [m] (4 passes) | spread | low-elevation passes, detrended RMS |
|---|---|---|---|
| M_FULL | 487.8 – 490.8 | **3.0 m** | 0.50 / 0.62 m |
| NO_MEDIA | 495.4 – 507.9 | 12.5 m | 1.01 / 1.16 m |
| NO_TIDE | as M_FULL to ≤ 0.16 m | | as M_FULL |
| MOON_NEWTONIAN (pre-C1 default, no media, no tide) | 504.5 – 517.1 | 12.6 m | 1.02 / 1.17 m |
| DE421 | M_FULL − 0.24 m (constant) | 3.0 m | as M_FULL |

Readings:
- **Media (C5-I) is confirmed on real data.** Without it, the 6–15° pass sits 10–20 m higher and carries an
  elevation-shaped residual (positive slope, detrended RMS doubled). With it, every pass of a station shares
  one delay to 3 m, and the low-elevation passes are the cleanest (0.50 m).
- The solid tide (cm level) and DE421 vs DE440 are below this day's resolution. That is expected: the C3/C4
  surveys found them material in OD (1–5σ), not in single-pass residuals.

## 4. The common slope (diagnostic, `phase_f1_slope_diagnostic.py`)

Every pass has a slope of −10 to −15 m/h. Within one pass the slope is almost collinear with the line-of-sight
range rate ρ̇, which sweeps from −1.5 to +1.5 km/s. Fits:

| PAD | slope fit RMS | fit bias + Δt·ρ̇: Δt [ms], RMS |
|---|---|---|
| 103 | 0.85 m | −2.74, 0.72 m |
| 103 | 0.70 | −3.26, 0.77 |
| 103 | 0.68 | −3.14, 0.68 |
| 103 | 0.50 | −3.81, 0.67 |
| 105 | 0.62 | −3.19, 0.68 |
| 126 | 0.65 | −3.17, 0.61 |

- A **single Δt ≈ −3.2 ms** explains all six passes at three stations equally well.
- Two candidates remain:
  1. a fixed time-tag convention in the USN range data. The GN handbook only says "time-tagged at receive time";
  2. about 5 m of along-track error in the reference orbit on this day (Δt × 1.6 km/s).
- One day cannot separate them. A time-tag offset is the same on every day, while an orbit along-track error
  changes along the 38-day arc. A second day in the same SPK arc decides it. That is F1b: one more TRK day,
  ~1 MB, reported before download.

## 5. Consequences

- The measurement chain built in C1–C5 reproduces real LRO S-band range against NASA's precision orbit to
  0.5–0.85 m within a pass, after removing a per-station constant and a 3 ms / 5 m common signature. Both will be
  estimated parameters in the real-data fit (F2).
- For F2 (OD on real data), the solve-for set must include per-station range biases and, unless F1b attributes
  it, a time-tag bias.

## 6. Limits

- One day, three USN stations, range only. Doppler (cumulative bias counts) is not yet used.
- Station coordinates come from the archive's WGS-84 table, used as ITRF93; the antenna reference point
  accuracy is unknown.
- The ionosphere is climatological, at 2091.32 / 2271.125 MHz (downlink from public sources, uplink from the
  240/221 turnaround ratio).
- ZWD is a fixed 0.15 m; no weather files were used (the archive has WEA files).
