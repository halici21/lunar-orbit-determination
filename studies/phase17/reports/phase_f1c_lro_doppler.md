# Phase F1c — Real LRO two-way S-band Doppler against the LOLA precision orbit

- **Date:** 2026-10-24
- **Predeclaration:** `studies/phase17/configs/phase_f1c_lro_doppler_predeclaration.json` (`1d495a4`), before any
  residual.
- **Evidence:**
  - `studies/phase17/results/phase_f1c_lro_doppler.json` (gates, from `phase_f1c_lro_doppler.py`);
  - `studies/phase17/results/phase_f1c_day285_diagnostic.json` (post hoc, not predeclared, from
    `phase_f1c_day285_diagnostic.py`).
- **Reader fix found here** (`8187d44`): the UTDF transmit frequency is a binary count of 10 Hz, as 453-HDBK-GN
  says. The PDS `lro_trk.fmt` labels it IEEE_REAL, which decodes the LRO records to ~1.9e-31 instead of
  2091.414 MHz.

## 0. Final status

```
GATES        R1 decode PASS (2 416 intervals, dN > 0, one f_t = 2 091 414 110 Hz at every station)
             R2 sign/scale PASS (residual ~11 mm/s vs ~850 m/s signal) | R3 light time PASS
H1, H2       NOT MET as computed: a common offset of -10.7..-11.7 mm/s on every pass of all three stations
             (RMS 11.0-12.0 mm/s, limit 5); with the offset removed the scatter is 1.5-3.3 mm/s
H3           MET: the media phase delay removes the low-elevation trends (14-17 -> 4-6 mm/s/h)
FINDING      the offset is a constant measurement bias: the same on 2009-285 (Dongara -10.4..-12.2 mm/s, scatter
             0.9-1.1 mm/s); it equals a 7.4e-11 relative frequency offset (~0.16 Hz at the uplink), i.e. a
             Doppler bias parameter for OD, not an orbit signature (an orbit velocity error would change sign
             with the line of sight)
```

## 1. Observable

- Average two-way range rate over each interval between consecutive valid records of one station (5 s here):
  R′ = −c·(ΔN/ΔT − 2.4·10⁸) / (2·f_t·K·M), with K = 240/221 and M = 1000.
- It is compared with `relativistic_light_time.counted_doppler` at the interval midpoint, using F1's M_FULL model
  with the media phase delay at f_t and f_t·240/221.

## 2. Residuals (observed − computed), 2009-260

| PAD | n | mean [mm/s] | RMS [mm/s] | std [mm/s] | slope [mm/s/h] | NO_MEDIA std / slope |
|---|---|---|---|---|---|---|
| 103 | 422 | −10.73 | 11.04 | 2.59 | 7.1 | 2.55 / 6.9 |
| 103 | 438 | −11.09 | 11.58 | 3.34 | 5.3 | 3.34 / 5.4 |
| 103 | 410 | −11.04 | 11.27 | 2.27 | 6.2 | 2.30 / 6.4 |
| 103 (6–15° el.) | 381 | −11.61 | 11.77 | 1.95 | 4.4 | **3.56 / 14.3** |
| 105 (6–15° el.) | 369 | −11.74 | 11.95 | 2.23 | 6.4 | **3.98 / 17.0** |
| 126 | 396 | −11.03 | 11.13 | 1.52 | 6.4 | 1.55 / 6.6 |

## 3. The common offset

- −11.1 mm/s, the same at Dongara, South Point and Kiruna, and again on 2009-285 at Dongara: −11.1, −10.4, −12.1,
  −12.2 mm/s, with scatter ~1 mm/s.
- A reference-orbit velocity error projected on the line of sight would change sign as ρ̇ sweeps from −1.5 to
  +1.5 km/s within each pass, so it cannot produce a constant.
- With f_rx = K′·f_t·(1 − 2ρ̇/c), the observed R′ is ρ̇·K′/K − c(K′ − K)/(2K). A constant −11.1 mm/s therefore
  means a relative offset ε = 7.4·10⁻¹¹ between the effective turnaround ratio or frequency reference and the
  nominal 240/221 used in the conversion: about 0.16 Hz at the uplink.
- Such a Doppler bias is a standard OD parameter. The archive does not document it.

## 4. Other findings

- **PAD 129 (Weilheim WU2S, 2009-285):** Doppler is unusable (residual mean 1.4·10⁷ mm/s). The count series is
  not continuous in that station's mode (counter reset or rollover), so its Doppler is excluded.
- **Slopes** of 4–7 mm/s/h remain after the media correction. They are of the size and sign expected from the
  reference orbit's along-track error that F1b found (~5–7 m), and are left to the OD.
- **Scatter after the offset:** 1.5–3.3 mm/s on 2009-260 and ~1 mm/s on 2009-285. These are the Doppler weights
  for a combined fit.

## 5. Consequences for real-data OD

- Doppler gives the along-track and cross-track information the F2c range-only fit lacked: F2c resolved radial
  to 3.2 m but along/cross-track only to ~90 m.
- A combined range + Doppler fit needs:
  1. a Doppler bias (one global, or per station);
  2. Doppler weights of ~2 mm/s;
  3. F2c's degree-200 gravity.
