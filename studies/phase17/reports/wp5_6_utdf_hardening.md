# F5 WP5.6: UTDF reader hardening (RD-11, REQ-DATA-005)

- **Date:** 2026-11-27
- **Predeclaration:** `studies/phase17/configs/wp5_6_utdf_hardening_predeclaration.json`, committed and pushed in
  `8598457` before any WP5.6 code, test or gate value. One look at the data came first: a byte-field survey of
  bytes 45–54 and 73–75 on the development days. The predeclaration records it.
- **Failing tests first:** `9ed5df5` (tests and fixtures); implementation as predeclared `7695605`; exploratory
  amendment A1 in the next commit.
- **Amendments:** two, both after a result (`docs/assurance/amendment-log.md`, 2026-11-27): the U_G1 time-window
  clause and A1, the discontinuity predictor.
- **Code:** `src/lunar_od/utdf.py`, additive. Tests in `tests/test_utdf.py`. Real fixtures in
  `tests/fixtures/utdf/`: two archived files of 3600 B each, with their SHA-256 in `provenance.json`.
- **Impact script:** `studies/phase17/scripts/wp5_6_utdf_impact.py`. Results:
  - `studies/phase17/results/wp5_6_utdf_impact.json` (A1 detector);
  - `studies/phase17/results/wp5_6_utdf_impact_as_predeclared.json` (predeclared detector).
- **Data read:** development days 2009260–263 and 2009285 only, each through `require_readable_day`. The
  hold-out days 264–267 and the excluded day 268 were never opened.
- **Study scripts and results changed:** none.

## Status

| gate | as computed |
|---|---|
| U_G1 real archived record | **FAIL as predeclared** on one clause; PASS after the amendment (see below) |
| U_G2 real F2n slip flagged | PASS: deviation 1.0298 m/s against F2n's 1.024 (0.6 %), no other code but GAP |
| U_G3 real control file | PASS: 0 codes |
| U_G4 rollover / frozen / reversal / out of range | PASS |
| U_G5 pad_xmit == pad_rcv, PAD vs file, link type | PASS |
| U_G6 leap-second crossing | PASS, including (d), the cross-check against naif0012 (it ran; no skip) |
| U_G7(a) record-derived f_T, K, M | PASS |
| U_G7(b) equality with F1c's expression | **FAIL as computed**: 10 826 of 10 828 intervals exact (0 ulp), 2 differ; both are real counter rollovers, which the new reader unwraps |
| U_G8(a) decode identity with main's reader | PASS: 22 743 records in 410 files, every field equal |
| U_G8(b) no study file changed | PASS |
| U_G8(c) impact (information) | see §3 |
| U_G9 regression | see §5 |

## 1. What changed in the reader

All changes are additive. `decode_record` and `read_utdf` return the same values in every existing field
(U_G8(a)). The only behaviour change: `UTDFRecord.utc` raises on a year-end leap-second record (23:59:60), which
`datetime` cannot represent. It used to return the next instant silently. No archived 2009 record is affected.

- **Raw fields:** byte 52 (`band_byte`) and bytes 53–54 (`tracking_mode`).
- **Properties:** `frequency_band`, `data_type`, `link_type` (mode bits 6–5: 1-, 2- or 3-way), and
  `turnaround_ratio` / `doppler_multiplier`. K and M follow the band, per 453-HDBK-GN 4.2.2.4(c):
  - S-band: 240/221 and 1000;
  - VHF: 1 and 1000;
  - X-band: 880/749 and 250;
  - any other band: `None`.
- **Time properties:**
  - `utc_iso`, which shows `:60` at a year-end leap second;
  - `tai_microseconds_past_j2000` / `tai_seconds_past_j2000`, read from a built-in leap-second table (`LEAP_TABLE`,
    1972–2017, 28 steps). The table is declared valid until 2026-07-01; a later epoch needs a caller-supplied
    `tai_minus_utc`.
- **`record_reason_codes(record, file_pad=None)`:** returns `PAD_XMIT_RCV_MISMATCH`, `PAD_FILE_MISMATCH`,
  `NOT_TWO_WAY`, `BAND_UNSUPPORTED`, `MICROSECONDS_OUT_OF_RANGE` or `SECONDS_OUT_OF_YEAR`.
- **`check_doppler_counts(records, ...)`:** returns one `DopplerInterval` per consecutive pair of Doppler-valid
  records.
  - Each interval carries the elapsed TAI time, the count difference (unwrapped across 2^48), the rate, and the
    handbook range rate with the record's f_T, K and M.
  - Interval codes: `TIME_NOT_INCREASING`, `GAP` (> 60 s), `FREQUENCY_CHANGED`, `COUNT_FROZEN`, `COUNT_ROLLOVER`,
    `COUNT_REVERSAL`, `RATE_OUT_OF_RANGE` (> 3000 m/s) and `RATE_DISCONTINUITY`.
  - Flagged intervals are returned, not dropped.

## 2. Gates

**U_G1 (real record).** Both fixtures decode, and their SHA-256 matches the archive. Every record checks out:
- year 2009, SIC 59, VID 1;
- pad_xmit = pad_rcv = 103 = the PAD in the file name;
- S-band, two-way, no record code;
- f_T = 2 091 414 110 Hz;
- one-way range of the range-valid records between 3.5e8 and 4.1e8 m.

The predeclared clause "tags inside [close time − 5 min, close time] of the file name" **failed**. The last
record of `..._0722.trk` is 07:22:20. The HHMM in the name is the last record's minute, truncated, not a close
time. The window is now [HHMM − 5 min, HHMM + 1 min); this change is in the amendment log. This is a fault in an
assumed archive convention, not a reader defect.

**U_G2 / U_G3 (real slip, real control).** On `..._1105.trk`, the interval starting 11:03:15 UTC is the only
`RATE_DISCONTINUITY`. Its deviation, 1.0298 m/s, matches F2n's 1.024 m/s residual against LOLA, from counts
alone. The other intervals of the pass deviate by at most 11.4 mm/s (the degraded pass F2n described). The
control file has no code. Both gates pass with the predeclared detector and with A1.

**U_G4 / U_G5 (synthetic, built from real records).**
- A 2^48 wrap placed inside the control stream is `COUNT_ROLLOVER` on exactly that interval. Its dn and range
  rate equal the unwrapped ones bit for bit.
- A counter frozen with the valid flag set is `COUNT_FROZEN`.
- A decrease that no wrap explains is `COUNT_REVERSAL`.
- A +1e9 count jump is `RATE_OUT_OF_RANGE` on that interval only.
- pad_xmit ≠ pad_rcv, including pad_xmit = 0 (the handbook's S-band three-way value), taints both intervals
  touching the record. Link type 3 is `NOT_TWO_WAY`.

**U_G6 (leap second).**
- Year end 2016: the record with seconds of year = 366 × 86400 reads `2016-12-31T23:59:60.000000`. TAI steps are
  5, 5, 1, 5, 5 s. Without that record, 23:59:55 → 00:00:00 is 6 s, while the `datetime` difference is 5 s; the
  range rate uses 6 s.
- 2015-06-30: the crossing interval is 6 s, and a repeated 00:00:00 tag is `TIME_NOT_INCREASING`.
- Without a leap second, dt equals the `datetime` difference exactly.
- The built-in table equals naif0012's DELTET/DELTA_AT at every step. SPICE `str2et` plus `unitim` gives the
  same TAI for the 23:59:60.5 record within 1e-6 s.
- An epoch after the table's horizon raises unless `tai_minus_utc` is given.

**U_G7 (frequencies from the record).**
- (a) S, VHF and X-band records give the handbook range rate with their own K and M. A C-band record is
  `BAND_UNSUPPORTED` with a NaN range rate. A changed f_T is `FREQUENCY_CHANGED`.
- (b) **FAIL as computed:** on the 10 828 intervals F1c forms on the five development days, 10 826 equal F1c's
  expression exactly (0 ulp). The 2 that differ are real 48-bit rollovers on PAD 103:
  - 2009-09-18 06:03:45: count 281 474 612 798 041 → 933 120 434;
  - 2009-09-20 07:06:05: count 281 474 592 127 439 → 920 927 179.

  On each, a pass starts just below 2^48 and wraps within its first minute. F1c's expression gives an impossible
  rate there; the new reader unwraps and flags it. The predeclaration did not expect rollovers in this data.
  The record-derived constants equal the studies' constants on every other interval (every 2009 record is
  S-band, and 2009 has no leap second).

## 3. Impact on the LRO studies (U_G8(c), information)

These are the Doppler intervals F1c forms (doppler_valid, PADs 103/105/126, 0 < dt ≤ 60 s), checked per PAD.

| day | intervals | COUNT_FROZEN | COUNT_ROLLOVER | RATE_OUT_OF_RANGE | RATE_DISCONTINUITY as predeclared | RATE_DISCONTINUITY A1 |
|---|---|---|---|---|---|---|
| 2009260 | 2416 | 0 | 0 | 0 | 0 | 0 |
| 2009261 | 2225 | 482 | 1 | 0 | 5 | 0 |
| 2009262 | 2693 | 0 | 0 | 0 | 19 | 2 |
| 2009263 | 2442 | 481 | 1 | 2 | 1 | 1 |
| 2009285 | 1052 | 0 | 0 | 0 | 0 | 0 |

**Frozen counter (RD-11 confirmed on real data).** 963 intervals carry the Doppler-valid flag while the counter
reads 0: validity byte 0x09 (Doppler valid, range invalid). They come in two stretches:
- PAD 105 on 2009-09-18, 02:17–03:11;
- PAD 103 on 2009-09-20, 08:57–09:51 (three stretches between the 80 s gaps).

The flag alone would have admitted them.

**No study result moves.** F2, F2l, F2n and F2r drop every interval with |ṙ| > 3000 m/s (F2e A1,
`MAX_DOPPLER_MPS`). On every development day, that filter drops exactly the 967 intervals the new reader flags
with a count code and keeps every other interval (`study_filter_vs_codes` in the result). The studies were
protected, but by a plausibility threshold that gave no reason. F1c itself ran on day 260, which has no flag.
No study script calls the new checks, so nothing was re-frozen.

**Discontinuities.**
- As predeclared, 25 intervals were flagged. Twenty-three were false alarms: every one was a 50–55 s interval
  bridging a short gap at PADs 126 (261, 262) and 105 (262).
- The predeclared predictor fitted interval averages as point values at their mid-times. Over a 50 s interval,
  the average range rate differs from the mid-time value by ρ⃛T²/24 ≈ 0.15 m/s for LRO's ~1.4e-3 m/s³ range
  jerk. That is physics, not a counter fault.
- A1 fits the cubic to interval means and predicts the interval's own mean. Threshold, window and neighbour
  count are unchanged; the change is exploratory and in the amendment log.
- With A1, three flags remain, and F2o's residual editor had already edited all three:
  - F2n's slip (1.0298 m/s);
  - the adjacent pair PAD 105 2009-09-19 03:31:40 and 03:31:45 (+22.5 and +22.0 mm/s).
- `test_mixed_interval_lengths_are_not_discontinuities` covers A1. It fails on the predeclared detector and passes
  on A1; it also requires that a planted 100 mm/s slip is still caught.

## 4. Conventions and limits (also in `docs/assurance/assumptions-and-limits.md`)

- **Time tags.** Bytes 11–18 are read as a UTC calendar count. A year-end leap second is distinct
  (seconds of year = year length).
- **Mid-year leap second.** At a mid-year (June 30) leap second, 23:59:60 cannot be told from 00:00:00 July 1.
  The handbook does not say how a leap second is counted, and no archived record crosses one (LRO 2009). The
  convention is declared, not verified on real data.
- **Leap-second table horizon.** The built-in table holds until 2026-07-01. Later epochs need
  `tai_minus_utc`, for example `station_time.tai_minus_utc` with the leapseconds kernel loaded.
- **Discontinuity threshold.** The 20 mm/s threshold makes this a counter-integrity check. Smaller outliers stay
  F2o's job (residual editing).
- **Sample rate.** Bytes 53–54 read a 1 s sample rate on PADs 103–129, while records are 5 s apart. The field is
  decoded but not checked.
- **For WP5.4.** The frozen pipeline should take its Doppler intervals from `check_doppler_counts` and use only
  `usable` ones (A1 detector). Its predeclaration should say so.

## 5. Regression

Full `pytest` (offscreen Qt): see the PR description for the junit totals of the final commit.
`tools/check_repository.py`: see the PR. `utdf.py` is not a protected file
(`examples/r2_measurement_fidelity_validation.py` does not list it), so no re-freeze table was added.
