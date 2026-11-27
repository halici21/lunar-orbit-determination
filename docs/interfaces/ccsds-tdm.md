# CCSDS Tracking Data Message (TDM)

V&V programme F6 WP6.1 (REQ-IO-001). The library reads and writes the KVN form of the
CCSDS Tracking Data Message, CCSDS 503.0-B-2 (June 2020, Cor. 1 October 2021), and
turns TDM records into the estimator's observation rows. XML TDMs are not supported.

| Module | Purpose |
|---|---|
| [`lunar_od.ccsds_tdm`](../../src/lunar_od/ccsds_tdm.py) | KVN parser/writer, message model, time-tag conversion |
| [`lunar_od.tdm_ingestion`](../../src/lunar_od/tdm_ingestion.py) | TDM → `IngestedMeasurements` rows and back |

`measurement_ingestion.py` is byte-frozen, so the TDM entry point is a separate module. It
returns the same `IngestedMeasurements` as `read_measurement_csv`.

## Reading and writing

```python
from lunar_od.ccsds_tdm import read_tdm, write_tdm, format_tdm

tdm = read_tdm("pass.tdm")             # TrackingDataMessage
segment = tdm.segments[0]
segment.time_system, segment.participants, segment.path
[r.value for r in segment.records_of("RANGE")]   # in RANGE_UNITS (km by default)
write_tdm(tdm, "copy.tdm")             # parsed back before it is returned
```

Values keep the units of the standard's table 3-5: RANGE in `RANGE_UNITS`, ANGLE_1/2 in
deg, DOPPLER_INSTANTANEOUS/INTEGRATED in km/s, frequencies in Hz, TRANSMIT_FREQ_RATE in Hz/s.
Measurement text is kept verbatim, so a file that is read and written again keeps every
record. Phase counts keep all their digits (4.3.11). A number the writer creates is the
shortest text that reads back to the same double, limited to 16 digits (4.3.4/4.3.5).

### Strict validation (fails closed)

`TDMFormatError` names the offending line for:

- a keyword outside the header, metadata or data tables (3-2, 3-3, 3-5), or one that
  refers to an undefined participant (`RECEIVE_FREQ_3` with two participants);
- header and metadata keywords out of the table order, or repeated;
- enumerated values that are not normative (`RANGE_UNITS`, `ANGLE_TYPE`, `MODE`, ...), and
  `TIME_SYSTEM` values outside the SANA registry;
- any token after the measurement: KVN TDMs never show units (4.4), so `384400.5 [km]`
  is refused;
- malformed time tags (both calendar and day-of-year forms are read), a `Z` suffix outside
  UTC, and a leap second outside UTC;
- numbers that are not integer, fixed-point or floating-point literals; NaN/Inf; more
  than 16 digits;
- a repeated keyword/time tag (3.4.11), records of one keyword that are not in time order
  (3.4.10), COMMENT lines outside their places (4.5.2), tabs and non-ASCII characters, and
  lines longer than 254 characters;
- missing mandatory items, PATH/MODE inconsistencies, `CORRECTION_*` without
  `CORRECTIONS_APPLIED`, and records missing from `DATA_TYPES` when that keyword is present.

## Time systems

`tdm_epochs_to_tdb_seconds(epochs, time_system, et0_s, site=None)` returns TDB seconds
past `et0` (itself TDB seconds past J2000):

| TIME_SYSTEM | Conversion |
|---|---|
| UTC | `station_time.utc_to_tdb_seconds` (needs the leapseconds kernel loaded) |
| TAI, GPS | TT = TAI + 32.184 s; GPS = TAI − 19 s |
| TT | TDB = TT + `station_time.tdb_minus_tt` (geocentric or at `site`), solved at the TDB epoch |
| TDB | used as is |

Any other system (UT1, TCB, SCLK, ...) raises an error.

## Ingestion into observation rows

```python
from lunar_od.tdm_ingestion import read_measurement_tdm

obs = read_measurement_tdm("pass.tdm", "position", station_names=("DSS-25", "DSS-63"),
                           et0_s=et0, two_way_convention="round_trip")
obs.obs_data   # [t_s, range_m, az_rad, el_rad, station_id, time_index]
```

The range-rate layout uses RANGE, DOPPLER_INSTANTANEOUS, ANGLE_1 and ANGLE_2. Records
of several segments are joined per station and epoch. `time_index` counts the file's
sorted unique epochs, so build the pass geometry on `np.unique(obs_data[:, 0])`.

The mapping refuses anything the rows cannot represent: unused data keywords,
`MODE = SINGLE_DIFF`, three-way and longer paths, a receiver outside `station_names`,
`TIMETAG_REF = TRANSMIT`, `RANGE_UNITS` other than km, a non-zero `RANGE_MODULUS`,
non-AZEL angles, non-zero transmit/receive delays, corrections that were not applied,
`DATA_QUALITY = DEGRADED`, more than one spacecraft, and incomplete epochs.

For two-way data the standard leaves it to the ICD whether RANGE and Doppler are the
round-trip value or half of it (3.5.2.3, 3.5.2.7). `two_way_convention` must say which
(`"round_trip"` values are halved; `"one_way_equivalent"` values are used as they are). It is
refused for one-way paths.

`ingested_to_tdm` writes the rows as a version-2.0 TDM with one segment per station:
TDB time tags, one-way `PATH = 1,2` from the spacecraft to the station, km and deg.
Reading that TDM back gives the same rows.

## Verification

[`tests/test_ccsds_tdm_wp6_1.py`](../../tests/test_ccsds_tdm_wp6_1.py) uses four example
TDMs in [`tests/fixtures/ccsds_tdm`](../../tests/fixtures/ccsds_tdm/two_way_range_angles_utc.kvn).
They were written for this project in the style of the standard's annex E (range, angles
and Doppler; two-way frequencies with phase counts; one-way TDB; media and weather). The
tests cover:

- round trips: parse → write → parse is the same message, and writing is idempotent;
- more than forty malformed-message cases and the ingestion refusals;
- time systems: TT and UTC against SPICE (`unitim`, `str2et`) to 5e-5 s (ERFA vs SPICE
  TDB − TT), and TAI/GPS/TT identities to 1e-9 s;
- ingestion against `read_measurement_csv` on the same rows, and rows → TDM → rows.

An Orekit cross-read (`TdmParser` on the written files) runs when `orekit_jpype` and
its orekit-data are installed. It is skipped otherwise, with that reason.
