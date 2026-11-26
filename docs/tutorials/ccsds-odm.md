# CCSDS orbit data export (OEM / OPM)

`lunar_od.ccsds_odm` writes estimated trajectories and covariances as CCSDS
Orbit Data Messages (CCSDS 502.0-B-3) in Keyword = Value Notation (KVN):

- **OEM** (Orbit Ephemeris Message): one or more segments of time-tagged
  states, each with optional covariance blocks.
- **OPM** (Orbit Parameter Message): one state vector and its covariance.

A minimal reader for the same subset (`read_oem`, `read_opm`) exists for
round-trip checks and inspection. It is not a general ODM parser. The OCM
(Orbit Comprehensive Message) is not implemented.

## Library-to-CCSDS contract

| Item | Library | Written as |
|---|---|---|
| Origin | Moon centre | `CENTER_NAME = MOON` |
| Axes | SPICE `J2000` (the code calls this frame "MCI") | `REF_FRAME = EME2000` (SPICE `J2000` is EME2000) |
| Epochs | SPICE ET: TDB seconds past J2000.0 (2000-01-01T12:00:00 TDB) | `TIME_SYSTEM = TDB`, calendar `YYYY-MM-DDThh:mm:ss.ffffff` |
| State units | m, m/s | km, km/s |
| Covariance units | m², m²/s, m²/s² | km\*\*2, km\*\*2/s, km\*\*2/s\*\*2 |
| Covariance layout | 6×6 symmetric | lower triangle, row by row, order X, Y, Z, X_DOT, Y_DOT, Z_DOT |

Every file repeats this contract in `COMMENT` lines. The writer converts no
frames. It accepts `REF_FRAME` values `EME2000` and `ICRF` only, and `ICRF`
only for states you have already rotated. Never write `REF_FRAME = MCI`: in
the SANA registry `MCI` is the *Mars*-centred inertial frame.

Scenario and filter times are elapsed seconds. Add the scenario's `et0_s` to
get ET: `epoch_et_s = et0_s + t_s`.

## Precision

- Each state and covariance value is written with 17 significant digits, so
  the kilometre double read back equals the kilometre double written. Back in
  metres the only change is the ×1000 scaling, at most one unit in the last
  place.
- An epoch is converted exactly from its binary value with rational
  arithmetic. It is rounded once to `epoch_decimals` fractional digits
  (default 6, at most 9), so it reads back within half a unit of the last
  written digit. TDB has no leap seconds.
- `creation_date` defaults to the current UTC time. Pass a fixed value for
  byte-reproducible files.

## Validation on write

The writer raises `OdmFormatError` for:

- states that are not N×6 or are not finite;
- epochs that are not strictly increasing within a segment;
- a covariance that is not 6×6, not finite, not symmetric (relative tolerance
  1e-9) or has a negative variance;
- a covariance epoch outside its segment;
- a frame or centre outside the supported lists;
- an `INTERPOLATION` without a degree, or a degree without an `INTERPOLATION`;
- inconsistent useable start/stop times;
- multi-line names or comments;
- an unsupported message version (`2.0` and `3.0` are supported). `MESSAGE_ID`
  requires `3.0`.

## Usage

```python
from lunar_od.ccsds_odm import OemCovariance, OemSegment, write_oem, write_opm

segment = OemSegment(epochs_et_s, states_m, "LRO", "2009-031A",
                     covariances=(OemCovariance(epochs_et_s[0], cov0_m),))
write_oem("lro.oem", segment, originator="LUNAR_OD")
write_opm("lro.opm", epochs_et_s[0], states_m[0], object_name="LRO",
          object_id="2009-031A", originator="LUNAR_OD", covariance_m=cov0_m)
```

`oem_segment_from_filter_result(result, et0_s, ...)` builds a segment from a
UKF or EKF result (`t_update_s`, `state_estimates`, `covariances`). It exports
only the leading 6×6 position/velocity block of an augmented state.

The [export example](../../examples/export_ccsds_odm.py) writes an OEM and an
OPM, either from an `.npz` (`epochs_et_s`, `states_m`, optional `covariances_m`)
or with `--demo`, which uses a closed-form circular orbit.

## Verification (REQ-IO-002)

[`tests/test_ccsds_odm_wp6_2.py`](../../tests/test_ccsds_odm_wp6_2.py) checks:

- round-trip bit-exactness of the kilometre values;
- the epoch calendar against hand-computed values, including rounding carry
  across midnight;
- covariance symmetry and the standard's lower-triangle ordering;
- the metadata and units;
- determinism of the output;
- malformed inputs and files.

When `orekit_jpype` and a JVM are installed, Orekit's own OEM and OPM parsers
read the written files and must agree with this module:

- positions and velocities: difference exactly 0;
- epochs, through Orekit's TDB scale: within half a microsecond;
- covariances: to 1e-15 relative.

These Orekit tests use a stub Moon body, because parsing data lines never
queries a body ephemeris, so they need no JPL files. Without Orekit the tests
skip and say why.

Desktop export is a follow-up and is not wired into the application yet.
