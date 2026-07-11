# Frame Transformation Contract

Status: documentation of the **existing, verified** production behavior
(frame audit, 2026-07-11). This document changes no code. Locked by
`tests/test_frame_transformations.py` (SPICE-free) and
`tests/test_frame_spice_validation.py` (kernel-gated).

Audit result: **no frame-correctness bug was found in the production one-way
measurement chain.** Known approximations and open items are listed in
"Known limitations".

## 1. Matrix and vector convention (repository-wide)

Vectors are NumPy `(3,)` / states `(6,)` column-style arrays,
`[x, y, z, vx, vy, vz]`, SI units (m, m/s, rad, s). Every frame
transformation is applied by **left multiplication with a
target-from-source operator**:

```text
r_B = C_{B<-A}(t) r_A            (3x3, orientation only)
x_B = X_{B<-A}(t) x_A            (6x6, position + velocity)
```

SPICE mapping (argument order is FROM, TO — the returned matrix is
target-from-source):

```text
C_{B<-A} = spice.pxform(A, B, et)
X_{B<-A} = spice.sxform(A, B, et)
```

Properties relied on and tested: `C^T C = I`, `det C = +1`,
`C_{A<-B} = C_{B<-A}^T`. The 6x6 state transform has the block structure

```text
X_{B<-A} = [ C      0 ]
           [ Cdot   C ]
```

and its inverse is **not** its transpose; the code inverts it with
`np.linalg.solve(X, x)` (e.g. `measurements.py`
`_station_position_mci_at_receive_epoch`, `radiometrics.py`
`_station_state_mci`). SPICE matrices are used directly and never
transposed manually. Batched forms use
`np.einsum('nij,nj->ni', C, v)` — still `C @ v` per row.

All transformations are **passive** (coordinate re-expression of an
unchanged physical vector) with one deliberate exception:
`measurements.apply_stellar_aberration` performs an **active** Rodrigues
rotation of the line-of-sight vector within the J2000 basis (physics, not a
frame change).

## 2. Origin versus orientation

These concepts are kept strictly separate in the code and must stay so:

- **Frame orientation**: the axis triad a vector is resolved in
  (J2000, ITRF93, SEZ, MOON_PA_*). Changed only by `C`/`X` multiplication.
- **Frame origin / state center**: the point positions are measured from
  (Moon, Earth, station, SSB). Changed only by **vector addition of an
  ephemeris offset in matching axes** — never by a rotation matrix.
- **Origin translation** examples (all in J2000-aligned axes):
  `r_sc/Earth = r_sc/Moon - r_Earth/Moon` (spacecraft re-centering,
  `measurements.py` `generate_position_measurements`), and
  `r_station/Moon = r_Earth/Moon + r_station/Earth`
  (`_station_position_mci_at_receive_epoch`).
- **Position rotation**: `rho_F = C_{F<-I} rho_I` — valid only after both
  endpoints share one origin.
- **Full state transformation**: `X_{B<-A}` (rotating frames) — the `Cdot`
  block supplies the frame-rotation velocity term; a position-only rotation
  of a velocity vector is wrong whenever the frame rotates.
- **Local topocentric transformation**: `C_{SEZ<-ECEF}(lat, lon)` — station
  attitude constants; origin shift to the station is the ECEF subtraction of
  `station.r_ecef_m` done *before* this rotation.

## 3. Frame inventory

| Name in code | Origin | Orientation | Rotating | SPICE equivalent |
|---|---|---|---|---|
| `MCI` (`*_mci`) | Moon center | **Moon-centered, J2000-aligned inertial coordinates** | no | orientation `J2000`; not a SPICE frame name |
| `ECI` (`dr_eci`, `state_sat_eci`) | Earth center | J2000 | no | `J2000` @ Earth |
| `J2000` | per SPICE call (Earth, Moon, SSB) | SPICE `J2000` (ICRF-aligned) | no | `J2000` |
| `ECEF` (`r_ecef*`) | Earth center | **ITRF93 Cartesian coordinates** — the name `ECEF` in production code always means ITRF93 | yes | `ITRF93` |
| `SEZ` | station | South / East / Zenith (geodetic normal) | station-fixed | none |
| `MOON_PA_DE421` / `MOON_PA_DE440` | Moon center | lunar principal axes | yes | same |
| legacy GST "ECEF" | Earth center | z-rotation by Curtis GST only | yes | none (see §10) |

`J2000` origins must always be distinguished: Moon-centered (`MCI`),
Earth-centered (`ECI`), and SSB-centered (SPICE-like stellar-aberration
observer velocity) representations share axes but not origins.
`IAU_EARTH`, `IAU_MOON`, ENU, and NED are not used.

## 4. Production frame graph (one-way range/az/el)

```text
spacecraft state in Moon-centered J2000-aligned MCI
  -> subtract Earth/Moon ephemeris for Earth-centered J2000 representation
     (origin translation: dr_eci = r_sc/M - r_E/M)
  -> J2000 -> ITRF93 at receive epoch
     (orientation: X = sxform("J2000","ITRF93", et0 + t_r), C = X[:3,:3])
  -> ECEF -> SEZ at the station
     (C_sez_ecef = ecef2sez_dcm(lat, lon); rho_sez = C_sez_ecef @ rho_ecef)
  -> range / azimuth / elevation
```

Station chain (reverse direction):

```text
WGS84 geodetic (lat, lon, h)
  -> geodetic_to_ecef_wgs84 -> ITRF93 Cartesian r_F     (config.Station.r_ecef_m)
  -> [r_F, 0] -> np.linalg.solve(X_{ITRF93<-J2000}, .) -> [r_I, v_I]  (J2000 @ Earth)
  -> + r_Earth/Moon ephemeris -> station state in MCI
```

The station's inertial velocity comes from the `Cdot` block of `sxform`;
applying `pxform` to the zero body-fixed velocity would (wrongly) give zero.
There is no hand-coded `omega x r` in the production chain — the analytic
`v_I ~ omega_E x r_I` is used only as a test oracle.

The finite-difference validation of this velocity uses the `sxform` velocity
as the exact reference and sweeps the position central-difference step. For the
2027 campaign epoch the error is U-shaped: coarse steps are truncation-limited,
the empirical crossover plateau is 10--1 s with its minimum near 3 s, and
sub-second steps are dominated by finite-precision cancellation at the large
absolute SPICE ET. The test therefore accepts the minimum over `{10, 3, 1}` s
against `2e-5 m/s`; it does not apply a plateau tolerance to the 10 s sample or
to the round-off-dominated fine steps.

## 5. Epoch contract (light-time / receive-tagged paths)

| Quantity | Epoch |
|---|---|
| Measurement time tag | receive `t_r` |
| Spacecraft state | **transmit** `t_t = t_r - tau` (cubic-Hermite interpolated) |
| Station position | receive |
| Station velocity | receive |
| Observer velocity (stellar aberration) | receive |
| `J2000 -> ITRF93` transformation | receive |
| SEZ basis | station-fixed constants, applied at receive |
| STM in implicit initial-state Jacobians | `Phi_r(t_t, t_0)` at transmit |

The observation frame is evaluated at receive epoch because the received
signal's direction is resolved against the station's local basis at the
moment of reception, independent of when it left the spacecraft. Evaluating
`C_{F<-I}` at `t_t` instead would rotate the topocentric basis by
`omega_E * tau` (~9.3e-5 rad at lunar light time) — the kernel-gated epoch
mutation test demonstrates this difference is measurable.

Instantaneous (non-light-time) paths evaluate everything at the tag time by
construction. Metadata already declares the epochs
(`station_position_epoch`, `frame_transformation_epoch`,
`spacecraft_position_epoch`, `station_velocity_model: "sxform"`).

## 6. Time scales

- `et0` is SPICE ET (TDB seconds past J2000). Sources:
  `et0 = spice.str2et(epoch_utc)` (UTC string; leap seconds via
  `naif0012.tls`) in campaign/desktop paths, or
  `et0 = (first_jd_TDB - 2451545.0) * 86400` for the `.mat` ephemeris chain
  (TDB interpretation verified in Phase 0).
- All internal times are scenario seconds relative to `et0`; every SPICE
  call uses `et0 + t_s` (adding coordinate seconds to ET is valid).
- Angles are radians everywhere; SPICE km / km/s are converted to m / m/s at
  the call boundary.

## 7. SEZ and azimuth/elevation conventions

`geometry.ecef2sez_dcm(lat_rad, lon_rad)` (geodetic latitude, east-positive
longitude, radians):

```text
S = ( sin(lat)cos(lon),  sin(lat)sin(lon), -cos(lat) )
E = (-sin(lon),          cos(lon),          0        )
Z = ( cos(lat)cos(lon),  cos(lat)sin(lon),  sin(lat) )
```

Orthonormal, det = +1, **right-handed with S x E = +Z**. At the exact
geographic pole the matrix remains orthonormal but the S/E directions are
longitude-dependent (no unique physical north there); tests assert
finiteness and orthogonality near the pole, not a unique azimuth.

```text
azimuth   = atan2(E, -S)   wrapped to [0, 2*pi)   (clockwise from north)
elevation = asin(Z / range)  in [-pi/2, +pi/2]
```

North 0, East 90, South 180, West 270 degrees. Residual differences in az/el
are wrapped with `wrap_to_pi`.

Zenith singularity policies (intentionally different, declared in metadata):

- Observable (`geometry.ecef2razel_sez`): horizontal norm < 1e-12 returns
  `az = 0.0` fallback (elevation and range remain valid).
- Implicit-light-time Jacobian
  (`measurements._position_measurement_jacobian_from_unit_los`): horizontal
  unit-LOS norm below `ANGLE_JACOBIAN_MIN_HORIZONTAL_UNIT_NORM` (1e-6)
  **raises** `MeasurementJacobianError`.
- Legacy vectorized analytic Jacobian: zeroes ill-defined azimuth rows.

## 8. Jacobian frame propagation

```text
J_SEZ = C_{SEZ<-F} C_{F<-I}(t_r) J_I
H     = H_tilde @ Phi(t_k, t_0)      (applied exactly once)
```

Frame-matrix derivatives are omitted because the receive tag is a fixed
datum and station coordinates / Earth orientation are not solve-for
parameters, so both `C` factors are constants with respect to the spacecraft
initial state. Light-time *does* depend on the state; that sensitivity is
carried in the line-of-sight term (`J_los = Phi_r - v_tx * dtau/dx0`), not
in the frame matrices. This contract breaks (and the code must be revisited)
if station coordinates, clock states, or Earth-orientation parameters ever
become estimated, or if the receive tag becomes state-dependent.

## 9. Station model

Stations are WGS84 geodetic coordinates (`config.Station`) converted once by
`geometry.geodetic_to_ecef_wgs84` (a = 6378137 m, 1/f = 298.257223563,
geodetic latitude, ellipsoidal height) and then used as **rigid ITRF93
Cartesian station coordinates**. The WGS84-vs-ITRF93 datum difference
(centimetre level) is neglected — far below the 5–94 m range sigmas. There
are **no plate-motion, solid-Earth-tide, ocean-loading, or other station
displacement models**, and no polar-motion sensitivity beyond what the
ITRF93 kernel chain provides.

## 10. Legacy GST visibility path (not production)

`visibility.analyze_visibility_gap` rotates by Greenwich sidereal time
(Curtis) from a naive-`datetime` UTC epoch with
`earth_rotation_rad_s = 7.292115e-5` (UTC treated as UT1, z-rotation only,
no precession/nutation/polar motion, no leap seconds). It is retained **only
as a comparison/diagnostic model** (`examples/compare_visibility_models.py`).
The production visibility and measurement paths use
`analyze_visibility_gap_with_transforms` /
`sample_j2000_to_itrf93_transforms` (SPICE `sxform`).

## 11. Two-way path and sxform interpolation

The two-way counted-Doppler path needs transforms at off-grid epochs and
**linearly interpolates the 6x6 sxform matrices in time**
(`radiometrics._interp_matrix`, `_interp_array_and_slope`;
`filters._interp_pass_values`). A linearly interpolated rotation block is
not exactly orthogonal (defect ~ (omega_E * dt)^2 at mid-segment). The
station-motion slope is differentiated self-consistently from the
interpolant (`d(X^-1 x_F)/dt = -X^-1 Xdot X^-1 x_F`). The kernel-gated
diagnostic `SxformInterpolationDiagnostics` measures orthogonality defect,
determinant error, station position/velocity error and the two-way
range/range-rate-level effect against exact `sxform` for 10/30/60 s grids;
its measured bounds are recorded in the test and in the frame-audit final
report. Production interpolation behavior is intentionally unchanged; if
the measured observable-level effect matters for a future campaign, fix it
in a separate phase.

## 12. Known limitations

- No EOP (Earth-orientation) solve-for sensitivity; no station-coordinate
  solve-for; no clock-dependent frame epoch.
- Rigid WGS84-as-ITRF93 stations (see §9).
- Two-way path uses linearly interpolated sxform matrices (see §11).
- The SR-UKF position path is instantaneous-only (ignores light-time
  measurement-model profiles); estimator-consistency work belongs to the
  measurement roadmap, not the frame contract.
- `tests/test_spice_snapshots.py` and its MATLAB fixture use the bare
  `MOON_PA` alias (pre-dates the explicit DE421/DE440 frame-name rule). It
  resolves deterministically under the DE421-only `REQUIRED_KERNELS` set.
  Renaming requires a fixture-regeneration decision — deliberately not
  changed in the frame-audit patch.
- Frame/epoch metadata additions to `measurement_model_metadata` (explicit
  `station_fixed_frame`, `line_of_sight_frame`, `topocentric_frame`,
  `azimuth_convention`, ... keys) are deferred until the parallel
  measurement-model work is committed.

## 13. Verification map

| Property | Test |
|---|---|
| SEZ orthogonality, determinant, handedness, near-pole behavior | `tests/test_frame_transformations.py` |
| Cardinal az/el incl. west wrap, nadir; degree/radian anchors | `tests/test_frame_transformations.py` |
| Transpose / direction / double-SEZ mutation detection | `tests/test_frame_transformations.py` |
| Zenith policy contracts (observable vs implicit Jacobian) | `tests/test_frame_transformations.py` |
| Frame-Jacobian chain vs finite difference (SPICE-free and SPICE) | both new test files |
| sxform block properties, station velocity vs FD sweep, round trip | `tests/test_frame_spice_validation.py` |
| Receive-vs-transmit epoch mutation | `tests/test_frame_spice_validation.py` |
| Range norm invariance through the chain | both new test files |
| Station-velocity contribution to range-rate (sign contract) | `tests/test_frame_spice_validation.py` |
| sxform linear-interpolation error bounds | `tests/test_frame_spice_validation.py` |
| SPICE-vs-MATLAB snapshot cross-check (pre-existing) | `tests/test_spice_snapshots.py` |
| SEZ chain-rule FD across geometries (pre-existing) | `tests/test_measurements.py` |

## 14. Measured local validation results (2026-07-11, Windows)

Environment: Windows 11, Python 3.13.12 (`C:\Users\erayh\miniforge3\python.exe`),
spiceypy 8.1.0, `LUNAR_OD_KERNEL_DIR=C:\Users\erayh\Documents\mice\kernels`.
Raw output: `frame_audit_run.log` (repository root, regenerated on this date).
All values below are measured on this machine with real kernels — none are
preflight/sandbox estimates.

Suite results:

| Run | Result |
|---|---|
| `tests/test_frame_transformations.py` (SPICE-free) | 17 passed |
| `tests/test_frame_spice_validation.py` (kernel-gated) | 12 passed, 0 skipped |
| `tests/test_stellar_aberration_vv.py` + `tests/test_measurements.py` | 53 passed |
| Full suite `tests/` | 472 passed, 20 skipped, 0 failed, no deselection, 38.8 s |

All 20 full-suite skips are `LUNAR_OD_RUN_SLOW_TESTS=1` slow-regression gates
in `tests/test_scenarios.py` and `tests/test_filters.py`. No SPICE test skipped.

### 14.1 Station-velocity central-difference sweep (measured)

Reference `|v_sxform| = 350.991350 m/s`; analytic `omega_E x r` oracle
relative magnitude error `2.295e-3`, direction cosine `0.999999890`.

| dt [s] | abs err [m/s] | rel err | ratio to previous |
|---:|---:|---:|---:|
| 100 | 3.111e-03 | 8.86e-06 | — |
| 30 | 2.808e-04 | 8.00e-07 | 0.090 |
| 10 | 3.373e-05 | 9.61e-08 | 0.120 |
| 3 | 1.157e-05 | 3.30e-08 | 0.343 |
| 1 | 2.666e-05 | 7.60e-08 | 2.304 |
| 0.1 | 1.800e-04 | 5.13e-07 | 6.750 |
| 0.01 | 2.971e-03 | 8.47e-06 | 16.510 |
| 0.001 | 1.640e-02 | 4.67e-05 | 5.520 |

The error is U-shaped as predicted: truncation-limited above ~10 s,
round-off-limited below ~1 s (large absolute SPICE ET cancellation), with the
crossover minimum at dt = 3 s. Plateau acceptance `min over {10, 3, 1} s =
1.157e-5 m/s < 2e-5 m/s` passes. The historical `2.666e-5 m/s` failure was the
dt = 1 s sample judged against a `5e-6 m/s` tolerance — a test-step/tolerance
issue, not a production frame bug (Case B of the acceptance decision tree).

### 14.2 sxform / round-trip / chain properties (measured)

| Property | Measured |
|---|---:|
| `max \|C^T C - I\|` | 2.22e-16 |
| `max \|det C - 1\|` | 1.11e-16 |
| Round-trip position / velocity | 1.86e-9 m / 5.68e-14 m/s |
| Solve-vs-inverse-`sxform` station state (pos / vel) | 9.31e-10 m / 1.71e-13 m/s |
| End-to-end LOS→az/el vs reference (daz / del / drange) | 0 rad / 0 rad / 1.79e-7 m |
| Frame-Jacobian chain vs FD, max relative mismatch | 2.24e-10 |
| Range norm invariance (ITRF93 / SEZ, relative) | 1.47e-16 / 2.94e-16 |
| Production-vs-inertial range-rate station term | 1.36e-12 m/s (contribution +307.70 m/s) |

### 14.3 Receive-vs-transmit epoch mutation (measured)

Receive ET `857217669.185389 s`, transmit ET `857217667.836742 s`, light time
`1.348647 s`. Evaluating `C_{F<-I}` at transmit instead of receive rotates the
LOS by `8.681e-5 rad = 17.907 arcsec` (predicted `omega_E * tau = 9.834e-5 rad`),
giving `d az = -5.710e-5 rad`, `d el = -6.667e-5 rad` — measurable, confirming
the receive-epoch contract matters at lunar light time.

### 14.4 sxform linear-interpolation error (measured, F1 diagnostics)

| Grid | Eval frac | Ortho defect | Pos err [m] | Vel err [m/s] | Range err [m] | RR(60 s) err [m/s] |
|---:|---:|---:|---:|---:|---:|---:|
| 10 s | 0.50 | 1.33e-07 | 3.20e-01 | 2.33e-05 | 3.64e-02 | 2.05e-05 |
| 10 s | 0.25 | 9.97e-08 | 2.40e-01 | 1.75e-05 | 2.72e-02 | 1.53e-05 |
| 30 s | 0.50 | 1.20e-06 | 2.88e+00 | 2.10e-04 | 3.37e-01 | 1.84e-04 |
| 30 s | 0.25 | 8.97e-07 | 2.16e+00 | 1.58e-04 | 2.51e-01 | 1.38e-04 |
| 60 s | 0.50 | 4.79e-06 | 1.15e+01 | 8.40e-04 | 1.40e+00 | 0 |
| 60 s | 0.25 | 3.59e-06 | 8.64e+00 | 6.30e-04 | 1.04e+00 | 5.52e-04 |

Worst case (60 s grid): station position error 11.5 m, velocity error
8.4e-4 m/s, two-way range effect 1.4 m, equivalent range-rate effect
5.5e-4 m/s. Whether these bounds are acceptable is a two-way (M3) decision:
at a 10 s transform grid the range effect is ~3.6 cm and the range-rate
effect ~2e-5 m/s; at 60 s the ~1.4 m range effect is significant relative to
typical range noise and should be revisited when M3 fixes its grid policy.

### 14.5 Conclusion

No production frame bug was found; the previously provisional audit
conclusion is now confirmed by real local SPICE execution. M2.3 and the
frame audit are closed (Case A): SPICE tests passed without skipping, the
full suite gives a natural `0 failed` with no deselection, and the
station-velocity FD failure is classified as a test-tolerance issue that the
plateau acceptance policy resolves.
