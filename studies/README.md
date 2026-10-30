# Research studies

Every study keeps its predeclaration (committed before any number), driver script, result
records and report. Reports state the outcome *as computed* by the predeclared rule and,
separately, any owner adjudication.

## Where things are

All studies share one directory tree so that the paths recorded inside result files,
report citations and frozen manifests stay valid:

| Directory | Contents |
|---|---|
| [`phase17/configs/`](phase17/configs/) | Predeclarations, grids and saved job specifications |
| [`phase17/scripts/`](phase17/scripts/) | Drivers, numerical oracles and plotting |
| [`phase17/results/`](phase17/results/) | Committed evidence: summaries, arrays and figures |
| [`phase17/reports/`](phase17/reports/) | Reports, status, defect ledger |

Files are named by campaign:

- `phase17_*` — the **Phase 17 campaign** (SRP observability, covariance qualification,
  tracking geometry, estimator hardening). Start with the [Phase 17 guide](phase17/README.md)
  and the [authoritative status](phase17/reports/phase17_authoritative_status.md).
- `phase_<c|d|f><n>*` — the **roadmap phases** below (measurement realism, dynamics, real data).

## Roadmap phases C1–F2

A materiality survey (no production change) decides what an implementation phase (`-I`) must add.

| Phase | Topic | Outcome | Report |
|---|---|---|---|
| C1 | Barycentric light time, Shapiro delay | Survey; implement list owner-accepted | [C1](phase17/reports/phase_c1_light_time_materiality.md) |
| C1-I | Light-time implementation | `C1I_QUALIFIED` | [C1-I](phase17/reports/phase_c1i_light_time_implementation.md) |
| C2 | Time scales UTC/TAI/TT/TDB, station clocks | Survey | [C2](phase17/reports/phase_c2_time_scale_materiality.md) |
| C2-I | Station time, clock consider parameter | `C2I_QUALIFIED` | [C2-I](phase17/reports/phase_c2i_station_time_implementation.md) |
| C3 | Station frames, EOP | Survey (`C3_INCONCLUSIVE` for the predict-PCK term) | [C3](phase17/reports/phase_c3_station_frame_materiality.md) |
| C3-I | Station coordinates, solid Earth tide | `C3I_QUALIFIED` | [C3-I](phase17/reports/phase_c3i_station_displacement_implementation.md) |
| C4 | DE440 vs DE421 | Survey; material | [C4](phase17/reports/phase_c4_de440_materiality.md) |
| C4-I | DE440 production ephemeris | `C4I_QUALIFIED` | [C4-I](phase17/reports/phase_c4i_de440_implementation.md) |
| C5 | Troposphere, ionosphere, ramped uplink | Survey | [C5](phase17/reports/phase_c5_media_materiality.md) |
| C5-I | Media calibration interface | `C5I_QUALIFIED` | [C5-I](phase17/reports/phase_c5i_media_implementation.md) |
| D1 | Integrator accuracy, shadow events | Characterization; closure hypothesis not confirmed | [D1](phase17/reports/phase_d1_integrator_characterization.md) |
| D2 | Propagation profile (opt-in) | `D2_QUALIFIED` | [D2](phase17/reports/phase_d2_propagation_consistency.md) |
| D3 | Extended Kalman filter, RTS smoother | `D3_QUALIFIED` | [D3](phase17/reports/phase_d3_ekf_rts_smoother.md) |
| D4 / D4-R | Iterated EKF + RTS smoother | D4 `NOT_QUALIFIED` as computed; `D4R_QUALIFIED` | [D4](phase17/reports/phase_d4_iterated_smoother.md) |
| F1 | Real LRO two-way range residuals | H1 not met as computed; H2, H3 met | [F1](phase17/reports/phase_f1_lro_residuals.md) |
| F1c | Real LRO two-way Doppler residuals | H1, H2 not met (constant Doppler bias); H3 met | [F1c](phase17/reports/phase_f1c_lro_doppler.md) |
| F2 | First LRO orbit determination (12 h, range) | `F2_NOT_QUALIFIED` as computed | [F2](phase17/reports/phase_f2_lro_od.md) |

LRO tracking data and reference orbits are not committed; each F report names its PDS source.
