"""PHASE 17-HARDEN - single source for the defect ledger and physical-contract inventory.

Writes
  artifacts/phase17_harden_defects.json
  docs/phase17_harden_defect_ledger.md
  artifacts/phase17_harden_physical_contracts.csv
from the tables below, so the machine-readable and human-readable forms cannot diverge.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# status vocabulary
FIXED = "FIXED"
FIXED_EARLIER = "FIXED_IN_EARLIER_PHASE_HARDENED_HERE"
QUARANTINED = "QUARANTINED_LEGACY_ONLY"
ARCHIVED = "ARCHIVED"
MITIGATED = "MITIGATED_RESIDUAL_DOCUMENTED"
ROOT_CAUSE = "ROOT_CAUSE_KNOWN_DOCUMENTED_APPROXIMATION"
FOLLOWUP = "OPEN_RECOMMENDED_FOLLOWUP"
KNOWN_APPROX = "KNOWN_DOCUMENTED_APPROXIMATION"
FIXED_F1 = "FIXED_IN_PHASE17_F1"

DEFECTS = [
    dict(id="HARDEN-D01", title="STM unpacked in C order (Phi^T used for Phi)",
         location="examples/phase17_r1o_core.py (pre-R1O-R) chain helper",
         discovered="Phase 17-R1O-R", root_cause="nom48[:, 6:42] is column-major (order='F'); the analysis helper reshaped it row-major",
         effect="R1O surrogate state Jacobians used the transpose of the STM; state columns wrong at O(1)",
         status=FIXED_EARLIER, klass="REQUIRED_FOR_CORRECTNESS",
         repair="order='F' everywhere; HARDEN audit found 25 production/analysis reshape sites, all column-major; only 3 labelled defect-reproduction oracles use C order",
         regression="tests/test_r1o_stm_layout.py; tests/test_harden_mutants.py::test_mutant_d_stm_c_order_is_detected",
         reproduction="examples/phase17_r1o_r_requalification.py (historical_nom48; PHASE17_LEGACY_REPRODUCTION=1)"),
    dict(id="HARDEN-D02", title="Earth ephemeris 504000 s early (epoch-origin mismatch)",
         location="campaign Earth table (od_gravity_covariance_campaign.get_earth_pos, C.ET0 = phase7 ET0 - 504000 s) fed to the production range model whose Earth rotation uses et0",
         discovered="Phase 17-FXR", root_cause="the Earth table and the measurement model counted seconds from different ET origins; the production generator accepted any Earth callable with no epoch metadata",
         effect="Earth centre displaced by ~5.8 days of lunar orbital motion relative to the Earth rotation used for station placement; all range/ΔDOR geometry inconsistent",
         status=FIXED_EARLIER, klass="REQUIRED_FOR_CORRECTNESS",
         repair="FXR: canonical epoch-tagged EarthEphemeris. HARDEN: lunar_od.measurement_contracts.generate_two_way_range_measurements_checked validates tagged providers (EphemerisEpochMismatchError) in front of the byte-frozen generator; scenarios.py and every Phase-17 builder use it (structural test); MoonCenteredEphemeris carries et0/frame/centre/units",
         regression="tests/test_fxr_earth_epoch_fixture.py; tests/test_harden_mutants.py mutant A (+ physical SPICE oracle)",
         reproduction="phase17_r1m_core.EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED (warns SupersededEarthFixtureWarning / LegacyEphemerisWarning)"),
    dict(id="HARDEN-D03", title="Earth table clamped at 28260 s",
         location="campaign Earth table end-of-coverage handling",
         discovered="Phase 17-FXR", root_cause="out-of-coverage queries returned the last sample instead of raising",
         effect="Earth frozen for every epoch after 28260 s of the arc (Earth moves ~1 km/s relative to the Moon)",
         status=FIXED_EARLIER, klass="REQUIRED_FOR_CORRECTNESS",
         repair="FXR: canonical fixture raises EarthEphemerisCoverageError. HARDEN: production MoonCenteredEphemeris raises EphemerisCoverageError beyond a 10 s margin instead of PCHIP extrapolation",
         regression="tests/test_harden_mutants.py mutant B; tests/test_harden_boundaries.py coverage tests",
         reproduction="EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED (clamp_after_s)"),
    dict(id="HARDEN-D04", title="ΔDOR / surrogate / Earth-LOS rows formed behind the Moon",
         location="phase17_r1o_core.build_ddor_arc, phase17_r1od_campaign.build_ddor_rows, phase17_r1o_r_requalification.earth_los_rows / bridge_reduced_rows",
         discovered="Phase 17-R1O-D2 (ΔDOR); Phase 17-HARDEN (Earth-LOS)",
         root_cause="each builder re-implemented visibility as an elevation mask only; the production elevation+occultation test existed but was not shared",
         effect="93 of 263 canonical ΔDOR rows (35%) and 230 of 590 Earth-LOS rows (39%) were physically impossible; R1O-D/FXR combined f_perp 0.3204 instead of 0.172351",
         status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="single contract phase17_r1m_core.station_visibility_mask (production analyze_visibility_gap_with_transforms); every builder defaults to it; elevation-only survives only as an explicitly named, warning-emitting legacy rule",
         regression="tests/test_harden_mutants.py mutant C (all builders + zero-radius mutant); tests/test_r1od2_ddor_contracts.py; tests/test_harden_boundaries.py limb/random-occultation oracles",
         reproduction="row_rule=DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY; examples/phase17_fxr_requalification.py reproduces FXR bitwise"),
    dict(id="HARDEN-D05", title="Range Jacobian: STM vs re-propagation discrepancy 1e-6..1e-4",
         location="lunar_od/dynamics.py dynamics_jacobian_a_matrix (variational equations)",
         discovered="Phase 17-R1O-D2 / HARDEN", root_cause="the variational A matrix omits the SRP position gradient K*dg/dr (documented Phase 13/16 approximation, mean 4.02e-13 of the gravity gradient); the gradient is concentrated in penumbra (max 1.09e-11 s^-2, median 7.7e-19) and its effect accumulates secularly",
         effect="range-Jacobian state columns relative error 2e-6..1e-5 over 4 orbits at rtol 1e-13; the event (light-time) derivative itself is exact (LIN error 1e-8..9e-8, falling with h)",
         status=FIXED_F1, klass="RECOMMENDED_FOLLOWUP",
         repair="Phase 17-F1 (commit 9a0e786, authorised protected change, additive R2 re-freeze 434b9c3): the exact position gradient of the production SRP force (distance, direction and penumbra-illumination terms) now enters the A matrix of every SRP-active variational path; the acceleration is unchanged (bitwise). Qualified: local gradient <= 1.5e-7 vs independent oracles in every regime; STM max best-resolved error 9.9e-8 (pre-F1 model 1.2e-6; fixed-step RK4 cross-check 2.2e-8 vs 3.4e-6); S_K 8.7e-7; all Phase-17 classes preserved (docs/phase17_f1_srp_variational_gradient_qualification.md). HARDEN history: none in HARDEN (dynamics.py was R2-protected). CLOSED as a mathematically understood, bounded approximation: NO_SHADOW and K=0 controls drop the error to the integrator floor; the first-order prediction from the omitted gradient reproduces all six columns (corr >= 0.986, residual <= 1.0e-7) once the FD oracle is integrated at rtol 3e-15. The partial fit at rtol 1e-14 (y0 0.908, vx0 0.737) was oracle integration error at the non-smooth penumbra RHS: it scaled with rtol and did not change with quadrature dt",
         regression="tests/test_f1_srp_variational_gradient.py; tests/test_harden_boundaries.py::test_srp_position_gradient_root_cause_and_its_f1_closure, ::test_range_jacobian_root_cause_evidence_is_consistent",
         reproduction="examples/phase17_harden_range_jacobian.py [--predict]; artifacts/phase17_harden_range_jacobian_oracle_convergence.json"),
    dict(id="HARDEN-D06", title="Stale optical docstring references a nonexistent script",
         location="lunar_od/lunar_landmark_optical.py", discovered="Phase 17-HARDEN",
         root_cause="docstring named examples/phase17_r1o_opt_light_time.py, which never existed",
         effect="documentation only", status=FIXED, klass="OPTIONAL_CLEANUP",
         repair="now cites examples/phase17_r1o_opt_oracles.py and OPTICAL_LIGHT_TIME_SIGNIFICANCE",
         regression="n/a (documentation)", reproduction="git show 2e8a53d"),
    dict(id="HARDEN-D07", title="R2 provenance gate: NameError + stale dynamics.py hash",
         location="examples/r2_measurement_fidelity_validation.py; tests/test_r2_measurement_fidelity.py",
         discovered="Phase 17-HARDEN (pre-existing full-suite non-pass: 1 failure + 10 cascading fixture errors)",
         root_cause="the qualified R2 gate hashed lunar_od/dynamics.py against its R2-closure value d0f7ab02 although commit 0e785f6 (authorised SRP-gradient documentation) legitimately changed the file; a loop referenced an undefined name",
         effect="R2 fidelity suite unrunnable (10 errors); no scientific effect",
         status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="documented re-freeze PHASE17_HARDEN_REFROZEN_CURRENT_TREE_SHA256 (old d0f7ab02 @ fc13cc4 -> new b369899a, authorising commit 0e785f6, blob c973822f); NameError fixed",
         regression="tests/test_r2_measurement_fidelity.py (112 passed with FA-06/desktop/P10)", reproduction="git show cbc3af6"),
    dict(id="HARDEN-D08", title="Anchor-convention check uses next-grid rotation for station B and a frozen Earth",
         location="examples/phase17_r1od_anchor_convention_check.py", discovered="Phase 17-HARDEN audit",
         root_cause="station B placed with the transform of the following grid epoch; Earth frozen at t=0",
         effect="its 'anchor immaterial' conclusion is not trustworthy; no current result depends on it (D2 uses MEASUREMENT_EPOCH)",
         status=QUARANTINED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="script refuses to run without PHASE17_LEGACY_REPRODUCTION=1",
         regression="tests/test_harden_mutants.py::test_historical_scripts_refuse_without_explicit_opt_in",
         reproduction="PHASE17_LEGACY_REPRODUCTION=1 python examples/phase17_r1od_anchor_convention_check.py"),
    dict(id="HARDEN-D09", title="Cross-worktree import leak (FA-06)",
         location="workspace-root pytest.ini (pythonpath = sibling python_port checkout)",
         discovered="pre-existing full-suite failure", root_cause="the worktree had no pytest.ini, so pytest used the workspace ini whose pythonpath imported lunar_od from another checkout",
         effect="tests could exercise a different checkout's production code", status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="worktree-local pytest.ini (no pythonpath) + tests/conftest.py pinning ROOT and desktop_app",
         regression="tests/test_harden_import_isolation.py (fails under -c ../pytest.ini)", reproduction="git show cbc3af6"),
    dict(id="HARDEN-D10", title="Machine-specific absolute paths",
         location="30 literals in examples/ and tests/ (C:/Users/.../Grad/...)", discovered="Phase 17-HARDEN audit",
         root_cause="paths hard-coded per session; two pointed at the repo only through the Grad junction",
         effect="non-portable; silently resolves to another checkout if the junction moves", status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="lunar_od/project_paths.py (env-overridable LUNAR_OD_PROJECT_ROOT / _CAMPAIGN_ROOT / _GRAVITY_ROOT) and repo-relative artifacts paths; zero literals remain in tracked .py outside archive/",
         regression="git grep 'Users/erayh' -- '*.py' ':!archive' is empty", reproduction="git show 91c7a9d"),
    dict(id="HARDEN-D11", title="Orphan R1O-D scripts importing nonexistent functions",
         location="examples/phase17_r1od_fd_convergence.py, examples/phase17_r1od_qualification.py (untracked)",
         discovered="hygiene report; HARDEN", root_cause="written against a pre-1045898 delta_dor API (delta_dor_k_column, spacecraft_differential_delay)",
         effect="fail on import; outputs superseded or nonexistent", status=ARCHIVED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="moved byte-for-byte to archive/phase17_orphans with README and SHA-256 manifest (LF hashes match the hygiene report)",
         regression="tests/test_harden_mutants.py::test_archived_orphans_are_not_importable", reproduction="archive/phase17_orphans/"),
    dict(id="HARDEN-D12", title="External campaign module prepends a sibling checkout to sys.path",
         location="od_covariance_campaign/06_scripts/od_gravity_covariance_campaign.py:30-32 (outside the repository, read-only)",
         discovered="Phase 17-HARDEN (caught by the new import-isolation test)",
         root_cause="sys.path.insert(0, '<...>/python_port_grail_pines_20260813') before 'from lunar_od ...'",
         effect="latent: that directory does not exist today so nothing was rebound; if recreated, the campaign (and every later importer) would bind lunar_od from another checkout",
         status=MITIGATED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="lunar_od.project_paths.import_campaign_module(): binds lunar_od to this repo first, strips foreign python_port* entries, verifies the campaign's lunar_od objects come from this repo; used by phase17_r1m_core and tests/conftest.py. The external file is not modified (outside scope).",
         regression="tests/test_harden_import_isolation.py::test_no_sibling_checkout_on_sys_path", reproduction="import od_gravity_covariance_campaign directly"),
    dict(id="HARDEN-D13", title="bridge_reduced_rows: frozen Earth and elevation-only visibility (used by FXR)",
         location="examples/phase17_r1o_r_requalification.py::bridge_reduced_rows", discovered="Phase 17-HARDEN audit",
         root_cause="R1O-D's reduced bridge placed stations on an Earth frozen at t=0",
         effect="FXR 'bridge_surrogate_5nrad' row is historical only", status=QUARANTINED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="always emits SupersededVisibilityWarning; documented historical-only; no visibility-correct variant by design",
         regression="tests/test_harden_mutants.py legacy-rule tests", reproduction="examples/phase17_harden_visibility_requalification.py"),
    dict(id="HARDEN-D14", title="Production range generator accepts untagged Earth callables",
         location="lunar_od/two_way_range.py generate_two_way_range_measurements (byte-frozen by R3-P21-M3)", discovered="Phase 17-HARDEN audit",
         root_cause="Earth providers are bare callables with no epoch/frame/unit metadata",
         effect="the FXR epoch offset passed silently", status=MITIGATED, klass="RECOMMENDED_FOLLOWUP",
         repair="contract layer lunar_od/measurement_contracts.py (the frozen module is not edited; a first attempt that edited it tripped the P21-M3 byte gate and was reverted): tagged providers are validated (epoch/frame/centre/units); every provider must give a physical Moon-Earth distance (catches km/m and wrong body); direct calls from scenarios.py / Phase-17 code are forbidden by test. Residual: an untagged callable with a wrong epoch but physical distance is still accepted; all Phase-17 science builders pass tagged providers (tested)",
         regression="tests/test_harden_mutants.py mutants A, E, F; ::test_science_earth_provider_is_tagged_and_matched; ::test_no_direct_calls_to_the_unchecked_range_generator; ::test_frozen_range_module_is_untouched_by_harden",
         reproduction="require_ephemeris_contract(lambda t: ..., et0) == 'UNTAGGED'"),
    dict(id="HARDEN-D15", title="MoonCenteredEphemeris extrapolated PCHIP silently outside its samples",
         location="lunar_od/ephemeris.py", discovered="Phase 17-HARDEN audit", root_cause="scipy PchipInterpolator extrapolates by default",
         effect="unbounded ephemeris error outside the sampled span", status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="EphemerisCoverageError beyond a 10 s margin", regression="tests/test_harden_mutants.py::test_mutant_b_clamp_replaced_by_coverage_error",
         reproduction="git show 2e8a53d"),
    dict(id="HARDEN-D16", title="ΔDOR sensitivity accepted a transmit epoch before the trajectory grid",
         location="lunar_od/delta_dor.py delta_dor_spacecraft_sensitivity_full", discovered="Phase 17-HARDEN audit",
         root_cause="state/STM interpolation extrapolated linearly and silently", effect="none in committed results (no D2/FXR row at grid index 0)",
         status=FIXED, klass="REQUIRED_FOR_CORRECTNESS", repair="DeltaDorEventError outside [t_grid[0], t_grid[-1]]; station B receive up to ~0.03 s past the last node remains a bounded (~1e-5 m) extrapolation",
         regression="tests/test_harden_boundaries.py light-time; R1O-D2 contract tests unchanged", reproduction="git show 2e8a53d"),
    dict(id="HARDEN-D17", title="Optical K-sensitivity accepted a flat/non-6x6 STM block",
         location="lunar_od/lunar_landmark_optical.py landmark_optical_state_and_k_sensitivity", discovered="Phase 17-HARDEN audit",
         root_cause="no shape check; a flat [6:42] row could be reshaped in the wrong order by a caller", effect="none observed", status=FIXED,
         klass="REQUIRED_FOR_CORRECTNESS", repair="raises unless phi.shape == (6, 6)", regression="optical tests (156 passed)", reproduction="git show 2e8a53d"),
    dict(id="HARDEN-D18", title="Numba ephemeris lerp extrapolates beyond its grid",
         location="lunar_od/accelerated.py _lerp_vec3_numba / make_lerp_vec3", discovered="Phase 17-HARDEN audit",
         root_cause="binary-search lerp uses the edge interval for out-of-grid t", effect="bounded in current use (propagation grids cover the arc); unbounded in principle",
         status=FOLLOWUP, klass="RECOMMENDED_FOLLOWUP", repair="not modified: accelerated.py is R2-protected", regression="none", reproduction="make_lerp_vec3(t, d)(t[-1] + 1e4)"),
    dict(id="HARDEN-D19", title="Unbounded linear extrapolation helpers in radiometrics/filters",
         location="lunar_od/radiometrics.py _interp_vector/_interp_state (edge branch); lunar_od/filters.py _interp_pass_values",
         discovered="Phase 17-HARDEN audit", root_cause="edge branch extrapolates linearly without a bound (filters also interpolates rotation matrices linearly)",
         effect="Phase-17 uses are bounded: ΔDOR t_tx now guarded (D16), t_B <= ~0.03 s past grid (~1e-5 m); filters path is not used by Phase-17 science",
         status=FOLLOWUP, klass="RECOMMENDED_FOLLOWUP", repair="route through lunar_od/history_domain bounds in a later phase", regression="none", reproduction="_interp_state(t, x, t[-1] + 100)"),
    dict(id="HARDEN-D20", title="Legacy analyze_visibility_gap carries three time origins; VisibilityConfig.epoch_utc redundant on the transform path",
         location="lunar_od/visibility.py", discovered="Phase 17-HARDEN audit", root_cause="legacy MATLAB-port API (GST from epoch_utc, Earth callable in scenario seconds, transforms from et0)",
         effect="none on the transform path used by Phase 17 (epoch_utc unused there)", status=FOLLOWUP, klass="OPTIONAL_CLEANUP",
         repair="Phase-17 code uses only the transform path via station_visibility_mask (epoch_utc='UNUSED_BY_THE_TRANSFORM_PATH')", regression="tests/test_harden_boundaries.py elevation/limb tests", reproduction="n/a"),
    dict(id="HARDEN-D21", title="earth_position_mci counts seconds from the campaign epoch implicitly",
         location="examples/phase17_r1m_core.py earth_position_mci", discovered="Phase 17-HARDEN audit",
         root_cause="helper takes campaign seconds without an explicit origin argument", effect="none (all callers use campaign seconds; FXR audited)",
         status=FOLLOWUP, klass="OPTIONAL_CLEANUP", repair="documented; station_visibility_mask takes the Earth history explicitly", regression="tests/test_fxr_earth_epoch_fixture.py", reproduction="n/a"),
    dict(id="HARDEN-D22", title="Range residuals below 1e-7 m zeroed", location="lunar_od/two_way_range.py compute_two_way_range_residuals",
         discovered="Phase 17-HARDEN audit", root_cause="synthetic-closure round-off suppression", effect="cosmetic; 1e-7 m << any noise level",
         status=FOLLOWUP, klass="OPTIONAL_CLEANUP", repair="none", regression="none", reproduction="n/a"),
    dict(id="HARDEN-D23", title="Counted Doppler interpolates Earth linearly (range/ΔDOR use Hermite)",
         location="lunar_od/radiometrics.py make_exact_counted_doppler_station_state_provider (design R3-P22; the "
                  "frozen R2 reference module two_way_counted_doppler_reference.py keeps the linear policy by design)",
         discovered="Phase 17-HARDEN cross-measurement audit; magnitude corrected by Phase 17-R1O-CD1 closure",
         root_cause="R3-P22 kept the Earth LINEAR so the exact path reproduced R2 model S. The count interval is "
                    "centred on a grid node at the RECEIVE epochs, where linear errors cancel; the uplink TRANSMIT "
                    "epochs (t_mid +/- Tc/2 - tau) sit asymmetrically in their grid intervals, so they do not",
         effect="CURRENT (CD1 closure, D23 qualification): near-constant counted-Doppler VALUE offset 1.67-1.83e-3 "
                "m/s (239,265 rows); Earth position <= 2.80 m between nodes; information metrics f_perp <= 4e-10. "
                "SUPERSEDED (HARDEN): ~1.5e-6 m/s, which differenced only the symmetric receive endpoints and the x "
                "component",
         status=FIXED, klass="REQUIRED_FOR_CORRECTNESS",
         repair="Phase 17-HARDEN-D23: cubic-Hermite Earth (radiometrics._interp_state, as M3 range and DeltaDOR); "
                "production vs direct-SPICE model S/F oracle over the CD1/KP1 population max 7.6e-8 m/s, rms 1.6e-8 "
                "(ET-exact oracle max 5.4e-9); docs/phase17_harden_d23_repair_qualification.md",
         regression="tests/test_harden_d23_counted_doppler_earth.py (historical linear mutant detected; transmit-"
                    "epoch mechanism); tests/test_doppler_model_review.py R3EarthEphemerisPolicyInvariance",
         reproduction="examples/phase17_d23_qualification.py (historical_linear_provider)"),
    dict(id="HARDEN-D24", title="Exact station transform evaluated at rounded absolute ET",
         location="lunar_od/radiometrics.py (and the M3 range / DeltaDOR exact providers): sxform(et0 + t)",
         discovered="Phase 17-HARDEN-D23 qualification (ET-exact oracle diagnostic)",
         root_cause="fl(et0 + t) is quantised to ulp(8.6e8 s) = 1.19e-7 s; the site moves ~465 m/s",
         effect="counted-Doppler value noise rms <= 1.0e-7 m/s (max 3.6e-7), below the most optimistic operational "
                "noise floor 2.1e-6 m/s; Earth itself is unaffected (Hermite works in exact scenario seconds)",
         status=KNOWN_APPROX, klass="RECOMMENDED_FOLLOWUP",
         repair="none (outside D23); first-order two-sum correction demonstrated in "
                "examples/phase17_d23_qualification.py et_exact_spice_provider",
         regression="none", reproduction="examples/phase17_d23_qualification.py --et-exact-diagnostic"),
]

# contract_id, quantity, producer, consumer, time_origin, frame, center, units, coverage, enforcement, tests
CONTRACTS = [
    ("PC-01", "Earth position/velocity (science)", "phase17_r1m_core.EarthEphemeris (CANONICAL)", "two_way_range, ΔDOR, visibility",
     "et_origin == pass_geo.et0_s", "J2000", "MOON", "m, m/s", "[t_min_s, t_max_s], raises", "require_ephemeris_contract + EarthEphemerisCoverageError", "test_harden_mutants A/B/E/F"),
    ("PC-02", "Earth/Sun samples (production)", "lunar_od.ephemeris.MoonCenteredEphemeris", "dynamics, SRP, measurements",
     "et0_s (tagged)", "J2000", "MOON", "m, m/s", "samples +/- 10 s, raises", "EphemerisCoverageError; et_origin/frame/center/units", "test_harden_mutants B; test_harden_boundaries"),
    ("PC-03", "Moon-Earth distance", "any Earth provider", "measurement_contracts.generate_two_way_range_measurements_checked",
     "n/a", "n/a", "MOON", "m", "3.40e8..4.20e8 m", "require_moon_earth_distance", "test_mutant_f_km_provider_rejected_even_untagged"),
    ("PC-04", "J2000->ITRF93 transform", "sample_j2000_to_itrf93_transforms (SPICE sxform)", "station placement, visibility",
     "et0 + t (same epoch as Earth)", "J2000->ITRF93", "Earth", "rad, rad/s", "SPICE kernels", "orthonormal, det=+1; station v = dr/dt", "test_station_velocity_is_the_time_derivative_of_station_position"),
    ("PC-05", "Station visibility (radio)", "phase17_r1m_core.station_visibility_mask", "all ΔDOR / surrogate / range rows",
     "grid epochs", "J2000/ITRF93", "MOON", "deg", "grid", "elevation >= mask AND not occulted by the Moon; legacy rule warns", "test_harden_mutants C; limb/elevation boundary tests"),
    ("PC-06", "Earth line of sight (surrogate)", "earth_los_rows", "R1O Earth-LOS rows", "grid epochs", "J2000", "MOON", "rad", "grid",
     "is_occulted_by_moon(spacecraft, Earth)", "test_mutant_c_occulted_rows_rejected_by_every_builder"),
    ("PC-07", "STM storage", "lunar_od.dynamics (augmented state)", "every Jacobian chain", "t0 of the propagation", "MCI", "MOON", "SI",
     "propagated grid", "column-major (order='F') in [6:42]; K sensitivity in [42:48]", "test_r1o_stm_layout; mutant D"),
    ("PC-08", "Variational A matrix", "dynamics_jacobian_a_matrix", "STM", "n/a", "MCI", "MOON", "s^-2", "n/a",
     "gravity gradients + (Phase 17-F1) the exact SRP position gradient when SRP is active", "test_f1_srp_variational_gradient; test_srp_position_gradient_root_cause_and_its_f1_closure"),
    ("PC-09", "SRP acceleration", "lunar_od.srp", "dynamics", "t (Sun at et0+t)", "MCI", "MOON", "m/s^2", "n/a",
     "a = K*P*nu*u; linear in K; nu in [0,1]; umbra -> 0", "test_srp_is_linear_in_k_and_vanishes_at_k_zero; shadow tests"),
    ("PC-10", "Third-body acceleration", "third_body_acceleration", "dynamics", "et0 + t", "J2000", "MOON", "m/s^2", "n/a",
     "direct minus indirect term", "test_third_body_is_direct_minus_indirect_at_the_right_epoch"),
    ("PC-11", "Two-way range event", "generate_two_way_range_measurements", "range design", "receive tag t3; et0 origin", "J2000", "MOON", "m, s",
     "events outside history raise", "TwoWayEventHistoryError; Earth contract via measurement_contracts (module byte-frozen)", "event ledgers; range tests; test_no_direct_calls_to_the_unchecked_range_generator"),
    ("PC-12", "ΔDOR common-transmit event", "solve_common_transmit_event + delta_dor_spacecraft_sensitivity_full", "ΔDOR design",
     "t_obs at station A", "J2000", "MOON", "s", "t_tx within grid, raises", "DeltaDorEventError", "test_common_transmit_event_satisfies_both_light_time_equations"),
    ("PC-13", "Counted Doppler geometry", "two_way_counted_doppler_reference", "Doppler design", "count interval", "J2000", "MOON", "m/s",
     "grid", "Earth linear interpolation (R3-P22, quantified)", "test_cross_measurement_geometry_and_derivatives_are_consistent"),
    ("PC-14", "Optical landmark sensitivity", "lunar_landmark_optical", "optical design", "grid", "MCI", "MOON", "rad", "n/a",
     "phi.shape == (6, 6)", "optical tests"),
    ("PC-15", "Project data paths", "lunar_od.project_paths", "examples, tests", "n/a", "n/a", "n/a", "n/a", "n/a",
     "env-overridable resolver, no absolute literals", "git grep check"),
    ("PC-16", "Import isolation", "tests/conftest.py, pytest.ini, import_campaign_module", "all tests", "n/a", "n/a", "n/a", "n/a", "n/a",
     "lunar_od from this worktree; no foreign python_port* on sys.path", "test_harden_import_isolation"),
    ("PC-17", "Historical reproduction opt-in", "phase17_r1m_core.require_legacy_reproduction_opt_in", "11 historical scripts", "n/a", "n/a", "n/a", "n/a", "n/a",
     "PHASE17_LEGACY_REPRODUCTION=1 required", "test_historical_scripts_refuse_without_explicit_opt_in"),
]


def write_all() -> None:
    art, docs = ROOT / "artifacts", ROOT / "docs"
    (art / "phase17_harden_defects.json").write_text(
        json.dumps({"phase": "17-HARDEN", "defects": DEFECTS}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(art / "phase17_harden_physical_contracts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["contract_id", "quantity", "producer", "consumer", "time_origin", "frame", "center",
                    "units", "coverage", "enforcement", "tests"])
        w.writerows(CONTRACTS)
    lines = ["# Phase 17-HARDEN defect ledger", "",
             "Generated by `examples/phase17_harden_ledger.py`, which also writes "
             "`artifacts/phase17_harden_defects.json`. The two files have one source.", "",
             "| ID | defect | status | class |", "|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s |" % (d["id"], d["title"], d["status"], d["klass"]) for d in DEFECTS]
    for d in DEFECTS:
        lines += ["", "## %s: %s" % (d["id"], d["title"]), ""]
        for key, label in (("location", "Location"), ("discovered", "Discovered"), ("root_cause", "Root cause"),
                           ("effect", "Physical / numerical / scientific effect"), ("status", "Status"),
                           ("klass", "Class"), ("repair", "Repair"), ("regression", "Regression"),
                           ("reproduction", "Historical reproduction path")):
            lines.append("- **%s:** %s" % (label, d[key]))
    (docs / "phase17_harden_defect_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_all()
    print("%d defects, %d contracts" % (len(DEFECTS), len(CONTRACTS)))
