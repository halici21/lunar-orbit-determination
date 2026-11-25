"""WP7.1 (REQ-SW-002/003, TST-01): the verification level of every test, kept in one table.

``tests/conftest.py`` reads this table at collection and gives every test exactly one
level marker plus its resource markers, so CI can select tests by marker
(``tools/run_tier.py``) without editing hundreds of test files.

Levels (the oracle a test relies on; docs/assurance/test-levels.md):

====  ===================  ==================================================================
L0    ``analytic``         closed form, convex reference problem, invariant, same-model recovery
L1    ``derivative``       finite-difference / Richardson check of a Jacobian, STM or partial
L2    ``external_oracle``  an independent tool or record: SPICE, ERFA/SOFA, Orekit, MATLAB, standards
L3    ``model_mismatch``   truth and estimator deliberately use different models
L4    ``statistical``      Monte Carlo, NIS/NEES, chi-square and whiteness consistency
L5    ``real_data``        real tracking or real gravity-field data
L6    ``golden``           frozen record of the project's own outputs (snapshot, approved bytes)
L7    ``contract_negative`` interface, schema, provenance, UI-workflow and fail-closed contracts
====  ===================  ==================================================================

Resource markers: ``spice`` (needs SPICE kernels), ``campaign`` (reads the external
campaign data), ``qt`` (PyQt5 desktop), ``numba`` (exercises the Numba twins) and
``slow`` (set automatically from ``tests/slow.py``; may also be listed here).

How a test is classified, first match wins:

1. a level marker written on the test itself (``@pytest.mark.golden``);
2. the longest :data:`NODE_LEVELS` key that prefixes the node id;
3. the :data:`FILE_LEVELS` row of its file.

A test that matches nothing is *unclassified*: the run reports the file and the
one table row that fixes it, and ``tools/run_tier.py`` (``--strict-levels``) refuses
to run. Add the row here, not a marker in the test file.
"""
from __future__ import annotations

LEVELS: dict[str, str] = {
    "analytic": "L0",
    "derivative": "L1",
    "external_oracle": "L2",
    "model_mismatch": "L3",
    "statistical": "L4",
    "real_data": "L5",
    "golden": "L6",
    "contract_negative": "L7",
}
RESOURCES: tuple[str, ...] = ("spice", "campaign", "qt", "numba", "slow")

#: REQ-SW-003: a ``spice`` test at one of these levels FAILS when the kernels are missing.
KERNEL_FAIL_CLOSED_LEVELS: frozenset[str] = frozenset(
    ("external_oracle", "model_mismatch", "statistical", "real_data", "golden")
)
#: Derived marker on every test that falls under REQ-SW-003; kernel-less runs deselect it explicitly.
REQUIRES_KERNELS = "requires_kernels"

_A, _D, _X, _M, _S, _R, _G, _C = (
    "analytic", "derivative", "external_oracle", "model_mismatch",
    "statistical", "real_data", "golden", "contract_negative",
)

#: file name -> (level, space-separated resource markers).  One row per tests/test_*.py.
FILE_LEVELS: dict[str, tuple[str, str]] = {
    "test_adaptive_tuning.py": (_S, ""),
    "test_archive_sources.py": (_C, "qt"),
    "test_archived_config_compat.py": (_C, ""),
    "test_assessment_evidence.py": (_X, "spice"),
    "test_batch_whitening_wp4_5b.py": (_S, ""),
    "test_bounded_least_squares.py": (_A, ""),
    "test_compact_recovery.py": (_C, "qt"),
    "test_config.py": (_C, ""),
    "test_consistency_diagnostics_wp4_6.py": (_S, ""),
    "test_constants.py": (_X, "spice"),
    "test_convergence_certification_wp4_2.py": (_A, ""),
    "test_correlated_noise_wp4_5.py": (_S, ""),
    "test_covariance_conditioning_wp4_3.py": (_A, ""),
    "test_de440_kernel_contract.py": (_X, "spice"),
    "test_delta_dor.py": (_A, ""),
    "test_desktop_bridge_provenance.py": (_G, ""),
    "test_desktop_tutorial.py": (_C, "qt"),
    "test_desktop_workflow.py": (_C, "qt"),
    "test_diagnostics.py": (_A, ""),
    "test_direct_spice_rhs.py": (_X, "spice"),
    "test_dmc.py": (_A, ""),
    "test_dmc_batch_range_rate_wp4_8e.py": (_A, ""),
    "test_dmc_batch_wp4_8d.py": (_A, ""),
    "test_dmc_process_noise_wp0_2.py": (_A, ""),
    "test_dmc_rtn_wp4_8b.py": (_D, ""),
    "test_doppler_model_review.py": (_D, ""),
    "test_dynamics.py": (_A, ""),
    "test_earth_j2.py": (_A, "numba"),
    "test_ekf_partials.py": (_D, ""),
    "test_ekf_smoother.py": (_A, ""),
    "test_environment_readiness.py": (_C, "qt spice"),
    "test_ephemeris.py": (_X, "spice"),
    "test_ephemeris_generation_wp1_2.py": (_C, ""),
    "test_epoch_translation_consistency.py": (_A, "numba"),
    "test_estimator_lm_consistency.py": (_A, "spice"),
    "test_estimators.py": (_A, "spice"),
    "test_experiment_orchestrator.py": (_C, ""),
    "test_f1_srp_variational_gradient.py": (_D, "spice campaign"),
    "test_f1_station_exact_epoch.py": (_A, ""),
    "test_f2_current_adapters.py": (_C, ""),
    "test_f2_qualified_adapter.py": (_C, ""),
    "test_fep_prior.py": (_A, ""),
    "test_filters.py": (_A, ""),
    "test_force_contract.py": (_C, ""),
    "test_force_identity_wp1_5.py": (_C, ""),
    "test_force_model_parity.py": (_C, ""),
    "test_force_models.py": (_A, "numba"),
    "test_frame_spice_validation.py": (_X, "spice"),
    "test_frame_transformations.py": (_A, ""),
    "test_fxr_earth_epoch_fixture.py": (_G, "spice campaign"),
    "test_geometry.py": (_A, ""),
    "test_gravity_frame_label_wp1_7.py": (_C, ""),
    "test_gravity_harmonics.py": (_A, ""),
    "test_gravity_harmonics_numba.py": (_A, "numba"),
    "test_gravity_harmonics_prototype.py": (_A, ""),
    "test_gravity_model_loader.py": (_C, ""),
    "test_ground_track_density.py": (_C, "qt"),
    "test_ground_track_keyboard.py": (_C, "qt"),
    "test_ground_track_layout.py": (_C, "qt"),
    "test_ground_track_legibility.py": (_C, "qt"),
    "test_ground_track_lifecycle.py": (_C, "qt"),
    "test_ground_track_models.py": (_C, "qt"),
    "test_ground_track_results.py": (_C, "qt"),
    "test_ground_track_workflow.py": (_C, "qt"),
    "test_harden_bls1_k_srp_recovery.py": (_A, "spice"),
    "test_harden_boundaries.py": (_A, "spice campaign"),
    "test_harden_d18_d19_interpolation_bounds.py": (_C, "numba"),
    "test_harden_d23_counted_doppler_earth.py": (_A, ""),
    "test_harden_d25_posterior_dynamics_contract.py": (_C, "spice"),
    "test_harden_d27_convergence_status.py": (_C, ""),
    "test_harden_d28_posterior_prior_contract.py": (_C, "spice"),
    "test_harden_d29_srif_k_prior_contract.py": (_C, "spice"),
    "test_harden_d30_acceptance_rescore.py": (_C, "spice"),
    "test_harden_import_isolation.py": (_C, "campaign"),
    "test_harden_mutants.py": (_C, "spice campaign"),
    "test_harmonics_dynamics_splice.py": (_A, "spice numba"),
    "test_history_domain.py": (_C, ""),
    "test_history_kinematic_consistency.py": (_A, ""),
    "test_interval_scale_derivative.py": (_D, ""),
    "test_k_sensitivity_fd_harness.py": (_D, "spice"),
    "test_k_srp_bls_solve_for.py": (_C, "spice"),
    "test_k_srp_square_root_covariance.py": (_A, ""),
    "test_k_srp_srif_solve_for.py": (_C, "spice"),
    "test_k_srp_srukf_solve_for.py": (_C, "spice"),
    "test_kernel_manifest.py": (_C, "spice"),
    "test_kp1_alternative_k.py": (_A, ""),
    "test_kp2_m3_recovery.py": (_A, ""),
    "test_large_workspace.py": (_C, "qt"),
    "test_legacy_layout.py": (_C, ""),
    "test_levels.py": (_C, ""),
    "test_lm_damping_wp4_2b.py": (_A, ""),
    "test_lro_data_partition.py": (_C, ""),
    "test_lunar_frames.py": (_X, "spice"),
    "test_lunar_landmark_optical.py": (_D, ""),
    "test_map_objective_consistency.py": (_A, ""),
    "test_mc_consistency_wp4_7b.py": (_S, ""),
    "test_meas05_sigma_ephemeris_fidelity.py": (_X, "spice numba"),
    "test_measurement_contracts.py": (_C, ""),
    "test_measurement_ingestion.py": (_C, ""),
    "test_measurement_model_safety_diagnostics.py": (_C, "spice"),
    "test_measurements.py": (_D, "spice"),
    "test_media.py": (_X, "spice"),
    "test_media_chain_consistency.py": (_D, ""),
    "test_mission_overview.py": (_C, "qt"),
    "test_mm_truth_field.py": (_M, ""),
    "test_model_profile_transport.py": (_C, "spice"),
    "test_monte_carlo.py": (_S, ""),
    "test_monte_carlo_seeds_wp4_7.py": (_S, ""),
    "test_moon_orientation_wp1_3.py": (_X, "spice"),
    "test_native_media_partials.py": (_D, "spice"),
    "test_noise_models.py": (_S, ""),
    "test_observability.py": (_D, ""),
    "test_observability_scaling_wp4_3b.py": (_A, ""),
    "test_od_contracts.py": (_C, ""),
    "test_orbit.py": (_A, ""),
    "test_orbit_designer_guidance.py": (_C, "qt"),
    "test_packaging_extras.py": (_C, ""),
    "test_paired_benchmark.py": (_G, ""),
    "test_phase13c_harmonics_validation.py": (_C, ""),
    "test_phase13g_gravity_orbit_effects.py": (_A, ""),
    "test_phase13g_synthesis_report.py": (_G, "spice campaign"),
    "test_phase_c6_lunar_radiation.py": (_A, ""),
    "test_phase_c7_dynamics_candidates.py": (_D, ""),
    "test_phase_f3_core.py": (_A, ""),
    "test_pines_gradient_numba_parity.py": (_A, "spice numba"),
    "test_profile_desktop_transport.py": (_C, "qt"),
    "test_propagation.py": (_X, ""),
    "test_propagation_profile_wp1_8.py": (_C, "numba"),
    "test_pytest_acceleration.py": (_A, "numba"),
    "test_pytest_desktop_app.py": (_C, "qt spice"),
    "test_pytest_numba_benchmarks.py": (_A, "numba"),
    "test_q_tuning.py": (_S, ""),
    "test_quickstart.py": (_G, ""),
    "test_r0a_force_config_parity.py": (_C, "spice numba"),
    "test_r0b_f1_executed_physics.py": (_C, "spice numba"),
    "test_r1_covariance_observability_parity.py": (_D, "numba"),
    "test_r1_r4_long_arc_qualification.py": (_C, ""),
    "test_r1o_cd1_closure.py": (_G, ""),
    "test_r1o_cd1_counted_doppler.py": (_G, "campaign"),
    "test_r1o_stm_layout.py": (_D, ""),
    "test_r1od2_ddor_contracts.py": (_G, "spice campaign"),
    "test_r1od3_tracking_design.py": (_G, "campaign"),
    "test_r1od4_gm_confirmation.py": (_G, "campaign"),
    "test_r2_measurement_fidelity.py": (_G, "spice"),
    "test_real_grail_optional.py": (_R, ""),
    "test_recorded_run_workspace.py": (_C, "qt"),
    "test_reference_config.py": (_C, "campaign"),
    "test_relativistic_light_time.py": (_X, "spice"),
    "test_relativistic_light_time_c2.py": (_X, "spice"),
    "test_release_manifest.py": (_C, ""),
    "test_reporting.py": (_A, ""),
    "test_result_summary.py": (_C, "qt"),
    "test_rf_link_budget.py": (_A, ""),
    "test_run_workspace_resume.py": (_C, "qt"),
    "test_scenario_comparison.py": (_M, ""),
    "test_scenario_config.py": (_C, ""),
    "test_scenario_config_harmonics.py": (_C, ""),
    "test_scenario_reproducibility_wp1_6.py": (_G, "spice"),
    "test_scenarios.py": (_A, "spice"),
    "test_shadow_contact_guard.py": (_A, ""),
    "test_shadow_epoch_translation.py": (_A, ""),
    "test_shadow_events.py": (_A, ""),
    "test_shadow_grazing_regression.py": (_A, ""),
    "test_shapiro_epoch_derivative.py": (_D, ""),
    "test_spice_loader.py": (_C, ""),
    "test_spice_snapshots.py": (_X, "spice"),
    "test_srif_acceptance_wp4_1.py": (_A, "spice"),
    "test_srp_force_identity.py": (_C, ""),
    "test_srp_k_sensitivity.py": (_D, ""),
    "test_srp_production.py": (_G, "campaign"),
    "test_station_catalog_filters.py": (_C, "qt"),
    "test_station_displacement.py": (_X, "spice"),
    "test_station_time.py": (_X, "spice"),
    "test_stellar_aberration_vv.py": (_X, "spice"),
    "test_thesis_matrix.py": (_C, ""),
    "test_tracking_console.py": (_C, "qt"),
    "test_tracking_render_semantics.py": (_C, "qt"),
    "test_two_way_counted_doppler_four_event.py": (_G, ""),
    "test_two_way_event_conditioning.py": (_X, "spice"),
    "test_two_way_range.py": (_D, ""),
    "test_two_way_range_integration.py": (_X, "spice"),
    "test_ukf_constraints_wp4_4b.py": (_A, ""),
    "test_ukf_dmc_wp4_8c.py": (_A, ""),
    "test_ukf_force_consistency.py": (_C, ""),
    "test_ukf_numerics_wp4_4a.py": (_A, ""),
    "test_utdf.py": (_C, ""),
    "test_visibility.py": (_A, ""),
    "test_visibility_workflow.py": (_C, "qt"),
    "test_workflow_polish.py": (_C, "qt"),
    "test_workspace_appearance.py": (_C, "qt"),
    "test_workspace_session.py": (_C, "qt"),
    "test_wp2_1_barycentric_counted_doppler.py": (_X, "spice"),
    "test_wp2_2_media_production.py": (_X, "spice"),
    "test_wp2_3_two_way_range_station_bias.py": (_D, "spice"),
    "test_wp2_4_dsn_stations.py": (_X, "spice"),
    "test_wp2_4_earth_orientation.py": (_X, "spice"),
    "test_wp2_5_measurement_defaults.py": (_D, "spice"),
    "test_wp2_6_delta_dor_ssb.py": (_X, "spice"),
    "test_wp2_6_landmark_conventions.py": (_D, "spice"),
    "test_wp2_7_visibility.py": (_X, "spice numba"),
    "test_wp3_1_force_terms.py": (_A, "spice"),
    "test_wp3_2_lunar_radiation.py": (_A, ""),
    "test_wp3_2_lunar_radiation_transport.py": (_C, "spice"),
    "test_wp3_2_lunar_tide.py": (_A, ""),
    "test_wp3_2_lunar_tide_transport.py": (_C, "spice"),
    "test_wp4_2r_bls_loosened_exhaustion.py": (_C, "spice"),
}

#: node-id prefix -> (level, resource markers); refines a file row for single tests or classes.
NODE_LEVELS: dict[str, tuple[str, str]] = {
    # The four external-SPICE integration cases tools/run_library_checks.py lists by name.
    "tests/test_estimators.py::EstimatorTests::test_position_srif_reduces_cost_on_spice_fixture_arc": (_X, "spice"),
    "tests/test_estimators.py::EstimatorTests::test_range_rate_srif_reduces_cost_on_spice_fixture_arc": (_X, "spice"),
    "tests/test_measurements.py::MeasurementTests::test_position_measurement_generation_and_clean_residual_closure": (_X, "spice"),
    "tests/test_measurements.py::MeasurementTests::test_range_rate_measurement_generation_and_clean_residual_closure": (_X, "spice"),
    # Gated by its own LUNAR_OD_RUN_SLOW_TESTS check rather than tests/slow.py.
    "tests/test_r1_r4_long_arc_qualification.py::BoundedQualificationSlice::test_slow_day_horizon_slice_remains_bitwise_at_zero_delay": (_C, "slow"),
}


def classify(nodeid: str) -> tuple[str, frozenset[str]] | None:
    """Level and resource markers of a node id (``tests/test_x.py::Class::test``), or None."""
    nodeid = nodeid.replace("\\", "/")
    best = None
    for prefix, row in NODE_LEVELS.items():
        if nodeid == prefix or nodeid.startswith(prefix + "::") or nodeid.startswith(prefix + "["):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, row)
    if best is None:
        file_name = nodeid.split("::", 1)[0].rsplit("/", 1)[-1]
        row = FILE_LEVELS.get(file_name)
        if row is None:
            return None
    else:
        row = best[1]
    return row[0], frozenset(row[1].split())


def suggested_row(file_name: str) -> str:
    """The one-line table row that classifies a new test file (level to be chosen)."""
    return f'    "{file_name}": (_A, ""),  # choose the level: _A _D _X _M _S _R _G _C'
