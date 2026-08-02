"""R2 strict-Option-B counted-Doppler reference and evidence gates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from examples import r2_measurement_fidelity_validation as campaign
from lunar_od.config import Station
from lunar_od.constants import J2_MOON_UNNORMALIZED
from lunar_od.filters import run_lunar_ukf
from lunar_od.force_contract import (
    FORCE_CONTRACT_SCHEMA_VERSION,
    ConsumerReadiness,
    ConsumerRole,
    consumer_capabilities_for,
)
from lunar_od.radiometrics import RangeRatePhysicsConfig
from lunar_od.scenario_config import (
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
)
from lunar_od.two_way_counted_doppler_reference import (
    EXACT_STATION_MODEL_ID,
    FOUR_EVENT_MODEL_ID,
    LEGACY_MODEL_ID,
    REFERENCE_STATION_STATE_METHOD,
    CountedDopplerReferenceConfig,
    evaluate_lsf_decomposition,
    exact_station_four_event_counted_doppler_jacobian,
    exact_station_four_event_counted_doppler_reference,
    exact_station_single_bounce_counted_doppler_reference,
    generate_four_event_counted_doppler_reference,
    make_exact_event_station_state_provider,
    reference_config_with_delay,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_case():
    station = Station(
        name="R2 synthetic",
        lat_deg=12.0,
        lon_deg=-35.0,
        alt_m=250.0,
        color_rgb=(0.2, 0.4, 0.6),
        sigma_range_m=5.0,
        sigma_angle_rad=1e-4,
        sigma_range_rate_mps=1e-4,
    )
    t_grid = np.linspace(-8.0, 8.0, 65)
    x0 = np.array([3.84e8, 1.2e6, -8.0e5, -40.0, 850.0, 25.0])
    augmented = campaign._linear_augmented_history(t_grid, x0)
    earth_position = np.zeros((t_grid.size, 3))
    earth_velocity = np.zeros((t_grid.size, 3))
    transforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    return {
        "station": station,
        "t_grid": t_grid,
        "x0": x0,
        "augmented": augmented,
        "earth_position": earth_position,
        "earth_velocity": earth_velocity,
        "transforms": transforms,
        "sxform": lambda _from, _to, _et: np.eye(6),
    }


def _synthetic_provider(case):
    return make_exact_event_station_state_provider(
        case["station"],
        12345.0,
        case["t_grid"],
        case["earth_position"],
        case["earth_velocity"],
        sxform_fn=case["sxform"],
    )


def _reference_config(delay_s: float = 0.0) -> CountedDopplerReferenceConfig:
    return CountedDopplerReferenceConfig(
        count_interval_s=1.0,
        tolerance_s=1e-13,
        equation_tolerance_s=1e-12,
        max_iter=50,
        transponder_delay_s=delay_s,
    )


@pytest.fixture(scope="module")
def spice_epoch_s():
    try:
        return campaign._spice_epoch_and_loader()
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f"R2 real-SPICE fixture unavailable: {exc}")


@pytest.fixture(scope="module")
def real_fixture(spice_epoch_s):
    return campaign.build_campaign_fixture(
        campaign.frozen_geometry_specs()[0], 3.0, et0_s=spice_epoch_s
    )


@pytest.fixture(scope="module")
def dimension_campaign(spice_epoch_s):
    return campaign.run_decomposition_campaign(
        et0_s=spice_epoch_s,
        intervals_s=campaign.FROZEN_INTERVALS_S,
        cadences_s=campaign.FROZEN_TRANSFORM_CADENCES_S,
        geometries=(campaign.frozen_geometry_specs()[0],),
    )


@pytest.fixture(scope="module")
def quick_campaign_dir(tmp_path_factory, spice_epoch_s):
    output_dir = tmp_path_factory.mktemp("r2_quick_campaign")
    manifest = campaign.run_campaign(output_dir, quick=True)
    assert manifest["quick"] is True
    return output_dir


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def test_reference_config_accepts_finite_nonnegative_delay():
    for delay in (0.0, 1e-6, 1e-3):
        config = _reference_config(delay)
        assert config.transponder_delay_s == delay
        assert config.as_two_way_range_config().transponder_delay_s == delay


@pytest.mark.parametrize("delay", [-1e-12, np.nan, np.inf, -np.inf])
def test_reference_config_rejects_negative_or_nonfinite_delay(delay):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _reference_config(delay)


def test_legacy_counted_doppler_nonzero_delay_remains_fail_closed():
    with pytest.raises(ValueError, match="legacy single-bounce counted-Doppler"):
        RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=60.0,
            transponder_delay_s=1e-6,
        )


def test_exact_station_provider_uses_target_from_source_sxform_direction():
    station = Station(
        "direction fixture", 17.0, 42.0, 100.0, (1.0, 0.0, 0.0), 5.0, 1e-4, 1e-3
    )
    t_grid = np.array([-1.0, 1.0])
    earth_position = np.repeat(np.array([[100.0, 200.0, 300.0]]), 2, axis=0)
    earth_velocity = np.repeat(np.array([[1.0, 2.0, 3.0]]), 2, axis=0)
    angle = np.deg2rad(90.0)
    c_matrix = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xform = np.zeros((6, 6))
    xform[:3, :3] = c_matrix
    xform[3:, 3:] = c_matrix
    calls: list[tuple[str, str, float]] = []

    def sxform_fn(source, target, et):
        calls.append((source, target, et))
        return xform

    provider = make_exact_event_station_state_provider(
        station,
        1000.0,
        t_grid,
        earth_position,
        earth_velocity,
        sxform_fn=sxform_fn,
    )
    state = provider.state(0.25)
    fixed = np.concatenate([station.r_ecef_m, np.zeros(3)])
    expected = np.array([100.0, 200.0, 300.0, 1.0, 2.0, 3.0]) + np.linalg.solve(
        xform, fixed
    )
    np.testing.assert_array_equal(state, expected)
    assert calls == [("J2000", "ITRF93", 1000.25)]
    assert provider.exact_sxform_call_count == 1
    assert provider.station_state_method == REFERENCE_STATION_STATE_METHOD


def test_single_bounce_reference_keeps_zero_delay_contract(synthetic_case):
    result = exact_station_single_bounce_counted_doppler_reference(
        0.0,
        _synthetic_provider(synthetic_case),
        synthetic_case["t_grid"],
        synthetic_case["augmented"][:, :6],
        _reference_config(),
    )
    assert result.model_id == EXACT_STATION_MODEL_ID
    assert result.endpoint_solution_count == 2
    assert result.physical_event_count == 6
    assert result.start_solution.t1_s < result.start_solution.t2_s < result.start_solution.t3_s
    assert result.end_solution.t1_s < result.end_solution.t2_s < result.end_solution.t3_s
    with pytest.raises(ValueError, match="zero transponder delay"):
        exact_station_single_bounce_counted_doppler_reference(
            0.0,
            _synthetic_provider(synthetic_case),
            synthetic_case["t_grid"],
            synthetic_case["augmented"][:, :6],
            _reference_config(1e-6),
        )


def test_four_event_reference_has_two_endpoint_eight_event_ordering(synthetic_case):
    delay = 1e-6
    result = exact_station_four_event_counted_doppler_reference(
        0.0,
        _synthetic_provider(synthetic_case),
        synthetic_case["t_grid"],
        synthetic_case["augmented"][:, :6],
        _reference_config(delay),
    )
    assert result.model_id == FOUR_EVENT_MODEL_ID
    assert result.endpoint_solution_count == 2
    assert result.physical_event_count == 8
    for solution in (result.start_solution, result.end_solution):
        assert solution.t1_s < solution.t2u_s <= solution.t2d_s < solution.t3_s
        relative = abs((solution.t2d_s - solution.t2u_s) - delay) / delay
        assert relative <= 1e-9
    assert result.observable == pytest.approx(
        (result.end_range_m - result.start_range_m) / result.nominal_count_interval_s,
        rel=0.0,
        abs=1e-12,
    )


def test_zero_delay_four_event_reference_is_deterministic(synthetic_case):
    values = []
    events = []
    for _ in range(2):
        result = exact_station_four_event_counted_doppler_reference(
            0.0,
            _synthetic_provider(synthetic_case),
            synthetic_case["t_grid"],
            synthetic_case["augmented"][:, :6],
            _reference_config(),
        )
        values.append(result.observable)
        events.append(
            (
                result.start_solution.t1_s,
                result.start_solution.t2u_s,
                result.start_solution.t2d_s,
                result.end_solution.t1_s,
                result.end_solution.t2u_s,
                result.end_solution.t2d_s,
            )
        )
    assert values[0] == values[1]
    assert events[0] == events[1]


def test_generation_and_direct_reference_share_endpoint_helper(synthetic_case):
    receive_times = np.array([-0.1, 0.0, 0.1])
    generated, generated_results = generate_four_event_counted_doppler_reference(
        receive_times,
        _synthetic_provider(synthetic_case),
        synthetic_case["t_grid"],
        synthetic_case["augmented"][:, :6],
        _reference_config(),
        noise_sigma=0.0,
    )
    direct = [
        exact_station_four_event_counted_doppler_reference(
            time_s,
            _synthetic_provider(synthetic_case),
            synthetic_case["t_grid"],
            synthetic_case["augmented"][:, :6],
            _reference_config(),
        )
        for time_s in receive_times
    ]
    np.testing.assert_array_equal(generated[:, 0], receive_times)
    np.testing.assert_array_equal(generated[:, 1], [row.observable for row in direct])
    assert [row.observable for row in generated_results] == [row.observable for row in direct]


def test_lsf_observable_and_jacobian_triangle_identities(synthetic_case):
    decomposition = evaluate_lsf_decomposition(
        0.0,
        synthetic_case["station"],
        12345.0,
        synthetic_case["t_grid"],
        synthetic_case["augmented"],
        synthetic_case["earth_position"],
        synthetic_case["earth_velocity"],
        synthetic_case["transforms"],
        campaign._legacy_config(1.0),
        sxform_fn=synthetic_case["sxform"],
    )
    assert decomposition.observable_triangle_residual == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(decomposition.jacobian_triangle_residual, 0.0, atol=1e-14)
    assert decomposition.exact_station_observable == pytest.approx(
        decomposition.legacy_observable, rel=0.0, abs=1e-7
    )
    assert decomposition.exact_station_sxform_calls > 0
    assert decomposition.four_event_sxform_calls > 0


def test_four_event_analytic_jacobian_matches_fd_sweep(spice_epoch_s):
    rows = campaign.run_reference_fd_sweep(
        et0_s=spice_epoch_s, spec=campaign.frozen_geometry_specs()[0]
    )
    selected = [row for row in rows if row["selected_for_gate"]]
    assert len(selected) == 6
    assert max(float(row["relative_error"]) for row in selected) <= 1e-6
    assert all(row["denominator_policy"] == campaign.CONTROLLED_EXCLUSION_POLICY for row in rows)
    for column in range(6):
        steps = [
            float(row["perturbation_step"])
            for row in rows
            if int(row["state_column"]) == column
        ]
        assert len(steps) >= 11
        assert min(steps) < max(steps)


def test_frozen_campaign_dimensions_and_six_geometry_classes():
    assert campaign.FROZEN_INTERVALS_S == (1.0, 10.0, 30.0, 60.0, 100.0)
    assert campaign.FROZEN_TRANSFORM_CADENCES_S == (0.1, 1.0, 3.0, 10.0, 30.0, 60.0, 120.0)
    assert campaign.FROZEN_TRANSPONDER_DELAYS_S == (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
    specs = campaign.frozen_geometry_specs()
    assert len(specs) == 6
    labels = " ".join(f"{spec.geometry_id} {spec.description}" for spec in specs).lower()
    for required in ("nominal", "low", "high", "difficult", "rapid", "long"):
        assert required in labels
    assert any(spec.history_end_s - spec.history_start_s <= 620.0 for spec in specs)
    assert any(spec.history_end_s - spec.history_start_s >= 3600.0 for spec in specs)


def test_interpolation_cadence_and_receive_interval_cross_product(dimension_campaign):
    rows = dimension_campaign["observable"]
    assert len(rows) == len(campaign.FROZEN_INTERVALS_S) * len(
        campaign.FROZEN_TRANSFORM_CADENCES_S
    )
    assert {float(row["count_interval_s"]) for row in rows} == set(
        campaign.FROZEN_INTERVALS_S
    )
    assert {float(row["legacy_transform_cadence_s"]) for row in rows} == set(
        campaign.FROZEN_TRANSFORM_CADENCES_S
    )
    assert all(abs(float(row["observable_triangle_residual_mps"])) <= 1e-14 for row in rows)
    assert len(dimension_campaign["jacobian"]) == 6 * len(rows)


def test_real_campaign_uses_exact_event_sxform_and_reports_station_budget(dimension_campaign):
    assert all(
        row["station_state_method_S"] == REFERENCE_STATION_STATE_METHOD
        and row["station_state_method_F"] == REFERENCE_STATION_STATE_METHOD
        for row in dimension_campaign["observable"]
    )
    assert all(int(row["S_exact_sxform_calls"]) > 0 for row in dimension_campaign["observable"])
    assert all(int(row["F_exact_sxform_calls"]) > 0 for row in dimension_campaign["observable"])
    assert max(float(row["station_position_error_m"]) for row in dimension_campaign["station"]) > 0.0


def test_delay_campaign_covers_frozen_values_and_p04(quick_campaign_dir):
    rows = _csv_rows(quick_campaign_dir / "r2_transponder_delay_sensitivity.csv")
    assert {float(row["transponder_delay_s"]) for row in rows} == set(
        campaign.FROZEN_TRANSPONDER_DELAYS_S
    )
    positive = [row for row in rows if float(row["transponder_delay_s"]) > 0.0]
    assert positive
    assert max(float(row["relative_delay_equation_error"]) for row in positive) <= 1e-9
    assert any(abs(float(row["observable_shift_from_zero_mps"])) > 0.0 for row in positive)


def test_estimator_impact_is_finite_deterministic_and_model_paired(quick_campaign_dir):
    rows = _csv_rows(quick_campaign_dir / "r2_estimator_state_bias_impact.csv")
    assert len(rows) == 14
    assert {row["estimator"] for row in rows} == {"BLS_LM", "SRIF"}
    assert {int(row["noise_seed"]) for row in rows} == {campaign.FROZEN_RANDOM_SEED}
    for row in rows:
        assert _truth(row["legacy_operational_success"])
        assert _truth(row["reference_operational_success"])
        assert _truth(row["legacy_converged"]) == _truth(row["reference_converged"])
        assert float(row["legacy_covariance_symmetry_error"]) <= 1e-10
        assert float(row["reference_covariance_symmetry_error"]) <= 1e-10
        assert float(row["legacy_covariance_min_eigenvalue"]) >= -1e-12
        assert float(row["reference_covariance_min_eigenvalue"]) >= -1e-12


def test_cost_campaign_characterizes_l_s_f_without_production_gate(quick_campaign_dir):
    rows = _csv_rows(quick_campaign_dir / "r2_accuracy_cost_tradeoff.csv")
    assert {row["model_id"] for row in rows} == {
        LEGACY_MODEL_ID,
        EXACT_STATION_MODEL_ID,
        FOUR_EVENT_MODEL_ID,
    }
    by_model = {row["model_id"]: row for row in rows}
    assert float(by_model[LEGACY_MODEL_ID]["relative_combined_runtime_ratio"]) == 1.0
    call_column = "exact_sxform_call_count_per_observable_plus_jacobian"
    assert int(by_model[LEGACY_MODEL_ID][call_column]) == 0
    assert int(by_model[EXACT_STATION_MODEL_ID][call_column]) > 0
    assert int(by_model[FOUR_EVENT_MODEL_ID][call_column]) > 0
    assert all(float(row["combined_runtime_s"]) > 0.0 for row in rows)
    assert len({int(row["history_grid_memory_bytes"]) for row in rows}) == 1


def test_denominator_policy_rejects_nonfinite_and_records_controlled_exclusions(
    quick_campaign_dir,
):
    with pytest.raises(RuntimeError, match="non-finite"):
        campaign._assert_finite_rows([{"bad": np.nan}], label="test")
    jacobian = _csv_rows(quick_campaign_dir / "r2_jacobian_error_decomposition.csv")
    fd_rows = _csv_rows(quick_campaign_dir / "r2_reference_jacobian_fd_sweep.csv")
    assert all(row["denominator_policy"] == campaign.CONTROLLED_EXCLUSION_POLICY for row in jacobian)
    assert all(row["denominator_policy"] == campaign.CONTROLLED_EXCLUSION_POLICY for row in fd_rows)
    for basename in (contract["basename"] for contract in campaign.PLOT_CONTRACT):
        rows = _csv_rows(quick_campaign_dir / f"{basename}.csv")
        for row in rows:
            assert all(
                value.strip().lower()
                not in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}
                for value in row.values()
            )


def _independent_headline(contract: dict[str, object], rows: list[dict[str, str]]) -> float:
    selected = rows
    selector = contract.get("selector")
    if selector:
        selected = [row for row in rows if _truth(row[str(selector)])]
    values = [float(row[str(contract["column"])]) for row in selected]
    assert values and np.all(np.isfinite(values))
    return max(abs(value) for value in values)


def test_each_plot_headline_recomputes_independently_from_its_own_csv(quick_campaign_dir):
    validation = {
        row["plot_id"]: row
        for row in _csv_rows(quick_campaign_dir / "r2_plot_validation_matrix.csv")
    }
    assert set(validation) == {str(row["plot_id"]) for row in campaign.PLOT_CONTRACT}
    for contract in campaign.PLOT_CONTRACT:
        source = quick_campaign_dir / f"{contract['basename']}.csv"
        rows = _csv_rows(source)
        independently_recomputed = _independent_headline(contract, rows)
        recorded = validation[str(contract["plot_id"])]
        assert float(recorded["headline_value"]) == independently_recomputed
        assert recorded["source_csv"] == source.name
        assert recorded["formula_id"] == contract["formula_id"]
        assert recorded["denominator_policy"] == contract["denominator_policy"]
        assert recorded["metric_contract_version"] == campaign.METRIC_CONTRACT_VERSION
        assert int(recorded["source_csv_rows"]) == len(rows)
        metadata = json.loads(recorded["scenario_metadata"])
        assert metadata["intervals_s"] == [1.0, 60.0]
        assert metadata["cadences_s"] == [1.0, 60.0]
        assert metadata["geometry_count"] == 2


def test_seven_authoritative_csv_png_pairs_exist_and_are_nonempty(quick_campaign_dir):
    assert len(campaign.PLOT_CONTRACT) == 7
    for contract in campaign.PLOT_CONTRACT:
        basename = str(contract["basename"])
        csv_path = quick_campaign_dir / f"{basename}.csv"
        png_path = quick_campaign_dir / f"{basename}.png"
        assert csv_path.stat().st_size > 100
        assert png_path.stat().st_size > 1000
        assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_scientific_decision_uses_zero_delay_lsf_not_hypothetical_delay(quick_campaign_dir):
    row = _csv_rows(quick_campaign_dir / "r2_decision_summary.csv")[0]
    assert float(row["max_abs_zero_delay_four_event_error_over_sigma"]) <= 0.1
    assert float(row["max_abs_delay_sensitivity_over_sigma"]) > 0.0
    assert row["scientific_decision"] == "EXACT_STATION_TRANSFORM_UPGRADE_REQUIRED"
    assert not _truth(row["operational_success_changed"])
    assert not _truth(row["convergence_changed"])


def test_report_and_decision_qualify_the_full_cadence_envelope(tmp_path):
    cadence_values = {
        0.1: 0.0692770208843285,
        1.0: 0.170714924934146,
        3.0: 198.871549109754,
        10.0: 728.172058966265,
        30.0: 3189.77691492364,
        60.0: 6882.61153566145,
        120.0: 14268.279471039,
    }
    observable_rows = [
        {
            "legacy_transform_cadence_s": cadence_s,
            "frame_error_over_sigma": value,
            "four_event_error_over_sigma": 0.0,
            "total_error_over_sigma": value,
        }
        for cadence_s, value in cadence_values.items()
    ]
    estimator_rows = [
        {
            "estimator": estimator,
            "parameter_row": True,
            "estimate_shift_over_reference_posterior_sigma": 0.009216843912656895,
            "operational_success_changed": False,
            "legacy_converged": False,
            "reference_converged": False,
            "legacy_operational_success": True,
        }
        for estimator in ("BLS_LM", "SRIF")
    ]
    decision = campaign._decision_rows(
        observable_rows,
        [{"observable_shift_over_sigma": 0.22590160369873047}],
        estimator_rows,
    )[0]
    qualification = json.loads(str(decision["cadence_qualification_json"]))
    assert [row["legacy_transform_cadence_s"] for row in qualification] == list(
        cadence_values
    )
    assert [row["classification"] for row in qualification] == [
        "negligible",
        "material",
        "blocker",
        "blocker",
        "blocker",
        "blocker",
        "blocker",
    ]
    assert decision["scientific_decision"] == "EXACT_STATION_TRANSFORM_UPGRADE_REQUIRED"
    assert decision["envelope_classification"] == "production_blocker"
    assert decision["every_production_configuration_classification"] == "NOT CLAIMED"
    assert decision["finest_cadence_classification"] == "negligible"
    assert decision["one_second_classification"] == "material"
    assert float(decision["worst_case_cadence_s"]) == 120.0

    cost_rows = [
        {
            "model_id": model_id,
            "implementation_nominal_relative_combined_runtime_ratio": (
                campaign.IMPLEMENTATION_NOMINAL_COST_RATIOS[model_id]
            ),
            "independent_validation_relative_combined_runtime_ratio": (
                campaign.INDEPENDENT_VALIDATION_COST_RATIOS[model_id]
            ),
            "structural_exact_sxform_call_count": (
                campaign.STRUCTURAL_EXACT_SXFORM_CALLS[model_id]
            ),
        }
        for model_id in (LEGACY_MODEL_ID, EXACT_STATION_MODEL_ID, FOUR_EVENT_MODEL_ID)
    ]
    report = campaign.write_implementation_report(
        tmp_path,
        decision,
        [
            {
                "acceptance_id": "R2-P08",
                "value": 1.0,
                "source_csv": "observable.csv",
                "formula_id": "p08",
            },
            {
                "acceptance_id": "R2-P09",
                "value": 0.01,
                "source_csv": "jacobian.csv",
                "formula_id": "p09",
                "denominator_policy": campaign.CONTROLLED_EXCLUSION_POLICY,
            },
        ],
        [{"plot_id": "R2-PLOT-4", "headline_value": 1e-7}],
        [{"pass": True}],
        observable_rows,
        estimator_rows,
        cost_rows,
    ).read_text(encoding="utf-8")
    expected_report_values = {
        0.1: "0.069",
        1.0: "0.171",
        3.0: "199",
        10.0: "728",
        30.0: "3190",
        60.0: "6883",
        120.0: "14268",
    }
    for cadence_s, value in expected_report_values.items():
        assert f"| {cadence_s:g} s | {value} |" in report
    assert "every production configuration is blocked is `NOT CLAIMED`" in report
    assert "Current production cadence is\nscenario-dependent" in report
    assert "current production is already a blocker" not in report.lower()


def test_canonical_patch_provenance_is_reproducible(tmp_path):
    git = shutil.which("git")
    if git is None:
        windows_git = Path(r"C:\Program Files\Git\cmd\git.exe")
        git = str(windows_git) if windows_git.is_file() else None
    assert git is not None
    patch_path = tmp_path / "r2_measurement_fidelity.patch"
    command = [
        git,
        "-C",
        str(ROOT),
        "diff",
        "--binary",
        "--full-index",
        campaign.R2_BASELINE_COMMIT,
        campaign.R2_ACCEPTED_IMPLEMENTATION_COMMIT,
        f"--output={patch_path}",
    ]
    subprocess.run(command, check=True)
    patch_bytes = patch_path.read_bytes()
    assert len(patch_bytes) == campaign.R2_CANONICAL_PATCH_BYTE_COUNT
    assert hashlib.sha256(patch_bytes).hexdigest() == campaign.R2_CANONICAL_PATCH_SHA256
    assert not patch_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in patch_bytes
    assert patch_bytes.startswith(b"diff --git a/")

    changed_paths = subprocess.run(
        [
            git,
            "-C",
            str(ROOT),
            "diff",
            "--name-only",
            campaign.R2_BASELINE_COMMIT,
            campaign.R2_ACCEPTED_IMPLEMENTATION_COMMIT,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert changed_paths == [
        "examples/r2_measurement_fidelity_validation.py",
        "lunar_od/two_way_counted_doppler_reference.py",
        "tests/test_r2_measurement_fidelity.py",
    ]
    tree = subprocess.run(
        [git, "-C", str(ROOT), "rev-parse", f"{campaign.R2_ACCEPTED_IMPLEMENTATION_COMMIT}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tree == campaign.R2_ACCEPTED_IMPLEMENTATION_TREE
    reference_blob = subprocess.run(
        [
            git,
            "-C",
            str(ROOT),
            "rev-parse",
            f"{campaign.R2_ACCEPTED_IMPLEMENTATION_COMMIT}:lunar_od/two_way_counted_doppler_reference.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert reference_blob == "1b2b0b1ff42c1004dfceeabf0f7180c4fdea5215"


def test_estimator_metadata_qualifies_nonconvergence_alias_and_selector(
    quick_campaign_dir,
):
    primary = quick_campaign_dir / campaign.ESTIMATOR_PRIMARY_ARTIFACT
    alias = quick_campaign_dir / campaign.ESTIMATOR_PLOT_ALIAS_ARTIFACT
    assert primary.read_bytes() == alias.read_bytes()
    rows = _csv_rows(primary)
    assert len(rows) == 14
    assert max(
        abs(float(row["estimate_shift_over_reference_posterior_sigma"]))
        for row in rows
    ) == 0.009216843912656895
    for row in rows:
        assert not _truth(row["legacy_converged"])
        assert not _truth(row["reference_converged"])
        assert _truth(row["legacy_operational_success"])
        assert _truth(row["reference_operational_success"])
        assert not _truth(row["legacy_strict_step_tolerance_met"])
        assert not _truth(row["reference_strict_step_tolerance_met"])
        assert not _truth(row["convergence_changed"])
        assert int(row["max_iterations"]) == campaign.ESTIMATOR_MAX_ITERATIONS
        assert row["result_qualification"] == campaign.ESTIMATOR_RESULT_QUALIFICATION
        assert row["converged_posterior_solution_claim"] == "NOT_CLAIMED"
        assert row["parameter_row_semantics"] == campaign.PARAMETER_ROW_SEMANTICS
        assert row["primary_scientific_artifact"] == primary.name
        assert row["plot_contract_alias_source_artifact"] == alias.name
        assert row["artifact_relationship"] == campaign.ESTIMATOR_ARTIFACT_RELATIONSHIP
    manifest = json.loads((quick_campaign_dir / "r2_campaign_manifest.json").read_text())
    assert manifest["parameter_row_semantics"] == campaign.PARAMETER_ROW_SEMANTICS
    assert manifest["estimator_artifacts"]["relationship"] == (
        campaign.ESTIMATOR_ARTIFACT_RELATIONSHIP
    )


def test_cost_metadata_is_informational_and_records_timing_variability(
    quick_campaign_dir,
):
    primary = quick_campaign_dir / "r2_accuracy_cost_tradeoff.csv"
    alias = quick_campaign_dir / "r2_cost_summary.csv"
    assert primary.read_bytes() == alias.read_bytes()
    rows = _csv_rows(primary)
    by_model = {row["model_id"]: row for row in rows}
    for model_id in (LEGACY_MODEL_ID, EXACT_STATION_MODEL_ID, FOUR_EVENT_MODEL_ID):
        row = by_model[model_id]
        assert row["metric_qualification"] == campaign.COST_METRIC_QUALIFICATION
        assert row["timing_variability"] == campaign.COST_TIMING_VARIABILITY
        assert row["production_performance_guarantee"] == "NOT_CLAIMED"
        assert row["qualitative_conclusion"] == campaign.COST_QUALITATIVE_CONCLUSION
        assert float(row["implementation_nominal_relative_combined_runtime_ratio"]) == (
            campaign.IMPLEMENTATION_NOMINAL_COST_RATIOS[model_id]
        )
        assert float(row["independent_validation_relative_combined_runtime_ratio"]) == (
            campaign.INDEPENDENT_VALIDATION_COST_RATIOS[model_id]
        )
        assert int(row["structural_exact_sxform_call_count"]) == (
            campaign.STRUCTURAL_EXACT_SXFORM_CALLS[model_id]
        )
    report = (quick_campaign_dir / "R2_IMPLEMENTATION_REPORT.md").read_text()
    assert "no production performance guarantee is claimed" in report
    assert "R2-P16 remains informational" in report


def test_strict_option_b_protected_production_files_are_byte_identical():
    for relative_path, expected in campaign.BASELINE_FILE_SHA256.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected, relative_path


def _force_contract(j2_moon: float):
    config = scenario_config_from_mapping(
        {
            "name": "r2-test",
            "measurement_type": "range_rate",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
            "j2_moon": j2_moon,
        }
    )
    return force_model_contract_from_scenario_config(config)


def test_r0b_fingerprints_schema_and_r1_readiness_are_frozen():
    zero = _force_contract(0.0)
    lunar_j2 = _force_contract(float(J2_MOON_UNNORMALIZED))
    assert zero.force_model_fingerprint() == campaign.FROZEN_ZERO_J2_FINGERPRINT
    assert lunar_j2.force_model_fingerprint() == campaign.FROZEN_LUNAR_J2_FINGERPRINT
    assert FORCE_CONTRACT_SCHEMA_VERSION == "r0b.force-model-contract.v1"
    payload = lunar_j2.to_canonical_payload()
    assert payload["consumer_capabilities"]["posterior_covariance"] == "verified"
    assert payload["consumer_capabilities"]["observability"] == "verified"


def test_earth_j2_and_harmonics_remain_fail_closed():
    earth = consumer_capabilities_for(
        lunar_j2_on=False, earth_j2_on=True, harmonics_on=False
    )
    assert all(value is ConsumerReadiness.UNSUPPORTED for value in earth.values())
    harmonics = consumer_capabilities_for(
        lunar_j2_on=False, earth_j2_on=False, harmonics_on=True
    )
    assert (
        harmonics[ConsumerRole.TRUTH_STATE]
        is ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY
    )
    for role in (
        ConsumerRole.ESTIMATOR_STATE,
        ConsumerRole.ESTIMATOR_STM,
        ConsumerRole.POSTERIOR_COVARIANCE,
        ConsumerRole.OBSERVABILITY,
    ):
        assert harmonics[role] is ConsumerReadiness.UNSUPPORTED


def test_ukf_two_way_range_rejection_is_unchanged():
    with pytest.raises(ValueError, match="not supported by the UKF in M3"):
        run_lunar_ukf(
            [0.0],
            np.zeros((1, 2)),
            np.zeros(6),
            np.eye(6),
            None,
            1.0,
            1.0,
            1.0,
            lambda _t: np.zeros(3),
            lambda _t: np.zeros(3),
            measurement_type="two_way_range",
        )


def test_compatibility_summary_carries_forward_1088_value_zero_j2_gate(
    quick_campaign_dir,
):
    rows = _csv_rows(quick_campaign_dir / "r2_compatibility_summary.csv")
    assert all(_truth(row["pass"]) for row in rows)
    numeric = next(row for row in rows if row["item"] == "accepted_r1_zero_j2_vector")
    assert numeric["expected"] == "1088 values; 0 mismatch; 0 ULP"
    assert numeric["actual"] == numeric["expected"]
    assert "unchanged" in numeric["evidence"]


def test_reference_module_is_opt_in_and_absent_from_production_dispatch():
    protected = (
        "lunar_od/measurements.py",
        "lunar_od/scenario_config.py",
        "lunar_od/scenarios.py",
        "lunar_od/reporting.py",
        "lunar_od/filters.py",
        "examples/run_scenario_config.py",
        "desktop_app/controllers/analysis_controller.py",
    )
    for relative_path in protected:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "two_way_counted_doppler_reference" not in source
