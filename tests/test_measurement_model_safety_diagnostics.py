"""Measurement-model safety diagnostics (D1 evidence + safety contracts).

FA-01 and FA-02 sections assert the P0A production safety contracts (hard
rejection and corrected metadata); the retained operator-mismatch test
documents the underlying physics rationale for the rejection. FA-03A asserts
the P0B-1 strict convergence contract. FA-03B remains executable evidence for
the separate history-domain task.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import lunar_od
import lunar_od.measurements as measurements_module
import lunar_od.radiometrics as radiometrics_module
from lunar_od import (
    C_LIGHT_MPS,
    HistoryDomainError,
    LightTimeConvergenceError,
    PassGeometry,
    RangeRatePhysicsConfig,
    RoundTripLightTimeConvergenceError,
    TwoWayRangeConfig,
    generate_position_measurements,
    measurement_model_metadata,
    one_way_light_time_range_sensitivity,
    solve_one_way_light_time,
)
from lunar_od.filters import (
    _position_measurement_from_state,
    _two_way_local_histories,
    run_lunar_ukf,
    validate_ukf_measurement_support,
)
from lunar_od.geometry import wrap_to_pi
from lunar_od.measurements import (
    ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S,
    ONE_WAY_LIGHT_TIME_MAX_ITERATIONS,
    ONE_WAY_LIGHT_TIME_TOLERANCE_S,
)
from lunar_od.radiometrics import (
    interp_state_history,
    solve_two_way_light_time,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)
from lunar_od.scenario_config import ScenarioConfig, scenario_config_from_mapping
from lunar_od.scenarios import _resolve_range_rate_physics

_CN_CNS_PROFILES = (
    "one_way_light_time",
    "one_way_light_time_aberrated_local_mci",
    "one_way_light_time_aberrated_spice_ssb",
)


class _OriginStation:
    name = "D1 origin station"
    lat_rad = 0.0
    lon_rad = 0.0
    sigma_range_m = 5.0
    sigma_angle_rad = 1.0e-5
    sigma_range_rate_mps = 1.0e-4
    bias = ()

    @property
    def r_ecef_m(self) -> np.ndarray:
        return np.zeros(3, dtype=float)


def _sample_constant(vector: np.ndarray):
    value = np.asarray(vector, dtype=float).reshape(3)

    def sample(t_s):
        count = np.asarray(t_s, dtype=float).reshape(-1).size
        return np.repeat(value[None, :], count, axis=0)

    return sample


def _linear_state_history(t_grid_s: np.ndarray, position_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
    position = np.asarray(position_m, dtype=float).reshape(3)
    velocity = np.asarray(velocity_mps, dtype=float).reshape(3)
    states = np.zeros((t_grid.size, 6), dtype=float)
    states[:, :3] = position + t_grid[:, None] * velocity
    states[:, 3:6] = velocity
    return states


def _augmented_identity_history(states: np.ndarray) -> np.ndarray:
    phi_flat = np.eye(6).reshape(-1, order="F")
    return np.hstack([states, np.repeat(phi_flat[None, :], states.shape[0], axis=0)])


def _position_delta(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    delta = np.asarray(value, dtype=float) - np.asarray(reference, dtype=float)
    delta[1] = wrap_to_pi(delta[1])
    delta[2] = wrap_to_pi(delta[2])
    return delta


@pytest.fixture(scope="module")
def generated_profile_evidence():
    """Generate all position profiles on one deterministic synthetic history."""
    t_grid = np.linspace(-5.0, 5.0, 21)
    receive_index = int(np.where(np.isclose(t_grid, 0.0))[0][0])
    states = _linear_state_history(
        t_grid,
        np.array([3.0e8, 2.0e8, 1.0e8]),
        np.array([1200.0, -800.0, 450.0]),
    )
    station = _OriginStation()
    vis_mask = np.zeros((t_grid.size, 1), dtype=bool)
    vis_mask[receive_index, 0] = True
    earth_position = _sample_constant(np.zeros(3))
    earth_velocity = _sample_constant(np.array([1500.0, 22000.0, -2000.0]))
    ssb_velocity_mps = np.array([30000.0, -5000.0, 1000.0])
    ssb_state_km = np.concatenate([np.zeros(3), ssb_velocity_mps / 1000.0])

    evidence = {}
    profiles = (
        "geometric_instantaneous",
        "one_way_light_time",
        "one_way_light_time_aberrated_local_mci",
        "one_way_light_time_aberrated_spice_ssb",
    )
    with (
        mock.patch("spiceypy.sxform", return_value=np.eye(6)),
        mock.patch("spiceypy.spkezr", return_value=(ssb_state_km, 0.0)),
    ):
        for profile in profiles:
            _obs, pass_geo, clean = generate_position_measurements(
                t_grid,
                states,
                (station,),
                vis_mask,
                earth_position,
                earth_velocity,
                0.0,
                noise=False,
                measurement_model_profile=profile,
            )
            row = clean[0]
            ukf_value = _position_measurement_from_state(states[receive_index], row, pass_geo)
            evidence[profile] = {
                "generated": row[1:4].copy(),
                "pass_geo": pass_geo,
                "ukf": ukf_value,
            }
    return evidence


def test_fa06_pytest_session_imports_isolated_repository(pytestconfig):
    """FA-06 guard: pytest must import lunar_od from this isolated worktree."""
    repository_root = Path(__file__).resolve().parents[1]
    package_path = Path(lunar_od.__file__).resolve()
    relevant_sys_path = [entry for entry in sys.path if "python_port" in entry.lower()]

    print(f"[FA-06] pytest rootdir={pytestconfig.rootpath}")
    print(f"[FA-06] pytest config={pytestconfig.inipath}")
    print(f"[FA-06] repository root={repository_root}")
    print(f"[FA-06] lunar_od.__file__={package_path}")
    print(f"[FA-06] sys.executable={sys.executable}")
    print(f"[FA-06] relevant sys.path={relevant_sys_path}")
    print(f"[FA-06] effective pytest pythonpath={pytestconfig.getini('pythonpath')}")

    assert package_path.is_relative_to(repository_root)
    assert pytestconfig.getini("pythonpath") == []


def test_fa01_operator_mismatch_rationale_for_rejection(generated_profile_evidence):
    """Underlying operator mismatch motivating the P0A rejection.

    The private UKF position operator is geometric-only; against CN/CN+S
    generated observables it produces km-scale range and arcsec-scale angle
    deltas at the truth state. This is the physics rationale for the hard
    rejection asserted below, kept as executable evidence (the private
    operator itself is intentionally unchanged by P0A).
    """
    evidence = generated_profile_evidence
    geometric = evidence["geometric_instantaneous"]["generated"]
    arcsec_per_rad = 180.0 * 3600.0 / np.pi

    for profile, record in evidence.items():
        generated = record["generated"]
        ukf_value = record["ukf"]
        delta = _position_delta(ukf_value, generated)
        print(
            f"[FA-01] {profile}: UKF-selected delta "
            f"range={delta[0]:.9f} m, az={delta[1]:.12e} rad "
            f"({delta[1] * arcsec_per_rad:.6f} arcsec), "
            f"el={delta[2]:.12e} rad ({delta[2] * arcsec_per_rad:.6f} arcsec)"
        )
        np.testing.assert_allclose(ukf_value, geometric, rtol=0.0, atol=1.0e-12)
        assert record["pass_geo"].measurement_model_profile == profile

    np.testing.assert_allclose(
        evidence["geometric_instantaneous"]["ukf"],
        evidence["geometric_instantaneous"]["generated"],
        rtol=0.0,
        atol=1.0e-12,
    )
    for profile in (
        "one_way_light_time",
        "one_way_light_time_aberrated_local_mci",
        "one_way_light_time_aberrated_spice_ssb",
    ):
        delta = _position_delta(evidence[profile]["ukf"], evidence[profile]["generated"])
        assert abs(delta[0]) > 1.0 or np.linalg.norm(delta[1:]) > 1.0e-8


def test_fa01_loader_rejects_ukf_with_cn_and_cns_profiles():
    """P0A contract: scenario_config_from_mapping rejects UKF + CN/CN+S."""
    base = {
        "name": "p0a_ukf_profile_gate",
        "measurement_type": "position",
        "estimator_type": "ukf",
        "start_mode": "cold",
        "network": "multi",
        "jacobian_model": "implicit_light_time",
    }
    for profile in _CN_CNS_PROFILES:
        with pytest.raises(ValueError, match="only the geometric instantaneous"):
            scenario_config_from_mapping({**base, "measurement_model_profile": profile})
    # Legacy booleans are an equivalent non-geometric selection.
    with pytest.raises(ValueError, match="only the geometric instantaneous"):
        scenario_config_from_mapping(
            {
                "name": "p0a_ukf_legacy_boolean_gate",
                "measurement_type": "position",
                "estimator_type": "ukf",
                "start_mode": "cold",
                "network": "multi",
                "apply_light_time": True,
            }
        )


def test_fa01_runtime_rejects_direct_scenario_config_bypass(generated_profile_evidence):
    """P0A defense-in-depth: direct ScenarioConfig construction bypasses the
    loader, but every UKF position run funnels through run_lunar_ukf, which
    applies the same shared helper once per arc before any sigma-point work."""
    for profile in _CN_CNS_PROFILES:
        # Direct dataclass construction does NOT run cross-field validation:
        bypassed = ScenarioConfig(
            name="p0a_direct_bypass",
            measurement_type="position",
            estimator_type="ukf",
            start_mode="cold",
            network="multi",
            measurement_model_profile=profile,
        )
        assert bypassed.measurement_model_profile == profile

        # ... and the runtime boundary still rejects the combination:
        pass_geo = generated_profile_evidence[profile]["pass_geo"]
        with pytest.raises(ValueError, match="only the geometric instantaneous"):
            run_lunar_ukf(
                np.array([0.0]),
                np.zeros((1, 6)),
                np.zeros(6),
                np.eye(6),
                pass_geo,
                4.9028e12,
                0.0,
                0.0,
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
                lambda t: np.zeros((np.size(np.atleast_1d(t)), 3)),
            )


def test_fa01_supported_combinations_remain_accepted(generated_profile_evidence):
    """P0A contract: geometric UKF and CN/CN+S batch estimators stay valid;
    the M3 two_way_range UKF rejection is preserved."""
    geometric_ukf = scenario_config_from_mapping(
        {
            "name": "p0a_geometric_ukf",
            "measurement_type": "position",
            "estimator_type": "ukf",
            "start_mode": "cold",
            "network": "multi",
        }
    )
    assert geometric_ukf.measurement_model_profile == "geometric_instantaneous"

    for estimator in ("bls_lm", "srif"):
        for profile in _CN_CNS_PROFILES:
            config = scenario_config_from_mapping(
                {
                    "name": f"p0a_{estimator}_batch",
                    "measurement_type": "position",
                    "estimator_type": estimator,
                    "start_mode": "cold",
                    "network": "multi",
                    "measurement_model_profile": profile,
                    "jacobian_model": "implicit_light_time",
                }
            )
            assert config.measurement_model_profile == profile

    # Shared helper is a no-op for non-UKF and non-position combinations.
    validate_ukf_measurement_support("srif", "position", "one_way_light_time")
    validate_ukf_measurement_support("ukf", "range_rate", "geometric_instantaneous")

    # Geometric UKF pass geometry still passes the runtime gate.
    validate_ukf_measurement_support(
        "ukf",
        "position",
        generated_profile_evidence["geometric_instantaneous"]["pass_geo"].measurement_model_profile,
    )

    with pytest.raises(ValueError, match="not supported by the UKF"):
        scenario_config_from_mapping(
            {
                "name": "p0a_m3_ukf_preserved",
                "measurement_type": "two_way_range",
                "estimator_type": "ukf",
                "start_mode": "cold",
                "network": "multi",
            }
        )


def test_fa02_generation_metadata_reports_one_way_solver_policy(generated_profile_evidence):
    """P0A contract: position metadata reports the one-way solver constants."""
    pass_geo = generated_profile_evidence["one_way_light_time"]["pass_geo"]
    metadata = pass_geo.measurement_metadata
    regenerated_metadata = measurement_model_metadata(pass_geo, noise_enabled=False)

    print(
        "[FA-02] reported one-way tolerance/max_iter="
        f"{metadata['light_time_tolerance_s']:.3e}/{metadata['light_time_max_iter']}"
    )
    assert metadata == regenerated_metadata
    assert metadata["light_time_tolerance_s"] == ONE_WAY_LIGHT_TIME_TOLERANCE_S
    assert metadata["light_time_max_iter"] == ONE_WAY_LIGHT_TIME_MAX_ITERATIONS
    assert (
        metadata["light_time_equation_tolerance_s"]
        == ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S
    )
    # The constants must stay the actual defaults of the one-way solver.
    solution = solve_one_way_light_time(
        0.0,
        np.zeros(3),
        lambda t: np.array([3.0e8, 0.0, 0.0]),
    )
    assert solution.converged
    assert solution.update_converged
    assert solution.equation_residual_s <= ONE_WAY_LIGHT_TIME_EQUATION_TOLERANCE_S
    assert solution.iterations <= ONE_WAY_LIGHT_TIME_MAX_ITERATIONS


def test_fa02_counted_doppler_metadata_is_unchanged():
    """P0A contract: range-rate metadata keeps RangeRatePhysicsConfig values."""
    station = _OriginStation()
    t_grid = np.array([0.0, 10.0])
    counted = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        light_time_tolerance_s=5.0e-10,
        light_time_max_iter=15,
        light_time_equation_tolerance_s=7.0e-12,
    )
    pass_geo = PassGeometry(
        t_s=t_grid,
        earth_pos_mci_m=np.zeros((2, 3)),
        earth_vel_mci_mps=np.zeros((2, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], 2, axis=0),
        stations=(station,),
        measurement_type="range_rate",
        range_rate_physics=counted,
        measurement_model_profile="two_way_counted_doppler",
    )
    metadata = measurement_model_metadata(pass_geo, noise_enabled=False)
    assert metadata["light_time_tolerance_s"] == 5.0e-10
    assert metadata["light_time_max_iter"] == 15
    assert metadata["light_time_equation_tolerance_s"] == 7.0e-12
    overridden = _resolve_range_rate_physics(counted, count_interval_s=30.0)
    assert overridden.light_time_equation_tolerance_s == 7.0e-12


def test_p0a_legacy_counted_delay_guard_matrix():
    """P0A contract: legacy counted Doppler rejects any nonzero fixed delay.

    The fixed scalar delay term cancels directly in the endpoint RTLT
    difference, but nonzero delay still affects counted Doppler through the
    t2u/t2d separation, spacecraft motion during the delay, and the changed
    uplink/downlink event geometry — which the single-bounce model cannot
    represent, hence the hard gate until the four-event model exists.
    """
    RangeRatePhysicsConfig(mode="two_way_counted_doppler", transponder_delay_s=0.0)
    RangeRatePhysicsConfig(mode="two_way_counted_doppler", transponder_delay_s=-0.0)

    for delay in (1.0e-15, 2.5e-6, 4.0e-6):
        with pytest.raises(ValueError, match="single-bounce"):
            RangeRatePhysicsConfig(
                mode="two_way_counted_doppler", transponder_delay_s=delay
            )
    for delay in (-1.0e-6, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RangeRatePhysicsConfig(
                mode="two_way_counted_doppler", transponder_delay_s=delay
            )

    # Non-counted modes keep accepting a delay (direct solver-level studies),
    # and the M3 four-event model's nonzero-delay support is unaffected.
    RangeRatePhysicsConfig(transponder_delay_s=2.5e-6)
    m3_config = TwoWayRangeConfig(transponder_delay_s=2.5e-6)
    assert m3_config.transponder_delay_s == 2.5e-6


def test_p0b1_one_way_nonconverged_last_iterate_is_rejected():
    """P0B-1: diagnostic result remains inspectable but consumers reject it."""
    t_grid = np.linspace(-5.0, 5.0, 41)
    states = _linear_state_history(
        t_grid,
        np.array([3.0e8, 0.0, 0.0]),
        np.array([0.05 * C_LIGHT_MPS, 0.0, 0.0]),
    )
    station = _OriginStation()
    receive_time_s = 0.0
    target = lambda t_s: interp_state_history(t_grid, states, t_s)[:3]
    solution = solve_one_way_light_time(
        receive_time_s,
        station.r_ecef_m,
        target,
        tolerance_s=1.0e-15,
        max_iter=1,
    )
    range_at_returned_epoch_m = float(
        np.linalg.norm(target(solution.transmit_time_s) - station.r_ecef_m)
    )
    equation_residual_s = abs(
        solution.light_time_s - range_at_returned_epoch_m / C_LIGHT_MPS
    )

    print(
        f"[FA-03A one-way] converged={solution.converged}, iterations={solution.iterations}, "
        f"t_tx={solution.transmit_time_s:.12f} s, equation residual={equation_residual_s:.12e} s, "
        f"equivalent range={equation_residual_s * C_LIGHT_MPS:.6f} m"
    )
    assert not solution.converged
    assert not solution.update_converged
    assert solution.iterations == 1
    assert solution.equation_residual_s == pytest.approx(
        equation_residual_s, rel=0.0, abs=np.spacing(equation_residual_s)
    )
    assert equation_residual_s > 1.0e-6

    with pytest.raises(LightTimeConvergenceError, match="observable solve") as nominal_error:
        measurements_module._apparent_position_observable(
            receive_time_s,
            station,
            t_grid,
            states,
            np.zeros(3),
            np.eye(6),
            tolerance_s=1.0e-15,
            max_iter=1,
        )
    assert "equation residual" in str(nominal_error.value)
    assert "refusing to use the last iterate" in str(nominal_error.value)

    with pytest.raises(LightTimeConvergenceError, match="sensitivity solve"):
        one_way_light_time_range_sensitivity(
            receive_time_s,
            station,
            t_grid,
            states,
            np.zeros(3),
            np.eye(6),
            tolerance_s=1.0e-15,
            max_iter=1,
        )


def _round_trip_equation_residuals(solution, t_grid: np.ndarray, states: np.ndarray, config):
    sc_t2 = interp_state_history(t_grid, states, solution.transponder_time_s)[:3]
    geometric_light_time = float(np.linalg.norm(sc_t2) / config.light_speed_mps)
    downlink = (solution.receive_time_s - solution.transponder_time_s) - geometric_light_time
    uplink = (
        solution.transponder_time_s
        - config.transponder_delay_s
        - solution.transmit_time_s
        - geometric_light_time
    )
    return np.array([downlink, uplink], dtype=float)


def test_p0b1_counted_observable_and_jacobian_reject_nonconverged_endpoints():
    """P0B-1: counted nominal and H reject either failed endpoint."""
    t_grid = np.linspace(-5.0, 5.0, 41)
    states = _linear_state_history(
        t_grid,
        np.array([2.0e8, 0.0, 0.0]),
        np.array([0.03 * C_LIGHT_MPS, 0.0, 0.0]),
    )
    earth = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    station = _OriginStation()
    config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=0.2,
        light_time_tolerance_s=1.0e-15,
        light_time_max_iter=1,
    )
    endpoints = (-0.1, 0.1)
    solutions = [
        solve_two_way_light_time(t3, station, t_grid, states, earth, earth, xforms, config)
        for t3 in endpoints
    ]
    equation_residuals = np.concatenate(
        [_round_trip_equation_residuals(solution, t_grid, states, config) for solution in solutions]
    )
    max_residual_s = float(np.max(np.abs(equation_residuals)))

    print(
        f"[FA-03A counted] endpoint converged={[s.converged for s in solutions]}, "
        f"max equation residual={max_residual_s:.12e} s, "
        f"equivalent range={max_residual_s * C_LIGHT_MPS:.6f} m"
    )
    assert not any(solution.converged for solution in solutions)
    for solution, residuals in zip(solutions, equation_residuals.reshape(2, 2)):
        np.testing.assert_allclose(
            [
                solution.downlink_equation_residual_s,
                solution.uplink_equation_residual_s,
            ],
            np.abs(residuals),
            rtol=0.0,
            atol=4.0 * np.spacing(max_residual_s),
        )
    assert max_residual_s > 1.0e-6

    with pytest.raises(
        RoundTripLightTimeConvergenceError, match="count-start endpoint"
    ) as nominal_error:
        two_way_counted_doppler_observable(
            0.0, station, t_grid, states, earth, earth, xforms, config
        )
    assert "equation residuals" in str(nominal_error.value)
    assert "refusing to use the last iterate" in str(nominal_error.value)

    converged_start = replace(solutions[0], converged=True)
    with mock.patch.object(
        radiometrics_module,
        "solve_two_way_light_time",
        side_effect=[converged_start, solutions[1]],
    ):
        with pytest.raises(
            RoundTripLightTimeConvergenceError, match="count-end endpoint"
        ):
            two_way_counted_doppler_observable(
                0.0, station, t_grid, states, earth, earth, xforms, config
            )

    with pytest.raises(
        RoundTripLightTimeConvergenceError,
        match="count-start Jacobian endpoint",
    ):
        two_way_counted_doppler_initial_state_jacobian(
            0.0,
            station,
            t_grid,
            _augmented_identity_history(states),
            earth,
            earth,
            xforms,
            config,
        )


def test_p0b1_dual_criterion_rejects_equation_residual_after_updates_converge():
    """Equation closure remains mandatory even when loose update checks pass."""
    t_grid = np.linspace(-5.0, 5.0, 41)
    station = _OriginStation()
    one_way_states = _linear_state_history(
        t_grid,
        np.array([3.0e8, 0.0, 0.0]),
        np.array([0.05 * C_LIGHT_MPS, 0.0, 0.0]),
    )
    one_way_target = lambda t_s: interp_state_history(t_grid, one_way_states, t_s)[:3]
    one_way = solve_one_way_light_time(
        0.0,
        station.r_ecef_m,
        one_way_target,
        tolerance_s=1.0,
        equation_tolerance_s=1.0e-12,
        max_iter=1,
    )
    assert one_way.update_converged
    assert not one_way.converged
    assert one_way.equation_residual_s > 1.0e-12
    with pytest.raises(LightTimeConvergenceError, match="equation residual"):
        measurements_module._apparent_position_observable(
            0.0,
            station,
            t_grid,
            one_way_states,
            np.zeros(3),
            np.eye(6),
            tolerance_s=1.0,
            equation_tolerance_s=1.0e-12,
            max_iter=1,
        )

    counted_states = _linear_state_history(
        t_grid,
        np.array([2.0e8, 0.0, 0.0]),
        np.array([0.03 * C_LIGHT_MPS, 0.0, 0.0]),
    )
    earth = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    counted_config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=0.2,
        light_time_tolerance_s=1.0,
        light_time_equation_tolerance_s=1.0e-12,
        light_time_max_iter=1,
    )
    counted = solve_two_way_light_time(
        -0.1,
        station,
        t_grid,
        counted_states,
        earth,
        earth,
        xforms,
        counted_config,
    )
    assert counted.downlink_update_converged
    assert counted.uplink_update_converged
    assert not counted.converged
    assert max(
        counted.downlink_equation_residual_s,
        counted.uplink_equation_residual_s,
    ) > counted_config.light_time_equation_tolerance_s
    with pytest.raises(RoundTripLightTimeConvergenceError, match="equation residuals"):
        two_way_counted_doppler_observable(
            0.0,
            station,
            t_grid,
            counted_states,
            earth,
            earth,
            xforms,
            counted_config,
        )


@pytest.mark.parametrize("bad_tolerance", [0.0, -1.0, np.nan, np.inf])
def test_p0b1_equation_tolerances_must_be_finite_and_positive(bad_tolerance):
    with pytest.raises(ValueError, match="finite and positive"):
        solve_one_way_light_time(
            0.0,
            np.zeros(3),
            lambda _t: np.array([3.0e8, 0.0, 0.0]),
            equation_tolerance_s=bad_tolerance,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        RangeRatePhysicsConfig(light_time_equation_tolerance_s=bad_tolerance)


def test_fa03b_one_way_boundary_matrix_enforces_domain():
    """T006 (P0B-2B): one-way nominal and sensitivity enforce closed support.

    Exact endpoints and at-most-two-ULP representation excursions are served
    with endpoint samples; anything farther outside raises
    ``HistoryDomainError`` from BOTH the apparent observable and the
    sensitivity path, before any extrapolation. The solver's returned event
    variable is never clipped by the guard.
    """
    from lunar_od.history_domain import HistoryDomainError

    t_grid = np.array([0.0, 10.0])
    light_time_s = 0.25  # static target at exactly 0.25 s light time
    range_m = light_time_s * C_LIGHT_MPS
    states = _linear_state_history(t_grid, np.array([range_m, 0.0, 0.0]), np.zeros(3))
    station = _OriginStation()
    ulp_one = np.nextafter(1.0, np.inf) - 1.0  # policy scale S = 1 here

    def run_both(receive_time_s: float):
        z, transmit_time_s, _lt, _it = measurements_module._apparent_position_observable(
            receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
        )
        solution, sensitivity = one_way_light_time_range_sensitivity(
            receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
        )
        assert np.isfinite(z).all()
        assert np.isfinite(sensitivity.d_range_d_state).all()
        return z, transmit_time_s, solution

    # Accepted: transmit exactly at start; 1- and 2-ULP below start
    # (endpoint sample; raw solver event stays outside and unclipped);
    # interior; receive exactly at end; receive 2 ULP after end.
    accepted_cases = {
        "transmit exact start": 0.25,
        "transmit 1 ULP below start": 0.25 - 1.0 * ulp_one,
        "transmit 2 ULP below start": 0.25 - 2.0 * ulp_one,
        "interior": 5.0,
        "receive exact end": 10.0,
        "receive 2 ULP after end": 10.0 + 2.0 * (np.nextafter(10.0, np.inf) - 10.0),
    }
    for label, receive_time_s in accepted_cases.items():
        z, transmit_time_s, solution = run_both(receive_time_s)
        if label == "transmit 2 ULP below start":
            assert solution.transmit_time_s < t_grid[0]  # guard never clips events
        print(f"[FA-03B one-way] accepted {label}: t_tx={transmit_time_s:.18f}")

    # Rejected symmetrically: 3-ULP excursions and clearly unsupported
    # epochs, from BOTH the nominal observable and the sensitivity path.
    # (A receive tag beyond the history end fails on the solver's very first
    # receive-epoch probe — the D1 refinement — so 'transmit at end' driven
    # by an out-of-support receive tag is a rejection case by design.)
    rejected_cases = {
        "transmit before start": 0.20,
        "transmit 3 ULP below start": 0.25 - 3.0 * ulp_one,
        "receive 3 ULP after end": 10.0 + 3.0 * (np.nextafter(10.0, np.inf) - 10.0),
        "receive after end (transmit would hit end)": 10.25,
        "receive far after end": 10.30,
    }
    for label, receive_time_s in rejected_cases.items():
        with pytest.raises(HistoryDomainError) as nominal_error:
            measurements_module._apparent_position_observable(
                receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
            )
        with pytest.raises(HistoryDomainError) as sensitivity_error:
            one_way_light_time_range_sensitivity(
                receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
            )
        for err in (nominal_error.value, sensitivity_error.value):
            assert err.history_name == "spacecraft_state"
            assert err.support_start_s == 0.0 and err.support_end_s == 10.0
            assert err.outside_distance_s > 0.0
        print(
            f"[FA-03B one-way] rejected {label}: outside="
            f"{nominal_error.value.outside_distance_s:.3e} s, pre-roll="
            f"{nominal_error.value.required_pre_roll_s:.3e} s, post-roll="
            f"{nominal_error.value.required_post_roll_s:.3e} s"
        )


def test_fa03b_counted_boundary_matrix_enforces_domain():
    """T013 (P0B-2C1): counted nominal and Jacobian paths enforce support.

    The lower boundary is driven by the count-start uplink event and the
    upper boundary by the count-end receive event. Exact and one/two policy
    ULP excursions are served with endpoint samples; three policy ULP and
    larger excursions raise before an extrapolator is reached.
    """
    from lunar_od.history_domain import HistoryDomainError

    t_grid = np.array([0.0, 10.0])
    light_time_s = 0.25
    range_m = light_time_s * C_LIGHT_MPS
    states = _linear_state_history(
        t_grid, np.array([range_m, 0.0, 0.0]), np.zeros(3)
    )
    augmented = _augmented_identity_history(states)
    earth = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    station = _OriginStation()
    config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=0.5,
        light_time_tolerance_s=1.0e-13,
    )
    lower_policy_ulp = np.nextafter(1.0, np.inf) - 1.0
    upper_policy_ulp = np.nextafter(10.0, np.inf) - 10.0

    def run_both(midpoint_s: float):
        value = two_way_counted_doppler_observable(
            midpoint_s,
            station,
            t_grid,
            states,
            earth,
            earth,
            xforms,
            config,
        )
        jacobian = two_way_counted_doppler_initial_state_jacobian(
            midpoint_s,
            station,
            t_grid,
            augmented,
            earth,
            earth,
            xforms,
            config,
        )
        assert np.isfinite(value)
        assert np.isfinite(jacobian).all()

    accepted_cases = {
        "count-start uplink exact lower bound": 0.75,
        "count-start uplink 1 policy ULP below": 0.75 - lower_policy_ulp,
        "count-start uplink 2 policy ULP below": 0.75 - 2.0 * lower_policy_ulp,
        "interior": 5.0,
        "count-end receive exact upper bound": 9.75,
        "count-end receive 1 policy ULP above": 9.75 + upper_policy_ulp,
        "count-end receive 2 policy ULP above": 9.75 + 2.0 * upper_policy_ulp,
    }
    for label, midpoint_s in accepted_cases.items():
        run_both(midpoint_s)
        print(f"[FA-03B counted] accepted {label}: midpoint={midpoint_s:.18f}")

    rejected_cases = {
        "count-start uplink 3 policy ULP below": (
            0.75 - 3.0 * lower_policy_ulp,
            "count-start",
            "uplink",
        ),
        "count-start uplink clearly before": (0.70, "count-start", "uplink"),
        "count-end receive 3 policy ULP above": (
            9.75 + 3.0 * upper_policy_ulp,
            "count-end",
            "downlink",
        ),
        "count-end receive clearly after": (9.80, "count-end", "downlink"),
    }
    for label, (midpoint_s, endpoint_fragment, event_label) in rejected_cases.items():
        with pytest.raises(HistoryDomainError) as nominal_error:
            two_way_counted_doppler_observable(
                midpoint_s,
                station,
                t_grid,
                states,
                earth,
                earth,
                xforms,
                config,
            )
        with pytest.raises(HistoryDomainError) as jacobian_error:
            two_way_counted_doppler_initial_state_jacobian(
                midpoint_s,
                station,
                t_grid,
                augmented,
                earth,
                earth,
                xforms,
                config,
            )
        for error in (nominal_error.value, jacobian_error.value):
            assert endpoint_fragment in error.endpoint_label
            assert error.event_label == event_label
            assert error.support_start_s == 0.0
            assert error.support_end_s == 10.0
            assert error.outside_distance_s > 0.0
        print(
            f"[FA-03B counted] rejected {label}: outside="
            f"{nominal_error.value.outside_distance_s:.3e} s"
        )


def test_fa03b_history_domain_matrix_documents_current_behavior():
    """FA-03B matrix: counted UKF source histories reject unsupported
    local intervals after P0B-2C3, matching the strict counted paths."""
    t_grid = np.array([0.0, 10.0])
    range_m = 0.25 * C_LIGHT_MPS
    states = _linear_state_history(t_grid, np.array([range_m, 0.0, 0.0]), np.zeros(3))
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    station = _OriginStation()

    matrix = {"UKF counted local history": {}}

    counted_config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=0.2,
        light_time_tolerance_s=1.0e-13,
        light_time_max_iter=20,
    )
    pass_geo = PassGeometry(
        t_s=t_grid,
        earth_pos_mci_m=np.column_stack([t_grid, 2.0 * t_grid, 3.0 * t_grid]),
        earth_vel_mci_mps=np.zeros((t_grid.size, 3)),
        x_j2000_to_itrf93=xforms,
        stations=(station,),
        measurement_type="range_rate",
        range_rate_physics=counted_config,
    )
    ukf_cases = {
        "Before start": 2.00,
        "Exact start": 2.10,
        "Exact end": 9.90,
        "After end": 10.00,
    }
    far_body = _sample_constant(np.array([1.0e9, 2.0e9, 3.0e9]))
    for label, midpoint_s in ukf_cases.items():
        if label in {"Before start", "After end"}:
            with pytest.raises(HistoryDomainError) as caught:
                _two_way_local_histories(
                    midpoint_s,
                    states[0],
                    range_m,
                    pass_geo,
                    counted_config.count_interval_s,
                    counted_config.light_speed_mps,
                    0.0,
                    0.0,
                    0.0,
                    far_body,
                    far_body,
                    1.0e-10,
                    1.0e-12,
                    "taylor3",
                )
            assert caught.value.history_name == "earth_position_mci"
            assert caught.value.model_context == "two_way_counted_doppler_ukf_local"
            assert caught.value.consumer == "_two_way_local_histories"
            assert caught.value.event_label == "ukf-local-source-interval"
            matrix["UKF counted local history"][label] = "controlled rejection"
            continue

        local_t, local_state, local_earth_pos, local_earth_vel, local_xforms = _two_way_local_histories(
            midpoint_s,
            states[0],
            range_m,
            pass_geo,
            counted_config.count_interval_s,
            counted_config.light_speed_mps,
            0.0,
            0.0,
            0.0,
            far_body,
            far_body,
            1.0e-10,
            1.0e-12,
            "taylor3",
        )
        assert np.isfinite(local_state).all()
        assert np.isfinite(local_earth_vel).all()
        assert np.isfinite(local_xforms).all()
        np.testing.assert_allclose(local_earth_pos[:, 0], local_t, rtol=0.0, atol=1.0e-12)
        if label == "Exact start":
            assert local_t[0] == pytest.approx(t_grid[0], abs=1.0e-13)
        else:
            assert local_t[-1] == pytest.approx(t_grid[-1], abs=1.0e-13)
        matrix["UKF counted local history"][label] = "interpolation"

    for path, cells in matrix.items():
        print(
            f"[FA-03B] {path}: "
            + ", ".join(f"{boundary}={classification}" for boundary, classification in cells.items())
        )

    assert matrix == {
        "UKF counted local history": {
            "Before start": "controlled rejection",
            "Exact start": "interpolation",
            "Exact end": "interpolation",
            "After end": "controlled rejection",
        },
    }
