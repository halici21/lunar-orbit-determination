"""D1 executable evidence for known measurement-model safety defects.

These tests document confirmed current defects and provide executable
behavior-freeze evidence. They are not approval of that behavior and must be
updated when the corresponding production safety fixes are implemented.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import lunar_od
import lunar_od.measurements as measurements_module
from lunar_od import (
    C_LIGHT_MPS,
    PassGeometry,
    RangeRatePhysicsConfig,
    generate_position_measurements,
    measurement_model_metadata,
    one_way_light_time_range_sensitivity,
    solve_one_way_light_time,
)
from lunar_od.filters import _position_measurement_from_state, _two_way_local_histories
from lunar_od.geometry import wrap_to_pi
from lunar_od.radiometrics import (
    interp_state_history,
    solve_two_way_light_time,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)
from lunar_od.scenario_config import scenario_config_from_mapping


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


def test_fa01_current_ukf_operator_is_geometric_for_cn_profiles(generated_profile_evidence):
    """Current-defect evidence: UKF silently ignores selected CN/CN+S physics."""
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


def test_fa01_config_acceptance_and_m3_rejection_are_explicit():
    """Current config accepts inconsistent UKF profiles while retaining M3 rejection."""
    base = {
        "name": "d1_ukf_profile_evidence",
        "measurement_type": "position",
        "estimator_type": "ukf",
        "start_mode": "cold",
        "network": "multi",
        "jacobian_model": "implicit_light_time",
    }
    profiles = (
        "one_way_light_time",
        "one_way_light_time_aberrated_local_mci",
        "one_way_light_time_aberrated_spice_ssb",
    )
    for profile in profiles:
        config = scenario_config_from_mapping({**base, "measurement_model_profile": profile})
        assert config.estimator_type == "ukf"
        assert config.measurement_model_profile == profile

    with pytest.raises(ValueError, match="not supported by the UKF"):
        scenario_config_from_mapping(
            {
                **base,
                "measurement_type": "two_way_range",
                "measurement_model_profile": "geometric_instantaneous",
                "jacobian_model": "analytic_exact_geometric",
            }
        )


def test_fa02_generation_metadata_reports_range_rate_solver_defaults(generated_profile_evidence):
    """Current-defect evidence: one-way metadata reports the wrong solver contract."""
    pass_geo = generated_profile_evidence["one_way_light_time"]["pass_geo"]
    metadata = pass_geo.measurement_metadata
    regenerated_metadata = measurement_model_metadata(pass_geo, noise_enabled=False)
    signature = inspect.signature(solve_one_way_light_time)
    actual_tolerance = signature.parameters["tolerance_s"].default
    actual_max_iter = signature.parameters["max_iter"].default
    range_rate_defaults = RangeRatePhysicsConfig()

    print(
        "[FA-02] actual one-way tolerance/max_iter="
        f"{actual_tolerance:.3e}/{actual_max_iter}; reported="
        f"{metadata['light_time_tolerance_s']:.3e}/{metadata['light_time_max_iter']}"
    )
    assert metadata == regenerated_metadata
    assert metadata["light_time_tolerance_s"] == range_rate_defaults.light_time_tolerance_s
    assert metadata["light_time_max_iter"] == range_rate_defaults.light_time_max_iter
    assert metadata["light_time_tolerance_s"] != actual_tolerance
    assert metadata["light_time_max_iter"] != actual_max_iter


def test_fa03a_one_way_nominal_consumes_nonconverged_last_iterate():
    """Current-defect evidence: nominal CN continues where its Jacobian refuses."""
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
    z, transmit_time_s, light_time_s, iterations = measurements_module._apparent_position_observable(
        receive_time_s,
        station,
        t_grid,
        states,
        np.zeros(3),
        np.eye(6),
        tolerance_s=1.0e-15,
        max_iter=1,
    )
    range_at_returned_epoch_m = float(np.linalg.norm(target(transmit_time_s) - station.r_ecef_m))
    equation_residual_s = light_time_s - range_at_returned_epoch_m / C_LIGHT_MPS

    print(
        f"[FA-03A one-way] converged={solution.converged}, iterations={iterations}, "
        f"t_tx={transmit_time_s:.12f} s, equation residual={equation_residual_s:.12e} s, "
        f"equivalent range={equation_residual_s * C_LIGHT_MPS:.6f} m"
    )
    assert not solution.converged
    assert iterations == 1
    assert transmit_time_s == solution.transmit_time_s
    assert light_time_s == solution.light_time_s
    assert np.isfinite(z).all()
    assert z[0] == pytest.approx(range_at_returned_epoch_m, abs=1.0e-6)
    assert abs(equation_residual_s) > 1.0e-6

    with pytest.raises(RuntimeError, match="did not converge"):
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


def test_fa03a_counted_observable_and_jacobian_consume_nonconverged_endpoints():
    """Current-defect evidence: counted nominal and H ignore endpoint flags."""
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
    observable = two_way_counted_doppler_observable(
        0.0, station, t_grid, states, earth, earth, xforms, config
    )
    expected = C_LIGHT_MPS * (
        solutions[1].round_trip_light_time_s - solutions[0].round_trip_light_time_s
    ) / (2.0 * config.count_interval_s)
    jacobian = two_way_counted_doppler_initial_state_jacobian(
        0.0,
        station,
        t_grid,
        _augmented_identity_history(states),
        earth,
        earth,
        xforms,
        config,
    )
    equation_residuals = np.concatenate(
        [_round_trip_equation_residuals(solution, t_grid, states, config) for solution in solutions]
    )
    max_residual_s = float(np.max(np.abs(equation_residuals)))

    print(
        f"[FA-03A counted] endpoint converged={[s.converged for s in solutions]}, "
        f"observable={observable:.9f} m/s, max equation residual={max_residual_s:.12e} s, "
        f"equivalent range={max_residual_s * C_LIGHT_MPS:.6f} m"
    )
    assert not any(solution.converged for solution in solutions)
    assert observable == pytest.approx(
        expected,
        rel=0.0,
        abs=4.0 * np.spacing(abs(expected)),
    )
    assert np.isfinite(observable)
    assert np.isfinite(jacobian).all()
    assert max_residual_s > 1.0e-6


def test_fa03b_history_domain_matrix_documents_current_behavior():
    """FA-03B matrix: legacy paths continue across unsupported history bounds."""
    t_grid = np.array([0.0, 10.0])
    range_m = 0.25 * C_LIGHT_MPS
    states = _linear_state_history(t_grid, np.array([range_m, 0.0, 0.0]), np.zeros(3))
    augmented = _augmented_identity_history(states)
    earth = np.zeros((t_grid.size, 3), dtype=float)
    xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
    station = _OriginStation()

    matrix = {
        "One-way observable": {},
        "One-way Jacobian": {},
        "Counted observable": {},
        "Counted Jacobian": {},
        "UKF counted local history": {},
    }

    one_way_cases = {
        "Before start": 0.20,
        "Exact start": 0.25,
        "Exact end": 10.25,
        "After end": 10.30,
    }
    for label, receive_time_s in one_way_cases.items():
        z, transmit_time_s, _light_time, _iterations = measurements_module._apparent_position_observable(
            receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
        )
        solution, sensitivity = one_way_light_time_range_sensitivity(
            receive_time_s, station, t_grid, states, np.zeros(3), np.eye(6)
        )
        assert np.isfinite(z).all()
        assert np.isfinite(sensitivity.d_range_d_state).all()
        assert transmit_time_s == pytest.approx(solution.transmit_time_s, abs=1.0e-14)
        if label == "Before start":
            assert transmit_time_s < t_grid[0]
        elif label == "Exact start":
            assert transmit_time_s == pytest.approx(t_grid[0], abs=1.0e-14)
        elif label == "Exact end":
            assert transmit_time_s == pytest.approx(t_grid[-1], abs=1.0e-14)
        else:
            assert transmit_time_s > t_grid[-1]
        classification = "silent extrapolation" if label in {"Before start", "After end"} else "interpolation"
        if label == "Exact end":
            classification = "silent extrapolation"
        matrix["One-way observable"][label] = classification
        matrix["One-way Jacobian"][label] = classification

    counted_config = RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=0.2,
        light_time_tolerance_s=1.0e-13,
        light_time_max_iter=20,
    )
    counted_cases = {
        "Before start": 0.50,
        "Exact start": 0.60,
        "Exact end": 9.90,
        "After end": 10.00,
    }
    for label, midpoint_s in counted_cases.items():
        value = two_way_counted_doppler_observable(
            midpoint_s, station, t_grid, states, earth, earth, xforms, counted_config
        )
        jacobian = two_way_counted_doppler_initial_state_jacobian(
            midpoint_s, station, t_grid, augmented, earth, earth, xforms, counted_config
        )
        half_count = 0.5 * counted_config.count_interval_s
        endpoint_solutions = [
            solve_two_way_light_time(
                midpoint_s + sign * half_count,
                station,
                t_grid,
                states,
                earth,
                earth,
                xforms,
                counted_config,
            )
            for sign in (-1.0, 1.0)
        ]
        event_min = min(solution.transmit_time_s for solution in endpoint_solutions)
        event_max = max(solution.receive_time_s for solution in endpoint_solutions)
        assert np.isfinite(value)
        assert np.isfinite(jacobian).all()
        if label == "Before start":
            assert event_min < t_grid[0]
        elif label == "Exact start":
            assert event_min == pytest.approx(t_grid[0], abs=1.0e-13)
        elif label == "Exact end":
            assert event_max == pytest.approx(t_grid[-1], abs=1.0e-13)
        else:
            assert event_max > t_grid[-1]
        classification = "silent extrapolation" if label in {"Before start", "After end"} else "interpolation"
        matrix["Counted observable"][label] = classification
        matrix["Counted Jacobian"][label] = classification

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
        if label == "Before start":
            assert local_t[0] < t_grid[0]
        elif label == "Exact start":
            assert local_t[0] == pytest.approx(t_grid[0], abs=1.0e-13)
        elif label == "Exact end":
            assert local_t[-1] == pytest.approx(t_grid[-1], abs=1.0e-13)
        else:
            assert local_t[-1] > t_grid[-1]
        matrix["UKF counted local history"][label] = (
            "silent extrapolation" if label in {"Before start", "After end"} else "interpolation"
        )

    for path, cells in matrix.items():
        print(
            f"[FA-03B] {path}: "
            + ", ".join(f"{boundary}={classification}" for boundary, classification in cells.items())
        )

    assert matrix == {
        "One-way observable": {
            "Before start": "silent extrapolation",
            "Exact start": "interpolation",
            "Exact end": "silent extrapolation",
            "After end": "silent extrapolation",
        },
        "One-way Jacobian": {
            "Before start": "silent extrapolation",
            "Exact start": "interpolation",
            "Exact end": "silent extrapolation",
            "After end": "silent extrapolation",
        },
        "Counted observable": {
            "Before start": "silent extrapolation",
            "Exact start": "interpolation",
            "Exact end": "interpolation",
            "After end": "silent extrapolation",
        },
        "Counted Jacobian": {
            "Before start": "silent extrapolation",
            "Exact start": "interpolation",
            "Exact end": "interpolation",
            "After end": "silent extrapolation",
        },
        "UKF counted local history": {
            "Before start": "silent extrapolation",
            "Exact start": "interpolation",
            "Exact end": "interpolation",
            "After end": "silent extrapolation",
        },
    }
