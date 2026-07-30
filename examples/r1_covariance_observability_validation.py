"""Generate the R1 covariance/observability scientific validation figures.

The reproducer uses synthetic, deterministic lunar arcs and production entry
points only.  Each PNG has a same-basename CSV containing the plotted values
and enough run metadata to identify the force and numerical configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from lunar_od import (
    ObservabilityNumericalError,
    PassGeometry,
    PreparedArc,
    Station,
    analyze_arc_observability,
    compute_position_residuals_analytic,
    estimate_position_bls_lm,
    estimate_position_srif,
    propagate_augmented_state,
    propagate_state,
    summarize_weighted_jacobian,
)
from lunar_od.constants import J2_MOON_UNNORMALIZED, R_MOON_M
from lunar_od.dynamics import dynamics_jacobian_a_matrix, f3body_moon
from lunar_od.force_models import body_j2_acceleration, body_j2_gravity_gradient
from lunar_od.scenario_config import (
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
)


MU_MOON = 4.9028000661e12
MU_EARTH = 3.986004354360959e14
MU_SUN = 1.327124400419393e20
EARTH_MCI = np.array([3.844e8, -1.2e7, 3.0e6])
SUN_MCI = np.array([1.496e11, 2.0e9, -1.0e9])
STATE0 = np.array(
    [1.8374e6, 30.0e3, -20.0e3, -15.0, np.sqrt(MU_MOON / 1.8374e6), 4.0]
)
FD_PROFILE = "r1.fd-sweep.v1"
RNG_SEED = 314159

# ---------------------------------------------------------------------------
# Plot metric traceability contract
# ---------------------------------------------------------------------------
# Invariant: every figure's reported headline metric must be recomputable from
# that figure's OWN authoritative raw CSV.  A metric belonging to a different
# scenario, arc, whitening or observability campaign may never be attached to a
# plot row, so each metric below carries an explicit name, formula identifier
# and source artifact.
METRIC_CONTRACT_VERSION = "r1.plot-metric-contract.v1"

#: Denominators at or below this magnitude are EXCLUDED from a relative-change
#: metric rather than floored into a huge finite value.  Chosen as a fixed,
#: deterministic constant (not machine-tiny) so a near-zero reference component
#: can never manufacture a spurious headline number.
DENOMINATOR_FLOOR = 1.0e-300
DENOMINATOR_POLICY = (
    "controlled_exclusion: components whose |reference| <= 1.0e-300 or whose "
    "reference/comparison value is nonfinite are excluded from the maximum; "
    "a controlled PlotMetricError is raised when no component remains valid"
)

PLOT5_METRIC_NAME = "plot5_max_component_relative_singular_value_change"
PLOT5_METRIC_FORMULA_ID = (
    "max_i(|sigma_on[i]-sigma_off[i]|/|sigma_off[i]|) over components with "
    "finite sigma_off[i], finite sigma_on[i] and |sigma_off[i]|>denominator_floor"
)
PLOT5_SOURCE_CSV = "r1_observability_singular_values.csv"

PLOT4_SIGMA_METRIC_NAME = "plot4_min_estimator_max_component_relative_sigma_change"
PLOT4_SIGMA_FORMULA_ID = (
    "min over estimator/measurement groups of "
    "max_i(|sigma_on[i]-sigma_off[i]|/|sigma_off[i]|), same validity rule"
)
PLOT4_RECONSTRUCTION_METRIC_NAME = (
    "plot4_max_finite_sqrt_information_reconstruction_relative"
)
PLOT4_RECONSTRUCTION_FORMULA_ID = (
    "max over rows with finite sqrt_information_reconstruction_relative; "
    "estimators without a square-root path report NaN and are excluded"
)
PLOT4_SOURCE_CSV = "r1_posterior_covariance_j2_comparison.csv"


class PlotMetricError(RuntimeError):
    """A plot headline metric could not be computed from valid components."""


def max_component_relative_change(
    reference,
    comparison,
    *,
    metric_name: str,
    formula_id: str,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> dict[str, object]:
    """Component-wise maximum relative change with an explicit denominator policy.

    Returns the metric together with the metadata a report row needs to prove
    where the number came from.  Never returns inf or NaN: invalid components
    are excluded, and an all-invalid input is a controlled error rather than a
    silently floored value.
    """
    reference = np.asarray(reference, dtype=float).reshape(-1)
    comparison = np.asarray(comparison, dtype=float).reshape(-1)
    if reference.size != comparison.size:
        raise PlotMetricError(
            f"{metric_name}: reference and comparison must have the same length "
            f"({reference.size} vs {comparison.size})."
        )
    if reference.size == 0:
        raise PlotMetricError(f"{metric_name}: no components were supplied.")
    if denominator_floor <= 0.0 or not np.isfinite(denominator_floor):
        raise PlotMetricError(
            f"{metric_name}: denominator_floor must be positive and finite."
        )
    valid = (
        np.isfinite(reference)
        & np.isfinite(comparison)
        & (np.abs(reference) > float(denominator_floor))
    )
    if not np.any(valid):
        raise PlotMetricError(
            f"{metric_name}: every component was excluded by the denominator "
            f"policy ({DENOMINATOR_POLICY}); the metric is undefined."
        )
    relative = np.abs(comparison[valid] - reference[valid]) / np.abs(reference[valid])
    argmax_local = int(np.argmax(relative))
    argmax_index = int(np.flatnonzero(valid)[argmax_local])
    return {
        "metric_name": metric_name,
        "formula_id": formula_id,
        "metric_value": float(relative[argmax_local]),
        "argmax_component_index": argmax_index + 1,
        "component_count": int(reference.size),
        "valid_component_count": int(np.count_nonzero(valid)),
        "excluded_component_count": int(reference.size - np.count_nonzero(valid)),
        "denominator_floor": float(denominator_floor),
        "denominator_policy": DENOMINATOR_POLICY,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }


def _read_raw_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _paired_spectra(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    off: dict[int, float] = {}
    on: dict[int, float] = {}
    for row in rows:
        index = int(row["singular_value_index"])
        value = float(row["singular_value"])
        if row["j2_state"] == "J2 OFF":
            off[index] = value
        elif row["j2_state"] == "J2 ON":
            on[index] = value
    if not off or sorted(off) != sorted(on):
        raise PlotMetricError(
            "PLOT-5 raw CSV does not contain a matched J2 OFF / J2 ON spectrum."
        )
    order = sorted(off)
    return (
        np.array([off[i] for i in order], dtype=float),
        np.array([on[i] for i in order], dtype=float),
    )


def plot5_metric_from_spectra(off, on) -> dict[str, object]:
    """PLOT-5 headline metric from the same arrays written to its raw CSV."""
    return max_component_relative_change(
        off,
        on,
        metric_name=PLOT5_METRIC_NAME,
        formula_id=PLOT5_METRIC_FORMULA_ID,
    )


def plot5_metric_from_csv(path) -> dict[str, object]:
    """PLOT-5 headline metric recomputed from its authoritative raw CSV."""
    path = Path(path)
    rows = _read_raw_csv(path)
    off, on = _paired_spectra(rows)
    metric = plot5_metric_from_spectra(off, on)
    conditions = sorted({float(row["condition_number"]) for row in rows})
    tolerances = sorted({float(row["rank_tolerance"]) for row in rows})
    ranks = sorted({int(row["rank"]) for row in rows})
    metric.update(
        {
            "source_csv": path.name,
            "source_csv_rows": len(rows),
            "scenario": "R1 observability spectrum figure campaign",
            "arc": "single 0-600 s position arc, 60 s cadence, arc_id=1",
            "whitening": "measurement-sigma whitened initial-state Jacobian",
            "measurement_type": sorted({row["measurement_type"] for row in rows})[0],
            "bias_mode": sorted({row["bias_mode"] for row in rows})[0],
            "condition_number_min": conditions[0],
            "condition_number_max": conditions[-1],
            "rank_tolerance_min": tolerances[0],
            "rank_tolerance_max": tolerances[-1],
            "rank": ranks[0] if len(ranks) == 1 else -1,
            "num_observations": int(rows[0]["num_observations"]),
            "num_parameters": int(rows[0]["num_parameters"]),
            "all_finite": bool(np.all(np.isfinite(np.concatenate([off, on])))),
        }
    )
    return metric


def plot4_metrics_from_csv(path) -> dict[str, object]:
    """PLOT-4 headline metrics recomputed from its authoritative raw CSV."""
    path = Path(path)
    rows = _read_raw_csv(path)
    grouped: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["estimator"], row["measurement_type"], row["j2_state"])
        grouped.setdefault(key, {})[int(row["component_index"])] = float(
            row["posterior_sigma"]
        )
    per_group: dict[str, dict[str, object]] = {}
    for (estimator, measurement, state), values in sorted(grouped.items()):
        if state != "J2 ON":
            continue
        off_values = grouped.get((estimator, measurement, "J2 OFF"))
        if off_values is None or sorted(off_values) != sorted(values):
            raise PlotMetricError(
                f"PLOT-4 raw CSV lacks a matched J2 OFF group for {estimator}/{measurement}."
            )
        order = sorted(values)
        per_group[f"{estimator}/{measurement}"] = max_component_relative_change(
            [off_values[i] for i in order],
            [values[i] for i in order],
            metric_name=f"{PLOT4_SIGMA_METRIC_NAME}[{estimator}/{measurement}]",
            formula_id=PLOT4_SIGMA_FORMULA_ID,
        )
    if not per_group:
        raise PlotMetricError("PLOT-4 raw CSV contains no J2 ON group.")
    weakest_key = min(per_group, key=lambda key: per_group[key]["metric_value"])
    reconstruction = [
        float(row["sqrt_information_reconstruction_relative"]) for row in rows
    ]
    finite_reconstruction = [value for value in reconstruction if np.isfinite(value)]
    if not finite_reconstruction:
        raise PlotMetricError(
            "PLOT-4 raw CSV has no finite sqrt-information reconstruction value."
        )
    return {
        "metric_name": PLOT4_SIGMA_METRIC_NAME,
        "formula_id": PLOT4_SIGMA_FORMULA_ID,
        "metric_value": float(per_group[weakest_key]["metric_value"]),
        "weakest_group": weakest_key,
        "per_group": per_group,
        "reconstruction_metric_name": PLOT4_RECONSTRUCTION_METRIC_NAME,
        "reconstruction_formula_id": PLOT4_RECONSTRUCTION_FORMULA_ID,
        "reconstruction_metric_value": float(max(finite_reconstruction)),
        "reconstruction_finite_count": len(finite_reconstruction),
        "reconstruction_excluded_count": len(reconstruction) - len(finite_reconstruction),
        "denominator_floor": float(DENOMINATOR_FLOOR),
        "denominator_policy": DENOMINATOR_POLICY,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "source_csv": path.name,
        "source_csv_rows": len(rows),
        "scenario": "R1 posterior covariance figure campaign",
        "arc": "single 0-360 s position arc, 60 s cadence, arc_id=1, max_iter=1",
        "whitening": "estimator measurement weights; default estimator prior",
        "measurement_type": sorted({row["measurement_type"] for row in rows})[0],
    }


def _constant_vector(value):
    vector = np.asarray(value, dtype=float)

    def _getter(t_s):
        times = np.atleast_1d(np.asarray(t_s, dtype=float))
        return np.repeat(vector[None, :], times.size, axis=0)

    return _getter


GET_EARTH = _constant_vector(EARTH_MCI)
GET_SUN = _constant_vector(SUN_MCI)


def _git_value(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        windows_git = Path(r"C:\Program Files\Git\cmd\git.exe")
        git = str(windows_git) if windows_git.is_file() else None
    if git is None:
        return "unavailable"
    try:
        return subprocess.check_output(
            [git, *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _force_fingerprint(j2_moon: float) -> str:
    config = scenario_config_from_mapping(
        {
            "name": "r1_scientific_validation",
            "measurement_type": "position",
            "estimator_type": "bls_lm",
            "start_mode": "cold",
            "network": "multi",
            "j2_moon": float(j2_moon),
        }
    )
    return force_model_contract_from_scenario_config(config).force_model_fingerprint()


def _metadata() -> dict[str, object]:
    return {
        "commit_sha": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current") or "detached",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "mu_moon_m3_s2": MU_MOON,
        "mu_earth_m3_s2": MU_EARTH,
        "mu_sun_m3_s2": MU_SUN,
        "j2_moon": J2_MOON_UNNORMALIZED,
        "reference_radius_m": R_MOON_M,
        "fd_profile": FD_PROFILE,
        "zero_j2_fingerprint": _force_fingerprint(0.0),
        "lunar_j2_fingerprint": _force_fingerprint(J2_MOON_UNNORMALIZED),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    if not rows:
        raise ValueError(f"No rows produced for {path.name}.")
    enriched = [{**metadata, **row} for row in rows]
    fieldnames = list(enriched[0])
    for row in enriched[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)


def _figure_metadata(fig, metadata: dict[str, object], scenario: str) -> None:
    fig.text(
        0.01,
        0.01,
        f"commit={metadata['commit_sha']}  branch={metadata['branch']}  scenario={scenario}",
        fontsize=7,
        color="0.35",
    )


def _station(index: int) -> Station:
    definitions = (
        (0.0, 0.0, 0.0),
        (0.0, 90.0, 0.0),
        (45.0, -30.0, 500.0),
        (-35.0, 150.0, 600.0),
    )
    lat_deg, lon_deg, alt_m = definitions[index]
    return Station(
        name=f"R1 synthetic {index + 1}",
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=5.0,
        sigma_angle_rad=2.0e-5,
        sigma_range_rate_mps=1.0e-4,
    )


STATIONS = tuple(_station(index) for index in range(4))


def _truth(times_s: np.ndarray, j2_moon: float) -> np.ndarray:
    return propagate_state(
        times_s,
        STATE0,
        MU_MOON,
        MU_EARTH,
        MU_SUN,
        GET_EARTH,
        GET_SUN,
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-12,
        j2_moon=j2_moon,
    )


def _position_arc(
    times_s: np.ndarray,
    truth: np.ndarray,
    *,
    arc_id: int,
    add_noise: bool = False,
) -> PreparedArc:
    pass_geo = PassGeometry(
        t_s=np.asarray(times_s, dtype=float),
        earth_pos_mci_m=np.repeat(EARTH_MCI[None, :], times_s.size, axis=0),
        earth_vel_mci_mps=np.zeros((times_s.size, 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], times_s.size, axis=0),
        stations=STATIONS,
        measurement_type="position",
    )
    rows = []
    for time_index, t_s in enumerate(times_s, start=1):
        for station_id in range(1, len(STATIONS) + 1):
            rows.append([t_s, 0.0, 0.0, 0.0, station_id, time_index, arc_id])
    observations = np.asarray(rows, dtype=float)
    _, modeled, _ = compute_position_residuals_analytic(truth, observations, pass_geo)
    observations[:, 1:4] = modeled
    if add_noise:
        rng = np.random.default_rng(RNG_SEED + arc_id)
        station_indices = observations[:, 4].astype(int) - 1
        sigmas = np.column_stack(
            [
                np.array([STATIONS[i].sigma_range_m for i in station_indices]),
                np.array([STATIONS[i].sigma_angle_rad for i in station_indices]),
                np.array([STATIONS[i].sigma_angle_rad for i in station_indices]),
            ]
        )
        observations[:, 1:4] += rng.standard_normal((observations.shape[0], 3)) * sigmas
    return PreparedArc(
        arc_id=arc_id,
        start_idx=0,
        end_idx=times_s.size - 1,
        t_pass_s=np.asarray(times_s, dtype=float),
        truth_state_history_mci=np.asarray(truth, dtype=float),
        obs_data=observations,
        pass_geo=pass_geo,
    )


def _derivative_sweep(output_dir: Path, metadata: dict[str, object]) -> None:
    state = np.array([1.81e6, 2.7e5, -1.4e5, -120.0, 1590.0, 35.0])
    epsilons = np.logspace(-4.0, 4.0, 17)
    configurations = (
        ("point-mass baseline", MU_MOON, 0.0, 0.0, 0.0),
        ("third-body only", 0.0, MU_EARTH, MU_SUN, 0.0),
        ("lunar J2 total F", MU_MOON, MU_EARTH, MU_SUN, J2_MOON_UNNORMALIZED),
    )
    rows: list[dict[str, object]] = []
    curves: dict[str, list[float]] = {}
    for label, mu_moon, mu_earth, mu_sun, j2_moon in configurations:
        analytic = dynamics_jacobian_a_matrix(
            state,
            mu_moon,
            mu_earth,
            mu_sun,
            EARTH_MCI,
            SUN_MCI,
            j2_moon=j2_moon,
        )[3:, :3]
        errors = []
        for epsilon in epsilons:
            finite_difference = np.empty((3, 3))
            for column in range(3):
                plus = state.copy()
                minus = state.copy()
                plus[column] += epsilon
                minus[column] -= epsilon
                finite_difference[:, column] = (
                    f3body_moon(
                        plus,
                        mu_moon,
                        mu_earth,
                        mu_sun,
                        EARTH_MCI,
                        SUN_MCI,
                        j2_moon=j2_moon,
                    )[3:]
                    - f3body_moon(
                        minus,
                        mu_moon,
                        mu_earth,
                        mu_sun,
                        EARTH_MCI,
                        SUN_MCI,
                        j2_moon=j2_moon,
                    )[3:]
                ) / (2.0 * epsilon)
            relative_error = float(
                np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic)
            )
            errors.append(relative_error)
            rows.append(
                {
                    "series": label,
                    "epsilon": epsilon,
                    "epsilon_unit": "m",
                    "relative_frobenius_error": relative_error,
                    "state_vector": json.dumps(state.tolist(), separators=(",", ":")),
                }
            )
        curves[label] = errors

    analytic_j2 = body_j2_gravity_gradient(
        state[:3], MU_MOON, R_MOON_M, J2_MOON_UNNORMALIZED, np.eye(3)
    )
    j2_errors = []
    for epsilon in epsilons:
        finite_difference = np.empty((3, 3))
        for column in range(3):
            plus = state[:3].copy()
            minus = state[:3].copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            finite_difference[:, column] = (
                body_j2_acceleration(
                    plus, MU_MOON, R_MOON_M, J2_MOON_UNNORMALIZED, np.eye(3)
                )
                - body_j2_acceleration(
                    minus, MU_MOON, R_MOON_M, J2_MOON_UNNORMALIZED, np.eye(3)
                )
            ) / (2.0 * epsilon)
        relative_error = float(
            np.linalg.norm(analytic_j2 - finite_difference) / np.linalg.norm(analytic_j2)
        )
        j2_errors.append(relative_error)
        rows.append(
            {
                "series": "J2 contribution dr block",
                "epsilon": epsilon,
                "epsilon_unit": "m",
                "relative_frobenius_error": relative_error,
                "state_vector": json.dumps(state.tolist(), separators=(",", ":")),
            }
        )
    curves["J2 contribution dr block"] = j2_errors

    velocity_errors = []
    velocity_plot_errors = []
    analytic_velocity_block = np.eye(3)
    for epsilon in epsilons:
        finite_difference = np.empty((3, 3))
        for column in range(3):
            plus = state.copy()
            minus = state.copy()
            plus[3 + column] += epsilon
            minus[3 + column] -= epsilon
            finite_difference[:, column] = (
                f3body_moon(
                    plus,
                    MU_MOON,
                    MU_EARTH,
                    MU_SUN,
                    EARTH_MCI,
                    SUN_MCI,
                    j2_moon=J2_MOON_UNNORMALIZED,
                )[:3]
                - f3body_moon(
                    minus,
                    MU_MOON,
                    MU_EARTH,
                    MU_SUN,
                    EARTH_MCI,
                    SUN_MCI,
                    j2_moon=J2_MOON_UNNORMALIZED,
                )[:3]
            ) / (2.0 * epsilon)
        relative_error = float(
            np.linalg.norm(analytic_velocity_block - finite_difference)
            / np.linalg.norm(analytic_velocity_block)
        )
        velocity_errors.append(relative_error)
        velocity_plot_errors.append(max(relative_error, np.finfo(float).eps))
        rows.append(
            {
                "series": "velocity-column block (linear exact)",
                "epsilon": epsilon,
                "epsilon_unit": "m/s",
                "relative_frobenius_error": relative_error,
                "plotted_relative_error": velocity_plot_errors[-1],
                "state_vector": json.dumps(state.tolist(), separators=(",", ":")),
            }
        )
    curves["velocity-column block (linear exact)"] = velocity_plot_errors

    for label, errors in curves.items():
        minimum_index = int(np.argmin(errors))
        accurate = errors[minimum_index] <= 1.0e-8
        nonlinear_u_shape = (
            label == "velocity-column block (linear exact)"
            or (
                errors[0] > errors[minimum_index]
                and errors[-1] > errors[minimum_index]
            )
        )
        if not (accurate and nonlinear_u_shape):
            raise RuntimeError(f"R1 derivative sweep failed for {label}.")

    csv_path = output_dir / "r1_j2_dynamics_jacobian_fd_sweep.csv"
    _write_csv(csv_path, rows, metadata)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for label, errors in curves.items():
        ax.loglog(epsilons, errors, marker="o", markersize=3.5, label=label)
    ax.set_xlabel("Perturbation epsilon (m; m/s for velocity-column series)")
    ax.set_ylabel("Relative Frobenius error (dimensionless)")
    ax.set_title("R1 analytic dynamics/J2 Jacobian central-difference sweep")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    _figure_metadata(fig, metadata, "synthetic low lunar orbit; 17-point FD sweep")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "r1_j2_dynamics_jacobian_fd_sweep.png", dpi=180)
    plt.close(fig)


def _stm_validation(output_dir: Path, metadata: dict[str, object]) -> None:
    times_s = np.arange(0.0, 601.0, 60.0)
    # These operating steps sit on the measured central-difference plateau;
    # smaller velocity steps are cancellation-limited at the initial epoch.
    perturbations = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    labels = ("x", "y", "z", "vx", "vy", "vz")
    rows: list[dict[str, object]] = []
    summary: dict[str, tuple[list[float], list[float]]] = {}
    for j2_label, j2_moon in (("J2 OFF", 0.0), ("J2 ON", J2_MOON_UNNORMALIZED)):
        augmented0 = np.concatenate([STATE0, np.eye(6).reshape(-1, order="F")])
        augmented = propagate_augmented_state(
            times_s,
            augmented0,
            MU_MOON,
            MU_EARTH,
            MU_SUN,
            GET_EARTH,
            GET_SUN,
            method="DOP853",
            rtol=1.0e-12,
            atol=1.0e-13,
            j2_moon=j2_moon,
        )
        phi_history = np.stack(
            [row[6:].reshape(6, 6, order="F") for row in augmented]
        )
        final_errors = []
        maximum_errors = []
        for column, epsilon in enumerate(perturbations):
            plus = STATE0.copy()
            minus = STATE0.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            propagated_plus = propagate_state(
                times_s,
                plus,
                MU_MOON,
                MU_EARTH,
                MU_SUN,
                GET_EARTH,
                GET_SUN,
                method="DOP853",
                rtol=1.0e-12,
                atol=1.0e-13,
                j2_moon=j2_moon,
            )
            propagated_minus = propagate_state(
                times_s,
                minus,
                MU_MOON,
                MU_EARTH,
                MU_SUN,
                GET_EARTH,
                GET_SUN,
                method="DOP853",
                rtol=1.0e-12,
                atol=1.0e-13,
                j2_moon=j2_moon,
            )
            finite_difference = (propagated_plus - propagated_minus) / (2.0 * epsilon)
            errors = np.linalg.norm(
                phi_history[:, :, column] - finite_difference, axis=1
            ) / np.maximum(
                np.linalg.norm(phi_history[:, :, column], axis=1), np.finfo(float).tiny
            )
            final_errors.append(float(errors[-1]))
            maximum_errors.append(float(np.max(errors)))
            for epoch, error in zip(times_s, errors):
                rows.append(
                    {
                        "j2_state": j2_label,
                        "column_index": column + 1,
                        "state_component": labels[column],
                        "epoch_s": epoch,
                        "perturbation": epsilon,
                        "perturbation_unit": "m" if column < 3 else "m/s",
                        "relative_column_error": float(error),
                        "integrator": "DOP853",
                        "rtol": 1.0e-12,
                        "atol": 1.0e-13,
                    }
                )
        if max(maximum_errors) > 1.0e-6:
            raise RuntimeError(f"R1 STM finite-difference gate failed for {j2_label}.")
        summary[j2_label] = (final_errors, maximum_errors)

    _write_csv(output_dir / "r1_stm_fd_validation.csv", rows, metadata)
    x = np.arange(6)
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    offset = -1.5 * width
    for j2_label, (final_errors, maximum_errors) in summary.items():
        ax.bar(x + offset, final_errors, width, label=f"{j2_label} final")
        offset += width
        ax.bar(x + offset, maximum_errors, width, label=f"{j2_label} arc max")
        offset += width
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_xlabel("STM initial-state column")
    ax.set_ylabel("Relative column error (dimensionless)")
    ax.set_title("R1 propagated STM versus central-difference state sensitivity")
    ax.axhline(1.0e-6, color="red", linestyle="--", label="1e-6 gate")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _figure_metadata(fig, metadata, "600 s arc; DOP853; J2 off/on")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "r1_stm_fd_validation.png", dpi=180)
    plt.close(fig)


def _run_position_estimator(
    estimator: str,
    arc: PreparedArc,
    j2_moon: float,
    *,
    max_iter: int,
):
    initial = arc.truth_state_history_mci[0, :6].copy()
    initial += np.array([40.0, -25.0, 15.0, 0.02, -0.01, 0.005])
    function = estimate_position_bls_lm if estimator == "BLS-LM" else estimate_position_srif
    estimate, stop_reason, stats = function(
        arc.t_pass_s,
        arc.obs_data,
        initial,
        arc.pass_geo,
        MU_MOON,
        MU_EARTH,
        MU_SUN,
        GET_EARTH,
        GET_SUN,
        max_iter=max_iter,
        rtol=1.0e-11,
        atol=1.0e-12,
        j2_moon=j2_moon,
        return_posterior=True,
    )
    return estimate, stop_reason, stats


def _estimation_envelope(output_dir: Path, metadata: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    position_errors = []
    velocity_errors = []
    position_bounds = []
    velocity_bounds = []
    arc_ids = []
    for arc_id, sample_count in enumerate((6, 9, 12), start=1):
        times_s = np.arange(0.0, float(sample_count) * 60.0, 60.0)
        truth = _truth(times_s, J2_MOON_UNNORMALIZED)
        arc = _position_arc(times_s, truth, arc_id=arc_id, add_noise=True)
        estimate, stop_reason, stats = _run_position_estimator(
            "BLS-LM", arc, J2_MOON_UNNORMALIZED, max_iter=12
        )
        covariance = np.asarray(stats.posterior_covariance, dtype=float)
        position_error = float(np.linalg.norm(estimate[:3] - truth[0, :3]))
        velocity_error = float(np.linalg.norm(estimate[3:6] - truth[0, 3:6]))
        position_bound = float(3.0 * np.sqrt(np.trace(covariance[:3, :3])))
        velocity_bound = float(3.0 * np.sqrt(np.trace(covariance[3:6, 3:6])))
        arc_ids.append(arc_id)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        position_bounds.append(position_bound)
        velocity_bounds.append(velocity_bound)
        rows.append(
            {
                "arc_id": arc_id,
                "arc_duration_s": float(times_s[-1] - times_s[0]),
                "num_observations": int(arc.obs_data.shape[0]),
                "position_error_m": position_error,
                "position_3sigma_rss_m": position_bound,
                "position_inside_3sigma": position_error <= position_bound,
                "velocity_error_mps": velocity_error,
                "velocity_3sigma_rss_mps": velocity_bound,
                "velocity_inside_3sigma": velocity_error <= velocity_bound,
                "estimator": "BLS-LM",
                "measurement_type": "position",
                "station_network": "four synthetic stations",
                "noise_seed": RNG_SEED + arc_id,
                "stop_reason": stop_reason,
                "truth_force_fingerprint": metadata["lunar_j2_fingerprint"],
                "estimator_force_fingerprint": metadata["lunar_j2_fingerprint"],
            }
        )
    _write_csv(output_dir / "r1_estimation_error_vs_3sigma.csv", rows, metadata)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))
    axes[0].semilogy(arc_ids, position_errors, "o-", label="position error")
    axes[0].semilogy(arc_ids, position_bounds, "s--", label="+3 sigma RSS")
    axes[0].set_ylabel("Position (m)")
    axes[1].semilogy(arc_ids, velocity_errors, "o-", label="velocity error")
    axes[1].semilogy(arc_ids, velocity_bounds, "s--", label="+3 sigma RSS")
    axes[1].set_ylabel("Velocity (m/s)")
    for ax in axes:
        ax.set_xlabel("Arc index")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("R1 matched nonzero-J2 estimation error versus formal 3-sigma")
    _figure_metadata(fig, metadata, "BLS-LM position; matched force fingerprints")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(output_dir / "r1_estimation_error_vs_3sigma.png", dpi=180)
    plt.close(fig)


def _posterior_comparison(output_dir: Path, metadata: dict[str, object]) -> None:
    times_s = np.arange(0.0, 361.0, 60.0)
    truth = _truth(times_s, J2_MOON_UNNORMALIZED)
    arc = _position_arc(times_s, truth, arc_id=1)
    labels = ("x", "y", "z", "vx", "vy", "vz")
    rows: list[dict[str, object]] = []
    sigmas: dict[tuple[str, str], np.ndarray] = {}
    sqrt_sigma: np.ndarray | None = None
    for estimator in ("BLS-LM", "SRIF"):
        for j2_label, j2_moon in (("J2 OFF", 0.0), ("J2 ON", J2_MOON_UNNORMALIZED)):
            _, _, stats = _run_position_estimator(estimator, arc, j2_moon, max_iter=1)
            covariance = np.asarray(stats.posterior_covariance, dtype=float)
            sigma = np.sqrt(np.diag(covariance))
            sigmas[(estimator, j2_label)] = sigma
            reconstruction_relative = float("nan")
            if stats.posterior_sqrt_information is not None:
                reconstructed = (
                    stats.posterior_sqrt_information.T
                    @ stats.posterior_sqrt_information
                )
                reconstruction_relative = float(
                    np.linalg.norm(reconstructed - stats.posterior_information)
                    / np.linalg.norm(stats.posterior_information)
                )
                if reconstruction_relative > 1.0e-10:
                    raise RuntimeError("SRIF sqrt-information reconstruction failed.")
                if j2_label == "J2 ON":
                    sqrt_sigma = np.sqrt(np.diag(np.linalg.inv(reconstructed)))
            for component, value in enumerate(sigma):
                rows.append(
                    {
                        "estimator": estimator,
                        "j2_state": j2_label,
                        "component_index": component + 1,
                        "state_component": labels[component],
                        "posterior_sigma": float(value),
                        "sigma_unit": "m" if component < 3 else "m/s",
                        "sqrt_information_reconstruction_relative": reconstruction_relative,
                        "measurement_type": "position",
                        "arc_duration_s": float(times_s[-1]),
                        "prior_covariance": "default estimator prior",
                    }
                )
    for estimator in ("BLS-LM", "SRIF"):
        # Same named metric and same arrays as the PLOT-4 raw CSV rows above.
        relative = float(
            max_component_relative_change(
                sigmas[(estimator, "J2 OFF")],
                sigmas[(estimator, "J2 ON")],
                metric_name=f"{PLOT4_SIGMA_METRIC_NAME}[{estimator}/position]",
                formula_id=PLOT4_SIGMA_FORMULA_ID,
            )["metric_value"]
        )
        if relative <= 1.0e-9:
            raise RuntimeError(f"Posterior J2 selection did not reach {estimator} covariance.")

    _write_csv(output_dir / "r1_posterior_covariance_j2_comparison.csv", rows, metadata)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    x_pos = np.arange(3)
    x_vel = np.arange(3)
    styles = {
        ("BLS-LM", "J2 OFF"): "o--",
        ("BLS-LM", "J2 ON"): "o-",
        ("SRIF", "J2 OFF"): "s--",
        ("SRIF", "J2 ON"): "s-",
    }
    for key, sigma in sigmas.items():
        axes[0].semilogy(x_pos, sigma[:3], styles[key], label=f"{key[0]} {key[1]}")
        axes[1].semilogy(x_vel, sigma[3:], styles[key], label=f"{key[0]} {key[1]}")
    if sqrt_sigma is not None:
        axes[0].semilogy(x_pos, sqrt_sigma[:3], "x", label="SRIF R^T R reconstruction")
        axes[1].semilogy(x_vel, sqrt_sigma[3:], "x", label="SRIF R^T R reconstruction")
    axes[0].set_xticks(x_pos, labels[:3])
    axes[0].set_ylabel("Posterior standard deviation (m)")
    axes[1].set_xticks(x_vel, labels[3:])
    axes[1].set_ylabel("Posterior standard deviation (m/s)")
    for ax in axes:
        ax.set_xlabel("State component")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("R1 lunar-J2 posterior covariance parity")
    _figure_metadata(fig, metadata, "position observable; BLS-LM and SRIF")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(output_dir / "r1_posterior_covariance_j2_comparison.png", dpi=180)
    plt.close(fig)


def _observability_plots(output_dir: Path, metadata: dict[str, object]) -> None:
    times_s = np.arange(0.0, 601.0, 60.0)
    truth = _truth(times_s, J2_MOON_UNNORMALIZED)
    full_arc = _position_arc(times_s, truth, arc_id=1)
    results = {}
    spectrum_rows: list[dict[str, object]] = []
    for j2_label, j2_moon in (("J2 OFF", 0.0), ("J2 ON", J2_MOON_UNNORMALIZED)):
        result = analyze_arc_observability(
            full_arc,
            "position",
            MU_MOON,
            MU_EARTH,
            MU_SUN,
            GET_EARTH,
            GET_SUN,
            rtol=1.0e-12,
            atol=1.0e-13,
            j2_moon=j2_moon,
        )
        results[j2_label] = result
        tolerance = float(
            max(result.weighted_jacobian.shape)
            * np.finfo(float).eps
            * result.singular_values[0]
        )
        for index, singular_value in enumerate(result.singular_values, start=1):
            spectrum_rows.append(
                {
                    "j2_state": j2_label,
                    "singular_value_index": index,
                    "singular_value": float(singular_value),
                    "rank_tolerance": tolerance,
                    "retained": bool(singular_value > tolerance),
                    "rank": result.rank,
                    "condition_number": result.condition_number,
                    "num_observations": result.num_observations,
                    "num_parameters": result.num_parameters,
                    "measurement_type": "position",
                    "bias_mode": "none",
                }
            )
    off = results["J2 OFF"].singular_values
    on = results["J2 ON"].singular_values
    # The gate uses the SAME named metric that the report row will quote, on the
    # SAME arrays that are written to r1_observability_singular_values.csv.
    spectrum_metric = plot5_metric_from_spectra(off, on)
    spectral_relative = float(spectrum_metric["metric_value"])
    if not np.all(np.isfinite(np.concatenate([off, on]))) or spectral_relative <= 1.0e-9:
        raise RuntimeError("Observability J2 spectrum gate failed.")
    _write_csv(output_dir / "r1_observability_singular_values.csv", spectrum_rows, metadata)
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    indices = np.arange(1, off.size + 1)
    for label, result in results.items():
        ax.semilogy(indices, result.singular_values, "o-", label=label)
        tolerance = (
            max(result.weighted_jacobian.shape)
            * np.finfo(float).eps
            * result.singular_values[0]
        )
        ax.axhline(tolerance, linestyle="--", alpha=0.7, label=f"{label} rank tol")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value of whitened H (dimensionless)")
    ax.set_title("R1 whitened observability singular-value spectrum")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    _figure_metadata(fig, metadata, "position; four stations; no bias states")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "r1_observability_singular_values.png", dpi=180)
    plt.close(fig)

    rank_rows: list[dict[str, object]] = []
    plot_values: dict[str, dict[str, list[float]]] = {
        "J2 OFF": {"rank": [], "condition": []},
        "J2 ON": {"rank": [], "condition": []},
    }
    arc_ids = []
    for arc_id, sample_count in enumerate((4, 7, 10), start=1):
        arc_times = times_s[:sample_count]
        arc = _position_arc(arc_times, truth[:sample_count], arc_id=arc_id)
        arc_ids.append(arc_id)
        per_state = {}
        for j2_label, j2_moon in (("J2 OFF", 0.0), ("J2 ON", J2_MOON_UNNORMALIZED)):
            result = analyze_arc_observability(
                arc,
                "position",
                MU_MOON,
                MU_EARTH,
                MU_SUN,
                GET_EARTH,
                GET_SUN,
                rtol=1.0e-12,
                atol=1.0e-13,
                j2_moon=j2_moon,
            )
            per_state[j2_label] = result
            plot_values[j2_label]["rank"].append(result.rank)
            plot_values[j2_label]["condition"].append(result.condition_number)
            rank_rows.append(
                {
                    "arc_id": arc_id,
                    "j2_state": j2_label,
                    "rank": result.rank,
                    "condition_number": result.condition_number,
                    "condition_unit_note": "unit-dependent unscaled m and m/s columns",
                    "gate_rejected": False,
                    "rank_tolerance": float(
                        max(result.weighted_jacobian.shape)
                        * np.finfo(float).eps
                        * result.singular_values[0]
                    ),
                    "num_observations": result.num_observations,
                    "station_network": "four synthetic stations",
                }
            )
        if per_state["J2 OFF"].rank == 6 and per_state["J2 ON"].rank != 6:
            raise RuntimeError("J2 changed full-rank observability classification.")

    gate_rejected = False
    try:
        bad = np.eye(6)
        bad[0, 0] = np.nan
        summarize_weighted_jacobian("position", 1, bad)
    except ObservabilityNumericalError:
        gate_rejected = True
    if not gate_rejected:
        raise RuntimeError("Nonfinite observability mutation was not rejected.")
    rank_rows.append(
        {
            "arc_id": "nonfinite mutation",
            "j2_state": "gate",
            "rank": "",
            "condition_number": "",
            "condition_unit_note": "nonfinite input rejected before decomposition",
            "gate_rejected": True,
            "rank_tolerance": "",
            "num_observations": 1,
            "station_network": "injected mutation",
        }
    )
    _write_csv(output_dir / "r1_observability_rank_condition.csv", rank_rows, metadata)
    fig, ax_rank = plt.subplots(figsize=(9.0, 5.2))
    ax_condition = ax_rank.twinx()
    for label, values in plot_values.items():
        ax_rank.plot(arc_ids, values["rank"], "o-", label=f"{label} rank")
        ax_condition.semilogy(
            arc_ids, values["condition"], "s--", label=f"{label} condition"
        )
    marker_x = max(arc_ids) + 1
    finite_conditions = [
        value
        for state in plot_values.values()
        for value in state["condition"]
        if np.isfinite(value)
    ]
    marker_y = max(finite_conditions) if finite_conditions else 1.0
    ax_condition.scatter(
        [marker_x], [marker_y], color="red", marker="x", s=80, label="nonfinite gate rejection"
    )
    ax_rank.set_xticks(arc_ids + [marker_x], [str(value) for value in arc_ids] + ["gate"])
    ax_rank.set_xlabel("Arc index / injected gate case")
    ax_rank.set_ylabel("Numerical rank (integer)")
    ax_condition.set_ylabel("Condition number (unit-dependent, log scale)")
    ax_rank.set_title("R1 observability rank, condition and fail-closed gate")
    lines_a, labels_a = ax_rank.get_legend_handles_labels()
    lines_b, labels_b = ax_condition.get_legend_handles_labels()
    ax_rank.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="best")
    ax_rank.grid(True, alpha=0.25)
    _figure_metadata(fig, metadata, "three synthetic arcs; unscaled state columns")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "r1_observability_rank_condition.png", dpi=180)
    plt.close(fig)


PLOT_VALIDATION_FIELDS = (
    "plot_id",
    "metric_name",
    "formula_id",
    "source_csv",
    "source_csv_rows",
    "scenario",
    "arc",
    "whitening",
    "condition_number",
    "rank_tolerance",
    "metric_value",
    "acceptance_gate",
    "result",
    "supporting_metrics",
    "denominator_policy",
    "metric_contract_version",
)

_NONLINEAR_FD_SERIES = (
    "point-mass baseline",
    "third-body only",
    "lunar J2 total F",
    "J2 contribution dr block",
)
_LINEAR_FD_SERIES = "velocity-column block (linear exact)"


def build_plot_validation_rows(output_dir: Path) -> list[dict[str, object]]:
    """Derive every figure's headline metric from that figure's own raw CSV.

    This is the single producer of ``plot_validation_matrix.csv``.  No row may
    quote a metric computed by a different scenario, arc or campaign: each value
    below is recomputed here from the same-basename authoritative CSV.
    """
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    # PLOT-1 -- dynamics/J2 Jacobian finite-difference sweep.
    fd_name = "r1_j2_dynamics_jacobian_fd_sweep.csv"
    fd_rows = _read_raw_csv(output_dir / fd_name)
    minima = {
        series: min(
            float(row["relative_frobenius_error"])
            for row in fd_rows
            if row["series"] == series
        )
        for series in sorted({row["series"] for row in fd_rows})
    }
    missing_series = [name for name in _NONLINEAR_FD_SERIES if name not in minima]
    if missing_series or _LINEAR_FD_SERIES not in minima:
        raise PlotMetricError(f"PLOT-1 raw CSV is missing series: {missing_series}")
    worst_nonlinear = max(minima[name] for name in _NONLINEAR_FD_SERIES)
    rows.append(
        {
            "plot_id": "PLOT-1",
            "metric_name": "plot1_worst_nonlinear_series_minimum_relative_frobenius_error",
            "formula_id": "max over nonlinear series of min over epsilon of relative_frobenius_error",
            "source_csv": fd_name,
            "source_csv_rows": len(fd_rows),
            "scenario": "R1 dynamics Jacobian FD sweep campaign",
            "arc": "single epoch; log-spaced position and velocity perturbations",
            "whitening": "not applicable",
            "condition_number": "not applicable",
            "rank_tolerance": "not applicable",
            "metric_value": repr(worst_nonlinear),
            "acceptance_gate": "<=1e-8",
            "result": "PASS" if worst_nonlinear <= 1.0e-8 else "FAIL",
            "supporting_metrics": (
                f"linear_velocity_block_minimum={minima[_LINEAR_FD_SERIES]!r}"
            ),
            "denominator_policy": "not applicable",
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        }
    )

    # PLOT-2 -- STM finite difference.
    stm_name = "r1_stm_fd_validation.csv"
    stm_rows = _read_raw_csv(output_dir / stm_name)
    stm_max = {
        state: max(
            float(row["relative_column_error"])
            for row in stm_rows
            if row["j2_state"] == state
        )
        for state in ("J2 OFF", "J2 ON")
    }
    stm_worst = max(stm_max.values())
    rows.append(
        {
            "plot_id": "PLOT-2",
            "metric_name": "plot2_max_relative_stm_column_error",
            "formula_id": "max over J2 state, column and epoch of relative_column_error",
            "source_csv": stm_name,
            "source_csv_rows": len(stm_rows),
            "scenario": "R1 STM finite-difference campaign",
            "arc": "single position arc; per-column central differences",
            "whitening": "not applicable",
            "condition_number": "not applicable",
            "rank_tolerance": "not applicable",
            "metric_value": repr(stm_worst),
            "acceptance_gate": "<=1e-6",
            "result": "PASS" if stm_worst <= 1.0e-6 else "FAIL",
            "supporting_metrics": (
                f"j2_off_max={stm_max['J2 OFF']!r}; j2_on_max={stm_max['J2 ON']!r}"
            ),
            "denominator_policy": "not applicable",
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        }
    )

    # PLOT-3 -- estimation error versus formal 3 sigma (non-blocking metric).
    env_name = "r1_estimation_error_vs_3sigma.csv"
    env_rows = _read_raw_csv(output_dir / env_name)
    fractions = {}
    for column in ("position_inside_3sigma", "velocity_inside_3sigma"):
        flags = [row[column] == "True" for row in env_rows]
        fractions[column] = sum(flags) / len(flags)
    rows.append(
        {
            "plot_id": "PLOT-3",
            "metric_name": "plot3_three_sigma_containment_fraction",
            "formula_id": "count(inside_3sigma)/count(rows), reported per position and velocity",
            "source_csv": env_name,
            "source_csv_rows": len(env_rows),
            "scenario": "R1 matched-force estimation envelope campaign",
            "arc": "three synthetic position arcs; matched truth/estimator J2",
            "whitening": "not applicable",
            "condition_number": "not applicable",
            "rank_tolerance": "not applicable",
            "metric_value": (
                f"position={fractions['position_inside_3sigma']!r}; "
                f"velocity={fractions['velocity_inside_3sigma']!r}"
            ),
            "acceptance_gate": "reported only (non-blocking observation)",
            "result": "PASS (non-blocking observation)",
            "supporting_metrics": f"arcs={len(env_rows)}",
            "denominator_policy": "not applicable",
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        }
    )

    # PLOT-4 -- posterior covariance J2 reachability and SRIF reconstruction.
    plot4 = plot4_metrics_from_csv(output_dir / PLOT4_SOURCE_CSV)
    per_group = "; ".join(
        f"{key}={value['metric_value']!r}" for key, value in sorted(plot4["per_group"].items())
    )
    plot4_pass = (
        plot4["metric_value"] > 1.0e-9
        and plot4["reconstruction_metric_value"] <= 1.0e-10
    )
    rows.append(
        {
            "plot_id": "PLOT-4",
            "metric_name": plot4["metric_name"],
            "formula_id": plot4["formula_id"],
            "source_csv": plot4["source_csv"],
            "source_csv_rows": plot4["source_csv_rows"],
            "scenario": plot4["scenario"],
            "arc": plot4["arc"],
            "whitening": plot4["whitening"],
            "condition_number": "not reported by this figure",
            "rank_tolerance": "not applicable",
            "metric_value": repr(plot4["metric_value"]),
            "acceptance_gate": ">1e-9 for sigma change; <=1e-10 for reconstruction",
            "result": "PASS" if plot4_pass else "FAIL",
            "supporting_metrics": (
                f"weakest_group={plot4['weakest_group']}; {per_group}; "
                f"{plot4['reconstruction_metric_name']}="
                f"{plot4['reconstruction_metric_value']!r}; "
                f"reconstruction_finite={plot4['reconstruction_finite_count']}; "
                f"reconstruction_excluded={plot4['reconstruction_excluded_count']}"
            ),
            "denominator_policy": plot4["denominator_policy"],
            "metric_contract_version": plot4["metric_contract_version"],
        }
    )

    # PLOT-5 -- observability singular-value spectrum J2 reachability.
    plot5 = plot5_metric_from_csv(output_dir / PLOT5_SOURCE_CSV)
    plot5_pass = bool(plot5["metric_value"] > 1.0e-9 and plot5["all_finite"])
    rows.append(
        {
            "plot_id": "PLOT-5",
            "metric_name": plot5["metric_name"],
            "formula_id": plot5["formula_id"],
            "source_csv": plot5["source_csv"],
            "source_csv_rows": plot5["source_csv_rows"],
            "scenario": plot5["scenario"],
            "arc": plot5["arc"],
            "whitening": plot5["whitening"],
            "condition_number": (
                f"{plot5['condition_number_min']!r}..{plot5['condition_number_max']!r}"
            ),
            "rank_tolerance": (
                f"{plot5['rank_tolerance_min']!r}..{plot5['rank_tolerance_max']!r}"
            ),
            "metric_value": repr(plot5["metric_value"]),
            "acceptance_gate": ">1e-9",
            "result": "PASS" if plot5_pass else "FAIL",
            "supporting_metrics": (
                f"argmax_component_index={plot5['argmax_component_index']}; "
                f"valid_components={plot5['valid_component_count']}/"
                f"{plot5['component_count']}; rank={plot5['rank']}; "
                f"all_finite={plot5['all_finite']}; "
                f"num_observations={plot5['num_observations']}; "
                f"num_parameters={plot5['num_parameters']}"
            ),
            "denominator_policy": plot5["denominator_policy"],
            "metric_contract_version": plot5["metric_contract_version"],
        }
    )

    # PLOT-6 -- per-arc rank/condition and nonfinite rejection.
    rank_name = "r1_observability_rank_condition.csv"
    rank_rows = _read_raw_csv(output_dir / rank_name)
    arc_ranks: dict[str, dict[str, int]] = {}
    conditions: list[float] = []
    gate_markers = 0
    tolerances: list[float] = []
    for row in rank_rows:
        if row["gate_rejected"] == "True":
            gate_markers += 1
            continue
        arc_ranks.setdefault(row["arc_id"], {})[row["j2_state"]] = int(row["rank"])
        conditions.append(float(row["condition_number"]))
        tolerances.append(float(row["rank_tolerance"]))
    rank_identical = all(
        states.get("J2 OFF") == states.get("J2 ON") for states in arc_ranks.values()
    )
    plot6_pass = bool(
        rank_identical
        and conditions
        and all(np.isfinite(conditions))
        and gate_markers >= 1
    )
    rows.append(
        {
            "plot_id": "PLOT-6",
            "metric_name": "plot6_rank_identity_and_finite_condition_range",
            "formula_id": (
                "rank(J2 OFF)==rank(J2 ON) per arc over non-gated rows; "
                "min/max condition_number over non-gated rows; count(gate_rejected)"
            ),
            "source_csv": rank_name,
            "source_csv_rows": len(rank_rows),
            "scenario": "R1 per-arc observability rank/condition campaign",
            "arc": "three nested position arcs (4, 7, 10 samples) plus one mutation row",
            "whitening": "measurement-sigma whitened initial-state Jacobian",
            "condition_number": f"{min(conditions)!r}..{max(conditions)!r}",
            "rank_tolerance": f"{min(tolerances)!r}..{max(tolerances)!r}",
            "metric_value": (
                f"rank_identical={rank_identical}; arcs={len(arc_ranks)}; "
                f"gate_markers={gate_markers}"
            ),
            "acceptance_gate": "rank identity, finite conditions, >=1 explicit gate marker",
            "result": "PASS" if plot6_pass else "FAIL",
            "supporting_metrics": "; ".join(
                f"arc {arc_id}: {states['J2 OFF']}/{states['J2 ON']}"
                for arc_id, states in sorted(arc_ranks.items())
            ),
            "denominator_policy": "not applicable",
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        }
    )
    return rows


def write_plot_validation_matrix(output_dir: Path) -> Path:
    """Write ``plot_validation_matrix.csv`` derived from the six raw CSVs."""
    output_dir = Path(output_dir)
    rows = build_plot_validation_rows(output_dir)
    path = output_dir / "plot_validation_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLOT_VALIDATION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata()
    _derivative_sweep(output_dir, metadata)
    _stm_validation(output_dir, metadata)
    _estimation_envelope(output_dir, metadata)
    _posterior_comparison(output_dir, metadata)
    _observability_plots(output_dir, metadata)
    expected = (
        "r1_j2_dynamics_jacobian_fd_sweep",
        "r1_stm_fd_validation",
        "r1_estimation_error_vs_3sigma",
        "r1_posterior_covariance_j2_comparison",
        "r1_observability_singular_values",
        "r1_observability_rank_condition",
    )
    missing = [
        f"{basename}{suffix}"
        for basename in expected
        for suffix in (".png", ".csv")
        if not (output_dir / f"{basename}{suffix}").is_file()
    ]
    if missing:
        raise RuntimeError(f"R1 validation output is incomplete: {missing}")
    validation_rows = build_plot_validation_rows(output_dir)
    matrix_path = write_plot_validation_matrix(output_dir)
    failed = [row["plot_id"] for row in validation_rows if row["result"].startswith("FAIL")]
    if failed:
        raise RuntimeError(f"R1 plot validation matrix reports failures: {failed}")
    return {
        "metadata": metadata,
        "outputs": list(expected),
        "plot_validation_matrix": matrix_path.name,
        "plot_metrics": {
            row["plot_id"]: {
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "source_csv": row["source_csv"],
                "result": row["result"],
            }
            for row in validation_rows
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args(argv)
    summary = generate(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
