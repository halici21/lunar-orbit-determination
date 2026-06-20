"""JSON scenario configuration schema and validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .filters import UKFAdaptiveConfig, UnscentedTransformConfig
from .radiometrics import RangeRatePhysicsConfig
from .scenarios import EstimatorType, MeasurementType, StartMode
from .thesis_matrix import (
    THESIS_ATOL,
    THESIS_DURATION_H,
    THESIS_MAX_ITER,
    THESIS_NETWORKS,
    THESIS_RTOL,
    THESIS_SAMPLE_STEP_S,
)

ALLOWED_MEASUREMENT_TYPES = ("position", "range_rate")
ALLOWED_ESTIMATOR_TYPES = ("bls_lm", "srif", "ukf")
ALLOWED_START_MODES = ("cold", "hot", "formal", "sqrt_formal")
ALLOWED_NETWORKS = tuple(network.name for network in THESIS_NETWORKS)
ALLOWED_BIAS_MODES = (None, "global", "station_angles", "station_full")
ALLOWED_RANGE_RATE_PHYSICS = ("geometric_instantaneous", "two_way_counted_doppler")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    measurement_type: MeasurementType
    estimator_type: EstimatorType
    start_mode: StartMode
    network: str
    duration_h: float = THESIS_DURATION_H
    sample_step_s: float = THESIS_SAMPLE_STEP_S
    max_iter: int = THESIS_MAX_ITER
    tol_cost_stability: float = 1e-8
    bls_lambda0: float = 1e-2
    rtol: float = THESIS_RTOL
    atol: float = THESIS_ATOL
    j2_moon: float = 0.0
    noise: bool = False
    bias_mode: str | None = None
    range_rate_physics: str = "geometric_instantaneous"
    count_interval_s: float = 60.0
    uplink_frequency_hz: float = 7.2e9
    turnaround_ratio: float = 880.0 / 749.0
    two_way_local_state_model: str = "ode"
    station_clock_offset_s: float = 0.0
    station_clock_drift: float = 0.0
    clock_reference_time_s: float = 0.0
    transponder_delay_s: float = 0.0
    apply_light_time: bool = False
    apply_stellar_aberration: bool = False
    stellar_aberration_model: str = "local_mci"
    ukf_alpha: float = 0.35
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_covariance_inflation: float = 1.0
    ukf_process_noise_model: str = "discrete"
    ukf_acceleration_psd_m2_s3: float | None = None
    ukf_adaptive_process_noise: bool = False
    ukf_initial_process_noise_scale: float = 1.0
    ukf_min_process_noise_scale: float = 0.1
    ukf_max_process_noise_scale: float = 100.0
    ukf_process_noise_adaptation_gain: float = 0.2
    ukf_adaptive_measurement_noise: bool = False
    ukf_max_measurement_noise_scale: float = 100.0
    ukf_nis_gate: float | None = None
    ukf_component_nis_gate: float | None = None
    ukf_component_gate_mode: str = "marginal"
    ukf_robust_measurement_update: bool = False
    ukf_robust_loss: str = "student_t"
    ukf_robust_student_t_dof: float = 5.0
    ukf_robust_huber_threshold: float = 3.0
    ukf_robust_min_component_weight: float = 0.05
    ukf_covariance_form: str = "square_root"
    ukf_auto_bias_constraints: bool = False
    ukf_bias_freeze_relative_information: float = 1e-12
    ukf_bias_regularize_relative_information: float = 1e-5
    ukf_bias_regularization_std: float = 1.0
    output_dir: str = "python_port/results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scenario_config_schema() -> dict[str, Any]:
    """Return a compact JSON-serializable schema description."""
    return {
        "required": ["name", "measurement_type", "estimator_type", "start_mode", "network"],
        "properties": {
            "name": {"type": "string"},
            "measurement_type": {"enum": list(ALLOWED_MEASUREMENT_TYPES)},
            "estimator_type": {"enum": list(ALLOWED_ESTIMATOR_TYPES)},
            "start_mode": {"enum": list(ALLOWED_START_MODES)},
            "network": {"enum": list(ALLOWED_NETWORKS)},
            "duration_h": {"type": "number", "default": THESIS_DURATION_H},
            "sample_step_s": {"type": "number", "default": THESIS_SAMPLE_STEP_S},
            "max_iter": {"type": "integer", "default": THESIS_MAX_ITER},
            "rtol": {"type": "number", "default": THESIS_RTOL},
            "atol": {"type": "number", "default": THESIS_ATOL},
            "noise": {"type": "boolean", "default": False},
            "bias_mode": {"enum": [None, "global", "station_angles", "station_full"], "default": None},
            "range_rate_physics": {
                "enum": list(ALLOWED_RANGE_RATE_PHYSICS),
                "default": "geometric_instantaneous",
            },
            "count_interval_s": {"type": "number", "default": 60.0},
            "uplink_frequency_hz": {"type": "number", "default": 7.2e9},
            "turnaround_ratio": {"type": "number", "default": 880.0 / 749.0},
            "two_way_local_state_model": {"enum": ["ode", "taylor3"], "default": "ode"},
            "station_clock_offset_s": {"type": "number", "default": 0.0},
            "station_clock_drift": {"type": "number", "default": 0.0},
            "clock_reference_time_s": {"type": "number", "default": 0.0},
            "transponder_delay_s": {"type": "number", "default": 0.0},
            "apply_light_time": {"type": "boolean", "default": False},
            "apply_stellar_aberration": {"type": "boolean", "default": False},
            "stellar_aberration_model": {"enum": ["local_mci", "spice_ssb"], "default": "local_mci"},
            "ukf_alpha": {"type": "number", "default": 0.35},
            "ukf_beta": {"type": "number", "default": 2.0},
            "ukf_kappa": {"type": "number", "default": 0.0},
            "ukf_covariance_inflation": {"type": "number", "default": 1.0},
            "ukf_process_noise_model": {
                "enum": ["discrete", "continuous_white_acceleration"],
                "default": "discrete",
            },
            "ukf_acceleration_psd_m2_s3": {"type": ["number", "null"], "default": None},
            "ukf_adaptive_process_noise": {"type": "boolean", "default": False},
            "ukf_initial_process_noise_scale": {"type": "number", "default": 1.0},
            "ukf_min_process_noise_scale": {"type": "number", "default": 0.1},
            "ukf_max_process_noise_scale": {"type": "number", "default": 100.0},
            "ukf_process_noise_adaptation_gain": {"type": "number", "default": 0.2},
            "ukf_adaptive_measurement_noise": {"type": "boolean", "default": False},
            "ukf_max_measurement_noise_scale": {"type": "number", "default": 100.0},
            "ukf_nis_gate": {"type": ["number", "null"], "default": None},
            "ukf_component_nis_gate": {"type": ["number", "null"], "default": None},
            "ukf_component_gate_mode": {"enum": ["marginal", "conditional"], "default": "marginal"},
            "ukf_robust_measurement_update": {"type": "boolean", "default": False},
            "ukf_robust_loss": {"enum": ["student_t", "huber"], "default": "student_t"},
            "ukf_robust_student_t_dof": {"type": "number", "default": 5.0},
            "ukf_robust_huber_threshold": {"type": "number", "default": 3.0},
            "ukf_robust_min_component_weight": {"type": "number", "default": 0.05},
            "ukf_covariance_form": {"enum": ["standard", "square_root"], "default": "square_root"},
            "ukf_auto_bias_constraints": {"type": "boolean", "default": False},
            "ukf_bias_freeze_relative_information": {"type": "number", "default": 1e-12},
            "ukf_bias_regularize_relative_information": {"type": "number", "default": 1e-5},
            "ukf_bias_regularization_std": {"type": "number", "default": 1.0},
            "output_dir": {"type": "string", "default": "python_port/results"},
        },
    }


def load_scenario_config_json(path) -> ScenarioConfig:
    """Load and validate one scenario config from JSON."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scenario config JSON must contain an object.")
    return scenario_config_from_mapping(payload)


def scenario_config_from_mapping(payload: dict[str, Any]) -> ScenarioConfig:
    """Validate and normalize a scenario config mapping."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a mapping.")
    missing = [key for key in scenario_config_schema()["required"] if key not in payload]
    if missing:
        raise ValueError(f"Missing required scenario config field(s): {', '.join(missing)}")

    config = ScenarioConfig(
        name=_as_nonempty_string(payload["name"], "name"),
        measurement_type=_enum_value(payload["measurement_type"], ALLOWED_MEASUREMENT_TYPES, "measurement_type"),
        estimator_type=_enum_value(payload["estimator_type"], ALLOWED_ESTIMATOR_TYPES, "estimator_type"),
        start_mode=_enum_value(payload["start_mode"], ALLOWED_START_MODES, "start_mode"),
        network=_enum_value(payload["network"], ALLOWED_NETWORKS, "network"),
        duration_h=_positive_float(payload.get("duration_h", THESIS_DURATION_H), "duration_h"),
        sample_step_s=_positive_float(payload.get("sample_step_s", THESIS_SAMPLE_STEP_S), "sample_step_s"),
        max_iter=_positive_int(payload.get("max_iter", THESIS_MAX_ITER), "max_iter"),
        tol_cost_stability=_positive_float(payload.get("tol_cost_stability", 1e-8), "tol_cost_stability"),
        bls_lambda0=_positive_float(payload.get("bls_lambda0", 1e-2), "bls_lambda0"),
        rtol=_positive_float(payload.get("rtol", THESIS_RTOL), "rtol"),
        atol=_positive_float(payload.get("atol", THESIS_ATOL), "atol"),
        j2_moon=_nonnegative_float(payload.get("j2_moon", 0.0), "j2_moon"),
        noise=_boolean(payload.get("noise", False), "noise"),
        bias_mode=_bias_mode(payload.get("bias_mode", None)),
        range_rate_physics=_range_rate_physics(payload.get("range_rate_physics", "geometric_instantaneous")),
        count_interval_s=_positive_float(payload.get("count_interval_s", 60.0), "count_interval_s"),
        uplink_frequency_hz=_positive_float(payload.get("uplink_frequency_hz", 7.2e9), "uplink_frequency_hz"),
        turnaround_ratio=_positive_float(
            payload.get("turnaround_ratio", 880.0 / 749.0),
            "turnaround_ratio",
        ),
        two_way_local_state_model=_enum_value(
            payload.get("two_way_local_state_model", "ode"),
            ("ode", "taylor3"),
            "two_way_local_state_model",
        ),
        station_clock_offset_s=_finite_float(
            payload.get("station_clock_offset_s", 0.0),
            "station_clock_offset_s",
        ),
        station_clock_drift=_finite_float(
            payload.get("station_clock_drift", 0.0),
            "station_clock_drift",
        ),
        clock_reference_time_s=_finite_float(
            payload.get("clock_reference_time_s", 0.0),
            "clock_reference_time_s",
        ),
        transponder_delay_s=_nonnegative_float(
            payload.get("transponder_delay_s", 0.0),
            "transponder_delay_s",
        ),
        apply_light_time=_boolean(payload.get("apply_light_time", False), "apply_light_time"),
        apply_stellar_aberration=_boolean(
            payload.get("apply_stellar_aberration", False), "apply_stellar_aberration"
        ),
        stellar_aberration_model=_enum_value(
            payload.get("stellar_aberration_model", "local_mci"),
            ("local_mci", "spice_ssb"),
            "stellar_aberration_model",
        ),
        ukf_alpha=_positive_float(payload.get("ukf_alpha", 0.35), "ukf_alpha"),
        ukf_beta=_nonnegative_float(payload.get("ukf_beta", 2.0), "ukf_beta"),
        ukf_kappa=_finite_float(payload.get("ukf_kappa", 0.0), "ukf_kappa"),
        ukf_covariance_inflation=_at_least_one_float(
            payload.get("ukf_covariance_inflation", 1.0),
            "ukf_covariance_inflation",
        ),
        ukf_process_noise_model=_enum_value(
            payload.get("ukf_process_noise_model", "discrete"),
            ("discrete", "continuous_white_acceleration"),
            "ukf_process_noise_model",
        ),
        ukf_acceleration_psd_m2_s3=_optional_positive_float(
            payload.get("ukf_acceleration_psd_m2_s3", None),
            "ukf_acceleration_psd_m2_s3",
        ),
        ukf_adaptive_process_noise=_boolean(
            payload.get("ukf_adaptive_process_noise", False),
            "ukf_adaptive_process_noise",
        ),
        ukf_initial_process_noise_scale=_positive_float(
            payload.get("ukf_initial_process_noise_scale", 1.0),
            "ukf_initial_process_noise_scale",
        ),
        ukf_min_process_noise_scale=_positive_float(
            payload.get("ukf_min_process_noise_scale", 0.1),
            "ukf_min_process_noise_scale",
        ),
        ukf_max_process_noise_scale=_positive_float(
            payload.get("ukf_max_process_noise_scale", 100.0),
            "ukf_max_process_noise_scale",
        ),
        ukf_process_noise_adaptation_gain=_unit_interval_float(
            payload.get("ukf_process_noise_adaptation_gain", 0.2),
            "ukf_process_noise_adaptation_gain",
        ),
        ukf_adaptive_measurement_noise=_boolean(
            payload.get("ukf_adaptive_measurement_noise", False),
            "ukf_adaptive_measurement_noise",
        ),
        ukf_max_measurement_noise_scale=_at_least_one_float(
            payload.get("ukf_max_measurement_noise_scale", 100.0),
            "ukf_max_measurement_noise_scale",
        ),
        ukf_nis_gate=_optional_positive_float(payload.get("ukf_nis_gate", None), "ukf_nis_gate"),
        ukf_component_nis_gate=_optional_positive_float(
            payload.get("ukf_component_nis_gate", None),
            "ukf_component_nis_gate",
        ),
        ukf_component_gate_mode=_enum_value(
            payload.get("ukf_component_gate_mode", "marginal"),
            ("marginal", "conditional"),
            "ukf_component_gate_mode",
        ),
        ukf_robust_measurement_update=_boolean(
            payload.get("ukf_robust_measurement_update", False),
            "ukf_robust_measurement_update",
        ),
        ukf_robust_loss=_enum_value(
            payload.get("ukf_robust_loss", "student_t"),
            ("student_t", "huber"),
            "ukf_robust_loss",
        ),
        ukf_robust_student_t_dof=_positive_float(
            payload.get("ukf_robust_student_t_dof", 5.0),
            "ukf_robust_student_t_dof",
        ),
        ukf_robust_huber_threshold=_positive_float(
            payload.get("ukf_robust_huber_threshold", 3.0),
            "ukf_robust_huber_threshold",
        ),
        ukf_robust_min_component_weight=_unit_interval_float(
            payload.get("ukf_robust_min_component_weight", 0.05),
            "ukf_robust_min_component_weight",
        ),
        ukf_covariance_form=_enum_value(
            payload.get("ukf_covariance_form", "square_root"),
            ("standard", "square_root"),
            "ukf_covariance_form",
        ),
        ukf_auto_bias_constraints=_boolean(
            payload.get("ukf_auto_bias_constraints", False),
            "ukf_auto_bias_constraints",
        ),
        ukf_bias_freeze_relative_information=_nonnegative_float(
            payload.get("ukf_bias_freeze_relative_information", 1e-12),
            "ukf_bias_freeze_relative_information",
        ),
        ukf_bias_regularize_relative_information=_nonnegative_float(
            payload.get("ukf_bias_regularize_relative_information", 1e-5),
            "ukf_bias_regularize_relative_information",
        ),
        ukf_bias_regularization_std=_positive_float(
            payload.get("ukf_bias_regularization_std", 1.0),
            "ukf_bias_regularization_std",
        ),
        output_dir=_as_nonempty_string(payload.get("output_dir", "python_port/results"), "output_dir"),
    )
    _validate_cross_field_rules(config)
    return config


def write_normalized_scenario_config(config: ScenarioConfig, path) -> Path:
    """Write a normalized scenario config JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def scenario_config_summary(config: ScenarioConfig) -> str:
    """Create a one-line human-readable config summary."""
    bias = config.bias_mode or "none"
    noise = "noise" if config.noise else "clean"
    physics = f", physics={config.range_rate_physics}" if config.measurement_type == "range_rate" else ""
    return (
        f"{config.name}: {config.network} {config.measurement_type} "
        f"{config.estimator_type}/{config.start_mode}, {noise}, bias={bias}{physics}, "
        f"duration={config.duration_h:g} h"
    )


def scenario_ukf_configs(config: ScenarioConfig) -> tuple[UnscentedTransformConfig, UKFAdaptiveConfig]:
    """Build validated UKF runtime configs from a normalized scenario."""
    return (
        UnscentedTransformConfig(
            alpha=config.ukf_alpha,
            beta=config.ukf_beta,
            kappa=config.ukf_kappa,
        ),
        UKFAdaptiveConfig(
            covariance_inflation=config.ukf_covariance_inflation,
            adaptive_process_noise=config.ukf_adaptive_process_noise,
            initial_process_noise_scale=config.ukf_initial_process_noise_scale,
            min_process_noise_scale=config.ukf_min_process_noise_scale,
            max_process_noise_scale=config.ukf_max_process_noise_scale,
            process_noise_adaptation_gain=config.ukf_process_noise_adaptation_gain,
            adaptive_measurement_noise=config.ukf_adaptive_measurement_noise,
            max_measurement_noise_scale=config.ukf_max_measurement_noise_scale,
            nis_gate=config.ukf_nis_gate,
            component_nis_gate=config.ukf_component_nis_gate,
            component_gate_mode=config.ukf_component_gate_mode,
            robust_measurement_update=config.ukf_robust_measurement_update,
            robust_loss=config.ukf_robust_loss,
            robust_student_t_dof=config.ukf_robust_student_t_dof,
            robust_huber_threshold=config.ukf_robust_huber_threshold,
            robust_min_component_weight=config.ukf_robust_min_component_weight,
        ),
    )


def scenario_range_rate_physics_config(config: ScenarioConfig) -> RangeRatePhysicsConfig:
    """Build the normalized radiometric model for a scenario."""
    return RangeRatePhysicsConfig(
        mode=config.range_rate_physics,
        count_interval_s=config.count_interval_s,
        uplink_frequency_hz=config.uplink_frequency_hz,
        turnaround_ratio=config.turnaround_ratio,
        local_state_model=config.two_way_local_state_model,
        station_clock_offset_s=config.station_clock_offset_s,
        station_clock_drift=config.station_clock_drift,
        clock_reference_time_s=config.clock_reference_time_s,
        transponder_delay_s=config.transponder_delay_s,
    )


def _validate_cross_field_rules(config: ScenarioConfig) -> None:
    if config.start_mode == "sqrt_formal" and config.estimator_type != "srif":
        raise ValueError("sqrt_formal start_mode requires estimator_type='srif'.")
    if config.bias_mode is not None and config.estimator_type not in {"srif", "ukf"}:
        raise ValueError("bias solve-for modes are supported here only for estimator_type='srif' or 'ukf'.")
    if config.range_rate_physics != "geometric_instantaneous" and config.measurement_type != "range_rate":
        raise ValueError("non-geometric range_rate_physics requires measurement_type='range_rate'.")
    if config.apply_stellar_aberration and not config.apply_light_time:
        raise ValueError("apply_stellar_aberration requires apply_light_time.")
    if (config.apply_light_time or config.apply_stellar_aberration) and config.measurement_type != "position":
        raise ValueError(
            "apply_light_time / apply_stellar_aberration apply only to measurement_type='position'."
        )
    scenario_range_rate_physics_config(config)
    if config.ukf_min_process_noise_scale > config.ukf_max_process_noise_scale:
        raise ValueError("ukf_min_process_noise_scale must not exceed ukf_max_process_noise_scale.")
    if not (
        config.ukf_min_process_noise_scale
        <= config.ukf_initial_process_noise_scale
        <= config.ukf_max_process_noise_scale
    ):
        raise ValueError("ukf_initial_process_noise_scale must be within the min/max bounds.")
    if config.ukf_adaptive_process_noise and config.ukf_acceleration_psd_m2_s3 is None:
        raise ValueError("adaptive UKF process noise requires ukf_acceleration_psd_m2_s3.")
    UKFAdaptiveConfig(
        robust_measurement_update=config.ukf_robust_measurement_update,
        robust_loss=config.ukf_robust_loss,
        robust_student_t_dof=config.ukf_robust_student_t_dof,
        robust_huber_threshold=config.ukf_robust_huber_threshold,
        robust_min_component_weight=config.ukf_robust_min_component_weight,
    )
    if config.ukf_bias_regularize_relative_information < config.ukf_bias_freeze_relative_information:
        raise ValueError(
            "ukf_bias_regularize_relative_information must not be below "
            "ukf_bias_freeze_relative_information."
        )
    if (
        config.ukf_acceleration_psd_m2_s3 is not None
        and config.ukf_process_noise_model != "continuous_white_acceleration"
    ):
        raise ValueError(
            "ukf_acceleration_psd_m2_s3 requires ukf_process_noise_model='continuous_white_acceleration'."
        )


def _enum_value(value: Any, allowed: tuple[str, ...], field_name: str):
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}; got {value!r}.")
    return value


def _bias_mode(value: Any) -> str | None:
    if value not in ALLOWED_BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {ALLOWED_BIAS_MODES}; got {value!r}.")
    return value


def _range_rate_physics(value: Any) -> str:
    if value not in ALLOWED_RANGE_RATE_PHYSICS:
        raise ValueError(f"range_rate_physics must be one of {ALLOWED_RANGE_RATE_PHYSICS}; got {value!r}.")
    return str(value)


def _as_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _nonnegative_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return result


def _at_least_one_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result < 1.0:
        raise ValueError(f"{field_name} must be at least 1.0.")
    return result


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    return _positive_float(value, field_name)


def _unit_interval_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return result


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return result
