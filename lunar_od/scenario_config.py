"""JSON scenario configuration schema and validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import (
    J2_EARTH_UNNORMALIZED,
    MU_EARTH_M3S2,
    MU_MOON_M3S2,
    MU_SUN_M3S2,
    R_EARTH_J2_REF_M,
    R_MOON_M,
)
from .force_contract import (
    FORCE_CONTRACT_SCHEMA_VERSION,
    ConsumerReadiness,
    ConsumerRole,
    EarthJ2ForceContract,
    ForceElementStatus,
    ForceModelContract,
    ForceModelMismatchPolicy,
    ForceModelParityDecision,
    LunarHarmonicsForceContract,
    LunarJ2ForceContract,
    OrientationPolicy,
    PointMassForceContract,
    ThirdBodyForceContract,
    capability_map,
    evaluate_force_model_parity,
    lunar_j2_enabled,
    sha256_hex_of_bytes,
)

from .filters import (
    UKFAdaptiveConfig,
    UnscentedTransformConfig,
    validate_ukf_measurement_support,
)
from .gravity_harmonics import SphericalHarmonicGravityModel
from .gravity_model_loader import load_lunar_gravity_model, resolve_gravity_dir
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

ALLOWED_MEASUREMENT_TYPES = ("position", "range_rate", "two_way_range")
ALLOWED_TWO_WAY_RANGE_CONVENTIONS = (
    "raw_half_round_trip",
    "delay_calibrated_half_round_trip",
)
ALLOWED_ESTIMATOR_TYPES = ("bls_lm", "srif", "ukf")
ALLOWED_START_MODES = ("cold", "hot", "formal", "sqrt_formal")
ALLOWED_NETWORKS = tuple(network.name for network in THESIS_NETWORKS)
ALLOWED_BIAS_MODES = (None, "global", "station_angles", "station_full")
ALLOWED_RANGE_RATE_PHYSICS = ("geometric_instantaneous", "two_way_counted_doppler")
ALLOWED_MEASUREMENT_MODEL_PROFILES = (
    "geometric_instantaneous",
    "one_way_light_time",
    "one_way_light_time_aberrated_local_mci",
    "one_way_light_time_aberrated_spice_ssb",
)
ALLOWED_COMPANION_GEOMETRIES = ("instantaneous", "apparent_one_way")
ALLOWED_JACOBIAN_MODELS = (
    "analytic_exact_geometric",
    "analytic_first_order_light_time",
    "implicit_light_time",
    "finite_difference_reference",
)
ALLOWED_EARTH_J2_MODES = ("indirect", "direct")
# Bare "MOON_PA" is deliberately NOT allowed: the DE421 and DE440 frame
# kernels both define the FRAME_MOON_PA alias, so a bare name silently follows
# whichever kernel pool was furnished last.  Explicit versioned names only.
ALLOWED_LUNAR_GRAVITY_FRAMES = ("MOON_PA_DE421", "MOON_PA_DE440")
ALLOWED_LUNAR_KERNEL_PROFILES = (None, "DE421", "DE440")
_LUNAR_FRAME_TO_PROFILE = {"MOON_PA_DE421": "DE421", "MOON_PA_DE440": "DE440"}


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
    enable_earth_j2: bool = False
    earth_j2_mode: str = "indirect"
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
    # transponder_delay_s is shared by two measurement types: it enters the
    # counted-Doppler RangeRatePhysicsConfig (measurement_type='range_rate')
    # and the M3 TwoWayRangeConfig (measurement_type='two_way_range').
    transponder_delay_s: float = 0.0
    two_way_range_convention: str = "delay_calibrated_half_round_trip"
    apply_light_time: bool = False
    apply_stellar_aberration: bool = False
    stellar_aberration_model: str = "local_mci"
    measurement_model_profile: str = "geometric_instantaneous"
    companion_geometry: str = "instantaneous"
    jacobian_model: str = "analytic_exact_geometric"
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
    enable_lunar_harmonics: bool = False
    lunar_gravity_model_path: str | None = None
    lunar_gravity_nmax: int | None = None
    lunar_gravity_mmax: int | None = None
    lunar_gravity_frame: str = "MOON_PA_DE421"
    lunar_gravity_rotation_cadence_s: float = 60.0
    lunar_gravity_rotation_margin_s: float | None = None
    lunar_gravity_kernel_profile: str | None = None
    # Run-level force-model execution policy (R0B-2). Deliberately NOT part of
    # ForceModelContract: the contract describes physics, this describes what
    # the run is permitted to do with it.
    allow_explicit_force_model_mismatch: bool = False
    force_model_mismatch_reason: str | None = None

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
            "enable_earth_j2": {"type": "boolean", "default": False},
            "earth_j2_mode": {"enum": list(ALLOWED_EARTH_J2_MODES), "default": "indirect"},
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
            "two_way_range_convention": {
                "enum": list(ALLOWED_TWO_WAY_RANGE_CONVENTIONS),
                "default": "delay_calibrated_half_round_trip",
            },
            "apply_light_time": {"type": "boolean", "default": False},
            "apply_stellar_aberration": {"type": "boolean", "default": False},
            "stellar_aberration_model": {"enum": ["local_mci", "spice_ssb"], "default": "local_mci"},
            "measurement_model_profile": {
                "enum": list(ALLOWED_MEASUREMENT_MODEL_PROFILES),
                "default": "geometric_instantaneous",
            },
            "companion_geometry": {
                "enum": list(ALLOWED_COMPANION_GEOMETRIES),
                "default": "instantaneous",
            },
            "jacobian_model": {
                "enum": list(ALLOWED_JACOBIAN_MODELS),
                "default": "analytic_exact_geometric",
            },
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
            "enable_lunar_harmonics": {"type": "boolean", "default": False},
            "lunar_gravity_model_path": {"type": ["string", "null"], "default": None},
            "lunar_gravity_nmax": {"type": ["integer", "null"], "default": None},
            "lunar_gravity_mmax": {"type": ["integer", "null"], "default": None},
            "lunar_gravity_frame": {
                "enum": list(ALLOWED_LUNAR_GRAVITY_FRAMES),
                "default": "MOON_PA_DE421",
            },
            "lunar_gravity_rotation_cadence_s": {"type": "number", "default": 60.0},
            "lunar_gravity_rotation_margin_s": {"type": ["number", "null"], "default": None},
            "lunar_gravity_kernel_profile": {
                "enum": list(ALLOWED_LUNAR_KERNEL_PROFILES),
                "default": None,
            },
            "allow_explicit_force_model_mismatch": {"type": "boolean", "default": False},
            "force_model_mismatch_reason": {"type": ["string", "null"], "default": None},
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
        enable_earth_j2=_boolean(payload.get("enable_earth_j2", False), "enable_earth_j2"),
        earth_j2_mode=_enum_value(
            payload.get("earth_j2_mode", "indirect"),
            ALLOWED_EARTH_J2_MODES,
            "earth_j2_mode",
        ),
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
        two_way_range_convention=_enum_value(
            payload.get("two_way_range_convention", "delay_calibrated_half_round_trip"),
            ALLOWED_TWO_WAY_RANGE_CONVENTIONS,
            "two_way_range_convention",
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
        measurement_model_profile=_enum_value(
            payload.get("measurement_model_profile", "geometric_instantaneous"),
            ALLOWED_MEASUREMENT_MODEL_PROFILES,
            "measurement_model_profile",
        ),
        companion_geometry=_enum_value(
            payload.get("companion_geometry", "instantaneous"),
            ALLOWED_COMPANION_GEOMETRIES,
            "companion_geometry",
        ),
        jacobian_model=_enum_value(
            payload.get("jacobian_model", "analytic_exact_geometric"),
            ALLOWED_JACOBIAN_MODELS,
            "jacobian_model",
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
        enable_lunar_harmonics=_boolean(
            payload.get("enable_lunar_harmonics", False),
            "enable_lunar_harmonics",
        ),
        lunar_gravity_model_path=_optional_nonempty_string(
            payload.get("lunar_gravity_model_path", None),
            "lunar_gravity_model_path",
        ),
        lunar_gravity_nmax=_optional_bounded_int(
            payload.get("lunar_gravity_nmax", None),
            "lunar_gravity_nmax",
            minimum=2,
        ),
        lunar_gravity_mmax=_optional_bounded_int(
            payload.get("lunar_gravity_mmax", None),
            "lunar_gravity_mmax",
            minimum=0,
        ),
        lunar_gravity_frame=_enum_value(
            payload.get("lunar_gravity_frame", "MOON_PA_DE421"),
            ALLOWED_LUNAR_GRAVITY_FRAMES,
            "lunar_gravity_frame",
        ),
        lunar_gravity_rotation_cadence_s=_positive_float(
            payload.get("lunar_gravity_rotation_cadence_s", 60.0),
            "lunar_gravity_rotation_cadence_s",
        ),
        lunar_gravity_rotation_margin_s=_optional_positive_float(
            payload.get("lunar_gravity_rotation_margin_s", None),
            "lunar_gravity_rotation_margin_s",
        ),
        lunar_gravity_kernel_profile=_lunar_kernel_profile(
            payload.get("lunar_gravity_kernel_profile", None)
        ),
        allow_explicit_force_model_mismatch=_boolean(
            payload.get("allow_explicit_force_model_mismatch", False),
            "allow_explicit_force_model_mismatch",
        ),
        force_model_mismatch_reason=_optional_reason(
            payload.get("force_model_mismatch_reason", None)
        ),
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
    profile = (
        f", profile={config.measurement_model_profile}"
        if config.measurement_model_profile != "geometric_instantaneous"
        else ""
    )
    companion = (
        f", companion={config.companion_geometry}"
        if config.measurement_type == "range_rate" and config.companion_geometry != "instantaneous"
        else ""
    )
    return (
        f"{config.name}: {config.network} {config.measurement_type} "
        f"{config.estimator_type}/{config.start_mode}, {noise}, bias={bias}{physics}{profile}{companion}, "
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


def scenario_two_way_range_config(config: ScenarioConfig) -> "TwoWayRangeConfig":
    """Build the normalized M3 two-way range model for a scenario.

    The shared ``transponder_delay_s`` scenario field feeds this config for
    measurement_type='two_way_range' (and the counted-Doppler config for
    measurement_type='range_rate').
    """
    from .two_way_range import TwoWayRangeConfig

    return TwoWayRangeConfig(
        transponder_delay_s=config.transponder_delay_s,
        convention=config.two_way_range_convention,
    )


def scenario_lunar_kernel_profile(config: ScenarioConfig) -> str:
    """Effective SPICE kernel profile: explicit value, or derived from the frame."""
    if config.lunar_gravity_kernel_profile is not None:
        return config.lunar_gravity_kernel_profile
    return _LUNAR_FRAME_TO_PROFILE[config.lunar_gravity_frame]


def resolve_lunar_gravity_model_path(config: ScenarioConfig) -> Path:
    """Resolve the configured lunar gravity coefficient file to an existing path.

    Single source of truth for the resolution + existence contract, shared by
    the model loader and the force-contract mapping so the two cannot drift.
    """
    if config.lunar_gravity_model_path is None:
        raise ValueError(
            "enable_lunar_harmonics=True requires lunar_gravity_model_path "
            "(no automatic model selection)."
        )
    if config.lunar_gravity_nmax is None:
        raise ValueError(
            "enable_lunar_harmonics=True requires an explicit lunar_gravity_nmax truncation."
        )
    path = Path(config.lunar_gravity_model_path)
    if not path.is_absolute():
        path = resolve_gravity_dir() / path
    if not path.is_file():
        raise FileNotFoundError(
            f"lunar gravity model file not found: {path} "
            f"(from lunar_gravity_model_path={config.lunar_gravity_model_path!r})."
        )
    return path


def scenario_lunar_gravity_model(config: ScenarioConfig) -> SphericalHarmonicGravityModel | None:
    """Load the configured lunar gravity model once, at setup time.

    Returns ``None`` (touching no file) when ``enable_lunar_harmonics`` is
    False.  Otherwise the model path is mandatory (no automatic model
    selection): an absolute path is used as-is; a relative path is resolved
    against ``resolve_gravity_dir()`` (env ``LUNAR_OD_GRAVITY_DIR`` ->
    ``~/Documents/mice/gravity`` -> ``<python_port>/data/gravity``).  The
    production loader is called exactly once; nothing here runs per-RHS.
    """
    if not config.enable_lunar_harmonics:
        return None
    path = resolve_lunar_gravity_model_path(config)
    model = load_lunar_gravity_model(
        path,
        nmax=config.lunar_gravity_nmax,
        mmax=config.lunar_gravity_mmax,
    )
    # Mirror of the propagation-time double-count guard for callers that
    # construct ScenarioConfig directly (bypassing scenario_config_from_mapping).
    if float(model.cbar[2, 0]) != 0.0 and config.j2_moon != 0.0:
        raise ValueError(
            "lunar harmonics model includes a nonzero C20 (J2) term; combining it "
            "with j2_moon != 0 would count J2 twice. Set j2_moon=0."
        )
    return model


_HARMONICS_DISABLED_CONTRACT = LunarHarmonicsForceContract(
    enabled=False,
    model_identity="not_configured",
    coefficient_file_sha256=None,
    degree_nmax=None,
    order_mmax=None,
    normalization="not_applicable",
    model_gravitational_parameter_m3_s2=None,
    reference_radius_m=None,
    body_frame="not_applicable",
    kernel_profile_policy="not_applicable",
    rotation_cadence_s=None,
    rotation_margin_s=None,
)


def _lunar_harmonics_force_contract(config: ScenarioConfig) -> LunarHarmonicsForceContract:
    """Build the harmonics contract, hashing coefficient CONTENT (never a path).

    When harmonics are disabled the path-derived fields stay neutral so an
    inert ``lunar_gravity_model_path`` cannot leak into the fingerprint.
    """
    if not config.enable_lunar_harmonics:
        return _HARMONICS_DISABLED_CONTRACT
    path = resolve_lunar_gravity_model_path(config)
    model = scenario_lunar_gravity_model(config)
    content_sha256 = sha256_hex_of_bytes(path.read_bytes())
    metadata = dict(getattr(model, "metadata", {}) or {})
    identity = str(
        metadata.get("model_name")
        or metadata.get("name")
        or metadata.get("title")
        or "unnamed_lunar_gravity_model"
    )
    return LunarHarmonicsForceContract(
        enabled=True,
        model_identity=identity,
        coefficient_file_sha256=content_sha256,
        degree_nmax=int(model.nmax),
        order_mmax=int(model.mmax),
        normalization=str(metadata.get("normalization", "fully_normalized_4pi")),
        model_gravitational_parameter_m3_s2=float(model.mu_m3_s2),
        reference_radius_m=float(model.r_ref_m),
        body_frame=str(config.lunar_gravity_frame),
        kernel_profile_policy=str(scenario_lunar_kernel_profile(config)),
        rotation_cadence_s=float(config.lunar_gravity_rotation_cadence_s),
        rotation_margin_s=(
            None
            if config.lunar_gravity_rotation_margin_s is None
            else float(config.lunar_gravity_rotation_margin_s)
        ),
    )


def _consumer_capabilities_for(
    *, lunar_j2_on: bool, earth_j2_on: bool, harmonics_on: bool
):
    """Capability matrix for the configured force set.

    Precedence is worst-status-first: an experimental or unsupported element
    downgrades the roles it reaches, and R0B never reports a role as
    ``verified`` on the strength of a force that R0A/R1 has not qualified
    there. SCI-003 is NOT claimed closed anywhere in this matrix.
    """
    if harmonics_on:
        # High-degree harmonics: direct truth trajectory only.
        statuses = {role: ConsumerReadiness.UNSUPPORTED for role in ConsumerRole}
        statuses[ConsumerRole.TRUTH_STATE] = ConsumerReadiness.UNSUPPORTED
        return capability_map(statuses)
    if earth_j2_on:
        # Earth J2 is unsupported on every official OD role (R0A fail-closed).
        return capability_map(
            {role: ConsumerReadiness.UNSUPPORTED for role in ConsumerRole}
        )
    if lunar_j2_on:
        # R0A verified the propagation roles; posterior/observability await R1.
        statuses = {role: ConsumerReadiness.VERIFIED for role in ConsumerRole}
        statuses[ConsumerRole.POSTERIOR_COVARIANCE] = ConsumerReadiness.PENDING_R1
        statuses[ConsumerRole.OBSERVABILITY] = ConsumerReadiness.PENDING_R1
        return capability_map(statuses)
    # Point-mass + third-body only: the long-standing verified baseline.
    return capability_map({role: ConsumerReadiness.VERIFIED for role in ConsumerRole})


def force_model_contract_from_scenario_config(
    config: ScenarioConfig,
    *,
    mu_moon_m3_s2: float = MU_MOON_M3S2,
    mu_earth_m3_s2: float = MU_EARTH_M3S2,
    mu_sun_m3_s2: float = MU_SUN_M3S2,
    force_ephemeris_policy: str = "moon_centered_sampled_ephemeris",
) -> ForceModelContract:
    """Derive the immutable force contract implied by ``config``.

    Consumes every force-related ScenarioConfig field, turning implicit
    defaults into explicit contract values. Gravitational parameters are
    supplied by the caller because they come from the run fixture, not from
    the scenario config. This function never calls a propagator.
    """
    j2_moon = float(config.j2_moon)
    lunar_j2_on = lunar_j2_enabled(j2_moon)
    earth_j2_on = bool(config.enable_earth_j2)
    harmonics = _lunar_harmonics_force_contract(config)
    return ForceModelContract(
        schema_version=FORCE_CONTRACT_SCHEMA_VERSION,
        lunar_point_mass=PointMassForceContract(
            enabled=True,
            body="moon",
            gravitational_parameter_m3_s2=float(mu_moon_m3_s2),
            policy="central_body_point_mass",
        ),
        earth_third_body=ThirdBodyForceContract(
            enabled=bool(mu_earth_m3_s2),
            body="earth",
            gravitational_parameter_m3_s2=float(mu_earth_m3_s2),
            policy="moon_centered_third_body_point_mass",
        ),
        sun_third_body=ThirdBodyForceContract(
            enabled=bool(mu_sun_m3_s2),
            body="sun",
            gravitational_parameter_m3_s2=float(mu_sun_m3_s2),
            policy="moon_centered_third_body_point_mass",
        ),
        lunar_j2=LunarJ2ForceContract(
            enabled=lunar_j2_on,
            coefficient=j2_moon,
            # The propagator pairs a nonzero J2 with R_MOON_M; a disabled term
            # contributes no radius to the evaluated physics.
            reference_radius_m=float(R_MOON_M) if lunar_j2_on else 0.0,
            orientation_policy=(
                OrientationPolicy.CONSTANT_IAU2006_MOON_MEAN_POLE
                if lunar_j2_on
                else OrientationPolicy.NOT_APPLICABLE
            ),
            status=ForceElementStatus.SUPPORTED,
        ),
        earth_j2=EarthJ2ForceContract(
            enabled=earth_j2_on,
            mode=str(config.earth_j2_mode),
            coefficient=float(J2_EARTH_UNNORMALIZED) if earth_j2_on else 0.0,
            reference_radius_m=float(R_EARTH_J2_REF_M) if earth_j2_on else 0.0,
            orientation_policy=(
                OrientationPolicy.EXPERIMENTAL_IDENTITY_J2000_TO_EARTH_BODY_FIXED
                if earth_j2_on
                else OrientationPolicy.NOT_APPLICABLE
            ),
            status=ForceElementStatus.UNSUPPORTED_OFFICIAL_OD,
        ),
        lunar_harmonics=harmonics,
        force_ephemeris_policy=str(force_ephemeris_policy),
        consumer_capabilities=_consumer_capabilities_for(
            lunar_j2_on=lunar_j2_on,
            earth_j2_on=earth_j2_on,
            harmonics_on=bool(config.enable_lunar_harmonics),
        ),
    )


def _optional_reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("force_model_mismatch_reason must be a string or null.")
    return value


def force_model_mismatch_policy_from_scenario_config(
    config: ScenarioConfig,
) -> ForceModelMismatchPolicy:
    """Return the run-level mismatch policy declared by ``config``."""
    return ForceModelMismatchPolicy(
        enabled=bool(config.allow_explicit_force_model_mismatch),
        reason=config.force_model_mismatch_reason,
    )


def scenario_force_model_preflight(
    config: ScenarioConfig,
    *,
    mu_moon_m3_s2: float = MU_MOON_M3S2,
    mu_earth_m3_s2: float = MU_EARTH_M3S2,
    mu_sun_m3_s2: float = MU_SUN_M3S2,
    truth_contract: ForceModelContract | None = None,
    estimator_contract: ForceModelContract | None = None,
    context: str = "scenario force-model preflight",
) -> ForceModelParityDecision:
    """Shared truth/estimator force-parity preflight (JSON runner + desktop).

    Both official entry points call THIS helper so the contract is never
    reproduced two different ways. Contracts default to the one implied by
    ``config`` (the matched case); campaigns that deliberately propagate truth
    with different physics pass an explicit ``truth_contract``.

    Raises before any fixture read, SPICE load, or propagation.
    """
    derived = force_model_contract_from_scenario_config(
        config,
        mu_moon_m3_s2=mu_moon_m3_s2,
        mu_earth_m3_s2=mu_earth_m3_s2,
        mu_sun_m3_s2=mu_sun_m3_s2,
    )
    return evaluate_force_model_parity(
        truth_contract if truth_contract is not None else derived,
        estimator_contract if estimator_contract is not None else derived,
        force_model_mismatch_policy_from_scenario_config(config),
        context=context,
    )


def validate_official_earth_j2_support(enable_earth_j2: bool, *, context: str) -> None:
    """Reject Earth J2 on official OD paths until a real Earth orientation exists (R0A).

    Shared fail-closed gate for both enforcement layers: the scenario-config
    loader (``_validate_cross_field_rules``) and the JSON scenario runner
    (``examples/run_scenario_config.py``), so directly constructed
    ``ScenarioConfig`` objects cannot bypass the rule. The dynamics-layer
    Earth-J2 terms currently use an identity J2000-to-Earth-body-fixed
    orientation that is only an experimental direct-trajectory approximation,
    not an IERS-compliant Earth orientation; the experimental direct-trajectory
    API itself remains available outside the official scenario/desktop/UKF
    paths. The rejection fires before any state/STM/sigma propagation.
    """
    if enable_earth_j2:
        raise ValueError(
            f"enable_earth_j2=True is not supported on the official OD path ({context}): "
            "the current J2000-to-Earth-body-fixed orientation is an identity-matrix "
            "experimental direct-trajectory approximation, not an IERS-compliant Earth "
            "orientation. Keep enable_earth_j2=False (Earth J2 remains available only "
            "through the experimental direct dynamics API) until a validated Earth "
            "orientation contract is implemented."
        )


def _validate_cross_field_rules(config: ScenarioConfig) -> None:
    # R0A fail-closed loader gate (also enforced by the JSON scenario runner).
    validate_official_earth_j2_support(
        config.enable_earth_j2, context="scenario_config cross-field validation"
    )
    if config.start_mode == "sqrt_formal" and config.estimator_type != "srif":
        raise ValueError("sqrt_formal start_mode requires estimator_type='srif'.")
    if config.bias_mode is not None and config.estimator_type not in {"srif", "ukf"}:
        raise ValueError("bias solve-for modes are supported here only for estimator_type='srif' or 'ukf'.")
    if config.range_rate_physics != "geometric_instantaneous" and config.measurement_type != "range_rate":
        raise ValueError("non-geometric range_rate_physics requires measurement_type='range_rate'.")
    # FA-01 loader gate: UKF position measurements only support the geometric
    # instantaneous profile (shared helper; also enforced at run_lunar_ukf).
    validate_ukf_measurement_support(
        config.estimator_type,
        config.measurement_type,
        config.measurement_model_profile,
        apply_light_time=config.apply_light_time,
        apply_stellar_aberration=config.apply_stellar_aberration,
    )
    if config.measurement_type == "two_way_range":
        if config.estimator_type == "ukf":
            raise ValueError(
                "measurement_type='two_way_range' is not supported by the UKF in M3; "
                "use estimator_type='bls_lm' or 'srif'."
            )
        if config.bias_mode is not None:
            raise ValueError(
                "measurement_type='two_way_range' does not support bias solve-for "
                "modes in M3."
            )
        scenario_two_way_range_config(config)
    elif config.two_way_range_convention != "delay_calibrated_half_round_trip":
        raise ValueError(
            "two_way_range_convention applies only to measurement_type='two_way_range'."
        )
    profile_controls_position = config.measurement_model_profile != "geometric_instantaneous"
    if profile_controls_position and config.measurement_type != "position":
        raise ValueError(
            "non-geometric measurement_model_profile applies only to measurement_type='position'; "
            "use companion_geometry for range_rate companion range/angles."
        )
    if not profile_controls_position:
        if config.apply_stellar_aberration and not config.apply_light_time:
            raise ValueError("apply_stellar_aberration requires apply_light_time.")
        if (config.apply_light_time or config.apply_stellar_aberration) and config.measurement_type != "position":
            raise ValueError(
                "apply_light_time / apply_stellar_aberration apply only to measurement_type='position'."
            )
    if config.companion_geometry != "instantaneous" and config.measurement_type != "range_rate":
        raise ValueError("companion_geometry applies only to measurement_type='range_rate'.")
    if config.jacobian_model == "implicit_light_time" and not (
        profile_controls_position or config.apply_light_time or config.companion_geometry == "apparent_one_way"
    ):
        raise ValueError(
            "jacobian_model='implicit_light_time' requires a light-time corrected "
            "measurement profile or apparent companion geometry."
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
    # Lunar harmonics rules run ONLY when enabled: with harmonics off the
    # legacy parse path must stay byte-identical and touch no file system.
    if config.enable_lunar_harmonics:
        if config.lunar_gravity_model_path is None:
            raise ValueError(
                "enable_lunar_harmonics=True requires lunar_gravity_model_path "
                "(no automatic model selection)."
            )
        if config.lunar_gravity_nmax is None:
            raise ValueError(
                "enable_lunar_harmonics=True requires an explicit lunar_gravity_nmax truncation."
            )
        if (
            config.lunar_gravity_mmax is not None
            and config.lunar_gravity_mmax > config.lunar_gravity_nmax
        ):
            raise ValueError("lunar_gravity_mmax must satisfy 0 <= mmax <= nmax.")
        expected_profile = _LUNAR_FRAME_TO_PROFILE[config.lunar_gravity_frame]
        if (
            config.lunar_gravity_kernel_profile is not None
            and config.lunar_gravity_kernel_profile != expected_profile
        ):
            raise ValueError(
                f"lunar_gravity_frame {config.lunar_gravity_frame!r} requires "
                f"lunar_gravity_kernel_profile {expected_profile!r} (or None to derive it); "
                f"got {config.lunar_gravity_kernel_profile!r}."
            )
        if config.j2_moon != 0.0:
            raise ValueError(
                "enable_lunar_harmonics=True with j2_moon != 0 would count lunar J2 "
                "twice; set j2_moon=0 (the model's C20 term already contains it)."
            )
        if config.estimator_type in ("bls_lm", "srif"):
            raise ValueError(
                "lunar harmonics gradient not implemented; STM-based estimators "
                "cannot use harmonics — use estimator_type='ukf' or disable lunar harmonics."
            )
        # TEMPORARY Phase 13B2a guard (removed in Phase 13B2b): the scenario
        # runner does not consume these fields yet; accepting the config would
        # silently propagate WITHOUT harmonics, which is worse than refusing.
        raise ValueError(
            "enable_lunar_harmonics is not yet consumed by the scenario runner "
            "(Phase 13B2b); use the direct propagate_state API"
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


def _optional_nonempty_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_nonempty_string(value, field_name)


def _optional_bounded_int(value: Any, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return result


def _lunar_kernel_profile(value: Any) -> str | None:
    if value not in ALLOWED_LUNAR_KERNEL_PROFILES:
        raise ValueError(
            f"lunar_gravity_kernel_profile must be one of "
            f"{ALLOWED_LUNAR_KERNEL_PROFILES}; got {value!r}."
        )
    return value


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
