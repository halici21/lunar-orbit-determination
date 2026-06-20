"""Scenario configuration model and built-in presets."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ScenarioModel:
    name: str = "unnamed"
    description: str = ""
    duration_days: float = 1.0
    state_step_s: float = 600.0
    ephemeris_step_s: float = 3600.0
    arc_mode: str = "prescribed"          # prescribed | visibility
    arc_duration_h: float = 2.0
    arc_stride_h: float = 6.0
    min_visibility_samples: int = 6
    max_arcs: Optional[int] = None
    visibility_enabled: bool = True
    elevation_mask_deg: float = 10.0
    gap_threshold_s: float = 300.0
    network_name: str = "multi"           # single | multi | dsn | itu
    measurement_type: str = "range_rate"  # position | range_rate
    range_rate_physics: str = "geometric_instantaneous"  # geometric_instantaneous | two_way_counted_doppler
    count_interval_s: float = 60.0
    uplink_frequency_hz: float = 7.2e9
    turnaround_ratio: float = 880.0 / 749.0
    apply_light_time: bool = False
    apply_stellar_aberration: bool = False
    stellar_aberration_model: str = "local_mci"
    noise_enabled: bool = True
    random_seed: int = 42
    estimator_type: str = "bls_lm"        # bls_lm | srif | ukf
    start_mode: str = "cold"              # cold | hot | formal | sqrt_formal
    bias_mode: str = "none"
    output_dir: str = "results"

    # BLS-LM parameters
    bls_max_iter: int = 10
    bls_damping_init: float = 1e-3
    bls_rtol: float = 1e-8
    bls_atol: float = 1e-10

    # SR-UKF parameters
    ukf_alpha: float = 0.35
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_covariance_inflation: float = 1.0
    ukf_nis_gate: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioModel":
        m = cls()
        valid_fields = {f.name for f in m.__dataclass_fields__.values()}
        for k, v in d.items():
            if k in valid_fields:
                setattr(m, k, v)
        return m

    @classmethod
    def from_json(cls, s: str) -> "ScenarioModel":
        return cls.from_dict(json.loads(s))


# Built-in scenario presets ---------------------------------------------------

PRESETS: dict[str, ScenarioModel] = {
    "Gaussian Baseline": ScenarioModel(
        name="gaussian_baseline",
        description="1-day BLS-LM vs SR-UKF under prescribed arcs. "
                    "Cold start, range-rate, multi-station.",
        duration_days=1.0,
        arc_mode="prescribed",
        arc_duration_h=2.0,
        arc_stride_h=6.0,
        measurement_type="range_rate",
        range_rate_physics="geometric_instantaneous",
        estimator_type="bls_lm",
        start_mode="cold",
        network_name="multi",
        noise_enabled=True,
        random_seed=42,
    ),
    "28-Day Stability": ScenarioModel(
        name="28day_stability",
        description="Long-duration numerical stability and estimator comparison. "
                    "112 prescribed arcs, formal handoff.",
        duration_days=28.0,
        arc_mode="prescribed",
        arc_duration_h=2.0,
        arc_stride_h=6.0,
        measurement_type="range_rate",
        estimator_type="bls_lm",
        start_mode="formal",
        network_name="multi",
        noise_enabled=True,
    ),
    "Sequential Tracking": ScenarioModel(
        name="sequential_tracking",
        description="3-day fragmented SPICE visibility. BLS arc-end markers "
                    "vs SR-UKF continuous history.",
        duration_days=3.0,
        arc_mode="visibility",
        visibility_enabled=True,
        elevation_mask_deg=10.0,
        gap_threshold_s=300.0,
        measurement_type="range_rate",
        estimator_type="ukf",
        start_mode="cold",
        network_name="multi",
        noise_enabled=True,
    ),
    "Two-Way Doppler": ScenarioModel(
        name="two_way_doppler",
        description="Simplified two-way counted-Doppler test. "
                    "1-day, cold vs formal, computational cost comparison.",
        duration_days=1.0,
        arc_mode="prescribed",
        arc_duration_h=2.0,
        arc_stride_h=6.0,
        measurement_type="range_rate",
        range_rate_physics="two_way_counted_doppler",
        count_interval_s=60.0,
        estimator_type="bls_lm",
        start_mode="cold",
        network_name="multi",
        noise_enabled=True,
    ),
    "BLS Ablation": ScenarioModel(
        name="bls_ablation",
        description="BLS-LM range-rate ablation: position-only vs range-rate, "
                    "cold/hot/formal starts, geometric vs Doppler.",
        duration_days=3.0,
        arc_mode="prescribed",
        measurement_type="range_rate",
        estimator_type="bls_lm",
        start_mode="cold",
        network_name="multi",
        noise_enabled=True,
    ),
    "28-Day Network Visibility": ScenarioModel(
        name="28day_visibility_gantt",
        description="28-day ITU + DSN station visibility Gantt chart. "
                    "No estimation — visibility analysis only.",
        duration_days=28.0,
        arc_mode="visibility",
        visibility_enabled=True,
        elevation_mask_deg=10.0,
        gap_threshold_s=300.0,
        measurement_type="position",
        network_name="multi",
        noise_enabled=False,
    ),
}
