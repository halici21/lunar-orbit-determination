"""Canonical thesis scenario matrix definitions.

This module freezes the compact factorial matrix used by
``examples/thesis_factorial_report.py`` so the thesis table is not an
implicit set of nested loops hidden inside the report script.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenarios import EstimatorType, MeasurementType

ThesisStartMode = str


@dataclass(frozen=True)
class ThesisNetwork:
    name: str
    station_names: tuple[str, ...]


@dataclass(frozen=True)
class ThesisScenarioCase:
    network: str
    measurement_type: MeasurementType
    estimator_type: EstimatorType
    start_mode: ThesisStartMode

    @property
    def label(self) -> str:
        return f"{self.network} {self.measurement_type} {self.estimator_type} {self.start_mode}"


THESIS_DURATION_H = 4.0
THESIS_SAMPLE_STEP_S = 240.0
THESIS_EPHEMERIS_STEP_S = 600.0
THESIS_MAX_GAP_S = 20.0 * 60.0
THESIS_MIN_ELEVATION_DEG = 10.0
THESIS_COLD_START_SIGMA_POS_M = 180.0
THESIS_COLD_START_SIGMA_VEL_MPS = 0.03
THESIS_MAX_ITER = 10
THESIS_RTOL = 1e-10
THESIS_ATOL = 1e-11

THESIS_NETWORKS: tuple[ThesisNetwork, ...] = (
    ThesisNetwork("single", ("Canberra DSN",)),
    ThesisNetwork("multi", ("Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga")),
)
THESIS_MEASUREMENT_TYPES: tuple[MeasurementType, ...] = ("position", "range_rate")
THESIS_ESTIMATOR_TYPES: tuple[EstimatorType, ...] = ("bls_lm", "srif", "ukf")
THESIS_START_MODES: tuple[ThesisStartMode, ...] = ("cold", "hot")

THESIS_FACTORIAL_CASES: tuple[ThesisScenarioCase, ...] = tuple(
    ThesisScenarioCase(
        network=network.name,
        measurement_type=measurement_type,
        estimator_type=estimator_type,
        start_mode=start_mode,
    )
    for network in THESIS_NETWORKS
    for measurement_type in THESIS_MEASUREMENT_TYPES
    for estimator_type in THESIS_ESTIMATOR_TYPES
    for start_mode in THESIS_START_MODES
)


def thesis_network_by_name(name: str) -> ThesisNetwork:
    """Return a canonical thesis network by name."""
    for network in THESIS_NETWORKS:
        if network.name == name:
            return network
    raise KeyError(f"Unknown thesis network: {name}")


def thesis_seed_for(network_name: str, measurement_type: str) -> int:
    """Stable cold-start seed for one network/measurement block."""
    return 240700 + sum(ord(ch) for ch in f"{network_name}:{measurement_type}")


def thesis_case_rows() -> tuple[dict[str, str], ...]:
    """Return simple serializable rows for the frozen thesis matrix."""
    return tuple(
        {
            "network": case.network,
            "measurement_type": case.measurement_type,
            "estimator_type": case.estimator_type,
            "start_mode": case.start_mode,
            "label": case.label,
        }
        for case in THESIS_FACTORIAL_CASES
    )
