"""Immutable force-model contracts, canonical payloads, and SHA-256 fingerprints.

R0B-1. This module is the canonical owner of the force-model identity used to
decide whether two runs evaluated the SAME physics. It answers one question:
"which forces, with which effective constants and capability status, were
actually evaluated?"

Import direction (enforced by ``tests/test_force_contract.py``)::

    scenario_config -> force_contract -> primitive values

``force_contract`` must never import ``scenario_config``, ``scenarios``,
``filters``, ``estimators``, ``reporting``, ``dynamics``, or the desktop app,
and ``dynamics`` must never import this module. Standard library only, so the
contract can be built and hashed without pulling in numerics or SPICE.

Deliberately NOT part of the contract (and therefore not of the fingerprint):
integrator method, rtol/atol, max step, output/measurement cadence, RNG seed,
run name, UI label, working directory, absolute paths, and output folders.
Those belong to a later numerical/run manifest, not to the force physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "FORCE_CONTRACT_SCHEMA_VERSION",
    "ForceContractError",
    "ForceElementStatus",
    "ConsumerReadiness",
    "ConsumerRole",
    "OrientationPolicy",
    "PointMassForceContract",
    "ThirdBodyForceContract",
    "LunarJ2ForceContract",
    "EarthJ2ForceContract",
    "LunarHarmonicsForceContract",
    "ForceModelContract",
    "lunar_j2_enabled",
    "capability_map",
    "to_canonical_payload",
    "canonical_json_bytes",
    "force_model_fingerprint",
    "sha256_hex_of_bytes",
]


FORCE_CONTRACT_SCHEMA_VERSION = "r0b.force-model-contract.v1"


class ForceContractError(ValueError):
    """Controlled failure while building, normalizing, or hashing a contract."""


class ForceElementStatus(str, Enum):
    """How far a single force element is qualified for official OD use."""

    SUPPORTED = "supported"
    EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY = "experimental_direct_trajectory_only"
    UNSUPPORTED_OFFICIAL_OD = "unsupported_official_od"


class ConsumerReadiness(str, Enum):
    """How far a consumer role is qualified for the contract's force set.

    There is deliberately no ``configured_but_not_consumed`` state: a force that
    is configured but never evaluated is a defect, not a reportable status.
    """

    VERIFIED = "verified"
    PENDING_R1 = "pending_r1"
    UNSUPPORTED = "unsupported"


class ConsumerRole(str, Enum):
    """Every place the force model is (or must be) evaluated."""

    TRUTH_STATE = "truth_state"
    ESTIMATOR_STATE = "estimator_state"
    ESTIMATOR_STM = "estimator_stm"
    UKF_STANDARD = "ukf_standard"
    UKF_SQUARE_ROOT = "ukf_square_root"
    UKF_FAST_SIGMA = "ukf_fast_sigma"
    POSTERIOR_COVARIANCE = "posterior_covariance"
    OBSERVABILITY = "observability"


class OrientationPolicy(str, Enum):
    """Body-orientation rule used to evaluate a body-fixed force term."""

    NOT_APPLICABLE = "not_applicable"
    # Constant IAU 2006 lunar mean pole (RA0 269.9949 deg, Dec0 66.5392 deg).
    CONSTANT_IAU2006_MOON_MEAN_POLE = "constant_iau2006_moon_mean_pole"
    # Identity J2000 -> Earth body-fixed: experimental direct-trajectory
    # approximation only; NOT an IERS-compliant Earth orientation.
    EXPERIMENTAL_IDENTITY_J2000_TO_EARTH_BODY_FIXED = (
        "experimental_identity_j2000_to_earth_body_fixed"
    )
    # Epoch-dependent lunar body-fixed rotation sampled on a cadence grid.
    SAMPLED_LUNAR_BODY_FIXED_ROTATION = "sampled_lunar_body_fixed_rotation"


def lunar_j2_enabled(coefficient: float) -> bool:
    """Single derivation rule for "is lunar J2 actually evaluated?".

    The propagator branches on the truthiness of ``j2_moon`` (``if j2_moon:``),
    so a zero coefficient means the term is not evaluated at all. Defined once
    here so config mapping, contracts, and tests cannot drift apart.
    """
    return bool(float(coefficient) != 0.0)


def capability_map(
    mapping: Mapping[ConsumerRole, ConsumerReadiness],
) -> Mapping[ConsumerRole, ConsumerReadiness]:
    """Return an immutable capability mapping covering every consumer role."""
    missing = [role for role in ConsumerRole if role not in mapping]
    if missing:
        raise ForceContractError(
            "consumer_capabilities must cover every ConsumerRole; missing: "
            + ", ".join(role.value for role in missing)
        )
    unknown = [key for key in mapping if not isinstance(key, ConsumerRole)]
    if unknown:
        raise ForceContractError(f"unknown consumer roles: {unknown!r}")
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class PointMassForceContract:
    """Central-body point-mass attraction."""

    enabled: bool
    body: str
    gravitational_parameter_m3_s2: float
    policy: str


@dataclass(frozen=True)
class ThirdBodyForceContract:
    """Third-body point-mass perturbation in the Moon-centered frame."""

    enabled: bool
    body: str
    gravitational_parameter_m3_s2: float
    policy: str


@dataclass(frozen=True)
class LunarJ2ForceContract:
    """Lunar oblateness (J2) term."""

    enabled: bool
    coefficient: float
    reference_radius_m: float
    orientation_policy: OrientationPolicy
    status: ForceElementStatus = ForceElementStatus.SUPPORTED


@dataclass(frozen=True)
class EarthJ2ForceContract:
    """Earth oblateness (J2) term.

    Unsupported on official OD paths: the current orientation is an identity
    J2000-to-Earth-body-fixed approximation (experimental direct-trajectory
    use only). R0A's fail-closed rejection is unchanged by this contract.
    """

    enabled: bool
    mode: str
    coefficient: float
    reference_radius_m: float
    orientation_policy: OrientationPolicy
    status: ForceElementStatus = ForceElementStatus.UNSUPPORTED_OFFICIAL_OD


@dataclass(frozen=True)
class LunarHarmonicsForceContract:
    """High-degree lunar spherical-harmonic gravity.

    Scientific identity comes from the coefficient CONTENT hash plus model
    metadata — never from an absolute path, and not from the file basename, so
    relocating or renaming the same coefficient file cannot change the
    fingerprint. Rotation cadence and margin are part of the force identity
    because they change the body transformation, hence the acceleration.
    """

    enabled: bool
    model_identity: str
    coefficient_file_sha256: str | None
    degree_nmax: int | None
    order_mmax: int | None
    normalization: str
    model_gravitational_parameter_m3_s2: float | None
    reference_radius_m: float | None
    body_frame: str
    kernel_profile_policy: str
    rotation_cadence_s: float | None
    rotation_margin_s: float | None
    status: ForceElementStatus = ForceElementStatus.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY


@dataclass(frozen=True)
class ForceModelContract:
    """Immutable, hashable-by-content description of one evaluated force model."""

    schema_version: str
    lunar_point_mass: PointMassForceContract
    earth_third_body: ThirdBodyForceContract
    sun_third_body: ThirdBodyForceContract
    lunar_j2: LunarJ2ForceContract
    earth_j2: EarthJ2ForceContract
    lunar_harmonics: LunarHarmonicsForceContract
    force_ephemeris_policy: str
    consumer_capabilities: Mapping[ConsumerRole, ConsumerReadiness]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_capabilities", capability_map(self.consumer_capabilities)
        )

    def to_canonical_payload(self) -> dict[str, Any]:
        return to_canonical_payload(self)

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def force_model_fingerprint(self) -> str:
        return force_model_fingerprint(self)


# ---------------------------------------------------------------------------
# Canonical normalization
# ---------------------------------------------------------------------------

def _normalize(value: Any) -> Any:
    """Recursively convert a contract value into canonical JSON primitives.

    One normalizer for every contract type. Rejects anything whose JSON form
    would be ambiguous or platform-dependent instead of silently coercing it.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return _normalize(value.value)
    # bool before int: bool is an int subclass.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ForceContractError("NaN is not a valid force-contract value.")
        if math.isinf(value):
            raise ForceContractError(
                "infinite values are not valid force-contract values "
                f"({'+inf' if value > 0 else '-inf'})."
            )
        # Collapse negative zero so -0.0 and 0.0 hash identically. Finite
        # subnormals are preserved exactly (Python float repr round-trips).
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            key_value = key.value if isinstance(key, Enum) else key
            if not isinstance(key_value, str):
                key_value = str(key_value)
            if key_value in normalized:
                raise ForceContractError(f"duplicate canonical mapping key: {key_value!r}")
            normalized[key_value] = _normalize(item)
        return dict(sorted(normalized.items()))
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    raise ForceContractError(
        f"unsupported force-contract value of type {type(value).__name__!r}; "
        "contracts must contain only dataclasses, enums, mappings, sequences, "
        "strings, finite numbers, booleans, or None."
    )


def to_canonical_payload(contract: ForceModelContract) -> dict[str, Any]:
    """Return the deterministic, JSON-ready payload for ``contract``."""
    payload = _normalize(contract)
    if not isinstance(payload, dict):
        raise ForceContractError("force contract must normalize to a JSON object.")
    return payload


def canonical_json_bytes(contract: ForceModelContract) -> bytes:
    """Return the canonical UTF-8 JSON encoding hashed by the fingerprint."""
    return json.dumps(
        to_canonical_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex_of_bytes(data: bytes) -> str:
    """Return ``sha256:<64-lowercase-hex>`` for ``data``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def force_model_fingerprint(contract: ForceModelContract) -> str:
    """Return the deterministic ``sha256:<hex>`` force fingerprint."""
    return sha256_hex_of_bytes(canonical_json_bytes(contract))
