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
    "consumer_capabilities_for",
    "ForceExecutionSpec",
    "bind_spec_to_runtime",
    "to_canonical_payload",
    "canonical_json_bytes",
    "force_model_fingerprint",
    "sha256_hex_of_bytes",
    "ESTIMATOR_CONSUMER_ROLES",
    "ForceModelParityError",
    "ForceModelMismatchPolicy",
    "ForceModelParityDecision",
    "evaluate_force_model_parity",
    "force_model_manifest",
    "manifest_canonical_bytes",
    "manifest_sha256",
    "scenario_result_force_fields",
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

    ``EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY`` marks a role (in practice only
    ``truth_state``) that a force can drive as an experimental direct-trajectory
    computation but that is not qualified for official OD — distinct from
    ``UNSUPPORTED`` (cannot be driven at all on that role) (R0B-V06/V10).
    """

    VERIFIED = "verified"
    PENDING_R1 = "pending_r1"
    EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY = "experimental_direct_trajectory_only"
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
            # Only str keys, or enums whose .value is a str. Anything else
            # (object/int/float/tuple keys) is rejected rather than coerced
            # through str(), which could emit an address-bearing repr and make
            # the fingerprint process- or run-dependent (R0B-V07).
            if isinstance(key, Enum):
                key_value = key.value
            else:
                key_value = key
            if not isinstance(key_value, str):
                raise ForceContractError(
                    "canonical mapping keys must be strings (or enums whose "
                    f"value is a string); got key of type {type(key).__name__!r}."
                )
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


# ---------------------------------------------------------------------------
# Executable force specification (R0B-F1)
# ---------------------------------------------------------------------------

def consumer_capabilities_for(
    *, lunar_j2_on: bool, earth_j2_on: bool, harmonics_on: bool
) -> Mapping[ConsumerRole, ConsumerReadiness]:
    """Capability matrix for a force set. Worst-status-first precedence.

    R1 closes SCI-003 for lunar-J2 posterior covariance and observability.
    High-degree harmonics remains qualified only as an experimental DIRECT
    truth trajectory (``truth_state`` -> experimental_direct_trajectory_only),
    and Earth J2 remains unsupported for official OD consumers.
    """
    if harmonics_on:
        statuses = {role: ConsumerReadiness.UNSUPPORTED for role in ConsumerRole}
        statuses[ConsumerRole.TRUTH_STATE] = (
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY
        )
        return capability_map(statuses)
    if earth_j2_on:
        return capability_map(
            {role: ConsumerReadiness.UNSUPPORTED for role in ConsumerRole}
        )
    if lunar_j2_on:
        statuses = {role: ConsumerReadiness.VERIFIED for role in ConsumerRole}
        statuses[ConsumerRole.POSTERIOR_COVARIANCE] = ConsumerReadiness.VERIFIED
        statuses[ConsumerRole.OBSERVABILITY] = ConsumerReadiness.VERIFIED
        return capability_map(statuses)
    return capability_map({role: ConsumerReadiness.VERIFIED for role in ConsumerRole})


@dataclass(frozen=True)
class ForceExecutionSpec:
    """Single source of one executed force model: runtime args AND contract.

    Built once from a scenario config plus the effective (fixture) GM values,
    then everything downstream is derived from THIS object — the primitives
    handed to the propagator (``mu_*``, ``j2_moon``), the ForceModelContract,
    its fingerprint, and the capability decision — so a run can never report
    physics it did not execute (R0B-V02). Truth and estimator each carry their
    own spec; a declared mismatch campaign simply uses two different specs
    (R0B-V05). Constant-derived values (reference radii, Earth-J2 coefficient)
    are passed in as data to keep this module standard-library only.
    """

    j2_moon: float
    mu_moon_m3_s2: float
    mu_earth_m3_s2: float
    mu_sun_m3_s2: float
    enable_earth_j2: bool
    earth_j2_mode: str
    lunar_harmonics: LunarHarmonicsForceContract
    lunar_j2_reference_radius_m: float
    earth_j2_coefficient: float
    earth_j2_reference_radius_m: float
    force_ephemeris_policy: str = "moon_centered_sampled_ephemeris"

    @property
    def lunar_j2_on(self) -> bool:
        return lunar_j2_enabled(self.j2_moon)

    @property
    def harmonics_on(self) -> bool:
        return bool(self.lunar_harmonics.enabled)

    def propagation_kwargs(self) -> dict[str, float]:
        """Force primitives that must reach the propagator for this spec."""
        return {
            "mu_moon_m3_s2": float(self.mu_moon_m3_s2),
            "mu_earth_m3_s2": float(self.mu_earth_m3_s2),
            "mu_sun_m3_s2": float(self.mu_sun_m3_s2),
            "j2_moon": float(self.j2_moon),
        }

    def contract(self) -> ForceModelContract:
        lunar_j2_on = self.lunar_j2_on
        earth_j2_on = bool(self.enable_earth_j2)
        return ForceModelContract(
            schema_version=FORCE_CONTRACT_SCHEMA_VERSION,
            lunar_point_mass=PointMassForceContract(
                enabled=True,
                body="moon",
                gravitational_parameter_m3_s2=float(self.mu_moon_m3_s2),
                policy="central_body_point_mass",
            ),
            earth_third_body=ThirdBodyForceContract(
                enabled=bool(self.mu_earth_m3_s2),
                body="earth",
                gravitational_parameter_m3_s2=float(self.mu_earth_m3_s2),
                policy="moon_centered_third_body_point_mass",
            ),
            sun_third_body=ThirdBodyForceContract(
                enabled=bool(self.mu_sun_m3_s2),
                body="sun",
                gravitational_parameter_m3_s2=float(self.mu_sun_m3_s2),
                policy="moon_centered_third_body_point_mass",
            ),
            lunar_j2=LunarJ2ForceContract(
                enabled=lunar_j2_on,
                coefficient=float(self.j2_moon),
                reference_radius_m=(
                    float(self.lunar_j2_reference_radius_m) if lunar_j2_on else 0.0
                ),
                orientation_policy=(
                    OrientationPolicy.CONSTANT_IAU2006_MOON_MEAN_POLE
                    if lunar_j2_on
                    else OrientationPolicy.NOT_APPLICABLE
                ),
                status=ForceElementStatus.SUPPORTED,
            ),
            earth_j2=EarthJ2ForceContract(
                enabled=earth_j2_on,
                mode=str(self.earth_j2_mode),
                coefficient=float(self.earth_j2_coefficient) if earth_j2_on else 0.0,
                reference_radius_m=(
                    float(self.earth_j2_reference_radius_m) if earth_j2_on else 0.0
                ),
                orientation_policy=(
                    OrientationPolicy.EXPERIMENTAL_IDENTITY_J2000_TO_EARTH_BODY_FIXED
                    if earth_j2_on
                    else OrientationPolicy.NOT_APPLICABLE
                ),
                status=ForceElementStatus.UNSUPPORTED_OFFICIAL_OD,
            ),
            lunar_harmonics=self.lunar_harmonics,
            force_ephemeris_policy=str(self.force_ephemeris_policy),
            consumer_capabilities=consumer_capabilities_for(
                lunar_j2_on=lunar_j2_on,
                earth_j2_on=earth_j2_on,
                harmonics_on=self.harmonics_on,
            ),
        )

    def fingerprint(self) -> str:
        return force_model_fingerprint(self.contract())


def bind_spec_to_runtime(
    spec: ForceExecutionSpec,
    runtime_kwargs: Mapping[str, float],
    *,
    context: str,
) -> None:
    """Assert the spec's force primitives exactly equal the runtime values.

    Bit-level float equality between the fingerprinted force values and the
    values actually handed to the propagator (R0B-V01). Raises a controlled
    binding error on any difference.
    """
    expected = spec.propagation_kwargs()
    for key, spec_value in expected.items():
        runtime_value = float(runtime_kwargs[key])
        # exact float equality (bit-level), NOT approximate
        if runtime_value != spec_value or (
            runtime_value == 0.0
            and math.copysign(1.0, runtime_value) != math.copysign(1.0, spec_value)
        ):
            raise ForceContractError(
                f"{context}: force primitive {key!r} bound to the propagator "
                f"({runtime_value!r}) does not exactly equal the fingerprinted "
                f"spec value ({spec_value!r})."
            )


# ---------------------------------------------------------------------------
# Parity enforcement (R0B-2)
# ---------------------------------------------------------------------------

#: Roles that decide whether an ESTIMATOR can consume a force model. The truth
#: side is governed by the per-element status instead, so a high-fidelity truth
#: trajectory can still drive an explicitly declared mismatch campaign.
ESTIMATOR_CONSUMER_ROLES = (
    ConsumerRole.ESTIMATOR_STATE,
    ConsumerRole.ESTIMATOR_STM,
    ConsumerRole.UKF_STANDARD,
    ConsumerRole.UKF_SQUARE_ROOT,
    ConsumerRole.UKF_FAST_SIGMA,
)


class ForceModelParityError(ForceContractError):
    """Truth/estimator force models are not an allowed combination."""


@dataclass(frozen=True)
class ForceModelMismatchPolicy:
    """Run-level permission to evaluate different truth/estimator physics.

    Deliberately NOT part of ``ForceModelContract``: the contract describes the
    physics, this describes what a run is allowed to do with it. Opt-in never
    bypasses an unsupported capability.
    """

    enabled: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not (self.reason or "").strip():
            raise ForceModelParityError(
                "allow_explicit_force_model_mismatch=True requires a non-empty "
                "force_model_mismatch_reason describing the planned campaign."
            )
        if not self.enabled and self.reason is not None:
            object.__setattr__(self, "reason", self.reason or None)


@dataclass(frozen=True)
class ForceModelParityDecision:
    """Outcome of the pre-execution truth/estimator force comparison.

    The manifest is stored as immutable canonical bytes and the SHA is computed
    from exactly those bytes. ``manifest`` re-parses a fresh copy on every
    access, so no nested mutation of a returned payload can ever leave the
    stored SHA stale (R0B-V08).
    """

    schema_version: str
    truth_fingerprint: str
    estimator_fingerprint: str
    match: bool
    policy: ForceModelMismatchPolicy
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def explicit_mismatch(self) -> bool:
        return bool(self.policy.enabled and not self.match)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a freshly parsed copy of the manifest (never a shared dict)."""
        return json.loads(self.manifest_bytes.decode("utf-8"))


def _unsupported_estimator_roles(contract: ForceModelContract) -> list[ConsumerRole]:
    return [
        role
        for role in ESTIMATOR_CONSUMER_ROLES
        if contract.consumer_capabilities[role] is ConsumerReadiness.UNSUPPORTED
    ]


def evaluate_force_model_parity(
    truth_contract: ForceModelContract,
    estimator_contract: ForceModelContract,
    policy: ForceModelMismatchPolicy | None = None,
    *,
    context: str = "force-model parity preflight",
) -> ForceModelParityDecision:
    """Compare truth/estimator force models before anything is propagated.

    Order of gates (both fail closed):

    1. capability -- the estimator contract must not rely on a force whose
       estimator-side roles are ``unsupported``. Mismatch opt-in cannot bypass
       this, so high-degree harmonics can never enter the estimator.
    2. parity -- differing fingerprints require an explicit opt-in plus a
       non-empty reason.
    """
    policy = policy or ForceModelMismatchPolicy()
    unsupported = _unsupported_estimator_roles(estimator_contract)
    if unsupported:
        raise ForceModelParityError(
            f"{context}: the estimator force model is unsupported for role(s) "
            + ", ".join(role.value for role in unsupported)
            + ". This capability gate cannot be bypassed by "
            "allow_explicit_force_model_mismatch."
        )
    truth_fingerprint = force_model_fingerprint(truth_contract)
    estimator_fingerprint = force_model_fingerprint(estimator_contract)
    match = truth_fingerprint == estimator_fingerprint
    if not match and not policy.enabled:
        raise ForceModelParityError(
            f"{context}: truth force fingerprint {truth_fingerprint} does not match "
            f"estimator force fingerprint {estimator_fingerprint}. Set "
            "allow_explicit_force_model_mismatch=True with a "
            "force_model_mismatch_reason to run this as a declared "
            "force-model mismatch campaign."
        )
    manifest = force_model_manifest(
        truth_contract, estimator_contract, policy, match=match
    )
    manifest_bytes = manifest_canonical_bytes(manifest)
    return ForceModelParityDecision(
        schema_version=FORCE_CONTRACT_SCHEMA_VERSION,
        truth_fingerprint=truth_fingerprint,
        estimator_fingerprint=estimator_fingerprint,
        match=match,
        policy=policy,
        manifest_bytes=manifest_bytes,
        manifest_sha256=sha256_hex_of_bytes(manifest_bytes),
    )


def _contract_manifest_section(contract: ForceModelContract) -> dict[str, Any]:
    payload = to_canonical_payload(contract)
    return {
        "fingerprint": force_model_fingerprint(contract),
        "canonical_payload": payload,
        "consumer_capabilities": payload["consumer_capabilities"],
    }


def force_model_manifest(
    truth_contract: ForceModelContract,
    estimator_contract: ForceModelContract,
    policy: ForceModelMismatchPolicy | None = None,
    *,
    match: bool | None = None,
) -> dict[str, Any]:
    """Build the deterministic, path-free force-model manifest payload."""
    policy = policy or ForceModelMismatchPolicy()
    truth_section = _contract_manifest_section(truth_contract)
    estimator_section = _contract_manifest_section(estimator_contract)
    if match is None:
        match = truth_section["fingerprint"] == estimator_section["fingerprint"]
    return {
        "schema_version": FORCE_CONTRACT_SCHEMA_VERSION,
        "truth": truth_section,
        "estimator": estimator_section,
        "match": bool(match),
        "mismatch_policy": {
            "enabled": bool(policy.enabled),
            "reason": policy.reason if policy.enabled else None,
        },
    }


def manifest_canonical_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return the canonical UTF-8 JSON bytes of a force-model manifest."""
    return json.dumps(
        _normalize(dict(manifest)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` of the canonical manifest bytes."""
    return sha256_hex_of_bytes(manifest_canonical_bytes(manifest))


def scenario_result_force_fields(decision: ForceModelParityDecision) -> dict[str, Any]:
    """Map a parity decision onto the append-only ScenarioResult fields.

    Posterior/observability statuses are read from the ESTIMATOR contract:
    R1 qualifies both roles for nonzero lunar J2 while Earth J2 and harmonics
    retain their fail-closed capability states.
    """
    capabilities = decision.manifest["estimator"]["consumer_capabilities"]
    return {
        "force_contract_schema_version": decision.schema_version,
        "truth_force_fingerprint": decision.truth_fingerprint,
        "estimator_force_fingerprint": decision.estimator_fingerprint,
        "force_model_match": bool(decision.match),
        "explicit_force_model_mismatch": bool(decision.explicit_mismatch),
        "force_model_mismatch_reason": decision.policy.reason or "",
        "force_contract_manifest_sha256": decision.manifest_sha256,
        "posterior_force_role_status": capabilities[
            ConsumerRole.POSTERIOR_COVARIANCE.value
        ],
        "observability_force_role_status": capabilities[ConsumerRole.OBSERVABILITY.value],
    }
