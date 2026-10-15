"""Schema generations, and the semantic overlay for the Phase 14 document.

TWO SCHEMA GENERATIONS EXIST
----------------------------
``0`` — the Phase 14 artifact. It predates this package and carries no
``schema_version`` field. That absence is treated as a *named* generation, not
as a default: a document with no version is generation 0, a document with a
version is read at that version, and a version this loader does not know is
rejected. Nothing is assumed.

``1`` — the generation this package writes. It is generation 0 plus the
structure below: interval semantics, unknown reasons, sigma semantics and
source discrepancies promoted from prose into fields.

WHY AN OVERLAY RATHER THAN A REWRITE
------------------------------------
Phase 14 states all of this correctly, but in ``limitations`` prose that only a
human can act on. Phase 15 needs it in fields. Editing the Phase 14 artifact to
add them would rewrite a historical science artifact, so instead the facts are
declared here, applied at load time, and tested against the prose they came
from.

WHAT THE OVERLAY DELIBERATELY DOES NOT DO
-----------------------------------------
It does not re-label any Phase 14 parameter status. ``PUBLIC_RF_HARDWARE_SOURCED``
exists in the vocabulary for future configurations, and the Iris parameters
would arguably qualify, but retroactively re-classifying a frozen parameter
would change what Phase 14 claimed. Statuses are carried through unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from .errors import UnsupportedConfigurationSchemaError

__all__ = [
    "PHASE14_DISCREPANCIES",
    "PHASE14_INTERVALS",
    "PHASE14_SIGMA_SEMANTICS",
    "PHASE14_UNKNOWN_SEMANTICS",
    "SCHEMA_VERSION_CURRENT",
    "SCHEMA_VERSION_PHASE14_UNVERSIONED",
    "SUPPORTED_SCHEMA_VERSIONS",
    "detect_schema_version",
]

SCHEMA_VERSION_PHASE14_UNVERSIONED = 0
SCHEMA_VERSION_CURRENT = 1
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (0, 1)

#: Sections a configuration document must define.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "identity", "dynamics", "srp", "rf", "navigation",
)
#: Identity fields that must be present in every generation.
REQUIRED_IDENTITY: tuple[str, ...] = (
    "configuration_id", "configuration_class",
)


def detect_schema_version(document: Mapping[str, Any]) -> int:
    """Return the generation of ``document``, or reject it.

    A missing ``schema_version`` means generation 0 — the Phase 14 document —
    and that is the only case in which a version is inferred rather than read.
    Anything that declares a version must declare one this loader supports.
    """
    if "schema_version" not in document:
        return SCHEMA_VERSION_PHASE14_UNVERSIONED
    raw = document["schema_version"]
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise UnsupportedConfigurationSchemaError(
            f"schema_version must be an integer; got {raw!r}"
        )
    if raw not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedConfigurationSchemaError(
            f"schema_version {raw} is not supported by this loader "
            f"(supported: {SUPPORTED_SCHEMA_VERSIONS}). Refusing to interpret a "
            "configuration written against an unknown schema."
        )
    return raw


# ======================================================================
# Phase 14 semantic overlay
# ======================================================================

#: ``path -> (unknown_reason, blocks)`` for every UNKNOWN in the Phase 14
#: document. Each entry restates, as structure, what that parameter's own
#: ``limitations`` prose already says.
PHASE14_UNKNOWN_SEMANTICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dynamics.solar_array_area_m2": (
        "NOT_PUBLIC",
        ("an upper bound on effective projected SRP area",),
    ),
    "dynamics.effective_projected_area_m2": (
        "BLOCKED_BY_DEPENDENCY",
        ("a point area-to-mass ratio", "a point K_SRP"),
    ),
    "dynamics.C_R": (
        "NOT_PUBLIC",
        ("a point K_SRP",),
    ),
    "dynamics.attitude_mode": (
        "NOT_PUBLIC",
        ("any box-wing or panel SRP model", "a time-varying projected area"),
    ),
    "srp.K_SRP": (
        "BLOCKED_BY_DEPENDENCY",
        ("a point SRP magnitude", "an SRP solve-for prior centred on truth"),
    ),
    "rf.spacecraft_tx_antenna_gain_dbi": (
        "NOT_PUBLIC",
        ("spacecraft EIRP", "downlink C/N0", "any thermal Doppler sigma"),
    ),
    "rf.spacecraft_tx_internal_losses_db": (
        "NOT_PUBLIC",
        ("spacecraft EIRP",),
    ),
    "rf.spacecraft_EIRP_dbw": (
        "BLOCKED_BY_DEPENDENCY",
        ("downlink C/N0", "any mission thermal Doppler sigma"),
    ),
    "rf.spacecraft_rx_antenna_gain_dbi": (
        "NOT_PUBLIC",
        ("spacecraft G/T", "the uplink leg of the two-way budget"),
    ),
    "rf.spacecraft_g_over_t_db_per_k": (
        "BLOCKED_BY_DEPENDENCY",
        ("the uplink leg of the two-way budget",),
    ),
    "rf.pointing_assumption": (
        "NOT_PUBLIC",
        ("an antenna pointing-loss term",),
    ),
    "rf.polarization_assumption": (
        "NOT_PUBLIC",
        ("a polarization-mismatch loss term",),
    ),
    "navigation.cross_time_correlation_model": (
        "NOT_MODELLED",
        ("a colored-noise R in production",),
    ),
}

#: Interval definitions built from parameter pairs already in the document.
#:
#: The C_R entry and the K_SRP entry differ in kind, and that difference is the
#: scientific content of this table. C_R is bounded by radiation-pressure
#: physics. K_SRP is not bounded at all here: its area factor is body-only, so
#: the true value can lie above the upper endpoint once the solar arrays are
#: included. One is PHYSICAL_BOUND; the other is a SCREENING_ENVELOPE that must
#: name what it leaves out.
PHASE14_INTERVALS: dict[str, dict[str, Any]] = {
    "C_R_physical_bound": dict(
        lower_path="dynamics.C_R_lower_bound",
        upper_path="dynamics.C_R_upper_bound",
        semantics="PHYSICAL_BOUND",
        justification=(
            "C_R = 1 is total absorption and C_R = 2 is total specular "
            "back-reflection along the incident ray. A cannonball reflectivity "
            "coefficient cannot lie outside this range."
        ),
        excludes=(),
    ),
    "area_to_mass_body_only_screening_envelope": dict(
        lower_path="dynamics.area_to_mass_lower_bound_m2_per_kg",
        upper_path="dynamics.area_to_mass_upper_bound_m2_per_kg",
        semantics="SCREENING_ENVELOPE",
        justification=(
            "Built from the stowed rectangular envelope only, across the two "
            "disagreeing published masses. The true illuminated area is larger, "
            "so the true A/m can lie above the upper endpoint."
        ),
        excludes=("solar array area", "deployed appendages"),
    ),
    "K_SRP_body_only_screening_envelope": dict(
        lower_path="srp.K_SRP_lower_bound_m2_per_kg",
        upper_path="srp.K_SRP_upper_bound_m2_per_kg",
        semantics="SCREENING_ENVELOPE",
        justification=(
            "C_R physical bounds times the body-only A/m screening envelope. "
            "This is a screening range for materiality work, NOT a bound on the "
            "spacecraft: the solar-array contribution and any attitude "
            "dependence of projected area are excluded, so the true K_SRP may "
            "exceed the upper endpoint."
        ),
        excludes=(
            "solar array area",
            "attitude-dependent projected area",
            "deployed appendages",
        ),
    ),
}

#: Source disagreements that must survive loading without being averaged.
PHASE14_DISCREPANCIES: tuple[dict[str, Any], ...] = (
    dict(
        parameter="dynamics.mass_kg",
        canonical_path="dynamics.mass_kg",
        conflicting_path="dynamics.mass_alternative_kg",
        reason_for_canonical_selection=(
            "The NASA fact sheet is the mission-owner source; eoPortal is a "
            "secondary compilation. Both values are retained and both enter the "
            "area-to-mass screening envelope as endpoints. They are NOT averaged: "
            "the disagreement is a provenance conflict between two publications, "
            "not a measured uncertainty in the spacecraft."
        ),
    ),
)

#: Source records the Phase 14 document cites but never registered.
#:
#: Six identity parameters carry ``source_id = "PHASE14"``, which resolves to
#: nothing in that document's own register. Phase 14's internal check special-
#: cased the string rather than registering it, so the gap survived. Rather than
#: special-casing it again — which would let a genuinely dangling id through —
#: the missing record is declared here, explicitly, as part of the overlay.
PHASE14_SUPPLEMENTARY_SOURCES: tuple[dict[str, str], ...] = (
    dict(
        source_id="PHASE14",
        title="Phase 14 reference spacecraft, RF and navigation configuration "
              "freeze (this document)",
        organization="this project",
        reference="05_reports/reference_spacecraft_rf_and_navigation_"
                  "configuration_freeze.md",
        revision="2026-09-05",
        source_type="INTERNAL_FREEZE",
        notes="self-reference: the identity fields are owner-frozen by the "
              "phase that created this configuration. Registered by the Phase 15 "
              "overlay because Phase 14 cited the id without defining it.",
    ),
)

#: Measurement-sigma semantics. The frozen Doppler sigma is the ground
#: frequency-standard component alone; the thermal term remains blocked by an
#: unknown spacecraft EIRP, so this is not a total Doppler noise figure.
PHASE14_SIGMA_SEMANTICS: dict[str, str] = {
    "navigation.sigma_doppler_mps": "GROUND_FREQUENCY_STANDARD_COMPONENT",
    "navigation.sigma_range_m": "PARAMETRIC_PLACEHOLDER",
}
