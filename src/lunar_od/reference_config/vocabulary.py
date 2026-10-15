"""Controlled vocabularies for reference-configuration semantics.

These are tuples rather than enums to match the convention already established
by ``lunar_od.rf.types.SCENARIO_CLASSES``, and so that a serialised
configuration round-trips as plain strings without an encoder.

The distinctions drawn here are the scientific content of this package. A number
in a configuration file means very little on its own; what it is, where it came
from, and what kind of interval it belongs to is the part that stops a screening
value from being quoted later as a measured spacecraft property.
"""

from __future__ import annotations

__all__ = [
    "CONFIGURATION_CLASSES",
    "INTERVAL_SEMANTICS",
    "KNOWN_STATUSES",
    "PARAMETER_STATUSES",
    "RESERVED_STATUSES",
    "SIGMA_SEMANTICS",
    "UNKNOWN_REASONS",
]

#: Where a parameter's authority comes from.
#:
#: ``PUBLIC_RF_HARDWARE_SOURCED`` is separated from ``PUBLIC_SPACECRAFT_SOURCED``
#: because a radio is a catalogue part: the Iris V2.1 specification describes
#: every unit built, while a spacecraft fact sheet describes one vehicle. Mixing
#: them would hide which claims travel with the hardware and which do not.
PARAMETER_STATUSES: tuple[str, ...] = (
    "MISSION_SOURCED",
    "PUBLIC_SPACECRAFT_SOURCED",
    "PUBLIC_BUS_SOURCED",
    "PUBLIC_RF_HARDWARE_SOURCED",
    "DERIVED_FROM_SOURCED_VALUES",
    "OWNER_FROZEN_REFERENCE",
    "PARAMETRIC",
    "UNKNOWN",
)

#: Statuses a non-mission configuration may not claim. Enforced at construction,
#: so a later edit cannot quietly promote a public reference to flight truth.
RESERVED_STATUSES: tuple[str, ...] = ("MISSION_SOURCED",)

#: Every status that asserts a value exists.
KNOWN_STATUSES: tuple[str, ...] = tuple(
    s for s in PARAMETER_STATUSES if s != "UNKNOWN"
)

#: What kind of interval a pair of endpoints actually is.
#:
#: The separation that matters here is PHYSICAL_BOUND versus SCREENING_ENVELOPE.
#: C_R in [1, 2] is a bound: no cannonball reflectivity can leave it. The K_SRP
#: interval built from a body-only projected area is not a bound at all — the
#: solar arrays are excluded, so the true value can sit outside it. Both are
#: intervals; only one constrains reality.
INTERVAL_SEMANTICS: tuple[str, ...] = (
    "PHYSICAL_BOUND",
    "SCREENING_ENVELOPE",
    "SOURCE_DISCREPANCY_RANGE",
    "PARAMETRIC_SWEEP_RANGE",
    "STATISTICAL_CONFIDENCE_INTERVAL",
)

#: Why a value is absent. "Unknown" is not one thing, and collapsing these into a
#: single null loses the difference between "nobody published it" and "this
#: cannot be computed until something else is published".
UNKNOWN_REASONS: tuple[str, ...] = (
    "NOT_PUBLIC",
    "NOT_MEASURED",
    "NOT_APPLICABLE",
    "NOT_REQUIRED",
    "NOT_MODELLED",
    "BLOCKED_BY_DEPENDENCY",
)

#: What a measurement sigma actually covers.
#:
#: The project's frozen Doppler sigma is the ground frequency-standard component
#: alone. Labelling it TOTAL_DOPPLER_NOISE would assert that the thermal and
#: media terms are included, which they are not — the thermal term is still
#: blocked by an unknown spacecraft EIRP.
SIGMA_SEMANTICS: tuple[str, ...] = (
    "TOTAL_DOPPLER_NOISE",
    "GROUND_FREQUENCY_STANDARD_COMPONENT",
    "THERMAL_COMPONENT",
    "WHITE_COMPONENT_ONLY",
    "PARAMETRIC_PLACEHOLDER",
)

#: What a whole configuration claims to be.
CONFIGURATION_CLASSES: tuple[str, ...] = (
    "MISSION_CONFIGURATION",
    "PUBLIC_SPACECRAFT_REFERENCE",
    "PUBLIC_BUS_REFERENCE",
    "THESIS_REFERENCE_SPACECRAFT",
)
