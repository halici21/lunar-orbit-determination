"""The typed configuration domain model.

A configuration is not a dictionary of numbers. Every scientifically meaningful
quantity here carries its unit, the authority behind it, and — when it was
computed rather than sourced — the formula and the inputs that produced it. That
is what lets a result written months from now still answer "which spacecraft
was this, and how much of it was actually known?".

Everything is frozen. A loaded configuration is a record of what was frozen at a
point in time, and code that could mutate it in place would make the fingerprint
a statement about nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .errors import (
    InvalidDerivedParameterError,
    InvalidIntervalSemanticsError,
    SourceDiscrepancyError,
    UnknownParameterStatusError,
    UnknownParameterValueError,
)
from .vocabulary import (
    CONFIGURATION_CLASSES,
    INTERVAL_SEMANTICS,
    PARAMETER_STATUSES,
    RESERVED_STATUSES,
    UNKNOWN_REASONS,
)

__all__ = [
    "Derivation",
    "Interval",
    "ParameterValue",
    "ReferenceConfiguration",
    "SourceDiscrepancy",
    "SourceRecord",
]

#: Section names, in the order they are serialised for fingerprinting.
SECTION_ORDER: tuple[str, ...] = (
    "identity",
    "dynamics",
    "srp",
    "rf",
    "navigation",
)


@dataclass(frozen=True)
class Derivation:
    """How a computed parameter was obtained, and from what."""

    formula: str
    inputs: tuple[str, ...] = ()
    input_source_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.formula.strip():
            raise InvalidDerivedParameterError(
                "a derivation must carry a non-empty formula"
            )

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"formula": self.formula}
        if self.inputs:
            d["inputs"] = list(self.inputs)
        if self.input_source_ids:
            d["input_source_ids"] = list(self.input_source_ids)
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass(frozen=True)
class ParameterValue:
    """One scientifically significant quantity, with its provenance attached.

    An UNKNOWN parameter carries ``value=None`` and must say why it is unknown
    and what that blocks. Reading it through :meth:`require` raises; there is
    deliberately no ``value_or(default)`` accessor, because the whole point of
    this package is that a missing spacecraft property must not quietly become a
    number somewhere downstream.
    """

    value: Any
    unit: str
    status: str
    source_id: str = ""
    derivation: Derivation | None = None
    notes: str = ""
    unknown_reason: str = ""
    blocks: tuple[str, ...] = ()
    #: What the quantity actually covers, where a bare number would be
    #: ambiguous. A Doppler sigma is the motivating case: the same float means
    #: something different if it is the whole noise budget rather than the
    #: ground frequency-standard term alone.
    semantics: str = ""

    def __post_init__(self) -> None:
        if self.status not in PARAMETER_STATUSES:
            raise UnknownParameterStatusError(
                f"status {self.status!r} is not in the controlled vocabulary "
                f"{PARAMETER_STATUSES}"
            )
        if self.status in RESERVED_STATUSES:
            raise UnknownParameterStatusError(
                f"status {self.status!r} is reserved for real mission "
                "configurations and must not appear in a reference configuration"
            )
        if not self.unit.strip():
            raise UnknownParameterStatusError(
                f"a parameter must record its unit; {self.status} parameter has none"
            )

        if self.status == "UNKNOWN":
            if self.value is not None:
                raise UnknownParameterStatusError(
                    "an UNKNOWN parameter must carry value None, not "
                    f"{self.value!r} — encoding an unknown as a number is exactly "
                    "the failure this type exists to prevent"
                )
            if self.unknown_reason not in UNKNOWN_REASONS:
                raise UnknownParameterStatusError(
                    f"unknown_reason {self.unknown_reason!r} is not in "
                    f"{UNKNOWN_REASONS}"
                )
            if not self.blocks:
                raise UnknownParameterStatusError(
                    "an UNKNOWN parameter must state what it blocks"
                )
        else:
            if self.value is None:
                raise UnknownParameterStatusError(
                    f"a {self.status} parameter must carry a value"
                )
            if not self.source_id:
                raise UnknownParameterStatusError(
                    f"a {self.status} parameter must cite a source_id"
                )
            if self.unknown_reason:
                raise UnknownParameterStatusError(
                    "a known parameter must not carry an unknown_reason"
                )

        if self.status == "DERIVED_FROM_SOURCED_VALUES" and self.derivation is None:
            raise InvalidDerivedParameterError(
                "a derived parameter must carry its Derivation"
            )
        if self.derivation is not None and self.status not in (
            "DERIVED_FROM_SOURCED_VALUES",
            "PUBLIC_RF_HARDWARE_SOURCED",
            "PUBLIC_SPACECRAFT_SOURCED",
            "OWNER_FROZEN_REFERENCE",
        ):
            raise InvalidDerivedParameterError(
                f"a {self.status} parameter may not carry a Derivation"
            )

    @property
    def is_known(self) -> bool:
        return self.status != "UNKNOWN"

    @property
    def is_derived(self) -> bool:
        return self.status == "DERIVED_FROM_SOURCED_VALUES"

    def require(self, context: str = "") -> Any:
        """Return the value, or raise if it is UNKNOWN.

        ``context`` is folded into the message so the traceback names the
        calculation that wanted the value, not just the missing field.
        """
        if not self.is_known:
            where = f" (needed for {context})" if context else ""
            raise UnknownParameterValueError(
                f"parameter is UNKNOWN{where}: reason={self.unknown_reason}, "
                f"blocks={', '.join(self.blocks)}"
            )
        return self.value

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
        }
        if self.source_id:
            d["source_id"] = self.source_id
        if self.derivation is not None:
            d["derivation"] = self.derivation.as_dict()
        if self.notes:
            d["notes"] = self.notes
        if self.unknown_reason:
            d["unknown_reason"] = self.unknown_reason
        if self.blocks:
            d["blocks"] = list(self.blocks)
        if self.semantics:
            d["semantics"] = self.semantics
        return d


@dataclass(frozen=True)
class Interval:
    """Two endpoints plus an explicit statement of what kind of interval it is.

    The semantics field is not decoration. ``PHYSICAL_BOUND`` asserts the true
    value cannot lie outside; ``SCREENING_ENVELOPE`` asserts only that the range
    is useful for screening, and explicitly allows the truth to sit outside it.
    Nothing here can be promoted from one to the other by writing a different
    string, because ``justification`` is required and validation reads it.
    """

    lower: float
    upper: float
    unit: str
    semantics: str
    justification: str
    source_ids: tuple[str, ...] = ()
    derivation: Derivation | None = None
    excludes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.semantics not in INTERVAL_SEMANTICS:
            raise InvalidIntervalSemanticsError(
                f"interval semantics {self.semantics!r} is not in "
                f"{INTERVAL_SEMANTICS}"
            )
        if not (self.lower <= self.upper):
            raise InvalidIntervalSemanticsError(
                f"interval is not ordered: [{self.lower}, {self.upper}]"
            )
        if not self.unit.strip():
            raise InvalidIntervalSemanticsError("an interval must record its unit")
        if not self.justification.strip():
            raise InvalidIntervalSemanticsError(
                "an interval must justify its semantics"
            )
        if self.semantics == "SCREENING_ENVELOPE" and not self.excludes:
            raise InvalidIntervalSemanticsError(
                "a SCREENING_ENVELOPE must name what it excludes; that omission "
                "is precisely why it is not a bound"
            )
        if self.semantics == "PHYSICAL_BOUND" and self.excludes:
            raise InvalidIntervalSemanticsError(
                "a PHYSICAL_BOUND cannot exclude a contributor and still bound "
                "the quantity; declare it a SCREENING_ENVELOPE instead"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, x: float) -> bool:
        return self.lower <= x <= self.upper

    def position_of(self, x: float) -> float:
        """Where ``x`` sits in the interval, as a fraction from lower to upper."""
        if self.width == 0.0:
            return 0.0
        return (x - self.lower) / self.width

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "semantics": self.semantics,
            "justification": self.justification,
        }
        if self.source_ids:
            d["source_ids"] = list(self.source_ids)
        if self.derivation is not None:
            d["derivation"] = self.derivation.as_dict()
        if self.excludes:
            d["excludes"] = list(self.excludes)
        return d


@dataclass(frozen=True)
class SourceDiscrepancy:
    """Two sources that disagree, both kept.

    HOW AVERAGING IS PREVENTED
    --------------------------
    Not by arithmetic. Once a mean has been written into a document the inputs
    are gone, and no later check can recover them — comparing a stored canonical
    against the midpoint of itself and one survivor is vacuous, because that
    equality is impossible whenever the two differ.

    What is enforceable is structural, and that is what this type enforces: the
    canonical value must be one a source actually reported, so it carries that
    source's own status. A mean is not reported by anyone, so it can only enter
    as a ``DERIVED_FROM_SOURCED_VALUES`` parameter — and a derived canonical is
    refused here. Selecting between publications is a provenance judgement with
    a stated reason, never a calculation.
    """

    parameter: str
    unit: str
    canonical_value: float
    canonical_source_id: str
    conflicting_value: float
    conflicting_source_id: str
    reason_for_canonical_selection: str
    canonical_status: str = "PUBLIC_SPACECRAFT_SOURCED"

    def __post_init__(self) -> None:
        if self.canonical_value == self.conflicting_value:
            raise SourceDiscrepancyError(
                f"{self.parameter}: values agree; this is not a discrepancy"
            )
        if not self.reason_for_canonical_selection.strip():
            raise SourceDiscrepancyError(
                f"{self.parameter}: choosing a canonical value requires a stated "
                "reason"
            )
        if self.canonical_source_id == self.conflicting_source_id:
            raise SourceDiscrepancyError(
                f"{self.parameter}: a source cannot disagree with itself"
            )
        if self.canonical_status == "DERIVED_FROM_SOURCED_VALUES":
            raise SourceDiscrepancyError(
                f"{self.parameter}: the canonical value of a source disagreement "
                "is DERIVED, meaning it was computed from the disagreeing "
                "sources rather than reported by one of them. That is what "
                "averaging looks like, and it is refused: pick a source and say "
                "why."
            )
        if self.canonical_status == "UNKNOWN":
            raise SourceDiscrepancyError(
                f"{self.parameter}: a canonical value cannot be UNKNOWN"
            )

    @property
    def midpoint(self) -> float:
        """The average of the two reported values — reported, never used.

        Present so a summary can show what was declined. Nothing in the loader
        consumes it, and no parameter in a valid configuration equals it unless
        a source actually reported that number.
        """
        return 0.5 * (self.canonical_value + self.conflicting_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "unit": self.unit,
            "canonical_value": self.canonical_value,
            "canonical_source_id": self.canonical_source_id,
            "conflicting_value": self.conflicting_value,
            "conflicting_source_id": self.conflicting_source_id,
            "reason_for_canonical_selection": self.reason_for_canonical_selection,
            "canonical_status": self.canonical_status,
        }


@dataclass(frozen=True)
class SourceRecord:
    """One entry in the source register."""

    source_id: str
    title: str
    organization: str
    reference: str = ""
    revision: str = ""
    source_type: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("source_id", "title", "organization"):
            if not str(getattr(self, name)).strip():
                raise SourceDiscrepancyError(
                    f"source record is missing {name}"
                )

    def as_dict(self) -> dict[str, Any]:
        d = {"source_id": self.source_id, "title": self.title,
             "organization": self.organization}
        for name in ("reference", "revision", "source_type", "notes"):
            v = getattr(self, name)
            if v:
                d[name] = v
        return d


@dataclass(frozen=True)
class ReferenceConfiguration:
    """A loaded, validated spacecraft / RF / navigation reference configuration.

    Sections are read-only mappings of parameter name to :class:`ParameterValue`.
    ``intervals`` holds quantities that were frozen as a range rather than a
    point, each declaring what kind of range it is.
    """

    schema_version: int
    configuration_id: str
    configuration_class: str
    sections: Mapping[str, Mapping[str, ParameterValue]]
    intervals: Mapping[str, Interval]
    discrepancies: tuple[SourceDiscrepancy, ...]
    sources: Mapping[str, SourceRecord]
    limitations: Mapping[str, str]
    origin: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.configuration_class not in CONFIGURATION_CLASSES:
            raise UnknownParameterStatusError(
                f"configuration_class {self.configuration_class!r} is not in "
                f"{CONFIGURATION_CLASSES}"
            )
        object.__setattr__(self, "sections", MappingProxyType({
            name: MappingProxyType(dict(params))
            for name, params in self.sections.items()
        }))
        object.__setattr__(self, "intervals",
                           MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "limitations",
                           MappingProxyType(dict(self.limitations)))
        object.__setattr__(self, "origin", MappingProxyType(dict(self.origin)))

    # -- access ---------------------------------------------------------
    def parameter(self, path: str) -> ParameterValue:
        """Look up ``"section.name"``."""
        try:
            section, name = path.split(".", 1)
        except ValueError:
            raise KeyError(
                f"parameter path must be 'section.name'; got {path!r}"
            ) from None
        try:
            return self.sections[section][name]
        except KeyError:
            raise KeyError(f"no parameter {path!r} in this configuration") from None

    def require(self, path: str, context: str = "") -> Any:
        """Value of ``path``, raising if it is UNKNOWN."""
        return self.parameter(path).require(context or path)

    def is_known(self, path: str) -> bool:
        return self.parameter(path).is_known

    def interval(self, name: str) -> Interval:
        try:
            return self.intervals[name]
        except KeyError:
            raise KeyError(f"no interval {name!r} in this configuration") from None

    def discrepancy(self, parameter: str) -> SourceDiscrepancy | None:
        for d in self.discrepancies:
            if d.parameter == parameter:
                return d
        return None

    # -- inventory ------------------------------------------------------
    def walk(self) -> Iterator[tuple[str, ParameterValue]]:
        for section in SECTION_ORDER:
            for name, p in self.sections.get(section, {}).items():
                yield f"{section}.{name}", p

    def unknown_parameters(self) -> tuple[str, ...]:
        return tuple(path for path, p in self.walk() if not p.is_known)

    def status_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in PARAMETER_STATUSES}
        for _, p in self.walk():
            counts[p.status] += 1
        return counts

    @property
    def parameter_count(self) -> int:
        return sum(1 for _ in self.walk())
