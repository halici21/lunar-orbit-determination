"""Reading a reference-configuration document into the typed domain model.

The loader is deliberately strict. It reads what the document says, applies the
declared semantic overlay for that schema generation, and validates the result.
It never fills a gap: a parameter the document leaves unknown stays unknown all
the way through, and a document that fails a semantic check raises rather than
loading in a degraded state.

Nothing here is loaded implicitly. There is no module-level default
configuration, and importing this module reads no files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import (
    MissingSourceReferenceError,
    ReferenceConfigurationError,
    UnsupportedConfigurationSchemaError,
)
from .fingerprint import legacy_phase14_fingerprint
from .model import (
    SECTION_ORDER,
    Derivation,
    Interval,
    ParameterValue,
    ReferenceConfiguration,
    SourceDiscrepancy,
    SourceRecord,
)
from .schema import (
    PHASE14_DISCREPANCIES,
    PHASE14_INTERVALS,
    PHASE14_SIGMA_SEMANTICS,
    PHASE14_SUPPLEMENTARY_SOURCES,
    PHASE14_UNKNOWN_SEMANTICS,
    REQUIRED_IDENTITY,
    REQUIRED_SECTIONS,
    SCHEMA_VERSION_CURRENT,
    SCHEMA_VERSION_PHASE14_UNVERSIONED,
    detect_schema_version,
)
from .validation import validate_configuration
from .vocabulary import SIGMA_SEMANTICS

__all__ = [
    "load_reference_configuration",
    "parse_reference_configuration",
    "to_document",
]


def _identity_value(document: Mapping[str, Any], name: str) -> str:
    entry = document.get("identity", {}).get(name)
    if entry is None:
        raise ReferenceConfigurationError(
            f"configuration is missing required identity field {name!r}"
        )
    value = entry.get("value") if isinstance(entry, Mapping) else entry
    if value is None or not str(value).strip():
        raise ReferenceConfigurationError(
            f"identity field {name!r} carries no value"
        )
    return str(value)


def _parameter_from_entry(
    path: str, entry: Mapping[str, Any], schema_version: int
) -> ParameterValue:
    """Build one ParameterValue from a raw document entry.

    Generation 0 (the Phase 14 document) states its unknown reasons and
    consequences in prose, so they come from the schema overlay. Generation 1
    carries them as fields, so the document speaks for itself and no overlay is
    consulted. Either way an unknown that cannot say what it blocks is refused.

    Phase 14 spells the unit key ``units`` and puts its caveats in
    ``limitations``; both are carried across without loss.
    """
    if not isinstance(entry, Mapping):
        raise ReferenceConfigurationError(
            f"{path}: expected an object with value/units/status, got "
            f"{type(entry).__name__} — a bare number carries no provenance"
        )
    for required in ("units", "status"):
        if required not in entry and required.rstrip("s") not in entry:
            raise ReferenceConfigurationError(
                f"{path}: entry is missing required field {required!r}"
            )

    unit = str(entry.get("units", entry.get("unit", "")))
    status = str(entry["status"])
    # Generation 0 writes a derivation as a bare formula string; generation 1
    # writes the full record with its inputs. Both are read.
    raw_derivation = entry.get("derivation")
    if isinstance(raw_derivation, str) and raw_derivation.strip():
        derivation = Derivation(formula=raw_derivation)
    elif isinstance(raw_derivation, Mapping):
        derivation = Derivation(
            formula=str(raw_derivation.get("formula", "")),
            inputs=tuple(str(i) for i in raw_derivation.get("inputs", ())),
            input_source_ids=tuple(
                str(i) for i in raw_derivation.get("input_source_ids", ())
            ),
            notes=str(raw_derivation.get("notes", "")),
        )
    else:
        derivation = None
    unknown_reason = ""
    blocks: tuple[str, ...] = ()
    if status == "UNKNOWN":
        if schema_version == SCHEMA_VERSION_PHASE14_UNVERSIONED:
            semantics = PHASE14_UNKNOWN_SEMANTICS.get(path)
            if semantics is None:
                raise ReferenceConfigurationError(
                    f"{path}: parameter is UNKNOWN but the generation-0 overlay "
                    "declares no reason or consequence for it. An unknown with "
                    "no stated consequence is not loadable."
                )
            unknown_reason, blocks = semantics
        else:
            unknown_reason = str(entry.get("unknown_reason", ""))
            blocks = tuple(str(b) for b in entry.get("blocks", ()))
            if not unknown_reason or not blocks:
                raise ReferenceConfigurationError(
                    f"{path}: an UNKNOWN parameter must carry unknown_reason and "
                    "blocks. An unknown with no stated consequence is not loadable."
                )

    if schema_version == SCHEMA_VERSION_PHASE14_UNVERSIONED:
        semantics_label = PHASE14_SIGMA_SEMANTICS.get(path, "")
    else:
        semantics_label = str(entry.get("semantics", ""))

    return ParameterValue(
        value=entry.get("value"),
        unit=unit,
        status=status,
        source_id=str(entry.get("source_id", "")),
        derivation=derivation,
        notes=str(entry.get("limitations", entry.get("notes", ""))),
        unknown_reason=unknown_reason,
        blocks=blocks,
        semantics=semantics_label,
    )


def _intervals_from_document(document: Mapping[str, Any]) -> dict[str, Interval]:
    """Generation 1: intervals are fields, so read them as written."""
    out: dict[str, Interval] = {}
    for name, raw in document.get("intervals", {}).items():
        d = raw.get("derivation")
        out[name] = Interval(
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            unit=str(raw["unit"]),
            semantics=str(raw["semantics"]),
            justification=str(raw["justification"]),
            source_ids=tuple(str(s) for s in raw.get("source_ids", ())),
            derivation=(
                Derivation(
                    formula=str(d["formula"]),
                    inputs=tuple(str(i) for i in d.get("inputs", ())),
                    input_source_ids=tuple(
                        str(i) for i in d.get("input_source_ids", ())
                    ),
                    notes=str(d.get("notes", "")),
                )
                if isinstance(d, Mapping) else None
            ),
            excludes=tuple(str(e) for e in raw.get("excludes", ())),
        )
    return out


def _discrepancies_from_document(
    document: Mapping[str, Any],
) -> tuple[SourceDiscrepancy, ...]:
    return tuple(
        SourceDiscrepancy(
            parameter=str(raw["parameter"]),
            unit=str(raw["unit"]),
            canonical_value=float(raw["canonical_value"]),
            canonical_source_id=str(raw["canonical_source_id"]),
            conflicting_value=float(raw["conflicting_value"]),
            conflicting_source_id=str(raw["conflicting_source_id"]),
            reason_for_canonical_selection=str(
                raw["reason_for_canonical_selection"]
            ),
            canonical_status=str(
                raw.get("canonical_status", "PUBLIC_SPACECRAFT_SOURCED")
            ),
        )
        for raw in document.get("discrepancies", [])
    )


def _build_intervals(
    sections: Mapping[str, Mapping[str, ParameterValue]],
) -> dict[str, Interval]:
    def lookup(path: str) -> ParameterValue | None:
        section, name = path.split(".", 1)
        return sections.get(section, {}).get(name)

    intervals: dict[str, Interval] = {}
    for name, spec in PHASE14_INTERVALS.items():
        lo_p, hi_p = lookup(spec["lower_path"]), lookup(spec["upper_path"])
        # A generation-0 document that does not carry these endpoints simply has
        # no such interval; that is not an error, and must not surface as a raw
        # KeyError from inside the overlay.
        if lo_p is None or hi_p is None:
            continue
        if not (lo_p.is_known and hi_p.is_known):
            continue
        source_ids = tuple(
            dict.fromkeys(s for s in (lo_p.source_id, hi_p.source_id) if s)
        )
        intervals[name] = Interval(
            lower=float(lo_p.value),
            upper=float(hi_p.value),
            unit=lo_p.unit,
            semantics=spec["semantics"],
            justification=spec["justification"],
            source_ids=source_ids,
            derivation=Derivation(
                formula=f"[{spec['lower_path']}, {spec['upper_path']}]",
                inputs=(spec["lower_path"], spec["upper_path"]),
                input_source_ids=source_ids,
            ),
            excludes=tuple(spec["excludes"]),
        )
    return intervals


def _build_discrepancies(
    sections: Mapping[str, Mapping[str, ParameterValue]],
) -> tuple[SourceDiscrepancy, ...]:
    def lookup(path: str) -> ParameterValue | None:
        section, name = path.split(".", 1)
        return sections.get(section, {}).get(name)

    out: list[SourceDiscrepancy] = []
    for spec in PHASE14_DISCREPANCIES:
        canonical = lookup(spec["canonical_path"])
        conflicting = lookup(spec["conflicting_path"])
        if canonical is None or conflicting is None:
            continue
        if not (canonical.is_known and conflicting.is_known):
            continue
        out.append(SourceDiscrepancy(
            parameter=spec["parameter"],
            unit=canonical.unit,
            canonical_value=float(canonical.value),
            canonical_source_id=canonical.source_id,
            canonical_status=canonical.status,
            conflicting_value=float(conflicting.value),
            conflicting_source_id=conflicting.source_id,
            reason_for_canonical_selection=spec["reason_for_canonical_selection"],
        ))
    return tuple(out)


def _build_sources(document: Mapping[str, Any]) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    for raw in document.get("provenance", {}).get("sources", []):
        sid = str(raw.get("source_id", "")).strip()
        if not sid:
            raise MissingSourceReferenceError(
                "source register contains an entry with no source_id"
            )
        if sid in records:
            raise MissingSourceReferenceError(
                f"source register defines {sid!r} more than once"
            )
        records[sid] = SourceRecord(
            source_id=sid,
            title=str(raw.get("title", "")),
            organization=str(raw.get("organization", "")),
            reference=str(raw.get("url", raw.get("reference", ""))),
            revision=str(raw.get("revision", "")),
            source_type=str(raw.get("source_type", "")),
            notes=str(raw.get("confidence", raw.get("notes", ""))),
        )
    for raw in PHASE14_SUPPLEMENTARY_SOURCES:
        if raw["source_id"] not in records:
            records[raw["source_id"]] = SourceRecord(**raw)
    return records


def parse_reference_configuration(
    document: Mapping[str, Any],
    *,
    origin: Mapping[str, str] | None = None,
    expected_ancestor_fingerprint: str | None = None,
) -> ReferenceConfiguration:
    """Build and validate a configuration from an already-parsed document."""
    if not isinstance(document, Mapping):
        raise ReferenceConfigurationError(
            "configuration document must be a JSON object"
        )

    schema_version = detect_schema_version(document)

    missing = [s for s in REQUIRED_SECTIONS if s not in document]
    if missing:
        raise ReferenceConfigurationError(
            f"configuration is missing required section(s): {', '.join(missing)}"
        )
    for name in REQUIRED_IDENTITY:
        _identity_value(document, name)

    if schema_version == SCHEMA_VERSION_PHASE14_UNVERSIONED:
        actual_ancestor = legacy_phase14_fingerprint(document)
        if (expected_ancestor_fingerprint is not None
                and actual_ancestor != expected_ancestor_fingerprint):
            raise ReferenceConfigurationError(
                f"document fingerprints as {actual_ancestor} under the Phase 14 "
                f"algorithm, but {expected_ancestor_fingerprint} was expected; "
                "the file on disk is not the document that was frozen"
            )
    else:
        actual_ancestor = ""

    sections: dict[str, dict[str, ParameterValue]] = {}
    for section in SECTION_ORDER:
        body = document.get(section)
        if body is None:
            continue
        if not isinstance(body, Mapping):
            raise ReferenceConfigurationError(
                f"section {section!r} must be an object"
            )
        sections[section] = {
            name: _parameter_from_entry(f"{section}.{name}", entry, schema_version)
            for name, entry in body.items()
        }

    legacy = schema_version == SCHEMA_VERSION_PHASE14_UNVERSIONED
    intervals = (
        _build_intervals(sections) if legacy
        else _intervals_from_document(document)
    )
    discrepancies = (
        _build_discrepancies(sections) if legacy
        else _discrepancies_from_document(document)
    )

    for path, semantics in PHASE14_SIGMA_SEMANTICS.items():
        if semantics not in SIGMA_SEMANTICS:
            raise UnsupportedConfigurationSchemaError(
                f"overlay declares sigma semantics {semantics!r} for {path}, "
                f"which is not in the controlled vocabulary"
            )

    limitations = {
        str(k): str(v) for k, v in document.get("limitations", {}).items()
    }
    origin_map = dict(origin or {})
    if actual_ancestor:
        origin_map["phase14_fingerprint"] = actual_ancestor

    config = ReferenceConfiguration(
        schema_version=schema_version,
        configuration_id=_identity_value(document, "configuration_id"),
        configuration_class=_identity_value(document, "configuration_class"),
        sections=sections,
        intervals=intervals,
        discrepancies=discrepancies,
        sources=_build_sources(document),
        limitations=limitations,
        origin=origin_map,
    )
    validate_configuration(config)
    return config


def load_reference_configuration(
    path: str | Path,
    *,
    expected_ancestor_fingerprint: str | None = None,
) -> ReferenceConfiguration:
    """Load, validate and return the configuration stored at ``path``.

    ``expected_ancestor_fingerprint``, when supplied, asserts that the file is
    the exact document a previous phase froze, using that phase's own hashing
    algorithm. Supplying it turns a silent substitution into a hard failure.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReferenceConfigurationError(
            f"cannot read configuration at {p}: {exc}"
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReferenceConfigurationError(
            f"configuration at {p} is not valid JSON: {exc}"
        ) from exc

    # The file name is recorded for humans, never for the fingerprint.
    return parse_reference_configuration(
        document,
        origin={"file_name": p.name},
        expected_ancestor_fingerprint=expected_ancestor_fingerprint,
    )


def to_document(config: ReferenceConfiguration) -> dict[str, Any]:
    """Serialise a configuration as a self-describing generation-1 document.

    Generation 1 carries the semantics that generation 0 kept in prose, so a
    document written here needs no overlay to be read back. The round trip is
    fingerprint-preserving: nothing scientific is added, dropped or reworded.

    Nothing about the run is written — no timestamp, no path, no host. Those
    belong to a result file, not to the configuration.
    """
    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION_CURRENT}
    for section in SECTION_ORDER:
        if section in config.sections:
            document[section] = {
                name: p.as_dict()
                for name, p in config.sections[section].items()
            }
    if config.intervals:
        document["intervals"] = {
            name: iv.as_dict() for name, iv in config.intervals.items()
        }
    if config.discrepancies:
        document["discrepancies"] = [d.as_dict() for d in config.discrepancies]
    document["provenance"] = {
        "sources": [s.as_dict() for s in config.sources.values()]
    }
    document["limitations"] = dict(config.limitations)
    return document
