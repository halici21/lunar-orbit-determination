"""Deterministic scientific fingerprinting of a reference configuration.

WHAT THE FINGERPRINT IS FOR
---------------------------
So that a result written later can name the configuration that produced it, and
so that two runs claiming the same configuration can be shown to mean it.

WHAT ENTERS IT
--------------
Identity, every parameter with its value, unit, status, source and derivation,
every interval with its semantics, every recorded source disagreement, the
source register, the scoping limitations, and the schema generation.

Limitations are included deliberately. They are prose, and prose churns — but a
configuration whose limitations stopped excluding the solar arrays would be
making a different scientific claim with identical numbers, and that must not
hash the same.

WHAT NEVER ENTERS IT
--------------------
Anything that varies between two runs of the same science: wall-clock
timestamps, runtime durations, absolute filesystem paths, hostnames, process
ids. Phase 14 found exactly this contamination in earlier artifacts, where a
stored dense/diagonal timing ratio moved from 4.79x to 35.55x between two runs
of identical code. A fingerprint that moved with it would be worthless.

The exclusion is structural rather than a filter: the projection is built by
naming what goes in, so a volatile field added to a document later cannot leak
into the hash by default.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .errors import ConfigurationFingerprintMismatchError
from .model import SECTION_ORDER, ReferenceConfiguration

__all__ = [
    "EXCLUDED_FROM_FINGERPRINT",
    "canonical_json",
    "fingerprint",
    "legacy_phase14_fingerprint",
    "scientific_projection",
    "section_fingerprints",
    "verify_fingerprint",
]

#: Documented, for the report and for tests. These names never reach the hash,
#: because the projection below does not read them.
EXCLUDED_FROM_FINGERPRINT: tuple[str, ...] = (
    "generated_utc",
    "frozen_utc",
    "loaded_utc",
    "runtime_seconds",
    "elapsed_ms",
    "wall_clock",
    "hostname",
    "process_id",
    "source_path",
    "absolute_path",
)


def scientific_projection(config: ReferenceConfiguration) -> dict[str, Any]:
    """The part of a configuration that determines its science.

    Built by explicit construction, not by deleting volatile keys from a dump.
    """
    return {
        "schema_version": config.schema_version,
        "configuration_id": config.configuration_id,
        "configuration_class": config.configuration_class,
        "sections": {
            name: {
                pname: p.as_dict()
                for pname, p in sorted(config.sections.get(name, {}).items())
            }
            for name in SECTION_ORDER
            if name in config.sections
        },
        "intervals": {
            name: iv.as_dict() for name, iv in sorted(config.intervals.items())
        },
        "discrepancies": [
            d.as_dict()
            for d in sorted(config.discrepancies, key=lambda d: d.parameter)
        ],
        "sources": {
            sid: s.as_dict() for sid, s in sorted(config.sources.items())
        },
        "limitations": dict(sorted(config.limitations.items())),
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Formatting-invariant serialisation: sorted keys, no incidental spacing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def fingerprint(config: ReferenceConfiguration) -> str:
    """SHA-256 over the canonical scientific projection."""
    return hashlib.sha256(
        canonical_json(scientific_projection(config)).encode("ascii")
    ).hexdigest()


def section_fingerprints(config: ReferenceConfiguration) -> dict[str, str]:
    """Per-section digests, so a campaign can say which half it depended on."""
    projection = scientific_projection(config)
    out: dict[str, str] = {}
    for name, body in projection["sections"].items():
        out[name] = hashlib.sha256(
            canonical_json(body).encode("ascii")
        ).hexdigest()
    return out


def verify_fingerprint(config: ReferenceConfiguration, expected: str) -> str:
    """Recompute and compare, raising on mismatch. Returns the actual value."""
    actual = fingerprint(config)
    if actual != expected:
        raise ConfigurationFingerprintMismatchError(
            f"configuration {config.configuration_id!r} fingerprints as {actual}, "
            f"but {expected} was expected"
        )
    return actual


def legacy_phase14_fingerprint(document: Mapping[str, Any]) -> str:
    """Reproduce the Phase 14 fingerprint from a raw Phase 14 document.

    Phase 14 hashed the canonical JSON of its five parameter sections and
    nothing else. Recomputing it here lets the loader prove that the file on
    disk is the document Phase 14 actually froze, before any Phase 15 semantics
    are layered on top.
    """
    payload = {
        name: document[name] for name in SECTION_ORDER if name in document
    }
    return hashlib.sha256(
        canonical_json(payload).encode("ascii")
    ).hexdigest()
