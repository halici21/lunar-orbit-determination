"""Read-only views: a campaign snapshot, and a human summary.

``configuration_snapshot`` is the piece future campaigns need. Dropping it into
a result file lets that result answer, years later, which spacecraft
configuration produced it — by id, by scientific fingerprint, and by schema
generation, with no ambiguity about which revision was meant.

The snapshot deliberately carries no timestamp and no path. Those belong to the
run, not to the configuration, and Phase 14 found that mixing the two is how a
results file ends up with a number that changes between identical runs.
"""

from __future__ import annotations

from typing import Any

from .fingerprint import fingerprint, section_fingerprints
from .model import ReferenceConfiguration

__all__ = ["configuration_snapshot", "summarize_reference_configuration"]


def configuration_snapshot(config: ReferenceConfiguration) -> dict[str, Any]:
    """Metadata a campaign should record alongside its results."""
    return {
        "REFERENCE_CONFIGURATION_ID": config.configuration_id,
        "REFERENCE_CONFIGURATION_CLASS": config.configuration_class,
        "REFERENCE_CONFIGURATION_FINGERPRINT": fingerprint(config),
        "CONFIGURATION_SCHEMA_VERSION": config.schema_version,
        "REFERENCE_CONFIGURATION_SECTION_FINGERPRINTS": section_fingerprints(config),
        "REFERENCE_CONFIGURATION_UNKNOWN_COUNT": len(config.unknown_parameters()),
    }


def summarize_reference_configuration(config: ReferenceConfiguration) -> str:
    """A deterministic plain-text view, grouped by what is actually known."""
    lines: list[str] = []
    add = lines.append

    add(f"configuration_id    : {config.configuration_id}")
    add(f"configuration_class : {config.configuration_class}")
    add(f"schema_version      : {config.schema_version}")
    add(f"fingerprint         : {fingerprint(config)}")
    add(f"parameters          : {config.parameter_count}")

    counts = config.status_counts()
    for status, n in sorted(counts.items()):
        if n:
            add(f"    {status:32s} {n:3d}")

    add("")
    add("UNKNOWN — carried through, never defaulted:")
    for path in config.unknown_parameters():
        p = config.parameter(path)
        add(f"    {path:46s} {p.unknown_reason:22s} blocks: "
            f"{'; '.join(p.blocks)}")

    if config.intervals:
        add("")
        add("INTERVALS — semantics decide what may be concluded:")
        for name, iv in sorted(config.intervals.items()):
            add(f"    {name}")
            add(f"        [{iv.lower:.6g}, {iv.upper:.6g}] {iv.unit}  "
                f"{iv.semantics}")
            if iv.excludes:
                add(f"        excludes: {', '.join(iv.excludes)}")

    if config.discrepancies:
        add("")
        add("SOURCE DISAGREEMENTS — both values kept, never averaged:")
        for d in config.discrepancies:
            add(f"    {d.parameter}")
            add(f"        canonical   {d.canonical_value} {d.unit}  "
                f"({d.canonical_source_id})")
            add(f"        conflicting {d.conflicting_value} {d.unit}  "
                f"({d.conflicting_source_id})")
            add(f"        NOT used    {d.midpoint} {d.unit}  (the average)")

    derived = [(path, p) for path, p in config.walk() if p.is_derived]
    if derived:
        add("")
        add("DERIVED — each with the formula that produced it:")
        for path, p in derived:
            formula = p.derivation.formula if p.derivation else "?"
            add(f"    {path:46s} = {formula}")

    return "\n".join(lines)
