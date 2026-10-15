"""Reference spacecraft / RF / navigation configuration infrastructure.

This package is OPT-IN and inert until called. Importing it reads no files,
registers no configurations, and changes no behaviour anywhere else in
``lunar_od``. There is no default spacecraft: ``DEFAULT_REFERENCE_CONFIGURATION``
is ``None``, the registry starts empty, and asking for a configuration nobody
registered raises rather than producing the only one that happens to exist.

It exists because Phase 10A and Phase 13 stopped at the same missing object —
nobody had said which spacecraft was being navigated — and Phase 14 froze one.
A frozen configuration in a JSON file is still only a document; this package
makes it a validated software object that a result can cite.

    from lunar_od.reference_config import (
        load_reference_configuration, configuration_snapshot,
    )

    config = load_reference_configuration(path_to_document)
    snapshot = configuration_snapshot(config)     # goes in the result metadata

    config.require("rf.uplink_frequency_hz")      # 7.2e9
    config.require("rf.spacecraft_EIRP_dbw")      # raises: UNKNOWN

The last line is the design. A spacecraft property nobody published stays
unknown all the way through, and reaching for it fails at the call site instead
of yielding a zero that looks like an answer.
"""

from __future__ import annotations

from .errors import (
    ConfigurationFingerprintMismatchError,
    InconsistentFrequencyPlanError,
    InvalidDerivedParameterError,
    InvalidIntervalSemanticsError,
    MissingSourceReferenceError,
    ReferenceConfigurationError,
    SourceDiscrepancyError,
    UnknownParameterStatusError,
    UnknownParameterValueError,
    UnsupportedConfigurationSchemaError,
)
from .fingerprint import (
    EXCLUDED_FROM_FINGERPRINT,
    canonical_json,
    fingerprint,
    legacy_phase14_fingerprint,
    scientific_projection,
    section_fingerprints,
    verify_fingerprint,
)
from .loader import (
    load_reference_configuration,
    parse_reference_configuration,
    to_document,
)
from .model import (
    Derivation,
    Interval,
    ParameterValue,
    ReferenceConfiguration,
    SourceDiscrepancy,
    SourceRecord,
)
from .registry import (
    DEFAULT_REFERENCE_CONFIGURATION,
    ReferenceConfigurationRegistry,
    RegistryEntry,
    clear_registry,
    get_reference_configuration,
    register_configuration_directory,
    register_configuration_file,
    registered_ids,
)
from .reporting import configuration_snapshot, summarize_reference_configuration
from .schema import (
    SCHEMA_VERSION_CURRENT,
    SCHEMA_VERSION_PHASE14_UNVERSIONED,
    SUPPORTED_SCHEMA_VERSIONS,
    detect_schema_version,
)
from .validation import CheckResult, run_checks, validate_configuration
from .vocabulary import (
    CONFIGURATION_CLASSES,
    INTERVAL_SEMANTICS,
    PARAMETER_STATUSES,
    SIGMA_SEMANTICS,
    UNKNOWN_REASONS,
)

__all__ = [
    "CONFIGURATION_CLASSES",
    "DEFAULT_REFERENCE_CONFIGURATION",
    "EXCLUDED_FROM_FINGERPRINT",
    "INTERVAL_SEMANTICS",
    "PARAMETER_STATUSES",
    "SCHEMA_VERSION_CURRENT",
    "SCHEMA_VERSION_PHASE14_UNVERSIONED",
    "SIGMA_SEMANTICS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "UNKNOWN_REASONS",
    "CheckResult",
    "ConfigurationFingerprintMismatchError",
    "Derivation",
    "InconsistentFrequencyPlanError",
    "Interval",
    "InvalidDerivedParameterError",
    "InvalidIntervalSemanticsError",
    "MissingSourceReferenceError",
    "ParameterValue",
    "ReferenceConfiguration",
    "ReferenceConfigurationError",
    "ReferenceConfigurationRegistry",
    "RegistryEntry",
    "SourceDiscrepancy",
    "SourceDiscrepancyError",
    "SourceRecord",
    "UnknownParameterStatusError",
    "UnknownParameterValueError",
    "UnsupportedConfigurationSchemaError",
    "canonical_json",
    "clear_registry",
    "configuration_snapshot",
    "detect_schema_version",
    "fingerprint",
    "get_reference_configuration",
    "legacy_phase14_fingerprint",
    "load_reference_configuration",
    "parse_reference_configuration",
    "register_configuration_directory",
    "register_configuration_file",
    "registered_ids",
    "run_checks",
    "scientific_projection",
    "section_fingerprints",
    "summarize_reference_configuration",
    "to_document",
    "validate_configuration",
    "verify_fingerprint",
]
