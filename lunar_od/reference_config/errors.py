"""Failures raised while loading or validating a reference configuration.

Every failure here is scientific rather than syntactic: a configuration that
parses as JSON can still assert a frequency plan that does not close, cite a
source that does not exist, or claim a derived value its own inputs do not
support. Those are rejected, not warned about.

The hierarchy follows the project convention set by ``ForceContractError`` and
``HistoryDomainError``: a domain base subclassing ``ValueError``, with specific
subclasses so a caller can distinguish a schema problem from a science problem.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationFingerprintMismatchError",
    "InconsistentFrequencyPlanError",
    "InvalidDerivedParameterError",
    "InvalidIntervalSemanticsError",
    "MissingSourceReferenceError",
    "ReferenceConfigurationError",
    "SourceDiscrepancyError",
    "UnknownParameterStatusError",
    "UnknownParameterValueError",
    "UnsupportedConfigurationSchemaError",
]


class ReferenceConfigurationError(ValueError):
    """Base for every reference-configuration failure."""


class UnsupportedConfigurationSchemaError(ReferenceConfigurationError):
    """The document declares a schema generation this loader cannot interpret.

    Raised rather than guessing. A future schema may reorganise or re-mean
    fields, and silently reading it under today's assumptions would produce a
    configuration that looks valid and is not.
    """


class UnknownParameterStatusError(ReferenceConfigurationError):
    """A parameter carries a status outside the controlled vocabulary."""


class UnknownParameterValueError(ReferenceConfigurationError):
    """A caller demanded the value of a parameter that is UNKNOWN.

    This is the type raised by :meth:`ParameterValue.require`. It exists so that
    reaching for an unknown spacecraft property fails loudly at the call site
    instead of quietly yielding a zero, a NaN or an engineering guess.
    """


class InvalidDerivedParameterError(ReferenceConfigurationError):
    """A derived parameter disagrees with its own formula, or lacks provenance."""


class MissingSourceReferenceError(ReferenceConfigurationError):
    """A parameter cites a source id that the register does not define."""


class InconsistentFrequencyPlanError(ReferenceConfigurationError):
    """An asserted coherent frequency relationship does not hold numerically."""


class InvalidIntervalSemanticsError(ReferenceConfigurationError):
    """An interval is malformed, or claims semantics its evidence cannot support.

    The motivating case: an interval derived from a body-only projected area is
    a screening envelope, not a bound on the whole spacecraft. Promoting one to
    the other is a scientific claim, so the type system refuses to do it by
    accident.
    """


class SourceDiscrepancyError(ReferenceConfigurationError):
    """A disagreement between sources is malformed — most often, averaged away."""


class ConfigurationFingerprintMismatchError(ReferenceConfigurationError):
    """A recomputed scientific fingerprint does not match the expected one."""
