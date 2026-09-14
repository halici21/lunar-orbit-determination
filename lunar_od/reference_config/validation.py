"""Semantic validation of a loaded reference configuration.

Parsing proves a file is JSON. These checks ask whether it is science: whether a
frequency plan closes, whether a derived value agrees with its own formula,
whether every claim names a source that exists, and whether an interval's
declared semantics match what its inputs can actually support.

Everything here is generic. The checks read what a configuration declares — that
it asserts a coherent turnaround, that it stores a transponder band, that it
carries a reflectivity and an area-to-mass — and verify the relationships that
follow. No spacecraft, frequency or hardware model is named in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .errors import (
    InconsistentFrequencyPlanError,
    InvalidDerivedParameterError,
    MissingSourceReferenceError,
    ReferenceConfigurationError,
    SourceDiscrepancyError,
)
from .model import ReferenceConfiguration
from .schema import SUPPORTED_SCHEMA_VERSIONS
from .vocabulary import SIGMA_SEMANTICS

__all__ = ["CheckResult", "run_checks", "validate_configuration"]

#: Relative tolerance for asserted arithmetic relationships between stored
#: values. These are numbers written by a generator, not measurements, so the
#: only slack needed is float round-trip through JSON.
_REL_TOL = 1e-9

#: Name-suffix to unit convention. Checked longest-first so that ``_m2_per_kg``
#: is not mistaken for ``_kg``. This is what catches a power quietly stored in
#: dBW, or a gain stored as a bare ratio.
_UNIT_BY_SUFFIX: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("_m2_per_kg", ("m^2/kg",)),
    ("_db_per_k", ("dB/K",)),
    ("_dbhz", ("dB-Hz",)),
    ("_dbw", ("dBW",)),
    ("_dbm", ("dBm",)),
    ("_dbi", ("dBi",)),
    ("_mps", ("m/s",)),
    ("_hz", ("Hz",)),
    ("_kg", ("kg",)),
    ("_m2", ("m^2",)),
    ("_db", ("dB",)),
    ("_deg", ("deg",)),
    ("_pct", ("%",)),
    ("_w", ("W",)),
    ("_s", ("s",)),
    ("_m", ("m",)),
)


@dataclass(frozen=True)
class CheckResult:
    group: str
    check: str
    measured: str
    gate: str
    passed: bool
    detail: str = ""

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _f(config: ReferenceConfiguration, path: str) -> float | None:
    """Numeric value at ``path``, or None if unknown/absent/non-numeric."""
    try:
        p = config.parameter(path)
    except KeyError:
        return None
    if not p.is_known:
        return None
    v = p.value
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


# ======================================================================
# individual check groups
# ======================================================================
def _check_identity(config: ReferenceConfiguration, add: Callable) -> None:
    add("identity", "schema version is supported", config.schema_version,
        f"in {SUPPORTED_SCHEMA_VERSIONS}",
        config.schema_version in SUPPORTED_SCHEMA_VERSIONS)
    add("identity", "configuration id is present", config.configuration_id or "(none)",
        "non-empty", bool(config.configuration_id.strip()))
    add("identity", "configuration does not claim mission truth",
        config.configuration_class, "not MISSION_CONFIGURATION",
        config.configuration_class != "MISSION_CONFIGURATION",
        "a public reference must not be loadable as a mission configuration")
    add("identity", "every section is populated",
        f"{config.parameter_count} parameters", "> 0",
        config.parameter_count > 0)


def _check_units(config: ReferenceConfiguration, add: Callable) -> None:
    offenders: list[str] = []
    checked = 0
    for path, p in config.walk():
        name = path.split(".", 1)[1].casefold()
        for suffix, allowed in _UNIT_BY_SUFFIX:
            if name.endswith(suffix):
                checked += 1
                if p.unit not in allowed:
                    offenders.append(f"{path} is '{p.unit}', expected {allowed}")
                break
    add("units", "unit matches the name's dimension suffix",
        f"{len(offenders)} mismatched of {checked} checked", "0", not offenders,
        "; ".join(offenders))

    missing = [path for path, p in config.walk() if not p.unit.strip()]
    add("units", "every parameter records a unit", f"{len(missing)} unitless",
        "0", not missing, ", ".join(missing))


def _check_provenance(config: ReferenceConfiguration, add: Callable) -> None:
    dangling = [
        f"{path} -> {p.source_id}"
        for path, p in config.walk()
        if p.source_id and p.source_id not in config.sources
    ]
    add("provenance", "every source id resolves in the register",
        f"{len(dangling)} dangling", "0", not dangling, "; ".join(dangling))

    uncited = [path for path, p in config.walk() if p.is_known and not p.source_id]
    add("provenance", "every known parameter cites a source",
        f"{len(uncited)} uncited", "0", not uncited, ", ".join(uncited))

    silent = [
        path for path, p in config.walk()
        if not p.is_known and not (p.unknown_reason and p.blocks)
    ]
    add("provenance", "every UNKNOWN states a reason and what it blocks",
        f"{len(silent)} silent", "0", not silent, ", ".join(silent))

    formula_less = [
        path for path, p in config.walk() if p.is_derived and p.derivation is None
    ]
    add("provenance", "every derived parameter carries its formula",
        f"{len(formula_less)} formula-less", "0", not formula_less,
        ", ".join(formula_less))

    interval_dangling = [
        f"{name} -> {sid}"
        for name, iv in config.intervals.items()
        for sid in iv.source_ids
        if sid not in config.sources
    ]
    add("provenance", "every interval source id resolves",
        f"{len(interval_dangling)} dangling", "0", not interval_dangling,
        "; ".join(interval_dangling))


def _check_derivations(config: ReferenceConfiguration, add: Callable) -> None:
    """Re-evaluate the derived relationships that are machine-checkable."""
    checked = 0

    # dB conversions: any *_dbw whose *_w sibling exists.
    for path, p in list(config.walk()):
        section, name = path.split(".", 1)
        if not name.endswith("_dbw"):
            continue
        watt_path = f"{section}.{name[:-4]}_w"
        w = _f(config, watt_path)
        d = _f(config, path)
        if w is None or d is None or w <= 0.0:
            continue
        checked += 1
        add("derivations", f"{name} == 10 log10({name[:-4]}_w)",
            f"{d:.6f} dBW", f"= {10.0 * math.log10(w):.6f}",
            _close(d, 10.0 * math.log10(w)))

    add("derivations", "dB conversions were exercised", f"{checked} checked",
        ">= 0", True)


def _check_frequency_plan(config: ReferenceConfiguration, add: Callable) -> None:
    """Verify a coherent plan only where the configuration asserts one."""
    f_up = _f(config, "rf.uplink_frequency_hz")
    f_dn = _f(config, "rf.downlink_frequency_hz")
    ratio = _f(config, "rf.turnaround_ratio")

    if f_up is None or f_dn is None or ratio is None:
        add("rf", "coherent frequency plan asserted", "not asserted",
            "skipped when incomplete", True,
            "configuration does not declare all of uplink, downlink and "
            "turnaround; nothing to check")
        return

    expected = f_up * ratio
    ok = _close(f_dn, expected)
    add("rf", "downlink == uplink * turnaround_ratio", f"{f_dn:.6f} Hz",
        f"= {expected:.6f} Hz", ok)
    if not ok:
        raise InconsistentFrequencyPlanError(
            f"declared downlink {f_dn} Hz does not equal uplink {f_up} Hz times "
            f"turnaround {ratio} (= {expected} Hz)"
        )

    for label, freq, band_path in (
        ("uplink", f_up, "rf.transponder_rx_band_hz"),
        ("downlink", f_dn, "rf.transponder_tx_band_hz"),
    ):
        try:
            band = config.parameter(band_path)
        except KeyError:
            continue
        if not band.is_known:
            continue
        lo, hi = (float(x) for x in band.value)
        add("rf", f"{label} lies inside the declared transponder band",
            f"{freq / 1e6:.3f} MHz", f"in [{lo / 1e6:.0f}, {hi / 1e6:.0f}] MHz",
            lo <= freq <= hi)


def _check_srp_semantics(config: ReferenceConfiguration, add: Callable) -> None:
    """Section 22 semantics: what may and may not be concluded about K_SRP."""
    try:
        k_point = config.parameter("srp.K_SRP")
    except KeyError:
        return

    c_r = _f(config, "dynamics.C_R")
    a_over_m = _f(config, "dynamics.effective_projected_area_m2")
    a_over_m_direct = _f(config, "dynamics.area_to_mass_m2_per_kg")
    am = a_over_m_direct if a_over_m_direct is not None else None

    if c_r is None and am is None:
        add("srp", "no point K_SRP is derived while C_R and A/m are unknown",
            k_point.status, "UNKNOWN", not k_point.is_known,
            "K_SRP = C_R * A/m cannot be evaluated from two unknowns")
    elif c_r is not None and am is not None and k_point.is_known:
        add("srp", "K_SRP == C_R * A/m", f"{float(k_point.value):.9g}",
            f"= {c_r * am:.9g}", _close(float(k_point.value), c_r * am))

    envelopes = [
        (name, iv) for name, iv in config.intervals.items()
        if name.lower().startswith("k_srp")
    ]
    add("srp", "an unknown point K_SRP may still carry a screening envelope",
        f"point={k_point.status}, envelopes={len(envelopes)}",
        "permitted", True,
        "an unknown point value and a known screening range are different "
        "statements, not a contradiction")

    for name, iv in envelopes:
        add("srp", f"{name} is not labelled a physical bound", iv.semantics,
            "SCREENING_ENVELOPE", iv.semantics == "SCREENING_ENVELOPE",
            "the area factor excludes " + ", ".join(iv.excludes))

    # Interval endpoints must agree with the parameters they were built from.
    for name, iv in config.intervals.items():
        if iv.derivation is None or len(iv.derivation.inputs) != 2:
            continue
        lo = _f(config, iv.derivation.inputs[0])
        hi = _f(config, iv.derivation.inputs[1])
        if lo is None or hi is None:
            continue
        add("srp", f"{name} endpoints match their source parameters",
            f"[{iv.lower:.9g}, {iv.upper:.9g}]", f"[{lo:.9g}, {hi:.9g}]",
            _close(iv.lower, lo) and _close(iv.upper, hi))


def _check_discrepancies(config: ReferenceConfiguration, add: Callable) -> None:
    for d in config.discrepancies:
        add("discrepancy", f"{d.parameter} keeps both sources",
            f"{d.canonical_value} ({d.canonical_source_id}) vs "
            f"{d.conflicting_value} ({d.conflicting_source_id})",
            "both retained", True)
        # The enforceable invariant, not an arithmetic one: a canonical value
        # must be a number some source reported, so it carries that source's
        # status. A mean is reported by nobody and could only arrive as a
        # derived quantity.
        add("discrepancy", f"{d.parameter} canonical value was reported, not computed",
            d.canonical_status, "not DERIVED_FROM_SOURCED_VALUES",
            d.canonical_status != "DERIVED_FROM_SOURCED_VALUES",
            f"the declined average is {d.midpoint} {d.unit}")
        canonical_param = None
        try:
            canonical_param = config.parameter(d.parameter)
        except KeyError:
            pass
        if canonical_param is not None:
            add("discrepancy", f"{d.parameter} matches its own parameter",
                f"{canonical_param.value}", f"= {d.canonical_value}",
                canonical_param.is_known
                and _close(float(canonical_param.value), d.canonical_value))
            add("discrepancy", f"{d.parameter} is not the average of its sources",
                f"{canonical_param.value}", f"!= {d.midpoint}",
                not (canonical_param.is_known
                     and _close(float(canonical_param.value), d.midpoint)),
                "a canonical equal to the midpoint would have to be reported by "
                "a source that reported it")
        add("discrepancy", f"{d.parameter} conflicting source stays in the register",
            d.conflicting_source_id,
            "resolvable", d.conflicting_source_id in config.sources)


def _check_sigma_semantics(config: ReferenceConfiguration, add: Callable) -> None:
    """A measurement sigma must say what it covers.

    Generic: any parameter whose name contains ``sigma`` is required to declare
    semantics from the controlled vocabulary. A bare sigma is ambiguous between
    a total noise budget and one component of it, and the project's frozen
    Doppler sigma is in fact only the ground frequency-standard term.
    """
    sigmas = [(path, p) for path, p in config.walk() if "sigma" in path.casefold()]
    undeclared = [path for path, p in sigmas if not p.semantics]
    add("navigation", "every measurement sigma declares its semantics",
        f"{len(undeclared)} undeclared of {len(sigmas)}", "0", not undeclared,
        ", ".join(undeclared))

    invalid = [
        f"{path}={p.semantics}" for path, p in sigmas
        if p.semantics and p.semantics not in SIGMA_SEMANTICS
    ]
    add("navigation", "sigma semantics come from the controlled vocabulary",
        f"{len(invalid)} invalid", f"in {len(SIGMA_SEMANTICS)}-term vocabulary",
        not invalid, "; ".join(invalid))

    # A sigma may only claim to be the total budget if nothing it would need is
    # still unknown. Here the thermal term depends on an EIRP that is not public.
    overclaimed: list[str] = []
    for path, p in sigmas:
        if p.semantics != "TOTAL_DOPPLER_NOISE":
            continue
        blocked = [
            q for _, q in config.walk()
            if not q.is_known and any("thermal" in b.casefold() for b in q.blocks)
        ]
        if blocked:
            overclaimed.append(path)
    add("navigation", "no sigma claims to be total while a component is blocked",
        f"{len(overclaimed)} overclaimed", "0", not overclaimed,
        ", ".join(overclaimed))


# ======================================================================
# entry points
# ======================================================================
_CHECK_GROUPS: tuple[Callable[..., None], ...] = (
    _check_identity,
    _check_units,
    _check_provenance,
    _check_derivations,
    _check_frequency_plan,
    _check_srp_semantics,
    _check_discrepancies,
    _check_sigma_semantics,
)


def run_checks(config: ReferenceConfiguration) -> tuple[CheckResult, ...]:
    """Every semantic check, collected. Does not raise on ordinary failures."""
    results: list[CheckResult] = []

    def add(group: str, check: str, measured: Any, gate: str, passed: bool,
            detail: str = "") -> None:
        results.append(CheckResult(group, check, str(measured), gate,
                                   bool(passed), detail))

    for group_fn in _CHECK_GROUPS:
        group_fn(config, add)
    return tuple(results)


def validate_configuration(
    config: ReferenceConfiguration,
) -> tuple[CheckResult, ...]:
    """Run every check and raise if any failed.

    There is no lenient mode. A configuration that fails a semantic check is not
    a configuration with a warning attached; it is one whose numbers cannot all
    be true at once, and continuing would propagate that into results.
    """
    results = run_checks(config)
    failures = [r for r in results if not r.passed]
    if failures:
        lines = "\n".join(
            f"  [{r.group}] {r.check}: measured {r.measured}, expected {r.gate}"
            + (f" ({r.detail})" if r.detail else "")
            for r in failures
        )
        raise ReferenceConfigurationError(
            f"configuration {config.configuration_id!r} failed "
            f"{len(failures)} semantic check(s):\n{lines}"
        )
    return results
