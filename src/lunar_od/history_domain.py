"""Shared history-domain primitives for FA-03B enforcement (P0B-2A).

Legacy production light-time paths silently extrapolated spacecraft/station/
Earth/transform histories outside their support (FA-03B). This module owns
the single shared contract used by every guarded legacy production boundary:

- closed support ``D = [t_start, t_end]`` with exact endpoints accepted;
- a fixed two-ULP representation allowance that normalizes a request lying at
  most ``MAX_BOUNDARY_ULPS`` representable steps (at the compared
  relative-time magnitude ``S = max(1, |q|, |b|)``) outside a bound onto that
  bound — an endpoint *sample*, never an extrapolation;
- :class:`HistoryDomainError` with mandatory quantitative diagnostics
  (support bounds, outside distance, required pre/post-roll) plus optional
  endpoint/event/observation context;
- :class:`HistoryDomainDropRecord` for structured generation-time candidate
  drops, and :func:`summarize_history_domain_drops` for the metadata
  aggregates.

M3 (`lunar_od/two_way_range.py`) keeps its own validated four-event history
contract and is intentionally NOT routed through this module. The low-level
interpolators keep their generic behavior; guards wrap legacy *production*
boundaries only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = [
    "MAX_BOUNDARY_ULPS",
    "HistoryDomainError",
    "HistoryDomainDropRecord",
    "normalize_supported_epoch",
    "guarded_history_callback",
    "summarize_history_domain_drops",
]

# Fixed by the FA-03B boundary policy: at most two representable float64
# steps at the compared magnitude may be normalized onto a support endpoint.
# This is deliberately NOT configurable — a larger allowance would begin to
# hide physically unsupported events, and the mutation test
# (test_boundary_policy_mutation_rejects_larger_allowance) kills any change.
MAX_BOUNDARY_ULPS = 2


def _representation_allowance_s(requested_epoch_s: float, bound_s: float) -> float:
    """Two-ULP allowance at the compared relative-time magnitude.

    ``S = max(1, |q|, |b|)``; ``ulp(S) = nextafter(S, +inf) - S``;
    allowance ``= MAX_BOUNDARY_ULPS * ulp(S)``. The allowance therefore does
    not scale with history cadence, range, or light time — only with the
    floating-point representation of the compared epochs themselves.
    """
    scale = max(1.0, abs(requested_epoch_s), abs(bound_s))
    return MAX_BOUNDARY_ULPS * (math.nextafter(scale, math.inf) - scale)


class HistoryDomainError(ValueError):
    """A production history lookup was requested outside its support.

    Mandatory quantitative fields are always populated; optional context
    fields identify the counted endpoint, light-time leg, and observation
    when the failing consumer supplies them. Aggregate (generation-summary)
    instances additionally carry the immutable per-candidate
    :class:`HistoryDomainDropRecord` tuple in ``drop_records``.
    """

    def __init__(
        self,
        message: str,
        *,
        history_name: str,
        requested_epoch_s: float,
        support_start_s: float,
        support_end_s: float,
        outside_distance_s: float,
        required_pre_roll_s: float,
        required_post_roll_s: float,
        model_context: str,
        consumer: str,
        endpoint_label: str | None = None,
        event_label: str | None = None,
        observation_index: int | None = None,
        drop_records: tuple["HistoryDomainDropRecord", ...] = (),
    ) -> None:
        super().__init__(message)
        self.history_name = str(history_name)
        self.requested_epoch_s = float(requested_epoch_s)
        self.support_start_s = float(support_start_s)
        self.support_end_s = float(support_end_s)
        self.outside_distance_s = float(outside_distance_s)
        self.required_pre_roll_s = float(required_pre_roll_s)
        self.required_post_roll_s = float(required_post_roll_s)
        self.model_context = str(model_context)
        self.consumer = str(consumer)
        self.endpoint_label = endpoint_label
        self.event_label = event_label
        self.observation_index = observation_index
        self.drop_records = tuple(drop_records)

    @classmethod
    def from_violation(
        cls,
        *,
        history_name: str,
        requested_epoch_s: float,
        support_start_s: float,
        support_end_s: float,
        model_context: str,
        consumer: str,
        endpoint_label: str | None = None,
        event_label: str | None = None,
        observation_index: int | None = None,
    ) -> "HistoryDomainError":
        """Build a single-violation error with derived distance/pre/post-roll.

        ``required_pre_roll_s``/``required_post_roll_s`` are the exact query
        deficiencies ``max(0, start - q)`` / ``max(0, q - end)``; automatic
        history extension belongs to a later architecture phase.
        """
        q = float(requested_epoch_s)
        start = float(support_start_s)
        end = float(support_end_s)
        if math.isfinite(q):
            pre = max(0.0, start - q)
            post = max(0.0, q - end)
        else:
            pre = math.inf
            post = math.inf
        outside = max(pre, post)
        context_bits = []
        if endpoint_label is not None:
            context_bits.append(f"endpoint={endpoint_label}")
        if event_label is not None:
            context_bits.append(f"event={event_label}")
        if observation_index is not None:
            context_bits.append(f"observation_index={observation_index}")
        context_suffix = f" ({', '.join(context_bits)})" if context_bits else ""
        message = (
            f"{model_context}: {consumer} requested {history_name} at epoch "
            f"{q!r} s outside the supported history "
            f"[{start:.9f}, {end:.9f}] s by {outside:.9e} s"
            f"{context_suffix}. Required pre-roll {pre:.9e} s, post-roll "
            f"{post:.9e} s; refusing to extrapolate."
        )
        return cls(
            message,
            history_name=history_name,
            requested_epoch_s=q,
            support_start_s=start,
            support_end_s=end,
            outside_distance_s=outside,
            required_pre_roll_s=pre,
            required_post_roll_s=post,
            model_context=model_context,
            consumer=consumer,
            endpoint_label=endpoint_label,
            event_label=event_label,
            observation_index=observation_index,
        )

    @classmethod
    def from_drop_summary(
        cls,
        drop_records: Sequence["HistoryDomainDropRecord"],
        *,
        model_context: str,
        consumer: str,
    ) -> "HistoryDomainError":
        """Aggregate error using the worst violation's mandatory values.

        The worst violation is the record with the largest outside distance;
        all records travel immutably on the exception so scenario callers can
        report every candidate without re-raising.
        """
        records = tuple(drop_records)
        if not records:
            raise ValueError("from_drop_summary requires at least one drop record.")
        worst = max(records, key=lambda record: record.outside_distance_s)
        message = (
            f"{model_context}: {consumer} dropped {len(records)} candidate "
            f"measurement(s) for unsupported history epochs; worst violation "
            f"{worst.history_name} at {worst.requested_epoch_s!r} s outside "
            f"[{worst.support_start_s:.9f}, {worst.support_end_s:.9f}] s by "
            f"{worst.outside_distance_s:.9e} s (required pre-roll "
            f"{worst.required_pre_roll_s:.9e} s, post-roll "
            f"{worst.required_post_roll_s:.9e} s)."
        )
        return cls(
            message,
            history_name=worst.history_name,
            requested_epoch_s=worst.requested_epoch_s,
            support_start_s=worst.support_start_s,
            support_end_s=worst.support_end_s,
            outside_distance_s=worst.outside_distance_s,
            required_pre_roll_s=worst.required_pre_roll_s,
            required_post_roll_s=worst.required_post_roll_s,
            model_context=model_context,
            consumer=consumer,
            endpoint_label=worst.endpoint_label,
            event_label=worst.event_label,
            observation_index=worst.observation_index,
            drop_records=records,
        )


@dataclass(frozen=True)
class HistoryDomainDropRecord:
    """One structured generation-time candidate drop (FA-03B).

    Serializes the mandatory :class:`HistoryDomainError` fields and may carry
    candidate context (arc, station/time indices, candidate ordinal) without
    introducing a second exception type.
    """

    history_name: str
    requested_epoch_s: float
    support_start_s: float
    support_end_s: float
    outside_distance_s: float
    required_pre_roll_s: float
    required_post_roll_s: float
    model_context: str
    consumer: str
    endpoint_label: str | None = None
    event_label: str | None = None
    observation_index: int | None = None
    arc_id: int | None = None
    station_index: int | None = None
    time_index: int | None = None
    candidate_ordinal: int | None = None

    @classmethod
    def from_error(
        cls,
        error: HistoryDomainError,
        *,
        arc_id: int | None = None,
        station_index: int | None = None,
        time_index: int | None = None,
        candidate_ordinal: int | None = None,
    ) -> "HistoryDomainDropRecord":
        return cls(
            history_name=error.history_name,
            requested_epoch_s=error.requested_epoch_s,
            support_start_s=error.support_start_s,
            support_end_s=error.support_end_s,
            outside_distance_s=error.outside_distance_s,
            required_pre_roll_s=error.required_pre_roll_s,
            required_post_roll_s=error.required_post_roll_s,
            model_context=error.model_context,
            consumer=error.consumer,
            endpoint_label=error.endpoint_label,
            event_label=error.event_label,
            observation_index=error.observation_index,
            arc_id=arc_id,
            station_index=station_index,
            time_index=time_index,
            candidate_ordinal=candidate_ordinal,
        )

    def as_dict(self) -> dict:
        """Deterministically ordered plain-dict form for metadata transport."""
        return {
            "history_name": self.history_name,
            "requested_epoch_s": self.requested_epoch_s,
            "support_start_s": self.support_start_s,
            "support_end_s": self.support_end_s,
            "outside_distance_s": self.outside_distance_s,
            "required_pre_roll_s": self.required_pre_roll_s,
            "required_post_roll_s": self.required_post_roll_s,
            "model_context": self.model_context,
            "consumer": self.consumer,
            "endpoint_label": self.endpoint_label,
            "event_label": self.event_label,
            "observation_index": self.observation_index,
            "arc_id": self.arc_id,
            "station_index": self.station_index,
            "time_index": self.time_index,
            "candidate_ordinal": self.candidate_ordinal,
        }


def normalize_supported_epoch(
    requested_epoch_s: float,
    support_start_s: float,
    support_end_s: float,
    *,
    history_name: str,
    model_context: str,
    consumer: str,
    endpoint_label: str | None = None,
    event_label: str | None = None,
    observation_index: int | None = None,
) -> float:
    """Return the supported lookup epoch for ``requested_epoch_s``.

    Inside the closed interval (endpoints included) the request is returned
    unchanged. A request at most :data:`MAX_BOUNDARY_ULPS` representable
    steps (at magnitude ``S = max(1, |q|, |b|)``) outside a bound is
    normalized onto that bound — the caller then evaluates an in-support
    endpoint sample; the original request stays available to the caller for
    diagnostics. Anything farther outside raises :class:`HistoryDomainError`
    immediately, before any interpolation or extrapolation, and never
    mutates a solver event variable.
    """
    start = float(support_start_s)
    end = float(support_end_s)
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        raise ValueError(
            f"history support for {history_name!r} must be a finite ordered "
            f"interval; got [{support_start_s!r}, {support_end_s!r}]."
        )
    q = float(requested_epoch_s)
    if math.isfinite(q):
        if start <= q <= end:
            return q
        if q < start and (start - q) <= _representation_allowance_s(q, start):
            return start
        if q > end and (q - end) <= _representation_allowance_s(q, end):
            return end
    raise HistoryDomainError.from_violation(
        history_name=history_name,
        requested_epoch_s=q,
        support_start_s=start,
        support_end_s=end,
        model_context=model_context,
        consumer=consumer,
        endpoint_label=endpoint_label,
        event_label=event_label,
        observation_index=observation_index,
    )


def guarded_history_callback(
    callback: Callable[[float], object],
    support_start_s: float,
    support_end_s: float,
    *,
    history_name: str,
    model_context: str,
    consumer: str,
    endpoint_label: str | None = None,
    event_label: str | None = None,
    observation_index: int | None = None,
) -> Callable[[float], object]:
    """Wrap an epoch-indexed lookup with the strict history-domain guard.

    Every evaluation — including intermediate solver probes — passes through
    :func:`normalize_supported_epoch` first, so the first unsupported probe
    raises before any model evaluation happens outside support.
    """

    def guarded(epoch_s: float):
        normalized = normalize_supported_epoch(
            epoch_s,
            support_start_s,
            support_end_s,
            history_name=history_name,
            model_context=model_context,
            consumer=consumer,
            endpoint_label=endpoint_label,
            event_label=event_label,
            observation_index=observation_index,
        )
        return callback(normalized)

    return guarded


def summarize_history_domain_drops(
    drop_records: Sequence[HistoryDomainDropRecord],
) -> dict:
    """Aggregate drop records into the FA-03B metadata fields.

    Returns the numeric aggregates used by ``PassGeometry`` metadata (and,
    in P0B-2D, the four scenario-level summary fields): drop count, ordered
    serialized records, and the maximum required pre-roll/post-roll across
    all candidates.
    """
    records = tuple(drop_records)
    return {
        "history_domain_dropped_measurements": len(records),
        "history_domain_drop_records": tuple(record.as_dict() for record in records),
        "history_domain_required_pre_roll_s": (
            max((record.required_pre_roll_s for record in records), default=0.0)
        ),
        "history_domain_required_post_roll_s": (
            max((record.required_post_roll_s for record in records), default=0.0)
        ),
    }
