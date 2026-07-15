"""P0B-2A primitive tests (T001-T005): FA-03B history-domain contract.

The boundary tests construct ULP excursions with repeated ``math.nextafter``
from the support bounds (never decimal approximations). The bounds are chosen
away from powers of two and with magnitude >= 1 so one representable step at
the bound equals one ULP of the policy scale ``S = max(1, |q|, |b|)`` in both
directions — making the 2-ULP acceptance and 3-ULP rejection exact.
"""

from __future__ import annotations

import math

import pytest

from lunar_od.history_domain import (
    MAX_BOUNDARY_ULPS,
    HistoryDomainDropRecord,
    HistoryDomainError,
    normalize_supported_epoch,
    summarize_history_domain_drops,
)
import lunar_od.history_domain as history_domain_module

_SUPPORT_CASES = (
    (10.0, 100.0),
    (-100.0, -10.0),  # symmetric negative-time support
)

_CONTEXT = {
    "history_name": "spacecraft_state",
    "model_context": "unit_test",
    "consumer": "test_history_domain",
}


def _steps_outside(bound: float, direction: float, count: int) -> float:
    value = bound
    for _ in range(count):
        value = math.nextafter(value, direction)
    return value


def test_exact_closed_interval_endpoints():
    """T001: exact lower/upper bounds are accepted and returned unchanged."""
    for start, end in _SUPPORT_CASES:
        assert normalize_supported_epoch(start, start, end, **_CONTEXT) == start
        assert normalize_supported_epoch(end, start, end, **_CONTEXT) == end
        interior = 0.5 * (start + end)
        assert normalize_supported_epoch(interior, start, end, **_CONTEXT) == interior


def test_two_ulp_outside_snaps_symmetrically():
    """T002: 1- and 2-ULP excursions snap onto the endpoint on both sides."""
    for start, end in _SUPPORT_CASES:
        for count in (1, 2):
            below = _steps_outside(start, -math.inf, count)
            above = _steps_outside(end, math.inf, count)
            assert below < start and above > end  # genuinely outside requests
            assert normalize_supported_epoch(below, start, end, **_CONTEXT) == start
            assert normalize_supported_epoch(above, start, end, **_CONTEXT) == end
            # The original request is retained by the caller (the helper is
            # pure); nothing mutates the requested value.
            assert below == _steps_outside(start, -math.inf, count)


def test_three_ulp_outside_rejected_symmetrically():
    """T003: the third representable float outside each bound raises with
    exact distance and pre/post-roll deficiencies."""
    for start, end in _SUPPORT_CASES:
        below = _steps_outside(start, -math.inf, MAX_BOUNDARY_ULPS + 1)
        with pytest.raises(HistoryDomainError) as lower:
            normalize_supported_epoch(below, start, end, **_CONTEXT)
        err = lower.value
        assert err.history_name == "spacecraft_state"
        assert err.requested_epoch_s == below
        assert err.support_start_s == start and err.support_end_s == end
        assert err.outside_distance_s == start - below
        assert err.required_pre_roll_s == start - below
        assert err.required_post_roll_s == 0.0

        above = _steps_outside(end, math.inf, MAX_BOUNDARY_ULPS + 1)
        with pytest.raises(HistoryDomainError) as upper:
            normalize_supported_epoch(above, start, end, **_CONTEXT)
        err = upper.value
        assert err.outside_distance_s == above - end
        assert err.required_post_roll_s == above - end
        assert err.required_pre_roll_s == 0.0

    # Non-finite requests are unconditional domain violations.
    with pytest.raises(HistoryDomainError):
        normalize_supported_epoch(float("nan"), 10.0, 100.0, **_CONTEXT)
    with pytest.raises(HistoryDomainError):
        normalize_supported_epoch(float("inf"), 10.0, 100.0, **_CONTEXT)


def test_boundary_policy_mutation_rejects_larger_allowance(monkeypatch):
    """T004: enlarging the fixed allowance from 2 to 3 ULP must be caught by
    the 3-ULP rejection contract (mutation kill)."""
    start, end = _SUPPORT_CASES[0]
    below = _steps_outside(start, -math.inf, 3)

    original = history_domain_module._representation_allowance_s

    def mutated(requested_epoch_s: float, bound_s: float) -> float:
        return original(requested_epoch_s, bound_s) * 3.0 / MAX_BOUNDARY_ULPS

    monkeypatch.setattr(
        history_domain_module, "_representation_allowance_s", mutated
    )
    # Under the mutation the 3-ULP excursion is (wrongly) accepted — i.e. the
    # T003 expectation fails, so the mutation is killed by that contract.
    assert normalize_supported_epoch(below, start, end, **_CONTEXT) == start
    monkeypatch.undo()
    with pytest.raises(HistoryDomainError):
        normalize_supported_epoch(below, start, end, **_CONTEXT)


def test_history_domain_error_fields_and_drop_serialization():
    """T005: mandatory fields are finite; records/aggregates deterministic."""
    lower = HistoryDomainError.from_violation(
        history_name="earth_position_mci",
        requested_epoch_s=5.0,
        support_start_s=10.0,
        support_end_s=100.0,
        model_context="one_way_light_time",
        consumer="generate_position_measurements",
        endpoint_label=None,
        event_label="transmit",
        observation_index=0,
    )
    upper = HistoryDomainError.from_violation(
        history_name="spacecraft_state",
        requested_epoch_s=112.5,
        support_start_s=10.0,
        support_end_s=100.0,
        model_context="two_way_counted_doppler",
        consumer="solve_two_way_light_time",
        endpoint_label="count-end",
        event_label="uplink",
        observation_index=3,
    )
    for err in (lower, upper):
        for value in (
            err.requested_epoch_s,
            err.support_start_s,
            err.support_end_s,
            err.outside_distance_s,
            err.required_pre_roll_s,
            err.required_post_roll_s,
        ):
            assert math.isfinite(value)
    assert lower.required_pre_roll_s == 5.0 and lower.required_post_roll_s == 0.0
    assert upper.required_post_roll_s == 12.5 and upper.required_pre_roll_s == 0.0
    assert upper.endpoint_label == "count-end" and upper.event_label == "uplink"

    records = (
        HistoryDomainDropRecord.from_error(
            lower, arc_id=1, station_index=0, time_index=0, candidate_ordinal=0
        ),
        HistoryDomainDropRecord.from_error(
            upper, arc_id=1, station_index=2, time_index=7, candidate_ordinal=5
        ),
    )
    # Deterministic serialization: identical construction -> identical dicts.
    assert [record.as_dict() for record in records] == [
        HistoryDomainDropRecord.from_error(
            lower, arc_id=1, station_index=0, time_index=0, candidate_ordinal=0
        ).as_dict(),
        HistoryDomainDropRecord.from_error(
            upper, arc_id=1, station_index=2, time_index=7, candidate_ordinal=5
        ).as_dict(),
    ]

    summary = summarize_history_domain_drops(records)
    assert summary["history_domain_dropped_measurements"] == 2
    assert summary["history_domain_required_pre_roll_s"] == 5.0
    assert summary["history_domain_required_post_roll_s"] == 12.5
    assert len(summary["history_domain_drop_records"]) == 2

    aggregate = HistoryDomainError.from_drop_summary(
        records, model_context="generation", consumer="build_measurement_arcs"
    )
    # Worst violation (largest outside distance) supplies mandatory values.
    assert aggregate.history_name == "spacecraft_state"
    assert aggregate.outside_distance_s == 12.5
    assert aggregate.drop_records == records
    with pytest.raises(ValueError):
        HistoryDomainError.from_drop_summary(
            (), model_context="generation", consumer="build_measurement_arcs"
        )
    assert summarize_history_domain_drops(())["history_domain_dropped_measurements"] == 0
