"""WP4.3b (EST-18): observability rank/condition and bias classification are unit-invariant.

Before WP4.3b the bias decision divided each bias's conditional information by the
largest one, so in station_full mode a range bias (m) was judged against angle biases
(rad) and a change of unit alone froze, regularised or kept it; rank, condition number
and the weakest direction came from the raw (unit-dependent) design.  Oracle: a pure
change of units (column scaling) is not a change of the estimation problem.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od import (
    BiasObservabilityPolicy,
    decide_bias_state_handling,
    summarize_weighted_jacobian,
)


def _design(seed=0, m=60):
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((m, 6))
    independent = rng.standard_normal((m, 2))                       # well separated biases
    weak = state @ rng.standard_normal(6) + 1e-4 * rng.standard_normal(m)   # nearly in span
    unseen = np.zeros(m)                                             # bias with no data
    return np.column_stack([state, independent, weak, unseen])


POLICY = BiasObservabilityPolicy(freeze_relative_information=1e-12,
                                 regularize_relative_information=1e-5,
                                 regularization_std=2.0)


def _decision(h):
    return decide_bias_state_handling(summarize_weighted_jacobian("position", 1, h), policy=POLICY)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("units", [
    [1, 1, 1, 1, 1, 1, 1e-6, 1e6, 1, 1],          # rad vs microrad vs metres on the biases
    [1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1, 1, 1e-4, 1],  # estimator bookkeeping on the state
    [1, 1, 1, 1, 1, 1, 1, 1, 1e8, 1],
])
def test_bias_decision_is_invariant_to_units(seed, units):
    h = _design(seed)
    reference, scaled = _decision(h), _decision(h * np.asarray(units, float))
    assert scaled.active_state_indices == reference.active_state_indices
    assert scaled.regularized_state_indices == reference.regularized_state_indices
    assert scaled.frozen_state_indices == reference.frozen_state_indices
    np.testing.assert_allclose(scaled.relative_information, reference.relative_information,
                               rtol=1e-6, atol=1e-12)


def test_bias_classification_follows_separability_not_magnitude():
    decision = _decision(_design() * np.array([1, 1, 1, 1, 1, 1, 1e-9, 1e3, 1, 1]))
    assert decision.active_state_indices == (6, 7)         # separable, whatever the unit
    assert decision.regularized_state_indices == (8,)      # survives at ~1e-8 of its own
    assert decision.frozen_state_indices == (9,)            # never seen by the data
    assert np.all((0.5 < decision.relative_information[:2]) & (decision.relative_information[:2] <= 1.0))


@pytest.mark.parametrize("seed", range(3))
def test_rank_condition_and_weakest_direction_are_invariant_to_units(seed):
    h = _design(seed)[:, :9]
    units = np.logspace(-6, 9, h.shape[1])
    a, b = summarize_weighted_jacobian("position", 1, h), summarize_weighted_jacobian("position", 1, h * units)
    assert a.rank == b.rank == h.shape[1]
    assert b.condition_number == pytest.approx(a.condition_number, rel=1e-6)
    # the same physical direction: map b's (scaled-coordinate) vector back to a's coordinates
    back = b.weakest_direction * units
    back /= np.linalg.norm(back)
    assert abs(float(back @ a.weakest_direction)) == pytest.approx(1.0, abs=1e-6)
