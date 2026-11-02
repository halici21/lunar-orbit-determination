"""WP4.7 (EST-20): Monte Carlo trial seeds from SeedSequence; common random numbers.

Before WP4.7 trial k of a run with base seed b had seed b + k - 1, so runs with base
seeds 100 and 101 shared every trial but one (identical noise streams, silently
correlated "independent" campaigns), and every campaign case got its own block of
seeds, so a case-to-case difference carried the full sampling variance of both cases
instead of being a paired comparison.  Now seed = SeedSequence([b, k]) (64-bit), the
same for every case of a campaign; ``seed_scheme="legacy_offset"`` reproduces the old
seeds for archived campaigns.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunar_od.monte_carlo import (
    MonteCarloCase,
    make_trial_specs,
    run_monte_carlo_campaign,
    run_monte_carlo_trials,
)


def test_trial_seed_is_the_seed_sequence_of_base_and_trial():
    specs = make_trial_specs(3, base_seed=100)
    expected = [int(np.random.SeedSequence([100, k]).generate_state(1, np.uint64)[0]) for k in (1, 2, 3)]
    assert [s.seed for s in specs] == expected


def test_neighbouring_base_seeds_share_no_trial_stream():
    def draw(spec, rng):
        return {"x": float(rng.standard_normal())}

    a = run_monte_carlo_trials(50, draw, base_seed=100)
    b = run_monte_carlo_trials(50, draw, base_seed=101)
    assert not set(r.seed for r in a) & set(r.seed for r in b)          # was 49 shared
    assert not set(r.metrics["x"] for r in a) & set(r.metrics["x"] for r in b)


def test_campaign_cases_use_common_random_numbers():
    """A paired case difference cancels the shared noise: with CRN the difference of
    two cases that only scale the same draw has zero variance across trials."""
    cases = (MonteCarloCase("a", {"scale": 1.0}), MonteCarloCase("b", {"scale": 1.0}))

    def trial(case, spec, rng):
        return {"x": case.parameters["scale"] * float(rng.standard_normal())}

    result = run_monte_carlo_campaign(cases, 40, trial, base_seed=5, max_workers=2)
    xa = np.array([t.metrics["x"] for t in result[0].trials])
    xb = np.array([t.metrics["x"] for t in result[1].trials])
    assert [t.seed for t in result[0].trials] == [t.seed for t in result[1].trials]
    np.testing.assert_array_equal(xa, xb)                                # was independent


def test_legacy_offset_scheme_reproduces_archived_seeds():
    specs = make_trial_specs(4, base_seed=100, seed_scheme="legacy_offset")
    assert [s.seed for s in specs] == [100, 101, 102, 103]
    cases = (MonteCarloCase("a"), MonteCarloCase("b"))
    result = run_monte_carlo_campaign(cases, 3, lambda c, s, r: {"v": 0.0}, base_seed=100,
                                      seed_scheme="legacy_offset")
    assert [t.seed for t in result[1].trials] == [103, 104, 105]


def test_unknown_seed_scheme_rejected():
    with pytest.raises(ValueError):
        make_trial_specs(2, seed_scheme="sequential")
