"""Opt-in marker for expensive Lunar OD regression tests."""

from __future__ import annotations

import os
import unittest

RUN_SLOW_TESTS = os.environ.get("LUNAR_OD_RUN_SLOW_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_skip_unless_slow = unittest.skipUnless(
    RUN_SLOW_TESTS,
    "slow regression; set LUNAR_OD_RUN_SLOW_TESTS=1 to run",
)

#: Attribute ``tests/conftest.py`` reads to add the ``slow`` pytest marker (WP7.1).
SLOW_ATTRIBUTE = "__lunar_od_slow__"


def slow(test_item):
    """Skip unless LUNAR_OD_RUN_SLOW_TESTS is set; tag the test or class for the ``slow`` marker."""
    test_item = _skip_unless_slow(test_item)
    setattr(test_item, SLOW_ATTRIBUTE, True)
    return test_item
