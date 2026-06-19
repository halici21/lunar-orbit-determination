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

slow = unittest.skipUnless(
    RUN_SLOW_TESTS,
    "slow regression; set LUNAR_OD_RUN_SLOW_TESTS=1 to run",
)
