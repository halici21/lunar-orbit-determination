"""WP7.1 / WP7.2 (REQ-SW-002, REQ-SW-003): verification levels, CI tiers and kernel fail-closed.

The level table lives in tests/levels.py and is applied by tests/conftest.py.  These
tests pin that (a) every test file has exactly one table row and every collected test
ends up with exactly one level marker, (b) the markers are registered for
``--strict-markers``, (c) the tier expressions of tools/run_tier.py only use those
markers and keep the commit tier kernel-free, and (d) an L2+ SPICE test FAILS without
kernels while a lower-level one keeps its skip unless ``--require-kernels`` is given.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tests import levels
from tests.slow import RUN_SLOW_TESTS, SLOW_ATTRIBUTE, slow

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "tools"))
import run_tier  # noqa: E402

_ALL_MARKERS = set(levels.LEVELS) | set(levels.RESOURCES) | {levels.REQUIRES_KERNELS}


def _env(tmp_path, **extra):
    """A subprocess environment with no kernels, campaign or gravity data and no run record."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for key, name in (("LUNAR_OD_KERNEL_DIR", "kernels"), ("LUNAR_OD_CAMPAIGN_ROOT", "campaign"),
                      ("LUNAR_OD_GRAVITY_ROOT", "gravity")):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        env[key] = str(folder)
    env["LUNAR_OD_RECORD_TESTS"] = "0"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("LUNAR_OD_STRICT_LEVELS", None)
    env.update(extra)
    return env


def test_every_test_file_has_one_table_row_and_no_row_is_stale():
    files = {p.name for p in TESTS.glob("test_*.py")}
    missing = sorted(files - set(levels.FILE_LEVELS))
    stale = sorted(set(levels.FILE_LEVELS) - files)
    assert not missing, ("test files without a verification level; add to FILE_LEVELS in tests/levels.py:\n"
                         + "\n".join(levels.suggested_row(name) for name in missing))
    assert not stale, f"FILE_LEVELS rows for files that no longer exist: {stale}"


def test_table_rows_name_one_known_level_and_known_resources():
    for key, (level, resources) in {**levels.FILE_LEVELS, **levels.NODE_LEVELS}.items():
        assert level in levels.LEVELS, (key, level)
        assert set(resources.split()) <= set(levels.RESOURCES), (key, resources)
    for prefix in levels.NODE_LEVELS:
        assert (ROOT / prefix.split("::", 1)[0]).is_file(), prefix
    assert levels.KERNEL_FAIL_CLOSED_LEVELS == set(levels.LEVELS) - {"analytic", "derivative", "contract_negative"}


def test_node_rows_refine_file_rows_and_unknown_files_stay_unclassified():
    node = next(iter(levels.NODE_LEVELS))
    assert levels.classify(node) == (levels.NODE_LEVELS[node][0], frozenset(levels.NODE_LEVELS[node][1].split()))
    assert levels.classify(node + "[case]")[0] == levels.NODE_LEVELS[node][0]
    assert levels.classify("tests/test_estimators.py::EstimatorTests::test_other")[0] == "analytic"
    assert levels.classify("tests\\test_dynamics.py::DynamicsTests::test_x")[0] == "analytic"
    assert levels.classify("tests/test_not_in_the_table.py::test_x") is None


def test_markers_are_registered_and_strict():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config =tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    registered = {entry.split(":", 1)[0].strip() for entry in config["markers"]}
    assert registered == _ALL_MARKERS
    assert "--strict-markers" in config["addopts"].split()


def test_slow_decorator_tags_tests_for_the_slow_marker():
    @slow
    def test_case():
        pass

    assert getattr(test_case, SLOW_ATTRIBUTE) is True
    assert getattr(test_case, "__unittest_skip__", False) is (not RUN_SLOW_TESTS)


def test_tier_expressions_use_registered_markers_and_commit_tier_is_kernel_free():
    for tier, (expression, _kernels, _slow, _purpose) in run_tier.TIERS.items():
        words = set(re.findall(r"[A-Za-z_]+", expression)) - {"and", "or", "not"}
        assert words <= _ALL_MARKERS, (tier, words - _ALL_MARKERS)
    commit = run_tier.TIERS["commit"][0]
    assert "not spice" in commit and not run_tier.TIERS["commit"][1]
    for level in ("analytic", "derivative", "contract_negative"):
        assert level in commit
    assert "external_oracle" in run_tier.TIERS["pr"][0] and run_tier.TIERS["pr"][1]
    assert run_tier.TIERS["nightly"][0] == "" and run_tier.TIERS["nightly"][2]
    assert run_tier.TIERS["release"][0] == "real_data"
    assert "--require-kernels" in run_tier.command_for("pr")
    assert "--require-kernels" not in run_tier.command_for("commit")
    assert "--strict-levels" in run_tier.command_for("commit")


_DUMP_PLUGIN = '''
import json, os
LEVELS = {levels!r}
def pytest_collection_finish(session):
    rows = {{item.nodeid: sorted({{m.name for m in item.iter_markers() if m.name in LEVELS}})
            for item in session.items}}
    with open(os.environ["LEVEL_DUMP"], "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
'''


def test_every_collected_test_has_exactly_one_level_marker(tmp_path):
    (tmp_path / "level_dump_plugin.py").write_text(_DUMP_PLUGIN.format(levels=sorted(levels.LEVELS)),
                                                  encoding="utf-8")
    dump = tmp_path / "levels.json"
    env = _env(tmp_path, PYTHONPATH=str(tmp_path), LEVEL_DUMP=str(dump))
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                             "-p", "level_dump_plugin", "tests"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert dump.is_file(), result.stdout[-3000:] + result.stderr[-3000:]
    rows = json.loads(dump.read_text(encoding="utf-8"))
    assert len(rows) > 1000
    wrong = {node: found for node, found in rows.items() if len(found) != 1}
    assert not wrong, (f"{len(wrong)} tests without exactly one level marker, e.g. "
                       + json.dumps(dict(list(wrong.items())[:10]), indent=1) + "\n" + result.stdout[-2000:])


def _run(tmp_path, *args):
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rA", *args],
                            cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout + result.stderr


def test_l2_spice_test_fails_without_kernels_instead_of_skipping(tmp_path):
    code, out = _run(tmp_path, "tests/test_spice_snapshots.py")
    assert code == 1, out[-3000:]
    assert "REQ-SW-003" in out and " skipped" not in out, out[-3000:]


def test_lower_level_spice_test_fails_only_when_kernels_are_required(tmp_path):
    node = "tests/test_srif_acceptance_wp4_1.py"
    code, out = _run(tmp_path, node, "--require-kernels")
    assert code == 1 and "REQ-SW-003" in out and " passed" not in out, out[-3000:]
    code, out = _run(tmp_path, node, "-m", "not spice")
    assert code == 5, out[-3000:]  # all deselected: nothing ran, nothing hidden


def test_strict_levels_names_the_row_that_fixes_an_unclassified_file(tmp_path):
    probe = TESTS / f"test_zz_unclassified_probe_{os.getpid()}.py"
    probe.write_text("def test_probe():\n    pass\n", encoding="utf-8")
    try:
        code, out = _run(tmp_path, str(probe.relative_to(ROOT)), "--strict-levels")
        assert code != 0 and f'"{probe.name}": (_A, "")' in out, out[-3000:]
        code, out = _run(tmp_path, str(probe.relative_to(ROOT)))
        assert code == 0 and "unclassified tests (WP7.1)" in out, out[-3000:]
    finally:
        probe.unlink()
