# Test suite

Tests retain their existing flat paths and pytest node identities because
many science and desktop tests reuse helpers across files. Default discovery
is `tests/` in the worktree-local `pytest.ini`; no tests were removed or hidden
as part of repository reorganization.

| Group | Representative files |
| --- | --- |
| Unit | `test_geometry.py`, `test_orbit.py`, `test_config.py` |
| Integration | `test_scenarios.py`, `test_scenario_config.py`, `test_desktop_workflow.py` |
| Regression/contracts | `test_od_contracts.py`, `test_force_contract.py`, `test_harden_import_isolation.py` |
| Scientific qualification | `test_r*.py`, `test_harden_*.py`, `test_f1_*.py`, `test_kp*.py` |
| Desktop | `test_tracking_*.py`, `test_workspace_*.py`, `test_mission_overview.py` |

Install the editable package first. Run `python -m pytest -p
tests.desktop_preferences -ra` for the combined suite with isolated desktop
preferences. `python -m pytest` also discovers the same suite; the preferences
plugin protects local user settings during GUI checks.

`python tools/run_public_checks.py` runs the existing public subset with
external-data overrides pointed at empty temporary directories. Scientific
tests keep their existing missing-data skip policies. Historical provenance
tests need Git history and Windows filtered-checkout line endings. Long study
campaigns are manual and are not added to default CI.

For the existing opt-in slow profile, set `LUNAR_OD_RUN_SLOW_TESTS=1` before
running the suite and remove the override afterward. It covers seeded Monte
Carlo, long-arc UKF, counted-Doppler light time, square-root stress and scenario
handoff regressions; its cost and input requirements make it a manual gate.
