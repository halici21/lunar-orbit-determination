# CI / Quality Gates

CI, pre-push checks, and merge readiness.

## Fast default gate
```bash
python -m pytest python_port/tests/ -q
```

## Targeted examples
```bash
python -m pytest python_port/tests/test_dynamics.py -q
python -m pytest python_port/tests/test_measurements.py -q
python -m pytest python_port/tests/test_estimators.py -q
python -m pytest python_port/tests/test_filters.py -q
python -m pytest python_port/tests/test_scenarios.py -q
python -m pytest python_port/tests/test_scenario_config.py -q
```

## Optional quality gates (only if installed / configured)
```bash
python -m ruff check python_port
python -m ruff format --check python_port
python -m mypy python_port/lunar_od python_port/desktop_app
python -m pytest python_port/tests/ --cov=lunar_od --cov-report=term-missing
python -m pip check
python -m pip-audit
python -m bandit -r python_port
```

## Rules
- Do not run long Monte Carlo or thesis campaigns in default CI.
- Mark long campaigns as manual / nightly / release-only / benchmark-only.
- CI must not depend on private local paths.
- CI must skip gracefully when SPICE kernels are unavailable.
- Separate fast tests from long scientific campaigns.

## Repository configuration status (expected / verify in repo)
- `pyproject.toml`, `setup.cfg` — **not present** at package-creation time.
- `.github/workflows/` — **not present** (no CI configured yet).
- `.pre-commit-config.yaml`, coverage / lint / type-check config — **not present**.
These are optional to add later; do not create them unless explicitly asked.
