# Dependency Hygiene Checklist

Dependencies, packaging, public-repo readiness, reproducible environments.

## Present at package-creation time
- `requirements.txt`, `requirements-dev.txt`, `requirements-accelerated.txt`
- `desktop_app/requirements_desktop.txt`

## Expected / verify in repo (not present at creation time)
- `pyproject.toml`, `setup.cfg`, `setup.py`, `environment.yml`,
  `poetry.lock`, `uv.lock`, `.pre-commit-config.yaml`

## Rules
- dependencies must be necessary; heavy dependencies must be justified.
- pin or bound versions when reproducibility matters.
- document optional desktop dependencies and SPICE-related dependencies.
- separate test / development dependencies where practical.
- missing kernels or optional artifacts should fail gracefully.
- no secrets, tokens, private paths, private PDFs, virtual environments, or
  unintended large generated artifacts committed.

## Critical rule
Dependency changes must not make scientific results irreproducible.
