# Validation

## Public regression gate

The [Windows workflow](../../.github/workflows/public-checks.yml) installs the
reference dependencies and package, verifies repository paths and preserved
records, builds a wheel, and runs the bounded public test set.

```powershell
python -m pip check
python tools/check_repository.py
python tools/run_public_checks.py --junitxml outputs/public-checks.xml
```

The runner redirects external data roots into empty temporary directories.
The tutorial tests cover deterministic shared noise and recovery of the known
initial state from noiseless synthetic measurements. Their metre / cm/s bounds
protect this educational case; they are not an operational accuracy requirement.

## Research regression

```powershell
python -m pytest -p tests.desktop_preferences tests --durations=15 -q
```

The full suite has existing external-data skips, kernel-dependent cases and
historical provenance requirements. See the [complete verification record](../validation/verification.md)
for setup, tiers, historical counts and qualification boundaries. Long campaigns
remain explicit manual runs rather than default CI jobs.

## Documentation and release

```powershell
python -m pip install -r requirements-docs.txt
python tools/build_docs.py --strict
```

Documentation builds are checked on pull requests. Only main is deployed to
GitHub Pages. Release validation also checks a clean source extraction, package
installation/imports and the data-free example. The
[release checklist](../development/releases.md) distinguishes source previews,
library wheels and the future standalone desktop installer.
