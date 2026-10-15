# Maintenance and verification automation

## Dependency changes

[Dependabot configuration](../../.github/dependabot.yml) pauses routine version-update
pull requests (`open-pull-requests-limit: 0`) for both GitHub Actions and Python.
This keeps `main` as the only persistent branch. The existing three bot proposals
were closed without merging their dependency changes. Automatic security-fix PRs
are also disabled in repository settings; dependency alerts and the mandatory PR
dependency review remain enabled. Review alerts and proposed version changes
manually, qualify a focused update branch, then delete it after merging.

To resume routine version proposals deliberately, raise the limits in that
configuration. No automatic merge is configured.

The pinned `requirements/windows.txt` profile remains the scientific/desktop
reference. Updating it requires passing public, library and package qualification
checks, inspecting the changed versions and recording new numerical evidence when
behavior changes. A dependency PR does not silently bless new study baselines.

[Dependency review](../../.github/workflows/dependency-review.yml) rejects new
moderate-or-higher known vulnerable dependencies reported by GitHub's dependency
graph. It uses read-only permissions and does not post PR comments. Repository
dependency alerts/graph are enabled. This does not scan every possible runtime
behavior or prove absence of vulnerabilities.

## Verification tiers

| Tier | Execution | Evidence |
|---|---|---|
| Public Windows | Every PR and main push | Fixture-only desktop/library checks, repository record/curated-media hashes, dependency consistency |
| Library matrix | Every PR and main push | Eight Windows/Linux × Python 3.10–3.13 jobs; installed-wheel imports/analytic orbit and data-free regressions |
| Documentation | Every PR and main push | Strict site build; Pages deployment only on main |
| Package qualification | Package-affecting PRs, manual builds and version tags | Fresh wheel/source installations and relocated bundled-runtime desktop execution/cancellation |
| Paired scientific campaign | Manual, explicitly scoped | Declared seeds, matched controls, raw data, source/environment identities and scientific review |

The matrix aggregate fails unless all eight jobs pass. The four external-SPICE
integration cases listed in the library runner remain available through
`python tools/run_library_checks.py --with-kernels`; they are intentionally
outside the data-free tier. Missing data never becomes an invented passing result.

The bounded published campaign is not scheduled repeatedly in default CI.
The release workflow and [release procedure](releases.md) define what an actual
published asset has passed. No Codex reminder, hidden nightly campaign or DOI
automation is created by this repository configuration.
