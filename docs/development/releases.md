# Qualified source, wheel and Windows portable releases

## Distribution types

- **Windows source preview:** tracked source and launch/setup scripts; Python and
  network access for dependencies are required. No standalone EXE is included.
- **Library wheel:** installs `lunar_od`; the desktop, examples, research archive
  and SPICE kernels are not bundled. Run tutorials from a source checkout/ZIP.
- **Windows x64 portable desktop:** includes CPython 3.13.2 and the reference
  dependencies. Extract the full ZIP and open `LunarOD.cmd`; Python installation
  is not required. This is an unsigned portable alpha, not an installer.
  Its libraries keep their own licenses/notices under the runtime's
  `Lib/site-packages/*.dist-info`; CPython's license is `runtime/LICENSE.txt`.

## Release gate

1. Select a clean committed source revision; require public CI and documentation builds.
2. Run the data-free quickstart and preserve the exact summary/source identity.
3. Check preserved scientific bytes and package dependencies; run the relevant research regressions.
4. Build the wheel and test imports in a fresh environment outside the checkout.
5. Extract the source ZIP to a clean path containing spaces, install its reference
   dependencies/package, and check launcher imports and the tutorial.
6. Keep kernels, local runs, environments and Git history out of release assets.
7. Extract the portable ZIP into a different path with spaces and `&`; run its
   verification with external Python/PYTHONPATH removed. Require bundled imports,
   QProcess execution/cancellation, a real tutorial solve and desktop shutdown.
8. Publish source ZIP, wheel, portable ZIP, verification JSON and SHA256SUMS with
   exact commit, scope and known limits.

## Automated build and publication

[Release qualification](../../.github/workflows/release.yml) runs package-affecting
PR checks and manual builds with read-only repository access. A new `v*` tag must
match `pyproject.toml` and belong to `main`. After every build/installation gate
passes, a separate job receives `contents: write`, checks all asset hashes, and
publishes the prerelease. It refuses to replace an existing release's assets.
PR and manual-dispatch runs never publish.

The source, wheel and portable tests run in fresh environments or freshly extracted
directories on the Windows runner. The portable check verifies all bundle hashes,
restricts PATH to system/bundled components, isolates external-data locations and
asserts the imported library/interpreter come from the bundle. This tests independence
from the runner's Python installation; it is not qualification of every Windows
edition or SPICE-backed scenario.

Generated Python/Numba caches are excluded from the ZIP and immutable content
identities. Qualification explicitly enables ordinary bytecode writes, executes
the real `LunarOD.cmd --check` launcher in the relocated directory, and verifies
the distributed files again after imports and desktop use. Cache creation must
not produce a false integrity failure; source/binary changes still fail.

Run locally from a clean committed Windows/Python 3.13 checkout:

```text
python -m pip install -r requirements/windows.txt -r requirements/release.txt
python -m pip install -e .
python tools/prepare_release.py
```

Qualified upload assets are isolated under `outputs/qualified-release/` so an
older version cannot accidentally enter a new publication. Build folders are
never committed. A failed build preserves its evidence; use a fresh clean
checkout/output directory for a new attempt. No PyPI credentials or publishing
step is configured.

Use a new version for changed assets; keep previous release assets immutable.
The Windows source ZIP preserves the checkout bytes of scientific records; a
plain GitHub auto-generated archive may apply different line endings. Use the
named, checksummed Windows source asset for the reference byte-identity checks.

## Zenodo DOI: owner account step

**Deferred at the repository owner's request.** The workflow does not attempt to
create an archive or invent a DOI.

Citation metadata is already present. A DOI must come from an actual archive
record; it cannot be generated locally or guessed.

1. Sign in to Zenodo with the repository owner's account.
2. Authorize its GitHub integration for this public repository.
3. Enable archiving for the repository, then create a new release.
4. Inspect the archived record, authors, license and version.
5. Add the verified version/concept DOI to `CITATION.cff` and the README, with a
   clear distinction between a specific version and the evolving software concept.

Until that account authorization and successful archive exist, the repository
states **no DOI assigned**. See the
[official GitHub procedure](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).
