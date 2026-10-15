# Source releases, library wheels and DOI

## Distribution types

- **Windows source preview:** tracked source and launch/setup scripts; Python and
  network access for dependencies are required. No standalone EXE is included.
- **Library wheel:** installs `lunar_od`; the desktop, examples, research archive
  and SPICE kernels are not bundled. Run tutorials from a source checkout/ZIP.
- **Future desktop installer:** separate qualification is required for the Python
  runtime, QProcess children, backend imports and external-data setup.

## Release gate

1. Select a clean committed source revision; require public CI and documentation builds.
2. Run the data-free quickstart and preserve the exact summary/source identity.
3. Check preserved scientific bytes and package dependencies; run the relevant research regressions.
4. Build the wheel and test imports in a fresh environment outside the checkout.
5. Extract the source ZIP to a clean path containing spaces, install its reference
   dependencies/package, and check launcher imports and the tutorial.
6. Keep kernels, local runs, environments and Git history out of release assets.
7. Publish source ZIP, wheel and SHA256SUMS with exact commit, scope and known limits.

Use a new version for changed assets; keep previous release assets immutable.
The Windows source ZIP preserves the checkout bytes of scientific records; a
plain GitHub auto-generated archive may apply different line endings. Use the
named, checksummed Windows source asset for the reference byte-identity checks.

## Zenodo DOI: owner account step

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
