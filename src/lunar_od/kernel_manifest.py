"""SPICE kernel identity manifest (WP7.4).

Every kernel :func:`lunar_od.spice_loader.load_spice_kernels` furnishes is
checked against the SHA-256 recorded here before it is loaded, so a replaced,
truncated or re-downloaded-but-different kernel fails closed instead of changing
the physics silently (CFG-04).  Text kernels are hashed with CRLF normalised to
LF: SPICE reads both, and a Windows checkout of an LF kernel is the same kernel.

The hashes were recorded from the validated local kernel set on 2026-11-01
(the set every full regression and Phase 17 record of that date ran on).  The
source is the NAIF generic_kernels path each file was obtained from; local
names keep the project's historical ``.txt`` suffix on text kernels.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

NAIF_GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"

#: name -> (sha256, NAIF source URL).
KERNEL_MANIFEST: dict[str, tuple[str, str]] = {
    "naif0012.tls.txt": (
        "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
        NAIF_GENERIC + "lsk/naif0012.tls"),
    "moon_de440_220930.txt": (
        "73eb6b216c06a27c3419c4cfeaded7ffce46a714e3d4f9f5142dda51c0710f76",
        NAIF_GENERIC + "fk/satellites/moon_de440_220930.tf"),
    "moon_pa_de440_200625.bpc": (
        "60cd55aa401ea2ea97360636f567554bfe4e37bb829f901b4460a455dfaf783f",
        NAIF_GENERIC + "pck/moon_pa_de440_200625.bpc"),
    "de440.bsp": (
        "a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5",
        NAIF_GENERIC + "spk/planets/de440.bsp"),
    "de421.bsp": (
        "08b20db2ae22488650641c5a9033e5bfda4b1c4b440cfeaf20f621cfa18ecdb3",
        NAIF_GENERIC + "spk/planets/de421.bsp"),
    "earth_assoc_itrf93.tf.txt": (
        "aab7bbc19b8a69bad11988ee1b4812a3963812a03a029c2776863e680719b336",
        NAIF_GENERIC + "fk/planets/earth_assoc_itrf93.tf"),
    "moon_080317.tf.txt": (
        "78732477b96f9863e7b0d65bcee3c22b8707ca5ed0db56d1173319cb2e8c7993",
        NAIF_GENERIC + "fk/satellites/moon_080317.tf"),
    "earth_2025_250826_2125_predict.bpc": (
        "f5ed4d1f460dbcf40bf24f657ab9021718048e18fb723093bfd229066b4e3a49",
        NAIF_GENERIC + "pck/earth_2025_250826_2125_predict.bpc"),
    # F2 WP2.4: the high-precision Earth PCK of the Earth-orientation profiles (spice_loader), by its versioned
    # NAIF name; last datum 2026-11-02, downloaded 2026-11-03 with the owner's approval.
    "earth_000101_270128_261102.bpc": (
        "feed146c1ba6fba13f1e579371d34b67f6c134612d158db27cec019892bca5ab",
        NAIF_GENERIC + "pck/earth_000101_270128_261102.bpc"),
    # The local copy of NAIF's rolling high-precision file (earth_000101_260419_260121.bpc, last datum 2026-01-21)
    # that the F1/F2 LRO study scripts furnish by this name; recorded 2026-11-03 so their records stay traceable.
    "earth_latest_high_prec.bpc": (
        "99ddb26ac25e1ddc0fd98083424801d9c648a342f01b033bf1695591cfae2ec5",
        NAIF_GENERIC + "pck/earth_latest_high_prec.bpc"),
    "moon_pa_de421_1900-2050.bpc": (
        "656f90616403d75a75f0cd6c8830fc5b44f8cb4facb5ccb8915e752b397520cf",
        NAIF_GENERIC + "pck/moon_pa_de421_1900-2050.bpc"),
    "gm_de431.tpc.txt": (
        "15756c162151853f329d473fa00c38d35e62d83c3e25ddceece6516bbd98738b",
        NAIF_GENERIC + "pck/gm_de431.tpc"),
    # WP1.2 (DE440 GMs with the DE440 ephemeris); recorded 2026-11-02 from the local kernel.
    "gm_de440.tpc": (
        "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140",
        NAIF_GENERIC + "pck/gm_de440.tpc"),
    "pck00010.tpc.txt": (
        "59468328349aa730d18bf1f8d7e86efe6e40b75dfb921908f99321b3a7a701d2",
        NAIF_GENERIC + "pck/pck00010.tpc"),
}

_BINARY_SUFFIXES = (".bsp", ".bpc", ".bc", ".bds")
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


class KernelIntegrityError(RuntimeError):
    """A kernel is not in the manifest or its content differs from the recorded hash.

    Deliberately not a ``FileNotFoundError``: callers that skip on missing kernels
    must not skip on a wrong one.
    """


def kernel_sha256(path: os.PathLike[str] | str) -> str:
    """SHA-256 of a kernel (text kernels with CRLF normalised to LF), cached by size and mtime."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    if path.name.lower().endswith(_BINARY_SUFFIXES):
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    else:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def verify_kernels(paths) -> list[dict[str, str]]:
    """Check every path against :data:`KERNEL_MANIFEST`; return the identity records.

    Raises :class:`KernelIntegrityError` naming every unknown or mismatched file.
    """
    records, problems = [], []
    for path in (Path(p) for p in paths):
        expected = KERNEL_MANIFEST.get(path.name)
        got = kernel_sha256(path)
        if expected is None:
            problems.append(f"{path.name}: not in the kernel manifest (sha256 {got})")
            continue
        if got != expected[0]:
            problems.append(f"{path.name}: sha256 {got}, manifest {expected[0]} ({expected[1]})")
            continue
        records.append({"name": path.name, "sha256": got, "source": expected[1]})
    if problems:
        raise KernelIntegrityError("SPICE kernel identity check failed:\n" + "\n".join(problems))
    return records
