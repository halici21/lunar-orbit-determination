"""GRAIL / lunar spherical-harmonic gravity model loader (Phase 12).

Reads spherical-harmonic coefficient files into the Phase 11B
:class:`~lunar_od.gravity_harmonics.SphericalHarmonicGravityModel` in one
canonical form: **SI units + fully-normalized Cbar/Sbar**, with the model
carrying its OWN gravitational parameter and reference radius.

Supported formats
-----------------
- **PDS SHADR ASCII table** (``*.tab`` / ``*.sha``; GRGM / GL GRAIL products).
  Header record: ``R_ref[km], GM[km^3/s^2], GM_sigma, degree, order,
  normalization_flag, ref_lon, ref_lat`` followed by ``n, m, C, S, sC, sS``
  records.  Native units are km / km^3/s^2 and are converted to SI here.
- **ICGEM ``.gfc``** (GFZ mirror of the GRAIL models).  Key-value header up to
  ``end_of_head`` (``earth_gravity_constant``/``gravity_constant`` [m^3/s^2],
  ``radius`` [m], ``max_degree``, ``norm``) followed by ``gfc n m C S ...``
  records.  Time-variable records (``gfct``/``trnd``/``asin``/``acos``/``dot``)
  are skipped (static field only) and counted in the metadata.

Both formats may declare unnormalized coefficients; those are converted to
fully-normalized via ``Cbar_nm = C_nm / N_nm`` with
``N_nm = sqrt((2 - delta_m0)(2n+1)(n-m)!/(n+m)!)`` (evaluated in log space).

GM / R_ref pairing rule
-----------------------
The loaded model's coefficients are only meaningful together with the GM and
reference radius of the SAME file (GRAIL models use R_ref = 1738.0 km, which is
NOT ``constants.R_MOON_M`` = 1737.4 km).  This loader therefore always returns
the file's own GM/R_ref; ``constants.MU_MOON_M3S2`` is used ONLY as a sanity
reference for validation, never as a substitute.

Metadata validation (hard failures, no silent fixes)
----------------------------------------------------
A file is rejected with ``ValueError`` when any of these fail:
GM within 0.1% of the lunar GM; R_ref within [1730, 1745] km; a (2, 0)
coefficient present (within the requested truncation) and ``Cbar20 < 0``
(oblate); ``C00 ~ 1`` and degree-1 terms ~ 0 when present (centre of mass at
the origin); coefficient degrees consistent with the declared maximum; no
duplicate (n, m) records; no sine term on order m = 0.

Data management
---------------
Large model files are NOT committed; they live in the first existing directory
of: ``LUNAR_OD_GRAVITY_DIR`` (environment variable), ``~/Documents/mice/gravity``,
``<python_port>/data/gravity`` (git-ignored).  Tests use tiny synthetic
fixtures under ``tests/fixtures/gravity/``.  Every load records the SHA-256 of
the file bytes in ``model.metadata['sha256']`` for reproducibility.

Scope: pure data layer.  No frame handling (epoch-dependent MOON_PA is Phase
12/13), no dynamics integration, no downloads.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import MU_MOON_M3S2
from .gravity_harmonics import SphericalHarmonicGravityModel

__all__ = [
    "load_lunar_gravity_model",
    "describe_model",
    "default_gravity_candidates",
    "resolve_gravity_dir",
]

# Validation thresholds (module docstring rules).
_GM_REL_TOL = 1e-3
_R_REF_MIN_M = 1.730e6
_R_REF_MAX_M = 1.745e6
_C00_TOL = 1e-6
_DEG1_TOL = 1e-7
# Loading a very high-degree file untruncated is usually a mistake; warn.
_FULL_DEGREE_WARN = 128

_TIME_VARIABLE_KEYS = {"gfct", "trnd", "asin", "acos", "dot"}


# ---------------------------------------------------------------------------
# Gravity data directory resolution (mirror of spice_loader kernel resolution)
# ---------------------------------------------------------------------------
def default_gravity_candidates() -> list[Path]:
    """Gravity-model directories in priority order (env var first)."""
    package_root = Path(__file__).resolve().parents[1]   # <python_port>
    candidates: list[Path] = []
    env_dir = os.environ.get("LUNAR_OD_GRAVITY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path.home() / "Documents" / "mice" / "gravity",
            package_root / "data" / "gravity",
        ]
    )
    return candidates


def resolve_gravity_dir(
    candidates: Iterable[os.PathLike[str] | str] | None = None,
) -> Path:
    """Find the first existing gravity-model directory."""
    search_dirs = [Path(item) for item in (candidates or default_gravity_candidates())]
    for gravity_dir in search_dirs:
        if gravity_dir.is_dir():
            return gravity_dir
    searched = "\n".join(str(item) for item in search_dirs)
    raise FileNotFoundError(f"Gravity model directory not found. Searched:\n{searched}")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _float(token: str) -> float:
    """Parse a float tolerating Fortran 'D' exponents."""
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _norm_factor(n: int, m: int) -> float:
    """N_nm = sqrt((2 - delta_m0)(2n+1)(n-m)!/(n+m)!), in log space.

    Converts unnormalized to fully-normalized coefficients via
    ``Cbar_nm = C_nm / N_nm`` (for J2: Cbar20 = -J2/sqrt(5)).
    """
    k = 2.0 if m > 0 else 1.0
    log_ratio = math.lgamma(n - m + 1) - math.lgamma(n + m + 1)
    return math.exp(0.5 * (math.log(k * (2.0 * n + 1.0)) + log_ratio))


def _split_record(line: str) -> list[str]:
    """Split a SHADR record on commas (PDS) or whitespace (tolerant)."""
    if "," in line:
        return [tok for tok in (t.strip() for t in line.split(",")) if tok != ""]
    return line.split()


def _resolve_truncation(file_degree: int, nmax, mmax) -> tuple[int, int]:
    nmax_eff = file_degree if nmax is None else int(nmax)
    if nmax_eff > file_degree:
        raise ValueError(
            f"requested nmax={nmax_eff} exceeds the file's maximum degree {file_degree}."
        )
    if nmax_eff < 2:
        raise ValueError("nmax must be >= 2 (degree 0/1 carry no perturbation).")
    mmax_eff = nmax_eff if mmax is None else int(mmax)
    if not (0 <= mmax_eff <= nmax_eff):
        raise ValueError("mmax must satisfy 0 <= mmax <= nmax.")
    return nmax_eff, mmax_eff


def _collect_records(
    records: Iterable[tuple[int, int, float, float]],
    file_degree: int,
    nmax_eff: int,
    mmax_eff: int,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Truncate, and guard duplicates / inconsistent degrees / m=0 sine terms."""
    coeffs: dict[tuple[int, int], tuple[float, float]] = {}
    for n, m, c, s in records:
        if n > file_degree:
            raise ValueError(
                f"coefficient record degree n={n} exceeds the declared maximum "
                f"degree {file_degree}."
            )
        if m > n:
            raise ValueError(f"coefficient record has order m={m} > degree n={n}.")
        if m == 0 and s != 0.0:
            raise ValueError(f"nonzero sine coefficient on order m=0 (n={n}).")
        if n > nmax_eff or m > mmax_eff:
            continue
        if (n, m) in coeffs:
            raise ValueError(f"duplicate coefficient record for (n={n}, m={m}).")
        coeffs[(n, m)] = (c, s)
    return coeffs


# ---------------------------------------------------------------------------
# Format parsers -> raw model dict
# ---------------------------------------------------------------------------
def _parse_shadr(text: str, nmax, mmax) -> dict:
    """PDS SHADR ASCII table.  Native units: km and km^3/s^2."""
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("empty SHADR file.")
    header = _split_record(lines[0])
    if len(header) < 6:
        raise ValueError(
            "not a SHADR header: expected at least 6 fields "
            "(R_ref, GM, GM_sigma, degree, order, normalization_flag)."
        )
    try:
        r_ref_m = _float(header[0]) * 1.0e3          # km -> m
        gm_m3_s2 = _float(header[1]) * 1.0e9          # km^3/s^2 -> m^3/s^2
        file_degree = int(_float(header[3]))
        file_order = int(_float(header[4]))
        norm_flag = int(_float(header[5]))
    except ValueError as exc:
        raise ValueError(f"unparseable SHADR header: {exc}") from exc
    if norm_flag not in (0, 1):
        raise ValueError(f"unknown SHADR normalization flag {norm_flag} (expected 0 or 1).")

    nmax_eff, mmax_eff = _resolve_truncation(file_degree, nmax, mmax)

    def records():
        for idx, line in enumerate(lines[1:], start=2):
            parts = _split_record(line)
            if len(parts) < 4:
                raise ValueError(f"bad SHADR coefficient record on line {idx}: {line!r}")
            try:
                yield (int(_float(parts[0])), int(_float(parts[1])),
                       _float(parts[2]), _float(parts[3]))
            except ValueError as exc:
                raise ValueError(
                    f"bad SHADR coefficient record on line {idx}: {line!r}"
                ) from exc

    coeffs = _collect_records(records(), file_degree, nmax_eff, mmax_eff)
    return {
        "gm_m3_s2": gm_m3_s2,
        "r_ref_m": r_ref_m,
        "normalized": norm_flag == 1,
        "file_degree": file_degree,
        "file_order": file_order,
        "coeffs": coeffs,
        "nmax_eff": nmax_eff,
        "mmax_eff": mmax_eff,
        "skipped_time_variable": 0,
        "format": "shadr",
    }


def _parse_gfc(text: str, nmax, mmax) -> dict:
    """ICGEM .gfc.  Native units: m and m^3/s^2 (SI already)."""
    lines = text.splitlines()
    header: dict[str, str] = {}
    data_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("end_of_head"):
            data_start = idx + 1
            break
        parts = stripped.split()
        if parts[0].lower() == "begin_of_head":
            continue
        if len(parts) >= 2:
            header[parts[0].lower()] = parts[1]
    if data_start is None:
        raise ValueError("not an ICGEM .gfc file: missing 'end_of_head' marker.")

    gm_key = "earth_gravity_constant" if "earth_gravity_constant" in header else "gravity_constant"
    if gm_key not in header:
        raise ValueError(".gfc header is missing the gravity constant key.")
    if "radius" not in header:
        raise ValueError(".gfc header is missing the 'radius' key.")
    if "max_degree" not in header:
        raise ValueError(".gfc header is missing the 'max_degree' key.")
    gm_m3_s2 = _float(header[gm_key])
    r_ref_m = _float(header["radius"])
    file_degree = int(_float(header["max_degree"]))
    norm_value = header.get("norm", "fully_normalized").lower()
    if norm_value not in ("fully_normalized", "unnormalized"):
        raise ValueError(f"unknown .gfc 'norm' value: {norm_value!r}.")

    nmax_eff, mmax_eff = _resolve_truncation(file_degree, nmax, mmax)
    skipped = 0

    def records():
        nonlocal skipped
        for idx, line in enumerate(lines[data_start:], start=data_start + 1):
            parts = line.split()
            if not parts:
                continue
            key = parts[0].lower()
            if key in _TIME_VARIABLE_KEYS:
                skipped += 1
                continue
            if key != "gfc":
                raise ValueError(f"unknown .gfc record type {parts[0]!r} on line {idx}.")
            if len(parts) < 5:
                raise ValueError(f"bad .gfc coefficient record on line {idx}: {line!r}")
            try:
                yield (int(parts[1]), int(parts[2]), _float(parts[3]), _float(parts[4]))
            except ValueError as exc:
                raise ValueError(
                    f"bad .gfc coefficient record on line {idx}: {line!r}"
                ) from exc

    coeffs = _collect_records(records(), file_degree, nmax_eff, mmax_eff)
    return {
        "gm_m3_s2": gm_m3_s2,
        "r_ref_m": r_ref_m,
        "normalized": norm_value == "fully_normalized",
        "file_degree": file_degree,
        "file_order": file_degree,
        "coeffs": coeffs,
        "nmax_eff": nmax_eff,
        "mmax_eff": mmax_eff,
        "skipped_time_variable": skipped,
        "format": "icgem-gfc",
    }


# ---------------------------------------------------------------------------
# Canonicalization + validation
# ---------------------------------------------------------------------------
def _canonicalize(raw: dict) -> dict:
    """Convert coefficients to fully normalized (no-op if already normalized)."""
    if raw["normalized"]:
        return raw
    converted = {}
    for (n, m), (c, s) in raw["coeffs"].items():
        factor = _norm_factor(n, m)
        converted[(n, m)] = (c / factor, s / factor)
    raw = dict(raw)
    raw["coeffs"] = converted
    return raw


def _validate(raw: dict, source: str) -> None:
    gm = raw["gm_m3_s2"]
    r_ref = raw["r_ref_m"]
    coeffs = raw["coeffs"]
    rel_gm = abs(gm - MU_MOON_M3S2) / MU_MOON_M3S2
    if not (rel_gm < _GM_REL_TOL):
        raise ValueError(
            f"{source}: GM={gm:.6e} m^3/s^2 is not a lunar GM "
            f"(relative difference {rel_gm:.2e} vs lunar reference, tol {_GM_REL_TOL})."
        )
    if not (_R_REF_MIN_M <= r_ref <= _R_REF_MAX_M):
        raise ValueError(
            f"{source}: reference radius {r_ref:.1f} m outside the plausible "
            f"lunar range [{_R_REF_MIN_M:.0f}, {_R_REF_MAX_M:.0f}] m."
        )
    if (2, 0) not in coeffs:
        raise ValueError(f"{source}: no (2, 0) coefficient within the requested truncation.")
    cbar20 = coeffs[(2, 0)][0]
    if not (cbar20 < 0.0):
        raise ValueError(
            f"{source}: Cbar20={cbar20:.6e} must be negative for an oblate body "
            "(sign/normalization convention violated)."
        )
    if (0, 0) in coeffs and abs(coeffs[(0, 0)][0] - 1.0) > _C00_TOL:
        raise ValueError(f"{source}: C00={coeffs[(0, 0)][0]!r} differs from 1.")
    for m in (0, 1):
        if (1, m) in coeffs:
            c, s = coeffs[(1, m)]
            if abs(c) > _DEG1_TOL or abs(s) > _DEG1_TOL:
                raise ValueError(
                    f"{source}: degree-1 coefficient (1, {m}) = ({c!r}, {s!r}) is not ~0 "
                    "(centre of mass not at the origin)."
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_lunar_gravity_model(
    path: os.PathLike[str] | str,
    nmax: int | None = None,
    mmax: int | None = None,
    *,
    fmt: str = "auto",
) -> SphericalHarmonicGravityModel:
    """Load a lunar spherical-harmonic model file into canonical form.

    Parameters
    ----------
    path : coefficient file (SHADR ``.tab``/``.sha`` or ICGEM ``.gfc``).
    nmax, mmax : truncation applied AT LOAD TIME (records beyond it are
        skipped).  ``nmax=None`` loads the file's full degree (a warning is
        emitted above degree 128 -- usually you want a truncation).
    fmt : ``"auto"`` (by extension, content sniff as fallback), ``"shadr"``
        or ``"gfc"``.

    Returns the model with the FILE's own GM and reference radius (SI) and
    fully-normalized coefficients; provenance (sha256, format, native
    normalization, declared degree, truncation) is in ``model.metadata``.
    """
    file_path = Path(path)
    raw_bytes = file_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="replace")

    fmt = fmt.lower()
    if fmt == "auto":
        suffix = file_path.suffix.lower()
        if suffix in (".tab", ".sha"):
            fmt = "shadr"
        elif suffix == ".gfc":
            fmt = "gfc"
        else:
            # Content sniff: gfc files start with alphabetic header keys,
            # SHADR files start with a numeric header record.
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            fmt = "gfc" if first[:1].isalpha() else "shadr"
    if fmt not in ("shadr", "gfc"):
        raise ValueError(f"unknown gravity file format {fmt!r} (expected 'shadr' or 'gfc').")

    raw = _parse_shadr(text, nmax, mmax) if fmt == "shadr" else _parse_gfc(text, nmax, mmax)
    if nmax is None and raw["file_degree"] > _FULL_DEGREE_WARN:
        warnings.warn(
            f"loading full degree {raw['file_degree']} from {file_path.name}; "
            "pass nmax= to truncate.",
            stacklevel=2,
        )
    raw = _canonicalize(raw)
    _validate(raw, file_path.name)

    nmax_eff = raw["nmax_eff"]
    mmax_eff = raw["mmax_eff"]
    cbar = np.zeros((nmax_eff + 1, nmax_eff + 1))
    sbar = np.zeros((nmax_eff + 1, nmax_eff + 1))
    for (n, m), (c, s) in raw["coeffs"].items():
        cbar[n, m] = c
        sbar[n, m] = s

    metadata = {
        "source_file": str(file_path),
        "sha256": sha256,
        "format": raw["format"],
        "native_normalization": "fully_normalized" if raw["normalized"] else "unnormalized",
        "file_degree": raw["file_degree"],
        "file_order": raw["file_order"],
        "records_loaded": len(raw["coeffs"]),
        "skipped_time_variable": raw["skipped_time_variable"],
        "requested_nmax": nmax,
        "requested_mmax": mmax,
    }
    return SphericalHarmonicGravityModel(
        mu_m3_s2=raw["gm_m3_s2"],
        r_ref_m=raw["r_ref_m"],
        cbar=cbar,
        sbar=sbar,
        nmax=nmax_eff,
        mmax=mmax_eff,
        frame="MOON_PA",
        metadata=metadata,
    )


def describe_model(model: SphericalHarmonicGravityModel) -> str:
    """One-line provenance string for logs and reports."""
    meta = model.metadata
    return (
        f"{Path(meta.get('source_file', '?')).name} [{meta.get('format', '?')}] "
        f"sha256={str(meta.get('sha256', '?'))[:12]} "
        f"native={meta.get('native_normalization', '?')} "
        f"file_degree={meta.get('file_degree', '?')} "
        f"loaded nmax={model.nmax} mmax={model.mmax} "
        f"GM={model.mu_m3_s2:.9e} m^3/s^2 R_ref={model.r_ref_m:.1f} m "
        f"frame={model.frame}"
    )
