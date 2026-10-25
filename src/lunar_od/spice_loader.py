"""SPICE kernel loading helpers for the Python port.

The MATLAB project uses MICE from `Documents/mice` and loads kernels from
`Documents/mice/kernels`. In Python, the toolkit layer is `spiceypy`; only the
kernel directory and kernel file list are shared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

#: DE440 lunar principal-axis capability. These two files are an ATOMIC PAIR:
#: the frame definition declares ``MOON_PA_DE440`` while the binary PCK carries
#: its orientation data, and NEITHER IS SUFFICIENT ALONE. Measured behaviour
#: with only one of them present:
#:
#:   frame definition only -> SpiceFRAMEDATANOTFOUND, and generic ``MOON_PA``
#:                            stops resolving as well (a partial configuration
#:                            is strictly worse than neither file)
#:   binary PCK only       -> SpiceUNKNOWNFRAME
#:
#: Keep them together. Removing one silently degrades the lunar frame contract.
MOON_PA_DE440_KERNELS = (
    "moon_de440_220930.txt",
    "moon_pa_de440_200625.bpc",
)

#: Planetary ephemeris (Phase C4-I).  DE440 is the production default: the C4
#: survey measured DE440 - DE421 at 3.5-3.9 m in the Moon-centred Earth, an OD
#: effect of 4.8 sigma, and DE440 is what real DSN data, NASA's LRO/GRAIL
#: orbits and Orekit use.  DE421 stays selectable so every record computed with
#: it remains reproducible.  Selection order: the explicit ``ephemeris=``
#: argument, else the ``LUNAR_OD_PLANETARY_EPHEMERIS`` environment variable,
#: else ``DEFAULT_PLANETARY_EPHEMERIS``.
PLANETARY_EPHEMERIS_KERNELS = {
    "DE440": "de440.bsp",
    "DE421": "de421.bsp",
}
DEFAULT_PLANETARY_EPHEMERIS = "DE440"
PLANETARY_EPHEMERIS_ENV = "LUNAR_OD_PLANETARY_EPHEMERIS"

#: ORDER IS LOAD-BEARING, not cosmetic. Both ``moon_de440_220930.txt`` and
#: ``moon_080317.tf.txt`` define the generic ``MOON_PA`` alias, and SPICE lets
#: the LAST text kernel loaded win that name. Listing the DE440 pair FIRST
#: therefore leaves generic ``MOON_PA`` bound to the historical DE421
#: realization while both explicit frames stay resolvable, so adding DE440
#: capability changes no existing consumer's physics. Appending the pair at the
#: end instead silently rebinds generic ``MOON_PA`` to DE440 -- a ~7.6e-7
#: rotation change (~1.65 m at the lunar surface) that was measured breaking
#: the MATLAB SPICE snapshot fixture. ``test_de440_kernel_contract`` pins this.
_KERNEL_TEMPLATE = (
    "naif0012.tls.txt",
    # DE440 lunar PA (atomic pair -- see MOON_PA_DE440_KERNELS above). Loaded
    # BEFORE moon_080317.tf.txt so it adds MOON_PA_DE440 without capturing the
    # generic alias. High-order gravity selects its realization by EXPLICIT
    # versioned frame name and never relies on generic ``MOON_PA``.
    *MOON_PA_DE440_KERNELS,
    # The planetary ephemeris slot (de421.bsp before Phase C4-I).
    "@PLANETARY_EPHEMERIS@",
    "earth_assoc_itrf93.tf.txt",
    "moon_080317.tf.txt",
    "earth_2025_250826_2125_predict.bpc",
    "moon_pa_de421_1900-2050.bpc",
    "gm_de431.tpc.txt",
    "pck00010.tpc.txt",
)


def planetary_ephemeris(ephemeris: str | None = None) -> str:
    """The selected planetary ephemeris name (``"DE440"`` or ``"DE421"``)."""
    name = ephemeris if ephemeris is not None else os.environ.get(PLANETARY_EPHEMERIS_ENV)
    name = DEFAULT_PLANETARY_EPHEMERIS if name in (None, "") else str(name).strip().upper()
    if name not in PLANETARY_EPHEMERIS_KERNELS:
        raise ValueError(
            f"unknown planetary ephemeris {name!r}; choose one of "
            f"{tuple(PLANETARY_EPHEMERIS_KERNELS)}"
        )
    return name


def required_kernels(ephemeris: str | None = None) -> tuple[str, ...]:
    """The required kernel list with the selected planetary ephemeris."""
    bsp = PLANETARY_EPHEMERIS_KERNELS[planetary_ephemeris(ephemeris)]
    return tuple(bsp if k == "@PLANETARY_EPHEMERIS@" else k for k in _KERNEL_TEMPLATE)


#: The required kernel list as resolved at import (default or environment).
REQUIRED_KERNELS = required_kernels()

#: Explicit versioned lunar PA frames. The generic ``MOON_PA`` alias is
#: deliberately absent: with both DE421 and DE440 orientation kernels furnished
#: it binds by kernel load order, so scientific code must name a realization.
MOON_PA_DE440_FRAME = "MOON_PA_DE440"
MOON_PA_DE421_FRAME = "MOON_PA_DE421"


class LunarFrameKernelError(RuntimeError):
    """Raised when the explicit DE440 lunar PA realization is unusable."""


def require_moon_pa_de440(et_s: float) -> None:
    """Fail closed unless ``MOON_PA_DE440`` is resolvable at ``et_s``.

    Queries the EXPLICIT versioned frame. There is deliberately no fallback to
    the generic ``MOON_PA`` alias, to ``MOON_PA_DE421``, or to identity: a
    caller that needs the DE440 realization must get that realization or an
    error, never a silently different one.

    Parameters
    ----------
    et_s : SPICE ET (TDB seconds past J2000) at which to test the frame.

    Raises
    ------
    LunarFrameKernelError
        When the frame cannot be evaluated, with the underlying SPICE error
        preserved as the exception cause.
    """
    import spiceypy as spice

    try:
        spice.pxform("J2000", MOON_PA_DE440_FRAME, float(et_s))
    except Exception as exc:  # noqa: BLE001 - normalised into a project error
        raise LunarFrameKernelError(
            f"{MOON_PA_DE440_FRAME} is not resolvable at et={float(et_s)!r}. "
            "The DE440 lunar principal-axis kernels are missing or incomplete; "
            f"both files of the atomic pair are required: {MOON_PA_DE440_KERNELS}. "
            "No fallback to the generic MOON_PA alias or to MOON_PA_DE421 is "
            "performed, because either would silently supply a different "
            "physical realization."
        ) from exc


def default_kernel_candidates() -> list[Path]:
    """Return kernel directories in the same priority order as MATLAB."""
    project_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []

    env_kernel_dir = os.environ.get("LUNAR_OD_KERNEL_DIR")
    if env_kernel_dir:
        candidates.append(Path(env_kernel_dir))

    candidates.extend(
        [
            Path.home() / "Documents" / "mice" / "kernels",
            project_root / "kernels",
        ]
    )
    return candidates


def resolve_kernel_dir(candidates: Iterable[os.PathLike[str] | str] | None = None) -> Path:
    """Find the first existing SPICE kernel directory."""
    search_dirs = [Path(item) for item in (candidates or default_kernel_candidates())]
    for kernel_dir in search_dirs:
        if kernel_dir.is_dir():
            return kernel_dir

    searched = "\n".join(str(item) for item in search_dirs)
    raise FileNotFoundError(f"SPICE kernel directory not found. Searched:\n{searched}")


def required_kernel_paths(
    kernel_dir: os.PathLike[str] | str | None = None,
    ephemeris: str | None = None,
) -> list[Path]:
    """Return full paths for the required kernel list and validate existence."""
    resolved_dir = resolve_kernel_dir([kernel_dir]) if kernel_dir is not None else resolve_kernel_dir()
    kernel_paths = [resolved_dir / kernel_name for kernel_name in required_kernels(ephemeris)]

    missing = [path.name for path in kernel_paths if not path.is_file()]
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing required SPICE kernel file(s):\n{missing_list}")

    return kernel_paths


def load_spice_kernels(
    kernel_dir: os.PathLike[str] | str | None = None,
    clear: bool = True,
    ephemeris: str | None = None,
) -> list[Path]:
    """Load the required kernels with spiceypy and return the loaded paths.

    ``ephemeris`` selects the planetary ephemeris (see ``planetary_ephemeris``).
    """
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "spiceypy is required for Python SPICE calls. Install it before "
            "running dynamics, frame-transform, or measurement code."
        ) from exc

    kernel_paths = required_kernel_paths(kernel_dir, ephemeris)

    if clear:
        spice.kclear()

    for kernel_path in kernel_paths:
        spice.furnsh(str(kernel_path))

    return kernel_paths

