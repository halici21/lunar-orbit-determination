"""SPICE kernel loading helpers for the Python port.

The MATLAB project uses MICE from `Documents/mice` and loads kernels from
`Documents/mice/kernels`. In Python, the toolkit layer is `spiceypy`; only the
kernel directory and kernel file list are shared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

REQUIRED_KERNELS = (
    "naif0012.tls.txt",
    "de421.bsp",
    "earth_assoc_itrf93.tf.txt",
    "moon_080317.tf.txt",
    "earth_2025_250826_2125_predict.bpc",
    "moon_pa_de421_1900-2050.bpc",
    "gm_de431.tpc.txt",
    "pck00010.tpc.txt",
)


def default_kernel_candidates() -> list[Path]:
    """Return kernel directories in the same priority order as MATLAB."""
    project_root = Path(__file__).resolve().parents[2]
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


def required_kernel_paths(kernel_dir: os.PathLike[str] | str | None = None) -> list[Path]:
    """Return full paths for the required kernel list and validate existence."""
    resolved_dir = resolve_kernel_dir([kernel_dir]) if kernel_dir is not None else resolve_kernel_dir()
    kernel_paths = [resolved_dir / kernel_name for kernel_name in REQUIRED_KERNELS]

    missing = [path.name for path in kernel_paths if not path.is_file()]
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing required SPICE kernel file(s):\n{missing_list}")

    return kernel_paths


def load_spice_kernels(kernel_dir: os.PathLike[str] | str | None = None, clear: bool = True) -> list[Path]:
    """Load the required kernels with spiceypy and return the loaded paths."""
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "spiceypy is required for Python SPICE calls. Install it before "
            "running dynamics, frame-transform, or measurement code."
        ) from exc

    kernel_paths = required_kernel_paths(kernel_dir)

    if clear:
        spice.kclear()

    for kernel_path in kernel_paths:
        spice.furnsh(str(kernel_path))

    return kernel_paths

