"""Shared scenario of the WP5.1 Orekit oracle (no Orekit import).

Every number both tools need comes from here or from the pinned ``scenario.json``
in ``tests/fixtures/external_oracle``. Units: SI (m, m/s, s). ``t`` is TDB seconds
since ``EPOCH_TDB``.

Predeclaration: ``studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "external_oracle"
PREDECLARATION = ROOT / "studies" / "phase17" / "configs" / "wp5_1_orekit_oracle_predeclaration.json"
RUN_DIR = ROOT / "outputs" / "wp5_1_oracle_runs"  # gitignored scratch (ladder rungs)

EPOCH_TDB = "2025-03-01T00:00:00"
ARC_S = 86400.0
OUTPUT_STEP_S = 600.0
K_SRP = 0.02  # m^2/kg, C_R * A / m
STATION_NAME = "Canberra DSN"
DYNAMICS_CASES = ("D1a", "D1b", "D2-50", "D2-100", "D3")

# measurement geometry (predeclaration measurement_cases)
T_MEAS = np.arange(3600.0, 10800.0 + 0.5, 60.0)
TC_S = 60.0
TRAJ_GRID = np.arange(0.0, 14400.0 + 0.5, 10.0)
STATION_GRID = np.arange(3000.0, 11400.0 + 0.5, 10.0)


def output_grid() -> np.ndarray:
    return np.arange(0.0, ARC_S + 0.5 * OUTPUT_STEP_S, OUTPUT_STEP_S)


def ephemeris_of(case: str) -> str:
    return "DE421" if case == "D1b" else "DE440"


def harmonic_degree(case: str) -> int | None:
    return int(case.split("-")[1]) if case.startswith("D2") else None


def kernel_dir() -> Path:
    from lunar_od.spice_loader import default_kernel_candidates

    for cand in default_kernel_candidates():
        if (Path(cand) / "de440.bsp").is_file():
            return Path(cand)
    raise FileNotFoundError("no SPICE kernel directory with de440.bsp (set LUNAR_OD_KERNEL_DIR)")


def gravity_tab() -> Path:
    from lunar_od.project_paths import gravity_data_root

    return gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"


def load_kernels(ephemeris: str = "DE440") -> None:
    """lunar_od's own kernel set for the named planetary ephemeris (manifest-verified)."""
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(kernel_dir(), clear=True, ephemeris=ephemeris)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scenario() -> dict:
    """The trial's scenario, rebuilt from lunar_od at the current commit."""
    import spiceypy as spice
    from lunar_od.config import range_rate_stations
    from lunar_od.constants import (AU_M, MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2, R_MOON_M, R_SUN_M,
                                    SOLAR_PRESSURE_1AU_N_M2)
    from lunar_od.orbit import lunar_initial_state_mci

    load_kernels("DE440")
    et0 = float(spice.str2et(EPOCH_TDB.replace("T", " ") + " TDB"))
    sx = np.asarray(spice.sxform("MOON_PA_DE440", "J2000", et0))
    _, _, x0 = lunar_initial_state_mci(R_MOON_M, 100e3, 0.0, np.pi / 2, 0.0, 0.0, 0.0, MU_MOON_M3S2, sx)
    st = next(s for s in range_rate_stations() if s.name == STATION_NAME)
    return {
        "epoch_tdb": EPOCH_TDB,
        "et0": et0,
        "arc_s": ARC_S,
        "output_step_s": OUTPUT_STEP_S,
        "state0_mci_j2000_m_mps": [float(v) for v in x0],
        "gm_m3_s2": {"MOON": MU_MOON_M3S2, "EARTH": MU_EARTH_M3S2, "SUN": MU_SUN_M3S2},
        "gm_source": "gm_de431 (lunar_od.constants MU_*), overridden in Orekit",
        "k_srp_m2_per_kg": K_SRP,
        "srp": {"P1AU_N_m2": SOLAR_PRESSURE_1AU_N_M2, "AU_m": AU_M, "R_sun_m": R_SUN_M, "R_occ_m": R_MOON_M},
        "station": {"name": st.name, "lat_deg": st.lat_deg, "lon_deg": st.lon_deg, "alt_m": st.alt_m},
        "measurements": {"t_meas_s": [T_MEAS[0], T_MEAS[-1], 60.0], "tc_s": TC_S,
                         "trajectory_grid_s": [TRAJ_GRID[0], TRAJ_GRID[-1], 10.0],
                         "station_grid_s": [STATION_GRID[0], STATION_GRID[-1], 10.0]},
    }


def load_scenario() -> dict:
    return json.loads((FIXTURE_DIR / "scenario.json").read_text(encoding="utf-8"))


def load_predeclaration() -> dict:
    return json.loads(PREDECLARATION.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# lossless CSV I/O (repr of float64 round-trips exactly)
# ---------------------------------------------------------------------------
def write_csv(path: Path, header: list[str], rows: np.ndarray, comment: str = "") -> None:
    rows = np.asarray(rows, dtype=float)
    lines = [f"# {c}" for c in comment.splitlines() if c] + [",".join(header)]
    lines += [",".join(repr(float(v)) for v in r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    header = lines[0].split(",")
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=float)
    return header, data
