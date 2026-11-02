"""WP1.2 (DYN-01, CFG-04, X-01): planetary ephemeris and GMs from one generation.

Before WP1.2 the runtime used the DE440 planetary ephemeris with gm_de431 GMs (the
MATLAB snapshot), in the SPICE pool, the constants module and the scenario runners.
Oracles: the NAIF gm_de440.tpc / gm_de431.tpc kernel pools themselves (BODY301,
BODY399, BODY10 GM) and the snapshot fixture the DE421 profile must reproduce bit for bit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lunar_od import constants
from lunar_od.ephemeris_generation import (
    GRAVITATIONAL_PARAMETERS,
    check_generation_consistency,
    gravitational_parameters,
)
from lunar_od.spice_loader import GM_KERNELS, required_kernels, resolve_kernel_dir

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "spice_snapshots.json"


def test_each_ephemeris_loads_its_own_gm_kernel():
    assert "gm_de440.tpc" in required_kernels("DE440") and "gm_de431.tpc.txt" not in required_kernels("DE440")
    assert "gm_de431.tpc.txt" in required_kernels("DE421") and "gm_de440.tpc" not in required_kernels("DE421")
    assert GM_KERNELS == {"DE440": "gm_de440.tpc", "DE421": "gm_de431.tpc.txt"}


def _kernel_gm(path: Path, code: int) -> float:
    """BODY<code>_GM from a NAIF text kernel, km^3/s^2 -> m^3/s^2 (no SPICE pool used)."""
    match = re.search(rf"BODY{code}_GM\s*=\s*\(\s*([0-9.+\-EeDd]+)\s*\)", path.read_text(encoding="utf-8"))
    return float(match.group(1).replace("D", "E")) * 1e9


@pytest.mark.parametrize("ephemeris", ["DE440", "DE421"])
def test_constants_match_the_generation_gm_kernel(ephemeris):
    try:
        kernel_dir = resolve_kernel_dir()
    except FileNotFoundError:
        pytest.skip("SPICE kernel directory not available")
    path = Path(kernel_dir) / GM_KERNELS[ephemeris]
    if not path.is_file():
        pytest.skip(f"{path.name} not installed")
    kernel = tuple(_kernel_gm(path, code) for code in (301, 399, 10))
    assert GRAVITATIONAL_PARAMETERS[ephemeris] == pytest.approx(kernel, rel=1e-15, abs=0.0)


def test_default_constants_are_de440_and_the_de421_profile_reproduces_the_snapshot():
    assert (constants.MU_MOON_M3S2, constants.MU_EARTH_M3S2, constants.MU_SUN_M3S2) == GRAVITATIONAL_PARAMETERS["DE440"]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    snapshot = (float(fixture["initial_state"]["mu_moon_m3_s2"]),
                float(fixture["constants"]["mu_earth_km3_s2"][0] if isinstance(fixture["constants"]["mu_earth_km3_s2"], list)
                      else fixture["constants"]["mu_earth_km3_s2"]) * 1e9,
                float(fixture["constants"]["mu_sun_km3_s2"][0] if isinstance(fixture["constants"]["mu_sun_km3_s2"], list)
                      else fixture["constants"]["mu_sun_km3_s2"]) * 1e9)
    assert gravitational_parameters("DE421") == snapshot             # bit for bit
    rel = abs(GRAVITATIONAL_PARAMETERS["DE440"][0] - snapshot[0]) / snapshot[0]
    assert rel == pytest.approx(1.0666e-8, rel=1e-3)                  # the X-01 GM(Moon) gap


def test_mixed_generation_is_rejected(monkeypatch):
    monkeypatch.delenv("LUNAR_OD_PLANETARY_EPHEMERIS", raising=False)
    assert check_generation_consistency(*GRAVITATIONAL_PARAMETERS["DE440"]) == "DE440"
    with pytest.raises(ValueError, match="one generation"):
        check_generation_consistency(*GRAVITATIONAL_PARAMETERS["DE421"])
    assert check_generation_consistency(*GRAVITATIONAL_PARAMETERS["DE421"], ephemeris="DE421") == "DE421"
    monkeypatch.setenv("LUNAR_OD_PLANETARY_EPHEMERIS", "DE421")
    assert gravitational_parameters() == GRAVITATIONAL_PARAMETERS["DE421"]
