"""WP7.5 / TST-06: lunar_od.constants against the published standards it cites.

Oracles are the defining documents (literal values typed from them), not this
module's own numbers; GM is checked against the SPICE kernel the runtime loads.
CFG-11: every module-level speed-of-light declaration must equal the SI value.
"""
from __future__ import annotations

import math

import pytest

import lunar_od.constants as K


def test_si_and_iau_exact_constants():
    assert K.SPEED_OF_LIGHT_M_S == 299_792_458.0          # SI, exact
    assert K.AU_M == 149_597_870_700.0                    # IAU 2012 Resolution B2, exact


def test_iau_2015_b3_nominal_solar_values():
    assert K.R_SUN_M == 695_700_000.0                     # nominal solar radius
    assert K.SOLAR_IRRADIANCE_1AU_W_M2 == 1361.0          # nominal TSI
    assert K.SOLAR_PRESSURE_1AU_N_M2 == pytest.approx(4.5398e-6, rel=1e-4)
    assert K.SOLAR_PRESSURE_1AU_N_M2 == K.SOLAR_IRRADIANCE_1AU_W_M2 / K.SPEED_OF_LIGHT_M_S


def test_earth_j2_is_egm96_unnormalized_c20_with_its_reference_radius():
    c20_bar_egm96 = -0.484165371736e-3                    # EGM96 normalized C(2,0)
    assert K.J2_EARTH_UNNORMALIZED == pytest.approx(-math.sqrt(5.0) * c20_bar_egm96, rel=1e-8)
    assert K.R_EARTH_J2_REF_M == 6_378_136.3              # EGM96 reference radius


def test_moon_radius_is_pck00010_mean_radius():
    assert K.R_MOON_M == 1_737_400.0


def test_moon_j2_and_radius_are_the_propagator_pair():
    from lunar_od import dynamics

    assert dynamics.MOON_J2 == K.J2_MOON_UNNORMALIZED
    assert dynamics.MOON_R_M == K.R_MOON_M


@pytest.mark.parametrize("module, name", [
    ("lunar_od.measurements", "C_LIGHT_MPS"),
    ("lunar_od.media", "C_LIGHT_MPS"),
    ("lunar_od.radiometrics", "C_LIGHT_MPS"),
    ("lunar_od.relativistic_light_time", "C_LIGHT_MPS"),
    ("lunar_od.rf.link_budget", "SPEED_OF_LIGHT_M_S"),
])
def test_every_speed_of_light_declaration_agrees(module, name):
    import importlib

    assert getattr(importlib.import_module(module), name) == K.SPEED_OF_LIGHT_M_S


@pytest.fixture(scope="module")
def spice_pool():
    spice = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels

    try:
        load_spice_kernels()
    except FileNotFoundError as exc:
        pytest.skip(f"SPICE kernels unavailable: {exc}")
    return spice


# WP1.2: the default (DE440) kernel set furnishes gm_de440.tpc; the legacy
# MU_*_M3S2 names keep their gm_de431 values for frozen baselines.
@pytest.mark.parametrize("body, name", [
    ("MOON", "MU_MOON_DE440_M3S2"),
    ("EARTH", "MU_EARTH_DE440_M3S2"),
    ("SUN", "MU_SUN_DE440_M3S2"),
])
def test_gm_equals_loaded_gm_kernel(spice_pool, body, name):
    """The DE440 constants match the furnished GM kernel exactly (km^3/s^2 -> m^3/s^2)."""
    (gm_km3_s2,) = spice_pool.bodvrd(body, "GM", 1)[1]
    assert getattr(K, name) == pytest.approx(gm_km3_s2 * 1e9, rel=1e-15)
