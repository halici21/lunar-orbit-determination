"""F3 WP3.2 S5: hold-out replication of the WP3.1 A2 materiality classification (gate S5_G1, S5 part).

Checks the population rule (strata, geometry, independence) and the term wiring of the driver against the
predeclaration studies/phase17/configs/wp3_2_s5_replication_predeclaration.json. No survey number is computed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
sys.path.insert(0, str(SCRIPTS))

PRE = ROOT / "studies" / "phase17" / "configs" / "wp3_2_s5_replication_predeclaration.json"

import wp3_2_s5_population as POP  # noqa: E402
import wp3_2_s5_replication as R  # noqa: E402


def _frame(seed: int = 3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=3)
    p = rng.normal(size=3)
    return s / np.linalg.norm(s), p / np.linalg.norm(p)


def test_predeclaration_names_the_s5_outputs():
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    out = pre["outputs"]
    assert out["population_script"].endswith("wp3_2_s5_population.py")
    assert out["driver"].endswith("wp3_2_s5_replication.py")
    assert out["tests"] == "tests/test_wp3_2_s5_replication.py"
    assert pre["population"]["seed"].startswith("numpy.random.default_rng(20261127)")
    assert POP.SEED == 20261127


def test_strata_and_altitude_bounds_match_the_predeclaration():
    assert POP.STRATA_DEG == ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))
    assert POP.ALTITUDE_RANGE_M == (50e3, 200e3)
    assert POP.ECLIPSE_OFFSET_RANGE_S == (6 * 3600.0, 18 * 3600.0)
    assert POP.WINDOW_UTC == ("2027-04-01T00:00:00", "2029-12-31T00:00:00")


@pytest.mark.parametrize("beta_deg", [0.0, 12.5, -44.0, 59.9, 89.0])
@pytest.mark.parametrize("omega", [0.0, 1.3, 4.0])
def test_orbit_normal_has_the_drawn_beta(beta_deg, omega):
    s, p = _frame()
    n = POP.orbit_normal(math.radians(beta_deg), omega, s, p)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-14
    assert abs(math.degrees(math.asin(float(np.dot(n, s)))) - beta_deg) < 1e-9


def test_initial_state_is_circular_at_the_altitude_and_in_the_plane():
    from lunar_od.constants import R_MOON_M
    s, p = _frame(5)
    n = POP.orbit_normal(math.radians(20.0), 2.0, s, p)
    mu = 4.9028e12
    x0 = POP.initial_state(n, p, 1.1, 123456.0, mu)
    r, v = x0[:3], x0[3:]
    assert abs(np.linalg.norm(r) - (R_MOON_M + 123456.0)) < 1e-6
    assert abs(np.linalg.norm(v) - math.sqrt(mu / np.linalg.norm(r))) < 1e-9
    assert abs(float(np.dot(r, v))) < 1e-6 * np.linalg.norm(r) * np.linalg.norm(v)
    assert abs(float(np.dot(np.cross(r, v) / np.linalg.norm(np.cross(r, v)), n)) - 1.0) < 1e-12


def test_cell_draw_respects_the_stratum_and_ranges():
    rng = np.random.default_rng(1)
    for k, (lo, hi) in enumerate(POP.STRATA_DEG):
        for _ in range(50):
            d = POP.draw_cell(rng, k)
            assert lo <= abs(d["beta_deg"]) < hi
            assert POP.ALTITUDE_RANGE_M[0] <= d["altitude_m"] <= POP.ALTITUDE_RANGE_M[1]
            assert 0.0 <= d["omega_rad"] < 2 * math.pi and 0.0 <= d["u_rad"] < 2 * math.pi


def _valid_population():
    cells = [dict(name=f"H{e}xS{k}", epoch=f"H{e}", altitude_m=60e3 + 1e3 * k) for e in range(1, 7)
             for k in range(1, 4)]
    epochs = {f"H{e}": dict(t0_utc=f"2028-0{e}-10T00:00:00", et0=1e7 * e, kind=kind, t_ge_utc=tge)
              for e, kind, tge in ((1, "NOMINAL", None), (2, "NOMINAL", None), (3, "TOTAL", "2028-03-10T05:00:00"),
                                   (4, "TOTAL", "2028-04-10T05:00:00"), (5, "NON_TOTAL", "2028-05-10T05:00:00"),
                                   (6, "NON_TOTAL", "2028-06-10T05:00:00"))}
    return dict(epochs=epochs, cells=cells)


def test_independence_check_passes_a_fresh_population():
    assert POP.independence(_valid_population())["PASS"]


def test_independence_check_refuses_a_wp3_1_altitude_or_an_early_epoch_or_close_epochs():
    pop = _valid_population()
    pop["cells"][4]["altitude_m"] = 100e3
    assert not POP.independence(pop)["PASS"]
    pop = _valid_population()
    pop["epochs"]["H1"]["t0_utc"] = "2027-03-02T00:00:00"
    assert not POP.independence(pop)["PASS"]
    pop = _valid_population()
    pop["epochs"]["H2"]["et0"] = pop["epochs"]["H1"]["et0"] + 86400.0
    assert not POP.independence(pop)["PASS"]
    pop = _valid_population()
    pop["epochs"]["H3"]["t_ge_utc"] = "2026-03-03T11:34:14"
    assert not POP.independence(pop)["PASS"]


def test_driver_terms_match_the_predeclaration():
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    assert R.PRODUCTION_TERMS == ("TIDE", "LRP", "EARTH_SHADOW", "SRP_LIGHT_TIME", "EARTH_J2")
    assert R.STUDY_TERMS == ("PN", "PLANETS")
    assert R.REPORTED == ("EARTH_J2_ITRF93",)
    assert set(R.PRODUCTION_TERMS + R.STUDY_TERMS) <= set(pre["terms"])
    assert R.THRESHOLD == 0.1 and R.MARGINAL_FLOOR == 0.01 and R.G4_MAX == 1e-3 and R.FLOOR_MAX_M == 1e-3
    assert R.MARGINAL == ("EARTH_J2", "SRP_LIGHT_TIME")


def test_production_options_select_exactly_one_term():
    from lunar_od.earth_shadow import EarthShadowOptions
    from lunar_od.force_models import J2_EARTH_UNNORMALIZED
    from lunar_od.lunar_radiation import LunarRadiationOptions
    from lunar_od.lunar_tide import LunarTideOptions
    from lunar_od.srp_light_time import SRPLightTimeOptions, SSBVelocityTable
    table = SSBVelocityTable(t_s=np.array([0.0, 60.0, 120.0, 180.0]), sun_vel_mps=np.zeros((4, 3)),
                             moon_vel_mps=np.zeros((4, 3)), et0_s=0.0)
    expected = {"TIDE": ("lunar_tide", LunarTideOptions), "LRP": ("lunar_radiation", LunarRadiationOptions),
                "EARTH_SHADOW": ("earth_shadow", EarthShadowOptions),
                "SRP_LIGHT_TIME": ("srp_light_time", SRPLightTimeOptions)}
    for term in R.PRODUCTION_TERMS:
        kw = R.production_options(term, table)
        if term == "EARTH_J2":
            assert kw == dict(j2_earth=J2_EARTH_UNNORMALIZED, earth_j2_mode="indirect")
            continue
        key, cls = expected[term]
        assert list(kw) == [key] and isinstance(kw[key], cls)
    assert R.production_options("TIDE", table)["lunar_tide"].static_field_convention == "TIDE_FREE"
    assert R.production_options("TIDE", table, off=True) == {"lunar_tide": None}
    assert R.production_options("EARTH_J2", table, off=True) == {"j2_earth": 0.0, "earth_j2_mode": "indirect"}


def _cell(name, epoch, response, dr, floor=1e-5, tfloor=1e-5, g3=True):
    return dict(cell=name, epoch=epoch, floor_m=floor, G3=g3,
                terms={"T": dict(response=response, response_at="RANGE_RANGE_RATE/DSN_REALISTIC", max_dr_m=dr,
                                 floor_m=tfloor)})


def test_classification_rules():
    cells = [_cell("H1xS1", "H1", 0.2, 1.0), _cell("H2xS1", "H2", 0.05, 1.0)]
    assert R.classify("T", cells)["hold_out_class"] == "IMPLEMENT"
    # a below-floor cell cannot make a term material
    cells = [_cell("H1xS1", "H1", 0.5, 1e-4), _cell("H2xS1", "H2", 0.05, 1.0)]
    c = R.classify("T", cells)
    assert c["hold_out_class"] == "DOCUMENT_ONLY" and c["max_usable_response"] == 0.05
    assert R.replication("TIDE", "IMPLEMENT", 0.2) == "REPLICATED"
    assert R.replication("TIDE", "DOCUMENT_ONLY", 0.05) == "NOT_REPLICATED"
    assert R.replication("PN", "DOCUMENT_ONLY", 0.05) == "REPLICATED"
    assert R.replication("PN", "IMPLEMENT", 0.12) == "NOT_REPLICATED"
    assert R.replication("EARTH_J2", "IMPLEMENT", 0.1) == "REPLICATED"
    assert R.replication("EARTH_J2", "DOCUMENT_ONLY", 0.0999) == "MARGINAL_NOT_REPLICATED"
    assert R.replication("SRP_LIGHT_TIME", "DOCUMENT_ONLY", 0.01) == "MARGINAL_NOT_REPLICATED"
    assert R.replication("SRP_LIGHT_TIME", "DOCUMENT_ONLY", 0.0099) == "CONTRADICTED"


def test_req_dyn_007_rule():
    ok = {t: "REPLICATED" for t in ("TIDE", "LRP", "EARTH_SHADOW", "PN", "PLANETS", "EARTH_J2", "SRP_LIGHT_TIME")}
    assert R.req_dyn_007(ok, gates_pass=True) == "passed"
    assert R.req_dyn_007(ok, gates_pass=False) == "partial"
    assert R.req_dyn_007(dict(ok, EARTH_J2="MARGINAL_NOT_REPLICATED"), gates_pass=True) == "passed"
    assert R.req_dyn_007(dict(ok, EARTH_J2="CONTRADICTED"), gates_pass=True) == "partial"
    assert R.req_dyn_007(dict(ok, PN="NOT_REPLICATED"), gates_pass=True) == "partial"
    assert R.req_dyn_007(dict(ok, LRP="S5_INCONCLUSIVE"), gates_pass=True) == "partial"
    assert R.req_dyn_007(dict(ok, SRP_LIGHT_TIME="S5_INCONCLUSIVE"), gates_pass=True) == "partial"
