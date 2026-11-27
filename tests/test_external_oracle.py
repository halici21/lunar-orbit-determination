"""F5 WP5.1: lunar_od against the pinned Orekit oracle (AC-3), without Orekit.

The reference values were produced by Orekit 13.1.9 with ``tools/oracle/generate.py``; their
provenance (versions, SHA-256 of every input and fixture, alignment, integrator settings) is
``tests/fixtures/external_oracle/provenance.json``. Thresholds are read from, and checked
against, ``studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json``.

Gates (predeclaration names):
* O_F  fixture integrity, provenance completeness, no Orekit/JPype import on the test path;
* O_D  dynamics, lunar_od force model integrated in TT (alignment A_TIME) vs Orekit: < 1 mm over 1 day;
* O_L  the production ``propagate_state`` equals that force model integrated in TDB: <= 0.1 mm;
* O_M2 two-way range, Moon-centred and barycentric light time: < 1 mm;
* O_M3 counted Doppler (Tc = 60 s), both frames: < 1 um/s; L_M3 production counted Doppler = range form.

Requires the SPICE kernels (and GL1800F for D2); without them the tests skip and name the file.
Results as computed: ``studies/phase17/reports/wp5_1_orekit_oracle.md``.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "oracle"
if str(ORACLE) not in sys.path:
    sys.path.insert(0, str(ORACLE))

import compare as C  # noqa: E402
import scenario as S  # noqa: E402
from tests.slow import RUN_SLOW_TESTS  # noqa: E402

FIX = S.FIXTURE_DIR
#: dynamics cases whose lunar_od side takes > 20 s here (predeclaration test_tiers)
SLOW_CASES: set[str] = {"D2-50", "D2-100", "D3"}
#: cases that fail a gate as computed (strict xfail, see the report); empty when all pass
_REPORT = "studies/phase17/reports/wp5_1_orekit_oracle.md"
XFAIL_O_D: dict[str, str] = {}
_O_L_REASON = ("O_L FAIL as computed: the production propagate_state (DOP853, no step cap) is not converged to "
               "0.1 mm with high-degree harmonics; finding for WP3.3, see " + _REPORT)
XFAIL_O_L: dict[str, str] = {"D2-50": _O_L_REASON, "D2-100": _O_L_REASON}
#: measurement keys that fail a gate as computed (strict xfail)
XFAIL_M: dict[str, str] = {
    "cd_ssb": ("O_M3 FAIL as computed (max 3.8 um/s): Orekit's barycentric two-way range carries twice "
               "lunar_od's double-precision noise; the Moon-centred series agree. See " + _REPORT),
}


def _need_spice():
    pytest.importorskip("spiceypy")
    try:
        kd = S.kernel_dir()
    except FileNotFoundError as exc:
        pytest.skip(f"SPICE kernels not available: {exc}")
    for name in ("de440.bsp", "de421.bsp", "moon_pa_de440_200625.bpc"):
        if not (kd / name).is_file():
            pytest.skip(f"SPICE kernel missing: {kd / name}")


def _need_gravity():
    if not S.gravity_tab().is_file():
        pytest.skip(f"GL1800F gravity field missing: {S.gravity_tab()}")


# ---------------------------------------------------------------------------
# O_F
# ---------------------------------------------------------------------------
def test_o_f_fixture_files_match_provenance_hashes():
    assert C.fixture_hash_mismatches() == []


def test_o_f_provenance_records_versions_inputs_and_settings():
    p = C.provenance()
    v = p["versions"]
    assert v["orekit-jpype"] == "13.1.9.0" and v["java"] and any(j.startswith("orekit-13.1.9") for j in v["orekit_jars"])
    for key in ("numpy", "scipy", "jplephem", "spiceypy", "JPype1"):
        assert v[key], key
    for key in ("orekit_data_dir_files", "jpl_DE440_binary", "jpl_DE421_binary", "gl1800f_tab", "de440.bsp",
                "de421.bsp", "moon_pa_de440_200625.bpc", "moon_de440_220930.txt", "naif0012.tls.txt",
                "oracle_helpers_jar_sources"):
        assert p["inputs"][key], key
    assert set(p["orekit_dynamics"]) == set(S.DYNAMICS_CASES)
    assert set(C.lunar_od_settings()["cases"]) == set(S.DYNAMICS_CASES)
    assert p["alignment"] == S.load_predeclaration()["alignment"]
    assert p["repository"]["commit"]


def test_o_f_scenario_is_the_trial_scenario():
    sc = C.scenario()
    assert sc["epoch_tdb"] == "2025-03-01T00:00:00" and sc["et0"] == 794059200.0
    from lunar_od.constants import MU_EARTH_M3S2, MU_MOON_M3S2, MU_SUN_M3S2

    assert sc["gm_m3_s2"] == {"MOON": MU_MOON_M3S2, "EARTH": MU_EARTH_M3S2, "SUN": MU_SUN_M3S2}


def test_o_f_test_path_imports_no_orekit():
    banned = ("orekit", "orekit_jpype", "jpype", "java", "org", "orekit_session")
    for path in (Path(__file__), ORACLE / "compare.py", ORACLE / "lunar_od_side.py", ORACLE / "scenario.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in banned, f"{path.name} imports {n}"


def test_o_f_thresholds_are_the_predeclared_ones():
    C.check_thresholds_match_predeclaration()
    assert C.THRESHOLDS == {"O_D_m": 1e-3, "O_L_m": 1e-4, "O_M2_m": 1e-3, "O_M3_mps": 1e-6, "L_M3_mps": 1e-8}


def test_o_p_preconditions_recorded_within_bounds():
    p = C.provenance()["preconditions"]
    for eph in ("DE440", "DE421"):
        assert p[f"a_reader_vs_cspice_{eph}_m"]["EARTH"] <= 1e-5
        assert p[f"a_reader_vs_cspice_{eph}_m"]["SUN"] <= 1e-3
    assert p["b_pck_euler_vs_pxform_max_element"] <= 1e-11
    assert p["c_moon_icrf_rotation_rate_rad_s"] <= 1e-15
    assert p["e_holmes_featherstone_c00_only_accel_mps2"] == 0.0
    assert p["f_ssb_minus_moon_two_way_m"]["min"] > 1.0
    # (d) failed its 50 ns bound; the predeclared remedy is the pinned Orekit time map
    if p["d_tdb_minus_tt_orekit_minus_spice_s"]["change_over_arc"] > 50e-9:
        assert (FIX / "orekit_time_map.csv").is_file()


# ---------------------------------------------------------------------------
# O_D, O_L
# ---------------------------------------------------------------------------
def _dyn_params(xfails: dict[str, str]):
    out = []
    for case in S.DYNAMICS_CASES:
        marks = []
        if case in SLOW_CASES:
            marks.append(pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow; set LUNAR_OD_RUN_SLOW_TESTS=1"))
        if case in xfails:
            marks.append(pytest.mark.xfail(strict=True, reason=xfails[case]))
        out.append(pytest.param(case, marks=marks, id=case))
    return out


_DYN_CACHE: dict[str, dict] = {}


def _dynamics(case: str) -> dict:
    _need_spice()
    if case.startswith("D2"):
        _need_gravity()
    if case not in _DYN_CACHE:
        _DYN_CACHE[case] = C.dynamics_case(case)
    return _DYN_CACHE[case]


@pytest.mark.parametrize("case", _dyn_params(XFAIL_O_D))
def test_o_d_dynamics_agree_with_orekit_within_1_mm_per_day(case):
    d = _dynamics(case)
    assert d["O_D_max_dr_m"] < C.THRESHOLDS["O_D_m"], json.dumps(d)


@pytest.mark.parametrize("case", _dyn_params(XFAIL_O_L))
def test_o_l_production_propagator_equals_the_aligned_force_model(case):
    d = _dynamics(case)
    assert d["O_L_max_dr_m"] <= C.THRESHOLDS["O_L_m"], json.dumps(d)


# ---------------------------------------------------------------------------
# O_M2, O_M3, L_M3
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def measurement_gates():
    _need_spice()
    return C.measurement_gates()


def _m_params(keys):
    return [pytest.param(k, id=k, marks=[pytest.mark.xfail(strict=True, reason=XFAIL_M[k])] if k in XFAIL_M else [])
            for k in keys]


@pytest.mark.parametrize("key", _m_params(["two_way_moon", "two_way_ssb"]))
def test_o_m2_two_way_range_agrees_with_orekit_within_1_mm(measurement_gates, key):
    assert measurement_gates[key]["max_abs"] < C.THRESHOLDS["O_M2_m"], measurement_gates[key]


@pytest.mark.parametrize("key", _m_params(["cd_moon", "cd_ssb"]))
def test_o_m3_counted_doppler_agrees_with_orekit_within_1_um_per_s(measurement_gates, key):
    assert measurement_gates[key]["max_abs"] < C.THRESHOLDS["O_M3_mps"], measurement_gates[key]


def test_l_m3_production_counted_doppler_equals_the_range_difference_form(measurement_gates):
    assert measurement_gates["L_M3_max_abs_mps"] <= C.THRESHOLDS["L_M3_mps"]


def test_measurement_inputs_are_the_pinned_grids():
    tg, traj, sg, st = C.shared_inputs()
    assert tg[0] <= S.T_MEAS[0] - 10.0 and tg[-1] >= S.T_MEAS[-1] + S.TC_S
    assert sg[0] <= S.T_MEAS[0] - S.TC_S and sg[-1] >= S.T_MEAS[-1] + S.TC_S
    assert np.all(np.isfinite(traj)) and np.all(np.isfinite(st))
