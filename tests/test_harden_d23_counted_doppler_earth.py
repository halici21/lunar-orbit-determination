"""Phase 17-HARDEN-D23 - counted-Doppler Earth ephemeris: permanent regression.

SPICE-free and analytic.  The Earth moves on a circular orbit about the Moon
(radius 3.844e8 m, sidereal rate), the site transform is a deterministic
Z-rotating Earth, and the spacecraft flies a straight line.  The exact-Earth
oracle evaluates the Earth ANALYTICALLY at every event epoch the production
solver requests, so it isolates the Earth policy alone.

The historical defect: R3 interpolated the Earth LINEARLY on the 90 s grid
(R3-P22).  The count interval is centred on a grid node at the RECEIVE epochs
(t_mid +/- Tc/2), where the linear-interpolation errors are equal and cancel in
the observable; but the uplink TRANSMIT epochs t_mid +/- Tc/2 - tau sit
asymmetrically inside their grid intervals, so the errors no longer cancel and
the value carries a near-constant offset (~1.7-1.8e-3 m/s in the CD1/KP1
population).  The repaired path uses the same cubic-Hermite Earth as M3 range.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lunar_od.radiometrics import (
    COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
    CountedDopplerStationStateProvider,
    RangeRatePhysicsConfig,
    _interp_vector,
    make_exact_counted_doppler_station_state_provider,
    solve_two_way_light_time,
    two_way_counted_doppler_observable,
)

ROOT = Path(__file__).resolve().parents[1]
R_EM = 3.844e8                        # m
OMEGA_EM = 2.6617e-6                  # rad/s, sidereal month
OMEGA_E = 7.292115e-5                 # rad/s
H_GRID = 90.0                         # s, the Phase-17 grid
TC = 60.0
VALUE_GATE_MPS = 1e-7                 # artifacts/phase17_d23_acceptance.json
QUALIFICATION = ROOT / "artifacts" / "phase17_d23_qualification.json"


@dataclass(frozen=True)
class _Station:
    name: str = "D23 equatorial"
    r_ecef_m: np.ndarray = None


STATION = _Station(r_ecef_m=np.array([6378137.0, 0.0, 0.0]))


def _earth(t):
    c, s = np.cos(OMEGA_EM * t), np.sin(OMEGA_EM * t)
    return np.array([R_EM * c, R_EM * s, 0.0, -R_EM * OMEGA_EM * s, R_EM * OMEGA_EM * c, 0.0])


def _sxform(_frm, _to, t):
    c, s = np.cos(OMEGA_E * t), np.sin(OMEGA_E * t)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    drot = OMEGA_E * np.array([[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 0.0]])
    out = np.zeros((6, 6))
    out[:3, :3] = rot
    out[3:, 3:] = rot
    out[3:, :3] = drot
    return out


def _fixture():
    t_grid = np.arange(-1800.0, 1800.0 + 1e-9, H_GRID)
    earth = np.array([_earth(t) for t in t_grid])
    sc = np.zeros((t_grid.size, 6))
    sc[:, 0] = 1.9e6 + 300.0 * t_grid
    sc[:, 1] = -4.0e5 + 1450.0 * t_grid
    sc[:, 2] = 2.0e5
    sc[:, 3], sc[:, 4] = 300.0, 1450.0
    xf = np.array([_sxform("J2000", "ITRF93", t) for t in t_grid])
    return t_grid, sc, earth, xf


def _config():
    return RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=TC,
                                  light_time_tolerance_s=1e-12, light_time_equation_tolerance_s=1e-11,
                                  light_time_max_iter=30)


def _site(t):
    return np.linalg.solve(_sxform("J2000", "ITRF93", t), np.concatenate([STATION.r_ecef_m, np.zeros(3)]))


def _exact_provider():
    return CountedDopplerStationStateProvider(state_fn=lambda t: _earth(t) + _site(t),
                                              earth_ephemeris_method="analytic_exact")


def _historical_linear_provider(t_grid, earth):
    """The pre-repair R3 provider, rebuilt verbatim: linear Earth, exact site."""
    def state_fn(t):
        e = np.concatenate([_interp_vector(t_grid, earth[:, :3], t), _interp_vector(t_grid, earth[:, 3:], t)])
        return e + _site(t)
    return CountedDopplerStationStateProvider(state_fn=state_fn, earth_ephemeris_method="historical_linear")


def _value(provider, t_mid):
    t_grid, sc, earth, xf = _fixture()
    return two_way_counted_doppler_observable(t_mid, STATION, t_grid, sc, earth[:, :3], earth[:, 3:], xf,
                                              _config(), et0_s=0.0, station_state_provider=provider)


def _production_provider():
    t_grid, _sc, earth, _xf = _fixture()
    return make_exact_counted_doppler_station_state_provider(STATION, 0.0, t_grid, earth[:, :3], earth[:, 3:],
                                                             sxform_fn=_sxform)


MIDS = (0.0, 270.0, 540.0, -630.0)        # receive midpoints ON grid nodes, as in CD1


@pytest.mark.parametrize("t_mid", MIDS)
def test_repaired_value_matches_the_exact_earth_oracle(t_mid):
    y_prod = _value(_production_provider(), t_mid)
    y_exact = _value(_exact_provider(), t_mid)
    assert abs(y_prod - y_exact) <= VALUE_GATE_MPS, (t_mid, y_prod - y_exact)


@pytest.mark.parametrize("t_mid", MIDS)
def test_historical_linear_earth_mutant_is_detected(t_mid):
    """The pre-repair behaviour violates the value gate by orders of magnitude."""
    t_grid, _sc, earth, _xf = _fixture()
    y_mut = _value(_historical_linear_provider(t_grid, earth), t_mid)
    y_exact = _value(_exact_provider(), t_mid)
    assert abs(y_mut - y_exact) > 1e3 * VALUE_GATE_MPS, (t_mid, y_mut - y_exact)


def test_production_declares_the_repaired_policy():
    assert COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD == "cubic_hermite_grid_interpolation"
    assert _production_provider().earth_ephemeris_method == "cubic_hermite_grid_interpolation"


def test_mechanism_receive_epochs_cancel_transmit_epochs_do_not():
    """Linear-Earth errors are equal at the symmetric receive epochs but not at
    the asymmetric transmit epochs; the repaired Earth is exact at both."""
    t_grid, sc, earth, xf = _fixture()
    lin, prod = _historical_linear_provider(t_grid, earth), _production_provider()
    t_mid = 0.0
    errs = {}
    for label, t3 in (("start", t_mid - TC / 2), ("end", t_mid + TC / 2)):
        sol = solve_two_way_light_time(t3, STATION, t_grid, sc, earth[:, :3], earth[:, 3:], xf, _config(),
                                       et0_s=0.0, station_state_provider=prod)
        for kind, t in (("receive", sol.receive_time_s), ("transmit", sol.transmit_time_s)):
            truth = _earth(t)[:3] + _site(t)[:3]
            errs[(label, kind, "linear")] = lin.state(t)[:3] - truth
            errs[(label, kind, "repaired")] = prod.state(t)[:3] - truth
    rx = np.linalg.norm(errs[("end", "receive", "linear")] - errs[("start", "receive", "linear")])
    tx = np.linalg.norm(errs[("end", "transmit", "linear")] - errs[("start", "transmit", "linear")])
    # receive errors cancel (symmetric about the node) up to the slow rotation of
    # the orbital acceleration between the two grid intervals (~5e-4 m here)
    assert rx < 1e-2 * tx
    assert tx > 1e-2                               # transmit errors do not (asymmetric): ~0.1 m scale
    assert max(np.linalg.norm(v) for k, v in errs.items() if k[2] == "repaired") < 1e-6


def test_repaired_earth_matches_the_exact_orbit_everywhere():
    t_grid, _sc, earth, _xf = _fixture()
    prod = _production_provider()
    ts = np.linspace(t_grid[0] + 1.0, t_grid[-1] - 1.0, 997)
    err = max(np.linalg.norm(prod.state(t)[:3] - _earth(t)[:3] - _site(t)[:3]) for t in ts)
    lin = max(np.linalg.norm(_historical_linear_provider(t_grid, earth).state(t)[:3] - _earth(t)[:3] - _site(t)[:3])
              for t in ts)
    assert err < 1e-6 and lin > 1.0                # m: Hermite ~1e-9 vs linear ~2.6 m at mid-nodes


@pytest.mark.skipif(not QUALIFICATION.exists(), reason="D23 qualification artifact not yet produced")
def test_committed_population_qualification_is_reported_strictly():
    """The frozen gates as computed, plus the adjudication of the one that failed.

    EARTH_EPOCH_CONTRACT failed as frozen: its diagnostic evaluated SPICE at
    fl(et0 + t) for arbitrary event epochs, so it measured the ORACLE's time
    rounding (<= |v_Earth| ulp(ET)/2 ~ 6.4e-5 m).  The epoch scan evaluates the
    same 1e-6 m criterion at exactly representable ETs over every grid interval
    of all 24 epochs.  Both facts are pinned here; neither is hidden."""
    q = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    g = q["gates"]
    for name in ("DIRECT_SPICE_VALUE_ORACLE", "REDUCTION", "HISTORICAL_D23_MUTANT_REPRODUCED", "SOLVER_FLOOR",
                 "CD1_INFORMATION", "KP1_INFORMATION", "CD1_FD_ORACLE"):
        assert g[name] == "PASS", name
    assert g["EARTH_EPOCH_CONTRACT"] == "FAIL" and g["PHASE17_HARDEN_D23_REPAIR_GATE"] == "FAIL"
    v = q["value"]
    assert v["prod_minus_oracleS_max"] <= VALUE_GATE_MPS and v["prod_minus_oracleF_max"] <= VALUE_GATE_MPS
    assert v["mutant_minus_oracleS_max"] > 1e-3 and v["reduction_factor"] > 1e4
    assert v["production_earth_method"] == ["cubic_hermite_grid_interpolation"]
    assert v["oracle_convergence_failures"] == []
    assert q["cd1"]["decision_identical"] is True and q["kp1"]["classes_identical"] is True
    # adjudication: the failing diagnostic equals the oracle time-rounding bound ...
    ulp_et, v_earth = 1.1920928955078125e-07, 1.1e3
    assert q["epochs"]["hermite_pos_err_max_m"] <= v_earth * ulp_et / 2.0
    # ... and the same frozen 1e-6 m criterion holds at exactly representable epochs
    scan = json.loads((ROOT / "artifacts" / "phase17_d23_epoch_scan.json").read_text(encoding="utf-8"))
    assert scan["et_rounding_max_s"] < 1e-12 and scan["points"] > 1.9e5
    assert scan["hermite_vs_et_exact_max_m"] <= 1e-6
