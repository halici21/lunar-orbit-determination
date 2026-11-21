"""F2 WP2.6 slice 2 (MEAS-13): delta-DOR with the barycentric spacecraft delay and the IERS 2010 consensus
quasar delay in one model.

Gates G7-G12 in studies/phase17/configs/wp2_6_predeclaration.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lunar_od.config import Station

GOLDSTONE = Station("Goldstone", 35.4267, -116.8900, 1000.0, (0, 0, 0), 2.0, 1e-4, 1e-4)
CANBERRA = Station("Canberra", -35.4014, 148.9817, 690.0, (0, 0, 0), 2.0, 1e-4, 1e-4)
MADRID = Station("Madrid", 40.4314, -4.2481, 834.0, (0, 0, 0), 2.0, 1e-4, 1e-4)
C = 299792458.0
LEGACY = json.loads((Path(__file__).parent / "fixtures" / "wp2_6_delta_dor_legacy_before.json").read_text())


@pytest.fixture(scope="module")
def arc():
    spice = pytest.importorskip("spiceypy")
    from lunar_od.spice_loader import load_spice_kernels
    try:
        load_spice_kernels(None, clear=False)
    except FileNotFoundError:
        pytest.skip("SPICE kernels not available")
    import lunar_od.relativistic_light_time as R
    from lunar_od.dynamics import propagate_augmented_state
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.ephemeris_generation import gravitational_parameters
    from lunar_od.station_time import Site

    et0 = spice.str2et("2027-03-02T16:00:00")
    t = np.arange(0.0, 3600.0 + 1.0, 60.0)
    moon_eph = sample_moon_centered_ephemeris(et0, np.arange(-600.0, t[-1] + 601.0, 60.0))
    gm = gravitational_parameters()
    r0 = 1737.4e3 + 100e3
    x0 = np.array([r0, 30e3, -20e3, -15.0, np.sqrt(gm[0] / r0), 4.0])

    def propagate(x_initial, augmented=False):
        y0 = np.concatenate([x_initial, np.eye(6).reshape(-1, order="F")]) if augmented else x_initial
        from lunar_od.dynamics import propagate_state
        if augmented:
            return propagate_augmented_state(t, y0, *gm, moon_eph.earth_position, moon_eph.sun_position,
                                             rtol=1e-12, atol=1e-13)
        return propagate_state(t, y0, *gm, moon_eph.earth_position, moon_eph.sun_position, rtol=1e-12, atol=1e-13)

    def station_fn(st):
        def state(t_s):
            # et0 + t rounds to ulp(et0) ~ 1.2e-7 s; the station is moved by velocity x the exact rounding error
            # (as BarycentricEphemeris._ssb_state). Without it a finite difference whose t_B shift (~1e-9 s) is
            # below the quantum sees a frozen station and misses the v_B/c term (amendment log, WP2.6 G10).
            t = float(t_s)
            et = et0 + t
            earth = np.asarray(spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")[0]) * 1e3
            s = earth + np.asarray(spice.sxform("ITRF93", "J2000", et)) @ np.concatenate([st.r_ecef_m, np.zeros(3)])
            s[:3] += s[3:6] * (t - (et - et0))
            return s
        return state

    return dict(spice=spice, R=R, et0=et0, t=t, x0=x0, propagate=propagate, eph=R.BarycentricEphemeris(et0),
                sc=R.SpacecraftHistory(t, propagate(x0, True)), station=station_fn,
                site=lambda st: Site.from_itrf_m(st.r_ecef_m))


def _quasar(a, t_obs, offset_deg=3.0):
    """3 deg from the geocentric direction of the spacecraft."""
    eph = a["eph"]
    los = a["sc"].state(t_obs)[:3] + eph.moon_state(t_obs)[:3] - eph.body_position("EARTH", t_obs)
    los /= np.linalg.norm(los)
    perp = np.cross(los, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    return np.cos(np.radians(offset_deg)) * los + np.sin(np.radians(offset_deg)) * perp


@pytest.mark.parametrize("pair", [(GOLDSTONE, CANBERRA), (MADRID, CANBERRA)], ids=["GDS-CAN", "MAD-CAN"])
def test_g7_same_model_quasar_delay_vs_iers_consensus(arc, pair):
    """G7 as amended after the result (amendment log, exploratory). The pre-registered oracle was a BCRS plane
    wave with ERFA-dtdb station time, the production delta_dor_ssb quasar term is now that same-model delay, and
    the gate checks it against the independent IERS 2010 consensus delay (gravity off in both) within the two
    explained model differences: dtdb's topocentric differential minus the exact V.b/c^2, and the
    (2U/c^2 + V^2/c^2) |b|/c scale terms the Galilean station positions leave out."""
    from lunar_od.station_time import tdb_minus_tt

    R, eph = arc["R"], arc["eph"]
    st_a, st_b = pair
    fa, fb = arc["station"](st_a), arc["station"](st_b)
    model = R.LightTimeModel(frame="SSB", interval_time_scale="TT", station_time="FULL_TOPOCENTRIC")
    sites = (arc["site"](st_a), arc["site"](st_b))
    for t_obs in np.linspace(0.0, 3600.0, 10):
        k_hat = _quasar(arc, t_obs)
        same = R.quasar_delay_station_time(t_obs, fa, fb, k_hat, eph, model, sites=sites)
        consensus = R.quasar_delay_consensus(t_obs, fa, fb, k_hat, eph, gravity_bodies=())
        geo = [f(t_obs)[:3] + eph.moon_state(t_obs)[:3] - eph.body_position("EARTH", t_obs) for f in (fa, fb)]
        b = geo[1] - geo[0]
        v = eph.body_velocity("EARTH", t_obs)
        topo = [float(tdb_minus_tt(arc["et0"], t_obs, s)) - float(tdb_minus_tt(arc["et0"], t_obs)) for s in sites]
        dtdb_error = (topo[1] - topo[0]) - float(v @ b) / C ** 2
        u = eph.gm("SUN") / np.linalg.norm(eph.body_position("EARTH", t_obs) - eph.body_position("SUN", t_obs))
        bound = abs(dtdb_error) + (2 * u / C ** 2 + float(v @ v) / C ** 2) * np.linalg.norm(b) / C + 1e-11
        assert abs(same.delay_s - consensus.delay_s) < bound, (t_obs, same.delay_s - consensus.delay_s, bound)


def test_g8_gravitational_delay_vs_moyer_one_way_shapiro(arc):
    R = arc["R"]
    model = R.LightTimeModel(frame="SSB", shapiro_bodies=("SUN", "EARTH"))
    eph = arc["eph"]
    fa, fb = arc["station"](GOLDSTONE), arc["station"](CANBERRA)
    worst = 0.0
    for t_obs in np.linspace(0.0, 3600.0, 5):
        k_hat = _quasar(arc, t_obs)
        q = R.quasar_delay_consensus(t_obs, fa, fb, k_hat, eph)
        xa = fa(t_obs)[:3] + eph.moon_state(t_obs)[:3]
        t_b = t_obs + q.tau_s
        xb = fb(t_b)[:3] + eph.moon_state(t_b)[:3]
        source = xa + 1e16 * k_hat
        s_a = R.shapiro_delay_s(model, eph, source, xa, t_obs)[0]
        s_b = R.shapiro_delay_s(model, eph, source, xb, t_b)[0]
        worst = max(worst, abs(q.gravitational_s - (s_b - s_a)))
    assert worst < 1e-12, worst


def test_g9_station_clocks_cancel_in_delta_dor(arc):
    R = arc["R"]
    fa, fb = arc["station"](GOLDSTONE), arc["station"](CANBERRA)
    sites = (arc["site"](GOLDSTONE), arc["site"](CANBERRA))
    for t_obs in (600.0, 2400.0):
        k_hat = _quasar(arc, t_obs)
        plain = R.delta_dor_ssb(t_obs, fa, fb, arc["sc"], arc["eph"], k_hat, sites=sites)
        clocked = R.delta_dor_ssb(t_obs, fa, fb, arc["sc"], arc["eph"], k_hat, sites=sites,
                                  clock_offsets_s=(1e-6, -2e-6))
        assert abs(clocked.spacecraft.delay_s - plain.spacecraft.delay_s) > 2.9e-6
        assert abs(clocked.quasar.delay_s - plain.quasar.delay_s) > 2.9e-6
        assert abs(clocked.value_s - plain.value_s) < 1e-11, clocked.value_s - plain.value_s


def test_g10_state_partial_vs_richardson_fd(arc):
    R = arc["R"]
    fa, fb = arc["station"](GOLDSTONE), arc["station"](CANBERRA)
    sites = (arc["site"](GOLDSTONE), arc["site"](CANBERRA))
    epochs = np.linspace(600.0, 3000.0, 5)
    quasars = [_quasar(arc, t) for t in epochs]

    def values(sc):
        return np.array([R.delta_dor_ssb(t, fa, fb, sc, arc["eph"], k, sites=sites).value_s
                         for t, k in zip(epochs, quasars)])

    analytic = np.array([R.delta_dor_ssb_partial(R.delta_dor_ssb(t, fa, fb, arc["sc"], arc["eph"], k, sites=sites),
                                                 arc["sc"]) for t, k in zip(epochs, quasars)])

    def central(scale):
        out = np.zeros_like(analytic)
        for j in range(6):
            h = scale * (10.0 if j < 3 else 1e-2)
            e = np.zeros(6)
            e[j] = h
            plus = values(R.SpacecraftHistory(arc["t"], arc["propagate"](arc["x0"] + e)))
            minus = values(R.SpacecraftHistory(arc["t"], arc["propagate"](arc["x0"] - e)))
            out[:, j] = (plus - minus) / (2 * h)
        return out

    fd = (4 * central(1.0) - central(2.0)) / 3
    rel = np.linalg.norm(analytic - fd, axis=1) / np.linalg.norm(fd, axis=1)
    assert rel.max() < 1e-6, rel


def test_g11_legacy_delta_dor_is_unchanged(arc):
    from lunar_od.delta_dor import (DELTA_DOR_MODEL_STATUS, QuasarDirection, delta_dor_observable_s,
                                    quasar_differential_delay_s, solve_common_transmit_event)

    assert DELTA_DOR_MODEL_STATUS == "legacy_synthetic"
    sc, fa, fb = arc["sc"], arc["station"](GOLDSTONE), arc["station"](CANBERRA)
    for row in LEGACY["rows"]:
        t_obs = row["t_obs"]
        ev = solve_common_transmit_event(t_obs, fa(t_obs)[:3], lambda t: fb(t)[:3], lambda t: sc.state(t)[:3])
        dq = quasar_differential_delay_s(fa(t_obs)[:3], fb(t_obs)[:3],
                                         QuasarDirection.from_unit_vector(_quasar(arc, t_obs)))
        assert [float(ev.spacecraft_differential_delay_s).hex(), float(dq).hex(),
                float(delta_dor_observable_s(ev, dq)).hex()] == row["hex"]


def test_g12_fail_closed(arc):
    R = arc["R"]
    fa, fb = arc["station"](GOLDSTONE), arc["station"](CANBERRA)
    sites = (arc["site"](GOLDSTONE), arc["site"](CANBERRA))
    k_hat = _quasar(arc, 600.0)
    with pytest.raises(ValueError):
        R.delta_dor_ssb(600.0, fa, fb, arc["sc"], arc["eph"], k_hat, sites=sites, model=R.DSN_LIGHT_TIME_MODEL)
    with pytest.raises(ValueError):
        R.delta_dor_ssb(600.0, fa, fb, arc["sc"], arc["eph"], k_hat, sites=(sites[0], None))
    with pytest.raises(ValueError):
        R.quasar_delay_consensus(600.0, fa, fb, k_hat, arc["eph"], gravity_bodies=("VENUS",))
