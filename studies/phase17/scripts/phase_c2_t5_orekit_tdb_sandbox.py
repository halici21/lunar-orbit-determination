"""Phase C2 term T5: is the C1-I G4 range difference (Orekit vs lunar_od relativistic_light_time, correlated
with range rate) the Orekit-vs-SPICE TDB-TT difference? Same setup as meas_corr_c1i.py; logs both TDB-TT
models per epoch and applies the C2 predeclared verdict (slope in [0.8, 1.2], RMS reduction >= 50%).
Writes out/meas/c2_t5_tdb.json.
"""
import json

import numpy as np

from common import OUT, load_lunar_od_kernels, load_scenario
from orekit_env import moon_icrf_frame, tdb_date
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
from org.orekit.estimation.measurements import GroundStation, ObservableSatellite, Range
from org.orekit.estimation.measurements.modifiers import ShapiroRangeModifier
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants, IERSConventions, PVCoordinates, TimeStampedPVCoordinates


def main():
    import spiceypy as spice
    from lunar_od import relativistic_light_time as R
    from lunar_od.dynamics import f3body_moon
    from lunar_od.radiometrics import interp_state_history

    sc = load_scenario()
    load_lunar_od_kernels("DE440")
    L = np.load(OUT / "meas" / "lunar_od.npz")
    grid, hist, t_meas, tc = L["grid"], L["hist"], L["t_meas"], float(L["tc"])
    et0, gm = sc["et0"], sc["gm_m3_s2"]
    mf = moon_icrf_frame("DE440")
    icrf = FramesFactory.getICRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    st = sc["station"]
    topo = TopocentricFrame(earth, GeodeticPoint(np.radians(st["lat_deg"]), np.radians(st["lon_deg"]), st["alt_m"]),
                            st["name"])
    station = GroundStation(topo)
    for drv in (station.getPrimeMeridianOffsetDriver(), station.getPrimeMeridianDriftDriver(),
                station.getPolarOffsetXDriver(), station.getPolarDriftXDriver(),
                station.getPolarOffsetYDriver(), station.getPolarDriftYDriver(),
                station.getClockOffsetDriver(), station.getClockDriftDriver(), station.getClockAccelerationDriver()):
        drv.setReferenceDate(tdb_date(0.0))
    sat = ObservableSatellite(0)

    def state_icrf(t, frame=None):
        frame = icrf if frame is None else frame
        x = interp_state_history(grid, hist, t)
        e = np.asarray(spice.spkpos("EARTH", et0 + t, "J2000", "NONE", "MOON")[0]) * 1e3
        s = np.asarray(spice.spkpos("SUN", et0 + t, "J2000", "NONE", "MOON")[0]) * 1e3
        a = f3body_moon(x, gm["MOON"], gm["EARTH"], gm["SUN"], e, s)[3:]
        pv = TimeStampedPVCoordinates(tdb_date(t), Vector3D(*x[:3]), Vector3D(*x[3:]), Vector3D(*a))
        pv_icrf = mf.getTransformTo(frame, tdb_date(t)).transformPVCoordinates(pv)
        return SpacecraftState(CartesianOrbit(pv_icrf, frame, float(gm["SUN"])))

    gcrf = FramesFactory.getGCRF()

    def orekit_range(t, shapiro=False, state_epoch=None, frame=None):
        """Orekit two-way range tagged at t; the spacecraft state handed over is at ``state_epoch``
        (default: the receive tag, as in the trial; the event epoch t2 removes Orekit's Taylor shift)."""
        m = Range(station, True, tdb_date(t), 0.0, 1.0, 1.0, sat)
        if shapiro:
            m.addModifier(ShapiroRangeModifier(float(gm["EARTH"])))
        s = state_icrf(t if state_epoch is None else state_epoch, frame)
        return float(m.estimateWithoutDerivatives([s]).getEstimatedValue()[0])

    def station_mci(t):
        pv = topo.getTransformTo(mf, tdb_date(float(t))).transformPVCoordinates(PVCoordinates.ZERO)
        p, v = pv.getPosition(), pv.getVelocity()
        return np.array([p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()])

    sch = R.SpacecraftHistory(grid, hist)
    eph = R.BarycentricEphemeris(float(et0))
    ssb = R.LightTimeModel(frame="SSB")
    from orekit_env import TDB as TDB_SCALE
    k_, eb_ = spice.gdpool("DELTET/K", 0, 1)[0], spice.gdpool("DELTET/EB", 0, 1)[0]
    m0_, m1_ = spice.gdpool("DELTET/M", 0, 2)

    def spice_tdb_tt(t):
        m = m0_ + m1_ * (et0 + t)
        return float(k_ * np.sin(m + eb_ * np.sin(m)))

    def orekit_tdb_tt(t):
        return float(TDB_SCALE.offsetFromTAI(tdb_date(t)).toDouble()) - 32.184

    rr, d_rng, d_tdb = [], [], []
    for t in t_meas:
        t = float(t)
        rr.append(R.counted_doppler(t, tc, station_mci, sch, eph, ssb)[0])
        d_rng.append(orekit_range(t) - R.two_way_range_m(R.solve_two_way(t, station_mci, sch, eph, ssb)))
        d_tdb.append(orekit_tdb_tt(t) - spice_tdb_tt(t))
    rr, d_rng, d_tdb = map(np.asarray, (rr, d_rng, d_tdb))
    pred = rr * d_tdb                                   # the predeclared regressor rho_dot * (TDB_orekit - TDB_spice)
    slope = float(np.dot(pred, d_rng) / np.dot(pred, pred))
    rms = lambda v: float(np.sqrt(np.mean(v * v)))     # noqa: E731
    reduction = 1.0 - rms(d_rng - pred) / rms(d_rng)
    tag = float(np.dot(rr, d_rng) / np.dot(rr, rr))     # informational: implied constant tag offset
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])      # noqa: E731
    out = dict(n=int(rr.size), et0=float(et0), ulp_et0_s=float(np.spacing(float(et0))),
               d_range_mm=dict(min=float(d_rng.min() * 1e3), max=float(d_rng.max() * 1e3), rms=rms(d_rng) * 1e3),
               corr_drange_rr=c(d_rng, rr),
               tdb_orekit_minus_spice_us=dict(min=float(d_tdb.min() * 1e6), max=float(d_tdb.max() * 1e6)),
               slope=slope, rms_reduction_with_prediction=reduction,
               verdict="EXPLAINED" if (0.8 <= slope <= 1.2 and reduction >= 0.5) else "UNEXPLAINED",
               informational=dict(implied_constant_tag_s=tag, implied_tag_in_ulp=tag / float(np.spacing(float(et0))),
                                  rms_reduction_with_constant_tag=1.0 - rms(d_rng - tag * rr) / rms(d_rng)))
    (OUT / "meas" / "c2_t5_tdb.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
