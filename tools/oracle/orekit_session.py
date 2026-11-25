"""Orekit session for the WP5.1 oracle generator (Orekit 13.1.9 via orekit-jpype).

Imported ONLY by ``generate.py orekit``; nothing under ``tests/`` imports it.
Configuration (all required, see README.md):

* ``JAVA_HOME``             JDK >= 21 (the generator was run with JDK 25)
* ``--orekit-data``         an unpacked orekit-data directory (IERS EOP, leap seconds, DE-440 files)
* ``--jpl-dir``             directory holding the JPL binary ephemerides ``linux_p1550p2650.440``
                            and ``lnxp1900p2053.421``
* the helper jar built by ``tools/oracle/jhelpers/build.py``

Ported from the 2026-10-21 trial harness (``orekit_env.py``) without changes to the physics.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from scenario import kernel_dir

DE_FILES = {"DE440": "linux_p1550p2650.440", "DE421": "lnxp1900p2053.421"}
_STATE: dict = {}


def start(orekit_data: Path, jpl_dir: Path, jar: Path, scenario: dict) -> None:
    """Start the JVM, register the data and import the Orekit names used below."""
    import orekit_jpype

    orekit_jpype.initVM(additional_classpaths=[str(jar)])
    from orekit_jpype.pyhelpers import setup_orekit_data

    setup_orekit_data(str(orekit_data), from_pip_library=False)
    from java.io import File
    from org.orekit.data import DataContext, DirectoryCrawler
    from org.orekit.frames import FramesFactory
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    DataContext.getDefault().getDataProvidersManager().addProvider(DirectoryCrawler(File(str(jpl_dir))))
    tdb = TimeScalesFactory.getTDB()
    _STATE.update(scenario=scenario, TDB=tdb, GCRF=FramesFactory.getGCRF(),
                  J2000_TDB=AbsoluteDate(2000, 1, 1, 12, 0, 0.0, tdb),
                  EPOCH=AbsoluteDate(*_epoch_fields(scenario["epoch_tdb"]), tdb))


def _epoch_fields(iso: str):
    date, time = iso.split("T")
    y, mo, d = (int(v) for v in date.split("-"))
    h, mi, s = time.split(":")
    return y, mo, d, int(h), int(mi), float(s)


def tdb_date(t_s: float = 0.0):
    """Scenario epoch + t_s seconds of TDB (not TAI)."""
    from org.orekit.time import AbsoluteDate

    return AbsoluteDate(_STATE["EPOCH"], float(t_s), _STATE["TDB"])


def et_of(date) -> float:
    return float(date.offsetFrom(_STATE["J2000_TDB"], _STATE["TDB"]))


def gcrf():
    return _STATE["GCRF"]


@lru_cache(maxsize=None)
def bodies(ephemeris: str = "DE440") -> dict:
    """Moon, Earth, Sun read by Orekit's own JPL DE reader from the named binary file."""
    from org.orekit.bodies import CelestialBodyFactory, JPLEphemeridesLoader

    regex = "^" + DE_FILES[ephemeris].replace(".", "\\.") + "$"
    out = {}
    for name in ("MOON", "EARTH", "SUN"):
        loader = JPLEphemeridesLoader(regex, getattr(JPLEphemeridesLoader.EphemerisType, name))
        out[name] = loader.loadCelestialBody(getattr(CelestialBodyFactory, name))
    return out


@lru_cache(maxsize=None)
def moon_icrf_frame(ephemeris: str = "DE440"):
    from oracle import BodyCenteredIcrfProvider
    from org.orekit.frames import Frame

    return Frame(gcrf(), BodyCenteredIcrfProvider(bodies(ephemeris)["MOON"], gcrf()), f"MOON_ICRF_{ephemeris}", True)


def gm_body(name: str, ephemeris: str = "DE440"):
    from oracle import GmOverrideBody

    return GmOverrideBody(bodies(ephemeris)[name], float(_STATE["scenario"]["gm_m3_s2"][name]))


# ---------------------------------------------------------------------------
# MOON_PA_DE440 from the binary PCK, read by jplephem (not CSPICE)
# ---------------------------------------------------------------------------
class PaEuler:
    """J2000 -> MOON_PA_DE440 from the 3-1-3 Euler angles in moon_pa_de440_200625.bpc (jplephem)."""

    def __init__(self, path: Path | None = None):
        from jplephem.pck import PCK

        self.pck = PCK.open(str(path or kernel_dir() / "moon_pa_de440_200625.bpc"))
        self.segs = [s for s in self.pck.segments if s.body == 31008]
        if not self.segs:
            raise LookupError("no MOON_PA_DE440 (31008) segment")

    def matrix(self, et: float) -> np.ndarray:
        for s in self.segs:
            if s.initial_second <= et <= s.final_second:
                phi, theta, psi = s.compute(2451545.0, et / 86400.0, derivative=False)
                return _r3(psi) @ _r1(theta) @ _r3(phi)
        raise ValueError(f"et {et} outside PCK coverage")


def _r1(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _r3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def pa_provider(et_lo: float, et_hi: float):
    """Java provider fed with the raw type-2 records of MOON_PA_DE440 covering [et_lo, et_hi]."""
    from jpype import JArray, JDouble
    from oracle import PckEulerProvider

    pa = PaEuler()
    seg = next(sg for sg in pa.segs if sg.initial_second <= et_lo and et_hi <= sg.final_second)
    init, intlen, rsize, n = seg.daf.read_array(seg.end_i - 3, seg.end_i)
    rsize, n = int(rsize), int(n)
    i0 = int((et_lo - init) // intlen)
    i1 = int((et_hi - init) // intlen) + 1
    raw = np.asarray(seg.daf.read_array(seg.start_i, seg.end_i - 4)).reshape(n, rsize)[i0:i1 + 1]
    return PckEulerProvider(float(init + i0 * intlen), float(intlen), rsize, JArray(JDouble)(raw.ravel().tolist()),
                            _STATE["J2000_TDB"], _STATE["TDB"])


@lru_cache(maxsize=None)
def moon_pa_frame(ephemeris: str = "DE440"):
    from org.orekit.frames import Frame

    et0 = float(_STATE["scenario"]["et0"])
    return Frame(moon_icrf_frame(ephemeris), pa_provider(et0 - 86400.0, et0 + 3 * 86400.0),
                 f"MOON_PA_DE440_PCK_{ephemeris}", False)


# ---------------------------------------------------------------------------
# GL1800F: coefficients parsed by lunar_od's loader (a shared INPUT; Orekit has no PDS SHADR reader).
# GM and reference radius are the file's own.
# ---------------------------------------------------------------------------
def gl1800f_provider(nmax: int, tab: Path):
    from jpype import JArray, JDouble
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem

    m = load_lunar_gravity_model(tab, nmax=nmax, mmax=nmax)
    c = [[float(m.cbar[n, k]) for k in range(n + 1)] for n in range(nmax + 1)]
    s = [[float(m.sbar[n, k]) for k in range(n + 1)] for n in range(nmax + 1)]
    c[0][0] = 1.0
    jc = JArray(JDouble, 2)([JArray(JDouble)(row) for row in c])
    js = JArray(JDouble, 2)([JArray(JDouble)(row) for row in s])
    return GravityFieldFactory.getNormalizedProvider(m.r_ref_m, m.mu_m3_s2, TideSystem.UNKNOWN, jc, js)


def vec(v) -> np.ndarray:
    return np.array([v.getX(), v.getY(), v.getZ()])


def pv_array(pv) -> np.ndarray:
    return np.concatenate([vec(pv.getPosition()), vec(pv.getVelocity())])
