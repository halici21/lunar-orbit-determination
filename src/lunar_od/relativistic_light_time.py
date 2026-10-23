"""Barycentric light time, Shapiro delay and station interval time scale (Phase C1-I).

The production observables (``two_way_range``, ``radiometrics`` counted Doppler, ``delta_dor``) solve
Newtonian light time in the Moon-centred inertial frame. Phase C1 showed that against the DSN/Moyer
formulation this costs 1.9-2.5 m in two-way range, up to 0.1 mm/s in counted Doppler and ~97 urad in
delta-DOR, and that the Sun and Earth Shapiro delays and the TT/TDB conversion of the measured interval
are also material (``studies/phase17/reports/phase_c1_light_time_materiality.md``).

This module evaluates the same observables under a configurable :class:`LightTimeModel`:

* ``frame="MOON"`` reproduces the production Moon-centred model (the historical default);
* ``frame="SSB"`` solves every light-time leg in the solar-system barycentric frame: an endpoint is its
  Moon-centred state plus the Moon's SSB state (SPICE, the loaded kernels) at that endpoint's own epoch;
* ``shapiro_bodies`` adds Moyer (2000) eq. 8-38 (gamma = 1) per body, evaluated at the leg's receive epoch;
* ``interval_time_scale="TT"`` reports the station-measured interval in TT instead of TDB;
* ``station_time="FULL_TOPOCENTRIC"`` (Phase C2-I) converts that interval with the full TDB - TT series and its
  topocentric terms at the station's site (:mod:`lunar_od.station_time`) instead of SPICE's geocentric DELTET
  model, and reports delta-DOR in station time, each reception at its own site (this carries the
  -(V_E . b)/c^2 term of the VLBI delay).

Per-station clock offsets follow the production ``RangeRatePhysicsConfig.station_clock_offset_s`` convention:
a receive tag ``t`` is read as ``t + b``.

Epochs are seconds past ``et0`` (SPICE ET, i.e. TDB). Spacecraft states are the production Moon-centred
histories, interpolated by the production cubic Hermite; station states come from the production
station-state providers. The production modules are imported, never modified.

Partials are arc-initial-state Jacobians: the spacecraft position sensitivity at each event (STM rows, and
the K_SRP sensitivity when the augmented history carries 48 columns) is folded in through the implicit event
equations, exactly as ``two_way_range_event_sensitivity`` does in the Moon-centred frame. Shapiro gradients
are included; they are ~1e-8 of the geometric term.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import station_time as _station_time
from .radiometrics import _interp_state, _interp_state_transition_position

C_LIGHT_MPS = 299792458.0
STATION_TIME_MODELS = ("DELTET_GEOCENTRIC", "FULL_TOPOCENTRIC")
SHAPIRO_BODY_CODES = {"SUN": "10", "EARTH": "399", "MOON": "301", "JUPITER": "5"}
DEFAULT_UPDATE_TOLERANCE_S = 1e-14
DEFAULT_MAX_ITER = 60
DEFAULT_TWO_CYCLE_TOLERANCE_S = 1e-11


class LightTimeConvergenceError(RuntimeError):
    """A light-time fixed point neither converged nor settled into a tolerated ephemeris two-cycle."""


@dataclass(frozen=True)
class LightTimeModel:
    """Light-time physics selection. The default is the historical Moon-centred Newtonian model."""

    frame: str = "MOON"
    shapiro_bodies: tuple[str, ...] = ()
    interval_time_scale: str = "TDB"
    update_tolerance_s: float = DEFAULT_UPDATE_TOLERANCE_S
    max_iter: int = DEFAULT_MAX_ITER
    two_cycle_tolerance_s: float = DEFAULT_TWO_CYCLE_TOLERANCE_S
    station_time: str = "DELTET_GEOCENTRIC"

    def __post_init__(self):
        if self.frame not in ("MOON", "SSB"):
            raise ValueError(f"frame must be 'MOON' or 'SSB'; got {self.frame!r}.")
        unknown = set(self.shapiro_bodies) - set(SHAPIRO_BODY_CODES)
        if unknown:
            raise ValueError(f"unknown Shapiro bodies {sorted(unknown)}; known: {sorted(SHAPIRO_BODY_CODES)}.")
        if self.shapiro_bodies and self.frame != "SSB":
            raise ValueError("Shapiro delay requires frame='SSB' (body positions are barycentric).")
        if self.interval_time_scale not in ("TDB", "TT"):
            raise ValueError(f"interval_time_scale must be 'TDB' or 'TT'; got {self.interval_time_scale!r}.")
        if self.station_time not in STATION_TIME_MODELS:
            raise ValueError(f"station_time must be one of {STATION_TIME_MODELS}; got {self.station_time!r}.")
        if self.station_time == "FULL_TOPOCENTRIC" and self.interval_time_scale != "TT":
            raise ValueError("station_time='FULL_TOPOCENTRIC' converts station intervals to TT; it requires "
                             "interval_time_scale='TT'.")

    @property
    def full_station_time(self) -> bool:
        return self.station_time == "FULL_TOPOCENTRIC"


#: The DSN/Moyer-faithful configuration selected by Phase C1 (Moon and Jupiter Shapiro immaterial).
DSN_LIGHT_TIME_MODEL = LightTimeModel(frame="SSB", shapiro_bodies=("SUN", "EARTH"), interval_time_scale="TT")
#: Phase C2-I: the C1 model with the station-time relation of Phase C2 (full TDB - TT series, topocentric).
DSN_C2_LIGHT_TIME_MODEL = LightTimeModel(frame="SSB", shapiro_bodies=("SUN", "EARTH"), interval_time_scale="TT",
                                         station_time="FULL_TOPOCENTRIC")


@dataclass
class BarycentricEphemeris:
    """SPICE access for one scenario epoch: the Moon's SSB state, body positions, GM, d(TDB-TT)/dTDB."""

    et0_s: float
    _moon: dict = field(default_factory=dict, repr=False)
    _gm: dict = field(default_factory=dict, repr=False)

    def _ssb_state(self, code: str, t_s: float) -> np.ndarray:
        """SSB state [m, m/s] of ``code`` at ``et0 + t_s`` without epoch quantisation.

        ``et0 + t_s`` rounds to ulp(et0) ~ 1.2e-7 s near 2027; at the Moon's ~30 km/s barycentric speed that
        is a 3.5 mm position step, which shows up as light-time two-cycles and ~1e-5 m/s Doppler noise.
        The ephemeris is read at the rounded epoch and moved by velocity x the exact rounding error
        (Fast2Sum, |et0| >= |t_s|).
        """
        import spiceypy as spice
        t = float(t_s)
        et = self.et0_s + t
        rounding = t - (et - self.et0_s)
        state = np.asarray(spice.spkezr(code, et, "J2000", "NONE", "SSB")[0], dtype=float) * 1e3
        state[:3] += state[3:6] * rounding
        return state

    def moon_state(self, t_s: float) -> np.ndarray:
        """The Moon's SSB state [m, m/s] at ``t_s`` (J2000)."""
        key = float(t_s)
        state = self._moon.get(key)
        if state is None:
            state = self._ssb_state("301", key)
            self._moon[key] = state
        return state

    def body_position(self, name: str, t_s: float) -> np.ndarray:
        return self._ssb_state(SHAPIRO_BODY_CODES[name], t_s)[:3]

    def gm(self, name: str) -> float:
        if name not in self._gm:
            import spiceypy as spice
            self._gm[name] = float(spice.bodvrd(SHAPIRO_BODY_CODES[name], "GM", 1)[1][0]) * 1e9
        return self._gm[name]

    def tdb_minus_tt_rate(self, t_s: float) -> float:
        """d(TDB - TT)/dTDB at ``t_s``: the derivative of SPICE's own model TDB - TT = K sin(E),
        E = M + EB sin(M), M = M0 + M1 et (leapseconds-kernel DELTET constants).

        Analytic on purpose: a finite difference of ``unitim`` at et ~ 8e8 s is quantised at ulp(et)
        ~ 1.2e-7 s, i.e. ~6e-10 noise on a ~3e-10 rate.
        """
        import spiceypy as spice
        k = spice.gdpool("DELTET/K", 0, 1)[0]
        eb = spice.gdpool("DELTET/EB", 0, 1)[0]
        m0, m1 = spice.gdpool("DELTET/M", 0, 2)
        m = m0 + m1 * (self.et0_s + float(t_s))
        e = m + eb * np.sin(m)
        return float(k * np.cos(e) * (1.0 + eb * np.cos(m)) * m1)


# ======================================================================
# frame translation, Shapiro
# ======================================================================
def _to_frame(model: LightTimeModel, eph: BarycentricEphemeris, t_s: float, state_mci: np.ndarray) -> np.ndarray:
    """A Moon-centred 6-state expressed in the model's frame at its own epoch."""
    if model.frame == "MOON":
        return np.asarray(state_mci, dtype=float)[:6]
    return np.asarray(state_mci, dtype=float)[:6] + eph.moon_state(t_s)


def shapiro_delay_s(model: LightTimeModel, eph: BarycentricEphemeris, r_tx: np.ndarray, r_rx: np.ndarray,
                    t_rx_s: float, d12: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Moyer eq. 8-38 summed over ``model.shapiro_bodies``; returns (delay, d/dr_tx, d/dr_rx).

    ``d12`` is the precise leg vector r_rx - r_tx when the caller has it (see :func:`_relative`)."""
    total, g_tx, g_rx = 0.0, np.zeros(3), np.zeros(3)
    if not model.shapiro_bodies:
        return total, g_tx, g_rx
    d12 = r_rx - r_tx if d12 is None else np.asarray(d12, dtype=float)
    r12 = float(np.linalg.norm(d12))
    u12 = d12 / r12
    for name in model.shapiro_bodies:
        b = eph.body_position(name, t_rx_s)
        a1, a2 = r_tx - b, r_rx - b
        r1, r2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
        k = 2.0 * eph.gm(name) / C_LIGHT_MPS ** 3
        num, den = r1 + r2 + r12, r1 + r2 - r12
        total += k * np.log(num / den)
        # d ln(num/den) = (dr1 + dr2)(1/num - 1/den) + dr12 (1/num + 1/den)
        s_sum, s_12 = 1.0 / num - 1.0 / den, 1.0 / num + 1.0 / den
        g_tx += k * (s_sum * a1 / r1 - s_12 * u12)
        g_rx += k * (s_sum * a2 / r2 + s_12 * u12)
    return total, g_tx, g_rx


# ======================================================================
# light-time legs
# ======================================================================
@dataclass(frozen=True)
class Leg:
    """One converged light-time leg in the model's frame."""

    t_tx_s: float
    t_rx_s: float
    tx: np.ndarray                 # transmitter 6-state in the model frame at t_tx
    rx: np.ndarray                 # receiver 6-state in the model frame at t_rx
    rel: np.ndarray                # precise r_rx - r_tx (see _relative)
    newtonian_s: float             # |rel| / c
    shapiro_s: float
    shapiro_grad_tx: np.ndarray
    shapiro_grad_rx: np.ndarray
    two_cycle: bool

    @property
    def light_time_s(self) -> float:
        return self.newtonian_s + self.shapiro_s


def _fixed_point(update: Callable[[float], float], tau0: float, model: LightTimeModel, label: str):
    """Iterate tau <- update(tau); accept convergence or a two-cycle within the ephemeris tolerance."""
    history = [tau0]
    tau = tau0
    for _ in range(model.max_iter):
        new = update(tau)
        if abs(new - tau) < model.update_tolerance_s:
            return new, False
        history.append(new)
        tau = new
    a, b, c = history[-3], history[-2], history[-1]
    if abs(c - a) < model.update_tolerance_s and abs(c - b) <= model.two_cycle_tolerance_s:
        return 0.5 * (b + c), True
    raise LightTimeConvergenceError(
        f"{label} light-time iteration did not converge (last update {abs(c - b):.3e} s; "
        f"two-cycle tolerance {model.two_cycle_tolerance_s:.1e} s).")


def _relative(model, eph, t_tx, t_rx, tau, tx_mci, rx_mci) -> np.ndarray:
    """Receiver minus transmitter position in the model's frame, without large-number cancellation.

    Barycentric positions are ~1.5e11 m, so their rounding (~2e-5 m per component) would put a ~1e-13 s floor
    under every SSB light time and ~1e-6 m/s of noise into counted Doppler. The leg vector is therefore formed
    from the Moon-centred end states plus the Moon's SSB displacement over the light time ``tau`` (trapezoid on
    the Moon's SSB velocity; the jerk term is ~1e-9 m for a 1.3 s leg).
    """
    d = np.asarray(rx_mci, dtype=float)[:3] - np.asarray(tx_mci, dtype=float)[:3]
    if model.frame == "SSB":
        d = d + 0.5 * (eph.moon_state(t_tx)[3:6] + eph.moon_state(t_rx)[3:6]) * tau
    return d


def _leg(model, eph, *, t_rx=None, rx_fn=None, rx_fixed=None, t_tx=None, tx_fn=None, tx_fixed=None, label):
    """Solve one leg. Exactly one end is fixed (an epoch with a known state), the other moves.

    ``rx_fn``/``tx_fn`` return Moon-centred 6-states at an epoch; ``rx_fixed``/``tx_fixed`` are the fixed
    end's Moon-centred state at ``t_rx``/``t_tx``. The light time ``tau`` is the iteration variable, so the
    leg never uses an epoch difference.
    """
    c = C_LIGHT_MPS
    shapiro_on = bool(model.shapiro_bodies)

    def light_time(t_tx_, t_rx_, tau, tx_m, rx_m):
        d = _relative(model, eph, t_tx_, t_rx_, tau, tx_m, rx_m)
        lt = float(np.linalg.norm(d)) / c
        if shapiro_on:
            lt += shapiro_delay_s(model, eph, _to_frame(model, eph, t_tx_, tx_m)[:3],
                                  _to_frame(model, eph, t_rx_, rx_m)[:3], t_rx_, d)[0]
        return lt

    if t_rx is not None:                      # receiver fixed, solve the transmit epoch backward
        rx_m = np.asarray(rx_fixed, dtype=float)[:6]
        tau0 = float(np.linalg.norm(rx_m[:3] - tx_fn(t_rx)[:3])) / c
        tau, cyc = _fixed_point(lambda tau: light_time(t_rx - tau, t_rx, tau, tx_fn(t_rx - tau), rx_m),
                                tau0, model, label)
        t_tx_s, t_rx_s = t_rx - tau, t_rx
        tx_m = np.asarray(tx_fn(t_tx_s), dtype=float)[:6]
    else:                                     # transmitter fixed, solve the receive epoch forward
        tx_m = np.asarray(tx_fixed, dtype=float)[:6]
        tau0 = float(np.linalg.norm(rx_fn(t_tx)[:3] - tx_m[:3])) / c
        tau, cyc = _fixed_point(lambda tau: light_time(t_tx, t_tx + tau, tau, tx_m, rx_fn(t_tx + tau)),
                                tau0, model, label)
        t_tx_s, t_rx_s = t_tx, t_tx + tau
        rx_m = np.asarray(rx_fn(t_rx_s), dtype=float)[:6]
    d = _relative(model, eph, t_tx_s, t_rx_s, tau, tx_m, rx_m)
    tx, rx = _to_frame(model, eph, t_tx_s, tx_m), _to_frame(model, eph, t_rx_s, rx_m)
    sh, g_tx, g_rx = shapiro_delay_s(model, eph, tx[:3], rx[:3], t_rx_s, d)
    return Leg(t_tx_s, t_rx_s, tx, rx, d, float(np.linalg.norm(d)) / c, sh, g_tx, g_rx, cyc)


def exact_epoch_station_state(station_state: Callable, et0_s: float, t_grid_s, earth_pos_mci_m, earth_vel_mci_mps
                              ) -> Callable[[float], np.ndarray]:
    """Wrap a production station-state provider so its Earth-rotation part is free of epoch quantisation.

    The production providers evaluate ``sxform`` at ``et0 + t``, quantised at ulp(et0) ~ 1.2e-7 s: at a DSN
    station's ~0.4 km/s rotation speed that is a ~4.5e-5 m position step (~4e-7 m/s of counted-Doppler noise).
    The Earth's Moon-centred motion is interpolated in ``t`` and is not quantised, so only the station's
    velocity relative to the Earth is used to move the position by the exact rounding error.
    """
    t_grid = np.asarray(t_grid_s, dtype=float)
    earth = np.hstack([np.asarray(earth_pos_mci_m, dtype=float), np.asarray(earth_vel_mci_mps, dtype=float)])
    et0 = float(et0_s)

    def state(t_s: float) -> np.ndarray:
        t = float(t_s)
        s = np.array(station_state(t), dtype=float)[:6]
        rounding = t - ((et0 + t) - et0)
        if rounding:
            s[:3] += (s[3:6] - _interp_state(t_grid, earth, t)[3:6]) * rounding
        return s
    return state


# ======================================================================
# spacecraft history access
# ======================================================================
class SpacecraftHistory:
    """Production cubic-Hermite access to a Moon-centred history; 42/48-column histories add sensitivities."""

    def __init__(self, t_grid_s, history):
        self.t_grid = np.asarray(t_grid_s, dtype=float).reshape(-1)
        self.hist = np.asarray(history, dtype=float)
        self.n_par = 0
        if self.hist.shape[1] >= 42:
            self.n_par = 7 if self.hist.shape[1] >= 48 else 6
            self.phi = np.array([row.reshape((6, 6), order="F") for row in self.hist[:, 6:42]], dtype=float)

    def state(self, t_s: float) -> np.ndarray:
        return _interp_state(self.t_grid, self.hist[:, :6], t_s)

    def position_sensitivity(self, t_s: float) -> np.ndarray:
        """d r_sc(t) / d[x0 (, K)] as a (3, n_par) matrix."""
        if not self.n_par:
            raise ValueError("partials need an augmented history (state + column-major STM [+ dx/dK]).")
        blocks = [_interp_state_transition_position(self.t_grid, self.phi, t_s)]
        if self.n_par == 7:
            blocks.append(_interp_state(self.t_grid, self.hist[:, 42:48], t_s)[:3].reshape(3, 1))
        return np.hstack(blocks)


# ======================================================================
# observables
# ======================================================================
@dataclass(frozen=True)
class TwoWaySolution:
    t3_s: float
    downlink: Leg                  # spacecraft (t2) -> station (t3)
    uplink: Leg                    # station (t1) -> spacecraft (t2)
    tdb_minus_tt_rate: float
    media_group_s: float = 0.0     # Phase C5-I: up + down media group delay (range)
    media_phase_s: float = 0.0     # up + down media phase delay (counted Doppler)

    @property
    def t1_s(self) -> float:
        return self.uplink.t_tx_s

    @property
    def round_trip_s(self) -> float:
        """Local delays, never an epoch difference (Phase 17C)."""
        return self.downlink.light_time_s + self.uplink.light_time_s

    @property
    def two_cycle(self) -> bool:
        return self.downlink.two_cycle or self.uplink.two_cycle


def solve_two_way(t3_s: float, station_state: Callable, sc: SpacecraftHistory, eph: BarycentricEphemeris,
                  model: LightTimeModel, *, site: "_station_time.Site | None" = None,
                  clock_offset_s: float = 0.0, media=None, media_site=None) -> TwoWaySolution:
    """Receive-tagged two-way chain t3 -> t2 -> t1 (zero transponder delay, the R3/M3 production default).

    ``site`` (the station's :class:`lunar_od.station_time.Site`) is required by ``station_time='FULL_TOPOCENTRIC'``
    and ignored otherwise. ``clock_offset_s`` reads the receive tag as ``t3 + b`` (production convention); the
    interval of one station clock is unchanged by it. ``media`` (:class:`lunar_od.media.MediaModel`, with the
    station's ``media_site``) adds the media delay of each leg at its ground event (Phase C5-I); its feedback on
    the event epochs (~1e-8 relative) is neglected.
    """
    t3 = float(t3_s)
    if clock_offset_s:
        t3 = t3 + float(clock_offset_s)
    down = _leg(model, eph, t_rx=t3, rx_fixed=station_state(t3), tx_fn=sc.state, label="downlink")
    up = _leg(model, eph, t_rx=down.t_tx_s, rx_fixed=sc.state(down.t_tx_s), tx_fn=station_state, label="uplink")
    if model.interval_time_scale != "TT":
        rate = 0.0
    elif model.full_station_time:
        if site is None:
            raise ValueError("station_time='FULL_TOPOCENTRIC' needs the station's site.")
        # D_site(t3) - D_site(t1) = round trip x the rate at the interval midpoint (exact to ~1e-20 s)
        mid = t3 - 0.5 * (down.light_time_s + up.light_time_s)
        rate = float(_station_time.tdb_minus_tt_rate(eph.et0_s, mid, site))
    else:
        rate = eph.tdb_minus_tt_rate(t3)
    if media is None:
        return TwoWaySolution(t3, down, up, rate)
    if media_site is None:
        raise ValueError("media delays need the station's media_site (lunar_od.media.MediaSite).")
    et0 = eph.et0_s
    gd, pd, _ = media.leg_delays(media_site, et0 + down.t_rx_s, -down.rel, uplink=False)
    gu, pu, _ = media.leg_delays(media_site, et0 + up.t_tx_s, up.rel, uplink=True)
    return TwoWaySolution(t3, down, up, rate, (gd + gu) / C_LIGHT_MPS, (pd + pu) / C_LIGHT_MPS)


def _interval_scale(sol: TwoWaySolution) -> float:
    return 1.0 - sol.tdb_minus_tt_rate


def two_way_range_m(sol: TwoWaySolution) -> float:
    """Half round trip [m], in the model's interval time scale (plus the media group delay, Phase C5-I)."""
    value = 0.5 * C_LIGHT_MPS * sol.round_trip_s * _interval_scale(sol)
    if sol.media_group_s:
        value = value + 0.5 * C_LIGHT_MPS * sol.media_group_s
    return value


def _round_trip_partial(sol: TwoWaySolution, sc: SpacecraftHistory) -> np.ndarray:
    """d(round trip)/d[x0 (, K)] from the implicit uplink/downlink equations (receive epoch t3 fixed).

    Unknowns y = [t1, t2]; G_d = t3 - t2 - |r_st(t3) - r_sc(t2)|/c - S_d, G_u = t2 - t1 - |r_sc(t2) - r_st(t1)|/c - S_u.
    Round trip = t3 - t1, so d/dx = -dt1/dx.
    """
    c = C_LIGHT_MPS
    d, u = sol.downlink, sol.uplink
    t2 = d.t_tx_s
    m = sc.position_sensitivity(t2)
    los_d = d.rel / np.linalg.norm(d.rel)     # spacecraft -> station
    los_u = u.rel / np.linalg.norm(u.rel)     # station -> spacecraft
    v_sc, v_st1 = d.tx[3:6], u.tx[3:6]
    # d|r_rx - r_tx|/d r_tx = -los, /d r_rx = +los; Shapiro gradients add to the same position partials.
    dGd_dr2 = (los_d / c) - d.shapiro_grad_tx          # w.r.t. spacecraft position at t2 (transmitter of downlink)
    dGu_dr2 = -(los_u / c) - u.shapiro_grad_rx         # spacecraft is the uplink receiver
    dGu_dr1 = (los_u / c) - u.shapiro_grad_tx          # station transmitter position at t1
    g_y = np.array([[0.0, -1.0 + float(dGd_dr2 @ v_sc)],
                    [-1.0 + float(dGu_dr1 @ v_st1), 1.0 + float(dGu_dr2 @ v_sc)]])
    g_x = np.vstack([dGd_dr2 @ m, dGu_dr2 @ m])
    dy = np.linalg.solve(g_y, -g_x)
    return -dy[0]


def two_way_range_partial(sol: TwoWaySolution, sc: SpacecraftHistory) -> np.ndarray:
    return 0.5 * C_LIGHT_MPS * _interval_scale(sol) * _round_trip_partial(sol, sc)


def _round_trip_receive_epoch_partial(sol: TwoWaySolution) -> float:
    """d(round trip)/d t3 [s/s] from the same implicit equations: dG_d/dt3 = 1 - (los_d/c + dS_d/dr_rx) . v_st(t3),
    dG_u/dt3 = 0; round trip = t3 - t1, so d/dt3 = 1 - dt1/dt3. The interval scale's own epoch dependence (~1e-10
    relative) is neglected, as in the state partials."""
    c = C_LIGHT_MPS
    d, u = sol.downlink, sol.uplink
    los_d = d.rel / np.linalg.norm(d.rel)
    los_u = u.rel / np.linalg.norm(u.rel)
    v_sc, v_st1, v_st3 = d.tx[3:6], u.tx[3:6], d.rx[3:6]
    dGd_dr2 = (los_d / c) - d.shapiro_grad_tx
    dGu_dr2 = -(los_u / c) - u.shapiro_grad_rx
    dGu_dr1 = (los_u / c) - u.shapiro_grad_tx
    g_y = np.array([[0.0, -1.0 + float(dGd_dr2 @ v_sc)],
                    [-1.0 + float(dGu_dr1 @ v_st1), 1.0 + float(dGu_dr2 @ v_sc)]])
    g_t3 = np.array([1.0 - float(((los_d / c) + d.shapiro_grad_rx) @ v_st3), 0.0])
    dy = np.linalg.solve(g_y, -g_t3)
    return 1.0 - float(dy[0])


def two_way_range_clock_partial(sol: TwoWaySolution) -> float:
    """d(two-way range)/d(station clock offset b) [m/s]: the receive tag is read as t3 + b."""
    return 0.5 * C_LIGHT_MPS * _interval_scale(sol) * _round_trip_receive_epoch_partial(sol)


def counted_doppler(t_mid_s: float, count_interval_s: float, station_state: Callable, sc: SpacecraftHistory,
                    eph: BarycentricEphemeris, model: LightTimeModel, *, with_partial: bool = False,
                    site: "_station_time.Site | None" = None, clock_offset_s: float = 0.0,
                    with_clock_partial: bool = False, media=None, media_site=None):
    """Receive-tagged counted Doppler, m/s equivalent: c/(2 Tc) (tau_end - tau_start), each tau in the
    model's interval time scale (plus the media phase delay, Phase C5-I). Returns (value, partial or None,
    (start, end) solutions); with ``with_clock_partial`` a fourth item, d(value)/d(clock offset) [m/s per s]."""
    tc = float(count_interval_s)
    kw = dict(site=site, clock_offset_s=clock_offset_s)
    if media is not None:
        kw.update(media=media, media_site=media_site)
    a = solve_two_way(t_mid_s - 0.5 * tc, station_state, sc, eph, model, **kw)
    b = solve_two_way(t_mid_s + 0.5 * tc, station_state, sc, eph, model, **kw)
    k = C_LIGHT_MPS / (2.0 * tc)
    value = k * (b.round_trip_s * _interval_scale(b) - a.round_trip_s * _interval_scale(a))
    if b.media_phase_s or a.media_phase_s:
        value = value + k * (b.media_phase_s - a.media_phase_s)
    partial = None
    if with_partial:
        partial = k * (_interval_scale(b) * _round_trip_partial(b, sc) - _interval_scale(a) * _round_trip_partial(a, sc))
    if with_clock_partial:
        clock = k * (_interval_scale(b) * _round_trip_receive_epoch_partial(b)
                     - _interval_scale(a) * _round_trip_receive_epoch_partial(a))
        return value, partial, (a, b), clock
    return value, partial, (a, b)


@dataclass(frozen=True)
class DeltaDorSolution:
    t_obs_s: float
    leg_a: Leg                     # spacecraft (t_tx) -> station A (t_obs, frozen)
    leg_b: Leg                     # spacecraft (t_tx) -> station B (t_B)

    @property
    def differential_delay_s(self) -> float:
        """t_B - t_obs as a difference of local delays."""
        return self.leg_b.light_time_s - self.leg_a.light_time_s

    @property
    def two_cycle(self) -> bool:
        return self.leg_a.two_cycle or self.leg_b.two_cycle


def solve_delta_dor(t_obs_s: float, station_a_state_at_obs: np.ndarray, station_b_state: Callable,
                    sc: SpacecraftHistory, eph: BarycentricEphemeris, model: LightTimeModel) -> DeltaDorSolution:
    """Common transmit event (production contract): station A fixed at t_obs, t_tx backward, t_B forward."""
    a = _leg(model, eph, t_rx=float(t_obs_s), rx_fixed=station_a_state_at_obs, tx_fn=sc.state, label="delta-DOR A")
    b = _leg(model, eph, t_tx=a.t_tx_s, tx_fixed=sc.state(a.t_tx_s), rx_fn=station_b_state, label="delta-DOR B")
    return DeltaDorSolution(float(t_obs_s), a, b)


def delta_dor_partial(sol: DeltaDorSolution, sc: SpacecraftHistory) -> np.ndarray:
    """d(t_B - t_obs)/d[x0 (, K)]. Unknowns y = [t_tx, t_B];
    G_A = t_obs - t_tx - |r_A - r_sc(t_tx)|/c - S_A, G_B = t_B - t_tx - |r_B(t_B) - r_sc(t_tx)|/c - S_B."""
    c = C_LIGHT_MPS
    a, b = sol.leg_a, sol.leg_b
    m = sc.position_sensitivity(a.t_tx_s)
    los_a = a.rel / np.linalg.norm(a.rel)
    los_b = b.rel / np.linalg.norm(b.rel)
    v_sc, v_b = a.tx[3:6], b.rx[3:6]
    dGa_drsc = (los_a / c) - a.shapiro_grad_tx
    dGb_drsc = (los_b / c) - b.shapiro_grad_tx
    dGb_drb = -(los_b / c) - b.shapiro_grad_rx
    g_y = np.array([[-1.0 + float(dGa_drsc @ v_sc), 0.0],
                    [-1.0 + float(dGb_drsc @ v_sc), 1.0 + float(dGb_drb @ v_b)]])
    g_x = np.vstack([dGa_drsc @ m, dGb_drsc @ m])
    return np.linalg.solve(g_y, -g_x)[1]


@dataclass(frozen=True)
class DeltaDorStationTime:
    """Delta-DOR as the two station clocks measure it (Phase C2-I)."""

    solution: DeltaDorSolution
    delay_s: float                     # measured delay: tag at B minus tag at A
    station_time_correction_s: float   # -[D_B(t_B) - D_A(t_A)], D = TDB - TT at each site (0 under DELTET)
    clock_term_s: float                # b_A - b_B


def delta_dor_station_time(t_obs_s: float, station_a_state: Callable, station_b_state: Callable,
                           sc: SpacecraftHistory, eph: BarycentricEphemeris, model: LightTimeModel, *,
                           sites: tuple = (None, None),
                           clock_offsets_s: tuple[float, float] = (0.0, 0.0),
                           media=None, media_sites: tuple = (None, None)) -> DeltaDorStationTime:
    """Delta-DOR in station time.

    A's tag ``t_obs`` is read as ``t_obs + b_A`` (the true reception at A, production convention); B's clock reads
    its true reception ``t_B`` as ``t_B - b_B``, so the measured delay is D(t_obs + b_A) + b_A - b_B. Under
    ``station_time='FULL_TOPOCENTRIC'`` each reception is converted to its own station's TT: D_TT = D_TDB -
    [D_B(t_B) - D_A(t_A)] (``sites`` required); otherwise the delay is the TDB delay of :func:`solve_delta_dor`
    (the C1-I contract).
    """
    b_a, b_b = (float(x) for x in clock_offsets_s)
    t_a = float(t_obs_s) + b_a if b_a else float(t_obs_s)
    sol = solve_delta_dor(t_a, np.asarray(station_a_state(t_a), dtype=float), station_b_state, sc, eph, model)
    corr = 0.0
    if model.full_station_time:
        site_a, site_b = sites
        if site_a is None or site_b is None:
            raise ValueError("station_time='FULL_TOPOCENTRIC' needs both stations' sites.")
        corr = -(float(_station_time.tdb_minus_tt(eph.et0_s, sol.leg_b.t_rx_s, site_b))
                 - float(_station_time.tdb_minus_tt(eph.et0_s, t_a, site_a)))
    clock = b_a - b_b
    delay = sol.differential_delay_s
    if corr:
        delay = delay + corr
    if clock:
        delay = delay + clock
    if media is not None:
        # Phase C5-I: the two downlinks' media group delays (delta-DOR is a group measurement)
        ma, mb = media_sites
        if ma is None or mb is None:
            raise ValueError("media delays need both stations' media_sites.")
        ga, _, _ = media.leg_delays(ma, eph.et0_s + sol.leg_a.t_rx_s, -sol.leg_a.rel, uplink=False)
        gb, _, _ = media.leg_delays(mb, eph.et0_s + sol.leg_b.t_rx_s, -sol.leg_b.rel, uplink=False)
        delay = delay + (gb - ga) / C_LIGHT_MPS
    return DeltaDorStationTime(sol, delay, corr, clock)


def delta_dor_clock_partials(sol: DeltaDorSolution) -> tuple[float, float]:
    """(d delay/d b_A, d delay/d b_B) [s/s]. With A's epoch moving, dG_A/dt_obs = 1 - (los_a/c + dS_A/dr_A) . v_A;
    d(measured)/d b_A = d(t_B - t_obs)/dt_obs + 1 = dt_B/dt_obs, and d/d b_B = -1."""
    c = C_LIGHT_MPS
    a, b = sol.leg_a, sol.leg_b
    los_a = a.rel / np.linalg.norm(a.rel)
    los_b = b.rel / np.linalg.norm(b.rel)
    v_sc, v_a, v_b = a.tx[3:6], a.rx[3:6], b.rx[3:6]
    dGa_drsc = (los_a / c) - a.shapiro_grad_tx
    dGb_drsc = (los_b / c) - b.shapiro_grad_tx
    dGb_drb = -(los_b / c) - b.shapiro_grad_rx
    g_y = np.array([[-1.0 + float(dGa_drsc @ v_sc), 0.0],
                    [-1.0 + float(dGb_drsc @ v_sc), 1.0 + float(dGb_drb @ v_b)]])
    g_t = np.array([1.0 - float(((los_a / c) + a.shapiro_grad_rx) @ v_a), 0.0])
    dy = np.linalg.solve(g_y, -g_t)
    return float(dy[1]), -1.0


def one_way_range(t3_s: float, station_state: Callable, sc: SpacecraftHistory, eph: BarycentricEphemeris,
                  model: LightTimeModel, *, with_partial: bool = False):
    """Downlink-only light distance c * tau_d to a receive epoch (the RAZEL one-way convention)."""
    t3 = float(t3_s)
    leg = _leg(model, eph, t_rx=t3, rx_fixed=station_state(t3), tx_fn=sc.state, label="one-way")
    value = C_LIGHT_MPS * leg.light_time_s
    partial = None
    if with_partial:
        c = C_LIGHT_MPS
        los = leg.rel / np.linalg.norm(leg.rel)
        dg_dr = (los / c) - leg.shapiro_grad_tx
        g_t = -1.0 + float(dg_dr @ leg.tx[3:6])
        dt2 = -(dg_dr @ sc.position_sensitivity(leg.t_tx_s)) / g_t
        partial = -c * dt2
    return value, partial, leg


# ======================================================================
# pass-level API: generation, nominal values + Jacobian, residuals (estimator dispatch)
# ======================================================================
#: ``PassGeometry.measurement_model_profile`` that routes the two-way range estimators to this module.
TWO_WAY_SSB_PROFILE = "two_way_light_time_ssb"


def _model_metadata(model: LightTimeModel) -> dict:
    return {"frame": model.frame, "shapiro_bodies": list(model.shapiro_bodies),
            "interval_time_scale": model.interval_time_scale, "update_tolerance_s": model.update_tolerance_s,
            "max_iter": model.max_iter, "two_cycle_tolerance_s": model.two_cycle_tolerance_s,
            "station_states": "exact_epoch", "leg_vectors": "relative", "station_time": model.station_time}


def pass_light_time_model(pass_geo) -> LightTimeModel:
    """The model a ``two_way_light_time_ssb`` pass was generated with (from its metadata). Metadata written
    before Phase C2-I has no ``station_time`` key and reads as the C1-I relation (DELTET_GEOCENTRIC)."""
    meta = (pass_geo.measurement_metadata or {}).get("light_time_model")
    if meta is None:
        raise ValueError(f"pass geometry with profile {TWO_WAY_SSB_PROFILE!r} carries no light_time_model metadata.")
    return LightTimeModel(frame=meta["frame"], shapiro_bodies=tuple(meta["shapiro_bodies"]),
                          interval_time_scale=meta["interval_time_scale"],
                          update_tolerance_s=meta["update_tolerance_s"], max_iter=meta["max_iter"],
                          two_cycle_tolerance_s=meta["two_cycle_tolerance_s"],
                          station_time=meta.get("station_time", "DELTET_GEOCENTRIC"))


def pass_station_kwargs(pass_geo) -> list[dict]:
    """Per-station ``site`` and ``clock_offset_s`` for :func:`solve_two_way`, in the pass's station order. The
    sites come from ``pass_geo.stations[i].r_ecef_m``; clock offsets from the metadata key
    ``station_clock_offsets_s`` (absent: all zero); ``media`` / ``media_site`` from the metadata key ``media``
    (Phase C5-I; absent: none)."""
    meta = pass_geo.measurement_metadata or {}
    offsets = meta.get("station_clock_offsets_s")
    stations = list(pass_geo.stations)
    offsets = [0.0] * len(stations) if offsets is None else [float(b) for b in offsets]
    if len(offsets) != len(stations):
        raise ValueError(f"station_clock_offsets_s has {len(offsets)} entries for {len(stations)} stations.")
    out = [dict(site=_station_time.Site.from_itrf_m(st.r_ecef_m), clock_offset_s=b)
           for st, b in zip(stations, offsets)]
    if meta.get("media") is not None:
        from .media import MediaModel, MediaSite
        model = MediaModel.from_metadata(meta["media"])
        for kw, st in zip(out, stations):
            kw.update(media=model, media_site=MediaSite.from_station(st))
    return out


def _pass_context(pass_geo):
    from .two_way_range import _station_state_providers_for_pass, two_way_range_config
    cfg = two_way_range_config(pass_geo.two_way_range)
    if cfg.transponder_delay_s != 0.0 or cfg.convention != "delay_calibrated_half_round_trip":
        raise NotImplementedError("the barycentric two-way range supports zero transponder delay and the "
                                  "delay-calibrated half-round-trip convention only.")
    stations = [exact_epoch_station_state(lambda t, p=p: np.asarray(p.state(t), dtype=float), pass_geo.et0_s,
                                          pass_geo.t_s, pass_geo.earth_pos_mci_m, pass_geo.earth_vel_mci_mps)
                for p in _station_state_providers_for_pass(pass_geo)]
    disp = (pass_geo.measurement_metadata or {}).get("station_displacement")
    if disp is not None:
        stations = [displaced_station_state(st, pass_geo.et0_s, s.r_ecef_m, **disp)
                    for st, s in zip(stations, pass_geo.stations)]
    return pass_light_time_model(pass_geo), stations, BarycentricEphemeris(float(pass_geo.et0_s))


#: Central-difference step [s] of the solid-tide displacement rate (the tide changes on hour scales).
TIDE_RATE_STEP_S = 10.0


def displaced_station_state(station_state: Callable, et0_s: float, r_ecef_m, *, solid_tide: bool = True,
                            coordinate_frame: str = "ITRF93", include_scale: bool = True) -> Callable:
    """Wrap a production station provider (Phase C3-I) with an Earth-fixed station offset rotated to J2000:

    * ``coordinate_frame='ITRF2020'``: the provider's ECEF vector is read as ITRF 2020 and moved to ITRF 93 (the
      frame of SPICE's Earth orientation) with :func:`lunar_od.station_frames.itrf2020_to_itrf93` at each epoch;
      ``include_scale=False`` drops the EPSG scale (Orekit's transcription);
    * ``solid_tide``: the IERS 2010 solid Earth tide (:func:`lunar_od.solid_tide.solid_tide_at_et`).

    The offset d(t) enters as [R d, R' d + R d'] with R, R' from ``sxform('ITRF93', 'J2000')`` and d' a central
    difference (h = TIDE_RATE_STEP_S) of the tide.
    """
    import spiceypy as spice

    from .solid_tide import solid_tide_at_et
    from .station_frames import itrf2020_to_itrf93

    if coordinate_frame not in ("ITRF93", "ITRF2020"):
        raise ValueError(f"coordinate_frame must be 'ITRF93' or 'ITRF2020'; got {coordinate_frame!r}.")
    r = np.asarray(r_ecef_m, dtype=float).reshape(3)
    et0 = float(et0_s)

    def offset(t: float) -> np.ndarray:
        d = np.zeros(3)
        if coordinate_frame == "ITRF2020":
            d = d + (itrf2020_to_itrf93(r, et0 + t, include_scale=include_scale) - r)
        if solid_tide:
            d = d + solid_tide_at_et(r, et0 + t)
        return d

    def state(t_s: float) -> np.ndarray:
        t = float(t_s)
        s = np.array(station_state(t), dtype=float)[:6]
        x = np.asarray(spice.sxform("ITRF93", "J2000", et0 + t), dtype=float)
        d = offset(t)
        h = TIDE_RATE_STEP_S
        d_dot = (offset(t + h) - offset(t - h)) / (2.0 * h)
        s[:3] += x[:3, :3] @ d
        s[3:] += x[3:, :3] @ d + x[3:, 3:] @ d_dot
        return s
    return state


def generate_two_way_range_measurements_ssb(t_pass_s, state_history_mci, stations, vis_mask_raw, get_earth_pos,
                                            get_earth_vel, et0, *, model: LightTimeModel = DSN_LIGHT_TIME_MODEL,
                                            noise: bool = True, rng: np.random.Generator | None = None,
                                            arc_id: int | None = None, config=None, noise_seed: int | None = None,
                                            station_clock_offsets_s=None, station_displacement: dict | None = None,
                                            media=None):
    """Receive-tagged two-way range under ``model`` (default: the DSN model). ``station_displacement`` (Phase
    C3-I; keyword arguments of :func:`displaced_station_state`) is stored in the metadata like the clock offsets.

    The receive tags, station rows and dropped-row rule are the production generator's
    (``two_way_range.generate_two_way_range_measurements``, run noise-free); every value is then recomputed
    with this module and noise is added with the station's ``sigma_range_m``. The returned pass geometry
    carries ``measurement_model_profile = 'two_way_light_time_ssb'`` and the model in its metadata, which
    routes the two-way range estimators here. ``station_clock_offsets_s`` (one per station, Phase C2-I) is
    stored in the metadata and applied to generation and to every later prediction of the pass.
    """
    from dataclasses import replace

    from .two_way_range import generate_two_way_range_measurements

    obs, pass_geo = generate_two_way_range_measurements(t_pass_s, state_history_mci, stations, vis_mask_raw,
                                                        get_earth_pos, get_earth_vel, et0, noise=False,
                                                        arc_id=arc_id, config=config, noise_seed=noise_seed)
    metadata = dict(pass_geo.measurement_metadata or {})
    metadata.update(measurement_model_profile=TWO_WAY_SSB_PROFILE, light_time_model=_model_metadata(model),
                    noise_enabled=bool(noise))
    if station_clock_offsets_s is not None:
        metadata["station_clock_offsets_s"] = [float(b) for b in station_clock_offsets_s]
    if station_displacement is not None:
        metadata["station_displacement"] = dict(station_displacement)
    if media is not None:
        metadata["media"] = media.to_metadata()
    pass_geo = replace(pass_geo, measurement_model_profile=TWO_WAY_SSB_PROFILE, measurement_metadata=metadata)
    _, station_states, eph = _pass_context(pass_geo)
    kw = pass_station_kwargs(pass_geo)
    sc = SpacecraftHistory(pass_geo.t_s, np.asarray(state_history_mci, dtype=float)[:, :6])
    rng = rng or np.random.default_rng()
    obs = np.array(obs, dtype=float, copy=True)
    for i in range(obs.shape[0]):
        sid = int(obs[i, 2]) - 1
        value = two_way_range_m(solve_two_way(float(obs[i, 0]), station_states[sid], sc, eph, model, **kw[sid]))
        obs[i, 1] = value + (stations[sid].sigma_range_m * rng.standard_normal() if noise else 0.0)
    return obs, pass_geo


def two_way_range_nominal_and_initial_jacobian_ssb(obs_data, pass_geo, augmented_state_history_mci):
    """Nominal two-way ranges (N,) and the arc-initial-state Jacobian (N, 6) under the pass's model; the
    signature and conventions of ``two_way_range.two_way_range_nominal_and_initial_jacobian``."""
    obs = np.asarray(obs_data, dtype=float)
    model, station_states, eph = _pass_context(pass_geo)
    kw = pass_station_kwargs(pass_geo)
    sc = SpacecraftHistory(pass_geo.t_s, np.asarray(augmented_state_history_mci, dtype=float)[:, :42])
    h_nom = np.zeros(obs.shape[0])
    h_x0 = np.zeros((obs.shape[0], 6))
    for i in range(obs.shape[0]):
        sid = int(obs[i, 2]) - 1
        sol = solve_two_way(float(obs[i, 0]), station_states[sid], sc, eph, model, **kw[sid])
        h_nom[i] = two_way_range_m(sol)
        h_x0[i] = two_way_range_partial(sol, sc)
    return h_nom, h_x0


def compute_two_way_range_residuals_ssb(state_history_mci, obs_data, pass_geo):
    """Observed-minus-computed two-way range [m] under the pass's model; the conventions of
    ``two_way_range.compute_two_way_range_residuals`` (including its 1e-7 m closure round-off suppression)."""
    obs = np.asarray(obs_data, dtype=float)
    model, station_states, eph = _pass_context(pass_geo)
    kw = pass_station_kwargs(pass_geo)
    sc = SpacecraftHistory(pass_geo.t_s, np.asarray(state_history_mci, dtype=float)[:, :6])
    h = np.array([two_way_range_m(solve_two_way(float(obs[i, 0]), station_states[int(obs[i, 2]) - 1], sc, eph,
                                                model, **kw[int(obs[i, 2]) - 1])) for i in range(obs.shape[0])])
    residuals = obs[:, 1] - h
    residuals[np.abs(residuals) < 1e-7] = 0.0
    return residuals, h
