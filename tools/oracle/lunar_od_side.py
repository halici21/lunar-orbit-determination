"""lunar_od side of the WP5.1 Orekit oracle comparison (no Orekit import).

Dynamics
    ``propagate_aligned`` integrates lunar_od's OWN force function with either
    TT or TDB seconds as the independent variable (alignment rule A_TIME). The
    force function is the 6-state RHS the production ``propagate_state``
    integrates (``dynamics._build_state_rhs``; harmonics in the exact-epoch
    DIRECT MOON_PA_DE440 mode; SRP through its production wrapper), fed with
    Earth/Sun from SPICE at the true TDB epoch. The adapter adds no force
    physics. ``kernel='reference'`` swaps in the reference kernel
    ``dynamics.f3body_moon`` (amendment 2026-11-25: the predeclared kernel, kept
    as information because its Python harmonics cost ~0.13 s per call at
    100x100). D3 always restarts at every shadow contact.

    ``propagate_production`` runs the production ``dynamics.propagate_state``
    for the O_L link gate.

Measurements
    ``measurements`` evaluates lunar_od's light-time solvers on the pinned
    spacecraft trajectory and the pinned (Orekit) station track (A_STATION).

Predeclaration: ``studies/phase17/configs/wp5_1_orekit_oracle_predeclaration.json``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenario import ephemeris_of, gravity_tab, harmonic_degree, load_kernels, output_grid


@dataclass(frozen=True)
class Setting:
    """One rung of an integrator ladder."""

    rtol: float = 1e-13
    atol: float = 1e-9
    max_step: float = float("inf")

    def label(self) -> str:
        cap = "nocap" if not np.isfinite(self.max_step) else f"h{self.max_step:g}"
        return f"rtol{self.rtol:g}_atol{self.atol:g}_{cap}"

    def as_dict(self) -> dict:
        return {"rtol": self.rtol, "atol": self.atol,
                "max_step_s": None if not np.isfinite(self.max_step) else self.max_step}

    @classmethod
    def from_dict(cls, d: dict) -> "Setting":
        ms = d.get("max_step_s")
        return cls(float(d["rtol"]), float(d["atol"]), float("inf") if ms is None else float(ms))


LADDER = (Setting(),) + tuple(Setting(max_step=h) for h in (60.0, 20.0, 10.0, 5.0, 2.0, 1.0))


def _force_setup(case: str, sc: dict, kernel: str = "production"):
    """Returns (ephem(t), force(t, x), srp, harm); ``t`` = TDB seconds since the scenario epoch.

    ``kernel='production'`` (the gate): the 6-state RHS ``dynamics.propagate_state`` integrates, built by
    ``dynamics._build_state_rhs`` (numba fast path where available), harmonics in the exact-epoch DIRECT
    MOON_PA_DE440 mode, SRP through its production wrapper. ``kernel='reference'`` (information): the
    reference kernel ``dynamics.f3body_moon`` with ``lunar_frames.moon_pa_de440_rotation_at_et``.
    """
    import spiceypy as spice
    from lunar_od.srp import SRPOptions

    load_kernels(ephemeris_of(case))
    et0 = float(sc["et0"])
    gm = sc["gm_m3_s2"]
    harm = None
    n = harmonic_degree(case)
    if n is not None:
        from lunar_od.gravity_model_loader import load_lunar_gravity_model

        harm = load_lunar_gravity_model(gravity_tab(), nmax=n, mmax=n)
    srp = SRPOptions(k_srp_m2_per_kg=float(sc["k_srp_m2_per_kg"]), shadow_model="CONICAL_PENUMBRA") \
        if case == "D3" else None
    cache: dict = {}

    def ephem(t):
        t = float(t)
        if cache.get("t") != t:
            et = et0 + t
            cache["t"] = t
            cache["v"] = (np.asarray(spice.spkpos("EARTH", et, "J2000", "NONE", "MOON")[0]) * 1e3,
                          np.asarray(spice.spkpos("SUN", et, "J2000", "NONE", "MOON")[0]) * 1e3)
        return cache["v"]

    if kernel == "production":
        from lunar_od.dynamics import _build_state_rhs, lunar_j2_orientation

        j2_rot, j2_r = lunar_j2_orientation(0.0, epoch_et0=et0)
        rhs, _ = _build_state_rhs(
            np.array([0.0, float(sc["arc_s"])]), gm["MOON"], gm["EARTH"], gm["SUN"],
            lambda t: ephem(t)[0], lambda t: ephem(t)[1], 0.0, 0.0, "indirect",
            harm, None, et0 if harm is not None else None, srp, j2_rot, j2_r)
        force = rhs
    elif kernel == "reference":
        from lunar_od.dynamics import f3body_moon
        from lunar_od.lunar_frames import moon_pa_de440_rotation_at_et

        def force(t, x):
            e, s = ephem(t)
            kw = {}
            if harm is not None:
                kw = dict(harmonic_model=harm, c_inertial_to_bf_harmonic=moon_pa_de440_rotation_at_et(et0 + float(t)))
            return f3body_moon(x, gm["MOON"], gm["EARTH"], gm["SUN"], e, s, srp=srp, **kw)
    else:
        raise ValueError(kernel)
    return et0, ephem, force, srp, harm


def propagate_aligned(case: str, sc: dict, *, time_arg: str = "TT", setting: Setting = Setting(),
                      t_out: np.ndarray | None = None,
                      time_map: tuple[np.ndarray, np.ndarray] | None = None,
                      kernel: str = "production") -> tuple[np.ndarray, np.ndarray, int]:
    """Integrate lunar_od's force model; returns (t_out [TDB s], states (N, 6), number of shadow events).

    ``time_arg``: ``"TDB"`` (lunar_od's convention), ``"TT"`` (TT seconds, SPICE's TDB-TT model) or
    ``"TT_OREKIT"`` (TT seconds with ``time_map`` = (t_tdb_s, s_tt_s) samples of Orekit's own TDB-TT
    model, interpolated by cubic splines both ways; used because precondition O_P(d) failed).
    """
    import spiceypy as spice
    from scipy.integrate import solve_ivp

    if time_arg not in ("TT", "TDB", "TT_OREKIT"):
        raise ValueError(time_arg)
    et0, ephem, force, srp, _ = _force_setup(case, sc, kernel)
    tt0 = spice.unitim(et0, "TDB", "TT")
    if time_arg == "TT_OREKIT":
        from scipy.interpolate import CubicSpline

        tk, sk = (np.asarray(a, dtype=float) for a in time_map)
        t_of_s_spl, s_of_t_spl = CubicSpline(sk, tk), CubicSpline(tk, sk)

    def t_of_s(s):
        """TDB seconds since the epoch at independent-variable value s."""
        if time_arg == "TDB":
            return s
        if time_arg == "TT":
            return spice.unitim(tt0 + s, "TT", "TDB") - et0
        return float(t_of_s_spl(s))

    def s_of_t(t):
        if time_arg == "TDB":
            return t
        if time_arg == "TT":
            return spice.unitim(et0 + t, "TDB", "TT") - tt0
        return float(s_of_t_spl(t))

    def rhs(s, x):
        return force(t_of_s(s), x)

    events = []
    if srp is not None:
        from lunar_od.srp import apparent_radii_and_separation

        def contact(kind):
            def g(s, x):
                _, sun = ephem(t_of_s(s))
                a_s, a_o, gam, _, _ = apparent_radii_and_separation(x[:3], sun, np.zeros(3),
                                                                    srp.occulting_body_radius_m)
                return gam - (a_s + a_o) if kind == "outer" else gam - (a_o - a_s)
            g.terminal = True
            return g
        events = [contact("outer"), contact("inner")]

    t_out = output_grid() if t_out is None else np.asarray(t_out, dtype=float)
    s_out = np.array([s_of_t(t) for t in t_out])
    x = np.asarray(sc["state0_mci_j2000_m_mps"], dtype=float)
    if t_out[0] != 0.0:
        raise ValueError("t_out must start at the epoch")
    kw = dict(method="DOP853", rtol=setting.rtol, atol=setting.atol, max_step=setting.max_step)
    s, res, n_events = 0.0, [x.copy()], 0
    for s_target in s_out[1:]:
        while True:
            sol = solve_ivp(rhs, (s, s_target), x, events=events or None, **kw)
            if sol.status < 0:
                raise RuntimeError(f"{case}: {sol.message}")
            x, s = sol.y[:, -1], sol.t[-1]
            if sol.status == 1:  # stopped at a shadow contact: restart just past it
                n_events += 1
                s_next = min(s + 1e-6, s_target)
                x = solve_ivp(rhs, (s, s_next), x, **kw).y[:, -1]
                s = s_next
                if s >= s_target:
                    break
                continue
            break
        res.append(x.copy())
    return t_out, np.array(res), n_events


def propagate_production(case: str, sc: dict, *, setting: Setting = Setting(),
                         rotation: str = "grid") -> np.ndarray:
    """O_L: the production ``propagate_state`` (DOP853) with direct-SPICE Earth/Sun callbacks.

    ``rotation='grid'`` (the gate): MOON_PA_DE440 sampled every 60 s; ``'direct'`` (information):
    the exact-epoch DIRECT mode (``harmonic_epoch_et0``).
    """
    from lunar_od.dynamics import propagate_state
    from lunar_od.lunar_frames import sample_moon_pa_rotations

    if np.isfinite(setting.max_step):
        raise ValueError("propagate_state takes no step cap; the link uses an uncapped rung")
    et0, ephem, _, srp, harm = _force_setup(case, sc)
    gm = sc["gm_m3_s2"]
    kw = {}
    if harm is not None:
        if rotation == "grid":
            grid = np.arange(-600.0, sc["arc_s"] + 601.0, 60.0)
            kw = dict(harmonic_model=harm,
                      harmonic_rotation=(grid, sample_moon_pa_rotations(et0, grid, frame="MOON_PA_DE440",
                                                                        load_kernels=False)))
        elif rotation == "direct":
            kw = dict(harmonic_model=harm, harmonic_epoch_et0=et0)
        else:
            raise ValueError(rotation)
    if srp is not None:
        kw.update(srp=srp, shadow_events=True)
    return propagate_state(output_grid(), sc["state0_mci_j2000_m_mps"], gm["MOON"], gm["EARTH"], gm["SUN"],
                           lambda t: ephem(t)[0], lambda t: ephem(t)[1],
                           method="DOP853", rtol=setting.rtol, atol=setting.atol, **kw)


def ladder_floor(a: np.ndarray, b: np.ndarray) -> float:
    """max over samples of |dr| between two runs [m]."""
    return float(np.max(np.linalg.norm(np.asarray(a)[:, :3] - np.asarray(b)[:, :3], axis=1)))


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def _track(grid: np.ndarray, states: np.ndarray):
    from lunar_od.radiometrics import interp_state_history

    g = np.asarray(grid, dtype=float)
    s = np.asarray(states, dtype=float)
    return lambda t: interp_state_history(g, s, float(t))


def measurements(sc: dict, traj_grid, traj, st_grid, st_states, t_meas, tc) -> dict:
    """lunar_od values for M1-MOON, M2/M3-MOON, M2/M3-SSB and the L_M3 production counted Doppler."""
    from lunar_od.measurements import solve_one_way_light_time
    from lunar_od.relativistic_light_time import (BarycentricEphemeris, LightTimeModel, SpacecraftHistory,
                                                  counted_doppler, solve_two_way, two_way_range_m)
    from lunar_od.two_way_range import (TwoWayRangeConfig, TwoWayStationStateProvider,
                                        solve_two_way_range_events, two_way_range_from_solution)

    load_kernels("DE440")
    station = _track(st_grid, st_states)
    sc_state = _track(traj_grid, traj)
    prov = TwoWayStationStateProvider(station, "pinned_orekit_station_track", "orekit_jpl_de440")
    cfg = TwoWayRangeConfig()

    def r2_moon(t):
        return two_way_range_from_solution(solve_two_way_range_events(t, prov, traj_grid, traj, cfg), cfg)

    eph = BarycentricEphemeris(float(sc["et0"]))
    hist = SpacecraftHistory(traj_grid, traj)
    model = LightTimeModel(frame="SSB", shapiro_bodies=(), interval_time_scale="TDB")

    def r2_ssb(t):
        return two_way_range_m(solve_two_way(t, station, hist, eph, model))

    out = {k: [] for k in ("one_way_moon", "two_way_moon", "cd_moon", "two_way_ssb", "cd_ssb",
                           "cd_ssb_production")}
    for t in t_meas:
        out["one_way_moon"].append(solve_one_way_light_time(t, station(t)[:3],
                                                            lambda tt: sc_state(tt)[:3]).range_m)
        out["two_way_moon"].append(r2_moon(t))
        out["cd_moon"].append((r2_moon(t + tc / 2) - r2_moon(t - tc / 2)) / tc)
        out["two_way_ssb"].append(r2_ssb(t))
        out["cd_ssb"].append((r2_ssb(t + tc / 2) - r2_ssb(t - tc / 2)) / tc)
        out["cd_ssb_production"].append(counted_doppler(t, tc, station, hist, eph, model)[0])
    return {k: np.asarray(v) for k, v in out.items()}


def own_station_track(sc: dict, st_grid) -> np.ndarray:
    """I_STATION: lunar_od's own Canberra DSN (SPICE ITRF93 PCK chain) on the station grid."""
    from lunar_od.config import range_rate_stations
    from lunar_od.ephemeris import sample_moon_centered_ephemeris
    from lunar_od.two_way_range import make_exact_sxform_station_state_provider

    load_kernels("DE440")
    et0 = float(sc["et0"])
    eph_t = np.arange(st_grid[0] - 600.0, st_grid[-1] + 601.0, 60.0)
    eph = sample_moon_centered_ephemeris(et0, eph_t)
    st = next(s for s in range_rate_stations() if s.name == sc["station"]["name"])
    prov = make_exact_sxform_station_state_provider(st, et0, eph_t, eph.earth_pos_m, eph.earth_vel_mps)
    return np.array([prov.state(t) for t in st_grid])
