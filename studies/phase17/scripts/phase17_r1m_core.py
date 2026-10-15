"""PHASE 17-R1M - shared analysis fixtures and information-geometry utilities.

ANALYSIS SPACE ONLY.  Nothing in this module changes production behaviour: it
imports the frozen estimator/measurement/dynamics code and reads design
matrices out of it.  No production file is modified by R1M (s12/s13).

The one substantive algebraic identity this module rests on, used repeatedly
below and worth stating once:

    with  A_x = W^(1/2) H_x0   (N x 6)   and   b_K = W^(1/2) h_K   (N,)

    I_KK    = b_K^T b_K
    I_K|x   = I_KK - I_Kx I_xx^-1 I_xK
            = ||b_K||^2 - ||P_{A_x} b_K||^2
            = ||b_K - P_{A_x} b_K||^2
            = ||b_K,perp||^2

So the Schur conditional K information IS the squared norm of the part of the
K design column that the six orbital-state columns cannot reproduce, and

    f_perp = ||b_K,perp|| / ||b_K|| = sqrt( I_K|x / I_KK )

is therefore not a separate quantity from the conditional information -- it is
the same fact expressed as a dimensionless fraction.  R1M reports both because
they answer different questions: I_K|x asks "how much independent K
information is there", f_perp asks "what fraction of K's own signature is
independent".  A configuration can raise the first while leaving the second
almost unchanged, which is exactly the distinction s33 asks for.

EARTH EPOCH CONTRACT (Phase 17-FXR)
-----------------------------------
Up to Phase 17-FXR, `build_range_arc` took the Sun, the Earth rotation and the
spacecraft state from the Phase-7 manifest epoch (`phase7_et0`) but fed the
MEASUREMENT model the campaign module's Earth table
(`od_gravity_covariance_campaign.get_earth_pos`), whose t = 0 is the canonical
campaign ET0, 504000 s (5.83 d) EARLIER, and which `np.interp` holds constant
after 4 orbits.  The range line of sight was therefore drawn to where the
Earth was 5.83 days before the measurement.  The production range model cannot
detect this: it receives Earth as an untagged callable of scenario seconds.

The canonical fixture is now `EARTH_FIXTURE_CANONICAL`: exact SPICE Earth
states at the SAME ET origin the production model uses for the Earth rotation
(`pass_geo.et0_s`), over an explicit coverage interval, failing loudly
(`EarthEphemerisCoverageError`) instead of clamping.  The defective fixture is
kept, only by explicit request, as `EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED` so
the published R1M..R1O-R numbers stay reproducible.  See
studies/phase17/reports/phase17_fxr_earth_epoch_fixture_erratum.md.

The dynamics are untouched: `build_range_arc` runs with mu_earth = mu_sun = 0,
so the constant `earth_at` it hands the propagator is never used as a force.
"""
from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- campaign data (read-only, outside the repo; the R1/R1G arc) ----------
# HARDEN-D10: resolved by lunar_od.project_paths (LUNAR_OD_CAMPAIGN_ROOT overrides).
from lunar_od.project_paths import campaign_root as _campaign_root  # noqa: E402

CAMPAIGN_ROOT = _campaign_root()

# HARDEN-D12: the external campaign module prepends a sibling checkout to
# sys.path; import it once through the guarded loader so it cannot rebind lunar_od.
try:
    from lunar_od.project_paths import import_campaign_module as _import_campaign  # noqa: E402
    _import_campaign()
except ModuleNotFoundError:      # campaign scripts not on PYTHONPATH: callers fail later
    pass

K_TRUTH = 0.01          # SYNTHETIC_CAMPAIGN_TRUTH
K_WRONG = 0.015
K_INITIAL = 0.012
CADENCE_S = 60.0
RTOL, ATOL = 1e-12, 1e-13

#: The R1/R1G physical scaling, frozen.  Position 1e6 m, velocity 1e3 m/s,
#: K 1e-2 m^2/kg -- the same diag(scale) the estimator itself uses.
SCALE_POS, SCALE_VEL, SCALE_K = 1.0e6, 1.0e3, 1.0e-2


def campaign_epoch() -> tuple[float, float]:
    m = json.loads(
        (CAMPAIGN_ROOT / "08_manifests"
         / "phase7_measurement_geometry_manifest.json").read_text())
    return float(m["phase7_et0"]), float(m["orbit_period_s"])


def campaign_initial_state() -> np.ndarray:
    t_ext = np.load(CAMPAIGN_ROOT / "03_data" / "phase8_T_ext.npy")
    x_ext = np.load(CAMPAIGN_ROOT / "03_data" / "phase8_X_ext.npy")
    return x_ext[int(np.argmin(np.abs(t_ext)))].copy()


# ======================================================================
# Earth ephemeris for the MEASUREMENT geometry (Phase 17-FXR)
# ======================================================================
#: The only Earth fixture for science: exact SPICE Earth at the measurement ET.
EARTH_FIXTURE_CANONICAL = "CANONICAL_EPOCH_CONSISTENT_SPICE"
#: SUPERSEDED, DEFECTIVE.  The R1M..R1O-R fixture: campaign Earth table from an
#: epoch 504000 s early, held constant after 28260 s (3.9986 orbits).  Reproduction only.
EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED = "LEGACY_OFFSET_CLAMPED_TABLE_SUPERSEDED_BY_PHASE17_FXR"
#: FXR s17 attribution diagnostics: each carries exactly ONE of the two defects.
EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED = "DIAGNOSTIC_OFFSET_EPOCH_UNCLAMPED"
EARTH_FIXTURE_DIAG_CORRECT_EPOCH_CLAMPED = "DIAGNOSTIC_CORRECT_EPOCH_CLAMPED"
_NON_CANONICAL_EARTH_FIXTURES = (
    EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED,
    EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED,
    EARTH_FIXTURE_DIAG_CORRECT_EPOCH_CLAMPED,
)


class EarthEphemerisCoverageError(ValueError):
    """An Earth state was requested outside the fixture's declared coverage."""


class SupersededEarthFixtureWarning(UserWarning):
    """A superseded or diagnostic Earth fixture was explicitly selected."""


@functools.lru_cache(maxsize=1)
def _campaign_et0() -> float:
    return campaign_epoch()[0]


def _spice_earth_state_mci(et: float) -> np.ndarray:
    """Moon-centred J2000 Earth state [m, m/s], geometric (no aberration)."""
    import spiceypy as spice
    return np.asarray(spice.spkezr("EARTH", float(et), "J2000", "NONE", "MOON")[0]) * 1e3


def _legacy_campaign_module():
    import od_gravity_covariance_campaign as C  # noqa: N811
    return C


@dataclass(frozen=True)
class EarthEphemeris:
    """Epoch-tagged Earth ephemeris handed to the production measurement model.

    ``pos(t)`` / ``vel(t)`` take scenario seconds ``t`` (the same ``t`` the
    production range model passes, measured from ``et_origin``) and return
    (n, 3) Moon-centred J2000 Earth position [m] / velocity [m/s].

    CANONICAL: SPICE at ``et_origin + t``; any ``t`` outside
    [t_min_s, t_max_s] raises EarthEphemerisCoverageError -- never a clamp.
    Non-canonical fixtures reproduce a defect on purpose and say which.
    """

    fixture: str
    et_origin: float            # SPICE ET (TDB s) the Earth's own t = 0 refers to
    t_min_s: float
    t_max_s: float
    clamp_after_s: float = float("inf")   # t beyond which Earth is held constant
    frame: str = "J2000"
    center: str = "MOON"
    units: str = "m, m/s"

    def _check(self, t: np.ndarray) -> None:
        if self.fixture != EARTH_FIXTURE_CANONICAL:
            return
        bad = (t < self.t_min_s) | (t > self.t_max_s)
        if np.any(bad):
            raise EarthEphemerisCoverageError(
                "Earth state requested at t=%r s, outside coverage [%r, %r] s "
                "(ET origin %r)" % (t[bad][:3].tolist(), self.t_min_s, self.t_max_s,
                                    self.et_origin))

    def _column(self, t, cols: slice) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        if self.fixture == EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED:
            C = _legacy_campaign_module()  # noqa: N806
            fn = C.get_earth_pos if cols.start == 0 else C.get_earth_vel
            return fn(t)
        self._check(tt)
        tt = np.minimum(tt, self.clamp_after_s)
        return np.array([_spice_earth_state_mci(self.et_origin + x)[cols] for x in tt])

    def pos(self, t) -> np.ndarray:
        return self._column(t, slice(0, 3))

    def vel(self, t) -> np.ndarray:
        return self._column(t, slice(3, 6))

    @property
    def legacy_reproduction_only(self) -> bool:
        """Read by `lunar_od.ephemeris.require_ephemeris_contract` (Phase 17-HARDEN):
        every non-canonical fixture is accepted by production only with a warning."""
        return self.fixture != EARTH_FIXTURE_CANONICAL

    def describe(self) -> dict:
        return dict(fixture=self.fixture, et_origin=self.et_origin, t_min_s=self.t_min_s,
                    t_max_s=self.t_max_s, clamp_after_s=self.clamp_after_s,
                    frame=self.frame, center=self.center, units=self.units)


def measurement_earth_ephemeris(earth_fixture: str, et_measurement_origin: float,
                                t_max_s: float, *, t_min_s: float = 0.0) -> EarthEphemeris:
    """The Earth ephemeris a range arc hands the measurement model.

    ``et_measurement_origin`` is the ET the production model adds scenario
    seconds to for the Earth ROTATION (``pass_geo.et0_s``).  The canonical
    fixture uses that same origin -- the whole point of the FXR repair.
    """
    if earth_fixture == EARTH_FIXTURE_CANONICAL:
        return EarthEphemeris(earth_fixture, float(et_measurement_origin),
                              float(t_min_s), float(t_max_s))
    if earth_fixture not in _NON_CANONICAL_EARTH_FIXTURES:
        raise ValueError("unknown earth_fixture %r" % (earth_fixture,))
    warnings.warn("Earth fixture %s selected: SUPERSEDED/DIAGNOSTIC geometry, not for "
                  "science (Phase 17-FXR)" % earth_fixture, SupersededEarthFixtureWarning,
                  stacklevel=3)
    C = _legacy_campaign_module()  # noqa: N806
    table_end = float(C.T_OBS[-1])
    if earth_fixture == EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED:
        # The table's t = 0 is the campaign ET0 whatever arc is being built.
        return EarthEphemeris(earth_fixture, float(C.ET0), float(C.T_OBS[0]), table_end,
                              clamp_after_s=table_end)
    if float(et_measurement_origin) != _campaign_et0():
        raise ValueError("attribution diagnostics are defined for arcs starting at the "
                         "campaign epoch only")
    if earth_fixture == EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED:
        return EarthEphemeris(earth_fixture, float(C.ET0), float(t_min_s), float(t_max_s))
    return EarthEphemeris(earth_fixture, float(et_measurement_origin), float(t_min_s),
                          float(t_max_s), clamp_after_s=table_end)


def _earth_state_abs(t_abs_s, earth_fixture: str, cols: slice) -> np.ndarray:
    tt = np.atleast_1d(np.asarray(t_abs_s, dtype=float))
    if earth_fixture == EARTH_FIXTURE_CANONICAL:
        et0 = _campaign_et0()
        return np.array([_spice_earth_state_mci(et0 + float(x))[cols] for x in tt])
    if earth_fixture == EARTH_FIXTURE_LEGACY_OFFSET_CLAMPED:
        C = _legacy_campaign_module()  # noqa: N806
        fn = C.get_earth_pos if cols.start == 0 else C.get_earth_vel
        return np.asarray(fn(t_abs_s), dtype=float)
    if earth_fixture in (EARTH_FIXTURE_DIAG_OFFSET_UNCLAMPED,
                         EARTH_FIXTURE_DIAG_CORRECT_EPOCH_CLAMPED):
        eph = measurement_earth_ephemeris(earth_fixture, _campaign_et0(), float(np.max(tt)))
        return eph.pos(tt) if cols.start == 0 else eph.vel(tt)
    raise ValueError("unknown earth_fixture %r" % (earth_fixture,))


def earth_position_mci(t_abs_s, earth_fixture: str = EARTH_FIXTURE_CANONICAL) -> np.ndarray:
    """Earth position (n, 3) [m] at campaign-epoch seconds, for analysis helpers
    that place stations outside `build_range_arc` (DDOR, LOS controls).

    Canonical: SPICE at campaign ET0 + t (SPICE raises outside kernel coverage).
    Legacy: the campaign table, as R1O/R1O-D/R1O-R read it.  The two FXR
    diagnostic fixtures are accepted for attribution only.
    """
    return _earth_state_abs(t_abs_s, earth_fixture, slice(0, 3))


def earth_velocity_mci(t_abs_s, earth_fixture: str = EARTH_FIXTURE_CANONICAL) -> np.ndarray:
    """Earth velocity (n, 3) [m/s]; same contract as `earth_position_mci`."""
    return _earth_state_abs(t_abs_s, earth_fixture, slice(3, 6))


# ======================================================================
# Station visibility contract (Phase 17-HARDEN)
# ======================================================================
#: THE Phase-17 radio visibility contract: a station sees the spacecraft iff the
#: production `lunar_od.visibility` test passes -- elevation >= mask AND the
#: station->spacecraft line of sight is not blocked by the Moon.  Every range,
#: ΔDOR and surrogate row generator uses it by default.
DDOR_ROW_RULE_VISIBLE = "PRODUCTION_VISIBILITY_ELEVATION_AND_OCCULTATION"
#: SUPERSEDED (Phase 17-D2).  The R1O / R1O-D / FXR rule: elevation only, lunar
#: occultation NOT tested -- 93 of 263 canonical FXR ΔDOR rows were behind the Moon.
DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY = "ELEVATION_ONLY_HISTORICAL_R1OD"
_ROW_RULES = (DDOR_ROW_RULE_VISIBLE, DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY)


class SupersededVisibilityWarning(UserWarning):
    """The superseded elevation-only visibility rule was explicitly selected."""


def require_row_rule(row_rule: str) -> str:
    """Validate a row rule; the legacy rule is accepted only with a warning."""
    if row_rule not in _ROW_RULES:
        raise ValueError("unknown row rule %r" % (row_rule,))
    if row_rule == DDOR_ROW_RULE_LEGACY_ELEVATION_ONLY:
        warnings.warn("row rule %s: lunar occultation NOT tested -- historical reproduction "
                      "only (Phase 17-D2/HARDEN)" % row_rule, SupersededVisibilityWarning,
                      stacklevel=3)
    return row_rule


def station_visibility_mask(t_local: np.ndarray, positions_mci: np.ndarray, stations,
                            earth_pos_hist_m: np.ndarray, xf: np.ndarray,
                            min_elevation_deg: float = 10.0) -> np.ndarray:
    """(N, M) production visibility (elevation + lunar occultation) of the
    spacecraft from each station, on the grid ``t_local`` whose Earth positions
    ``earth_pos_hist_m`` (N, 3, Moon-centred J2000, m) and J2000->ITRF93
    transforms ``xf`` were evaluated at the SAME epochs by the caller."""
    from lunar_od.constants import R_MOON_M
    from lunar_od.visibility import VisibilityConfig, analyze_visibility_gap_with_transforms

    t_local = np.asarray(t_local, float)
    earth = np.asarray(earth_pos_hist_m, float).reshape(t_local.size, 3)
    cfg = VisibilityConfig(r_moon_mean_m=R_MOON_M, earth_rotation_rad_s=7.292115e-5,
                           epoch_utc="UNUSED_BY_THE_TRANSFORM_PATH",
                           min_elevation_deg=min_elevation_deg)

    def earth_on_grid(t):
        if np.asarray(t, float).shape != t_local.shape:
            raise ValueError("visibility Earth history requested on a different grid")
        return earth

    _, _, vis, _ = analyze_visibility_gap_with_transforms(
        t_local, np.asarray(positions_mci, float)[:, :3], tuple(stations), earth_on_grid, xf,
        0.0, cfg)
    return np.asarray(vis, dtype=bool).reshape(t_local.size, len(stations))


# ======================================================================
# Historical-only scripts (Phase 17-HARDEN quarantine)
# ======================================================================
LEGACY_REPRODUCTION_ENV = "PHASE17_LEGACY_REPRODUCTION"


class LegacyReproductionRequired(RuntimeError):
    """A historical-only script was run without the explicit reproduction opt-in."""


def require_legacy_reproduction_opt_in(script: str, reason: str) -> None:
    """Refuse to run a script whose physics is known-superseded unless the
    caller sets PHASE17_LEGACY_REPRODUCTION=1 (historical reproduction only)."""
    import os

    if os.environ.get(LEGACY_REPRODUCTION_ENV) != "1":
        raise LegacyReproductionRequired(
            "%s is LEGACY_REPRODUCTION_ONLY (%s). Set %s=1 to reproduce the historical "
            "result; it is not valid current science." % (script, reason, LEGACY_REPRODUCTION_ENV))
    warnings.warn("%s: running as LEGACY_REPRODUCTION_ONLY (%s)" % (script, reason),
                  SupersededVisibilityWarning, stacklevel=2)


@dataclass(frozen=True)
class ArcFixture:
    """One contiguous observation window on the frozen R1/R1G arc."""

    label: str
    t_grid: np.ndarray          # propagation grid, seconds from campaign epoch
    x_true: np.ndarray          # (N,6) truth trajectory on t_grid
    nom48: np.ndarray           # (N,48) state + STM + dx/dK
    obs: np.ndarray             # observation rows
    pass_geo: object
    station_name: str
    h_x0: np.ndarray            # (M,6) arc-initial-state Jacobian
    h_k: np.ndarray             # (M,)  K column
    w: np.ndarray               # (M,)  measurement weight diagonal (1/sigma^2)
    earth: EarthEphemeris | None = None   # what the measurement model was given (FXR)

    @property
    def earth_fixture(self) -> str:
        return self.earth.fixture if self.earth is not None else "UNRECORDED"

    @property
    def n_obs(self) -> int:
        return int(self.obs.shape[0])

    @property
    def elapsed_s(self) -> float:
        return float(self.t_grid[-1] - self.t_grid[0])


def build_range_arc(
    t_start_s: float,
    duration_s: float,
    *,
    label: str,
    station_filter=None,
    k_srp: float = K_TRUTH,
    cadence_s: float = CADENCE_S,
    earth_fixture: str = EARTH_FIXTURE_CANONICAL,
) -> ArcFixture:
    """Propagate one window of the frozen arc and read its design matrices.

    ``t_start_s`` is measured from the campaign epoch.  The propagation always
    starts at the CAMPAIGN epoch and the window is sliced out afterwards, so
    every arc in a multi-arc study shares one physically continuous truth
    trajectory -- arcs differ by which observations are used, never by a
    different truth.

    ``earth_fixture`` selects the Earth ephemeris the MEASUREMENT model sees
    (station placement, visibility, light time).  Only the default is valid
    science; see the module docstring (Phase 17-FXR).
    """
    import spiceypy as spice

    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.estimators import (
        _two_way_range_k_srp_column,
        _two_way_range_weight_diagonal,
    )
    from lunar_od.srp import SRPOptions
    from lunar_od.two_way_range import (
        TwoWayRangeConfig,
        two_way_range_nominal_and_initial_jacobian,
    )
    from lunar_od.visibility import (
        VisibilityConfig,
        analyze_visibility_gap_with_transforms,
        sample_j2000_to_itrf93_transforms,
    )

    import od_gravity_covariance_campaign as C  # noqa: N811

    et0, _ = campaign_epoch()

    def sun_at(t_s):
        return spice.spkezr("SUN", et0 + float(t_s), "J2000", "NONE",
                            "MOON")[0][:3] * 1000.0

    def earth_at(_t_s):
        return np.array([384_400e3, 0.0, 0.0])

    t_grid = np.arange(t_start_s, t_start_s + duration_s + 0.5 * cadence_s,
                       cadence_s)
    x0 = campaign_initial_state()
    if t_start_s > 0.0:
        # Propagate the ONE truth trajectory from the campaign epoch up to the
        # window start, then continue on the window grid: arcs are windows on a
        # single physical trajectory, not independent re-initialisations.
        lead = np.concatenate([np.arange(0.0, t_start_s, cadence_s),
                               [t_start_s]])
        lead_hist = propagate_state_with_k_sensitivity(
            lead, x0, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=k_srp), rtol=RTOL, atol=ATOL,
            j2_moon=J2_MOON_UNNORMALIZED)
        window_x0 = lead_hist[-1, :6]
    else:
        window_x0 = x0

    nom48 = propagate_state_with_k_sensitivity(
        t_grid - t_grid[0], window_x0, C.MU, 0.0, 0.0,
        lambda t: earth_at(t + t_grid[0]), lambda t: sun_at(t + t_grid[0]),
        srp=SRPOptions(k_srp_m2_per_kg=k_srp), rtol=RTOL, atol=ATOL,
        j2_moon=J2_MOON_UNNORMALIZED)
    x_true = nom48[:, :6]

    t_local = t_grid - t_grid[0]
    earth = measurement_earth_ephemeris(earth_fixture, et0 + t_grid[0], float(t_local[-1]))
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    vis_cfg = VisibilityConfig(
        r_moon_mean_m=1_737_400.0, earth_rotation_rad_s=7.292115e-5,
        epoch_utc=spice.et2utc(et0 + t_grid[0], "ISOC", 3),
        min_elevation_deg=10.0)
    _, _, vis, _ = analyze_visibility_gap_with_transforms(
        t_local, x_true, C.STATIONS, earth.pos, xf, 0.0, vis_cfg)
    vis = np.asarray(vis, dtype=bool).reshape(t_local.size, len(C.STATIONS))

    if station_filter is None:
        sj = int(np.argmax(vis.sum(axis=0)))
        stations, vis_used = (C.STATIONS[sj],), vis[:, [sj]]
        station_name = C.STATIONS[sj].name
    else:
        idx = [i for i, s in enumerate(C.STATIONS) if s.name in station_filter]
        stations = tuple(C.STATIONS[i] for i in idx)
        vis_used = vis[:, idx]
        station_name = ";".join(s.name for s in stations)

    tight = TwoWayRangeConfig(tolerance_s=1e-13, equation_tolerance_s=1e-14,
                              max_iter=200)
    from lunar_od.measurement_contracts import generate_two_way_range_measurements_checked  # HARDEN
    obs, pass_geo = generate_two_way_range_measurements_checked(
        t_local, x_true, stations, vis_used, earth.pos, earth.vel,
        et0 + t_grid[0], noise=False, rng=None, config=tight)

    _, h_x0 = two_way_range_nominal_and_initial_jacobian(
        obs, pass_geo, nom48[:, :42])
    h_k = _two_way_range_k_srp_column(obs, nom48, h_x0)
    w = _two_way_range_weight_diagonal(obs, pass_geo)

    return ArcFixture(label=label, t_grid=t_grid, x_true=x_true, nom48=nom48,
                      obs=obs, pass_geo=pass_geo, station_name=station_name,
                      h_x0=np.asarray(h_x0, float), h_k=np.asarray(h_k, float),
                      w=np.asarray(w, float), earth=earth)


# ======================================================================
# Information geometry
# ======================================================================
def scale_matrix(n_states: int = 6, n_k: int = 1) -> np.ndarray:
    diag = ([SCALE_POS] * 3 + [SCALE_VEL] * 3) * (n_states // 6) + [SCALE_K] * n_k
    return np.diag(diag)


def whitened(h_x0: np.ndarray, h_k: np.ndarray, w: np.ndarray):
    """Return (A_x, b_K) in W^(1/2)-whitened coordinates, PHYSICAL units."""
    s = np.sqrt(np.asarray(w, float))
    return h_x0 * s[:, None], np.asarray(h_k, float) * s


def orthogonal_decomposition(a_x: np.ndarray, b_k: np.ndarray) -> dict:
    """Split the whitened K column into state-reproducible and independent parts.

    Uses a QR projector rather than forming (A^T A)^-1: the six-state normal
    matrix here is itself badly conditioned, and squaring it to answer a
    question ABOUT conditioning would be self-defeating.
    """
    q, _ = np.linalg.qr(a_x)
    b_par = q @ (q.T @ b_k)
    b_perp = b_k - b_par
    norm_k = float(np.linalg.norm(b_k))
    norm_par = float(np.linalg.norm(b_par))
    norm_perp = float(np.linalg.norm(b_perp))
    return dict(
        k_column_norm=norm_k,
        projected_norm=norm_par,
        orthogonal_norm=norm_perp,
        orthogonal_fraction=(norm_perp / norm_k) if norm_k > 0 else float("nan"),
        i_kk=norm_k ** 2,
        i_k_given_x=norm_perp ** 2,
    )


def information_matrix(h_full: np.ndarray, w: np.ndarray,
                       scale: np.ndarray) -> np.ndarray:
    hs = h_full @ scale
    return hs.T @ (np.asarray(w, float)[:, None] * hs)


def schur_conditional(info: np.ndarray, k_index: int = -1) -> float:
    """I_KK - I_Kx I_xx^-1 I_xK, via a solve rather than an explicit inverse."""
    n = info.shape[0]
    ki = k_index % n
    keep = [i for i in range(n) if i != ki]
    i_xx = info[np.ix_(keep, keep)]
    i_xk = info[np.ix_(keep, [ki])]
    i_kk = float(info[ki, ki])
    return float(i_kk - (i_xk.T @ np.linalg.solve(i_xx, i_xk))[0, 0])


def spectrum(info: np.ndarray) -> dict:
    sv = np.linalg.svd(info, compute_uv=False)
    u, s, vt = np.linalg.svd(info)
    return dict(
        singular_values=sv.tolist(),
        smallest=float(sv[-1]),
        largest=float(sv[0]),
        condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        rank_default=int(np.linalg.matrix_rank(info)),
        weakest_vector=vt[-1].tolist(),
        weakest_k_component=float(abs(vt[-1][-1])),
    )
