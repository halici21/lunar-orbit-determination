"""Ground Track page — selenographic orbit track with Earth station visibility.

Static view: full track color-coded by which station network sees the spacecraft.
Animation:   spacecraft dot moves along the track with a fading tail.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QSizePolicy,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QPlainTextEdit,
    QTabWidget, QScrollArea, QGroupBox, QFormLayout, QDoubleSpinBox, QSlider,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from styles.theme import C, FONT, MPL_DARK, NET_COLORS

UI_PATH  = DESKTOP_APP_DIR / "ui" / "pages" / "ground_track_page.ui"
FIXTURE  = PYTHON_PORT / "fixtures" / "spice_snapshots.json"

_SAMPLE_STEP_S = 240.0    # 4-minute propagation step (smoother ground track)
_EPHEM_STEP_S  = 3600.0   # 1-hour ephemeris grid

_NO_VIS_COLOR  = "#2a3448"   # color for segments with no station coverage
_MOON_BG       = "#111318"   # Moon surface background color
_MOON_GRID     = "#1e2535"   # lat/lon grid lines
_MOON_NEARSIDE = "#1a2240"   # subtle near-side highlight (lon ±90°)
_TAIL_ALPHA    = 0.75        # animation tail opacity
_TAIL_LEN      = 25          # number of trail points

# Moon physical constants (used by Keplerian orbit designer)
R_MOON_M   = 1_737_400.0   # Moon mean radius, m
MU_MOON_M3 = 4.9048695e12  # Moon GM, m³/s²
OMEGA_MOON = 2.6617e-6     # Moon sidereal rotation rate, rad/s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _network(name: str) -> str:
    for word in name.strip().split():
        if word in NET_COLORS:
            return word
    return "OTHER"


def _dominant_color(vis_mask: np.ndarray, step_idx: int, station_names: list[str]) -> str:
    """Return the network color of the first visible station at step_idx, or no-vis color."""
    for si, name in enumerate(station_names):
        if si < vis_mask.shape[0] and vis_mask[si, step_idx]:
            return NET_COLORS.get(_network(name), C.TEXT_MUTED)
    return _NO_VIS_COLOR


def _visible_station_names(vis_mask: np.ndarray, step_idx: int, station_names: list[str]) -> list[str]:
    return [
        name for si, name in enumerate(station_names)
        if si < vis_mask.shape[0] and vis_mask[si, step_idx]
    ]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _TrackWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # data dict
    failed   = pyqtSignal(str)

    def __init__(
        self,
        station_names: list[str],
        duration_days: float,
        elev_deg: float,
        max_gap_min: float,
    ) -> None:
        super().__init__()
        self.station_names = station_names
        self.duration_days = duration_days
        self.elev_deg      = elev_deg
        self.max_gap_s     = max_gap_min * 60.0

    def run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _do_run(self) -> None:
        import spiceypy as spice
        from lunar_od import (
            VisibilityConfig,
            analyze_visibility_gap_with_transforms,
            load_spice_kernels,
            propagate_truth_with_ephemeris,
            range_rate_stations,
            sample_j2000_to_itrf93_transforms,
            sample_moon_centered_ephemeris,
        )
        from lunar_od.thesis_matrix import THESIS_MAX_GAP_S

        self.progress.emit("Loading fixture…")
        fixture   = json.loads(FIXTURE.read_text(encoding="utf-8"))
        initial   = fixture["initial_state"]
        constants = fixture["constants"]
        epoch_utc = fixture["epoch_utc"]

        x0      = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
        mu_moon = float(initial["mu_moon_m3_s2"])
        mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
        mu_sun   = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)
        r_moon   = float(initial["r_moon_mean_m"])

        t_eval  = np.arange(0.0, self.duration_days * 86400.0 + _SAMPLE_STEP_S, _SAMPLE_STEP_S)
        t_ephem = np.arange(0.0, self.duration_days * 86400.0 + _EPHEM_STEP_S,  _EPHEM_STEP_S)

        self.progress.emit("Loading SPICE kernels…")
        load_spice_kernels()
        try:
            et0 = float(spice.str2et(epoch_utc))

            self.progress.emit("Propagating orbit…")
            ephemeris = sample_moon_centered_ephemeris(et0, t_ephem)
            xforms    = sample_j2000_to_itrf93_transforms(et0, t_eval)
            states    = propagate_truth_with_ephemeris(
                t_eval, x0, mu_moon, mu_earth, mu_sun, ephemeris,
                rtol=1e-9, atol=1e-10,
            )

            self.progress.emit("Computing selenographic coordinates…")
            lons, lats = self._selenographic(
                np.asarray(states)[:, :3], t_eval, et0, spice
            )

            self.progress.emit("Computing station visibility…")
            by_name  = {s.name: s for s in range_rate_stations()}
            stations = [by_name[n] for n in self.station_names if n in by_name]
            if not stations:
                self.failed.emit("No valid stations selected.")
                return

            vis_cfg = VisibilityConfig(
                r_moon_mean_m=r_moon,
                earth_rotation_rad_s=7.292115e-5,
                epoch_utc=epoch_utc,
                min_elevation_deg=self.elev_deg,
            )
            _seg_s, _seg_e, raw_mask, _filled = analyze_visibility_gap_with_transforms(
                t_eval, states, stations,
                ephemeris.earth_position, xforms,
                self.max_gap_s, vis_cfg,
            )
        finally:
            spice.kclear()

        raw_mask = np.asarray(raw_mask, dtype=bool)
        # Ensure shape (n_stations, n_steps)
        if raw_mask.ndim == 2 and raw_mask.shape[0] == len(t_eval) and raw_mask.shape[1] == len(stations):
            raw_mask = raw_mask.T

        self.finished.emit({
            "lons":          lons,
            "lats":          lats,
            "t_s":           t_eval,
            "vis_mask":      raw_mask,
            "station_names": [s.name for s in stations],
            "duration_days": self.duration_days,
            "epoch_utc":     epoch_utc,
        })

    @staticmethod
    def _selenographic(
        positions_mci: np.ndarray,
        t_eval_s: np.ndarray,
        et0: float,
        spice: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform MCI (J2000) positions to selenographic lat/lon in degrees."""
        n = len(positions_mci)
        lons = np.empty(n)
        lats = np.empty(n)
        for i, (pos, t) in enumerate(zip(positions_mci, t_eval_s)):
            rot = spice.pxform("J2000", "IAU_MOON", et0 + t)
            pf  = rot @ pos
            lats[i] = np.degrees(np.arctan2(pf[2], np.sqrt(pf[0] ** 2 + pf[1] ** 2)))
            lons[i] = np.degrees(np.arctan2(pf[1], pf[0]))
        return lons, lats


# ---------------------------------------------------------------------------
# Moon map background
# ---------------------------------------------------------------------------

def _draw_moon_bg(ax) -> None:
    """Draw Moon map background: surface, near-side highlight, grid."""
    ax.set_facecolor(_MOON_BG)

    # Subtle near-side band (sub-Earth point at ~0°, 0° for tidally locked Moon)
    from matplotlib.patches import Rectangle
    near_patch = Rectangle(
        (-90, -90), 180, 180,
        facecolor=_MOON_NEARSIDE, edgecolor="none", zorder=1, alpha=0.35,
    )
    ax.add_patch(near_patch)

    # Latitude / longitude grid
    for lat in range(-90, 91, 30):
        lw = 0.7 if lat == 0 else 0.3
        ax.axhline(lat, color=_MOON_GRID, linewidth=lw, zorder=2)
    for lon in range(-180, 181, 30):
        lw = 0.7 if lon == 0 else 0.3
        ax.axvline(lon, color=_MOON_GRID, linewidth=lw, zorder=2)

    # Sub-Earth label
    ax.text(
        2, 2, "● Sub-Earth",
        color="#344060", fontsize=FONT.SIZE_XS,
        fontfamily=FONT.MONO, zorder=3, va="bottom",
    )


# ---------------------------------------------------------------------------
# Static track drawing
# ---------------------------------------------------------------------------

def _pass_boundaries(lats: np.ndarray, lons: np.ndarray | None = None) -> np.ndarray:
    """Return split indices for drawing individual pole-to-pole passes.

    Splits at:
    - Latitude direction reversals (pole crossings detected via dlat sign change)
    - Large longitude jumps > 90° (catches cases where sparse 10-min sampling
      straddles a pole crossing, causing one sample to be near-side and the
      next to be far-side of the Moon without a detected lat reversal)
    """
    n = len(lats)
    if n < 3:
        return np.array([0, n])

    dlat = np.diff(lats)
    # indices in lats[] where lat reverses direction (pole-crossing index)
    lat_splits = np.where(np.diff(np.sign(dlat)) != 0)[0] + 1

    splits = set(lat_splits.tolist())

    if lons is not None:
        dlon = np.abs(np.diff(lons))
        # also split wherever consecutive points jump > 90° in longitude
        for idx in np.where(dlon > 90)[0]:
            splits.add(int(idx) + 1)

    all_splits = np.array(sorted(splits), dtype=int)
    return np.concatenate([[0], all_splits, [n]])


def _ground_track_runs(lon, lat, state,
                       lon_wrap_deg: float = 180.0,
                       polar_lat_deg: float = 89.0,
                       polar_jump_deg: float = 90.0):
    """Split a selenographic track into per-state polylines for clean plotting.

    Inserts ``np.nan`` at antimeridian wraps (|d_lon| > 180 deg) and polar
    crossings (|lat| > 89 deg with |d_lon| > 90 deg) so the renderer breaks the
    line there, and colours by *segment* (the start sample's state) so adjacent
    colour runs share their boundary point -> no gaps at coverage transitions.

    Returns a list of ``(state_value, x, y)``; plot each polyline in its colour
    (matplotlib breaks the line at the inserted NaNs automatically).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    state = np.asarray(state)
    n = lon.size
    if n == 0:
        return []
    if n == 1:
        return [(state[0], lon.copy(), lat.copy())]

    # break flags per segment i (point i -> i+1), length n-1
    dlon = np.abs(np.diff(lon))
    near_pole = np.maximum(np.abs(lat[:-1]), np.abs(lat[1:])) > polar_lat_deg
    brk = (dlon > lon_wrap_deg) | (near_pole & (dlon > polar_jump_deg))

    # colour each segment by its start sample, then group consecutive runs
    seg_state = state[:-1]
    change = np.flatnonzero(seg_state[1:] != seg_state[:-1]) + 1
    run_starts = np.concatenate(([0], change))
    run_ends = np.concatenate((change, [n - 1]))

    runs = []
    for s, e in zip(run_starts, run_ends):
        # run = segments [s, e) -> points [s, e] inclusive (point e shared)
        xs = lon[s:e + 1].copy()
        ys = lat[s:e + 1].copy()
        local = np.flatnonzero(brk[s:e])
        if local.size:
            ins = local + 1
            xs = np.insert(xs, ins, np.nan)
            ys = np.insert(ys, ins, np.nan)
        runs.append((seg_state[s], xs, ys))
    return runs


def _build_static_figure(data: dict) -> "Figure":
    """Create the full static track figure with visibility color coding."""
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lons = data["lons"]
    lats = data["lats"]
    vis_mask = data["vis_mask"]
    station_names = data["station_names"]
    n = len(lons)

    # Precompute per-step dominant color
    step_colors = [_dominant_color(vis_mask, i, station_names) for i in range(n)]

    # Split track at pole crossings so each pass is drawn independently.
    # Within one pass (south→north or north→south) the longitude is nearly
    # constant for a polar orbit, so no fan-shaped artifacts appear.
    boundaries = _pass_boundaries(lats, lons)

    with plt.rc_context(MPL_DARK):
        fig = Figure(figsize=(11, 5.5), tight_layout={"pad": 0.7})
        ax  = fig.add_subplot(111)

    _draw_moon_bg(ax)

    # When there is no station-visibility data (Orbit Designer / Keplerian
    # preview), color the track by elapsed time so the slow westward drift
    # is visible; otherwise keep the per-station visibility coloring.
    _time_colored = (vis_mask.size == 0) and ("t_s" in data) and (n > 1)
    if _time_colored:
        from matplotlib.collections import LineCollection
        _t = np.asarray(data["t_s"], dtype=float)
        _span = max(float(_t[-1] - _t[0]), 1e-9)
        _cmap = plt.get_cmap("plasma")
        _lc = None
        for ki in range(len(boundaries) - 1):
            p0, p1 = int(boundaries[ki]), int(boundaries[ki + 1])
            if p1 - p0 < 2:
                continue
            _pts = np.column_stack([lons[p0:p1], lats[p0:p1]])
            _segs = np.stack([_pts[:-1], _pts[1:]], axis=1)
            _lc = LineCollection(_segs, cmap=_cmap, zorder=4, linewidth=1.6, alpha=0.9)
            _lc.set_array((_t[p0:p1 - 1] - _t[0]) / _span)
            _lc.set_clim(0.0, 1.0)
            ax.add_collection(_lc)
        if _lc is not None:
            _cb = fig.colorbar(_lc, ax=ax, fraction=0.025, pad=0.01)
            _cb.set_label("orbit time  (start → end)", color=C.TEXT_MUTED, fontsize=FONT.SIZE_XS)
            _cb.ax.tick_params(colors=C.TEXT_TICK, labelsize=FONT.SIZE_XS)
            _cb.outline.set_edgecolor(C.BORDER_MAIN)
    else:
        # Robust per-state polylines: NaN breaks at antimeridian wraps and
        # polar crossings, plus shared boundary points across coverage changes
        # (no gaps). step_colors already encodes the per-sample colour/state.
        for _color, _xs, _ys in _ground_track_runs(lons, lats, step_colors):
            ax.plot(
                _xs, _ys, "-", color=str(_color),
                linewidth=1.6, alpha=0.88, zorder=4,
                solid_capstyle="round",
            )

    # Start / end markers
    ax.plot(lons[0],  lats[0],  "o", color=C.GREEN,  ms=6, zorder=7, label="Start")
    ax.plot(lons[-1], lats[-1], "s", color=C.YELLOW, ms=5, zorder=7, label="End")

    # Coverage stats
    covered = int(np.any(vis_mask, axis=0).sum()) if vis_mask.size else 0
    pct = 100.0 * covered / max(n, 1)

    # Axes
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(colors=C.TEXT_TICK, labelsize=FONT.SIZE_XS)
    ax.set_xlabel("Selenographic Longitude (°)", color=C.TEXT_MUTED, fontsize=FONT.SIZE_SM - 1)
    ax.set_ylabel("Selenographic Latitude (°)",  color=C.TEXT_MUTED, fontsize=FONT.SIZE_SM - 1)
    ax.set_title(
        f"Selenographic Ground Track — {data['duration_days']:.0f} days  "
        f"|  Coverage: {pct:.0f}%",
        color=C.TEXT_SECONDARY, fontsize=FONT.SIZE_SM,
    )
    for sp in ax.spines.values():
        sp.set_edgecolor(C.BORDER_MAIN)

    # Legend
    seen_nets: dict[str, str] = {}
    for si, name in enumerate(station_names):
        net = _network(name)
        if net not in seen_nets:
            seen_nets[net] = NET_COLORS.get(net, C.TEXT_MUTED)
    handles = [
        Line2D([0], [0], color=col, linewidth=2.5, label=net)
        for net, col in seen_nets.items()
    ]
    handles += [
        Patch(facecolor=_NO_VIS_COLOR, label="No coverage"),
        Line2D([0], [0], marker="o", color=C.GREEN,  linestyle="None", ms=6, label="Start"),
        Line2D([0], [0], marker="s", color=C.YELLOW, linestyle="None", ms=5, label="End"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        fontsize=FONT.SIZE_XS, framealpha=0.7,
        facecolor=C.BG_PANEL, edgecolor=C.BORDER_MID,
    )
    for t in leg.get_texts():
        t.set_color(C.TEXT_SECONDARY)

    ax.text(
        0.985, 0.025,
        "Moon rotates ~27× slower than Earth — passes drift only ~1°/orbit,\n"
        "so selenographic tracks stay near-vertical.",
        transform=ax.transAxes, ha="right", va="bottom",
        color=C.TEXT_MUTED, fontsize=FONT.SIZE_XS - 1, alpha=0.9, zorder=8,
    )

    return fig


# ---------------------------------------------------------------------------
# Net label helper (sidebar)
# ---------------------------------------------------------------------------

def _net_label(net: str, color: str) -> QLabel:
    lbl = QLabel(f"  {net}")
    lbl.setStyleSheet(
        f"QLabel {{ color:{color}; font-size:{FONT.SIZE_SM}px; font-weight:bold;"
        f" border-bottom:1px solid {color}; margin-top:6px; }}"
    )
    return lbl


# ---------------------------------------------------------------------------
# Keplerian orbit designer helpers (no SPICE — for quick exploration only)
# ---------------------------------------------------------------------------

def _solve_kepler(M: np.ndarray, e: float) -> np.ndarray:
    """Newton iterations on Kepler's equation M = E − e·sin(E)."""
    E = M.copy()
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-11:
            break
    return E


def _keplerian_ground_track(
    alt_km: float, e: float,
    inc_deg: float, raan_deg: float, argp_deg: float, nu0_deg: float,
    duration_days: float, step_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Keplerian propagation → selenographic (lon, lat) in degrees.

    Selenographic frame: simplified as a constant-rate z-rotation (OMEGA_MOON)
    starting aligned with J2000 at t = 0.  Good enough for design/exploration.
    """
    a    = R_MOON_M + alt_km * 1_000.0
    n_mm = np.sqrt(MU_MOON_M3 / a ** 3)
    T    = 2 * np.pi / n_mm
    if step_s is None:
        step_s = max(20.0, T / 300)

    t    = np.arange(0.0, duration_days * 86400.0 + step_s, step_s)
    inc  = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu0  = np.radians(nu0_deg)

    # True anomaly → mean anomaly at epoch
    E0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2),
                           np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)

    M  = (M0 + n_mm * t) % (2 * np.pi)
    E  = _solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                           np.sqrt(1 - e) * np.cos(E / 2))
    r  = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))

    # Perifocal → inertial via P, Q unit vectors
    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    ca, sa = np.cos(argp), np.sin(argp)

    Px =  cr * ca - sr * sa * ci;  Qx = -cr * sa - sr * ca * ci
    Py =  sr * ca + cr * sa * ci;  Qy = -sr * sa + cr * ca * ci
    Pz =          sa * si;         Qz =           ca * si

    xp = r * np.cos(nu);  yp = r * np.sin(nu)
    rx = xp * Px + yp * Qx
    ry = xp * Py + yp * Qy
    rz = xp * Pz + yp * Qz

    # Selenographic: Moon rotates around z at OMEGA_MOON
    th  = OMEGA_MOON * t
    cth, sth = np.cos(th), np.sin(th)
    rxs =  rx * cth + ry * sth
    rys = -rx * sth + ry * cth
    rzs =  rz

    rm   = np.sqrt(rxs ** 2 + rys ** 2 + rzs ** 2)
    lats = np.degrees(np.arcsin(np.clip(rzs / rm, -1.0, 1.0)))
    lons = np.degrees(np.arctan2(rys, rxs))
    return t, lons, lats


def _draw_orbit_preview(
    alt_km: float, e: float,
    inc_deg: float, raan_deg: float, argp_deg: float, nu0_deg: float,
    *,
    fig: "Figure | None" = None,
) -> "Figure":
    """3-D Moon-centered orbit preview for the orbit designer."""
    from matplotlib.figure import Figure

    if fig is None:
        fig = Figure(figsize=(5.2, 5.2), tight_layout={"pad": 0.35})
    else:
        fig.clf()
    fig.patch.set_facecolor(C.BG_DEEP)
    ax = fig.add_subplot(111, projection="3d")
    _draw_moon_sphere(ax)

    x, y, z = _keplerian_orbit_xyz(
        alt_km=alt_km, e=e, inc_deg=inc_deg,
        raan_deg=raan_deg, argp_deg=argp_deg,
    )
    ax.plot(x, y, z, color=C.CYAN, linewidth=1.9, alpha=0.92, zorder=5)

    x0, y0, z0 = _keplerian_orbit_xyz(
        alt_km=alt_km, e=e, inc_deg=inc_deg,
        raan_deg=raan_deg, argp_deg=argp_deg,
        nu_deg=[nu0_deg],
    )
    ax.plot(x0, y0, z0, "o", color="white", ms=6,
            markeredgecolor=C.CYAN, markeredgewidth=1.2, zorder=9)

    xp, yp, zp = _keplerian_orbit_xyz(
        alt_km=alt_km, e=e, inc_deg=inc_deg,
        raan_deg=raan_deg, argp_deg=argp_deg,
        nu_deg=[0.0],
    )
    xa, ya, za = _keplerian_orbit_xyz(
        alt_km=alt_km, e=e, inc_deg=inc_deg,
        raan_deg=raan_deg, argp_deg=argp_deg,
        nu_deg=[180.0],
    )
    ax.plot(xp, yp, zp, "v", color=C.GREEN, ms=5, zorder=8)
    ax.plot(xa, ya, za, "^", color=C.YELLOW, ms=5, zorder=8)

    metrics = _orbit_metrics(alt_km, e)
    _configure_orbit_axis(ax, _orbit_axis_limit(metrics))
    ax.set_title(
        f"a = {metrics['a_km']:.0f} km    e = {e:.3f}    i = {inc_deg:.0f} deg",
        fontsize=9, color=C.TEXT_SECONDARY, pad=5,
    )
    return fig


def _orbit_basis(
    inc_deg: float, raan_deg: float, argp_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return perifocal P/Q unit vectors in the inertial frame."""
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    ca, sa = np.cos(argp), np.sin(argp)
    p_vec = np.array([
        cr * ca - sr * sa * ci,
        sr * ca + cr * sa * ci,
        sa * si,
    ])
    q_vec = np.array([
        -cr * sa - sr * ca * ci,
        -sr * sa + cr * ca * ci,
        ca * si,
    ])
    return p_vec, q_vec


def _keplerian_orbit_xyz(
    alt_km: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    nu_deg: list[float] | np.ndarray | None = None,
    samples: int = 420,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Moon-centered inertial orbit points in km."""
    a_km = R_MOON_M / 1_000.0 + alt_km
    p_km = a_km * (1.0 - e ** 2)
    if nu_deg is None:
        nu = np.linspace(0.0, 2.0 * np.pi, samples)
    else:
        nu = np.radians(np.asarray(nu_deg, dtype=float))
    r_km = p_km / (1.0 + e * np.cos(nu))
    p_vec, q_vec = _orbit_basis(inc_deg, raan_deg, argp_deg)
    xyz = (r_km * np.cos(nu))[:, None] * p_vec + (r_km * np.sin(nu))[:, None] * q_vec
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


def _orbit_metrics(alt_km: float, e: float) -> dict[str, float]:
    a_km = R_MOON_M / 1_000.0 + alt_km
    r_peri_km = a_km * (1.0 - e)
    r_apo_km = a_km * (1.0 + e)
    period_s = 2.0 * np.pi * np.sqrt((a_km * 1_000.0) ** 3 / MU_MOON_M3)
    return {
        "a_km": a_km,
        "peri_alt_km": r_peri_km - R_MOON_M / 1_000.0,
        "apo_alt_km": r_apo_km - R_MOON_M / 1_000.0,
        "apo_radius_km": r_apo_km,
        "period_min": period_s / 60.0,
    }


def _orbit_axis_limit(metrics: dict[str, float]) -> float:
    moon_km = R_MOON_M / 1_000.0
    return max(moon_km * 1.35, metrics["apo_radius_km"] * 1.08)


def _draw_moon_sphere(ax) -> None:
    moon_km = R_MOON_M / 1_000.0
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    x = moon_km * np.outer(np.cos(u), np.sin(v))
    y = moon_km * np.outer(np.sin(u), np.sin(v))
    z = moon_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x, y, z,
        color="#30384f", edgecolor="#46516d", linewidth=0.12,
        alpha=0.78, shade=False, zorder=1,
    )
    equ = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.plot(
        moon_km * np.cos(equ), moon_km * np.sin(equ), np.zeros_like(equ),
        color=C.BORDER_EQUATOR, linewidth=0.8, alpha=0.7, zorder=2,
    )


def _configure_orbit_axis(ax, limit_km: float) -> None:
    ax.set_facecolor(C.BG_DEEP)
    ax.set_xlim(-limit_km, limit_km)
    ax.set_ylim(-limit_km, limit_km)
    ax.set_zlim(-limit_km, limit_km)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=23, azim=38)
    ax.set_xlabel("X km", fontsize=8, color=C.TEXT_MUTED, labelpad=-3)
    ax.set_ylabel("Y km", fontsize=8, color=C.TEXT_MUTED, labelpad=-3)
    ax.set_zlabel("Z km", fontsize=8, color=C.TEXT_MUTED, labelpad=-3)
    ax.tick_params(colors=C.TEXT_TICK, labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor(C.BG_PANEL)
            axis.pane.set_edgecolor(C.BORDER_MAIN)
            axis._axinfo["grid"]["color"] = C.BORDER_MAIN
            axis._axinfo["grid"]["linewidth"] = 0.4
        except Exception:
            pass


# Orbital element slider config: (key, label, unit, lo, hi, default, step, decimals)
_OE_CONFIG: list[tuple] = [
    ("alt_km",   "Altitude (h)",      "km", 50,  9000, 200, 50,    0),
    ("e",        "Eccentricity (e)",  "",   0,   0.9,  0.0, 0.001, 3),
    ("inc_deg",  "Inclination (i)",   "°",  0,   180,  90,  1,     0),
    ("raan_deg", "RAAN (Ω)",          "°",  0,   360,  0,   1,     0),
    ("argp_deg", "Arg. Perigee (ω)",  "°",  0,   360,  0,   1,     0),
    ("nu0_deg",  "True Anomaly (ν₀)", "°",  0,   360,  0,   1,     0),
]


class _OrbitDesigner(QWidget):
    """Orbital-element editor with a live 3-D orbit preview."""

    def __init__(self, gt_ctrl: "GroundTrackController", parent=None) -> None:
        super().__init__(parent)
        self._gt      = gt_ctrl
        self._spins:   dict[str, QDoubleSpinBox] = {}
        self._sliders: dict[str, QSlider]         = {}
        self._target_elems: dict[str, float] = {}
        self._display_elems: dict[str, float] = {}
        self._preview_phase_deg = 0.0
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(66)
        self._preview_timer.timeout.connect(self._animate_preview)
        self._fig = None
        self._ax = None
        self._canvas = None
        self._orbit_line = None
        self._craft_dot = None
        self._peri_marker = None
        self._apo_marker = None
        self._preview_caption = None
        self._build()
        self._target_elems = self._get_elements()
        self._display_elems = dict(self._target_elems)
        self._init_preview_canvas()
        self._redraw_preview(self._display_elems)
        self._preview_timer.start()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        from PyQt5.QtWidgets import QSplitter

        _box_ss = (
            f"QGroupBox {{ color:{C.TEXT_SECONDARY}; font-size:12px;"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:4px;"
            f" margin-top:8px; padding-top:4px; }}"
            f"QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}"
        )
        _spin_ss = (
            f"QDoubleSpinBox {{ background:{C.BG_DEEP}; color:{C.TEXT_PRIMARY};"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:3px;"
            f" font-size:11px; padding:1px 3px; }}"
        )
        _slider_ss = (
            f"QSlider::groove:horizontal {{ height:4px; background:{C.BORDER_MAIN};"
            f" border-radius:2px; }}"
            f"QSlider::handle:horizontal {{ width:12px; height:12px; margin:-4px 0;"
            f" background:{C.CYAN}; border-radius:6px; }}"
            f"QSlider::sub-page:horizontal {{ background:{C.CYAN}; border-radius:2px; }}"
        )
        _btn_ss = (
            f"QPushButton {{ background:{C.BG_HOVER}; color:{C.TEXT_PRIMARY};"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:4px;"
            f" padding:5px 12px; font-size:12px; }}"
            f"QPushButton:hover {{ background:{C.BG_ACTIVE}; }}"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # ── Left: 3-D orbit preview ─────────────────────────────────
        self._preview_holder = QWidget()
        ph_lay = QVBoxLayout(self._preview_holder)
        ph_lay.setContentsMargins(0, 0, 0, 0)
        self._preview_holder.setMinimumWidth(300)

        # ── Right: controls ──────────────────────────────────────────
        ctrl_inner = QWidget()
        ctrl_lay   = QVBoxLayout(ctrl_inner)
        ctrl_lay.setSpacing(8)
        ctrl_lay.setContentsMargins(6, 6, 6, 6)

        # Orbital elements group
        elems_box = QGroupBox("Orbital Elements")
        elems_box.setStyleSheet(_box_ss)
        elems_lay = QVBoxLayout(elems_box)
        elems_lay.setSpacing(5)

        for key, label, unit, lo, hi, default, step, dec in _OE_CONFIG:
            lbl = QLabel(label + ":")
            lbl.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:11px;")

            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(dec)
            spin.setValue(default)
            spin.setFixedWidth(95)
            spin.setStyleSheet(_spin_ss)
            if unit:
                spin.setSuffix(f" {unit}")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setStyleSheet(_slider_ss)
            _rng = hi - lo
            slider.setValue(int((default - lo) / _rng * 1000) if _rng else 0)

            def _sl_ch(val, s=spin, lo_=lo, hi_=hi):
                s.blockSignals(True)
                s.setValue(lo_ + (hi_ - lo_) * val / 1000)
                s.blockSignals(False)
                self._set_preview_target()

            def _sp_ch(val, sl=slider, lo_=lo, hi_=hi):
                sl.blockSignals(True)
                rng_ = hi_ - lo_
                sl.setValue(max(0, min(1000, int((val - lo_) / rng_ * 1000))) if rng_ else 0)
                sl.blockSignals(False)
                self._set_preview_target()

            slider.valueChanged.connect(_sl_ch)
            spin.valueChanged.connect(_sp_ch)
            self._spins[key]   = spin
            self._sliders[key] = slider

            elems_lay.addWidget(lbl)
            sl_row = QWidget()
            sl_lay = QHBoxLayout(sl_row)
            sl_lay.setContentsMargins(0, 0, 0, 0)
            sl_lay.setSpacing(6)
            sl_lay.addWidget(slider, 1)
            sl_lay.addWidget(spin)
            elems_lay.addWidget(sl_row)

        ctrl_lay.addWidget(elems_box)

        # Duration group
        dur_box = QGroupBox("Propagation")
        dur_box.setStyleSheet(_box_ss)
        dur_form = QFormLayout(dur_box)
        dur_form.setLabelAlignment(Qt.AlignRight)
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.1, 30)
        self._dur_spin.setValue(3.0)
        self._dur_spin.setSuffix(" days")
        self._dur_spin.setDecimals(1)
        self._dur_spin.setStyleSheet(_spin_ss)
        dur_form.addRow("Duration:", self._dur_spin)
        ctrl_lay.addWidget(dur_box)

        gen_btn = QPushButton("▶  Generate Ground Track")
        gen_btn.setStyleSheet(_btn_ss)
        gen_btn.setMinimumHeight(38)
        gen_btn.setToolTip(
            "Propagate this Keplerian orbit and display the selenographic ground track.\n"
            "Simplified model (no n-body perturbations). Use 'Compute Ground Track'\n"
            "on the Ground Track tab for accurate SPICE-based results."
        )
        gen_btn.clicked.connect(self._generate_track)
        ctrl_lay.addWidget(gen_btn)

        note = QLabel(
            "Keplerian only — no n-body perturbations or realistic ephemeris.\n"
            "Good for orbit design; use SPICE computation for accurate results."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C.TEXT_TICK}; font-size:10px;")
        ctrl_lay.addWidget(note)
        ctrl_lay.addStretch()

        # Scroll wrapper
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setWidget(ctrl_inner)
        ctrl_scroll.setMinimumWidth(280)
        ctrl_scroll.setMaximumWidth(360)
        ctrl_scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:{C.BG_PANEL}; }}"
        )

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._preview_holder)
        splitter.addWidget(ctrl_scroll)
        splitter.setSizes([440, 310])
        outer.addWidget(splitter)

    # ------------------------------------------------------------------
    def _get_elements(self) -> dict:
        return {k: sp.value() for k, sp in self._spins.items()}

    def _update_preview(self) -> None:
        self._display_elems = self._get_elements()
        self._redraw_preview(self._display_elems)

    def _init_preview_canvas(self) -> None:
        self._redraw_preview(self._display_elems)

    def _redraw_preview(self, elems: dict) -> None:
        try:
            if self._canvas is None:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
                fig = _draw_orbit_preview(**elems)
                canvas = FigureCanvasQTAgg(fig)
                self._fig = fig
                self._ax = fig.axes[0] if fig.axes else None
                self._canvas = canvas
                lay = self._preview_holder.layout()
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w:
                        w.setParent(None)
                        w.deleteLater()
                lay.addWidget(canvas)
            else:
                _draw_orbit_preview(**elems, fig=self._fig)
                self._ax = self._fig.axes[0] if self._fig.axes else None
                self._canvas.draw_idle()
        except Exception:
            pass

    def _set_preview_target(self) -> None:
        self._target_elems = self._get_elements()
        self._display_elems = dict(self._target_elems)
        self._redraw_preview(self._display_elems)

    def _animate_preview(self) -> None:
        if self._ax is None or self._canvas is None:
            return
        self._preview_phase_deg = (self._preview_phase_deg + 2.4) % 360
        try:
            self._ax.view_init(elev=25, azim=self._preview_phase_deg)
            self._canvas.draw_idle()
        except Exception:
            pass

    # Pause the expensive 3-D redraw loop whenever the designer is not
    # visible, so it does not bog down the rest of the UI on other pages.
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._preview_timer.stop()

    def _generate_track(self) -> None:
        elems = self._get_elements()
        dur   = self._dur_spin.value()

        a      = R_MOON_M + elems["alt_km"] * 1_000
        r_peri = a * (1 - elems["e"])
        if r_peri < R_MOON_M * 1.005:
            QMessageBox.warning(
                self, "Invalid orbit",
                "Periapsis is below (or too close to) the lunar surface.\n"
                "Reduce eccentricity or increase altitude.",
            )
            return

        try:
            t, lons, lats = _keplerian_ground_track(duration_days=dur, **elems)
        except Exception as exc:
            QMessageBox.critical(self, "Propagation failed", str(exc))
            return

        n    = len(t)
        data = {
            "lons":          lons,
            "lats":          lats,
            "t_s":           t,
            "vis_mask":      np.zeros((0, n), dtype=bool),
            "station_names": [],
            "duration_days": dur,
            "epoch_utc":     "Keplerian (analytical)",
        }
        data["_pass_bounds"] = _pass_boundaries(lats, lons)
        self._gt._show_keplerian_track(data, elems)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GroundTrackController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)

        self._checkboxes: dict[str, QCheckBox] = {}
        self._worker: _TrackWorker | None = None
        self._canvas     = None
        self._fig        = None
        self._data: dict | None = None

        # Animation state
        self._anim_timer  = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_idx    = 0
        self._anim_dot    = None
        self._anim_tail   = None
        self._anim_label  = None
        self._anim_ax     = None

        self._tab_widget: "QTabWidget | None" = None

        self._setup()
        self._add_orbit_designer_tab()

    def _setup(self) -> None:
        self._build_checkboxes()
        self.selectAllBtn.clicked.connect(self._select_all)
        self.deselectAllBtn.clicked.connect(self._deselect_all)
        self.computeBtn.clicked.connect(self._compute)
        self.playBtn.clicked.connect(self._play_pause)
        self.resetBtn.clicked.connect(self._reset_anim)
        self._build_export_buttons()
        self._build_log_panel()
        self._set_status("Select stations and compute.", C.TEXT_MUTED)
        self._init_placeholder()

    def _build_log_panel(self) -> None:
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFixedHeight(82)
        self._log_edit.setPlaceholderText("Computation log…")
        self._log_edit.setStyleSheet(
            f"QPlainTextEdit {{ background:{C.BG_DEEP}; color:{C.TEXT_MUTED};"
            f" font-family:Consolas,monospace; font-size:10px;"
            f" border:1px solid {C.BORDER_MAIN}; border-radius:3px; }}"
        )
        self._log_edit.setVisible(False)
        self.controlLayout.addWidget(self._log_edit)

    def _log(self, msg: str) -> None:
        self._log_edit.setVisible(True)
        self._log_edit.appendPlainText(msg)
        self._log_edit.ensureCursorVisible()

    def _build_export_buttons(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        self.exportPngBtn = QPushButton("Export PNG")
        self.exportCsvBtn = QPushButton("Export CSV")
        self.exportPngBtn.setEnabled(False)
        self.exportCsvBtn.setEnabled(False)
        self.exportPngBtn.clicked.connect(self._export_png)
        self.exportCsvBtn.clicked.connect(self._export_csv)
        row.addWidget(self.exportPngBtn)
        row.addWidget(self.exportCsvBtn)
        self.controlLayout.insertLayout(self.controlLayout.indexOf(self.statusLabel), row)

    def on_shown(self) -> None:
        self._sync_duration_from_settings()

    def _sync_duration_from_settings(self) -> None:
        from PyQt5.QtCore import QSettings
        s = QSettings("LunarOD", "DesktopApp")
        try:
            dur_days = float(s.value("dynamics/duration_days", self.durationSpin.value()))
            self.durationSpin.setValue(dur_days)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Checkbox setup
    # ------------------------------------------------------------------

    def _build_checkboxes(self) -> None:
        try:
            from lunar_od import range_rate_stations
            stations = range_rate_stations()
        except Exception:
            return

        layout = self.stationCheckLayout
        groups: dict[str, list] = {}
        for s in stations:
            groups.setdefault(_network(s.name), []).append(s)

        for net, slist in sorted(groups.items()):
            color = NET_COLORS.get(net, C.TEXT_MUTED)
            layout.addWidget(_net_label(net, color))
            for s in slist:
                cb = QCheckBox(s.name)
                cb.setChecked(True)
                cb.setStyleSheet(
                    f"QCheckBox {{ color:{C.TEXT_SECONDARY}; font-size:{FONT.SIZE_MD}px; }}"
                    f"QCheckBox::indicator:checked {{ background:{color}; border:1px solid {color}; }}"
                )
                layout.addWidget(cb)
                self._checkboxes[s.name] = cb
        layout.addStretch()

    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        selected = [n for n, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            self._set_status("Select at least one station.", C.YELLOW)
            return

        self._anim_timer.stop()
        self.computeBtn.setEnabled(False)
        self.animGroup.setEnabled(False)
        self.progressBar.setVisible(True)
        self._log_edit.clear()
        self._log(f"Starting: {len(selected)} station(s), {self.durationSpin.value():.0f} days")
        self._set_status("Computing…", C.YELLOW)

        self._worker = _TrackWorker(
            station_names=selected,
            duration_days=self.durationSpin.value(),
            elev_deg=self.elevSpin.value(),
            max_gap_min=self.gapSpin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_data)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._set_status(msg, C.YELLOW)
        self._log(msg)

    def _on_data(self, data: dict) -> None:
        self.computeBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        self._data = data
        # Pre-compute pass boundaries once; used both in drawing and animation tail
        self._data["_pass_bounds"] = _pass_boundaries(data["lats"], data["lons"])
        n = len(data["lons"])
        dur = data["duration_days"]
        covered = int(np.any(data["vis_mask"], axis=0).sum()) if data["vis_mask"].size else 0
        pct = 100.0 * covered / max(n, 1)
        done_msg = f"{n} points, {dur:.0f} days — coverage {pct:.0f}%"
        self._set_status(done_msg, C.GREEN)
        self._log(f"Done: {done_msg}")

        fig = _build_static_figure(data)
        self._fig = fig
        self._set_canvas(fig)
        self.exportPngBtn.setEnabled(True)
        self.exportCsvBtn.setEnabled(True)

        # Prepare animated artists on the same axes
        self._anim_ax = fig.axes[0]
        self._anim_dot,  = self._anim_ax.plot([], [], "o", color="white",
                                                ms=8, zorder=10,
                                                markeredgecolor=C.CYAN,
                                                markeredgewidth=1.5)
        self._anim_tail, = self._anim_ax.plot([], [], "-", color=C.CYAN,
                                               linewidth=2.2, alpha=_TAIL_ALPHA, zorder=9)
        self._anim_label = self._anim_ax.text(
            0.99, 0.97, "",
            transform=self._anim_ax.transAxes,
            ha="right", va="top",
            color=C.TEXT_PRIMARY,
            fontsize=FONT.SIZE_SM,
            fontfamily=FONT.MONO,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=C.BG_PANEL, alpha=0.8,
                      edgecolor=C.BORDER_MID),
            zorder=11,
        )
        self._anim_idx = 0
        self.animGroup.setEnabled(True)
        self.playBtn.setText("▶ Play")

    def _on_fail(self, msg: str) -> None:
        self.computeBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        self.animGroup.setEnabled(False)
        short = msg.splitlines()[0][:120] if msg else "Unknown error"
        self._set_status(f"Error: {short}", C.RED)
        self._log(f"FAILED: {short}")
        _show_error_dialog(self, "Computation failed", msg)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _play_pause(self) -> None:
        if self._data is None:
            return
        if self._anim_timer.isActive():
            self._anim_timer.stop()
            self.playBtn.setText("▶ Play")
        else:
            interval_ms = max(40, 200 // self.speedSpin.value())
            self._anim_timer.start(interval_ms)
            self.playBtn.setText("⏸ Pause")

    def _reset_anim(self) -> None:
        self._anim_timer.stop()
        self.playBtn.setText("▶ Play")
        self._anim_idx = 0
        if self._anim_dot is not None:
            self._anim_dot.set_data([], [])
            self._anim_tail.set_data([], [])
            self._anim_label.set_text("")
            if self._canvas:
                self._canvas.draw_idle()

    def _anim_step(self) -> None:
        if self._data is None or self._anim_dot is None:
            return

        lons = self._data["lons"]
        lats = self._data["lats"]
        t_s  = self._data["t_s"]
        n    = len(lons)

        skip = max(1, self.speedSpin.value())
        self._anim_idx = (self._anim_idx + skip) % n

        idx = self._anim_idx

        # Dot
        self._anim_dot.set_data([lons[idx]], [lats[idx]])

        # Tail: last _TAIL_LEN points clipped to the current pole-to-pole pass
        # so it never crosses a pole boundary (which would create fan artifacts).
        bounds = self._data.get("_pass_bounds", np.array([0, n]))
        pass_start = int(bounds[np.searchsorted(bounds, idx, side="right") - 1])
        tail_start = max(pass_start, idx - _TAIL_LEN)
        t_lons = lons[tail_start : idx + 1]
        t_lats = lats[tail_start : idx + 1]
        self._anim_tail.set_data(t_lons, t_lats)

        # Overlay text: elapsed time + visible stations
        t_day = t_s[idx] / 86400.0
        vis_names = _visible_station_names(
            self._data["vis_mask"], idx, self._data["station_names"]
        )
        lines = [f"D {t_day:.2f}"]
        if vis_names:
            lines.append("Tracking:")
            lines.extend(f"  {n}" for n in vis_names[:4])
        else:
            lines.append("(no coverage)")
        self._anim_label.set_text("\n".join(lines))

        if self._canvas:
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------

    def _set_canvas(self, fig: "Figure") -> None:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        canvas  = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, self.canvasHolder)
        toolbar.setStyleSheet(
            f"QToolBar {{ background:{C.BG_PANEL}; border:none; }}"
            f"QToolButton {{ color:{C.TEXT_SECONDARY}; background:transparent; }}"
        )

        lay = self.canvasHolder.layout()
        if lay is None:
            lay = QVBoxLayout(self.canvasHolder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        self._canvas = canvas

    def _init_placeholder(self) -> None:
        lay = QVBoxLayout(self.canvasHolder)
        lay.setContentsMargins(0, 0, 0, 0)
        ph = QLabel(
            "Ground track will appear here.\n"
            "Select stations on the left and click 'Compute Ground Track'."
        )
        ph.setAlignment(Qt.AlignCenter)
        ph.setStyleSheet(f"color:{C.TEXT_MUTED}; font-size:{FONT.SIZE_LG}px;")
        lay.addWidget(ph)

    def _set_status(self, text: str, color: str = C.TEXT_MUTED) -> None:
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(
            f"QLabel {{ color:{color}; font-size:{FONT.SIZE_MD}px; }}"
        )

    def _export_png(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "Export unavailable", "Compute ground track first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ground track PNG",
            str(Path.home() / "ground_track.png"),
            "PNG images (*.png);;PDF files (*.pdf)",
        )
        if path:
            self._fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=self._fig.get_facecolor())

    def _export_csv(self) -> None:
        if self._data is None:
            QMessageBox.information(self, "Export unavailable", "Compute ground track first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ground track CSV",
            str(Path.home() / "ground_track_visibility.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        data = self._data
        t_s = np.asarray(data["t_s"], dtype=float)
        lons = np.asarray(data["lons"], dtype=float)
        lats = np.asarray(data["lats"], dtype=float)
        mask = np.asarray(data["vis_mask"], dtype=bool)
        names = list(data["station_names"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "time_days", "lon_deg", "lat_deg", *names, "network_visible"])
            for idx, t in enumerate(t_s):
                row_mask = mask[:, idx] if mask.size else np.zeros(len(names), dtype=bool)
                writer.writerow([
                    f"{t:.3f}",
                    f"{t / 86400.0:.8f}",
                    f"{lons[idx]:.8f}",
                    f"{lats[idx]:.8f}",
                    *[int(v) for v in row_mask],
                    int(np.any(row_mask)),
                ])


    # ------------------------------------------------------------------
    # Orbit designer tab
    # ------------------------------------------------------------------

    def _add_orbit_designer_tab(self) -> None:
        """Wrap the existing splitter in a QTabWidget and add the Orbit Designer tab."""
        self._tab_widget = QTabWidget(self)
        self._tab_widget.setStyleSheet(
            f"QTabBar::tab {{ background:{C.BG_SIDEBAR}; color:{C.TEXT_MUTED};"
            f" padding:6px 18px; font-size:12px; border:none;"
            f" border-bottom:2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color:{C.NAV_ACTIVE_FG};"
            f" border-bottom-color:{C.NAV_ACTIVE}; }}"
            f"QTabBar::tab:hover {{ background:{C.BG_ACTIVE};"
            f" color:{C.TEXT_SECONDARY}; }}"
            f"QTabWidget::pane {{ border:none; }}"
        )
        self.mainLayout.removeWidget(self.splitter)
        self._tab_widget.addTab(self.splitter, "Ground Track")
        self._orbit_designer = _OrbitDesigner(self)
        self._tab_widget.addTab(self._orbit_designer, "Orbit Designer")
        self.mainLayout.addWidget(self._tab_widget)

    def _show_keplerian_track(self, data: dict, elems: dict) -> None:
        """Display a Keplerian ground track coming from the Orbit Designer."""
        self._anim_timer.stop()
        self._data = data
        n   = len(data["lons"])
        dur = data["duration_days"]
        self._set_status(
            f"Keplerian — {dur:.1f} d · {n} pts  "
            f"h={elems['alt_km']:.0f} km  e={elems['e']:.3f}  i={elems['inc_deg']:.0f}°",
            C.CYAN,
        )
        fig = _build_static_figure(data)
        self._fig = fig
        self._set_canvas(fig)
        self.exportPngBtn.setEnabled(True)
        self.exportCsvBtn.setEnabled(True)

        # Animated artists on the same axes
        self._anim_ax = fig.axes[0]
        self._anim_dot,  = self._anim_ax.plot([], [], "o", color="white",
                                                ms=8, zorder=10,
                                                markeredgecolor=C.CYAN,
                                                markeredgewidth=1.5)
        self._anim_tail, = self._anim_ax.plot([], [], "-", color=C.CYAN,
                                               linewidth=2.2, alpha=_TAIL_ALPHA, zorder=9)
        self._anim_label = self._anim_ax.text(
            0.99, 0.97, "",
            transform=self._anim_ax.transAxes,
            ha="right", va="top", color=C.TEXT_PRIMARY,
            fontsize=FONT.SIZE_SM, fontfamily=FONT.MONO,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C.BG_PANEL,
                      alpha=0.8, edgecolor=C.BORDER_MID),
            zorder=11,
        )
        self._anim_idx = 0
        self.animGroup.setEnabled(True)
        self.playBtn.setText("▶ Play")
        if self._tab_widget is not None:
            self._tab_widget.setCurrentIndex(0)


def _show_error_dialog(parent, title: str, msg: str) -> None:
    """Show a QMessageBox with the full traceback in the details pane."""
    from PyQt5.QtWidgets import QMessageBox
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Critical)
    short = msg.splitlines()[0][:200] if msg else "An error occurred."
    box.setText(short)
    if "\n" in msg:
        box.setDetailedText(msg)
    box.exec_()
