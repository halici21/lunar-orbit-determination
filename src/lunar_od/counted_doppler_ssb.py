"""Barycentric DSN counted Doppler for the production estimators (F2 WP2.1, MEAS-01).

``RangeRatePhysicsConfig(counted_doppler_model='barycentric_dsn')`` routes the production counted-Doppler
observable and its arc-initial-state Jacobian here, through the same dispatch point as the R4 four-event
model, so generation, residuals, BLS-LM/SRIF Jacobians and observability all consume one model.

The physics is :data:`lunar_od.relativistic_light_time.DSN_LIGHT_TIME_MODEL` (Phase C1-I): every light-time
leg solved in the solar-system barycentric frame, Sun and Earth Shapiro delays (Moyer eq. 8-38, gamma = 1)
and the station-measured interval in TT. Against the production Moon-centred Newtonian model this moves
counted Doppler by -68...+87 um/s (Phase C1 materiality report).

Station states are the production exact event-epoch ``sxform`` provider, made free of epoch quantisation
(:func:`lunar_od.relativistic_light_time.exact_epoch_station_state`). Spacecraft states are the production
cubic-Hermite history. The receive tags, count interval and clock offset follow the production convention
(count endpoints at ``t_mid -/+ Tc/2``, a receive tag ``t`` read as ``t + b``).

Unsupported production options fail closed: clock drift, transponder delay, ``output_unit='hz'``, the
legacy interpolated station grid and a non-default ``light_speed_mps``.
"""
from __future__ import annotations

import numpy as np

from . import relativistic_light_time as _rlt
from . import station_time as _station_time
from .history_domain import require_supported_epoch
from .radiometrics import BARYCENTRIC_DSN_COUNTED_DOPPLER_MODEL_VERSION as MODEL_VERSION  # noqa: F401
LIGHT_TIME_MODEL = _rlt.DSN_LIGHT_TIME_MODEL


class _GuardedHistory(_rlt.SpacecraftHistory):
    """Spacecraft history that refuses every epoch outside its support, as the production counted-Doppler
    solver does (``HistoryDomainError``; generators drop such candidates) instead of extrapolating."""

    def _require(self, t_s: float) -> None:
        require_supported_epoch(float(t_s), float(self.t_grid[0]), float(self.t_grid[-1]),
                                history_name="spacecraft_state", model_context=MODEL_VERSION,
                                consumer="barycentric_dsn counted Doppler")

    def state(self, t_s: float) -> np.ndarray:
        self._require(t_s)
        return super().state(t_s)

    def position_sensitivity(self, t_s: float) -> np.ndarray:
        self._require(t_s)
        return super().position_sensitivity(t_s)


def _require_supported(cfg) -> None:
    if cfg.station_clock_drift != 0.0:
        raise NotImplementedError("barycentric_dsn counted Doppler does not model station clock drift.")
    if cfg.transponder_delay_s != 0.0:
        raise NotImplementedError("barycentric_dsn counted Doppler supports zero transponder delay only.")
    if cfg.output_unit != "mps_equivalent":
        raise NotImplementedError("barycentric_dsn counted Doppler reports the m/s equivalent only.")
    if cfg.light_speed_mps != _rlt.C_LIGHT_MPS:
        raise NotImplementedError("barycentric_dsn counted Doppler uses the defined speed of light only.")
    if not cfg.exact_event_epoch_enabled:
        raise ValueError("barycentric_dsn counted Doppler requires the exact event-epoch station transform.")


def _context(cfg, station, t_grid_s, state_history_mci, earth_pos_mci_m, earth_vel_mci_mps, et0_s, n_cols):
    from .two_way_range import make_exact_sxform_station_state_provider

    _require_supported(cfg)
    if et0_s is None:
        raise ValueError("barycentric_dsn counted Doppler needs et0_s (SPICE ET of scenario t=0).")
    provider = make_exact_sxform_station_state_provider(station, float(et0_s), t_grid_s, earth_pos_mci_m,
                                                        earth_vel_mci_mps)
    station_state = _rlt.exact_epoch_station_state(lambda t: np.asarray(provider.state(t), dtype=float),
                                                   float(et0_s), t_grid_s, earth_pos_mci_m, earth_vel_mci_mps)
    sc = _GuardedHistory(t_grid_s, np.asarray(state_history_mci, dtype=float)[:, :n_cols])
    eph = _rlt.BarycentricEphemeris(float(et0_s))
    site = _station_time.Site.from_itrf_m(station.r_ecef_m)
    return station_state, sc, eph, site


def _media_kwargs(cfg, station) -> dict:
    """F2 WP2.2: the default-climatology media phase delay of both legs for 'barycentric_dsn_media'."""
    from .media import MediaModel, MediaSite
    from .radiometrics import BARYCENTRIC_DSN_MEDIA_COUNTED_DOPPLER_MODEL

    if cfg.counted_doppler_model != BARYCENTRIC_DSN_MEDIA_COUNTED_DOPPLER_MODEL:
        return {}
    return {"media": MediaModel(), "media_site": MediaSite.from_station(station)}


def barycentric_counted_doppler_observable(receive_mid_time_s, station, t_grid_s, state_history_mci,
                                           earth_pos_mci_m, earth_vel_mci_mps, cfg, *, et0_s=None) -> float:
    """Counted Doppler [m/s equivalent] under the DSN barycentric light-time model."""
    station_state, sc, eph, site = _context(cfg, station, t_grid_s, state_history_mci, earth_pos_mci_m,
                                            earth_vel_mci_mps, et0_s, 6)
    value, _, _ = _rlt.counted_doppler(float(receive_mid_time_s), cfg.count_interval_s, station_state, sc, eph,
                                       LIGHT_TIME_MODEL, site=site, clock_offset_s=cfg.station_clock_offset_s,
                                       **_media_kwargs(cfg, station))
    return float(value)


def barycentric_counted_doppler_initial_state_jacobian(receive_mid_time_s, station, t_grid_s,
                                                       augmented_state_history_mci, earth_pos_mci_m,
                                                       earth_vel_mci_mps, cfg, *, et0_s=None) -> np.ndarray:
    """d(counted Doppler)/d(arc-initial state), shape (6,), from the 42-column augmented history."""
    station_state, sc, eph, site = _context(cfg, station, t_grid_s, augmented_state_history_mci,
                                            earth_pos_mci_m, earth_vel_mci_mps, et0_s, 42)
    _, partial, _ = _rlt.counted_doppler(float(receive_mid_time_s), cfg.count_interval_s, station_state, sc, eph,
                                         LIGHT_TIME_MODEL, with_partial=True, site=site,
                                         clock_offset_s=cfg.station_clock_offset_s, **_media_kwargs(cfg, station))
    return np.asarray(partial, dtype=float).reshape(6)
