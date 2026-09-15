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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- campaign data (read-only, outside the repo; the R1/R1G arc) ----------
CAMPAIGN_ROOT = Path(r"C:/Users/erayh/Documents/Python/Grad/od_covariance_campaign")

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
) -> ArcFixture:
    """Propagate one window of the frozen arc and read its design matrices.

    ``t_start_s`` is measured from the campaign epoch.  The propagation always
    starts at the CAMPAIGN epoch and the window is sliced out afterwards, so
    every arc in a multi-arc study shares one physically continuous truth
    trajectory -- arcs differ by which observations are used, never by a
    different truth.
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
        generate_two_way_range_measurements,
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
    xf = sample_j2000_to_itrf93_transforms(et0 + t_grid[0], t_local)
    vis_cfg = VisibilityConfig(
        r_moon_mean_m=1_737_400.0, earth_rotation_rad_s=7.292115e-5,
        epoch_utc=spice.et2utc(et0 + t_grid[0], "ISOC", 3),
        min_elevation_deg=10.0)
    _, _, vis, _ = analyze_visibility_gap_with_transforms(
        t_local, x_true, C.STATIONS, C.get_earth_pos, xf, 0.0, vis_cfg)
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
    obs, pass_geo = generate_two_way_range_measurements(
        t_local, x_true, stations, vis_used, C.get_earth_pos, C.get_earth_vel,
        et0 + t_grid[0], noise=False, rng=None, config=tight)

    _, h_x0 = two_way_range_nominal_and_initial_jacobian(
        obs, pass_geo, nom48[:, :42])
    h_k = _two_way_range_k_srp_column(obs, nom48, h_x0)
    w = _two_way_range_weight_diagonal(obs, pass_geo)

    return ArcFixture(label=label, t_grid=t_grid, x_true=x_true, nom48=nom48,
                      obs=obs, pass_geo=pass_geo, station_name=station_name,
                      h_x0=np.asarray(h_x0, float), h_k=np.asarray(h_k, float),
                      w=np.asarray(w, float))


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
