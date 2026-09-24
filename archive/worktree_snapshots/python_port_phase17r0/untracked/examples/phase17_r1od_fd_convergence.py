"""PHASE 17-R1O-D - proper FD convergence sweep for the state Jacobian and the
K_SRP E2E sensitivity, following Phase 17A-R's plateau-then-roundoff
methodology rather than trusting one arbitrarily narrow step range.

The first qualification pass used only 4 step sizes for each and found
relative error INCREASING monotonically as the step shrank (1e-5 -> 1.25,
1e-2 -> 9.97e-4) -- i.e. the sweep never left the truncation-error arm and
never found the plateau, so declaring FAIL off that sweep would have been
premature. This script sweeps a much wider range (10 steps per quantity,
spanning 4 decades either side of the first sweep) to find the actual
minimum-error region before any pass/fail judgement.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from lunar_od.delta_dor import delta_dor_k_column, spacecraft_differential_delay  # noqa: E402

ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
DDOR_BASELINE = ("Goldstone DSN", "Canberra DSN")
W15_ORBITS = 15.0


def phi_6x6_history(nom48: np.ndarray) -> np.ndarray:
    n = nom48.shape[0]
    phi = np.empty((n, 6, 6), dtype=float)
    for i in range(n):
        phi[i] = nom48[i, 6:42].reshape((6, 6), order="F")
    return phi


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    from lunar_od.visibility import sample_j2000_to_itrf93_transforms
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    import spiceypy as spice
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    arc = build_range_arc(0.0, W15_ORBITS * t_orbit, label="W15_fd",
                          station_filter=ALL_STATIONS, cadence_s=90.0)
    by_name = {s.name: s for s in C.STATIONS}
    station_a, station_b = by_name[DDOR_BASELINE[0]], by_name[DDOR_BASELINE[1]]
    phi_hist = phi_6x6_history(arc.nom48)
    t_local = arc.t_grid - arc.t_grid[0]
    xf = sample_j2000_to_itrf93_transforms(et0 + arc.t_grid[0], t_local)

    def sun_at(t_s):
        return spice.spkezr("SUN", et0 + float(t_s), "J2000", "NONE", "MOON")[0][:3] * 1000.0

    def earth_at(_t_s):
        return np.array([384_400e3, 0.0, 0.0])

    i_test = len(arc.t_grid) // 3
    t_test = arc.t_grid[i_test]
    earth_pos_test = C.get_earth_pos(t_test)
    xf_test = xf[i_test]
    x0_nom = arc.nom48[0, :6].copy()
    t_grid_local = arc.t_grid - arc.t_grid[0]
    t_target_local = t_test - arc.t_grid[0]
    k0 = 0.01

    def d_s_at_x0(x0_pert):
        hist = propagate_state_with_k_sensitivity(
            t_grid_local, x0_pert, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=k0), rtol=1e-12, atol=1e-13,
            j2_moon=J2_MOON_UNNORMALIZED)
        phi_p = phi_6x6_history(hist)
        d_s_p, _, _, _, _, _ = spacecraft_differential_delay(
            t_test, station_a, station_b, arc.t_grid, hist[:, :6], phi_p,
            earth_pos_test, xf_test)
        return d_s_p

    def d_s_at_k(k_val):
        hist = propagate_state_with_k_sensitivity(
            t_grid_local, x0_nom, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=k_val), rtol=1e-12, atol=1e-13,
            j2_moon=J2_MOON_UNNORMALIZED)
        phi_p = phi_6x6_history(hist)
        d_s_p, _, _, _, _, _ = spacecraft_differential_delay(
            t_test, station_a, station_b, arc.t_grid, hist[:, :6], phi_p,
            earth_pos_test, xf_test)
        return d_s_p

    # analytic references
    d_s_t, h_x0_t, _, _, _, _ = spacecraft_differential_delay(
        t_test, station_a, station_b, arc.t_grid, arc.nom48[:, :6], phi_hist,
        earth_pos_test, xf_test)
    h_k_composed = delta_dor_k_column(h_x0_t, arc.nom48[i_test, 6:42], arc.nom48[i_test, 42:48])

    # ---------------- s28 wide FD sweep: position component 0 -------------
    hdr("s28 - WIDE FD CONVERGENCE SWEEP, STATE COMPONENT 0 (position x)")
    print("  analytic dDs/dx0[0] = %.10e" % h_x0_t[0])
    rows_x0 = []
    steps_m = np.geomspace(1e-6, 1e4, 12)
    for h in steps_m:
        xp = x0_nom.copy(); xp[0] += h
        xm = x0_nom.copy(); xm[0] -= h
        fd = (d_s_at_x0(xp) - d_s_at_x0(xm)) / (2 * h)
        rel = abs(fd - h_x0_t[0]) / abs(h_x0_t[0])
        rows_x0.append(dict(step_m=h, fd=fd, analytic=h_x0_t[0], relative_error=rel))
        print("  h=%.3e m   FD=%.10e   rel_err=%.3e" % (h, fd, rel))
    best_x0 = min(rows_x0, key=lambda r: r["relative_error"])
    print("  best: h=%.3e m, rel_err=%.3e" % (best_x0["step_m"], best_x0["relative_error"]))
    with (ARTIFACTS / "r1od_fd_convergence_state.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows_x0[0].keys()))
        w_.writeheader(); w_.writerows(rows_x0)

    # ---------------- s28 wide FD sweep: K -------------------------------
    hdr("s28 - WIDE FD CONVERGENCE SWEEP, K_SRP")
    print("  composed H_K = %.10e" % h_k_composed)
    rows_k = []
    dk_fracs = np.geomspace(1e-7, 3e-1, 12)
    for frac in dk_fracs:
        dk = frac * k0
        fd = (d_s_at_k(k0 + dk) - d_s_at_k(k0 - dk)) / (2 * dk)
        rel = abs(fd - h_k_composed) / abs(h_k_composed)
        rows_k.append(dict(dk_fraction=frac, dk=dk, fd=fd, composed=h_k_composed,
                           relative_error=rel))
        print("  dK/K=%.3e   FD=%.10e   rel_err=%.3e" % (frac, fd, rel))
    best_k = min(rows_k, key=lambda r: r["relative_error"])
    print("  best: dK/K=%.3e, rel_err=%.3e" % (best_k["dk_fraction"], best_k["relative_error"]))
    with (ARTIFACTS / "r1od_fd_convergence_k.csv").open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows_k[0].keys()))
        w_.writeheader(); w_.writerows(rows_k)

    print("\n=== VERDICT ===")
    print("best state-Jacobian FD relative error: %.3e" % best_x0["relative_error"])
    print("best K E2E FD relative error: %.3e" % best_k["relative_error"])


if __name__ == "__main__":
    main()
