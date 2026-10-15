"""PHASE 17-R1O-OPT - state Jacobian, K composition, and end-to-end K chain
(s34-s39) for the PRODUCTION optical measurement, on the real campaign arc.

ANALYSIS SPACE ONLY.  Qualifies `lunar_od.lunar_landmark_optical`'s analytic
image-plane Jacobian and K_SRP sensitivity chain against independently
re-propagated trajectories, reusing frozen production dynamics.

FIDELITY NOTE inherited from R1O-D: every FD check below must resolve a
signal ABOVE its own harness floor.  Unlike DDOR (whose d(D_S)/dK ~ 3.5e-9
s/(m^2/kg) sat below a 90-s linear interpolant's truncation floor), the
optical observable's K sensitivity is of order PIXELS per (m^2/kg) at this
geometry, so the end-to-end FD here is expected to be well conditioned --
which the sweep below verifies rather than assumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phase17_r1m_core import build_range_arc, campaign_epoch  # noqa: E402
from phase17_r1o_opt_core import PREDECLARED_EPOCH_FRACTIONS  # noqa: E402

from lunar_od.constants import R_MOON_M  # noqa: E402
from lunar_od.lunar_landmark_optical import (  # noqa: E402
    DEFAULT_CAMERA, landmark_inertial_position_m,
    landmark_optical_state_and_k_sensitivity, nadir_pointing_camera_frame,
    pinhole_position_jacobian, pinhole_project,
)

GATES: dict[str, bool] = {}


def hdr(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def gate(name, ok, detail=""):
    GATES[name] = bool(ok)
    print("  %-52s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels
    import od_gravity_covariance_campaign as C  # noqa: N811

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    arc = build_range_arc(0.0, 15.0 * t_orbit, label="W15_for_optical_jacobian",
                          cadence_s=90.0)
    t_grid = arc.t_grid
    t_local = t_grid - t_grid[0]

    # ------------------------------------------------------------------
    hdr("s34 -- DIRECT MEASUREMENT K DEPENDENCE (structural, not assumed)")
    import inspect
    from lunar_od import lunar_landmark_optical as opt_mod

    proj_src = inspect.getsource(opt_mod.pinhole_project)
    jac_src = inspect.getsource(opt_mod.pinhole_position_jacobian)
    offending = [tok for tok in ("k_srp", "K_SRP", "srp", "SRP")
                 if tok in proj_src or tok in jac_src]
    gate("DIRECT_MEASUREMENT_K_DEPENDENCE_IS_NO", not offending,
         "no SRP/K token appears anywhere in the projection or its Jacobian; "
         "K can only enter through r_sc(t)")

    # ------------------------------------------------------------------
    hdr("s35 -- ANALYTIC IMAGE-PLANE JACOBIAN AT A REAL CAMPAIGN EPOCH")
    # Pick the pre-declared landmark nearest the boresight at the arc midpoint,
    # so the Jacobian check runs on a genuinely in-FOV observation.
    i_mid = len(t_grid) // 2
    r_sc = arc.nom48[i_mid, :3]
    et_mid = et0 + float(t_grid[i_mid])
    c_ci = nadir_pointing_camera_frame(r_sc)

    from phase17_r1o_core import _landmark_latlon_from_nadir
    latlon = _landmark_latlon_from_nadir(arc.nom48, t_grid, et0, PREDECLARED_EPOCH_FRACTIONS)
    best = None
    for li in range(latlon.shape[0]):
        r_lm = landmark_inertial_position_m(
            np.radians(latlon[li, 0]), np.radians(latlon[li, 1]), R_MOON_M, et_mid)
        p = pinhole_project(r_sc, r_lm, c_ci, DEFAULT_CAMERA)
        if p.in_front_of_camera and (best is None or p.off_boresight_rad < best[1]):
            best = (li, p.off_boresight_rad, r_lm, p)
    if best is None:
        raise SystemExit("No pre-declared landmark was in front of the camera at the midpoint.")
    li, off_bore, r_lm_mid, proj_mid = best
    print(f"  landmark #{li} lat={latlon[li, 0]:.4f} deg lon={latlon[li, 1]:.4f} deg")
    print(f"  range={proj_mid.range_m / 1e3:.3f} km, off-boresight={np.degrees(off_bore):.4f} deg")
    print(f"  (u, v) = ({proj_mid.u_px:.4f}, {proj_mid.v_px:.4f}) px, in FOV={proj_mid.within_fov}")

    dg_dr = pinhole_position_jacobian(r_sc, r_lm_mid, c_ci, DEFAULT_CAMERA)
    # order="F": the convention lunar_od.dynamics packs Phi with (see s40).
    h_x0, h_k = landmark_optical_state_and_k_sensitivity(
        dg_dr, arc.nom48[i_mid, 6:42].reshape((6, 6), order="F"), arc.nom48[i_mid, 42:48])
    print(f"  d(u,v)/dr      =\n{np.array2string(dg_dr, precision=9)}")
    print(f"  d(u,v)/dx0 row u = {np.array2string(h_x0[0], precision=6)}")
    print(f"  d(u,v)/dK        = {np.array2string(h_k, precision=9)}  px per (m^2/kg)")

    # ------------------------------------------------------------------
    hdr("s36/s38 -- END-TO-END STATE JACOBIAN FD (re-propagated trajectories)")
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    import spiceypy as spice

    x0_nominal = arc.x_true[0].copy()
    k0 = 0.01

    def sun_at(t):
        return spice.spkezr("SUN", et0 + float(t), "J2000", "NONE", "MOON")[0][:3] * 1000.0

    def earth_at(_t):
        return np.array([384_400e3, 0.0, 0.0])

    def propagate(x0, k, rtol=1e-12, atol=1e-13):
        return propagate_state_with_k_sensitivity(
            t_local, x0, C.MU, 0.0, 0.0, earth_at, sun_at,
            srp=SRPOptions(k_srp_m2_per_kg=k), rtol=rtol, atol=atol,
            j2_moon=J2_MOON_UNNORMALIZED,
        )

    def uv_from_history(hist):
        """(u, v) of the SAME landmark at the SAME epoch, with the camera
        attitude held at its NOMINAL (unperturbed) orientation.

        Holding C_ci fixed is not a convenience: it is the model this
        phase qualifies.  Attitude is an EXTERNAL input (star-tracker
        knowledge), not a function slaved to the instantaneous true
        position, so d(u,v)/dr must be evaluated at fixed attitude -- and
        the FD must therefore perturb only what the analytic Jacobian
        claims to differentiate.
        """
        r = hist[i_mid, :3]
        p = pinhole_project(r, r_lm_mid, c_ci, DEFAULT_CAMERA)
        return np.array([p.u_px, p.v_px])

    steps_m = np.array([1e0, 3e0, 1e1, 3e1, 1e2, 3e2])
    max_rel_err = 0.0
    print("  component | analytic d(u)/dx0  | best FD            | rel_err")
    for i, name in enumerate(("x", "y", "z", "vx", "vy", "vz")):
        scale = 1.0 if i < 3 else 1e-3
        fd_vals = []
        for h in steps_m * scale:
            dx0 = np.zeros(6)
            dx0[i] = h
            u_plus = uv_from_history(propagate(x0_nominal + dx0, k0))
            dx0[i] = -h
            u_minus = uv_from_history(propagate(x0_nominal + dx0, k0))
            fd_vals.append((u_plus - u_minus) / (2 * h))
        fd_vals = np.asarray(fd_vals)          # (n_steps, 2)
        analytic = h_x0[:, i]                   # (2,) both u and v rows
        errs = np.max(np.abs(fd_vals - analytic[None, :]), axis=1)
        j_best = int(np.argmin(errs))
        denom = max(float(np.max(np.abs(analytic))), 1e-30)
        rel_err = float(errs[j_best] / denom)
        max_rel_err = max(max_rel_err, rel_err)
        print("  %-9s | %+.9e | %+.9e | %.3e  (step %.3g)"
              % (name, analytic[0], fd_vals[j_best, 0], rel_err, (steps_m * scale)[j_best]))
    gate("OPTICAL_STATE_JACOBIAN_GATE", max_rel_err < 1e-4,
         "max rel err = %.3e" % max_rel_err)

    # ------------------------------------------------------------------
    hdr("s37 -- K COMPOSITION CHAIN (dg/dK = dg/dr . S_K[:3])")
    s_k_mid = arc.nom48[i_mid, 42:48]
    h_k_manual = dg_dr @ s_k_mid[:3]
    comp_rel = float(np.max(np.abs(h_k_manual - h_k)) / max(np.max(np.abs(h_k)), 1e-30))
    print(f"  S_K[:3] at midpoint = {np.array2string(s_k_mid[:3], precision=6)} m per (m^2/kg)")
    print(f"  composition dg/dK   = {np.array2string(h_k_manual, precision=9)}")
    gate("OPTICAL_K_COMPOSITION_SENSITIVITY_GATE", comp_rel < 1e-14,
         "composition vs module rel err = %.3e" % comp_rel)

    # ------------------------------------------------------------------
    hdr("s39 -- END-TO-END K SENSITIVITY FD (re-propagate with perturbed K)")
    # Sweep spans four decades so the noise-floor-limited and
    # truncation-limited regimes are both visible and the optimum is
    # MEASURED rather than assumed (Phase 17A-R floor consistency).
    dk_steps = np.array([1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]) * k0
    print("  expected FD signal at the smallest step: |dg/dK|*h = %.3e px"
          % (np.max(np.abs(h_k)) * dk_steps[0]))
    denom = max(float(np.max(np.abs(h_k))), 1e-30)

    # FLOOR CONSISTENCY (Phase 17A-R methodology, and the lesson R1O-D paid
    # for): before trusting -- or distrusting -- the agreement, establish what
    # limits the ORACLE.  Two diagnostics below:
    #   (a) propagate twice at effectively identical K.  The integrator is
    #       deterministic, so this isolates random noise from systematic
    #       truncation.
    #   (b) repeat the whole sweep at a TIGHTER integration tolerance.  If the
    #       residual is the oracle's own truncation error it must shrink with
    #       the tolerance; if it were a model error it would not move.
    noise_probe = float(np.max(np.abs(
        uv_from_history(propagate(x0_nominal, k0 * (1 + 1e-14)))
        - uv_from_history(propagate(x0_nominal, k0)))))
    print("  (a) two near-identical propagations differ by %.3e px"
          " -> the integrator is deterministic; any residual below is"
          " SYSTEMATIC truncation, not random noise" % noise_probe)

    results = {}
    for rtol, atol, tag in ((1e-12, 1e-13, "campaign default"),
                            (1e-13, 1e-15, "tightened")):
        e2e = []
        for h in dk_steps:
            u_plus = uv_from_history(propagate(x0_nominal, k0 + h, rtol, atol))
            u_minus = uv_from_history(propagate(x0_nominal, k0 - h, rtol, atol))
            e2e.append((u_plus - u_minus) / (2 * h))
        e2e = np.asarray(e2e)
        errs = np.max(np.abs(e2e - h_k[None, :]), axis=1)
        j_best = int(np.argmin(errs))
        results[tag] = (float(errs[j_best] / denom), dk_steps[j_best])
        print("  (b) rtol=%.0e (%s):" % (rtol, tag))
        for j, h in enumerate(dk_steps):
            print("        dK=%.3e  FD d(u)/dK=%+.9e  d(v)/dK=%+.9e  rel_err=%.3e"
                  % (h, e2e[j, 0], e2e[j, 1], errs[j] / denom))
        print("        best rel_err = %.3e at dK=%.3e"
              % (results[tag][0], results[tag][1]))

    print("  analytic            d(u)/dK=%+.9e  d(v)/dK=%+.9e" % (h_k[0], h_k[1]))
    default_rel, _ = results["campaign default"]
    tight_rel, tight_step = results["tightened"]
    improved = tight_rel < default_rel
    print("  residual shrinks with integrator tolerance (%.3e -> %.3e): %s"
          % (default_rel, tight_rel, "YES -> oracle-limited" if improved
             else "NO -> would indicate a genuine model discrepancy"))
    gate("OPTICAL_K_E2E_SENSITIVITY_GATE", tight_rel < 1e-4 and improved,
         "best rel err = %.3e at dK=%.3e (tightened integrator), and the"
         " residual tracks the oracle's own tolerance" % (tight_rel, tight_step))
    e2e_rel = tight_rel

    # ------------------------------------------------------------------
    hdr("SUMMARY")
    for k, v in GATES.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  ALL_JACOBIAN_GATES_PASS = {all(GATES.values())}")


if __name__ == "__main__":
    main()
