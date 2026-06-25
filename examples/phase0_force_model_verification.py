"""Phase 0 - Existing Force Model Verification (blocks A-E, pure force model).

Measurement only: imports and calls existing production functions; modifies
nothing. Reports numerical equivalence/consistency for the current Moon-J2 +
third-body model across the Python and Numba paths.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od.dynamics import (  # noqa: E402
    f3body_moon, ode_fun_v3, dynamics_jacobian_a_matrix,
    point_mass_acceleration, third_body_acceleration,
    MOON_J2, MOON_R_M, _MCI_TO_MOON_BF,
)
from lunar_od.accelerated import f3body_rhs, ode42_rhs  # noqa: E402

# gm_de431 GM constants -> m^3/s^2
MU_MOON = 4902.800066163796e9
MU_EARTH = 398600.4354360959e9
MU_SUN = 132712440041.9393e9
# representative Earth / Sun position relative to Moon (m), from the SPICE fixture
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])


def _states():
    """Representative MCI 6-states: equatorial / polar / oblique, LLO + high."""
    out = []
    for alt in (100e3, 100e3, 100e3, 2000e3):
        r0 = 1737.4e3 + alt
        v0 = np.sqrt(MU_MOON / r0)
        out += [
            np.array([r0, 0, 0, 0, v0, 0.0]),          # equatorial
            np.array([0, 0, r0, v0, 0, 0.0]),          # polar (max J2 z-effect)
            np.array([r0*0.6, r0*0.5, r0*0.62, -0.4*v0, 0.7*v0, 0.3*v0]),  # oblique
        ]
    return out


def _a_py(state, j2):
    return f3body_moon(state, MU_MOON, MU_EARTH, MU_SUN, R_ME, R_MS, j2_moon=j2)[3:]


def _a_nb(state, j2):
    mr = MOON_R_M if j2 else 0.0
    return np.asarray(f3body_rhs(state, MU_MOON, MU_EARTH, MU_SUN, R_ME, R_MS, j2, mr, _MCI_TO_MOON_BF))[3:]


def block_A():
    print("\n=== A. Python vs Numba Moon-J2 ACCELERATION equivalence ===")
    for label, j2 in (("J2-OFF", 0.0), ("J2-ON", MOON_J2)):
        worst = 0.0
        for s in _states():
            ap, an = _a_py(s, j2), _a_nb(s, j2)
            rel = np.linalg.norm(ap - an) / max(np.linalg.norm(ap), 1e-30)
            worst = max(worst, rel)
        print(f"  {label}: max |Δa|/|a| = {worst:.3e}   (kabul < 1e-10)  -> {'PASS' if worst < 1e-10 else 'FAIL'}")


def block_B():
    print("\n=== B. Python vs Numba Moon-J2 GRAVITY-GRADIENT / STM (phi_dot) equivalence ===")
    PhiI = np.eye(6).reshape(-1, order="F")
    for label, j2 in (("J2-OFF", 0.0), ("J2-ON", MOON_J2)):
        worst_x = worst_phi = 0.0
        for s in _states():
            xaug = np.concatenate([s, PhiI])
            dpy = ode_fun_v3(0.0, xaug, MU_MOON, MU_EARTH, MU_SUN, lambda t: R_ME, lambda t: R_MS, j2)
            mr = MOON_R_M if j2 else 0.0
            dnb = np.asarray(ode42_rhs(xaug, MU_MOON, MU_EARTH, MU_SUN, R_ME, R_MS, j2, mr, _MCI_TO_MOON_BF))
            rx = np.linalg.norm(dpy[:6] - dnb[:6]) / max(np.linalg.norm(dpy[:6]), 1e-30)
            rp = np.linalg.norm(dpy[6:] - dnb[6:]) / max(np.linalg.norm(dpy[6:]), 1e-30)
            worst_x, worst_phi = max(worst_x, rx), max(worst_phi, rp)
        ok = worst_phi < 1e-9 and worst_x < 1e-9
        print(f"  {label}: max |Δx_dot|/|.|={worst_x:.3e}  max |Δphi_dot|/|.|={worst_phi:.3e}  (kabul < 1e-9) -> {'PASS' if ok else 'FAIL'}")


def block_C():
    print("\n=== C. Moon-J2 analytic gravity-gradient vs FINITE-DIFFERENCE ===")
    def a_of_r(r, j2):
        return _a_py(np.concatenate([r, np.zeros(3)]), j2)
    for label, j2 in (("J2-OFF", 0.0), ("J2-ON", MOON_J2)):
        worst = 0.0
        for s in _states():
            r = s[:3]
            A = dynamics_jacobian_a_matrix(s, MU_MOON, MU_EARTH, MU_SUN, R_ME, R_MS, j2_moon=j2)
            G = A[3:6, 0:3]
            h = 1.0
            G_fd = np.zeros((3, 3))
            for j in range(3):
                e = np.zeros(3); e[j] = h
                G_fd[:, j] = (a_of_r(r + e, j2) - a_of_r(r - e, j2)) / (2 * h)
            rel = np.linalg.norm(G_fd - G) / max(np.linalg.norm(G), 1e-30)
            worst = max(worst, rel)
        print(f"  {label}: max ‖G_FD − G‖/‖G‖ = {worst:.3e}   (kabul < 1e-4) -> {'PASS' if worst < 1e-4 else 'FAIL'}")


def block_D():
    print("\n=== D. J2-OFF regression: a(j2=0) == point_mass + third_body(Earth)+third_body(Sun) ===")
    worst = 0.0
    for s in _states():
        r = s[:3]
        a_ref = (point_mass_acceleration(r, MU_MOON)
                 + third_body_acceleration(r, R_ME, MU_EARTH)
                 + third_body_acceleration(r, R_MS, MU_SUN))
        a0 = _a_py(s, 0.0)
        worst = max(worst, np.linalg.norm(a0 - a_ref) / max(np.linalg.norm(a_ref), 1e-30))
    print(f"  max |Δ|/|a| = {worst:.3e}   (kabul < 1e-13, no-op dal) -> {'PASS' if worst < 1e-13 else 'FAIL'}")


def block_E():
    print("\n=== E. Third-body point-mass regression: function vs analytic indirect form ===")
    worst = 0.0
    for s in _states():
        r = s[:3]
        for r_tb, mu_tb in ((R_ME, MU_EARTH), (R_MS, MU_SUN)):
            d = r_tb - r
            ana = mu_tb * (d / np.linalg.norm(d)**3 - r_tb / np.linalg.norm(r_tb)**3)
            got = third_body_acceleration(r, r_tb, mu_tb)
            worst = max(worst, np.linalg.norm(got - ana) / max(np.linalg.norm(ana), 1e-30))
    print(f"  max |Δ|/|a| = {worst:.3e}   (kabul < 1e-13) -> {'PASS' if worst < 1e-13 else 'FAIL'}")


if __name__ == "__main__":
    print("Phase 0 — blocks A-E (pure force model). n_states =", len(_states()))
    block_A(); block_B(); block_C(); block_D(); block_E()
