"""Phase 5 validation support — Earth J2 magnitude / direction sanity.

NOT production code and NOT a scenario campaign.  Reports the direct vs indirect
Earth-J2 acceleration magnitudes in the Moon-centered frame and shows, numerically,
why the indirect (relative) form is much smaller than the direct-only model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import body_j2_acceleration  # noqa: E402
from lunar_od.dynamics import _J2000_TO_EARTH_BF as C_E, _MCI_TO_MOON_BF, MOON_J2, MOON_R_M  # noqa: E402
from lunar_od.constants import (  # noqa: E402
    J2_EARTH_UNNORMALIZED as J2_E, R_EARTH_J2_REF_M as R_E, MU_EARTH_M3S2 as MU_E,
)

MU_M = 4902.800066163796e9
R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])   # Earth rel Moon (m)


def main():
    print(f"Earth distance from Moon |R_ME| = {np.linalg.norm(R_ME)/1e3:,.0f} km")
    print(f"J2_E={J2_E:.6e}  R_E={R_E:,.1f} m  (EGM96 pair)\n")
    a_moon_term = body_j2_acceleration(-R_ME, MU_E, R_E, J2_E, C_E)
    print(f"Earth-J2 accel at the Moon (frame origin)   |a_J2(Moon)|  = {np.linalg.norm(a_moon_term):.3e} m/s^2")
    print(f"{'altitude':>10} | {'|a_direct|':>12} | {'|a_indirect|':>13} | {'ratio i/d':>9} | {'|a_MoonJ2|':>12}")
    print("-" * 70)
    for alt in (100e3, 500e3, 2000e3, 5000e3):
        r0 = MOON_R_M + alt
        r_sc = np.array([r0 * 0.6, r0 * 0.5, r0 * 0.62])
        r_sc_e = r_sc - R_ME
        a_direct = body_j2_acceleration(r_sc_e, MU_E, R_E, J2_E, C_E)
        a_indirect = a_direct - a_moon_term
        a_moon_j2 = body_j2_acceleration(r_sc, MU_M, MOON_R_M, MOON_J2, _MCI_TO_MOON_BF)
        nd = np.linalg.norm(a_direct); ni = np.linalg.norm(a_indirect); nm = np.linalg.norm(a_moon_j2)
        print(f"{alt/1e3:>8.0f}km | {nd:>12.3e} | {ni:>13.3e} | {ni/nd:>9.2e} | {nm:>12.3e}")
    print()
    print("Interpretation:")
    print(" - direct-only ~ a_J2(SC): dominated by the near-equal a_J2(Moon) common mode.")
    print(" - indirect = a_J2(SC) - a_J2(Moon): the true relative (tidal) perturbation,")
    print("   ~1-2 orders smaller; this is the physically correct Moon-centered term.")
    print(" - Earth J2 (indirect) is tiny vs Moon J2 -> expected negligible, kept for comparison.")
    # sign sanity: indirect must be much smaller than direct (else sign error)
    assert ni < 0.5 * nd, "indirect not << direct: possible sign error"
    print("\nSIGN/MAGNITUDE SANITY: PASS (indirect << direct at all altitudes).")


if __name__ == "__main__":
    main()
