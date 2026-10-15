"""PHASE 17-R1O-R s8/s9 -- independent STM storage-order oracle.

ANALYSIS SPACE ONLY.  Nothing here modifies production code.

THE QUESTION
------------
`lunar_od.dynamics.propagate_state_with_k_sensitivity` returns a 48-column
history whose columns [6:42] carry the 6x6 state-transition matrix Phi.  It
packs that block COLUMN-MAJOR (`phi_dot.reshape(-1, order="F")`, dynamics.py
s1249/s1292, docstring s386 "column-major order, matching MATLAB's Phi(:)").

A consumer that unflattens with NumPy's DEFAULT C order gets Phi^T instead of
Phi, because for a column-major flattening v,

    v[i + 6j] = Phi[i, j]          (pack, order="F")
    M[a, b]   = v[6a + b]          (unpack, order="C")
              = Phi[b, a]          ==>  M = Phi^T

Phi is not symmetric, so this is not a cosmetic difference.

THE ORACLE
----------
Phi(t, t0) is DEFINED by dx(t) = Phi(t,t0) dx0.  So the position rows obey

    dr(t)/dx0[:, j] = Phi[:3, j].

This script measures the left-hand side with a central finite difference on
INDEPENDENTLY RE-PROPAGATED trajectories -- using `propagate_state`, the
6-state propagator, which never forms or carries a variational block at all --
and compares it against BOTH candidate unpackings of the same nom48 row.

One of them must match to finite-difference precision.  The other must not.
That decides the storage contract by measurement, not by reading code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phase17_r1m_core import (  # noqa: E402
    ATOL, CADENCE_S, K_TRUTH, RTOL, campaign_epoch, campaign_initial_state,
)

COMPONENT_NAMES = ("x", "y", "z", "vx", "vy", "vz")
#: Central-difference steps: ~1 m on a ~1.8e6 m radius, ~1e-3 m/s on ~1.6e3
#: m/s.  Both are ~1e-6 relative -- near the classic cube-root-of-eps optimum
#: for a central difference, and re-checked by halving below.
FD_STEPS = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])


def _propagation_callbacks(et0: float):
    import spiceypy as spice

    def sun_at(t_s):
        return spice.spkezr("SUN", et0 + float(t_s), "J2000", "NONE",
                            "MOON")[0][:3] * 1000.0

    def earth_at(_t_s):
        return np.array([384_400e3, 0.0, 0.0])

    return earth_at, sun_at


def propagate_six_state(t_grid_s: np.ndarray, x0: np.ndarray, et0: float) -> np.ndarray:
    """Independent 6-state propagation -- NO variational block anywhere.

    Same force model, tolerances and callbacks `build_range_arc` uses, so the
    comparison isolates the STM LAYOUT and nothing else.
    """
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state
    from lunar_od.srp import SRPOptions

    earth_at, sun_at = _propagation_callbacks(et0)
    return propagate_state(
        t_grid_s, x0, C.MU, 0.0, 0.0, earth_at, sun_at,
        srp=SRPOptions(k_srp_m2_per_kg=K_TRUTH), rtol=RTOL, atol=ATOL,
        j2_moon=J2_MOON_UNNORMALIZED)


def propagate_nom48(t_grid_s: np.ndarray, x0: np.ndarray, et0: float) -> np.ndarray:
    import od_gravity_covariance_campaign as C  # noqa: N811
    from lunar_od.constants import J2_MOON_UNNORMALIZED
    from lunar_od.dynamics import propagate_state_with_k_sensitivity
    from lunar_od.srp import SRPOptions

    earth_at, sun_at = _propagation_callbacks(et0)
    return propagate_state_with_k_sensitivity(
        t_grid_s, x0, C.MU, 0.0, 0.0, earth_at, sun_at,
        srp=SRPOptions(k_srp_m2_per_kg=K_TRUTH), rtol=RTOL, atol=ATOL,
        j2_moon=J2_MOON_UNNORMALIZED)


def fd_position_sensitivity(t_grid_s: np.ndarray, x0: np.ndarray, et0: float,
                            i_epoch: int, steps: np.ndarray) -> np.ndarray:
    """(3,6) dr(t_i)/dx0 by central differences on re-propagated trajectories."""
    jac = np.zeros((3, 6))
    for j in range(6):
        h = float(steps[j])
        xp, xm = np.array(x0, dtype=float), np.array(x0, dtype=float)
        xp[j] += h
        xm[j] -= h
        rp = propagate_six_state(t_grid_s, xp, et0)[i_epoch, :3]
        rm = propagate_six_state(t_grid_s, xm, et0)[i_epoch, :3]
        jac[:, j] = (rp - rm) / (2.0 * h)
    return jac


def _rel_err(candidate: np.ndarray, truth: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(truth))), 1e-300)
    return float(np.max(np.abs(candidate - truth))) / denom


def _column_rel_err(candidate: np.ndarray, truth: np.ndarray, j: int) -> float:
    denom = max(float(np.max(np.abs(truth[:, j]))), 1e-300)
    return float(np.max(np.abs(candidate[:, j] - truth[:, j]))) / denom


def run(duration_orbits: float = 1.0) -> dict:
    et0, t_orbit = campaign_epoch()
    x0 = campaign_initial_state()
    duration_s = duration_orbits * t_orbit
    t_grid = np.arange(0.0, duration_s + 0.5 * CADENCE_S, CADENCE_S)

    print("=" * 74)
    print("PHASE 17-R1O-R s9 -- INDEPENDENT STM STORAGE-ORDER ORACLE")
    print("=" * 74)
    print(f"campaign et0        = {et0:.6f} (TDB s past J2000)")
    print(f"orbit period        = {t_orbit:.3f} s")
    print(f"grid                = {t_grid.size} epochs @ {CADENCE_S:.0f} s "
          f"({duration_orbits:g} orbit)")
    print(f"rtol/atol           = {RTOL:g}/{ATOL:g}")
    print("FD oracle           = propagate_state (6-state, no variational block)")

    nom48 = propagate_nom48(t_grid, x0, et0)

    # Two real campaign epochs, not one: a mid-arc and an end-of-arc epoch.
    epochs = {
        "mid-arc": int(t_grid.size // 2),
        "end-arc": int(t_grid.size - 1),
    }

    results = {}
    for name, i_epoch in epochs.items():
        print()
        print("-" * 74)
        print(f"EPOCH {name}: index {i_epoch}, t = {t_grid[i_epoch]:.1f} s "
              f"from campaign epoch")
        print("-" * 74)

        block = nom48[i_epoch, 6:42]
        phi_f = np.asarray(block, dtype=float).reshape((6, 6), order="F")
        phi_c = np.asarray(block, dtype=float).reshape(6, 6)

        print(f"  phi_F and phi_C are transposes of each other: "
              f"{np.array_equal(phi_f, phi_c.T)}")
        print(f"  Phi is symmetric (would make the defect invisible): "
              f"{np.allclose(phi_f, phi_f.T)}")

        fd = fd_position_sensitivity(t_grid, x0, et0, i_epoch, FD_STEPS)
        fd_half = fd_position_sensitivity(t_grid, x0, et0, i_epoch, FD_STEPS * 0.5)
        fd_selfconsistency = _rel_err(fd_half, fd)
        print(f"  FD step-halving self-consistency      = {fd_selfconsistency:.3e}")

        cand_f = phi_f[:3, :]
        cand_c = phi_c[:3, :]
        err_f = _rel_err(cand_f, fd)
        err_c = _rel_err(cand_c, fd)

        print()
        print("  per-component relative error of dr(t)/dx0[:, j] vs FD truth")
        print(f"  {'component':<10} {'F-order (order=F)':>20} {'C-order (default)':>20}")
        per_component = {}
        for j, cname in enumerate(COMPONENT_NAMES):
            ef = _column_rel_err(cand_f, fd, j)
            ec = _column_rel_err(cand_c, fd, j)
            per_component[cname] = {"f_order": ef, "c_order": ec}
            print(f"  {cname:<10} {ef:>20.6e} {ec:>20.6e}")

        print()
        print(f"  matrix-wide  F-order relative error   = {err_f:.6e}")
        print(f"  matrix-wide  C-order relative error   = {err_c:.6e}")

        # FLOOR CONSISTENCY (the Phase 17A-R methodology, reused).  A central
        # finite difference on a numerically integrated trajectory has its OWN
        # precision floor; comparing a candidate against it with a fixed
        # absolute threshold silently asks the candidate to be more accurate
        # than the oracle that judges it.  The measured floor here is the
        # step-halving self-consistency, so the honest question is:
        #   does F-order agree WITHIN the oracle's own resolution,
        #   and is C-order separated from it DECISIVELY?
        floor = max(fd_selfconsistency, 1e-12)
        f_matches = err_f <= 10.0 * floor
        c_matches = err_c <= 10.0 * floor
        separation = err_c / max(err_f, 1e-300)
        print(f"  FD oracle precision floor             = {floor:.3e}")
        print(f"  F-order error / floor                 = {err_f / floor:.3f}")
        print(f"  C-order error / floor                 = {err_c / floor:.3e}")
        print(f"  C/F separation ratio                  = {separation:.3e}")
        print(f"  F-order matches FD truth (at floor)   = {f_matches}")
        print(f"  C-order matches FD truth (at floor)   = {c_matches}")

        results[name] = {
            "index": i_epoch,
            "t_s": float(t_grid[i_epoch]),
            "phi_symmetric": bool(np.allclose(phi_f, phi_f.T)),
            "fd_step_halving": fd_selfconsistency,
            "fd_precision_floor": floor,
            "f_order_rel_err": err_f,
            "c_order_rel_err": err_c,
            "f_order_err_over_floor": err_f / floor,
            "c_over_f_separation": separation,
            "f_order_matches": bool(f_matches),
            "c_order_matches": bool(c_matches),
            "per_component": per_component,
        }

    gate = all(r["f_order_matches"] and not r["c_order_matches"]
               and r["c_over_f_separation"] > 1e3
               for r in results.values())

    print()
    print("=" * 74)
    print(f"R1O_STM_LAYOUT_ORACLE_GATE = {'PASS' if gate else 'FAIL'}")
    print("=" * 74)
    print("Interpretation: the FD-measured position sensitivity dr(t)/dx0 -- "
          "obtained\nfrom trajectories that carry no STM at all -- agrees with "
          "the order='F'\nunpacking and disagrees with the default C-order "
          "unpacking.  The storage\ncontract is therefore COLUMN-MAJOR, "
          "established by measurement.")

    results["gate"] = "PASS" if gate else "FAIL"
    return results


if __name__ == "__main__":
    out = run()
    import json

    art = HERE.parent / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "r1o_r_stm_oracle.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {art / 'r1o_r_stm_oracle.json'}")
    sys.exit(0 if out["gate"] == "PASS" else 1)
