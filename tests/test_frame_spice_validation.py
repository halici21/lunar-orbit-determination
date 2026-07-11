"""Kernel-gated SPICE frame/state validation tests (frame audit, 2026-07).

Validates the production J2000<->ITRF93 contract against live SPICE:
rotation-block properties, the [r_F, 0] -> [r_I, v_I] station state pattern,
finite-difference station velocity, round trips, receive-vs-transmit epoch
mutation, the end-to-end LOS->SEZ->az/el chain, the frame-Jacobian chain,
range-norm invariance, the station-velocity range-rate contribution, and the
linear sxform interpolation used by the two-way path (diagnostic only).

Skips (with an explicit reason, no collection error) when ``spiceypy`` or the
DE421 kernel set is unavailable — same optional pattern as
``tests/test_real_grail_optional.py``. Production frame math is only called,
never modified; deliberate mutations are computed inside the tests.

Diagnostic numbers (FD sweep, interpolation errors) are printed so a
``pytest -s`` run records them for the frame-audit report.
"""

import math
import unittest

import numpy as np

from lunar_od.config import Station
from lunar_od.geometry import ecef2razel_sez, ecef2sez_dcm, wrap_to_pi
from lunar_od.measurements import (
    _position_measurement_jacobian_from_unit_los,
    _station_relative_state_j2000_at_receive_epoch,
)
from lunar_od.radiometrics import _interp_matrix, instantaneous_geometric_range_rate


def _spice_status() -> tuple[bool, str]:
    """Detect spiceypy + required kernels without raising at import time."""
    try:
        import spiceypy  # noqa: F401
    except Exception as exc:  # ImportError or CSPICE load failure
        return False, f"spiceypy unavailable: {exc}"
    try:
        from lunar_od.spice_loader import required_kernel_paths

        required_kernel_paths()
    except FileNotFoundError as exc:
        return False, f"SPICE kernels unavailable: {exc}"
    return True, ""


_SPICE_OK, _SKIP_REASON = _spice_status()

_ET_UTC = "2027-03-02 00:00:00"  # campaign epoch family; inside all kernel spans
_OMEGA_EARTH_RAD_S = 7.292115e-5

_ITU = Station(
    name="ITU Ayazaga",
    lat_deg=41.101,
    lon_deg=29.023,
    alt_m=100.0,
    color_rgb=(0.0, 0.447, 0.741),
    sigma_range_m=94.0,
    sigma_angle_rad=math.radians(0.005),
)


def setUpModule():  # noqa: N802 (unittest naming)
    if not _SPICE_OK:
        raise unittest.SkipTest(_SKIP_REASON)
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels()


def _et0() -> float:
    import spiceypy as spice

    return float(spice.str2et(_ET_UTC))


def _xform(et: float) -> np.ndarray:
    import spiceypy as spice

    return np.asarray(spice.sxform("J2000", "ITRF93", float(et)), dtype=float)


def _station_state_j2000(et: float, station: Station = _ITU) -> np.ndarray:
    """Call the production receive-epoch station-state transformation."""
    return _station_relative_state_j2000_at_receive_epoch(station, _xform(et))


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class SxformPropertyTests(unittest.TestCase):
    def test_rotation_block_orthogonality_determinant_and_structure(self):
        et0 = _et0()
        max_ortho = 0.0
        max_det = 0.0
        for dt in (0.0, 1.0, 10.0, 60.0, 3600.0, 86400.0):
            x = _xform(et0 + dt)
            c = x[:3, :3]
            max_ortho = max(max_ortho, float(np.max(np.abs(c.T @ c - np.eye(3)))))
            max_det = max(max_det, abs(float(np.linalg.det(c)) - 1.0))
            # Block structure X = [[C, 0], [Cdot, C]].
            np.testing.assert_allclose(x[3:, 3:], c, atol=1e-13)
            np.testing.assert_allclose(x[:3, 3:], np.zeros((3, 3)), atol=1e-15)
        print(f"\n[sxform] max |C^T C - I| = {max_ortho:.3e}, max |det C - 1| = {max_det:.3e}")
        self.assertLess(max_ortho, 1e-13)
        self.assertLess(max_det, 1e-13)

    def test_solve_pattern_matches_inverse_direction_sxform(self):
        import spiceypy as spice

        et = _et0()
        state_solve = _station_state_j2000(et)
        x_if = np.asarray(spice.sxform("ITRF93", "J2000", et), dtype=float)
        state_direct = x_if @ np.concatenate([_ITU.r_ecef_m, np.zeros(3)])
        pos_err = float(np.max(np.abs(state_solve[:3] - state_direct[:3])))
        vel_err = float(np.max(np.abs(state_solve[3:] - state_direct[3:])))
        print(f"[station state] solve-vs-sxform: pos {pos_err:.3e} m, vel {vel_err:.3e} m/s")
        self.assertLess(pos_err, 1e-6)
        self.assertLess(vel_err, 1e-9)

    def test_station_velocity_analytic_earth_rotation_sanity(self):
        state = _station_state_j2000(_et0())
        r_i, v_i = state[:3], state[3:]
        # Simplified rigid-rotation oracle: v ~ omega_E z_hat x r (ITRF93 spin
        # axis is within ~arcseconds of J2000 z over this era).
        v_oracle = np.cross(np.array([0.0, 0.0, _OMEGA_EARTH_RAD_S]), r_i)
        rel_mag = abs(np.linalg.norm(v_i) - np.linalg.norm(v_oracle)) / np.linalg.norm(v_oracle)
        cos_dir = float(v_i @ v_oracle / (np.linalg.norm(v_i) * np.linalg.norm(v_oracle)))
        print(f"[station velocity] |v| = {np.linalg.norm(v_i):.6f} m/s, "
              f"oracle rel mag err {rel_mag:.3e}, direction cos {cos_dir:.9f}")
        self.assertLess(rel_mag, 5e-3)
        self.assertGreater(cos_dir, 0.999)

    def test_station_velocity_matches_central_finite_difference(self):
        et = _et0()
        v_ref = _station_state_j2000(et)[3:]
        v_mag = float(np.linalg.norm(v_ref))
        errors = {}
        relative_errors = {}
        ratios = {}
        steps = (100.0, 30.0, 10.0, 3.0, 1.0, 0.1, 0.01, 0.001)
        print(f"[station velocity FD] |v_sxform| = {v_mag:.9f} m/s")
        print("[station velocity FD]  dt [s]   abs err [m/s]   rel err      ratio/previous")
        previous_error = None
        for dt in steps:
            r_plus = _station_state_j2000(et + dt)[:3]
            r_minus = _station_state_j2000(et - dt)[:3]
            v_fd = (r_plus - r_minus) / (2.0 * dt)
            errors[dt] = float(np.linalg.norm(v_fd - v_ref))
            relative_errors[dt] = errors[dt] / v_mag
            ratios[dt] = float("nan") if previous_error is None else errors[dt] / previous_error
            ratio_text = "n/a" if previous_error is None else f"{ratios[dt]:.6e}"
            print(
                f"[station velocity FD] {dt:7.3f}   {errors[dt]:.6e}   "
                f"{relative_errors[dt]:.6e}   {ratio_text}"
            )
            previous_error = errors[dt]

        # The sweep has a U-shaped truncation/round-off crossover. On this
        # kernel profile the empirical plateau is 10--1 s with its minimum at
        # 3 s. Sub-second differencing of transforms at ET~8.6e8 s is dominated
        # by finite-precision cancellation and must not set the acceptance.
        plateau_steps = (10.0, 3.0, 1.0)
        plateau_min = min(errors[step] for step in plateau_steps)
        plateau_tolerance_mps = 2.0e-5
        print(
            "[station velocity FD] selected plateau H={10,3,1} s: "
            f"min={plateau_min:.6e} m/s, acceptance={plateau_tolerance_mps:.6e} m/s"
        )

        # Coarse-step truncation decreases toward the crossover plateau.
        self.assertLess(errors[30.0], errors[100.0])
        self.assertLess(errors[10.0], errors[30.0])
        self.assertLess(errors[3.0], errors[10.0])
        self.assertLess(plateau_min, plateau_tolerance_mps)

        # Required diagnostic steps expose the round-off side of the sweep.
        self.assertGreater(errors[0.1], plateau_min)
        self.assertGreater(errors[0.01], errors[0.1])
        self.assertGreater(errors[0.001], errors[0.01])

    def test_round_trip_state_transform(self):
        import spiceypy as spice

        et = _et0()
        x_fi = _xform(et)
        x_if = np.asarray(spice.sxform("ITRF93", "J2000", et), dtype=float)
        state_i = _station_state_j2000(et) + np.array([1e5, -2e5, 3e5, 1.0, -2.0, 3.0])
        rt = x_if @ (x_fi @ state_i)
        pos_err = float(np.max(np.abs(rt[:3] - state_i[:3])))
        vel_err = float(np.max(np.abs(rt[3:] - state_i[3:])))
        print(f"[round trip] pos {pos_err:.3e} m, vel {vel_err:.3e} m/s")
        self.assertLess(pos_err, 1e-6)
        self.assertLess(vel_err, 1e-9)

    def test_from_to_reversal_mutation_is_detected(self):
        import spiceypy as spice

        et = _et0()
        r_i = _station_state_j2000(et)[:3]
        c_ok = _xform(et)[:3, :3]
        c_bad = np.asarray(spice.sxform("ITRF93", "J2000", et), dtype=float)[:3, :3]
        # Applying the reversed-direction matrix must move the result by a
        # macroscopic amount (Earth rotation is far from identity).
        diff = float(np.linalg.norm(c_ok @ r_i - c_bad @ r_i))
        self.assertGreater(diff, 1e3)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class LosChainTests(unittest.TestCase):
    def _lunar_los_inertial(self, et: float) -> tuple[np.ndarray, float]:
        """Station->Moon LOS in Earth-centered J2000 axes, and light time."""
        import spiceypy as spice

        moon_km, lt = spice.spkpos("MOON", et, "J2000", "NONE", "EARTH")
        r_moon = np.asarray(moon_km, dtype=float) * 1000.0
        r_station = _station_state_j2000(et)[:3]
        return r_moon - r_station, float(lt)

    def test_receive_vs_transmit_epoch_mutation(self):
        et = _et0()
        rho_i, light_time_s = self._lunar_los_inertial(et)
        et_tx = et - light_time_s
        c_rx = _xform(et)[:3, :3]
        c_tx = _xform(et_tx)[:3, :3]  # deliberate wrong epoch
        u_rx = c_rx @ (rho_i / np.linalg.norm(rho_i))
        u_tx = c_tx @ (rho_i / np.linalg.norm(rho_i))
        angle = math.acos(float(np.clip(u_rx @ u_tx, -1.0, 1.0)))
        expected_scale = _OMEGA_EARTH_RAD_S * light_time_s

        # Full observable-level report: SEZ direction and az/el differences.
        lat, lon = _ITU.lat_rad, _ITU.lon_rad
        c_sez = ecef2sez_dcm(lat, lon)
        sez_diff = float(np.linalg.norm(c_sez @ u_rx - c_sez @ u_tx))
        az_rx, el_rx, _ = ecef2razel_sez(c_rx @ rho_i, lat, lon)
        az_tx, el_tx, _ = ecef2razel_sez(c_tx @ rho_i, lat, lon)
        d_az = float(wrap_to_pi(az_tx - az_rx))
        d_el = float(el_tx - el_rx)
        arcsec = math.degrees(angle) * 3600.0
        print(f"[epoch mutation] receive ET {et:.6f} s, transmit ET {et_tx:.6f} s, "
              f"light time {light_time_s:.6f} s")
        print(f"[epoch mutation] LOS rotation {angle:.6e} rad = {arcsec:.3f} arcsec "
              f"(omega*tau = {expected_scale:.6e} rad)")
        print(f"[epoch mutation] |d u_SEZ| = {sez_diff:.6e}, "
              f"d az = {d_az:+.6e} rad, d el = {d_el:+.6e} rad")
        self.assertGreater(angle, 0.3 * expected_scale)
        self.assertLess(angle, 1.5 * expected_scale)
        # The mutation must be visible at the observable level too.
        self.assertGreater(max(abs(d_az), abs(d_el)), 0.1 * expected_scale)
        # Identity check: the correct epoch used twice is exactly consistent.
        self.assertAlmostEqual(float(u_rx @ u_rx), 1.0, places=12)

    def test_end_to_end_los_to_azimuth_elevation(self):
        et = _et0()
        lat, lon = _ITU.lat_rad, _ITU.lon_rad
        c_sez = ecef2sez_dcm(lat, lon)
        c_fi = _xform(et)[:3, :3]
        az_target, el_target = math.radians(135.0), math.radians(35.0)
        u_sez = np.array(
            [
                -math.cos(el_target) * math.cos(az_target),
                math.cos(el_target) * math.sin(az_target),
                math.sin(el_target),
            ]
        )
        rho = 4.0e8
        rho_i = c_fi.T @ (c_sez.T @ (rho * u_sez))  # invert the chain
        # Production direction: rotate at receive epoch, then SEZ observable.
        rho_ecef = c_fi @ rho_i
        az, el, rng = ecef2razel_sez(rho_ecef, lat, lon)
        print(f"[end-to-end] daz {abs(wrap_to_pi(az - az_target)):.3e} rad, "
              f"del {abs(el - el_target):.3e} rad, drange {abs(rng - rho):.3e} m")
        self.assertAlmostEqual(float(wrap_to_pi(az - az_target)), 0.0, places=10)
        self.assertAlmostEqual(el, el_target, places=10)
        self.assertAlmostEqual(rng, rho, delta=1e-4)

    def test_range_norm_invariance_through_chain(self):
        et = _et0()
        rho_i, _ = self._lunar_los_inertial(et)
        c_fi = _xform(et)[:3, :3]
        rho_f = c_fi @ rho_i
        rho_sez = ecef2sez_dcm(_ITU.lat_rad, _ITU.lon_rad) @ rho_f
        n_i = float(np.linalg.norm(rho_i))
        rel_f = abs(np.linalg.norm(rho_f) - n_i) / n_i
        rel_sez = abs(np.linalg.norm(rho_sez) - n_i) / n_i
        print(f"[norm invariance] |rho_F| rel {rel_f:.3e}, |rho_SEZ| rel {rel_sez:.3e}")
        self.assertLess(rel_f, 1e-13)
        self.assertLess(rel_sez, 1e-13)

    def test_frame_jacobian_matches_finite_difference_with_real_sxform(self):
        et = _et0()
        x_fi = _xform(et)
        c_sez = ecef2sez_dcm(_ITU.lat_rad, _ITU.lon_rad)
        rho_i, _ = self._lunar_los_inertial(et)
        rng = np.random.default_rng(7)
        j_los = rng.normal(size=(3, 6)) * np.array([1.0] * 3 + [500.0] * 3)
        range_m = float(np.linalg.norm(rho_i))
        u = rho_i / range_m
        j_unit = (np.eye(3) - np.outer(u, u)) @ j_los / range_m
        block = _position_measurement_jacobian_from_unit_los(
            u @ j_los, u, j_unit, _ITU, x_fi
        )

        def azel(rho_vec: np.ndarray) -> np.ndarray:
            u_sez = c_sez @ (x_fi[:3, :3] @ (rho_vec / np.linalg.norm(rho_vec)))
            az = math.atan2(u_sez[1], -u_sez[0])
            el = math.atan2(u_sez[2], math.hypot(u_sez[0], u_sez[1]))
            return np.array([az, el])

        max_rel = 0.0
        for col in range(6):
            # Step sized so the LOS moves ~1e-5 * range (beats cancellation,
            # keeps O(h^2) truncation negligible).
            col_gain = float(np.linalg.norm(j_los[:, col]))
            h = 1.0e-5 * range_m / max(col_gain, 1e-12)
            dx = np.zeros(6)
            dx[col] = h
            fd = wrap_to_pi(azel(rho_i + j_los @ dx) - azel(rho_i - j_los @ dx)) / (2.0 * h)
            for row, fd_val in ((1, fd[0]), (2, fd[1])):
                scale = max(abs(fd_val), 1e-12 / range_m)
                max_rel = max(max_rel, abs(block[row, col] - fd_val) / scale)
        print(f"[frame jacobian FD] max relative mismatch {max_rel:.3e}")
        self.assertLess(max_rel, 5e-6)

    def test_station_velocity_contribution_to_range_rate(self):
        import spiceypy as spice

        et = _et0()
        moon_state_km, _lt = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
        moon_state = np.asarray(moon_state_km, dtype=float) * 1000.0
        # Spacecraft ~100 km above the lunar surface with LLO-like velocity.
        sc_state_eci = moon_state + np.array([1.837e6, 0.0, 0.0, 0.0, 1.633e3, 0.0])
        station_state = _station_state_j2000(et)
        rho_i = sc_state_eci[:3] - station_state[:3]
        u = rho_i / np.linalg.norm(rho_i)
        rr_inertial = float(u @ (sc_state_eci[3:] - station_state[3:]))

        # Production formulation: full 6x6 sxform to ITRF93, fixed station.
        x_fi = _xform(et)
        state_ecef = x_fi @ sc_state_eci
        rho_ecef = state_ecef[:3] - _ITU.r_ecef_m
        rr_production = instantaneous_geometric_range_rate(rho_ecef, state_ecef[3:])
        frame_consistency = abs(rr_production - rr_inertial)

        # Diagnostic: dropping the station inertial velocity shifts range-rate
        # by exactly +u.v_station (repository sign: rr = u.(v_sc - v_st)).
        rr_no_station = float(u @ sc_state_eci[3:])
        delta = rr_no_station - rr_inertial
        expected = float(u @ station_state[3:])
        print(f"[station velocity in rr] production-vs-inertial {frame_consistency:.3e} m/s, "
              f"station contribution {expected:+.6f} m/s")
        self.assertLess(frame_consistency, 1e-8)
        self.assertAlmostEqual(delta, expected, delta=1e-9)


@unittest.skipUnless(_SPICE_OK, _SKIP_REASON)
class SxformInterpolationDiagnostics(unittest.TestCase):
    """Quantify the two-way path's linear 6x6 sxform interpolation (F item).

    Production behavior is intentionally unchanged; this records the error of
    ``radiometrics._interp_matrix`` (the helper the two-way chain uses)
    against exact ``spice.sxform`` so a future tolerance/bug-fix decision can
    cite measured numbers instead of estimates.
    """

    GRID_INTERVALS_S = (10.0, 30.0, 60.0)
    EVAL_FRACTIONS = (0.5, 0.25)  # midpoint and off-midpoint
    COUNT_INTERVAL_S = 60.0

    def test_interpolated_sxform_error_against_exact(self):
        import spiceypy as spice

        et0 = _et0()
        report_lines = []
        worst = {"pos": 0.0, "vel": 0.0, "ortho": 0.0, "det": 0.0, "rt": 0.0,
                 "range": 0.0, "rr": 0.0}
        moon_km, _lt = spice.spkpos("MOON", et0, "J2000", "NONE", "EARTH")
        r_target_i = np.asarray(moon_km, dtype=float) * 1000.0

        for grid_dt in self.GRID_INTERVALS_S:
            # Grid wide enough that the count-interval endpoints stay strictly
            # inside it (production two-way grids include a light-time margin;
            # extrapolation behavior is deliberately not measured here).
            span = 2.0 * self.COUNT_INTERVAL_S + 4.0 * grid_dt
            t_grid = np.arange(0.0, span + 0.5 * grid_dt, grid_dt)
            xforms = np.stack([_xform(et0 + t) for t in t_grid])
            mid_idx = t_grid.size // 2
            for frac in self.EVAL_FRACTIONS:
                t_eval = float(t_grid[mid_idx] + frac * grid_dt)
                self.assertGreater(t_eval - 0.5 * self.COUNT_INTERVAL_S, t_grid[0])
                self.assertLess(t_eval + 0.5 * self.COUNT_INTERVAL_S, t_grid[-1])
                x_lin = _interp_matrix(t_grid, xforms, t_eval)
                x_exact = _xform(et0 + t_eval)

                c_lin = x_lin[:3, :3]
                ortho = float(np.max(np.abs(c_lin.T @ c_lin - np.eye(3))))
                det_err = abs(float(np.linalg.det(c_lin)) - 1.0)

                fixed_state = np.concatenate([_ITU.r_ecef_m, np.zeros(3)])
                st_lin = np.linalg.solve(x_lin, fixed_state)
                st_exact = np.linalg.solve(x_exact, fixed_state)
                pos_err = float(np.linalg.norm(st_lin[:3] - st_exact[:3]))
                vel_err = float(np.linalg.norm(st_lin[3:] - st_exact[3:]))

                # Round trip through the exact inverse exposes the defect.
                rt_err = float(
                    np.linalg.norm(
                        (np.linalg.solve(x_exact, x_lin @ st_exact) - st_exact)[:3]
                    )
                )

                # Observable-level effect: one-way range to the Moon direction
                # and the equivalent averaged range-rate over a count interval.
                range_lin = float(np.linalg.norm(r_target_i - st_lin[:3]))
                range_exact = float(np.linalg.norm(r_target_i - st_exact[:3]))
                range_err = abs(range_lin - range_exact)
                # Two-sided count-interval difference (counted-Doppler analog).
                t_a = t_eval - 0.5 * self.COUNT_INTERVAL_S
                t_b = t_eval + 0.5 * self.COUNT_INTERVAL_S
                dr = []
                for t_ab in (t_a, t_b):
                    x_l = _interp_matrix(t_grid, xforms, t_ab)
                    x_e = _xform(et0 + t_ab)
                    p_l = np.linalg.solve(x_l, fixed_state)[:3]
                    p_e = np.linalg.solve(x_e, fixed_state)[:3]
                    dr.append(
                        float(np.linalg.norm(r_target_i - p_l))
                        - float(np.linalg.norm(r_target_i - p_e))
                    )
                rr_err = abs(dr[1] - dr[0]) / self.COUNT_INTERVAL_S

                worst["pos"] = max(worst["pos"], pos_err)
                worst["vel"] = max(worst["vel"], vel_err)
                worst["ortho"] = max(worst["ortho"], ortho)
                worst["det"] = max(worst["det"], det_err)
                worst["rt"] = max(worst["rt"], rt_err)
                worst["range"] = max(worst["range"], range_err)
                worst["rr"] = max(worst["rr"], rr_err)
                report_lines.append(
                    f"  grid {grid_dt:4.0f} s, frac {frac:4.2f}: ortho {ortho:.3e}, "
                    f"|det-1| {det_err:.3e}, pos {pos_err:.3e} m, vel {vel_err:.3e} m/s, "
                    f"round-trip {rt_err:.3e} m, range {range_err:.3e} m, "
                    f"rr({self.COUNT_INTERVAL_S:.0f}s) {rr_err:.3e} m/s"
                )

        print("\n[sxform linear interpolation vs exact]")
        for line in report_lines:
            print(line)
        print(
            f"  WORST: pos {worst['pos']:.3e} m, vel {worst['vel']:.3e} m/s, "
            f"range {worst['range']:.3e} m, rr {worst['rr']:.3e} m/s"
        )

        # Loose sanity bounds only (documented policy: measure first, decide
        # tolerances in a follow-up). Analytic scale: defect ~ (omega*dt)^2/8
        # -> ~15 m station position at a 60 s grid midpoint.
        self.assertLess(worst["ortho"], 1e-4)
        self.assertLess(worst["det"], 1e-4)
        self.assertLess(worst["pos"], 100.0)
        self.assertLess(worst["vel"], 0.05)
        self.assertLess(worst["range"], 100.0)


if __name__ == "__main__":
    unittest.main()
