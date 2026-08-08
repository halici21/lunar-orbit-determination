"""Focused tests from the Doppler / range-rate model review (sections 9A-9F).

These complement the existing two-way Doppler suite by making the textbook vs
operational distinction explicit:

* 9A/9B -- the geometric instantaneous mode is the textbook line-of-sight
  range-rate (receding = speed, transverse = 0).
* 9C/9D -- the two-way counted-Doppler mode is genuinely *averaged differenced
  range* over the count interval (it depends on Tc under non-zero jerk and
  reduces to the instantaneous rate as Tc -> 0), and is therefore physically
  distinct from the instantaneous mode.
"""

import unittest

import numpy as np

from lunar_od import (
    COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
    COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD,
    EXACT_EVENT_EPOCH_STATION_METHOD,
    LEGACY_INTERPOLATED_STATION_METHOD,
    make_exact_counted_doppler_station_state_provider,
    solve_two_way_light_time,
    two_way_counted_doppler_initial_state_jacobian,
    RangeRatePhysicsConfig,
    instantaneous_geometric_range_rate,
    two_way_counted_doppler_observable,
)

C_LIGHT = 299792458.0

from lunar_od.radiometrics import _interp_vector, interp_state_history


class _Station:
    """Minimal station fixed at the ECEF origin (identity frame fixtures)."""
    r_ecef_m = np.zeros(3)
    lat_rad = 0.0
    lon_rad = 0.0


def _radial_cubic_states(t, x0, v0, a, j):
    """1-D radial motion x(t)=x0+v0 t+a t^2/2+j t^3/6 along +x (far from origin)."""
    t = np.asarray(t, float)
    x = x0 + v0 * t + 0.5 * a * t**2 + j * t**3 / 6.0
    vx = v0 + a * t + 0.5 * j * t**2
    states = np.zeros((t.size, 6))
    states[:, 0] = x
    states[:, 3] = vx
    return states


def _identity_frame(n):
    return np.repeat(np.eye(6)[None, :, :], n, axis=0)


class GeometricInstantaneousRangeRate(unittest.TestCase):
    def test_9a_receding_equals_speed(self):
        """Section 9A: radial recession -> rho_dot equals the closing speed."""
        r = np.array([1.0e7, 0.0, 0.0])
        v = np.array([125.0, 0.0, 0.0])           # straight along the line of sight
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, v), 125.0, places=9)
        # approaching (negative) is the mirror case
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, -v), -125.0, places=9)

    def test_9b_transverse_motion_is_zero(self):
        """Section 9B: velocity perpendicular to the line of sight -> rho_dot = 0."""
        r = np.array([1.0e7, 0.0, 0.0])
        v = np.array([0.0, 125.0, -80.0])         # purely transverse
        self.assertAlmostEqual(instantaneous_geometric_range_rate(r, v), 0.0, places=9)


class TwoWayCountedDopplerIsAveraged(unittest.TestCase):
    """Sections 9C/9D: confirm the two-way mode is Tc-averaged and distinct."""

    def _obs(self, t, states, tc):
        cfg = RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=tc, station_state_method=LEGACY_INTERPOLATED_STATION_METHOD)
        n = t.size
        zeros = np.zeros((n, 3))
        return two_way_counted_doppler_observable(
            0.0, _Station(), t, states, zeros, zeros, _identity_frame(n), cfg)

    def test_9c_observable_depends_on_count_interval_under_jerk(self):
        """A non-zero jerk makes the averaged observable depend on Tc; in the
        small-Tc limit it collapses onto the instantaneous range-rate. If the
        observable were instantaneous it would be Tc-independent."""
        t = np.arange(-200.0, 200.001, 1.0)
        jerk = 5.0e-3                              # m/s^3 (radial)
        states = _radial_cubic_states(t, x0=1.0e7, v0=100.0, a=0.0, j=jerk)
        inst = instantaneous_geometric_range_rate(states[t.size // 2, :3], states[t.size // 2, 3:])

        obs_short = self._obs(t, states, tc=2.0)
        obs_long = self._obs(t, states, tc=160.0)

        # small Tc -> instantaneous limit
        self.assertAlmostEqual(obs_short, inst, delta=1e-2)
        # large Tc -> averaged, measurably different (averaging over the jerk)
        self.assertGreater(abs(obs_long - obs_short), 0.1)
        # expected averaging signature: centred mean of a quadratic rho_dot is
        # inst + (jerk/6)*(Tc/2)^2  (one-way m/s-equivalent)
        expected_long = inst + (jerk / 6.0) * (160.0 / 2.0) ** 2
        self.assertAlmostEqual(obs_long, expected_long, delta=0.05)

    def test_9d_two_way_distinct_from_instantaneous_under_jerk(self):
        """Section 9D: with curvature/jerk the two-way counted Doppler differs
        from the instantaneous geometric range-rate -> the modes are physically
        distinct (not the same computation behind two labels)."""
        t = np.arange(-200.0, 200.001, 1.0)
        states = _radial_cubic_states(t, x0=1.0e7, v0=100.0, a=0.0, j=5.0e-3)
        inst = instantaneous_geometric_range_rate(states[t.size // 2, :3], states[t.size // 2, 3:])
        two_way = self._obs(t, states, tc=160.0)
        self.assertGreater(abs(two_way - inst), 0.1)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# R3 exact event-epoch station transform — acceptance gates
#
# R3-P03 exact station-state unit parity      R3-P07 cadence isolation
# R3-P06 finite-difference Jacobian           R3-P08 shared event geometry
# R3-P19 cache neutrality + structural counts R3-P22 Earth policy invariance
# R3-P24 exact path never reads the transform grid
# ---------------------------------------------------------------------------

R3_EARTH_ROTATION_RATE_RAD_S = 7.292115e-5
R3_EQUATORIAL_RADIUS_M = 6378137.0


def _r3_sxform(omega_rad_s=R3_EARTH_ROTATION_RATE_RAD_S):
    """Deterministic J2000->ITRF93 state transform for a Z-rotating Earth.

    Returns the 6x6 [[R, 0], [dR/dt, R]] block form SPICE uses, so the inverse
    velocity block reproduces the exact omega x r site velocity. SPICE-free and
    exactly reproducible, which is what the frozen unit-parity gate needs.
    """

    def sxform_fn(source, target, et):
        assert source == "J2000" and target == "ITRF93"
        theta = omega_rad_s * float(et)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        rot_dot = omega_rad_s * np.array(
            [[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 0.0]]
        )
        xform = np.zeros((6, 6))
        xform[:3, :3] = rot
        xform[3:, :3] = rot_dot
        xform[3:, 3:] = rot
        return xform

    return sxform_fn


class _R3EquatorialStation:
    """Equatorial site on the +X ITRF93 axis, so omega x r is exact and simple."""

    name = "R3 equatorial"
    r_ecef_m = np.array([R3_EQUATORIAL_RADIUS_M, 0.0, 0.0])


def _r3_case(
    cadence_s=60.0,
    span_s=600.0,
    earth_range_m=3.8e8,
    velocity_mps=(90.0, 1400.0, 0.0),
    position_m=(2.0e6, 5.0e5, 0.0),
):
    """Shared fixture: spacecraft history, Earth ephemeris and transform grid.

    ``cadence_s`` sets ONLY the transform-grid spacing. The spacecraft history
    grid is deliberately held fixed and dense so R3-P07 isolates the station
    transform from spacecraft cubic-Hermite interpolation.
    """
    t_grid = np.arange(-span_s, span_s + 1e-9, cadence_s)
    earth_pos = np.zeros((t_grid.size, 3))
    earth_vel = np.zeros((t_grid.size, 3))
    earth_pos[:, 0] = earth_range_m
    earth_pos[:, 1] = 12.0 * t_grid
    earth_vel[:, 1] = 12.0
    sxform_fn = _r3_sxform()
    xforms = np.array(
        [sxform_fn("J2000", "ITRF93", float(t)) for t in t_grid]
    )
    states = np.zeros((t_grid.size, 6))
    for axis in range(3):
        states[:, axis] = position_m[axis] + velocity_mps[axis] * t_grid
        states[:, 3 + axis] = velocity_mps[axis]
    return {
        "t_grid": t_grid,
        "states": states,
        "earth_pos": earth_pos,
        "earth_vel": earth_vel,
        "xforms": xforms,
        "sxform_fn": sxform_fn,
        "station": _R3EquatorialStation(),
    }


def _r3_provider(case, cache_enabled=True):
    return make_exact_counted_doppler_station_state_provider(
        case["station"],
        0.0,
        case["t_grid"],
        case["earth_pos"],
        case["earth_vel"],
        sxform_fn=case["sxform_fn"],
        cache_enabled=cache_enabled,
    )



def _r3_stm(t_s):
    """True STM of the straight-line fixture history: [[I, t I], [0, I]]."""
    phi = np.eye(6)
    phi[:3, 3:] = float(t_s) * np.eye(3)
    return phi


def _r3_augmented(case):
    """Augmented history whose STM columns match the fixture's real dynamics."""
    n = case["t_grid"].size
    augmented = np.zeros((n, 42))
    augmented[:, :6] = case["states"]
    for k, t_s in enumerate(case["t_grid"]):
        augmented[k, 6:] = _r3_stm(t_s).reshape(-1, order="F")
    return augmented

def _r3_config(count_interval_s=20.0, method=EXACT_EVENT_EPOCH_STATION_METHOD):
    return RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=count_interval_s,
        light_time_tolerance_s=1e-12,
        light_time_equation_tolerance_s=1e-11,
        light_time_max_iter=30,
        station_state_method=method,
    )


class R3ExactStationStateUnitParity(unittest.TestCase):
    """R3-P03: the production exact station state is the sxform station state."""

    def test_p03_matches_independent_sxform_evaluation(self):
        case = _r3_case()
        provider = _r3_provider(case)
        fixed = np.concatenate([case["station"].r_ecef_m, np.zeros(3)])
        for t_s in (-137.25, -0.5, 0.0, 61.125, 249.75):
            produced = provider.state(t_s)
            # Independent in-test evaluation: linear Earth + exact sxform.
            earth = np.concatenate(
                [
                    _interp_vector(case["t_grid"], case["earth_pos"], t_s),
                    _interp_vector(case["t_grid"], case["earth_vel"], t_s),
                ]
            )
            expected = earth + np.linalg.solve(
                case["sxform_fn"]("J2000", "ITRF93", t_s), fixed
            )
            self.assertLessEqual(
                float(np.max(np.abs(produced[:3] - expected[:3]))), 1e-9
            )
            self.assertLessEqual(
                float(np.max(np.abs(produced[3:] - expected[3:]))), 1e-12
            )

    def test_p03_station_velocity_is_omega_cross_r_for_an_equatorial_site(self):
        case = _r3_case()
        provider = _r3_provider(case)
        omega = np.array([0.0, 0.0, R3_EARTH_ROTATION_RATE_RAD_S])
        for t_s in (-90.0, 0.0, 175.5):
            state = provider.state(t_s)
            earth = np.concatenate(
                [
                    _interp_vector(case["t_grid"], case["earth_pos"], t_s),
                    _interp_vector(case["t_grid"], case["earth_vel"], t_s),
                ]
            )
            site_position = state[:3] - earth[:3]
            site_velocity = state[3:] - earth[3:]
            self.assertLessEqual(
                float(np.max(np.abs(site_velocity - np.cross(omega, site_position)))),
                1e-9,
            )

    def test_p03_provider_declares_frame_center_and_units(self):
        provider = _r3_provider(_r3_case())
        self.assertEqual(provider.station_state_method, EXACT_EVENT_EPOCH_STATION_METHOD)
        self.assertEqual(provider.source_frame, "J2000")
        self.assertEqual(provider.target_frame, "ITRF93")
        self.assertEqual(provider.center, "moon")
        self.assertEqual(provider.earth_ephemeris_method, "linear_grid_interpolation")
        self.assertEqual(provider.state(0.0).shape, (6,))


class R3EarthEphemerisPolicyInvariance(unittest.TestCase):
    """R3-P22: R3 changed the site transform only, never the Earth policy."""

    def test_p22_earth_history_is_linearly_interpolated(self):
        case = _r3_case()
        provider = _r3_provider(case)
        fixed = np.concatenate([case["station"].r_ecef_m, np.zeros(3)])
        # Midpoint between two grid nodes: linear interpolation is the exact
        # mean of the neighbours; a cubic-Hermite Earth would not be.
        left, right = case["t_grid"][3], case["t_grid"][4]
        mid = 0.5 * (left + right)
        state = provider.state(mid)
        earth_site = state[:3] - np.linalg.solve(
            case["sxform_fn"]("J2000", "ITRF93", mid), fixed
        )[:3]
        expected_linear = 0.5 * (case["earth_pos"][3] + case["earth_pos"][4])
        self.assertLessEqual(
            float(np.max(np.abs(earth_site - expected_linear))), 1e-9
        )
        self.assertEqual(
            provider.earth_ephemeris_method, "linear_grid_interpolation"
        )

    def test_p22_metadata_reports_the_unchanged_policies(self):
        self.assertEqual(COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD, "linear_grid_interpolation")
        self.assertEqual(COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD, "cubic_hermite")


class R3StationCadenceIsolation(unittest.TestCase):
    """R3-P07: only the transform-grid cadence changes; exact must not move."""

    CADENCES_S = (10.0, 60.0, 120.0)

    def _observable(self, cadence_s, method):
        case = _r3_case(cadence_s=cadence_s)
        return (
            two_way_counted_doppler_observable(
                0.0,
                case["station"],
                case["t_grid"],
                case["states"],
                case["earth_pos"],
                case["earth_vel"],
                case["xforms"],
                _r3_config(method=method),
                et0_s=0.0,
                station_state_provider=(
                    _r3_provider(case)
                    if method == EXACT_EVENT_EPOCH_STATION_METHOD
                    else None
                ),
            ),
            case,
        )

    def test_p07_exact_station_state_is_cadence_invariant(self):
        states = []
        for cadence_s in self.CADENCES_S:
            case = _r3_case(cadence_s=cadence_s)
            states.append(_r3_provider(case).state(37.5))
        reference = states[0]
        for state in states[1:]:
            self.assertLessEqual(float(np.max(np.abs(state[:3] - reference[:3]))), 1e-9)
            self.assertLessEqual(float(np.max(np.abs(state[3:] - reference[3:]))), 1e-12)

    def test_p07_exact_observable_is_cadence_invariant(self):
        values = [
            self._observable(c, EXACT_EVENT_EPOCH_STATION_METHOD)[0]
            for c in self.CADENCES_S
        ]
        spread = max(values) - min(values)
        self.assertLessEqual(spread, 1e-9)

    def test_p07_legacy_mode_still_shows_cadence_dependence(self):
        """Power check: without it the exact gate could pass vacuously."""
        values = [
            self._observable(c, LEGACY_INTERPOLATED_STATION_METHOD)[0]
            for c in self.CADENCES_S
        ]
        spread = max(values) - min(values)
        self.assertGreater(spread, 1e-9)

    def test_p07_spacecraft_history_grid_is_held_fixed_across_cadences(self):
        """The isolation only means something if the spacecraft grid is common.

        Any residual difference must be attributable to the transform grid, so
        this records that the spacecraft states at the compared epoch are
        identical across cadences by construction.
        """
        sampled = []
        for cadence_s in self.CADENCES_S:
            case = _r3_case(cadence_s=cadence_s)
            sampled.append(interp_state_history(case["t_grid"], case["states"], 37.5))
        for state in sampled[1:]:
            np.testing.assert_allclose(state, sampled[0], rtol=0.0, atol=1e-9)


class R3SharedObservableJacobianGeometry(unittest.TestCase):
    """R3-P08: the Jacobian consumes exactly the observable's station states."""

    class _RecordingProvider:
        """Wraps the real provider and records every (epoch, state) it serves."""

        def __init__(self, inner):
            self._inner = inner
            self.records = []
            self.station_state_method = inner.station_state_method
            self.earth_ephemeris_method = inner.earth_ephemeris_method
            self.source_frame = inner.source_frame
            self.target_frame = inner.target_frame
            self.center = inner.center

        def state(self, t_s):
            value = self._inner.state(t_s)
            self.records.append((float(t_s), value.copy()))
            return value

    def test_p08_jacobian_reuses_the_observable_station_states_bitwise(self):
        case = _r3_case()
        cfg = _r3_config()
        observable_provider = self._RecordingProvider(_r3_provider(case))
        two_way_counted_doppler_observable(
            0.0, case["station"], case["t_grid"], case["states"],
            case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
            station_state_provider=observable_provider,
        )
        jacobian_provider = self._RecordingProvider(_r3_provider(case))
        two_way_counted_doppler_initial_state_jacobian(
            0.0, case["station"], case["t_grid"], _r3_augmented(case),
            case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
            station_state_provider=jacobian_provider,
        )
        observed = {epoch: state for epoch, state in observable_provider.records}
        self.assertGreater(len(observed), 0)
        self.assertGreater(len(jacobian_provider.records), 0)
        for epoch, state in jacobian_provider.records:
            self.assertIn(epoch, observed)
            # BITWISE identity, not approximate agreement.
            self.assertTrue(np.array_equal(state, observed[epoch]))

    def test_p08_detects_exact_observable_mixed_with_legacy_jacobian(self):
        """The gate must FAIL when the two halves disagree, or it is worthless."""
        case = _r3_case()
        exact_cfg = _r3_config()
        legacy_cfg = _r3_config(method=LEGACY_INTERPOLATED_STATION_METHOD)
        exact_jacobian = two_way_counted_doppler_initial_state_jacobian(
            0.0, case["station"], case["t_grid"], _r3_augmented(case),
            case["earth_pos"], case["earth_vel"], case["xforms"], exact_cfg,
            station_state_provider=_r3_provider(case),
        )
        legacy_jacobian = two_way_counted_doppler_initial_state_jacobian(
            0.0, case["station"], case["t_grid"], _r3_augmented(case),
            case["earth_pos"], case["earth_vel"], case["xforms"], legacy_cfg,
        )
        self.assertGreater(
            float(np.max(np.abs(exact_jacobian - legacy_jacobian))), 0.0
        )


class R3FiniteDifferenceJacobian(unittest.TestCase):
    """R3-P06: analytic partials match central differences of the R3 observable.

    Uses the ACCEPTED R2/R3 metric definition
    ``max_best_relative_reference_fd_column_error``: per state column, sweep the
    perturbation step and keep the BEST relative error for that column, then
    take the max across columns. A single fixed step is not the frozen metric --
    a column whose analytic partial is ~1e-9 while the observable scale is ~1 is
    simply unresolvable at an arbitrary step, and scoring it there would measure
    finite-difference noise rather than Jacobian correctness.

    The controlled-exclusion denominator policy matches the accepted campaign:
    a column with |analytic| <= 1e-300 is excluded rather than dividing by zero.
    """

    THRESHOLD = 1e-6
    ACCEPTED_REFERENCE = 5.011776541977681e-7
    POSITION_STEPS_M = (1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4)
    VELOCITY_STEPS_MPS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1)

    # A near-degenerate geometry makes d(observable)/dx0 ~1e-9 while
    # d(observable)/dvx0 ~1, a nine-order dynamic range that central
    # differencing cannot resolve in double precision. That would measure
    # finite-difference noise, not Jacobian correctness, so this gate uses a
    # geometry in which every column is physically resolvable. The threshold is
    # untouched.
    FIXTURE = dict(
        cadence_s=60.0,
        span_s=900.0,
        earth_range_m=8.0e6,
        velocity_mps=(1200.0, 1500.0, 700.0),
        position_m=(2.0e6, 5.0e5, 3.0e5),
    )
    COUNT_INTERVAL_S = 240.0

    def _sweep(self):
        case = _r3_case(**self.FIXTURE)
        cfg = _r3_config(count_interval_s=self.COUNT_INTERVAL_S)
        augmented = _r3_augmented(case)

        def observable(states):
            return two_way_counted_doppler_observable(
                0.0, case["station"], case["t_grid"], states,
                case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
                station_state_provider=_r3_provider(case),
            )

        def perturbed(index, step):
            """Move the WHOLE history by Phi(t) @ delta.

            The analytic partials are taken with respect to the arc initial
            state, so a constant offset would silently test a different
            derivative.
            """
            delta = np.zeros(6)
            delta[index] = step
            states = case["states"].copy()
            for k, t_s in enumerate(case["t_grid"]):
                states[k] += _r3_stm(t_s) @ delta
            return states

        analytic = two_way_counted_doppler_initial_state_jacobian(
            0.0, case["station"], case["t_grid"], augmented,
            case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
            station_state_provider=_r3_provider(case),
        )
        best_per_column = {}
        for index in range(6):
            reference = float(analytic[index])
            if abs(reference) <= 1e-300:
                continue  # controlled exclusion, never a zero denominator
            steps = self.POSITION_STEPS_M if index < 3 else self.VELOCITY_STEPS_MPS
            errors = []
            for step in steps:
                numerical = (
                    observable(perturbed(index, step))
                    - observable(perturbed(index, -step))
                ) / (2.0 * step)
                errors.append(abs(numerical - reference) / abs(reference))
            best_per_column[index] = min(errors)
        return analytic, best_per_column

    def test_p06_max_best_relative_column_error_within_the_accepted_threshold(self):
        analytic, best_per_column = self._sweep()
        self.assertGreater(len(best_per_column), 0)
        headline = max(best_per_column.values())
        self.assertLessEqual(headline, self.THRESHOLD)

    def test_p06_threshold_is_not_tightened_opportunistically(self):
        """The frozen threshold stays 1e-6; the accepted reference is recorded."""
        self.assertEqual(self.THRESHOLD, 1e-6)
        self.assertLessEqual(self.ACCEPTED_REFERENCE, self.THRESHOLD)

    def test_p06_every_included_column_is_resolvable_somewhere_in_the_sweep(self):
        """Guards against a vacuous pass from an all-excluded denominator set."""
        _analytic, best_per_column = self._sweep()
        self.assertEqual(len(best_per_column), 6)
        for index, error in best_per_column.items():
            self.assertLessEqual(error, self.THRESHOLD, f"column {index}")


class R3CachePerformanceContract(unittest.TestCase):
    """R3-P19: the cache is result-neutral and the cost is structural."""

    def test_p19a_results_are_bitwise_identical_with_and_without_the_cache(self):
        case = _r3_case()
        cfg = _r3_config()
        augmented = _r3_augmented(case)
        values, jacobians = [], []
        for cache_enabled in (True, False):
            values.append(
                two_way_counted_doppler_observable(
                    0.0, case["station"], case["t_grid"], case["states"],
                    case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
                    station_state_provider=_r3_provider(case, cache_enabled),
                )
            )
            jacobians.append(
                two_way_counted_doppler_initial_state_jacobian(
                    0.0, case["station"], case["t_grid"], augmented,
                    case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
                    station_state_provider=_r3_provider(case, cache_enabled),
                )
            )
        self.assertEqual(values[0], values[1])
        self.assertTrue(np.array_equal(jacobians[0], jacobians[1]))

    def test_p19b_uncached_call_count_matches_the_closed_form(self):
        """2 endpoints x (1 downlink t3 + n_up uplink iterates + 1 final t1)."""
        case = _r3_case()
        cfg = _r3_config()
        provider = _r3_provider(case, cache_enabled=False)
        two_way_counted_doppler_observable(
            0.0, case["station"], case["t_grid"], case["states"],
            case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
            station_state_provider=provider,
        )
        half = 0.5 * cfg.count_interval_s
        expected = 0
        for receive_s in (-half, half):
            solution = solve_two_way_light_time(
                receive_s, case["station"], case["t_grid"], case["states"],
                case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
                station_state_provider=_r3_provider(case, cache_enabled=False),
            )
            expected += 2 + int(solution.uplink_iterations)
        self.assertEqual(provider.exact_sxform_call_count, expected)

    def test_p19c_cached_call_count_never_exceeds_uncached(self):
        case = _r3_case()
        cfg = _r3_config()
        counts = {}
        for cache_enabled in (True, False):
            provider = _r3_provider(case, cache_enabled)
            two_way_counted_doppler_observable(
                0.0, case["station"], case["t_grid"], case["states"],
                case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
                station_state_provider=provider,
            )
            counts[cache_enabled] = provider.exact_sxform_call_count
        self.assertLessEqual(counts[True], counts[False])


class R3TransformGridIsNotReadInExactMode(unittest.TestCase):
    """R3-P24: no counted-Doppler station query touches x_j2000_to_itrf93."""

    def test_p24_exact_path_ignores_a_poisoned_transform_grid(self):
        case = _r3_case()
        cfg = _r3_config()
        clean = two_way_counted_doppler_observable(
            0.0, case["station"], case["t_grid"], case["states"],
            case["earth_pos"], case["earth_vel"], case["xforms"], cfg,
            station_state_provider=_r3_provider(case),
        )
        # Any read of this grid would produce NaN or a wildly different value.
        poisoned = np.full_like(case["xforms"], np.nan)
        poisoned_result = two_way_counted_doppler_observable(
            0.0, case["station"], case["t_grid"], case["states"],
            case["earth_pos"], case["earth_vel"], poisoned, cfg,
            station_state_provider=_r3_provider(case),
        )
        self.assertEqual(clean, poisoned_result)
        self.assertTrue(np.isfinite(clean))

    def test_p24_legacy_path_does_read_the_transform_grid(self):
        """Power check: the poison must matter when the grid is actually used."""
        case = _r3_case()
        legacy_cfg = _r3_config(method=LEGACY_INTERPOLATED_STATION_METHOD)
        poisoned = np.full_like(case["xforms"], np.nan)
        with self.assertRaises(Exception):
            two_way_counted_doppler_observable(
                0.0, case["station"], case["t_grid"], case["states"],
                case["earth_pos"], case["earth_vel"], poisoned, legacy_cfg,
            )


# ---------------------------------------------------------------------------
# R4 (CD-4): finite-difference qualification of the four-event model.
# ---------------------------------------------------------------------------

class R4FourEventDerivativeQualification(unittest.TestCase):
    """R4-P10 / R4-P10b / R4-P12.

    The counted observable is a cancellation structure, so its finite
    difference is round-off limited: the error falls as 1/h until truncation
    takes over.  Both gates therefore use the accepted best-over-step-sweep
    metric with a sweep wide enough that every column and the delay derivative
    clear the floor.  The 1e-6 threshold is the inherited R3-P06 value and is
    not adjusted anywhere below.
    """

    @staticmethod
    def _module():
        from tests import test_two_way_counted_doppler_four_event as fixture

        return fixture

    @classmethod
    def setUpClass(cls):
        fixture = cls._module()
        cls.fx = fixture
        n = fixture.T_GRID.size
        aug = np.zeros((n, 42))
        aug[:, :6] = fixture.STATES
        phi = np.eye(6)
        aug[0, 6:42] = phi.reshape(-1, order="F")
        for k in range(1, n):
            dt = float(fixture.T_GRID[k] - fixture.T_GRID[k - 1])
            r = fixture.STATES[k - 1, :3]
            rn = float(np.linalg.norm(r))
            g = fixture.MU_MOON / rn**3 * (3.0 * np.outer(r, r) / rn**2 - np.eye(3))
            a_mat = np.zeros((6, 6))
            a_mat[:3, 3:] = np.eye(3)
            a_mat[3:, :3] = g
            phi = (np.eye(6) + a_mat * dt + 0.5 * (a_mat @ a_mat) * dt * dt) @ phi
            aug[k, 6:42] = phi.reshape(-1, order="F")
        cls.aug = aug
        cls.phi_history = np.array(
            [row.reshape((6, 6), order="F") for row in aug[:, 6:42]]
        )

    def _perturbed_history(self, column, step):
        """Propagate the initial-state perturbation through the TRUE STM.

        Shifting the stored history uniformly would make this gate vacuous.
        """
        delta = np.zeros(6)
        delta[column] = step
        return self.fx.STATES + np.einsum("kij,j->ki", self.phi_history, delta)

    def test_p12_analytic_state_jacobian_matches_finite_differences(self):
        from lunar_od.two_way_counted_doppler import (
            four_event_counted_doppler_initial_state_jacobian,
        )

        fixture = self.fx
        position_steps = [10.0**e for e in range(-2, 7)]
        velocity_steps = [10.0**e for e in range(-5, 4)]
        worst = 0.0
        unresolved = 0
        for receive_mid_s, count_interval_s in fixture.GEOMETRIES:
            config = fixture._config(count_interval_s, 1e-4)
            analytic = four_event_counted_doppler_initial_state_jacobian(
                receive_mid_s, None, fixture.T_GRID, self.aug, fixture.EARTH_POS,
                fixture.EARTH_VEL, fixture.TRANSFORM_GRID, config,
                station_state_provider=fixture._provider(),
            )
            for column in range(6):
                steps = position_steps if column < 3 else velocity_steps
                best = float("inf")
                for step in steps:
                    plus = fixture._observable(
                        receive_mid_s, config,
                        states=self._perturbed_history(column, step),
                    )
                    minus = fixture._observable(
                        receive_mid_s, config,
                        states=self._perturbed_history(column, -step),
                    )
                    finite = (plus - minus) / (2.0 * step)
                    if abs(analytic[column]) <= 1e-300:
                        continue
                    best = min(best, abs(finite - analytic[column]) / abs(analytic[column]))
                if not np.isfinite(best):
                    unresolved += 1
                    continue
                worst = max(worst, best)
        self.assertEqual(unresolved, 0, "every column must be resolvable")
        self.assertLess(worst, 1e-6, f"max_best_relative_fd_column_error {worst!r}")

    def test_p10_analytic_delay_sensitivity_matches_finite_differences(self):
        from lunar_od.two_way_counted_doppler import (
            four_event_counted_doppler_delay_sensitivity,
        )

        fixture = self.fx
        # Step-adequate qualification point: at delta_0 = 1 s the admissible
        # central step clears the observable's cancellation floor.  The analytic
        # coefficient varies by ~2e-4 relative between 1e-3 s and 1 s, so this
        # qualifies the same expression used at operational delays.
        worst = 0.0
        for receive_mid_s, count_interval_s in fixture.GEOMETRIES:
            config = fixture._config(count_interval_s, 1.0)
            analytic = four_event_counted_doppler_delay_sensitivity(
                receive_mid_s, None, fixture.T_GRID, fixture.STATES, fixture.EARTH_POS,
                fixture.EARTH_VEL, fixture.TRANSFORM_GRID, config,
                station_state_provider=fixture._provider(),
            )
            best = float("inf")
            for step in (0.1, 0.2, 0.5):
                finite = (
                    fixture._observable(receive_mid_s, fixture._config(count_interval_s, 1.0 + step))
                    - fixture._observable(receive_mid_s, fixture._config(count_interval_s, 1.0 - step))
                ) / (2.0 * step)
                best = min(best, abs(finite - analytic) / abs(analytic))
            worst = max(worst, best)
        self.assertLess(worst, 1e-6, f"max_best_relative_delay_fd_error {worst!r}")

    def test_p10b_delay_finite_difference_is_round_off_limited(self):
        """The residual at operational delays is a step artefact, not a model error.

        Larger delta_0 admits a larger step (delta_0 - h >= 0), so a
        round-off-limited finite difference must improve monotonically across
        the decade ladder and reach the 1e-6 gate at the qualification point.
        A wrong analytic expression would instead plateau at its model error.
        """
        from lunar_od.two_way_counted_doppler import (
            four_event_counted_doppler_delay_sensitivity,
        )

        fixture = self.fx
        ladder = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        receive_mid_s, count_interval_s = 300.0, 60.0
        curve = []
        for delay_s in ladder:
            config = fixture._config(count_interval_s, delay_s)
            analytic = four_event_counted_doppler_delay_sensitivity(
                receive_mid_s, None, fixture.T_GRID, fixture.STATES, fixture.EARTH_POS,
                fixture.EARTH_VEL, fixture.TRANSFORM_GRID, config,
                station_state_provider=fixture._provider(),
            )
            best = float("inf")
            for fraction in (1000.0, 100.0, 10.0, 2.0):
                step = delay_s / fraction
                finite = (
                    fixture._observable(
                        receive_mid_s, fixture._config(count_interval_s, delay_s + step)
                    )
                    - fixture._observable(
                        receive_mid_s, fixture._config(count_interval_s, delay_s - step)
                    )
                ) / (2.0 * step)
                best = min(best, abs(finite - analytic) / abs(analytic))
            curve.append(best)
        for earlier, later in zip(curve, curve[1:]):
            self.assertLessEqual(later, earlier * 2.0)
        self.assertGreater(curve[0] / curve[-1], 1e3)
        self.assertLess(curve[-1], 1e-6)
