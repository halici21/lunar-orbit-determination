"""R4 (CD-4) four-event counted Doppler with a constant transponder delay.

Covers the frozen acceptance rows R4-P01..P11, P14..P19 and P22.  The FD-based
rows (P10, P10b, P12) live in ``test_doppler_model_review.py`` next to the
existing R3 derivative gates.

Fixtures are deliberately SPICE-free: the station state is an analytic
Earth-rotation model injected through the accepted R3
``CountedDopplerStationStateProvider``, which is exactly the composition the R4
architecture requires (R3's linear-Earth provider, never M3's cubic-Hermite
provider factory).
"""

import math
import pathlib
import unittest

import numpy as np
from scipy.optimize import brentq, root

import lunar_od.radiometrics as radiometrics
from lunar_od.radiometrics import (
    COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
    COUNTED_DOPPLER_MODEL_VERSION,
    FOUR_EVENT_COUNTED_DOPPLER_MODEL_VERSION,
    CountedDopplerStationStateProvider,
    RangeRatePhysicsConfig,
    _interp_state,
    two_way_counted_doppler_initial_state_jacobian,
    two_way_counted_doppler_observable,
)
from lunar_od.two_way_counted_doppler import (
    FourEventCountedDopplerError,
    FourEventStationStateAdapter,
    four_event_counted_doppler_delay_sensitivity,
    four_event_counted_doppler_endpoints,
    four_event_counted_doppler_initial_state_jacobian,
    four_event_counted_doppler_measurement_metadata,
    four_event_counted_doppler_observable,
)
from lunar_od.two_way_counted_doppler_reference import (
    CountedDopplerReferenceConfig,
    ExactEventStationStateProvider,
    exact_station_four_event_counted_doppler_reference,
    exact_station_single_bounce_counted_doppler_reference,
    reference_config_with_delay,
)

C_LIGHT = 299792458.0
MU_MOON = 4.9028e12
R_EARTH_MOON = 3.84400e8
OMEGA_EARTH_MOON = 2.0 * math.pi / (27.321661 * 86400.0)
R_EARTH = 6.371e6
OMEGA_EARTH = 7.2921159e-5
STATION_LAT = math.radians(35.0)
STATION_LON0 = math.radians(20.0)

# Frozen R2 hypothetical delay sweep, recovered from repository authority
# (examples/r2_measurement_fidelity_validation.py FROZEN_TRANSPONDER_DELAYS_S).
FROZEN_DELAYS_S = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
GEOMETRIES = ((300.0, 10.0), (300.0, 60.0), (700.0, 60.0), (700.0, 30.0), (500.0, 100.0))

# --- Owner Addendum 06: independent numerical-accuracy oracle ----------------
# Structurally independent of production: it solves the LOCAL light-time
# unknowns with a scalar root finder and never calls a production event solver
# or the R2 reference. Validated against a 60-digit Decimal formulation
# (agreement < 1e-14 s) in the Phase-1A oracle-validation artifact.
NOMINAL_ROUND_TRIP_LIGHT_TIME_S = 2.56
# Accumulated binary64 rounding of the repaired arithmetic, in ulp(rho):
# two range/c divisions + local light-time representation + two additions
# assembling the round-trip light time + the endpoint difference.
INDEPENDENT_ORACLE_K = 5


def _independent_local_round_trip_light_time(t3: float) -> float:
    """Round-trip light time from a scalar root solve on local unknowns."""
    station_rx = _station_state(t3)

    def downlink_residual(tau):
        sc = _interp_state(T_GRID, STATES, t3 - tau)
        return tau - float(np.linalg.norm(sc[:3] - station_rx[:3])) / C_LIGHT

    tau_down = brentq(downlink_residual, 1.0, 4.0, xtol=1e-15, rtol=8.9e-16,
                      maxiter=200)
    t2 = t3 - tau_down
    sc_t2 = _interp_state(T_GRID, STATES, t2)

    def uplink_residual(tau):
        g1 = _station_state(t2 - tau)
        return tau - float(np.linalg.norm(sc_t2[:3] - g1[:3])) / C_LIGHT

    tau_up = brentq(uplink_residual, 1.0, 4.0, xtol=1e-15, rtol=8.9e-16, maxiter=200)
    return tau_down + tau_up


def _independent_observable(receive_mid_s: float, count_interval_s: float) -> float:
    """Zero-delay counted-Doppler observable from the independent oracle."""
    half = 0.5 * count_interval_s
    rho_start = _independent_local_round_trip_light_time(receive_mid_s - half)
    rho_end = _independent_local_round_trip_light_time(receive_mid_s + half)
    return C_LIGHT * ((rho_end - rho_start) / count_interval_s) / 2.0



def _station_state(t_s: float) -> np.ndarray:
    """Analytic Moon-centred inertial station state (deterministic, no SPICE)."""
    t = float(t_s)
    ce, se = math.cos(OMEGA_EARTH_MOON * t), math.sin(OMEGA_EARTH_MOON * t)
    earth_pos = np.array([R_EARTH_MOON * ce, R_EARTH_MOON * se, 0.0])
    earth_vel = np.array(
        [-R_EARTH_MOON * OMEGA_EARTH_MOON * se, R_EARTH_MOON * OMEGA_EARTH_MOON * ce, 0.0]
    )
    ang = OMEGA_EARTH * t + STATION_LON0
    cos_lat = math.cos(STATION_LAT)
    site_pos = np.array(
        [
            R_EARTH * cos_lat * math.cos(ang),
            R_EARTH * cos_lat * math.sin(ang),
            R_EARTH * math.sin(STATION_LAT),
        ]
    )
    site_vel = np.array(
        [
            -R_EARTH * cos_lat * OMEGA_EARTH * math.sin(ang),
            R_EARTH * cos_lat * OMEGA_EARTH * math.cos(ang),
            0.0,
        ]
    )
    return np.hstack([earth_pos + site_pos, earth_vel + site_vel])


def _spacecraft_history(t_grid, a_m=1.938e6, inclination=math.radians(85.0)):
    mean_motion = math.sqrt(MU_MOON / a_m**3)
    speed = math.sqrt(MU_MOON / a_m)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    states = np.empty((t_grid.size, 6))
    for index, t_s in enumerate(t_grid):
        cos_u, sin_u = math.cos(mean_motion * t_s), math.sin(mean_motion * t_s)
        states[index, :3] = (a_m * cos_u, a_m * sin_u * cos_i, a_m * sin_u * sin_i)
        states[index, 3:] = (-speed * sin_u, speed * cos_u * cos_i, speed * cos_u * sin_i)
    return states


T_GRID = np.arange(-200.0, 1200.0 + 1e-9, 0.5)
STATES = _spacecraft_history(T_GRID)
EARTH_POS = np.zeros((T_GRID.size, 3))
EARTH_VEL = np.zeros((T_GRID.size, 3))
TRANSFORM_GRID = np.zeros((T_GRID.size, 6, 6))


def _provider(cache_enabled: bool = True) -> CountedDopplerStationStateProvider:
    return CountedDopplerStationStateProvider(
        state_fn=_station_state, cache_enabled=cache_enabled
    )


def _reference_provider() -> ExactEventStationStateProvider:
    return ExactEventStationStateProvider(state_fn=_station_state)


def _config(count_interval_s=60.0, delay_s=0.0, four_event=True, **kwargs):
    if four_event:
        kwargs["counted_doppler_model"] = "four_event_delay"
    return RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=count_interval_s,
        transponder_delay_s=delay_s,
        **kwargs,
    )


def _observable(receive_mid_s, config, provider=None, states=STATES):
    return four_event_counted_doppler_observable(
        receive_mid_s,
        None,
        T_GRID,
        states,
        EARTH_POS,
        EARTH_VEL,
        TRANSFORM_GRID,
        config,
        station_state_provider=provider or _provider(),
    )


def _endpoints(receive_mid_s, config, provider=None):
    return four_event_counted_doppler_endpoints(
        receive_mid_s,
        None,
        T_GRID,
        STATES,
        EARTH_POS,
        EARTH_VEL,
        TRANSFORM_GRID,
        config,
        station_state_provider=provider or _provider(),
    )


class FourEventPhysicsAndEventConvention(unittest.TestCase):
    """R4-P01/P02/P03/P06: event structure, naming and the count convention."""

    def test_p01_p02_four_events_are_distinct_and_correctly_related(self):
        for receive_mid_s, count_interval_s in GEOMETRIES:
            for delay_s in FROZEN_DELAYS_S:
                with self.subTest(t=receive_mid_s, tc=count_interval_s, delay=delay_s):
                    config = _config(count_interval_s, delay_s)
                    _, _, start, end, _ = _endpoints(receive_mid_s, config)
                    for solution in (start, end):
                        self.assertAlmostEqual(
                            solution.t2d_s - solution.t2u_s, delay_s, delta=1e-11
                        )
                        self.assertTrue(solution.converged)

    def test_p06_event_ordering_holds_for_every_geometry_and_delay(self):
        for receive_mid_s, count_interval_s in GEOMETRIES:
            for delay_s in FROZEN_DELAYS_S:
                with self.subTest(t=receive_mid_s, tc=count_interval_s, delay=delay_s):
                    _, _, start, end = _endpoints(receive_mid_s, _config(count_interval_s, delay_s))[:4]
                    for solution in (start, end):
                        self.assertLess(solution.t1_s, solution.t2u_s)
                        self.assertLessEqual(solution.t2u_s, solution.t2d_s)
                        self.assertLess(solution.t2d_s, solution.t3_s)
                        self.assertGreater(solution.uplink_light_time_s, 0.0)
                        self.assertGreater(solution.downlink_light_time_s, 0.0)

    def test_p02_zero_delay_collapses_the_two_spacecraft_events(self):
        _, _, start, _, _ = _endpoints(300.0, _config(60.0, 0.0))
        self.assertEqual(start.t2u_s, start.t2d_s)

    def test_p02_nonzero_delay_separates_the_two_spacecraft_events(self):
        _, _, start, _, _ = _endpoints(300.0, _config(60.0, 1e-3))
        self.assertLess(start.t2u_s, start.t2d_s)

    def test_p03_count_interval_is_defined_on_the_ground_receive_epochs(self):
        for count_interval_s in (10.0, 30.0, 60.0, 100.0):
            with self.subTest(tc=count_interval_s):
                start_s, end_s, start, end = _endpoints(
                    300.0, _config(count_interval_s, 1e-4)
                )[:4]
                self.assertAlmostEqual(end_s - start_s, count_interval_s, places=9)
                self.assertEqual(start.t3_s, start_s)
                self.assertEqual(end.t3_s, end_s)


class FourEventSolverCorrectness(unittest.TestCase):
    """R4-P05/P07: per-leg residuals and the independent simultaneous root."""

    def test_p05_per_leg_equation_residuals_meet_the_frozen_tolerance(self):
        worst = 0.0
        for receive_mid_s, count_interval_s in GEOMETRIES:
            for delay_s in FROZEN_DELAYS_S:
                _, _, start, end, _ = _endpoints(
                    receive_mid_s, _config(count_interval_s, delay_s)
                )
                for solution in (start, end):
                    worst = max(
                        worst,
                        solution.uplink_equation_residual_s,
                        solution.downlink_equation_residual_s,
                        solution.transponder_equation_residual_s,
                    )
        self.assertLess(worst, 1e-11, f"worst per-leg equation residual {worst!r} s")

    def test_p07_production_solver_matches_an_independent_simultaneous_root(self):
        """Structurally different oracle: scipy solves all three events at once."""
        worst = 0.0
        for receive_mid_s, count_interval_s in GEOMETRIES:
            for delay_s in (0.0, 1e-4, 1e-3):
                config = _config(count_interval_s, delay_s)
                _, _, start, end, adapter = _endpoints(receive_mid_s, config)
                for solution in (start, end):
                    station_rx = adapter.state(solution.t3_s)

                    def event_equations(y, _rx=station_rx, _sol=solution, _d=delay_s):
                        t1, t2u, t2d = y
                        station_tx = adapter.state(t1)
                        sc_up = _interp_state(T_GRID, STATES, t2u)
                        sc_down = _interp_state(T_GRID, STATES, t2d)
                        return [
                            C_LIGHT * (t2u - t1)
                            - np.linalg.norm(sc_up[:3] - station_tx[:3]),
                            C_LIGHT * (_sol.t3_s - t2d)
                            - np.linalg.norm(_rx[:3] - sc_down[:3]),
                            (t2d - t2u) - _d,
                        ]

                    reference = root(
                        event_equations,
                        [solution.t1_s, solution.t2u_s, solution.t2d_s],
                        method="hybr",
                        tol=1e-13,
                    )
                    self.assertTrue(reference.success)
                    worst = max(
                        worst,
                        abs(reference.x[0] - solution.t1_s),
                        abs(reference.x[1] - solution.t2u_s),
                        abs(reference.x[2] - solution.t2d_s),
                    )
        self.assertLess(worst, 1e-10, f"worst independent-root epoch error {worst!r} s")


class ZeroDelayReductionPivot(unittest.TestCase):
    """R4-P08: the four-event model must reduce to the accepted zero-delay physics."""

    def test_p08_reduces_to_the_r3_production_observable_bitwise(self):
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                r3_value = two_way_counted_doppler_observable(
                    receive_mid_s,
                    None,
                    T_GRID,
                    STATES,
                    EARTH_POS,
                    EARTH_VEL,
                    TRANSFORM_GRID,
                    _config(count_interval_s, 0.0, four_event=False),
                    station_state_provider=_provider(),
                )
                r4_value = _observable(receive_mid_s, _config(count_interval_s, 0.0))
                self.assertEqual(
                    r4_value,
                    r3_value,
                    "R4 at zero delay must reproduce the accepted R3 observable bitwise",
                )

    def test_p08_matches_the_accepted_model_s_within_the_ulp_budget(self):
        """P08C: historical model-S characterization (Owner Addendum 06).

        This assertion previously served as the primary NUMERICAL-ACCURACY
        oracle for the zero-delay observable. Owner Addendum 06 superseded that
        role: the long-arc time-conditioning repair (Q1-F01/Q1-F04) removed an
        O(ulp(pass-relative epoch)) error from production, and the accepted
        model-S reference -- which is byte-protected and therefore still
        reconstructs short intervals from large rounded epochs -- retains it.
        An independent 60-digit oracle that calls neither side shows repaired
        production closer to truth in every fixture case, by 20x to 280x.

        The test identity is deliberately preserved rather than renamed, and the
        model-S comparison is still computed. What changed is its role: it now
        CHARACTERIZES the historical reference's conditioning instead of bounding
        production. Numerical accuracy is gated by
        ``test_p08b_matches_the_independent_high_precision_oracle``.
        """
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                reference = exact_station_single_bounce_counted_doppler_reference(
                    receive_mid_s,
                    _reference_provider(),
                    T_GRID,
                    STATES,
                    CountedDopplerReferenceConfig(count_interval_s=count_interval_s),
                ).observable
                r4_value = _observable(receive_mid_s, _config(count_interval_s, 0.0))
                truth = _independent_observable(receive_mid_s, count_interval_s)
                production_error = abs(r4_value - truth)
                reference_error = abs(reference - truth)
                # The characterization claim: production is the more accurate of
                # the two. If this ever inverts, the supersession premise fails.
                self.assertLess(
                    production_error,
                    reference_error,
                    "repaired production must remain closer to independent truth "
                    "than the historical model-S reference",
                )
                # Model-S must still be reproducible and physically sane.
                self.assertTrue(math.isfinite(reference))
                self.assertLess(abs(reference - r4_value) / abs(truth), 1e-6)

    def test_p08b_matches_the_independent_high_precision_oracle(self):
        """P08B: the post-repair numerical-accuracy gate (Owner Addendum 06).

        The bound is derived, not fitted: accumulating the binary64 rounding of
        the repaired arithmetic -- two range/c divisions, the local light-time
        representation, two additions assembling the round-trip light time, and
        the endpoint difference -- gives K = 5 ulp(rho), which the counted
        observable scales by c / (2 Tc).
        """
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                truth = _independent_observable(receive_mid_s, count_interval_s)
                r4_value = _observable(receive_mid_s, _config(count_interval_s, 0.0))
                budget = (
                    INDEPENDENT_ORACLE_K
                    * C_LIGHT
                    * np.spacing(NOMINAL_ROUND_TRIP_LIGHT_TIME_S)
                    / (2.0 * count_interval_s)
                )
                self.assertLessEqual(abs(r4_value - truth), budget)

    def test_p08b_oracle_is_structurally_independent_of_production(self):
        """The accuracy oracle must not be production wearing a different hat."""
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        body = source.split("def _independent_local_round_trip_light_time")[1]
        body = body.split(chr(10) + "def ")[0]
        for forbidden in ("solve_two_way_light_time", "solve_two_way_range_events",
                          "four_event_counted_doppler", "counted_doppler_reference"):
            self.assertNotIn(forbidden, body)

    def test_p08b_oracle_agrees_with_a_second_independent_formulation(self):
        """Cross-check the scalar-root oracle against a coarse fixed-point solve."""
        for receive_mid_s, _tc in GEOMETRIES[:3]:
            with self.subTest(t=receive_mid_s):
                a = _independent_local_round_trip_light_time(receive_mid_s)
                # Independent fixed-point on the same local unknowns.
                station_rx = _station_state(receive_mid_s)
                tau_d = 0.0
                for _ in range(80):
                    sc = _interp_state(T_GRID, STATES, receive_mid_s - tau_d)
                    tau_d = float(np.linalg.norm(sc[:3] - station_rx[:3])) / C_LIGHT
                t2 = receive_mid_s - tau_d
                sc2 = _interp_state(T_GRID, STATES, t2)
                tau_u = 0.0
                for _ in range(80):
                    g1 = _station_state(t2 - tau_u)
                    tau_u = float(np.linalg.norm(sc2[:3] - g1[:3])) / C_LIGHT
                self.assertAlmostEqual(a, tau_d + tau_u, delta=1e-13)


class ModelFParity(unittest.TestCase):
    """R4-P09: identity with the accepted, byte-protected R2 model-F reference."""

    def test_p09_matches_model_f_across_the_frozen_delay_sweep(self):
        worst = 0.0
        for receive_mid_s, count_interval_s in GEOMETRIES:
            for delay_s in FROZEN_DELAYS_S:
                reference = exact_station_four_event_counted_doppler_reference(
                    receive_mid_s,
                    _reference_provider(),
                    T_GRID,
                    STATES,
                    reference_config_with_delay(
                        CountedDopplerReferenceConfig(count_interval_s=count_interval_s),
                        delay_s,
                    ),
                )
                self.assertEqual(reference.physical_event_count, 8)
                r4_value = _observable(receive_mid_s, _config(count_interval_s, delay_s))
                worst = max(
                    worst, abs(r4_value - reference.observable) / abs(reference.observable)
                )
        self.assertLess(worst, 1e-9, f"worst model-F relative difference {worst!r}")

    def test_p09_event_epochs_match_model_f_to_solver_precision(self):
        for delay_s in (0.0, 1e-3):
            with self.subTest(delay=delay_s):
                reference = exact_station_four_event_counted_doppler_reference(
                    300.0,
                    _reference_provider(),
                    T_GRID,
                    STATES,
                    reference_config_with_delay(
                        CountedDopplerReferenceConfig(count_interval_s=60.0), delay_s
                    ),
                )
                _, _, start, end, _ = _endpoints(300.0, _config(60.0, delay_s))
                self.assertAlmostEqual(
                    start.t1_s, reference.start_solution.t1_s, delta=1e-10
                )
                self.assertAlmostEqual(end.t1_s, reference.end_solution.t1_s, delta=1e-10)


class ConstantDelaySensitivity(unittest.TestCase):
    """R4-P10: the constant-delay sensitivity is NONZERO and first order."""

    def test_p10_delay_sensitivity_is_not_zero(self):
        """The withdrawn structural-zero claim would make this term vanish."""
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                sensitivity = four_event_counted_doppler_delay_sensitivity(
                    receive_mid_s,
                    None,
                    T_GRID,
                    STATES,
                    EARTH_POS,
                    EARTH_VEL,
                    TRANSFORM_GRID,
                    _config(count_interval_s, 1e-3),
                    station_state_provider=_provider(),
                )
                self.assertGreater(abs(sensitivity), 1e-3)

    def test_p10_delay_effect_is_first_order_in_the_delay(self):
        """Halving the delay must halve the observable shift, not quarter it."""
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                baseline = _observable(receive_mid_s, _config(count_interval_s, 0.0))
                orders = []
                previous = None
                for power in range(5):
                    delay_s = 1e-2 / (2.0**power)
                    shift = _observable(
                        receive_mid_s, _config(count_interval_s, delay_s)
                    ) - baseline
                    if previous is not None:
                        orders.append(
                            math.log(abs(previous[1] / shift))
                            / math.log(previous[0] / delay_s)
                        )
                    previous = (delay_s, shift)
                for order in orders:
                    self.assertAlmostEqual(order, 1.0, delta=0.15)


class NoFallbackAndFailureContracts(unittest.TestCase):
    """R4-P14/P15/P19 and the R4-Fxx contracts: everything fails closed."""

    def test_p14_no_legacy_or_single_bounce_fallback_after_a_failure(self):
        calls = {"legacy": 0, "single_bounce": 0}
        original_mci = radiometrics._station_state_mci
        original_solver = radiometrics.solve_two_way_light_time

        def counting_mci(*args, **kwargs):
            calls["legacy"] += 1
            return original_mci(*args, **kwargs)

        def counting_solver(*args, **kwargs):
            calls["single_bounce"] += 1
            return original_solver(*args, **kwargs)

        radiometrics._station_state_mci = counting_mci
        radiometrics.solve_two_way_light_time = counting_solver
        try:
            with self.assertRaises(ValueError):
                _observable(float(T_GRID[0]) + 0.05, _config(60.0, 1e-4))
        finally:
            radiometrics._station_state_mci = original_mci
            radiometrics.solve_two_way_light_time = original_solver
        self.assertEqual(calls["legacy"], 0)
        self.assertEqual(calls["single_bounce"], 0)

    def test_p15_p0a_gate_still_rejects_nonzero_delay_for_the_r3_model(self):
        with self.assertRaises(ValueError) as ctx:
            RangeRatePhysicsConfig(
                mode="two_way_counted_doppler", transponder_delay_s=1e-3
            )
        self.assertIn("single-bounce counted-Doppler model", str(ctx.exception))

    def test_four_event_model_rejects_the_legacy_station_grid(self):
        with self.assertRaises(ValueError) as ctx:
            RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                counted_doppler_model="four_event_delay",
                station_state_method="legacy_interpolated_transform_grid",
            )
        self.assertIn("exact_event_epoch_sxform", str(ctx.exception))

    def test_delay_must_be_smaller_than_the_count_interval(self):
        with self.assertRaises(ValueError) as ctx:
            _config(10.0, 10.0)
        self.assertIn("smaller than count_interval_s", str(ctx.exception))

    def test_negative_and_non_finite_delays_are_rejected(self):
        for bad in (-1e-9, float("nan"), float("inf")):
            with self.subTest(delay=bad):
                with self.assertRaises(ValueError):
                    _config(60.0, bad)

    def test_p19_event_epoch_outside_the_history_fails_closed(self):
        with self.assertRaises(ValueError) as ctx:
            _observable(float(T_GRID[0]) + 0.05, _config(60.0, 1e-4))
        self.assertIn("pre-roll", str(ctx.exception).lower() + " pre-roll")

    def test_provider_with_the_wrong_contract_is_rejected(self):
        bad_provider = CountedDopplerStationStateProvider(
            state_fn=_station_state, center="earth"
        )
        with self.assertRaises(radiometrics.StationStateEvaluationError):
            FourEventStationStateAdapter(bad_provider, T_GRID)

    def test_missing_provider_raises_the_four_event_error(self):
        with self.assertRaises(FourEventCountedDopplerError):
            FourEventStationStateAdapter(None, T_GRID)


class ProvenanceTruthfulness(unittest.TestCase):
    """R4-P17: every emitted label describes what actually executed."""

    def test_p17_model_version_follows_the_selected_model(self):
        self.assertEqual(
            _config(60.0, 0.0, four_event=False).counted_doppler_model_version,
            COUNTED_DOPPLER_MODEL_VERSION,
        )
        self.assertEqual(
            _config(60.0, 1e-3).counted_doppler_model_version,
            FOUR_EVENT_COUNTED_DOPPLER_MODEL_VERSION,
        )

    def test_p17_event_model_label_is_truthful(self):
        self.assertEqual(_config(60.0, 0.0, four_event=False).event_model, "single_bounce")
        self.assertEqual(_config(60.0, 1e-3).event_model, "four_event")

    def test_p17_metadata_reports_linear_earth_interpolation(self):
        adapter = FourEventStationStateAdapter(_provider(), T_GRID)
        metadata = four_event_counted_doppler_measurement_metadata(
            _config(60.0, 1e-3), adapter
        )
        self.assertEqual(
            metadata["earth_ephemeris_method"], COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD
        )
        self.assertEqual(metadata["earth_ephemeris_method"], "linear_grid_interpolation")
        self.assertNotIn("cubic_hermite_grid_interpolation", str(metadata))
        self.assertEqual(metadata["physical_event_count"], 8)
        self.assertIs(metadata["transponder_delay_is_solve_for"], False)
        self.assertEqual(metadata["transponder_delay_model"], "constant_scalar")
        self.assertEqual(metadata["count_interval_reference"], "ground_receive_epochs")


class DispatchAndCacheNeutrality(unittest.TestCase):
    """R4-P11/P18: consumers reach R4 through the existing abstraction."""

    def test_radiometrics_dispatch_matches_the_direct_r4_call_bitwise(self):
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                config = _config(count_interval_s, 1e-4)
                dispatched = two_way_counted_doppler_observable(
                    receive_mid_s,
                    None,
                    T_GRID,
                    STATES,
                    EARTH_POS,
                    EARTH_VEL,
                    TRANSFORM_GRID,
                    config,
                    station_state_provider=_provider(),
                )
                self.assertEqual(dispatched, _observable(receive_mid_s, config))

    def test_p18_cached_and_uncached_observables_are_identical(self):
        for receive_mid_s, count_interval_s in GEOMETRIES:
            with self.subTest(t=receive_mid_s, tc=count_interval_s):
                config = _config(count_interval_s, 1e-4)
                cached = _observable(receive_mid_s, config, provider=_provider(True))
                uncached = _observable(receive_mid_s, config, provider=_provider(False))
                self.assertEqual(cached, uncached)

    def test_p18_adjacent_distinct_epochs_are_never_merged(self):
        provider = _provider(True)
        epoch = 300.0
        provider.state(epoch)
        provider.state(np.nextafter(epoch, np.inf))
        self.assertEqual(len(provider._cache), 2)

    def test_p11_provider_declares_the_linear_earth_policy(self):
        adapter = FourEventStationStateAdapter(_provider(), T_GRID)
        self.assertEqual(adapter.earth_ephemeris_method, "linear_grid_interpolation")
        self.assertEqual(adapter.station_state_method, "exact_event_epoch_sxform")


class MutationDiscrimination(unittest.TestCase):
    """R4-P22: representative frozen mutations must be detected, not survive."""

    def test_m03_single_bounce_collapse_is_detected_at_nonzero_delay(self):
        config = _config(60.0, 1e-3)
        _, _, start, end, adapter = _endpoints(300.0, config)
        reference = _observable(300.0, config)

        def collapsed(solution):
            station_tx = adapter.state(solution.t1_s)
            station_rx = adapter.state(solution.t3_s)
            sc_down = _interp_state(T_GRID, STATES, solution.t2d_s)
            return (
                np.linalg.norm(sc_down[:3] - station_tx[:3])
                + np.linalg.norm(station_rx[:3] - sc_down[:3])
            ) / C_LIGHT

        mutated = C_LIGHT * ((collapsed(end) - collapsed(start)) / 60.0) / 2.0
        self.assertGreater(abs(mutated - reference), 1e-9)

    def test_m12_zeroing_the_delay_sensitivity_is_detected(self):
        analytic = four_event_counted_doppler_delay_sensitivity(
            300.0,
            None,
            T_GRID,
            STATES,
            EARTH_POS,
            EARTH_VEL,
            TRANSFORM_GRID,
            _config(60.0, 1.0),
            station_state_provider=_provider(),
        )
        step = 0.5
        finite_difference = (
            _observable(300.0, _config(60.0, 1.0 + step))
            - _observable(300.0, _config(60.0, 1.0 - step))
        ) / (2.0 * step)
        self.assertGreater(abs(0.0 - finite_difference) / abs(finite_difference), 1e-6)
        self.assertLess(
            abs(analytic - finite_difference) / abs(finite_difference), 1e-6
        )


if __name__ == "__main__":
    unittest.main()
