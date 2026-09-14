import ast
import hashlib
import subprocess
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import lunar_od.radiometrics as radiometrics_module
import lunar_od.two_way_range as two_way_range_module
from examples import r2_measurement_fidelity_validation as campaign_module
from lunar_od.constants import J2_MOON_UNNORMALIZED
from lunar_od.force_contract import (
    FORCE_CONTRACT_SCHEMA_VERSION,
    ConsumerReadiness,
    ConsumerRole,
    consumer_capabilities_for,
)
from lunar_od.scenario_config import (
    force_model_contract_from_scenario_config,
    scenario_config_from_mapping,
)
from lunar_od import (
    EXACT_EVENT_EPOCH_STATION_METHOD,
    LEGACY_INTERPOLATED_STATION_METHOD,
    RangeRatePhysicsConfig,
    StationStateEvaluationError,
    make_exact_counted_doppler_station_state_provider,
    two_way_counted_doppler_observable,
    PassGeometry,
    RangeRatePhysicsConfig,
    Station,
    analyze_state_bias_correlation,
    compute_range_rate_residuals,
    compute_range_rate_residuals_analytic,
    measurement_covariance_matrix,
    measurement_sigma_vector,
    summarize_arc_observability_combinations,
    summarize_weighted_jacobian,
)


class ODContractTests(unittest.TestCase):
    def test_range_rate_weighting_order_is_range_rr_az_el_per_observation(self):
        stations = (
            _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5),
            _station("B", sigma_range_m=5.0, sigma_rr_mps=0.05, sigma_angle_rad=5e-5),
        )
        pass_geo = _pass_geo(np.array([0.0]), stations)
        obs_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            ]
        )

        sigma = measurement_sigma_vector(obs_data, pass_geo, "range_rate")

        np.testing.assert_allclose(sigma, [2.0, 0.02, 2e-5, 2e-5, 5.0, 0.05, 5e-5, 5e-5])

    def test_measurement_covariance_r_is_diag_sigma_squared_in_residual_order(self):
        stations = (
            _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5),
            _station("B", sigma_range_m=5.0, sigma_rr_mps=0.05, sigma_angle_rad=5e-5),
        )
        pass_geo = _pass_geo(np.array([0.0]), stations)
        obs_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            ]
        )

        r_cov = measurement_covariance_matrix(obs_data, pass_geo, "range_rate")

        expected_sigma = np.array([2.0, 0.02, 2e-5, 2e-5, 5.0, 0.05, 5e-5, 5e-5])
        np.testing.assert_allclose(r_cov, np.diag(expected_sigma**2), rtol=0.0, atol=0.0)

    def test_two_way_doppler_uses_same_mps_equivalent_r_order(self):
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(np.array([0.0]), (station,))
        object.__setattr__(
            pass_geo,
            "range_rate_physics",
            RangeRatePhysicsConfig(mode="two_way_counted_doppler", count_interval_s=60.0),
        )
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

        sigma = measurement_sigma_vector(obs_data, pass_geo, "range_rate")
        r_cov = measurement_covariance_matrix(obs_data, pass_geo, "range_rate")

        np.testing.assert_allclose(sigma, [2.0, 0.02, 2e-5, 2e-5])
        np.testing.assert_allclose(np.diag(r_cov), [4.0, 4e-4, 4e-10, 4e-10])

    def test_range_rate_residual_contract_is_observed_minus_computed(self):
        t_s = np.array([0.0])
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(t_s, (station,))
        state_history = np.array([[station.r_ecef_m[0] + 1000.0, station.r_ecef_m[1], station.r_ecef_m[2], 1.5, 0.0, 0.0]])
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        _, h_meas = compute_range_rate_residuals(state_history, obs_data, pass_geo)

        offset = np.array([3.0, -0.02, 4e-5, -5e-5])
        obs_data[:, 1:5] = h_meas + offset.reshape(1, 4)
        residuals, _ = compute_range_rate_residuals(state_history, obs_data, pass_geo)

        np.testing.assert_allclose(residuals, offset, rtol=0.0, atol=1e-14)

    def test_range_rate_jacobian_is_computed_measurement_derivative(self):
        t_s = np.array([0.0])
        station = _station("A", sigma_range_m=2.0, sigma_rr_mps=0.02, sigma_angle_rad=2e-5)
        pass_geo = _pass_geo(t_s, (station,))
        state = np.array([station.r_ecef_m[0] + 1000.0, station.r_ecef_m[1] + 30.0, station.r_ecef_m[2] + 20.0, 1.5, 0.2, -0.1])
        state_history = state.reshape(1, 6)
        obs_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        _, h0, h_tilde = compute_range_rate_residuals_analytic(state_history, obs_data, pass_geo)
        obs_data[:, 1:5] = h0
        residual0, _, h_tilde = compute_range_rate_residuals_analytic(state_history, obs_data, pass_geo)

        step = np.array([0.2, -0.1, 0.05, 1e-4, -2e-4, 3e-4])
        residual1, h1, _ = compute_range_rate_residuals_analytic((state + step).reshape(1, 6), obs_data, pass_geo)

        np.testing.assert_allclose(h1.reshape(-1) - h0.reshape(-1), h_tilde @ step, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(residual1 - residual0, -(h_tilde @ step), rtol=0.0, atol=2e-5)

    def test_state_bias_correlation_diagnostic_flags_coupling(self):
        covariance = np.eye(8)
        covariance[0, 6] = covariance[6, 0] = 0.5
        covariance[3, 7] = covariance[7, 3] = -0.25

        result = analyze_state_bias_correlation(covariance, num_state=6)

        self.assertEqual(result.num_bias, 2)
        self.assertAlmostEqual(result.correlation_matrix[0, 0], 0.5)
        self.assertAlmostEqual(result.correlation_matrix[3, 1], -0.25)
        self.assertAlmostEqual(result.max_abs_correlation, 0.5)

    def test_arc_combinations_can_improve_rank(self):
        arc1 = summarize_weighted_jacobian("position", 1, np.array([[1.0, 0.0], [0.0, 0.0]]), rank_tol=1e-12)
        arc2 = summarize_weighted_jacobian("position", 1, np.array([[0.0, 0.0], [0.0, 2.0]]), rank_tol=1e-12)

        combos = summarize_arc_observability_combinations(((1, arc1), (2, arc2)), rank_tol=1e-12)
        by_ids = {combo.arc_ids: combo.observability for combo in combos}

        self.assertEqual(by_ids[(1,)].rank, 1)
        self.assertEqual(by_ids[(2,)].rank, 1)
        self.assertEqual(by_ids[(1, 2)].rank, 2)
        self.assertEqual(by_ids[(1, 2)].rank_deficiency, 0)


def _station(name: str, *, sigma_range_m: float, sigma_rr_mps: float, sigma_angle_rad: float) -> Station:
    return Station(
        name=name,
        lat_deg=0.0,
        lon_deg=0.0,
        alt_m=0.0,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=sigma_range_m,
        sigma_angle_rad=sigma_angle_rad,
        sigma_range_rate_mps=sigma_rr_mps,
    )


def _pass_geo(t_s, stations):
    return PassGeometry(
        t_s=np.asarray(t_s, dtype=float),
        earth_pos_mci_m=np.zeros((len(t_s), 3)),
        earth_vel_mci_mps=np.zeros((len(t_s), 3)),
        x_j2000_to_itrf93=np.repeat(np.eye(6)[None, :, :], len(t_s), axis=0),
        stations=tuple(stations),
        measurement_type="range_rate",
    )



# ---------------------------------------------------------------------------
# R3 exact event-epoch station transform — contract gates
#
# R3-P12 nonzero transponder delay stays fail-closed
# R3-P16 every failure condition fails closed, with no legacy fallback (F01-F14)
# R3-P17 R3 changes nothing outside the measurement model
# R3-P21 provider protocol is shape-compatible with M3 without importing it
# ---------------------------------------------------------------------------


class _R3Station:
    name = "R3 contract station"
    r_ecef_m = np.array([6378137.0, 0.0, 0.0])


def _r3_identity_sxform(_source, _target, _et):
    return np.eye(6)


def _r3_grid(n=9, span=400.0):
    t_grid = np.linspace(-span, span, n)
    earth_pos = np.zeros((n, 3))
    earth_vel = np.zeros((n, 3))
    earth_pos[:, 0] = 3.8e8
    return t_grid, earth_pos, earth_vel


class R3NonzeroDelayRemainsFailClosed(unittest.TestCase):
    """R3-P12: R3 adds no silent four-event support."""

    def test_p12_nonzero_delay_rejected_at_config_construction(self):
        for method in (
            EXACT_EVENT_EPOCH_STATION_METHOD,
            LEGACY_INTERPOLATED_STATION_METHOD,
        ):
            with self.assertRaises(ValueError) as ctx:
                RangeRatePhysicsConfig(
                    mode="two_way_counted_doppler",
                    transponder_delay_s=1e-6,
                    station_state_method=method,
                )
            message = str(ctx.exception)
            self.assertIn("four-event", message)
            self.assertIn("legacy single-bounce counted-Doppler", message)

    def test_p12_zero_delay_is_still_accepted_in_both_methods(self):
        for method in (
            EXACT_EVENT_EPOCH_STATION_METHOD,
            LEGACY_INTERPOLATED_STATION_METHOD,
        ):
            config = RangeRatePhysicsConfig(
                mode="two_way_counted_doppler",
                transponder_delay_s=0.0,
                station_state_method=method,
            )
            self.assertEqual(config.transponder_delay_s, 0.0)


class R3FailureContract(unittest.TestCase):
    """R3-P16: F01-F14 fail closed; F14 is the defining no-fallback rule."""

    def _provider_kwargs(self, **overrides):
        t_grid, earth_pos, earth_vel = _r3_grid()
        kwargs = dict(
            station=_R3Station(),
            et0_s=0.0,
            t_grid_s=t_grid,
            earth_pos_mci_m=earth_pos,
            earth_vel_mci_mps=earth_vel,
            sxform_fn=_r3_identity_sxform,
        )
        kwargs.update(overrides)
        return kwargs

    def _make(self, **overrides):
        kwargs = self._provider_kwargs(**overrides)
        return make_exact_counted_doppler_station_state_provider(
            kwargs.pop("station"),
            kwargs.pop("et0_s"),
            kwargs.pop("t_grid_s"),
            kwargs.pop("earth_pos_mci_m"),
            kwargs.pop("earth_vel_mci_mps"),
            **kwargs,
        )

    def test_f01_f02_f03_sxform_failure_is_wrapped_and_names_the_context(self):
        def exploding(_source, _target, _et):
            raise RuntimeError("SPICE(NOFRAMECONNECT) kernel pool empty")

        provider = self._make(sxform_fn=exploding)
        with self.assertRaises(StationStateEvaluationError) as ctx:
            provider.state(0.0)
        message = str(ctx.exception)
        self.assertIn("R3 contract station", message)
        self.assertIn("J2000->ITRF93", message)
        self.assertIn("ET", message)
        self.assertIn("no legacy fallback was used", message.lower())

    def test_f04_wrong_shape_or_nonfinite_transform_is_rejected(self):
        for bad in (np.eye(5), np.full((6, 6), np.nan), np.full((6, 6), np.inf)):
            provider = self._make(sxform_fn=lambda _s, _t, _e, m=bad: m)
            with self.assertRaises(StationStateEvaluationError) as ctx:
                provider.state(0.0)
            self.assertIn("no legacy fallback was used", str(ctx.exception).lower())

    def test_f05_malformed_station_site_is_rejected_at_construction(self):
        class _Bad:
            name = "bad site"
            r_ecef_m = np.array([1.0, np.nan, 3.0])

        with self.assertRaises(ValueError) as ctx:
            self._make(station=_Bad())
        self.assertIn("bad site", str(ctx.exception))

    def test_f06_nonfinite_earth_history_fails_closed(self):
        t_grid, earth_pos, earth_vel = _r3_grid()
        earth_pos = earth_pos.copy()
        earth_pos[4, 1] = np.nan
        provider = self._make(earth_pos_mci_m=earth_pos)
        with self.assertRaises(StationStateEvaluationError) as ctx:
            provider.state(float(t_grid[4]))
        message = str(ctx.exception)
        self.assertIn("Earth position history is non-finite", message)
        self.assertIn("no legacy fallback was used", message.lower())

    def test_f07_provider_must_declare_the_mci_moon_centred_contract(self):
        provider = self._make()
        wrong = replace(provider, center="earth")
        t_grid, earth_pos, earth_vel = _r3_grid()
        with self.assertRaises(StationStateEvaluationError) as ctx:
            radiometrics_module._counted_station_state(
                0.0,
                _R3Station(),
                t_grid,
                earth_pos,
                earth_vel,
                np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0),
                provider=wrong,
                endpoint_label="contract",
                event_label="downlink",
                consumer="test",
            )
        self.assertIn("Moon-centred", str(ctx.exception))

    def test_f11_missing_or_nonfinite_et0_is_an_explicit_failure(self):
        for bad_et0 in (None, float("nan"), float("inf")):
            with self.assertRaises(ValueError) as ctx:
                self._make(et0_s=bad_et0)
            message = str(ctx.exception)
            self.assertIn("et0_s must be finite", message)
            self.assertIn("deliberate opt-in", message)

    def test_f14_exact_never_falls_back_when_legacy_data_is_available(self):
        """The defining rule: exact configured + no et0_s + legacy grid present.

        Expected: explicit failure, the legacy helper is never called, and no
        provenance claims a successful exact evaluation.
        """
        t_grid, earth_pos, earth_vel = _r3_grid(n=41, span=400.0)
        states = np.zeros((t_grid.size, 6))
        states[:, 0] = 2.0e6 + 90.0 * t_grid
        states[:, 3] = 90.0
        # A perfectly usable legacy transform grid is deliberately supplied.
        xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        exact_cfg = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=20.0,
            station_state_method=EXACT_EVENT_EPOCH_STATION_METHOD,
        )
        calls = []
        original = radiometrics_module._station_state_mci

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        radiometrics_module._station_state_mci = counting
        try:
            with self.assertRaises(ValueError) as ctx:
                two_way_counted_doppler_observable(
                    0.0, _R3Station(), t_grid, states, earth_pos, earth_vel,
                    xforms, exact_cfg, et0_s=None,
                )
        finally:
            radiometrics_module._station_state_mci = original
        self.assertIn("et0_s must be finite", str(ctx.exception))
        self.assertEqual(len(calls), 0, "legacy helper must never be reached")

    def test_f14_legacy_remains_reachable_only_by_explicit_opt_in(self):
        t_grid, earth_pos, earth_vel = _r3_grid(n=41, span=400.0)
        states = np.zeros((t_grid.size, 6))
        states[:, 0] = 2.0e6 + 90.0 * t_grid
        states[:, 3] = 90.0
        xforms = np.repeat(np.eye(6)[None, :, :], t_grid.size, axis=0)
        legacy_cfg = RangeRatePhysicsConfig(
            mode="two_way_counted_doppler",
            count_interval_s=20.0,
            station_state_method=LEGACY_INTERPOLATED_STATION_METHOD,
        )
        value = two_way_counted_doppler_observable(
            0.0, _R3Station(), t_grid, states, earth_pos, earth_vel,
            xforms, legacy_cfg,
        )
        self.assertTrue(np.isfinite(value))


class R3ProviderProtocolCompatibility(unittest.TestCase):
    """R3-P21: shape-compatible with the M3 provider, without importing it."""

    def test_p21_exposes_the_m3_provider_surface(self):
        t_grid, earth_pos, earth_vel = _r3_grid()
        provider = make_exact_counted_doppler_station_state_provider(
            _R3Station(), 0.0, t_grid, earth_pos, earth_vel,
            sxform_fn=_r3_identity_sxform,
        )
        for attribute in ("state", "station_state_method", "earth_ephemeris_method"):
            self.assertTrue(hasattr(provider, attribute), attribute)
        self.assertEqual(provider.state(0.0).shape, (6,))
        m3_provider = two_way_range_module.TwoWayStationStateProvider
        self.assertTrue(callable(getattr(m3_provider, "state", None)))
        m3_fields = set(getattr(m3_provider, "__dataclass_fields__", {}))
        for attribute in ("state_fn", "station_state_method", "earth_ephemeris_method"):
            self.assertIn(attribute, m3_fields, attribute)
        # The R3 provider carries the same surface, so a later Option-C
        # unification stays a rename rather than a redesign.
        r3_fields = set(type(provider).__dataclass_fields__)
        self.assertTrue(m3_fields.issubset(r3_fields), m3_fields - r3_fields)

    def test_p21_radiometrics_does_not_import_two_way_range_or_the_reference(self):
        """Checked on the import graph, not on raw text.

        The words appear legitimately in prose (for example the nonzero-delay
        message naming M3 two-way range); only real imports are forbidden.
        """
        tree = ast.parse(
            Path(radiometrics_module.__file__).read_text(encoding="utf-8")
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module)
                imported.update(
                    f"{node.module or ''}.{alias.name}" for alias in node.names
                )
        forbidden = {"two_way_range", "two_way_counted_doppler_reference"}
        for name in imported:
            leaf = name.lstrip(".").split(".")[-1]
            self.assertNotIn(leaf, forbidden, f"radiometrics imports {name}")
        # Import-time module graph confirms it too.
        self.assertNotIn(
            "two_way_range",
            {
                getattr(value, "__name__", "").split(".")[-1]
                for value in vars(radiometrics_module).values()
                if isinstance(value, type(radiometrics_module))
            },
        )

    # --- Owner Addendum 07B: M3 source-protection identity transition -------
    #
    # This gate used to assert
    #
    #     current repaired production blob == pre-repair production blob
    #
    # which became logically unsatisfiable the moment Owner Addendum 05
    # authorized editing lunar_od/two_way_range.py to repair the confirmed
    # long-arc time-conditioning defect. Its scientific purpose was never
    # "the source may never change"; it was PROTECTION / PROVENANCE.
    #
    # It is therefore replaced by three separate hard contracts (P21-A/B/C),
    # none of which involves a physical threshold. Nothing is weakened: the
    # historical blob stays permanently pinned, the repaired blob is pinned
    # too, and the diff between them is bounded to the authorized scope.

    #: Pre-repair M3 blob. Present at BOTH the R3 baseline (632560d) and the
    #: canonical R4 baseline (ec4b6871) -- two_way_range.py did not change
    #: between them -- so this single object identity anchors the whole
    #: pre-repair lineage.
    M3_HISTORICAL_BLOB = "a3561b252d4ca627f8fdda05deff7baf86858eeb"
    M3_CANONICAL_PRE_REPAIR_COMMIT = "ec4b6871cc8a6bf1f15e7a1dc5d0b7fb013f83d8"
    M3_R3_BASELINE_COMMIT = "632560d72d51b3d77b401365b1839908c2c8e85f"

    #: Repaired CANDIDATE blob. Deliberately not called "canonical": canonical
    #: still points at the pre-repair source and no merge has occurred.
    M3_REPAIRED_CANDIDATE_BLOB = "e3a05e2447e61babcfb79de0ec790eb560ac4bf9"

    def _git_blob_id(self, root, commit):
        """Blob object id of M3 at ``commit``, from the Git object store.

        Object lookup only -- deliberately no working-tree fallback, so a
        dirty or reverted checkout can never make this gate pass.
        """
        git = campaign_module.resolve_git_executable()
        return subprocess.run(
            [git, "-C", str(root), "rev-parse", f"{commit}:lunar_od/two_way_range.py"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def test_p21a_m3_historical_pre_repair_blob_identity_is_preserved(self):
        """P21-A: the pre-repair M3 object is permanently protected."""
        root = Path(radiometrics_module.__file__).resolve().parents[1]
        for commit in (self.M3_CANONICAL_PRE_REPAIR_COMMIT, self.M3_R3_BASELINE_COMMIT):
            self.assertEqual(
                self._git_blob_id(root, commit),
                self.M3_HISTORICAL_BLOB,
                f"historical M3 blob moved at {commit}",
            )

    def test_p21b_m3_repaired_candidate_blob_identity_is_fixed(self):
        """P21-B: the active repaired source has a pinned, auditable identity."""
        git = campaign_module.resolve_git_executable()
        root = Path(radiometrics_module.__file__).resolve().parents[1]
        # hash-object over the working tree, so an unrecorded local edit to the
        # repaired module is caught rather than silently qualified.
        current = subprocess.run(
            [git, "-C", str(root), "hash-object", "lunar_od/two_way_range.py"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(current, self.M3_REPAIRED_CANDIDATE_BLOB)
        self.assertNotEqual(
            current,
            self.M3_HISTORICAL_BLOB,
            "repaired candidate must not be the pre-repair blob",
        )

    def test_p21c_m3_repair_diff_stays_inside_the_authorized_scope(self):
        """P21-C: historical -> repaired diff is conditioning-only.

        Owner Addendum 05 authorized a numerical time-conditioning repair and
        nothing else. This bounds the diff to that scope by checking, in the
        changed lines, that the physical light-time equation, the event
        definitions, the range/delay conventions and the speed of light are
        untouched.
        """
        git = campaign_module.resolve_git_executable()
        root = Path(radiometrics_module.__file__).resolve().parents[1]
        diff = subprocess.run(
            [git, "-C", str(root), "diff", "-U0",
             self.M3_CANONICAL_PRE_REPAIR_COMMIT, "--",
             "lunar_od/two_way_range.py"],
            check=True, capture_output=True, text=True,
        ).stdout

        changed = [
            line[1:].strip()
            for line in diff.splitlines()
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith(("+++", "---"))
        ]
        code = [
            line for line in changed
            if line and not line.startswith("#")
        ]

        # Every changed CODE line must be one of the three authorized
        # conditioning repairs -- residual evaluation, or round-trip assembly.
        authorized_substrings = (
            "downlink_equation_residual_s =",
            "uplink_equation_residual_s =",
            "round_trip_light_time_s =",
        )
        for line in code:
            self.assertTrue(
                any(token in line for token in authorized_substrings),
                f"M3 repair touched an unauthorized code line: {line!r}",
            )

        # And these invariants must appear nowhere in the changed code.
        forbidden = (
            "light_speed_mps =", "C_LIGHT", "transponder_delay_s =",
            "def solve_two_way_range_events", "station_state_provider =",
            "_interp_state(", "earth_", "sxform", "j2_", "force",
        )
        for line in code:
            for token in forbidden:
                self.assertNotIn(
                    token, line,
                    f"M3 repair changed a protected construct ({token}): {line!r}",
                )


class R3CompatibilityUnchangedOutsideTheMeasurementModel(unittest.TestCase):
    """R3-P17: a measurement-model update never moves the force contract."""

    ZERO_J2_FINGERPRINT = (
        "sha256:9b93897a545d0d2f1cb2b5329ce6be79051fef97e1529bff3bea565c4c31418d"
    )
    LUNAR_J2_FINGERPRINT = (
        "sha256:11d33466c53e4c4a48cb8f72984ad1740e81de26ef342b4ecd106a30a7ddb52d"
    )

    def _contract(self, j2_moon):
        config = scenario_config_from_mapping(
            {
                "name": "r3-compat",
                "measurement_type": "range_rate",
                "estimator_type": "bls_lm",
                "start_mode": "cold",
                "network": "multi",
                "j2_moon": j2_moon,
            }
        )
        return force_model_contract_from_scenario_config(config)

    def test_p17_fingerprints_and_schema_are_unchanged(self):
        self.assertEqual(
            self._contract(0.0).force_model_fingerprint(), self.ZERO_J2_FINGERPRINT
        )
        self.assertEqual(
            self._contract(float(J2_MOON_UNNORMALIZED)).force_model_fingerprint(),
            self.LUNAR_J2_FINGERPRINT,
        )
        self.assertEqual(FORCE_CONTRACT_SCHEMA_VERSION, "r0b.force-model-contract.v1")

    def test_p17_readiness_and_fail_closed_roles_are_unchanged(self):
        lunar = consumer_capabilities_for(
            lunar_j2_on=True, earth_j2_on=False, harmonics_on=False
        )
        self.assertEqual(
            lunar[ConsumerRole.POSTERIOR_COVARIANCE], ConsumerReadiness.VERIFIED
        )
        self.assertEqual(lunar[ConsumerRole.OBSERVABILITY], ConsumerReadiness.VERIFIED)
        earth = consumer_capabilities_for(
            lunar_j2_on=False, earth_j2_on=True, harmonics_on=False
        )
        self.assertTrue(
            all(value == ConsumerReadiness.UNSUPPORTED for value in earth.values())
        )
        harmonics = consumer_capabilities_for(
            lunar_j2_on=False, earth_j2_on=False, harmonics_on=True
        )
        self.assertEqual(
            harmonics[ConsumerRole.TRUTH_STATE],
            ConsumerReadiness.EXPERIMENTAL_DIRECT_TRAJECTORY_ONLY,
        )
        self.assertEqual(
            harmonics[ConsumerRole.POSTERIOR_COVARIANCE], ConsumerReadiness.UNSUPPORTED
        )

    def test_p17_station_method_does_not_reach_the_force_contract(self):
        """Selecting either station strategy must not move a fingerprint."""
        before = self._contract(float(J2_MOON_UNNORMALIZED)).force_model_fingerprint()
        for method in (
            EXACT_EVENT_EPOCH_STATION_METHOD,
            LEGACY_INTERPOLATED_STATION_METHOD,
        ):
            payload = {
                "name": "r3-compat",
                "measurement_type": "range_rate",
                "estimator_type": "bls_lm",
                "start_mode": "cold",
                "network": "multi",
                "j2_moon": float(J2_MOON_UNNORMALIZED),
                "range_rate_physics": "two_way_counted_doppler",
                "station_state_method": method,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = scenario_config_from_mapping(payload)
            after = force_model_contract_from_scenario_config(
                config
            ).force_model_fingerprint()
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
