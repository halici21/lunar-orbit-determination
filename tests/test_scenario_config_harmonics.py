"""Phase 13B2a -- lunar-harmonics scenario config fields and validation.

Covers the eight additive ``lunar_gravity_*`` / ``enable_lunar_harmonics``
fields, their cross-field rules (explicit path + nmax, frame/profile
consistency, double-count and STM guards), the TEMPORARY Phase 13B2a
not-yet-consumed guard, and the ``scenario_lunar_gravity_model`` builder.

No SPICE, no kernels, no network, no real GRAIL data: model-loading tests use
only the committed synthetic fixture ``tests/fixtures/gravity/*.tab``.
"""
import os
import unittest
from pathlib import Path
from unittest import mock

from lunar_od import scenario_config as sc
from lunar_od.scenario_config import (
    ScenarioConfig,
    scenario_config_from_mapping,
    scenario_config_schema,
    scenario_lunar_gravity_model,
    scenario_lunar_kernel_profile,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gravity" / "synthetic_norm_sha.tab"

_BASE = {
    "name": "harm",
    "measurement_type": "position",
    "estimator_type": "ukf",
    "start_mode": "cold",
    "network": "multi",
}


def _on_payload(**overrides):
    """A harmonics-on payload that passes every lunar rule except the 13B2a guard."""
    payload = {
        **_BASE,
        "enable_lunar_harmonics": True,
        "lunar_gravity_model_path": str(FIXTURE),
        "lunar_gravity_nmax": 3,
    }
    payload.update(overrides)
    return payload


class HarmonicsOffTests(unittest.TestCase):
    # 1 -- new fields default correctly; legacy fields untouched -------------
    def test_default_mapping_new_fields(self):
        config = scenario_config_from_mapping(dict(_BASE))
        self.assertFalse(config.enable_lunar_harmonics)
        self.assertIsNone(config.lunar_gravity_model_path)
        self.assertIsNone(config.lunar_gravity_nmax)
        self.assertIsNone(config.lunar_gravity_mmax)
        self.assertEqual(config.lunar_gravity_frame, "MOON_PA_DE421")
        self.assertEqual(config.lunar_gravity_rotation_cadence_s, 60.0)
        self.assertIsNone(config.lunar_gravity_rotation_margin_s)
        self.assertIsNone(config.lunar_gravity_kernel_profile)
        # legacy behavior unchanged
        self.assertEqual(config.j2_moon, 0.0)
        self.assertFalse(config.enable_earth_j2)

    # 2 -- harmonics off: no path required, no file-system access ------------
    def test_off_requires_nothing_and_touches_no_fs(self):
        with mock.patch.object(
            sc, "resolve_gravity_dir",
            side_effect=AssertionError("resolve_gravity_dir must not be called"),
        ), mock.patch.object(
            sc, "load_lunar_gravity_model",
            side_effect=AssertionError("loader must not be called"),
        ):
            config = scenario_config_from_mapping(dict(_BASE))
            self.assertIsNone(scenario_lunar_gravity_model(config))

    def test_schema_advertises_new_fields(self):
        props = scenario_config_schema()["properties"]
        self.assertIn("enable_lunar_harmonics", props)
        self.assertEqual(props["enable_lunar_harmonics"]["default"], False)
        self.assertEqual(
            props["lunar_gravity_frame"]["enum"], ["MOON_PA_DE421", "MOON_PA_DE440"]
        )
        self.assertEqual(props["lunar_gravity_rotation_cadence_s"]["default"], 60.0)
        self.assertEqual(props["lunar_gravity_kernel_profile"]["enum"], [None, "DE421", "DE440"])


class HarmonicsOnValidationTests(unittest.TestCase):
    # 3/4 -- explicit path and nmax are mandatory -----------------------------
    def test_missing_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires lunar_gravity_model_path"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_model_path=None))

    def test_missing_nmax_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit lunar_gravity_nmax"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_nmax=None))

    # 5 -- nmax type/range rules ----------------------------------------------
    def test_invalid_nmax_rejected(self):
        for bad in (1, 0, -3):
            with self.assertRaisesRegex(ValueError, "lunar_gravity_nmax must be >= 2"):
                scenario_config_from_mapping(_on_payload(lunar_gravity_nmax=bad))
        with self.assertRaisesRegex(ValueError, "not a boolean"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_nmax=True))

    # 6 -- mmax rules ----------------------------------------------------------
    def test_invalid_mmax_rejected(self):
        with self.assertRaisesRegex(ValueError, "lunar_gravity_mmax must be >= 0"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_mmax=-1))
        with self.assertRaisesRegex(ValueError, "0 <= mmax <= nmax"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_mmax=5))

    # 7/8 -- cadence and margin -------------------------------------------------
    def test_invalid_cadence_rejected(self):
        for bad in (0.0, -60.0):
            with self.assertRaisesRegex(ValueError, "lunar_gravity_rotation_cadence_s"):
                scenario_config_from_mapping(
                    _on_payload(lunar_gravity_rotation_cadence_s=bad)
                )

    def test_invalid_margin_rejected(self):
        with self.assertRaisesRegex(ValueError, "lunar_gravity_rotation_margin_s"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_rotation_margin_s=-1.0))

    # 9/10/11 -- frame rules -----------------------------------------------------
    def test_allowed_frames_accepted(self):
        for frame in ("MOON_PA_DE421", "MOON_PA_DE440"):
            # all lunar rules pass; ONLY the temporary 13B2a guard fires
            with self.assertRaisesRegex(ValueError, "not yet consumed"):
                scenario_config_from_mapping(_on_payload(lunar_gravity_frame=frame))

    def test_bare_moon_pa_rejected(self):
        with self.assertRaisesRegex(ValueError, "lunar_gravity_frame must be one of"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_frame="MOON_PA"))

    def test_unknown_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, "lunar_gravity_frame must be one of"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_frame="MOON_ME"))

    # 12/13/14 -- kernel profile rules --------------------------------------------
    def test_profile_none_derives_from_frame(self):
        de421 = ScenarioConfig(**{**_BASE, "lunar_gravity_frame": "MOON_PA_DE421"})
        de440 = ScenarioConfig(**{**_BASE, "lunar_gravity_frame": "MOON_PA_DE440"})
        explicit = ScenarioConfig(**{**_BASE, "lunar_gravity_kernel_profile": "DE440"})
        self.assertEqual(scenario_lunar_kernel_profile(de421), "DE421")
        self.assertEqual(scenario_lunar_kernel_profile(de440), "DE440")
        self.assertEqual(scenario_lunar_kernel_profile(explicit), "DE440")

    def test_profile_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires lunar_gravity_kernel_profile"):
            scenario_config_from_mapping(
                _on_payload(lunar_gravity_frame="MOON_PA_DE440",
                            lunar_gravity_kernel_profile="DE421")
            )
        with self.assertRaisesRegex(ValueError, "requires lunar_gravity_kernel_profile"):
            scenario_config_from_mapping(
                _on_payload(lunar_gravity_frame="MOON_PA_DE421",
                            lunar_gravity_kernel_profile="DE440")
            )

    def test_matching_profile_accepted(self):
        with self.assertRaisesRegex(ValueError, "not yet consumed"):
            scenario_config_from_mapping(
                _on_payload(lunar_gravity_frame="MOON_PA_DE440",
                            lunar_gravity_kernel_profile="DE440")
            )

    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ValueError, "lunar_gravity_kernel_profile must be one of"):
            scenario_config_from_mapping(_on_payload(lunar_gravity_kernel_profile="CUSTOM"))

    # 15 -- double-count guard ------------------------------------------------------
    def test_double_count_guard(self):
        with self.assertRaisesRegex(ValueError, "count lunar J2\\s*twice|J2\\s*twice"):
            scenario_config_from_mapping(_on_payload(j2_moon=2.0330530e-4))

    # 16 -- STM guard ------------------------------------------------------------
    def test_stm_estimators_rejected(self):
        for estimator in ("bls_lm", "srif"):
            with self.assertRaisesRegex(ValueError, "STM-based estimators"):
                scenario_config_from_mapping(_on_payload(estimator_type=estimator))

    # 17 -- temporary not-yet-consumed guard (UKF path) ---------------------------
    def test_ukf_hits_temporary_guard(self):
        with self.assertRaisesRegex(
            ValueError,
            "not yet consumed by the scenario runner .*Phase 13B2b.*propagate_state",
        ):
            scenario_config_from_mapping(_on_payload())

    # 18 -- Earth J2 stays composable at the config level --------------------------
    def test_earth_j2_not_rejected(self):
        # the ONLY error must be the temporary guard, not an Earth-J2 rule
        with self.assertRaisesRegex(ValueError, "not yet consumed"):
            scenario_config_from_mapping(_on_payload(enable_earth_j2=True))


class ScenarioLunarGravityModelTests(unittest.TestCase):
    """Builder tests construct the dataclass directly (bypassing the temporary
    parse guard, which is exactly the intended 13B2a escape hatch)."""

    def _config(self, **overrides):
        fields = {
            **_BASE,
            "enable_lunar_harmonics": True,
            "lunar_gravity_model_path": str(FIXTURE),
            "lunar_gravity_nmax": 3,
        }
        fields.update(overrides)
        return ScenarioConfig(**fields)

    # 19/20 -- absolute fixture path loads once ------------------------------------
    def test_absolute_path_loads(self):
        model = scenario_lunar_gravity_model(self._config())
        self.assertIsNotNone(model)
        self.assertEqual(model.nmax, 3)
        self.assertEqual(model.mmax, 3)
        self.assertLess(float(model.cbar[2, 0]), 0.0)
        self.assertTrue(model.metadata["sha256"])
        self.assertGreater(model.mu_m3_s2, 0.0)

    def test_mmax_honored(self):
        model = scenario_lunar_gravity_model(self._config(lunar_gravity_mmax=0))
        self.assertEqual(model.nmax, 3)
        self.assertEqual(model.mmax, 0)

    # 21 -- relative path resolves via resolve_gravity_dir --------------------------
    def test_relative_path_resolves_via_gravity_dir(self):
        previous = os.environ.get("LUNAR_OD_GRAVITY_DIR")
        os.environ["LUNAR_OD_GRAVITY_DIR"] = str(FIXTURE.parent)
        try:
            model = scenario_lunar_gravity_model(
                self._config(lunar_gravity_model_path=FIXTURE.name)
            )
        finally:
            if previous is None:
                del os.environ["LUNAR_OD_GRAVITY_DIR"]
            else:
                os.environ["LUNAR_OD_GRAVITY_DIR"] = previous
        self.assertIsNotNone(model)
        self.assertEqual(model.nmax, 3)

    # 22 -- missing file: clear error naming both paths ------------------------------
    def test_missing_file_error(self):
        bad = str(FIXTURE.parent / "does_not_exist.tab")
        with self.assertRaisesRegex(FileNotFoundError, "does_not_exist.tab"):
            scenario_lunar_gravity_model(self._config(lunar_gravity_model_path=bad))

    # 23 -- builder repeats the mandatory-field errors --------------------------------
    def test_builder_requires_path_and_nmax(self):
        with self.assertRaisesRegex(ValueError, "requires lunar_gravity_model_path"):
            scenario_lunar_gravity_model(self._config(lunar_gravity_model_path=None))
        with self.assertRaisesRegex(ValueError, "explicit lunar_gravity_nmax"):
            scenario_lunar_gravity_model(self._config(lunar_gravity_nmax=None))

    # 24 -- post-load double-count mirror ----------------------------------------------
    def test_builder_double_count_mirror(self):
        with self.assertRaisesRegex(ValueError, "count J2 twice"):
            scenario_lunar_gravity_model(self._config(j2_moon=2.0330530e-4))


if __name__ == "__main__":
    unittest.main()
