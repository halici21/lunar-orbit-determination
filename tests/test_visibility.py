import json
import unittest
from pathlib import Path

import numpy as np

from lunar_od import VisibilityConfig, analyze_visibility_gap, analyze_visibility_gap_with_transforms
from lunar_od import calc_gst_curtis, is_occulted_by_moon, range_rate_stations


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class VisibilityTests(unittest.TestCase):
    def test_occultation_basic_cases(self):
        r_obs = np.array([-2.0, 0.0, 0.0])
        r_tgt_blocked = np.array([2.0, 0.0, 0.0])
        r_tgt_clear = np.array([2.0, 3.0, 0.0])
        r_body = np.array([0.0, 0.0, 0.0])

        self.assertTrue(bool(is_occulted_by_moon(r_obs, r_tgt_blocked, r_body, 1.0)[0]))
        self.assertFalse(bool(is_occulted_by_moon(r_obs, r_tgt_clear, r_body, 1.0)[0]))

    def test_visibility_against_matlab_fixture(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        initial = fixture["initial_state"]
        truth = fixture["truth_propagation"]
        visibility = fixture["visibility"]

        t_eval_s = np.asarray(visibility["t_eval_s"], dtype=float)
        state_history = np.asarray(truth["state_history_mci_m_mps"], dtype=float)
        t_ephem_s = np.asarray(truth["t_ephem_s"], dtype=float)
        earth_pos_grid_m = np.asarray(truth["earth_pos_grid_m"], dtype=float)

        def earth_position(t_s):
            from scipy.interpolate import PchipInterpolator

            return PchipInterpolator(t_ephem_s, earth_pos_grid_m, axis=0)(t_s)

        config = VisibilityConfig(
            r_moon_mean_m=initial["r_moon_mean_m"],
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=fixture["epoch_utc"],
            min_elevation_deg=visibility["min_elevation_deg"],
        )

        stations_by_name = {station.name: station for station in range_rate_stations()}

        single_stations = [stations_by_name[name] for name in visibility["single_station_names"]]
        self._assert_visibility_case(
            t_eval_s,
            state_history,
            single_stations,
            earth_position,
            visibility["max_gap_s"],
            config,
            visibility["single_seg_starts_1based"],
            visibility["single_seg_ends_1based"],
            visibility["single_vis_mask_raw"],
            visibility["single_net_vis_filled"],
        )

        multi_stations = [stations_by_name[name] for name in visibility["multi_station_names"]]
        self._assert_visibility_case(
            t_eval_s,
            state_history,
            multi_stations,
            earth_position,
            visibility["max_gap_s"],
            config,
            visibility["multi_seg_starts_1based"],
            visibility["multi_seg_ends_1based"],
            visibility["multi_vis_mask_raw"],
            visibility["multi_net_vis_filled"],
        )

    def test_transform_visibility_matches_gst_when_using_same_rotation(self):
        fixture_path = FIXTURES_DIR / "spice_snapshots.json"
        if not fixture_path.is_file():
            self.skipTest("MATLAB spice_snapshots.json fixture has not been exported yet.")

        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        initial = fixture["initial_state"]
        truth = fixture["truth_propagation"]
        visibility = fixture["visibility"]

        t_eval_s = np.asarray(visibility["t_eval_s"], dtype=float)
        state_history = np.asarray(truth["state_history_mci_m_mps"], dtype=float)
        t_ephem_s = np.asarray(truth["t_ephem_s"], dtype=float)
        earth_pos_grid_m = np.asarray(truth["earth_pos_grid_m"], dtype=float)

        def earth_position(t_s):
            from scipy.interpolate import PchipInterpolator

            return PchipInterpolator(t_ephem_s, earth_pos_grid_m, axis=0)(t_s)

        config = VisibilityConfig(
            r_moon_mean_m=initial["r_moon_mean_m"],
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=fixture["epoch_utc"],
            min_elevation_deg=visibility["min_elevation_deg"],
        )

        xforms = _gst_position_transforms(t_eval_s, config)
        stations_by_name = {station.name: station for station in range_rate_stations()}
        stations = [stations_by_name[name] for name in visibility["multi_station_names"]]

        seg_starts_gst, seg_ends_gst, vis_gst, net_gst = analyze_visibility_gap(
            t_eval_s,
            state_history,
            stations,
            earth_position,
            visibility["max_gap_s"],
            config,
        )
        seg_starts_xf, seg_ends_xf, vis_xf, net_xf = analyze_visibility_gap_with_transforms(
            t_eval_s,
            state_history,
            stations,
            earth_position,
            xforms,
            visibility["max_gap_s"],
            config,
        )

        np.testing.assert_array_equal(seg_starts_xf, seg_starts_gst)
        np.testing.assert_array_equal(seg_ends_xf, seg_ends_gst)
        np.testing.assert_array_equal(vis_xf, vis_gst)
        np.testing.assert_array_equal(net_xf, net_gst)

    def _assert_visibility_case(
        self,
        t_eval_s,
        state_history,
        stations,
        earth_position,
        max_gap_s,
        config,
        expected_seg_starts_1based,
        expected_seg_ends_1based,
        expected_vis_mask_raw,
        expected_net_vis_filled,
    ):
        seg_starts, seg_ends, vis_mask_raw, net_vis_filled = analyze_visibility_gap(
            t_eval_s,
            state_history,
            stations,
            earth_position,
            max_gap_s,
            config,
        )

        expected_seg_starts = np.asarray(expected_seg_starts_1based, dtype=int) - 1
        expected_seg_ends = np.asarray(expected_seg_ends_1based, dtype=int) - 1
        np.testing.assert_array_equal(seg_starts, expected_seg_starts)
        np.testing.assert_array_equal(seg_ends, expected_seg_ends)
        expected_vis = np.asarray(expected_vis_mask_raw, dtype=bool)
        if expected_vis.ndim == 1:
            expected_vis = expected_vis.reshape(-1, 1)
        np.testing.assert_array_equal(vis_mask_raw, expected_vis)
        np.testing.assert_array_equal(net_vis_filled, np.asarray(expected_net_vis_filled, dtype=bool))


def _gst_position_transforms(t_eval_s, config):
    from datetime import datetime

    dt = datetime.strptime(config.epoch_utc, "%Y-%m-%d %H:%M:%S")
    gst0_rad = float(calc_gst_curtis(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
    gst_vec = gst0_rad + config.earth_rotation_rad_s * np.asarray(t_eval_s, dtype=float)
    xforms = np.zeros((len(t_eval_s), 3, 3), dtype=float)
    for idx, theta in enumerate(gst_vec):
        c = np.cos(theta)
        s = np.sin(theta)
        xforms[idx, :, :] = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return xforms


if __name__ == "__main__":
    unittest.main()
