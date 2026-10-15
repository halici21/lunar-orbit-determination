"""Public tutorial contracts; same-model recovery is not external validation."""
import json
import numpy as np

from examples.quickstart_od import CONFIG, build_case, run


def test_quickstart_noise_is_seeded_and_shared():
    config = json.loads(CONFIG.read_text())
    first = build_case(config)[3]
    np.testing.assert_array_equal(first, build_case(config)[3])
    changed = build_case({**config, "seed": config["seed"] + 1})[3]
    assert not np.array_equal(first[:, 1:4], changed[:, 1:4])
    np.testing.assert_array_equal(first[:, [0, 4, 5, 6]], changed[:, [0, 4, 5, 6]])


def test_quickstart_recovers_known_state_without_measurement_noise(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["noise_enabled"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "run"
    result = run(path, output)
    # Known injected truth is the oracle. Metre / cm/s bounds are much smaller
    # than the 197 m / 0.071 m/s injected error and are generous to the solver's
    # stopping tolerance. They are tutorial recovery checks, not flight accuracy.
    for item in result["metrics"].values():
        assert item["initial_epoch_position_error_m"] < 1.0
        assert item["arc_end_position_error_m"] < 1.0
        assert item["arc_end_velocity_error_mps"] < 0.01
        assert item["rank"] == 6
        assert item["posterior_covariance_finite"]
    assert result["observations"] == 124
    assert result["scope"]["truth_level"] == 0
    for filename in result["artifact_sha256"]:
        assert (output / filename).is_file()
