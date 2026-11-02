"""WP1.6: a scenario JSON fully defines its run (CFG-02/CFG-04/CFG-05/CFG-10)."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

from lunar_od.run_provenance import RUN_PROVENANCE_SCHEMA, canonical_config_sha256
from lunar_od.scenario_config import (
    SCENARIO_CONFIG_FIELDS,
    load_scenario_config_json,
    scenario_config_from_mapping,
    scenario_config_schema,
    write_normalized_scenario_config,
)

_ROOT = Path(__file__).resolve().parents[1]
_PAYLOAD = {"name": "wp1_6", "measurement_type": "range_rate", "estimator_type": "bls_lm",
            "start_mode": "cold", "network": "multi"}


def _write(tmp_path, **fields):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({**_PAYLOAD, **fields}), encoding="utf-8")
    return path


def test_unknown_config_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario config field.*bls_lamda0"):
        load_scenario_config_json(_write(tmp_path, bls_lamda0=0.1))


def test_noise_requires_measurement_seed(tmp_path):
    with pytest.raises(ValueError, match="measurement_seed"):
        load_scenario_config_json(_write(tmp_path, noise=True))
    config = load_scenario_config_json(_write(tmp_path, noise=True, measurement_seed=7, cold_start_seed=11))
    assert (config.measurement_seed, config.cold_start_seed) == (7, 11)


@pytest.mark.parametrize("value", [-1, True, 1.5, "x"])
def test_invalid_seed_rejected(value):
    with pytest.raises(ValueError, match="measurement_seed"):
        scenario_config_from_mapping({**_PAYLOAD, "measurement_seed": value})


def test_normalized_config_round_trips_seeds(tmp_path):
    config = scenario_config_from_mapping({**_PAYLOAD, "noise": True, "measurement_seed": 3})
    path = write_normalized_scenario_config(config, tmp_path / "normalized.json")
    assert load_scenario_config_json(path) == config


def test_schema_lists_the_seed_fields():
    properties = scenario_config_schema()["properties"]
    assert {"measurement_seed", "cold_start_seed"} <= set(properties) <= SCENARIO_CONFIG_FIELDS


def test_config_hash_changes_with_seed():
    a = scenario_config_from_mapping({**_PAYLOAD, "measurement_seed": 1}).to_dict()
    b = scenario_config_from_mapping({**_PAYLOAD, "measurement_seed": 2}).to_dict()
    assert canonical_config_sha256(a) == canonical_config_sha256(dict(a))
    assert canonical_config_sha256(a) != canonical_config_sha256(b)


def _runner():
    spec = importlib.util.spec_from_file_location("wp1_6_runner", _ROOT / "examples" / "run_scenario_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_plot(scenarios, output_path, *, title=""):
    Path(output_path).write_bytes(b"\x89PNG\r\n")
    return Path(output_path)


def _run_cli(runner, tmp_path, name, **fields):
    out = tmp_path / name
    out.mkdir()
    cfg = _write(out, name="wp1_6_e2e", duration_h=1.0, sample_step_s=600.0, max_iter=3,
                 output_dir=str(out), **fields)
    with mock.patch.object(runner, "plot_scenario_comparison", side_effect=_fake_plot):
        assert runner.main([str(cfg)]) == 0
    with (out / "wp1_6_e2e_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    provenance = json.loads((out / "wp1_6_e2e_run_provenance.json").read_text(encoding="utf-8"))
    return row, provenance


_RESULT_COLUMNS = ("num_observations", "initial_position_error_m", "initial_velocity_error_mps",
                   "final_position_error_m", "final_velocity_error_mps", "final_cost", "iterations",
                   "truth_force_fingerprint", "force_contract_manifest_sha256")


def test_scenario_json_fully_defines_run(tmp_path):
    """Same JSON (noise on, seeded) -> identical results and identical provenance."""
    pytest.importorskip("spiceypy")
    runner = _runner()
    seeds = {"noise": True, "measurement_seed": 20261101, "cold_start_seed": 5}
    row_a, prov_a = _run_cli(runner, tmp_path, "a", **seeds)
    row_b, prov_b = _run_cli(runner, tmp_path, "b", **seeds)
    # Bit-identical estimates: the CSV carries repr-exact floats.
    for key in _RESULT_COLUMNS:
        assert row_a[key] == row_b[key], key

    assert prov_a["schema"] == RUN_PROVENANCE_SCHEMA
    assert prov_a["seeds"] == {"measurement_seed": 20261101, "measurement_noise": True, "cold_start_seed": 5}
    assert prov_a["scenario_config_sha256"] == canonical_config_sha256(prov_a["scenario_config"])
    assert prov_a["ephemeris"] == prov_b["ephemeris"]
    assert prov_a["ephemeris"]["truth"]["samples_sha256"]
    assert prov_a["spice_kernels"] and all(k["sha256"] for k in prov_a["spice_kernels"])
    assert prov_a["spice_kernels"] == prov_b["spice_kernels"]

    row_c, _ = _run_cli(runner, tmp_path, "c", **{**seeds, "measurement_seed": 20261102})
    assert row_a["final_cost"] != row_c["final_cost"]
