"""Protect comparison fairness and the published trial identities."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from examples import benchmark_estimators as benchmark


def test_three_estimators_receive_identical_data_and_explicit_prior(monkeypatch, tmp_path):
    calls = []
    for name, sequential in (("estimate_position_bls_lm", False),
                             ("estimate_position_srif", False), ("run_lunar_ukf", True)):
        original = getattr(benchmark, name)

        def wrapped(*args, _original=original, _sequential=sequential, **kwargs):
            calls.append((args[0].copy(), args[1].copy(), args[2].copy(),
                          args[3].copy() if _sequential else kwargs["prior_covariance"].copy()))
            if _sequential:
                assert kwargs["process_noise"] is None
                assert kwargs["covariance_form"] == "square_root"
                assert "adaptive_config" not in kwargs
            return _original(*args, **kwargs)
        monkeypatch.setattr(benchmark, name, wrapped)
    plan = json.loads(benchmark.PLAN.read_text())
    base = json.loads(benchmark.CONFIG.read_text())
    rows = benchmark.run_trial(41000, 1., plan, base, tmp_path)
    assert len(calls) == 3
    for other in calls[1:]:
        for reference, received in zip(calls[0], other):
            np.testing.assert_array_equal(received, reference)
    assert len({row["input_sha256"] for row in rows}) == 1
    assert all(not row["failure"] for row in rows)
    assert all(row["observations"] == 124 for row in rows)
    with np.load(tmp_path / "seed-41000-noise-1.npz") as data:
        np.testing.assert_array_equal(data["prior_covariance"], np.diag(np.array(plan["initial_std_m_mps"]) ** 2))
        assert data["ukf_times_s"][-1] == 1800.
        assert data["ukf_states"].shape == (124, 6)
        for label in ("BLS_LM", "SRIF", "SR_UKF"):
            assert data[label + "_final_covariance"].shape == (6, 6)


def test_published_campaign_has_all_declared_trials_and_verified_raw_members():
    folder = Path(__file__).resolve().parents[1] / "docs/figures/benchmark"
    summary = json.loads((folder / "summary.json").read_text())
    rows = json.loads((folder / "trials.json").read_text())
    plan = summary["plan"]
    expected = {(seed, scale, label) for seed in plan["seeds"] for scale in plan["noise_scales"] for label in benchmark.ESTIMATORS}
    assert {(r["seed"], r["noise_scale"], r["estimator"]) for r in rows} == expected
    assert len(rows) == len(expected) == 192
    assert summary["complete"] and not summary["pilot"] and not summary["timed_out"]
    receipt = json.loads((folder / "published.json").read_text())
    for name, expected_hash in receipt["artifact_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected_hash
    with zipfile.ZipFile(folder / "raw-trials.zip") as archive:
        assert len(archive.namelist()) == 64
        for name in archive.namelist():
            assert hashlib.sha256(archive.read(name)).hexdigest() == summary["artifact_sha256"][name]
