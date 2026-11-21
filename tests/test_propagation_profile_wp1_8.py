"""WP1.8 (DYN-14, DYN-15, X-04): shared propagation profile and integrator fidelity.

* Truth and estimator profiles are compared field by field; every scenario's
  numerical-profile manifest records whether they shared one numerical path,
  and a comparison can be refused unless they did (or the caller says why not).
* DYN-14's single-``atol`` concern is measured, not assumed: with the default
  rtol the scalar atol leaves every STM block at round-off level.
* DYN-15: the Numba Pines kernels (``fastmath=True``) match the pure-Python
  reference at N = 300 with a real field.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_od import dynamics as dyn
from lunar_od.dynamics import PropagationProfile
from lunar_od.propagation_contract import (
    SHARED_PROFILE_FIELDS,
    numerical_metadata,
    numerical_profile_differences,
    require_shared_numerical_profile,
    truth_numerical_metadata,
)
from lunar_od.reporting import write_numerical_profile_manifest


def test_default_truth_and_batch_estimator_do_not_share_a_profile():
    """The historical defaults: truth DOP853 at caller tolerances, batch VODE
    Adams with adaptive estimator tolerances (X-04)."""
    truth = truth_numerical_metadata(None, None, 1e-11, 1e-12)
    batch = numerical_metadata(None, None, 1e-11, 1e-12)
    assert numerical_profile_differences(truth, batch) == {
        "method": ("DOP853", "ADAMS"),
        "error_control": ("fixed_caller_tolerances", "legacy_adaptive_estimator_tolerances"),
    }
    with pytest.raises(ValueError, match="different numerical profiles"):
        require_shared_numerical_profile(truth, batch)
    assert require_shared_numerical_profile(truth, batch, mismatch_reason="legacy default")


def test_one_explicit_profile_makes_truth_and_estimators_share_the_path():
    profile = PropagationProfile()
    truth = truth_numerical_metadata(None, profile, 1e-11, 1e-12)
    batch = numerical_metadata(None, profile, 1e-11, 1e-12)
    assert require_shared_numerical_profile(truth, batch) == [{}]
    # The UKF fast sigma propagator is a different path even under one profile.
    ukf = numerical_metadata(None, profile, 1e-11, 1e-12, fast_sigma=True)
    assert set(numerical_profile_differences(truth, ukf)) == {"method", "error_control"}
    assert numerical_profile_differences(truth, None).keys() == set(SHARED_PROFILE_FIELDS)
    with pytest.raises(ValueError):
        require_shared_numerical_profile(truth, batch, ukf, mismatch_reason="  ")


def test_numerical_manifest_records_profile_parity(tmp_path):
    truth = truth_numerical_metadata(None, None, 1e-11, 1e-12)
    scenario = SimpleNamespace(force_contract_manifest_sha256="sha256:x",
                               numerical_profile=numerical_metadata(None, None, 1e-11, 1e-12))
    path = write_numerical_profile_manifest(truth, scenario, tmp_path / "default.json")
    manifest = json.loads(path.read_text())
    assert manifest["shared_profile"] is False
    assert manifest["profile_differences"]["method"] == ["DOP853", "ADAMS"]
    profile = PropagationProfile()
    scenario.numerical_profile = numerical_metadata(None, profile, 1e-11, 1e-12)
    path = write_numerical_profile_manifest(truth_numerical_metadata(None, profile, 1e-11, 1e-12),
                                            scenario, tmp_path / "shared.json")
    manifest = json.loads(path.read_text())
    assert manifest["shared_profile"] is True and manifest["profile_differences"] == {}


def _far(position):
    return lambda t: np.tile(position, (np.size(np.asarray(t)), 1))


def test_dyn14_scalar_atol_keeps_every_stm_block_at_round_off():
    """Oracle: the same propagation at atol = 1e-22 (pure relative control)."""
    earth, sun = _far(np.array([384400e3, 0.0, 0.0])), _far(np.array([149.6e9, 0.0, 0.0]))
    x0 = np.array([1.8374e6, 0.0, 2.0e5, 0.0, 1.6e3, 100.0])
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    t = np.array([0.0, 8481.0])

    def stm(atol):
        y = np.asarray(dyn.propagate_augmented_state(t, aug0, 4902.800066e9, 0.0, 0.0, earth, sun,
                                                     rtol=1e-11, atol=atol, method="DOP853"))
        return y[-1, 6:].reshape(6, 6, order="F")

    got, ref = stm(1e-12), stm(1e-22)
    for rows in (slice(0, 3), slice(3, 6)):
        for cols in (slice(0, 3), slice(3, 6)):
            block = np.abs(ref[rows, cols]).max()
            # measured 2.5e-12 .. 5.0e-12, including dv/dr whose entries are ~1e-2
            assert np.abs(got[rows, cols] - ref[rows, cols]).max() / block < 1e-10


def test_dyn15_numba_pines_matches_python_at_degree_300():
    from lunar_od import accelerated as acc
    from lunar_od import gravity_harmonics as gh
    from lunar_od.gravity_model_loader import load_lunar_gravity_model
    from lunar_od.project_paths import gravity_data_root

    if not acc.NUMBA_AVAILABLE:
        pytest.skip("Numba not available")
    path = gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"
    if not path.is_file():
        pytest.skip("GL1800F gravity field not installed")
    m = load_lunar_gravity_model(str(path), nmax=300, mmax=300)
    args = (m.mu_m3_s2, m.r_ref_m, m.cbar, m.sbar, m.nmax, m.mmax)
    rng = np.random.default_rng(20261102)
    for altitude_m in (30e3, 100e3):
        u = rng.normal(size=3)
        r = u / np.linalg.norm(u) * (1737.4e3 + altitude_m)
        a_py = gh._pines_acceleration_bf(r, *args)
        a_nb = acc.pines_accel_bf_fast(r, *args)
        assert np.linalg.norm(a_nb - a_py) / np.linalg.norm(a_py) < 1e-13   # measured <= 1.2e-14
        g_py = np.asarray(gh._pines_gradient_bf(r, *args))
        g_nb = np.asarray(acc.pines_gradient_bf_fast(r, *args))
        assert np.linalg.norm(g_nb - g_py) / np.linalg.norm(g_py) < 1e-13
