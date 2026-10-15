"""Phase 17-MM - the mismatch truth field is exactly a_GL + alpha (a_GR - a_GL).

The masquerade experiment (examples/phase17_mm_experiment.py) builds its truth field by mixing two GRAIL
solutions with different GM in GL1800F's normalisation; the engine's perturbing acceleration is linear in
mu R^n C_nm, so the rescaled mixture must reproduce the acceleration mixture exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_od.gravity_harmonics import spherical_harmonic_acceleration
from lunar_od.gravity_model_loader import load_lunar_gravity_model
from lunar_od.project_paths import gravity_data_root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
GL = gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"
GR = gravity_data_root() / "grgm1200l" / "gggrx_1200l_sha.tab"


def _mixed_field():
    src = (ROOT / "examples" / "phase17_mm_experiment.py").read_text(encoding="utf-8")
    ns = {}
    start = src.index("def mixed_field(")
    end = src.index("\ndef _fields(")
    exec("import dataclasses\nimport numpy as np\n" + src[start:end], ns)   # the function alone, no campaign imports
    return ns["mixed_field"]


@pytest.mark.skipif(not (GL.exists() and GR.exists()), reason="GRAIL coefficient files not available")
def test_mixed_truth_field_is_the_acceleration_mixture():
    gl = load_lunar_gravity_model(str(GL), nmax=8, mmax=8)
    gr = load_lunar_gravity_model(str(GR), nmax=8, mmax=8)
    mixed_field = _mixed_field()
    c = np.eye(3)
    rng = np.random.default_rng(3)
    for _ in range(4):
        r = rng.normal(size=3)
        r *= (1737.4e3 + 100e3) / np.linalg.norm(r)
        a_gl = spherical_harmonic_acceleration(r, gl, c)
        a_gr = spherical_harmonic_acceleration(r, gr, c)
        for alpha in (0.0, 0.3, 1.0):
            a = spherical_harmonic_acceleration(r, mixed_field(alpha, gl, gr), c)
            assert np.linalg.norm(a - (a_gl + alpha * (a_gr - a_gl))) <= 1e-9 * np.linalg.norm(a_gr - a_gl)
    m = mixed_field(0.3, gl, gr)
    assert m.mu_m3_s2 == gl.mu_m3_s2 and m.r_ref_m == gl.r_ref_m and m.frame == gl.frame


def test_predeclaration_freezes_the_decision_rule():
    pre = json.loads((ROOT / "artifacts" / "phase17_mm_predeclaration.json").read_text(encoding="utf-8"))
    assert pre["design"]["alpha"] == [1.0, 0.3]
    assert [c[:2] for c in pre["design"]["cells"]] == [["CANON", 0], ["O1_LOW_BETA_ECLIPSE_RICH", 12],
                                                       ["O2_FULL_SUN_HIGH_BETA", 0], ["O5_FAVOURABLE_GEO_MAX_FPERP", 0]]
    assert "13.82" in pre["thresholds"]["detect"] and "2 sqrt(2N)" in pre["thresholds"]["undetectable_residual"]
