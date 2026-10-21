"""Phase 17-FE-P - the GL1800F uncertainty prior is mapped and applied as predeclared.

The prior functions of studies/phase17/scripts/phase17_fe_prior.py are executed alone (no campaign imports): the sigma
parse must reproduce the predeclared spot values, and the assembled whitened design must carry exactly
diag(1 / sigma^2) on the field-correction block of its normal matrix and nothing else.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from lunar_od.project_paths import gravity_data_root

ROOT = Path(__file__).resolve().parents[1]
GL = gravity_data_root() / "gl1800f" / "jggrx_1800f_sha.tab"
SRC = (ROOT / "studies" / "phase17" / "scripts" / "phase17_fe_prior.py").read_text(encoding="utf-8")


def _functions(*names, **globs):
    ns = dict(np=np, **globs)
    for name in names:
        start = SRC.index(f"def {name}(")
        end = SRC.index("\ndef ", start + 1)
        exec(SRC[start:end], ns)
    return ns


def _coeffs(nmax):
    out = []
    for n in range(2, nmax + 1):
        out += [("C", n, m) for m in range(0, n + 1)]
        out += [("S", n, m) for m in range(1, n + 1)]
    return out


@pytest.mark.skipif(not GL.exists(), reason="GL1800F coefficient file not available")
def test_sigma_mapping_matches_the_predeclared_spot_values():
    ns = _functions("parse_gl1800f", "sigma_vector", GL_TAB=GL)
    coeffs = _coeffs(6)
    sig = ns["sigma_vector"](coeffs, ns["parse_gl1800f"](6))
    assert sig.shape == (45,) and np.all(sig > 0)
    assert sig[coeffs.index(("C", 2, 0))] == 4.593682635805e-12
    assert sig[coeffs.index(("C", 2, 1))] == 1.303864780869e-12
    assert sig[coeffs.index(("S", 2, 1))] == 1.390771628190e-12


def test_prior_rows_add_exactly_inverse_variance_on_the_field_block():
    k2 = types.SimpleNamespace(PRIOR_SD=np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0]))
    ns = _functions("_assemble", K2=k2)
    rng = np.random.default_rng(7)
    coeffs = _coeffs(6)
    d = dict(hx=rng.normal(size=(80, 6)), hk3=rng.normal(size=(80, 3)), nu=rng.normal(size=(80, 4)),
             h_fe=rng.normal(size=(80, 45)), coeffs=np.array(coeffs, dtype=object))
    sig = rng.uniform(0.5, 2.0, 45)
    for n_deg, n_fe in ((4, 21), (6, 45)):
        a, n_data, fe_idx, sel = ns["_assemble"](d, n_deg, sig)
        a0, _, _, _ = ns["_assemble"](d, n_deg, None)
        assert n_data == 80 and len(sel) == n_fe and a.shape[0] == 80 + 6 + n_fe
        expect = a0.T @ a0
        expect[np.ix_(fe_idx, fe_idx)] += np.diag(1.0 / sig[sel] ** 2)
        np.testing.assert_allclose(a.T @ a, expect, rtol=1e-13, atol=1e-12)
