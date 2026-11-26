"""F3 WP3.2 S3: exploratory gate set E_G3(a)-A (written after the E_G3(a) precondition FAIL; amendment log
2026-11-25).

Predeclaration: studies/phase17/configs/wp3_2_s3_earth_shadow_predeclaration.json. As predeclared, the oracle's
small-angle precondition gave 1.41e-8 against 1e-9, so E_G3(a) is FAIL as predeclared and stays FAIL.

Hypothesis, recorded with the failure: the oracle's quad uses the predeclared absolute tolerance epsabs = 1e-15 sr.
With every angle scaled by 1e-3 the solar cap is ~6.8e-11 sr, so quad may stop at ~1.5e-5 of the cap; at full
scale the cap is ~6.8e-5 sr and the same epsabs is ~1.5e-11 of it, so the gated value does not depend on it.

E_G3(a)-A, on the same samples and the same 50 precondition samples:
- A1: the oracle with a relative absolute tolerance, epsabs = 1e-13 x the cap's solid angle (breakpoints, epsrel
  and limit unchanged), reproduces the planar lens in the 1e-3-scaled limit to <= 1e-9;
- A2: that oracle against production on the unscaled samples: max |nu_prod - nu_ref| <= 2.5e-4;
- A3: the predeclared oracle and that oracle agree on the unscaled samples to <= 1e-9.

usage: python wp3_2_s3_eg3a_oracle_diagnostic.py OUT.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT, ROOT / "studies" / "phase17" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import lunar_od.earth_shadow as es  # noqa: E402
from tests import test_wp3_2_earth_shadow as M  # noqa: E402


def spherical_fraction_relative(a_s, a_o, gamma):
    """The predeclared oracle (M.spherical_fraction) with epsabs relative to the cap's solid angle."""
    if gamma >= a_s + a_o:
        return 1.0
    cap = 4.0 * math.pi * math.sin(0.5 * a_s) ** 2
    points = [p for p in (abs(gamma - a_o), gamma + a_o) if 0.0 < p < a_s]
    hidden, _ = quad(lambda rho: math.sin(rho) * M._hidden_arc(rho, a_o, gamma), 0.0, a_s,
                     points=points or None, epsabs=1e-13 * cap, epsrel=1e-13, limit=400)
    return 1.0 - hidden / cap


def run():
    r, s, e = M.make_samples()
    a1 = a2 = a3 = 0.0
    a1_original = 0.0
    n_pre = n_pen = 0
    for i in range(r.shape[0]):
        nu = es.earth_illumination(r[i], s[i], e[i], M.ES)
        if float(np.dot(r[i] - e[i], s[i] - e[i])) > 0.0:
            continue
        a_s, a_o, gamma = M.oracle_angles(r[i], s[i], e[i], M.ES.radius_m)
        ref = spherical_fraction_relative(a_s, a_o, gamma)
        a2 = max(a2, abs(nu - ref))
        a3 = max(a3, abs(M.spherical_fraction(a_s, a_o, gamma) - ref))
        if 0.0 < nu < 1.0:
            n_pen += 1
            if n_pre < 50:
                k = 1e-3
                planar = M._planar(k * a_s, k * a_o, k * gamma)
                a1 = max(a1, abs(spherical_fraction_relative(k * a_s, k * a_o, k * gamma) - planar))
                a1_original = max(a1_original, abs(M.spherical_fraction(k * a_s, k * a_o, k * gamma) - planar))
                n_pre += 1
    return dict(A1_precondition_relative_tol=a1, A1_threshold=1e-9, A1_pass=a1 <= 1e-9,
                predeclared_oracle_precondition=a1_original, n_precondition=n_pre,
                A2_worst=a2, A2_threshold=2.5e-4, A2_pass=a2 <= 2.5e-4,
                A3_oracles_agree=a3, A3_threshold=1e-9, A3_pass=a3 <= 1e-9, n_penumbra=n_pen)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    result = dict(study="F3 WP3.2 S3 E_G3(a)-A (exploratory)", commit=commit, status=status, **run())
    print(json.dumps(result, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
