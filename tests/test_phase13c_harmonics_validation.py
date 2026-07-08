"""Phase 13C -- campaign-script smoke tests (NOT the campaign itself).

Verifies only that the script imports cleanly (no .mat / SPICE side effects at
import time), the synthetic model set builds, the metric helpers work, and a
10-minute mini propagation through ``run_case`` runs with a SYNTHETIC rotation
grid and frozen third-body vectors.  The real MOON_PA campaign is executed by
running ``examples/phase13c_harmonics_validation.py`` directly.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from lunar_od.constants import MU_MOON_M3S2, R_MOON_M  # noqa: E402

import phase13c_harmonics_validation as p13c  # noqa: E402

R_ME = np.array([-83446893.0, 354010875.0, 178558253.0])
R_MS = np.array([1.40753701450e11, -4.21884124439e10, -1.82638284191e10])
GE = lambda t: R_ME    # noqa: E731
GS = lambda t: R_MS    # noqa: E731


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


class Phase13cSmokeTests(unittest.TestCase):
    # 1 -- import happened without side effects; model set builds ------------
    def test_models_build(self):
        models = p13c.build_models()
        self.assertEqual(set(models), {"M1", "M2", "M3", "M4", "M4_n3", "M4_n2", "M4_m0"})
        m1, m2, m4 = models["M1"], models["M2"], models["M4"]
        self.assertEqual((m1.nmax, m1.mmax), (2, 0))
        self.assertLess(m1.cbar[2, 0], 0.0)                    # Cbar20 = -J2/sqrt(5)
        self.assertEqual(m2.cbar[2, 2], 3.47e-5)
        self.assertEqual((m4.nmax, m4.mmax), (4, 4))
        self.assertEqual(models["M4_m0"].mmax, 0)
        for m in models.values():
            self.assertEqual(m.mu_m3_s2, MU_MOON_M3S2)         # NOT GRAIL 1738 km
            self.assertEqual(m.r_ref_m, R_MOON_M)

    # 2 -- metric helpers ------------------------------------------------------
    def test_diff_metrics(self):
        base = np.zeros((5, 6))
        traj = base.copy()
        traj[:, 0] = [0.0, 1.0, 2.0, 2.0, 1.0]
        m = p13c.diff_metrics(traj, base)
        self.assertEqual(m["final_dpos_m"], 1.0)
        self.assertEqual(m["max_dpos_m"], 2.0)
        self.assertGreater(m["rms_dpos_m"], 0.0)
        self.assertEqual(m["final_dvel_mps"], 0.0)

    def test_classify_ratio(self):
        self.assertIn("ACCEPT", p13c.classify_ratio(1e-3))
        self.assertIn("MARGINAL", p13c.classify_ratio(0.05))
        self.assertIn("REJECT", p13c.classify_ratio(0.5))

    # 3 -- 10-minute mini propagation through run_case (synthetic grid) --------
    def test_mini_propagation_run_case(self):
        models = p13c.build_models()
        teval = np.arange(0.0, 600.0 + 1, 60.0)
        rate = 2.0 * math.pi / (27.321661 * 86400.0)
        t_grid = np.arange(-120.0, 600.0 + 120.0 + 30.0, 60.0)
        grid = np.stack([_rotz(rate * t) for t in t_grid])
        s0 = p13c.polar_state()
        res = p13c.run_case(GE, GS, s0, teval,
                            harmonic_model=models["M2"],
                            harmonic_rotation=(t_grid, grid))
        self.assertEqual(res["traj"].shape, (teval.size, 6))
        self.assertTrue(bool(np.all(np.isfinite(res["traj"]))))
        self.assertGreater(res["rhs_evals"], 0)
        self.assertGreater(res["runtime_s"], 0.0)
        self.assertFalse(res["surface_crossing"])
        self.assertGreater(res["min_altitude_m"], 0.0)

    # 4 -- rhs-eval counter counts ----------------------------------------------
    def test_counting_getter(self):
        g = p13c._CountingGetter(GE)
        for _ in range(5):
            g(0.0)
        self.assertEqual(g.calls, 5)


if __name__ == "__main__":
    unittest.main()
