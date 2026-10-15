"""Check wheel imports and a closed-form circular orbit outside the checkout."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import pkgutil
import sys
import numpy as np
import lunar_od


def main():
    package = Path(lunar_od.__file__).resolve()
    if "site-packages" not in package.parts:
        raise RuntimeError(f"Expected an installed wheel, got {package}")
    modules = [item.name for item in pkgutil.walk_packages(lunar_od.__path__, "lunar_od.")]
    for name in modules:
        importlib.import_module(name)
    mu, radius, duration = 4.902800066e12, 1.8374e6, 300.0
    speed = np.sqrt(mu / radius)
    angle = speed / radius * duration
    state = np.array([radius, 0, 0, 0, speed, 0])
    initial = np.r_[state, np.eye(6).reshape(-1, order="F")]
    body = lambda t: np.tile([384400000., 0., 0.], (np.size(t), 1))
    history = lunar_od.propagate_augmented_state(
        [0., duration], initial, mu, 0., 0., body, body, rtol=1e-11, atol=1e-12)
    expected = [radius * np.cos(angle), radius * np.sin(angle), 0.,
                -speed * np.sin(angle), speed * np.cos(angle), 0.]
    # Analytic two-body circular orbit: 1 cm position / 0.1 mm/s velocity bounds.
    np.testing.assert_allclose(history[-1, :3], expected[:3], rtol=0, atol=.01)
    np.testing.assert_allclose(history[-1, 3:6], expected[3:], rtol=0, atol=1e-4)
    print(json.dumps({"python": sys.version, "package": str(package),
                      "imported_modules": len(modules) + 1, "analytic_orbit": "passed"}))


if __name__ == "__main__":
    main()
