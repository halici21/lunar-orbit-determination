"""WP2.7: record the sha256 "before" values of tests/test_wp2_7_visibility.py from the functions BEFORE the change.

Run once from the worktree on the pre-change tree (f754ab2); writes tests/fixtures/wp2_7_visibility_before.json.
Inputs are built with the test module's own helpers, so the test recomputes exactly the same arrays.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_wp2_7_visibility as T  # noqa: E402
from lunar_od.visibility import analyze_visibility_gap, analyze_visibility_gap_with_transforms  # noqa: E402


def main():
    out = {"recorded_on": "f754ab2 (pre-change tree)", "g1_directions": {}, "west_of_north": {}, "cardinals": {},
           "realistic": {}, "transforms": {}, "gst": {}}
    vectors = [T.sez_to_ecef_at_origin(*r) for seed in (7, 8) for r in T.g1_directions(seed)]
    for name, fn in (("python", T.azel_python), ("numba", T.azel_numba)):
        g1 = fn(vectors)
        out["g1_directions"][name] = {"az": T.sha(g1[:, 0]), "range": T.sha(g1[:, 2])}
        west = fn(T.WEST_OF_NORTH)
        out["west_of_north"][name] = {"returned_two_pi": [bool(x == T.TWO_PI) for x in west[:, 0]],
                                      "az_hex": [float(x).hex() for x in west[:, 0]]}
        out["cardinals"][name] = [float(x).hex() for x in fn(list(T.CARDINALS.values()))[:, 0]]

    a = T.build_arc()
    mask = analyze_visibility_gap_with_transforms(a["t"], a["x"], a["stations"], a["earth"], a["xf"], 1200.0,
                                                  a["config"])[2]
    rho = T.visible_rho_ecef(a, mask)
    from lunar_od.geometry import ecef2razel_sez
    for st in a["stations"]:
        vec = rho[st.name]
        py = np.array([ecef2razel_sez(v, st.lat_rad, st.lon_rad) for v in vec])
        old = np.array([T.old_python_azel(v, st.lat_rad, st.lon_rad) for v in vec])
        assert np.array_equal(py, old), "the test reimplementation must be the pre-change body bitwise"
        nb = T.azel_numba(vec, st.lat_rad, st.lon_rad)
        out["realistic"][st.name] = {name: {"az": T.sha(o[:, 0]), "el": T.sha(o[:, 1]), "range": T.sha(o[:, 2])}
                                     for name, o in (("python", py), ("numba", nb))}
        out["realistic"][st.name]["samples"] = int(vec.shape[0])

    for name, (t, x, xf) in T.uniform_grids(a).items():
        for max_gap in T.MAX_GAPS:
            res = analyze_visibility_gap_with_transforms(t, x, a["stations"], a["earth"], xf, max_gap, a["config"])
            out["transforms"][f"{name}/{max_gap:g}"] = [T.sha(res[0], np.int64), T.sha(res[1], np.int64),
                                                        T.sha(res[2], np.uint8), T.sha(res[3], np.uint8)]
            if name != "linspace":
                res = analyze_visibility_gap(t, x, a["stations"], a["earth"], max_gap, a["config"])
                out["gst"][f"{name}/{max_gap:g}"] = [T.sha(res[0], np.int64), T.sha(res[1], np.int64),
                                                     T.sha(res[2], np.uint8), T.sha(res[3], np.uint8)]
    T.BEFORE_PATH.write_text(json.dumps(out, indent=1))
    print({k: (len(v) if isinstance(v, dict) else v) for k, v in out.items()})
    print({st: v["samples"] for st, v in out["realistic"].items()})


if __name__ == "__main__":
    main()
