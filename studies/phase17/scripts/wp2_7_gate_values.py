"""WP2.7: the measured values behind gates G1-G11 (the tests assert them; this records them).

Uses the inputs of tests/test_wp2_7_visibility.py. Run from the worktree after the change.
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
import test_wp2_7_visibility as T  # noqa: E402
from lunar_od.geometry import ecef2razel_sez  # noqa: E402
from lunar_od.visibility import (_bridge_network_gaps, _segments_from_mask,  # noqa: E402
                                 analyze_visibility_gap_with_transforms, sample_j2000_to_itrf93_transforms)


def main():
    out = {"arc": "2027-03-02T00:00 TDB + 1 d, 10 s, 100 km polar lunar orbit, DSN complexes, 10 deg mask"}
    worst = {"python": 0.0, "numba": 0.0}
    for seed in (7, 8):
        rows = T.g1_directions(seed)
        vec = [T.sez_to_ecef_at_origin(*r) for r in rows]
        py, nb = T.azel_python(vec), T.azel_numba(vec)
        for (s, e, z), a, b in zip(rows, py[:, 1], nb[:, 1]):
            ref = T.ref_elevation(s, e, z)
            worst["python"] = max(worst["python"], abs(float(Decimal(float(a)) - ref)))
            worst["numba"] = max(worst["numba"], abs(float(Decimal(float(b)) - ref)))
    out["G1_max_elevation_error_rad"] = worst
    out["G2_west_of_north_azimuth"] = {name: [float(x) for x in fn(T.WEST_OF_NORTH)[:, 0]]
                                      for name, fn in (("python", T.azel_python), ("numba", T.azel_numba))}

    arc = T.build_arc()
    arc["mask"] = analyze_visibility_gap_with_transforms(arc["t"], arc["x"], arc["stations"], arc["earth"], arc["xf"],
                                                         1200.0, arc["config"])[2]
    rho = T.visible_rho_ecef(arc, arc["mask"])
    g3 = {}
    for st in arc["stations"]:
        vec = rho[st.name]
        old = np.array([T.old_python_azel(v, st.lat_rad, st.lon_rad) for v in vec])
        new = np.array([ecef2razel_sez(v, st.lat_rad, st.lon_rad) for v in vec])
        nb = T.azel_numba(vec, st.lat_rad, st.lon_rad)
        g3[st.name] = {"max_new_minus_old_rad": float(np.max(np.abs(new[:, 1] - old[:, 1]))),
                       "max_numba_minus_python_rad": float(np.max(np.abs(nb[:, 1] - new[:, 1]))),
                       "samples": int(len(vec))}
    out["G3_realistic_elevation"] = g3

    (t_a, vis_a), (t_b, vis_b) = T._synthetic()
    out["G4_segments_s"] = {
        "dense_then_coarse_1500s": [[float(t_a[i]), float(t_a[j])] for i, j in
                                    zip(*_segments_from_mask(_bridge_network_gaps(t_a, vis_a[:, None], 1200.0)))],
        "coarse_then_dense_600s": [[float(t_b[i]), float(t_b[j])] for i, j in
                                   zip(*_segments_from_mask(_bridge_network_gaps(t_b, vis_b[:, None], 1200.0)))]}

    t = arc["t"]
    g5 = {}
    for name, keep in (("full_10s", np.ones(t.size, dtype=bool)),
                       ("10s_then_120s", (t <= 21600.0) | (np.mod(t, 120.0) == 0.0)),
                       ("120s_then_10s", (t > 21600.0) | (np.mod(t, 120.0) == 0.0))):
        s, e, _, _ = analyze_visibility_gap_with_transforms(t[keep], arc["x"][keep], arc["stations"], arc["earth"],
                                                            arc["xf"][keep], 1200.0, arc["config"])
        g5[name] = {"segments": int(s.size), "starts_s": [float(x) for x in t[keep][s]],
                    "ends_s": [float(x) for x in t[keep][e]]}
    out["G5_segments"] = g5

    g9 = {}
    for mode in ("light_time_downlink", "light_time_two_way"):
        prod = analyze_visibility_gap_with_transforms(t, arc["x"], arc["stations"], arc["earth"], arc["xf"], 1200.0,
                                                      arc["config"], geometry=mode)[2]
        oracle = T._oracle_masks(arc, t, arc["x"], mode == "light_time_two_way")
        g9[mode] = {"samples_differing_from_oracle": int(np.sum(prod != oracle)),
                    "samples_differing_from_geometric": int(np.sum(prod != arc["mask"]))}
    out["G9_light_time_masks"] = g9

    shifts = {"occultation_down": [], "occultation_two_way": [], "set_down": [], "set_two_way": [],
              "rise_down": [], "rise_two_way": []}
    usable = total = 0
    from lunar_od.geometry import ecef2sez_dcm
    mask = arc["mask"]
    for j, st in enumerate(arc["stations"]):
        for k in np.where(mask[:-1, j] != mask[1:, j])[0]:
            if k < 2 or k > t.size - 3:
                continue
            total += 1
            rho_k = np.einsum("nij,nj->ni", arc["xf"][k:k + 2, :3, :3],
                              arc["x"][k:k + 2, :3] - arc["earth"](t[k:k + 2])) - st.r_ecef_m
            sez = np.einsum("ij,nj->ni", ecef2sez_dcm(st.lat_rad, st.lon_rad), rho_k)
            el = np.arctan2(sez[:, 2], np.hypot(sez[:, 0], sez[:, 1]))
            kind = "elevation" if (el[0] > T.MIN_EL) != (el[1] > T.MIN_EL) else "occultation"
            rising = bool(mask[k + 1, j])
            tw = 0.5 * (t[k] + t[k + 1]) + np.arange(-6.0, 6.0 + 1e-9, 0.02)
            xw, xfw = T.hermite(t, arc["x"], tw), sample_j2000_to_itrf93_transforms(arc["et0"], tw)
            flips = {m: T._flip_times(tw, analyze_visibility_gap_with_transforms(
                tw, xw, [st], arc["earth"], xfw, 1200.0, arc["config"], geometry=m)[2][:, 0])
                for m in ("geometric", "light_time_downlink", "light_time_two_way")}
            if any(v.size != 1 for v in flips.values()):
                continue
            usable += 1
            down = float(flips["light_time_downlink"][0] - flips["geometric"][0])
            two = float(flips["light_time_two_way"][0] - flips["geometric"][0])
            key = "occultation" if kind == "occultation" else ("rise" if rising else "set")
            shifts[f"{key}_down"].append(down)
            shifts[f"{key}_two_way"].append(two)
    out["G10_boundary_shift_s"] = {k: ({"n": len(v), "min": min(v), "max": max(v)} if v else None)
                                   for k, v in shifts.items()}
    out["G10_windows"] = {"usable": usable, "total": total}
    path = Path("studies/phase17/results/wp2_7_gate_values.json")
    path.write_text(json.dumps(out, indent=1))
    brief = dict(out)
    brief["G5_segments"] = {k: v["segments"] for k, v in g5.items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
