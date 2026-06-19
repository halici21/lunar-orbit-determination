"""Compare GST visibility against SPICE ITRF93-transform visibility.

Run from the project root:

    python python_port/examples/compare_visibility_models.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    VisibilityConfig,
    analyze_visibility_gap,
    analyze_visibility_gap_with_transforms,
    load_spice_kernels,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
)


def main() -> None:
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]

    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

    duration_h = 12.0
    sample_step_s = 120.0
    ephem_step_s = 600.0
    max_gap_s = 20.0 * 60.0
    min_elevation_deg = 5.0

    t_eval_s = np.arange(0.0, duration_h * 3600.0 + sample_step_s, sample_step_s)
    t_ephem_s = np.arange(0.0, duration_h * 3600.0 + ephem_step_s, ephem_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        xforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
    finally:
        spice.kclear()

    print(f"Propagating {duration_h:.1f} h truth trajectory with {t_eval_s.size} samples...")
    state_history = propagate_truth_with_ephemeris(
        t_eval_s,
        x0_mci,
        mu_moon,
        mu_earth,
        mu_sun,
        ephemeris,
        rtol=1e-10,
        atol=1e-11,
    )

    config = VisibilityConfig(
        r_moon_mean_m=float(initial["r_moon_mean_m"]),
        earth_rotation_rad_s=7.292115e-5,
        epoch_utc=epoch_utc,
        min_elevation_deg=min_elevation_deg,
    )

    station_names = ["Goldstone DSN", "Madrid DSN", "Canberra DSN", "ITU Ayazaga"]
    stations_by_name = {station.name: station for station in range_rate_stations()}
    stations = [stations_by_name[name] for name in station_names]

    gst = analyze_visibility_gap(
        t_eval_s,
        state_history,
        stations,
        ephemeris.earth_position,
        max_gap_s,
        config,
    )
    spice_vis = analyze_visibility_gap_with_transforms(
        t_eval_s,
        state_history,
        stations,
        ephemeris.earth_position,
        xforms,
        max_gap_s,
        config,
    )

    out_dir = Path("python_port") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "visibility_gst_vs_spice_summary.csv"
    png_path = out_dir / "visibility_gst_vs_spice_comparison.png"

    _write_comparison_csv(csv_path, station_names, t_eval_s, gst, spice_vis)
    _plot_comparison(png_path, station_names, t_eval_s, gst, spice_vis)

    diff_count = int(np.sum(gst[2] != spice_vis[2]))
    net_diff_count = int(np.sum(gst[3] != spice_vis[3]))
    print(f"Station mask differing samples: {diff_count}")
    print(f"Filled network differing samples: {net_diff_count}")
    print(f"GST arcs={len(gst[0])}, SPICE arcs={len(spice_vis[0])}")
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


def _write_comparison_csv(output_path, station_names, t_s, gst, spice_vis):
    seg_gst, end_gst, vis_gst, net_gst = gst
    seg_spice, end_spice, vis_spice, net_spice = spice_vis
    raw_gst = np.any(vis_gst, axis=1)
    raw_spice = np.any(vis_spice, axis=1)

    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "name", "value"])
        writer.writerow(["duration_s", "scenario", float(t_s[-1] - t_s[0])])
        writer.writerow(["sample_step_s", "scenario", float(t_s[1] - t_s[0])])
        writer.writerow(["num_samples", "scenario", int(t_s.size)])
        writer.writerow(["station_mask_diff_samples", "all_stations", int(np.sum(vis_gst != vis_spice))])
        writer.writerow(["raw_network_diff_samples", "network", int(np.sum(raw_gst != raw_spice))])
        writer.writerow(["filled_network_diff_samples", "network", int(np.sum(net_gst != net_spice))])
        writer.writerow(["gst_num_arcs", "network", int(seg_gst.size)])
        writer.writerow(["spice_num_arcs", "network", int(seg_spice.size)])

        for idx, station_name in enumerate(station_names):
            writer.writerow(["station_diff_samples", station_name, int(np.sum(vis_gst[:, idx] != vis_spice[:, idx]))])
            writer.writerow(["station_gst_visible_fraction", station_name, float(np.mean(vis_gst[:, idx]))])
            writer.writerow(["station_spice_visible_fraction", station_name, float(np.mean(vis_spice[:, idx]))])

        writer.writerow([])
        writer.writerow(["model", "arc_id", "start_s", "end_s", "duration_s"])
        for model, starts, ends in (("gst", seg_gst, end_gst), ("spice", seg_spice, end_spice)):
            for arc_id, (start_idx, end_idx) in enumerate(zip(starts, ends), start=1):
                writer.writerow([model, arc_id, float(t_s[start_idx]), float(t_s[end_idx]), float(t_s[end_idx] - t_s[start_idx])])


def _plot_comparison(output_path, station_names, t_s, gst, spice_vis):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    _, _, vis_gst, net_gst = gst
    _, _, vis_spice, net_spice = spice_vis
    raw_gst = np.any(vis_gst, axis=1)
    raw_spice = np.any(vis_spice, axis=1)
    t_h = np.asarray(t_s, dtype=float) / 3600.0

    diff_code = np.zeros_like(vis_gst, dtype=int)
    diff_code[vis_gst & ~vis_spice] = 1
    diff_code[~vis_gst & vis_spice] = 2

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 7.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]},
    )
    fig.suptitle("GST vs SPICE-ITRF Visibility Comparison", fontsize=13, fontweight="bold")

    ax_diff, ax_raw, ax_filled = axes
    extent = [float(t_h[0]), float(t_h[-1]), -0.5, len(station_names) - 0.5]
    ax_diff.imshow(
        diff_code.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#f8fafc", "#f59e0b", "#2563eb"]),
        vmin=0,
        vmax=2,
    )
    ax_diff.set_title("Station Visibility Differences: orange=GST only, blue=SPICE only")
    ax_diff.set_yticks(np.arange(len(station_names)))
    ax_diff.set_yticklabels(station_names)
    ax_diff.set_ylabel("Station")

    ax_raw.step(t_h, raw_gst.astype(int), where="post", color="#f59e0b", linewidth=1.7, label="GST raw network")
    ax_raw.step(t_h, raw_spice.astype(int), where="post", color="#2563eb", linewidth=1.4, linestyle="--", label="SPICE raw network")
    ax_raw.set_title("Raw Network Visibility")
    ax_raw.set_ylim(-0.08, 1.15)
    ax_raw.set_yticks([0, 1])
    ax_raw.set_ylabel("visible")
    ax_raw.legend(fontsize=8)
    ax_raw.grid(True, axis="x", alpha=0.25)

    ax_filled.step(t_h, net_gst.astype(int), where="post", color="#f59e0b", linewidth=1.7, label="GST filled network")
    ax_filled.step(t_h, net_spice.astype(int), where="post", color="#2563eb", linewidth=1.4, linestyle="--", label="SPICE filled network")
    ax_filled.set_title("Gap-Filled Network Visibility")
    ax_filled.set_ylim(-0.08, 1.15)
    ax_filled.set_yticks([0, 1])
    ax_filled.set_ylabel("visible")
    ax_filled.legend(fontsize=8)
    ax_filled.grid(True, axis="x", alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Time since epoch [h]")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
