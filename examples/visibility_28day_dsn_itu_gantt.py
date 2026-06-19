"""Generate a 28-day SPICE visibility Gantt chart for the DSN + ITU network."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    VisibilityConfig,
    analyze_visibility_gap_with_transforms,
    load_spice_kernels,
    propagate_truth_with_ephemeris,
    range_rate_stations,
    sample_j2000_to_itrf93_transforms,
    sample_moon_centered_ephemeris,
)


def _intervals(mask: np.ndarray, step_s: float) -> list[tuple[float, float]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (start * step_s / 86400.0, (stop - start) * step_s / 86400.0)
        for start, stop in zip(starts, stops)
    ]


def main() -> None:
    fixture_path = Path("python_port") / "fixtures" / "spice_snapshots.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    initial = fixture["initial_state"]
    constants = fixture["constants"]
    epoch_utc = fixture["epoch_utc"]

    x0_mci = np.asarray(initial["state_mci_j2000_m_mps"], dtype=float)
    mu_moon = float(initial["mu_moon_m3_s2"])
    mu_earth = float(np.asarray(constants["mu_earth_km3_s2"]).reshape(-1)[0] * 1e9)
    mu_sun = float(np.asarray(constants["mu_sun_km3_s2"]).reshape(-1)[0] * 1e9)

    duration_days = 28.0
    sample_step_s = 600.0
    ephem_step_s = 3600.0
    max_gap_s = 1800.0
    t_eval_s = np.arange(0.0, duration_days * 86400.0 + sample_step_s, sample_step_s)
    t_ephem_s = np.arange(0.0, duration_days * 86400.0 + ephem_step_s, ephem_step_s)

    import spiceypy as spice

    load_spice_kernels()
    try:
        et0 = float(spice.str2et(epoch_utc))
        ephemeris = sample_moon_centered_ephemeris(et0, t_ephem_s)
        transforms = sample_j2000_to_itrf93_transforms(et0, t_eval_s)
        states = propagate_truth_with_ephemeris(
            t_eval_s,
            x0_mci,
            mu_moon,
            mu_earth,
            mu_sun,
            ephemeris,
            rtol=1e-10,
            atol=1e-11,
        )

        by_name = {station.name: station for station in range_rate_stations()}
        station_names = ["ITU Ayazaga", "Goldstone DSN", "Madrid DSN", "Canberra DSN"]
        stations = [by_name[name] for name in station_names]
        config = VisibilityConfig(
            r_moon_mean_m=float(initial["r_moon_mean_m"]),
            earth_rotation_rad_s=7.292115e-5,
            epoch_utc=epoch_utc,
            min_elevation_deg=10.0,
        )
        starts, ends, raw_masks, filled_network = analyze_visibility_gap_with_transforms(
            t_eval_s,
            states,
            stations,
            ephemeris.earth_position,
            transforms,
            max_gap_s,
            config,
        )
    finally:
        spice.kclear()

    raw_masks = np.asarray(raw_masks, dtype=bool)
    if raw_masks.shape[0] == t_eval_s.size:
        raw_masks = raw_masks.T
    if raw_masks.shape != (len(station_names), t_eval_s.size):
        raise ValueError(
            f"Unexpected visibility-mask shape {raw_masks.shape}; "
            f"expected {(len(station_names), t_eval_s.size)}."
        )
    raw_network = np.any(raw_masks, axis=0)
    colors = ["#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15]},
    )

    ax = axes[0]
    for row, (name, mask, color) in enumerate(zip(station_names, raw_masks, colors)):
        ax.broken_barh(_intervals(mask, sample_step_s), (row - 0.34, 0.68), facecolors=color)
    ax.set_yticks(range(len(station_names)), station_names)
    ax.set_ylim(-0.7, len(station_names) - 0.3)
    ax.invert_yaxis()
    ax.set_ylabel("Ground station")
    ax.set_title("28-Day DSN + ITU Lunar Tracking Visibility")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.broken_barh(_intervals(raw_network, sample_step_s), (0.65, 0.5), facecolors="#4c78a8")
    ax.broken_barh(_intervals(filled_network, sample_step_s), (-0.15, 0.5), facecolors="#e45756")
    ax.set_yticks([0.9, 0.1], ["Raw network", "30-min stitched"])
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Elapsed time [days]")
    ax.set_xlim(0.0, duration_days)
    ax.set_xticks(np.arange(0.0, duration_days + 0.1, 2.0))
    ax.grid(axis="x", alpha=0.25)

    fig.text(
        0.5,
        0.005,
        f"Epoch: {epoch_utc}; 10 deg elevation mask; spherical lunar occultation; 600 s sampling",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))

    out_dir = Path("python_port") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "visibility_28day_dsn_itu_gantt.png"
    csv_path = out_dir / "visibility_28day_dsn_itu_summary.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", "visible_fraction", "visible_hours", "interval_count"])
        for name, mask in zip(station_names, raw_masks):
            writer.writerow(
                [
                    name,
                    float(np.mean(mask)),
                    float(np.count_nonzero(mask) * sample_step_s / 3600.0),
                    len(_intervals(mask, sample_step_s)),
                ]
            )
        writer.writerow(
            [
                "Raw network",
                float(np.mean(raw_network)),
                float(np.count_nonzero(raw_network) * sample_step_s / 3600.0),
                len(_intervals(raw_network, sample_step_s)),
            ]
        )
        writer.writerow(
            [
                "30-min stitched network",
                float(np.mean(filled_network)),
                float(np.count_nonzero(filled_network) * sample_step_s / 3600.0),
                len(starts),
            ]
        )

    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
